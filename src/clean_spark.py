"""
EXTRA CREDIT (d) — Clean Spark Spread with Emissions Cost
==========================================================

Clean Spark Spread per MWh:
    CSS = Power - HR × Gas - HR × CO2_rate × CO2_price - VOM

Where:
    HR        = heat rate (MMBtu/MWh)
    CO2_rate  = emissions (tons CO2 / MMBtu of gas burned)
                  ≈ 0.053 tons/MMBtu for natural gas (well established)
                Per-MWh emissions = HR × 0.053
    CO2_price = carbon price ($/ton)

Equivalent representation using per-MWh emissions intensity (tons/MWh):
    EMI = HR × 0.053
    CSS = Power - HR × Gas - EMI × CO2_price - VOM

The project Appendix Table 1 reports Antelope Station at 0.5 tons CO2/MWh.
Cross-checking: HR=9.2, gas CO2 factor 0.0531, → 0.488 tons/MWh ✓

We compare:
  - Peaker SCGT (9.5 HR, 0.504 tons/MWh, +$3 VOM toll)
  - Antelope Station (9.2 HR, 0.500 tons/MWh)
  - Modern CCGT (6.8 HR, 0.361 tons/MWh)

Carbon price: $50/ton (voluntary market mid, per user choice).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Natural gas combustion: ~117 lb CO2/MMBtu ≈ 0.053 tons/MMBtu
CO2_FACTOR_NG = 0.0531  # tons CO2 per MMBtu


@dataclass
class GenTech:
    name: str
    heat_rate: float           # MMBtu/MWh
    vom_gas_side: float = 0.0  # $/MMBtu, added to gas price
    fixed_OM: float = 0.0       # $/MWh, fixed operating cost

    @property
    def emissions_intensity(self) -> float:
        """Tons CO2 per MWh."""
        return self.heat_rate * CO2_FACTOR_NG


# Standard set of generators for comparison
TECHS = {
    'peaker_SCGT':  GenTech('Peaker SCGT (your toll)', heat_rate=9.5, vom_gas_side=3.0),
    'antelope':     GenTech('Antelope Station',         heat_rate=9.2, vom_gas_side=0.0),
    'modern_CCGT':  GenTech('Modern CCGT',              heat_rate=6.8, vom_gas_side=0.0),
}


def clean_spark_spread(power: np.ndarray,
                       gas:   np.ndarray,
                       tech:  GenTech,
                       co2_price: float) -> np.ndarray:
    """
    Per-hour clean spark spread:
        CSS = Power - HR×(Gas + VOM_gas) - EMI × CO2_price - FOM
    """
    fuel_cost  = tech.heat_rate * (gas + tech.vom_gas_side)
    carbon_cost = tech.emissions_intensity * co2_price
    return power - fuel_cost - carbon_cost - tech.fixed_OM


def value_clean_strip(sim,
                      power_var: str,
                      gas_var: str,
                      tech: GenTech,
                      co2_price: float,
                      volume_mw: float = 100.0,
                      strike: float = 0.0) -> pd.DataFrame:
    """Monthly value of the clean spark spread call option."""
    power = sim.get(power_var)
    gas   = sim.get(gas_var)
    css   = clean_spark_spread(power, gas, tech, co2_price)
    payoff = np.maximum(0.0, css - strike) * volume_mw

    ym = sim.timestamps.to_period('M')
    results = []
    for period in sorted(set(ym)):
        mask = (ym == period)
        month_total = payoff[:, mask].sum(axis=1)
        results.append({
            'month': str(period),
            'mean':  float(month_total.mean()),
            'p5':    float(np.quantile(month_total, 0.05)),
            'p95':   float(np.quantile(month_total, 0.95)),
            'exercise_rate': float((payoff[:, mask] > 0).mean()),
        })
    return pd.DataFrame(results).set_index('month')


def compare_all_techs(sim,
                      power_var: str,
                      gas_var: str,
                      co2_price: float,
                      volume_mw: float = 100.0,
                      strike: float = 0.0) -> pd.DataFrame:
    """Compare all generator technologies side-by-side."""
    out = {}
    for key, tech in TECHS.items():
        df = value_clean_strip(sim, power_var, gas_var, tech, co2_price,
                                volume_mw, strike)
        out[tech.name] = df['mean']
    return pd.DataFrame(out)


def cost_stack_summary(sim,
                       power_var: str,
                       gas_var: str,
                       co2_price: float) -> pd.DataFrame:
    """
    Show average $/MWh cost stack for each tech:
      Fuel cost | Carbon cost | Total marginal cost | Power price | Net margin
    """
    power = sim.get(power_var).flatten()
    gas   = sim.get(gas_var).flatten()
    rows = []
    for key, tech in TECHS.items():
        fuel_cost = tech.heat_rate * (gas.mean() + tech.vom_gas_side)
        carbon    = tech.emissions_intensity * co2_price
        total_cost = fuel_cost + carbon
        avg_lmp   = power.mean()
        rows.append({
            'tech': tech.name,
            'HR':   tech.heat_rate,
            'emi_tons_MWh': tech.emissions_intensity,
            'fuel_cost_$/MWh':    round(fuel_cost, 2),
            'carbon_cost_$/MWh':  round(carbon, 2),
            'total_cost_$/MWh':   round(total_cost, 2),
            'avg_LMP_$/MWh':      round(avg_lmp, 2),
            'avg_margin_$/MWh':   round(avg_lmp - total_cost, 2),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
    CO2_PRICE = 50.0   # $/ton, user choice

    print("=" * 70)
    print(f"PHASE 3a: CLEAN SPARK SPREAD @ ${CO2_PRICE}/ton CO2")
    print("=" * 70)

    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-06-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=10_000, seed=42)

    print("\n── Average $/MWh cost stack across techs (using HSC gas) ──")
    stack = cost_stack_summary(sim, 'power_houston', 'gas_hsc', CO2_PRICE)
    print(stack.to_string(index=False))

    print("\n── Monthly Clean Spark Spread strip values ($, 100 MW, HSC gas) ──")
    comp = compare_all_techs(sim, 'power_houston', 'gas_hsc', CO2_PRICE)
    print(comp.round(0).to_string())
    print(f"\nTotal 6-mo value by tech:")
    print(comp.sum().round(0))

    # Sensitivity: vs no-carbon case
    print("\n── Carbon cost impact: with vs without $50/ton ──")
    comp_no_carbon = compare_all_techs(sim, 'power_houston', 'gas_hsc', 0.0)
    diff = comp_no_carbon - comp
    print("Reduction in option value from $50/ton CO2:")
    print(diff.sum().round(0))
    print("(% reduction)")
    print(((diff.sum() / comp_no_carbon.sum()) * 100).round(1).astype(str) + '%')

    # ─── Plot ───
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: monthly values by tech
    ax = axes[0]
    x = np.arange(len(comp))
    width = 0.27
    colors = ['crimson', 'darkorange', 'forestgreen']
    for i, (col, color) in enumerate(zip(comp.columns, colors)):
        ax.bar(x + (i - 1) * width, comp[col] / 1000, width=width,
               color=color, alpha=0.85, label=col)
    ax.set_xticks(x); ax.set_xticklabels(comp.index, rotation=45)
    ax.set_ylabel('Monthly Clean Spark Spread value ($K, 100 MW)')
    ax.set_title(f'Clean Spark Spread Strip @ ${CO2_PRICE:.0f}/ton CO2 — HSC gas')
    ax.legend(loc='upper right', fontsize=9)

    # Right: cost stack chart
    ax = axes[1]
    techs = stack['tech'].tolist()
    fuel = stack['fuel_cost_$/MWh'].values
    carbon = stack['carbon_cost_$/MWh'].values
    ax.bar(techs, fuel, color='steelblue', label='Fuel cost')
    ax.bar(techs, carbon, bottom=fuel, color='darkorange',
           label=f'CO₂ cost @ ${CO2_PRICE:.0f}/ton')
    ax.axhline(stack['avg_LMP_$/MWh'].iloc[0], color='red', ls='--',
               lw=1.5, label=f"Avg HB_HOUSTON LMP = ${stack['avg_LMP_$/MWh'].iloc[0]:.0f}")
    ax.set_ylabel('$/MWh average')
    ax.set_title('Marginal Cost Stack vs Avg LMP — Houston')
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(EXHIBITS / '08_clean_spark_spread.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nExhibit saved: {EXHIBITS / '08_clean_spark_spread.png'}")

    comp.to_csv(EXHIBITS / 'clean_spark_strip.csv')
    stack.to_csv(EXHIBITS / 'clean_spark_cost_stack.csv', index=False)
