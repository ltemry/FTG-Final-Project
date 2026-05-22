"""
EXTRA CREDIT (e) — Waste Heat Recovery Investment Justification
================================================================

Setup (per project):
  - Data center: 80 MW compute → 100 MW total (PUE = 1.25)
  - Heat pumps boost waste heat (30-40°C → >100°C)
  - Boosted heat preheats feedwater in adjacent Rankine cycle generator
  - Less steam extracted for preheating → more steam through turbine
  - Net: ~5% incremental MW from existing thermal generation

Per the project text: "roughly equivalent to a free ~5% more megawatts
of power for the data center."

INTERPRETATION:
  5% of 80 MW compute = 4 MW additional output
  OR: 5% of 100 MW total facility ≈ 5 MW

We use 4 MW (the more conservative, project-text-aligned figure).

VALUATION FRAMEWORKS (we compute both):

  Framework A — LMP-only (treat as new generation capacity)
    Annual revenue = 4 MW × hours × avg LMP
    No fuel cost (waste heat is "free")

  Framework B — Spark-spread-augmented (treats incremental MW as
                 thermal generation displacing grid purchase)
    Annual margin = 4 MW × hours × max(LMP - marginal_cost, 0)
    Used when generation only runs when economic

Discount: 10% WACC; 15-year project life (standard CCGT life).

The project asks: "Justify the upper bound on the investment cost
needed to install this technology, at prevailing IMHRs."

So we solve for max-justifiable capex = NPV(positive revenue) at NPV=0.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class WHRConfig:
    incremental_mw: float = 4.0           # 5% of 80 MW compute
    capacity_factor: float = 1.0          # full utilization assumption
    hours_per_year: float = 8760.0
    discount_rate: float = 0.10
    project_life_years: int = 15
    om_cost_per_mwh: float = 2.0          # heat pump electric O&M
    auxiliary_load_pct: float = 0.15      # heat pumps need power (~15% of recovered)

    @property
    def annual_hours(self) -> float:
        return self.hours_per_year * self.capacity_factor

    @property
    def net_mw(self) -> float:
        """Net incremental MW after heat pump's own auxiliary load."""
        return self.incremental_mw * (1 - self.auxiliary_load_pct)


def annuity_pv_factor(rate: float, years: int) -> float:
    """Present value factor of an annuity of $1 for `years` at `rate`."""
    return (1 - (1 + rate) ** -years) / rate


def waste_heat_npv(sim,
                   power_var: str,
                   gas_var: str | None = None,
                   tech_heat_rate: float | None = None,
                   tech_vom: float = 0.0,
                   config: WHRConfig = None,
                   framework: str = 'A') -> dict:
    """
    Compute NPV of waste heat recovery.

    Framework A: Revenue = Net MW × LMP × hours (treat as new generation)
    Framework B: Revenue = Net MW × max(LMP - marginal_cost, 0) × hours
                  (treat as spark-spread amplifier, ONLY runs when economic)
    """
    if config is None:
        config = WHRConfig()

    power = sim.get(power_var)  # [paths, time], hourly $/MWh

    if framework == 'A':
        # Revenue per hour per MW = LMP - OM
        per_hour_per_mw = power - config.om_cost_per_mwh
    elif framework == 'B':
        assert gas_var is not None and tech_heat_rate is not None
        gas = sim.get(gas_var)
        marginal_cost = tech_heat_rate * (gas + tech_vom)
        spark = power - marginal_cost
        per_hour_per_mw = np.maximum(0.0, spark - config.om_cost_per_mwh)
    else:
        raise ValueError(framework)

    # Annualize the simulation period to a yearly basis
    sim_hours = power.shape[1]
    annual_scale = config.annual_hours / sim_hours

    # Annual revenue per path × Net MW
    revenue_per_path = per_hour_per_mw.sum(axis=1) * annual_scale * config.net_mw

    # Aggregate across paths to expected annual revenue
    expected_annual = float(revenue_per_path.mean())
    p5  = float(np.quantile(revenue_per_path, 0.05))
    p95 = float(np.quantile(revenue_per_path, 0.95))

    # Discount to PV via annuity formula
    pv_factor = annuity_pv_factor(config.discount_rate, config.project_life_years)
    max_justifiable_capex = expected_annual * pv_factor

    return {
        'framework':     framework,
        'config':        config,
        'avg_LMP_$/MWh': float(power.mean()),
        'capture_rate':  float((per_hour_per_mw > 0).mean()),
        'annual_revenue_mean':  expected_annual,
        'annual_revenue_p5':    p5,
        'annual_revenue_p95':   p95,
        'pv_factor':            pv_factor,
        'max_justifiable_capex_mean': max_justifiable_capex,
        'max_justifiable_capex_p5':   p5 * pv_factor,
        'max_justifiable_capex_p95':  p95 * pv_factor,
        'per_mw_capex_$M':            max_justifiable_capex / config.net_mw / 1e6,
    }


def sensitivity_table(sim, power_var, gas_var, config_base) -> pd.DataFrame:
    """Run sensitivity on key assumptions."""
    rows = []

    base = waste_heat_npv(sim, power_var, config=config_base, framework='A')
    rows.append({'scenario': 'Base — Framework A (LMP)', **_extract_npv(base)})

    # Framework B with peaker economics
    b_peaker = waste_heat_npv(sim, power_var, gas_var=gas_var,
                               tech_heat_rate=9.5, tech_vom=3.0,
                               config=config_base, framework='B')
    rows.append({'scenario': 'Framework B — peaker spark', **_extract_npv(b_peaker)})

    # Framework B with CCGT economics
    b_ccgt = waste_heat_npv(sim, power_var, gas_var=gas_var,
                             tech_heat_rate=6.8, tech_vom=0.0,
                             config=config_base, framework='B')
    rows.append({'scenario': 'Framework B — CCGT spark', **_extract_npv(b_ccgt)})

    # Sensitivity: discount rate
    for dr in [0.06, 0.08, 0.12]:
        c = WHRConfig(discount_rate=dr,
                      project_life_years=config_base.project_life_years,
                      incremental_mw=config_base.incremental_mw,
                      auxiliary_load_pct=config_base.auxiliary_load_pct,
                      om_cost_per_mwh=config_base.om_cost_per_mwh,
                      capacity_factor=config_base.capacity_factor)
        r = waste_heat_npv(sim, power_var, config=c, framework='A')
        rows.append({'scenario': f'A @ discount={dr*100:.0f}%', **_extract_npv(r)})

    # Sensitivity: project life
    for yrs in [10, 20, 25]:
        c = WHRConfig(project_life_years=yrs,
                      discount_rate=config_base.discount_rate,
                      incremental_mw=config_base.incremental_mw,
                      auxiliary_load_pct=config_base.auxiliary_load_pct,
                      om_cost_per_mwh=config_base.om_cost_per_mwh,
                      capacity_factor=config_base.capacity_factor)
        r = waste_heat_npv(sim, power_var, config=c, framework='A')
        rows.append({'scenario': f'A @ life={yrs}yr', **_extract_npv(r)})

    # Sensitivity: capacity factor (compute may not always run)
    for cf in [0.7, 0.85, 0.95]:
        c = WHRConfig(capacity_factor=cf,
                      discount_rate=config_base.discount_rate,
                      project_life_years=config_base.project_life_years,
                      incremental_mw=config_base.incremental_mw,
                      auxiliary_load_pct=config_base.auxiliary_load_pct,
                      om_cost_per_mwh=config_base.om_cost_per_mwh)
        r = waste_heat_npv(sim, power_var, config=c, framework='A')
        rows.append({'scenario': f'A @ CF={cf*100:.0f}%', **_extract_npv(r)})

    return pd.DataFrame(rows)


def _extract_npv(d: dict) -> dict:
    return {
        'annual_revenue_$M':   round(d['annual_revenue_mean'] / 1e6, 2),
        'max_capex_$M':        round(d['max_justifiable_capex_mean'] / 1e6, 2),
        'capex_$M_per_MW':     round(d['per_mw_capex_$M'], 2),
        'capex_p5_$M':         round(d['max_justifiable_capex_p5'] / 1e6, 2),
        'capex_p95_$M':        round(d['max_justifiable_capex_p95'] / 1e6, 2),
    }


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

    print("=" * 70)
    print("PHASE 3b: WASTE HEAT RECOVERY INVESTMENT NPV")
    print("=" * 70)

    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-06-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=5_000, seed=42)

    config = WHRConfig()
    print(f"\nBase assumptions:")
    print(f"  Incremental MW (gross):  {config.incremental_mw}")
    print(f"  Net MW (after 15% aux):  {config.net_mw}")
    print(f"  Annual hours @ CF 100%:  {config.annual_hours:.0f}")
    print(f"  Discount rate:           {config.discount_rate * 100:.1f}%")
    print(f"  Project life:            {config.project_life_years} years")
    print(f"  O&M cost:                ${config.om_cost_per_mwh}/MWh")

    # Framework A: simple LMP capture
    print(f"\n── Framework A: New Generation (sells at LMP) ──")
    print(f"   Houston (uses HB_HOUSTON LMP)")
    res_a_h = waste_heat_npv(sim, 'power_houston', config=config, framework='A')
    print(f"   Avg LMP:                ${res_a_h['avg_LMP_$/MWh']:.2f}/MWh")
    print(f"   Capture rate:           {res_a_h['capture_rate']*100:.1f}% of hours")
    print(f"   Annual revenue (mean):  ${res_a_h['annual_revenue_mean']/1e6:.2f}M "
          f"(p5={res_a_h['annual_revenue_p5']/1e6:.2f}M, "
          f"p95={res_a_h['annual_revenue_p95']/1e6:.2f}M)")
    print(f"   Max justifiable capex:  ${res_a_h['max_justifiable_capex_mean']/1e6:.2f}M total "
          f"(${res_a_h['per_mw_capex_$M']:.2f}M/MW)")
    print(f"   ${res_a_h['max_justifiable_capex_p5']/1e6:.2f}M – "
          f"${res_a_h['max_justifiable_capex_p95']/1e6:.2f}M at 5–95th pctile")

    print(f"\n   West (uses HB_WEST LMP)")
    res_a_w = waste_heat_npv(sim, 'power_west', config=config, framework='A')
    print(f"   Avg LMP:                ${res_a_w['avg_LMP_$/MWh']:.2f}/MWh")
    print(f"   Annual revenue (mean):  ${res_a_w['annual_revenue_mean']/1e6:.2f}M")
    print(f"   Max justifiable capex:  ${res_a_w['max_justifiable_capex_mean']/1e6:.2f}M total")

    # Framework B: only run when economic
    print(f"\n── Framework B: Spark-spread-amplifier (only runs when SS > 0) ──")
    print(f"   Houston with peaker economics (HSC, 9.5 HR, $3 VOM)")
    res_b_p = waste_heat_npv(sim, 'power_houston', 'gas_hsc',
                              tech_heat_rate=9.5, tech_vom=3.0,
                              config=config, framework='B')
    print(f"   Capture rate:           {res_b_p['capture_rate']*100:.1f}% of hours")
    print(f"   Annual revenue (mean):  ${res_b_p['annual_revenue_mean']/1e6:.2f}M")
    print(f"   Max justifiable capex:  ${res_b_p['max_justifiable_capex_mean']/1e6:.2f}M total")

    print(f"\n   Houston with CCGT economics (HSC, 6.8 HR, no VOM)")
    res_b_c = waste_heat_npv(sim, 'power_houston', 'gas_hsc',
                              tech_heat_rate=6.8, tech_vom=0.0,
                              config=config, framework='B')
    print(f"   Capture rate:           {res_b_c['capture_rate']*100:.1f}% of hours")
    print(f"   Annual revenue (mean):  ${res_b_c['annual_revenue_mean']/1e6:.2f}M")
    print(f"   Max justifiable capex:  ${res_b_c['max_justifiable_capex_mean']/1e6:.2f}M total")

    # Sensitivity table
    print(f"\n── Sensitivity analysis (Houston) ──")
    sens = sensitivity_table(sim, 'power_houston', 'gas_hsc', config)
    print(sens.to_string(index=False))
    sens.to_csv(EXHIBITS / 'waste_heat_sensitivity.csv', index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: framework comparison
    ax = axes[0]
    scenarios = ['LMP capture\n(Houston)', 'LMP capture\n(West)',
                 'Spark spread\n(peaker)', 'Spark spread\n(CCGT)']
    capex_means = [res_a_h['max_justifiable_capex_mean'] / 1e6,
                   res_a_w['max_justifiable_capex_mean'] / 1e6,
                   res_b_p['max_justifiable_capex_mean'] / 1e6,
                   res_b_c['max_justifiable_capex_mean'] / 1e6]
    p5s = [res_a_h['max_justifiable_capex_p5'] / 1e6,
           res_a_w['max_justifiable_capex_p5'] / 1e6,
           res_b_p['max_justifiable_capex_p5'] / 1e6,
           res_b_c['max_justifiable_capex_p5'] / 1e6]
    p95s = [res_a_h['max_justifiable_capex_p95'] / 1e6,
            res_a_w['max_justifiable_capex_p95'] / 1e6,
            res_b_p['max_justifiable_capex_p95'] / 1e6,
            res_b_c['max_justifiable_capex_p95'] / 1e6]
    yerr = [[m - p5 for m, p5 in zip(capex_means, p5s)],
            [p95 - m for p95, m in zip(p95s, capex_means)]]
    ax.bar(scenarios, capex_means, yerr=yerr, color='steelblue',
           alpha=0.85, capsize=4)
    ax.set_ylabel('Max Justifiable CapEx ($M)')
    ax.set_title('Waste Heat Recovery — Max CapEx at NPV=0\n(3.4 MW net, 10% WACC, 15-yr life)')

    # Right: sensitivity heatmap-style
    ax = axes[1]
    rates = [0.06, 0.08, 0.10, 0.12, 0.15]
    lives = [10, 12, 15, 20, 25]
    grid = np.zeros((len(rates), len(lives)))
    for i, dr in enumerate(rates):
        for j, yrs in enumerate(lives):
            c = WHRConfig(discount_rate=dr, project_life_years=yrs)
            r = waste_heat_npv(sim, 'power_houston', config=c, framework='A')
            grid[i, j] = r['max_justifiable_capex_mean'] / 1e6
    im = ax.imshow(grid, aspect='auto', cmap='YlGn', origin='lower')
    ax.set_xticks(range(len(lives))); ax.set_xticklabels(lives)
    ax.set_yticks(range(len(rates))); ax.set_yticklabels([f'{r*100:.0f}%' for r in rates])
    ax.set_xlabel('Project Life (years)')
    ax.set_ylabel('Discount Rate')
    ax.set_title('Max Justifiable CapEx ($M) — sensitivity')
    for i in range(len(rates)):
        for j in range(len(lives)):
            ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(EXHIBITS / '09_waste_heat_npv.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nExhibit saved: {EXHIBITS / '09_waste_heat_npv.png'}")
