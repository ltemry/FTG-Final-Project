"""
Option Pricing on Monte Carlo Paths
====================================

Values monthly 100-MW spark spread call options for the operating period.

Spark Spread Call (per hour):
    Payoff = max(0, Power_t - (Gas_t + VOM) × HR - K) × Volume

Heat Rate Call Option (HRCO, Extra Credit b):
    Payoff = max(0, Power_t - Gas_t × HR_strike) × Volume

Both are valued under the zero-cost-of-carry assumption from the project
brief (spot price = forward price = expected future price under risk-neutral
measure, with r = 0).

Output: monthly option values, with 5/50/95 percentile bands across paths.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from monte_carlo import SimResult


# ──────────────────────────────────────────────────────────────────────────────
# SPARK SPREAD OPTION
# ──────────────────────────────────────────────────────────────────────────────

def spark_spread_payoffs(power: np.ndarray,
                         gas: np.ndarray,
                         heat_rate: float,
                         vom: float = 0.0,
                         strike: float = 0.0) -> np.ndarray:
    """
    Per-hour payoff array: max(0, power - (gas + vom) × HR - K).
    Inputs are [paths, time] arrays. Returns same-shape array.
    """
    return np.maximum(0.0, power - (gas + vom) * heat_rate - strike)


def value_monthly_strip(sim: SimResult,
                        power_var: str,
                        gas_var: str,
                        heat_rate: float,
                        vom: float = 0.0,
                        strike: float = 0.0,
                        volume_mw: float = 100.0) -> pd.DataFrame:
    """
    Per-month expected value (across paths) of the spark spread call strip.

    Returns DataFrame indexed by year-month with columns:
      mean, p5, p50, p95, n_hours, exercise_rate
    Values are in $ (volume × $/MWh × hours).
    """
    power = sim.get(power_var)
    gas   = sim.get(gas_var)
    payoffs = spark_spread_payoffs(power, gas, heat_rate, vom, strike) * volume_mw

    ts = sim.timestamps
    ym = ts.to_period('M')

    results = []
    for period in sorted(set(ym)):
        mask = (ym == period)
        # Sum payoffs over hours in month, per path
        month_total = payoffs[:, mask].sum(axis=1)  # [n_paths]
        exercise_rate = (payoffs[:, mask] > 0).mean()
        results.append({
            'month':         str(period),
            'n_hours':       int(mask.sum()),
            'mean':          float(month_total.mean()),
            'p5':            float(np.quantile(month_total, 0.05)),
            'p50':           float(np.quantile(month_total, 0.50)),
            'p95':           float(np.quantile(month_total, 0.95)),
            'exercise_rate': float(exercise_rate),
        })

    return pd.DataFrame(results).set_index('month')


def value_strip_multi_strike(sim: SimResult,
                             power_var: str,
                             gas_var: str,
                             heat_rate: float,
                             vom: float,
                             strikes: list[float],
                             volume_mw: float = 100.0) -> pd.DataFrame:
    """
    Value the strip for a list of strikes. Returns wide DataFrame:
      rows = month, cols = strike
    """
    out = {}
    for K in strikes:
        df = value_monthly_strip(sim, power_var, gas_var, heat_rate, vom, K, volume_mw)
        out[f'K=${K:.0f}'] = df['mean']
    return pd.DataFrame(out)


# ──────────────────────────────────────────────────────────────────────────────
# HEAT RATE CALL OPTION (Extra Credit b)
# ──────────────────────────────────────────────────────────────────────────────

def value_hrco_monthly(sim: SimResult,
                       power_var: str,
                       gas_var: str,
                       strike_heat_rate: float,
                       volume_mw: float = 100.0) -> pd.DataFrame:
    """
    HRCO payoff: max(0, Power - Gas × HR_strike) × Volume
    No VOM, no separate strike — the strike heat rate IS the strike.

    This is essentially a spark spread call with VOM=0, K=0, HR=HR_strike.
    """
    return value_monthly_strip(sim, power_var, gas_var,
                                heat_rate=strike_heat_rate, vom=0.0,
                                strike=0.0, volume_mw=volume_mw)


# ──────────────────────────────────────────────────────────────────────────────
# OPERATING POLICY: Tolling vs Grid Procurement Cost
# ──────────────────────────────────────────────────────────────────────────────

def hourly_procurement_cost(sim: SimResult,
                            power_var: str,
                            gas_var: str,
                            heat_rate: float,
                            vom: float,
                            volume_mw: float = 100.0) -> dict:
    """
    Operating policy: at each hour, choose min(LMP, toll_cost).

    Returns dict of total-cost arrays [n_paths] and savings vs grid-only.
    """
    power = sim.get(power_var)
    gas   = sim.get(gas_var)

    toll_cost = (gas + vom) * heat_rate     # $/MWh, same shape as power
    # Optimal: pay whichever is cheaper per hour
    optimal = np.minimum(power, toll_cost)
    # Note: data center pays POWER (since it's consuming, not selling)
    # If power < toll: better to buy from grid; if toll < power: better to toll
    # Cost per path = sum of min × volume
    cost_grid_only = (power * volume_mw).sum(axis=1)
    cost_optimal   = (optimal * volume_mw).sum(axis=1)
    savings        = cost_grid_only - cost_optimal

    return {
        'cost_grid_only': cost_grid_only,
        'cost_optimal':   cost_optimal,
        'cost_toll_only': (toll_cost * volume_mw).sum(axis=1),
        'savings_vs_grid': savings,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BUNDLED REPORT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_strip_report(sim: SimResult,
                       configurations: list[dict]) -> pd.DataFrame:
    """
    Run multiple configurations and concatenate results.

    Each configuration is dict with:
      label, power_var, gas_var, heat_rate, vom, strike, volume_mw
    """
    parts = []
    for cfg in configurations:
        df = value_monthly_strip(
            sim,
            power_var=cfg['power_var'],
            gas_var=cfg['gas_var'],
            heat_rate=cfg['heat_rate'],
            vom=cfg.get('vom', 0.0),
            strike=cfg.get('strike', 0.0),
            volume_mw=cfg.get('volume_mw', 100.0),
        )
        df['config'] = cfg['label']
        parts.append(df)
    out = pd.concat(parts)
    return out


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

    print("=" * 70)
    print("PHASE 2: MONTHLY SPARK SPREAD STRIP VALUATION")
    print("=" * 70)

    # ─── Load, calibrate, simulate ───
    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    print("\nSimulating 10,000 paths for June 1 – Dec 1, 2026...")
    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-06-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=10_000, seed=42)
    print(f"Done. Array: {sim.paths.shape}, memory {sim.paths.nbytes / 1e6:.0f} MB")

    # ═══════════════════════════════════════════════════════════════════════
    # 1) Monthly Spark Spread Strip — Houston tolled vs HSC and HH at K=0
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Spark Spread Strip: 100 MW × Houston (HSC + $3 VOM, HR=9.5) ──")
    HEAT_RATE = 9.5
    HOU_VOM = 3.0

    strip_hsc = value_monthly_strip(sim, 'power_houston', 'gas_hsc',
                                     heat_rate=HEAT_RATE, vom=HOU_VOM,
                                     strike=0.0, volume_mw=100)
    print(strip_hsc.round(0).to_string())

    print("\n── Spark Spread Strip: 100 MW × Houston (HH + $3 VOM, HR=9.5) ──")
    strip_hh = value_monthly_strip(sim, 'power_houston', 'gas_hh',
                                    heat_rate=HEAT_RATE, vom=HOU_VOM,
                                    strike=0.0, volume_mw=100)
    print(strip_hh.round(0).to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # 2) Multi-strike sensitivity (Houston, HSC)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Multi-strike sensitivity — Houston HSC ──")
    strikes = [0, 10, 25, 50, 100]
    multi = value_strip_multi_strike(sim, 'power_houston', 'gas_hsc',
                                      heat_rate=HEAT_RATE, vom=HOU_VOM,
                                      strikes=strikes, volume_mw=100)
    print(multi.round(0).to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # 3) HRCO (Extra Credit b) — Houston and West, at various strike HRs
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── HRCO valuation: Houston (HSC), 100 MW ──")
    hrco_hou = {}
    for hr_strike in [7.0, 9.5, 12.0]:
        df = value_hrco_monthly(sim, 'power_houston', 'gas_hsc',
                                 strike_heat_rate=hr_strike, volume_mw=100)
        hrco_hou[f'HR_strike={hr_strike}'] = df['mean']
    print(pd.DataFrame(hrco_hou).round(0).to_string())

    print("\n── HRCO valuation: West (Waha) — virtual financial only, 100 MW ──")
    hrco_west = {}
    for hr_strike in [7.0, 9.5, 12.0]:
        df = value_hrco_monthly(sim, 'power_west', 'gas_waha',
                                 strike_heat_rate=hr_strike, volume_mw=100)
        hrco_west[f'HR_strike={hr_strike}'] = df['mean']
    print(pd.DataFrame(hrco_west).round(0).to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # 4) Operating policy: optimal procurement (min(grid, toll) at Houston)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Optimal Procurement Policy at Houston (100 MW, HSC) ──")
    proc = hourly_procurement_cost(sim, 'power_houston', 'gas_hsc',
                                    heat_rate=HEAT_RATE, vom=HOU_VOM,
                                    volume_mw=100)
    print(f"  Cost grid-only:       ${proc['cost_grid_only'].mean():>15,.0f} "
          f"(p5={np.quantile(proc['cost_grid_only'], 0.05):,.0f}, "
          f"p95={np.quantile(proc['cost_grid_only'], 0.95):,.0f})")
    print(f"  Cost toll-only:       ${proc['cost_toll_only'].mean():>15,.0f} "
          f"(p5={np.quantile(proc['cost_toll_only'], 0.05):,.0f}, "
          f"p95={np.quantile(proc['cost_toll_only'], 0.95):,.0f})")
    print(f"  Cost optimal (min):   ${proc['cost_optimal'].mean():>15,.0f} "
          f"(p5={np.quantile(proc['cost_optimal'], 0.05):,.0f}, "
          f"p95={np.quantile(proc['cost_optimal'], 0.95):,.0f})")
    print(f"  Savings vs grid:      ${proc['savings_vs_grid'].mean():>15,.0f} "
          f"(p5={np.quantile(proc['savings_vs_grid'], 0.05):,.0f}, "
          f"p95={np.quantile(proc['savings_vs_grid'], 0.95):,.0f})")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: monthly strip values with CI for Houston HSC vs HH
    ax = axes[0]
    months = strip_hsc.index.tolist()
    x = np.arange(len(months))
    ax.bar(x - 0.2, strip_hsc['mean'] / 1000, width=0.4, color='steelblue',
           yerr=[strip_hsc['mean']/1000 - strip_hsc['p5']/1000,
                 strip_hsc['p95']/1000 - strip_hsc['mean']/1000],
           capsize=4, label='HSC gas', alpha=0.85)
    ax.bar(x + 0.2, strip_hh['mean'] / 1000, width=0.4, color='darkorange',
           yerr=[strip_hh['mean']/1000 - strip_hh['p5']/1000,
                 strip_hh['p95']/1000 - strip_hh['mean']/1000],
           capsize=4, label='Henry Hub gas', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(months, rotation=45)
    ax.set_ylabel('Monthly value ($K) — 100 MW, K=$0')
    ax.set_title('Houston Spark Spread Option Strip (June – Nov 2026)\n5th–95th percentile across MC paths')
    ax.legend()

    # Right: multi-strike sensitivity
    ax = axes[1]
    for col in multi.columns:
        ax.plot(x, multi[col] / 1000, marker='o', label=col)
    ax.set_xticks(x); ax.set_xticklabels(months, rotation=45)
    ax.set_ylabel('Monthly value ($K)')
    ax.set_title('Strike Sensitivity — Houston HSC')
    ax.legend(title='Strike $/MWh')

    plt.tight_layout()
    plt.savefig(EXHIBITS / '07_monthly_strip_values.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nExhibit saved: {EXHIBITS / '07_monthly_strip_values.png'}")

    # Save tables to CSV
    strip_hsc.to_csv(EXHIBITS / 'strip_houston_HSC.csv')
    strip_hh.to_csv(EXHIBITS / 'strip_houston_HH.csv')
    multi.to_csv(EXHIBITS / 'strip_multi_strike.csv')
    pd.DataFrame(hrco_hou).to_csv(EXHIBITS / 'hrco_houston.csv')
    pd.DataFrame(hrco_west).to_csv(EXHIBITS / 'hrco_west.csv')

    # Procurement summary
    proc_summary = pd.DataFrame({
        k: [v.mean(), np.quantile(v, 0.05), np.quantile(v, 0.50), np.quantile(v, 0.95)]
        for k, v in proc.items()
    }, index=['mean', 'p5', 'p50', 'p95']).round(0)
    print("\nProcurement cost summary saved to procurement_summary.csv")
    proc_summary.to_csv(EXHIBITS / 'procurement_summary.csv')
