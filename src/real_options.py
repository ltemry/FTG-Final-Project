"""
Real Options Analysis Module
=============================
Formal real options framing of the tolling agreement.

The tolling contract embeds a STRIP of European spark spread call options.
Each hour t in the operating period, the data center holds the right
(not obligation) to substitute tolled power for grid procurement at a
heat rate of 9.5 MMBtu/MWh with a $3/MMBtu VOM adder.

We compute:
  1. STATIC NPV BASELINES — no flexibility (grid-only and toll-only)
  2. DYNAMIC NPV — full hourly flexibility (optimal min(grid, toll))
  3. REAL OPTION PREMIUM — difference between dynamic and static-min
  4. VOLATILITY SENSITIVITY — option value vs. simulated vol parameter
  5. EMBEDDED OPTIONS INVENTORY — comprehensive list

Framework References:
  - Dixit & Pindyck (1994), Investment Under Uncertainty
  - Trigeorgis (1996), Real Options
  - Geman (2005), Commodities and Commodity Derivatives
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


# ──────────────────────────────────────────────────────────────────────────────
# 1. NPV DECOMPOSITION: STATIC VS DYNAMIC
# ──────────────────────────────────────────────────────────────────────────────

def decompose_npv(sim,
                  power_var: str,
                  gas_var: str,
                  heat_rate: float,
                  vom: float,
                  volume_mw: float = 100.0) -> dict:
    """
    Decompose total procurement cost into static-baseline and dynamic-optimal
    components. The DIFFERENCE between static-cheaper and dynamic is the
    real option premium.

    Static strategies (no flexibility — commit to one source for all hours):
      Strategy A: Grid-only      → cost = Σ LMP_t × MW
      Strategy B: Toll-only      → cost = Σ (Gas_t + VOM) × HR × MW

    Dynamic strategy (full hourly flexibility):
      Strategy C: Optimal        → cost = Σ min(LMP_t, toll_cost_t) × MW

    Real option premium = min(static_A, static_B) − dynamic_C
    """
    power = sim.get(power_var)
    gas   = sim.get(gas_var)

    toll_cost   = (gas + vom) * heat_rate
    optimal_cost = np.minimum(power, toll_cost)

    # Annualize from simulation window
    sim_hours = power.shape[1]
    annual_factor = 8760.0 / sim_hours

    grid_only_path  = power.sum(axis=1)        * volume_mw * annual_factor
    toll_only_path  = toll_cost.sum(axis=1)    * volume_mw * annual_factor
    dynamic_path    = optimal_cost.sum(axis=1) * volume_mw * annual_factor

    static_min_path = np.minimum(grid_only_path, toll_only_path)
    premium_path    = static_min_path - dynamic_path

    def stats(arr):
        return {'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'p5': float(np.quantile(arr, 0.05)),
                'p95': float(np.quantile(arr, 0.95)),
                'std': float(arr.std())}

    return {
        'grid_only_annual':       stats(grid_only_path),
        'toll_only_annual':       stats(toll_only_path),
        'dynamic_optimal_annual': stats(dynamic_path),
        'static_min_annual':      stats(static_min_path),
        'real_option_premium':    stats(premium_path),
        # Pct of paths where toll is the cheaper static strategy
        'toll_dominates_grid_pct': float((toll_only_path < grid_only_path).mean() * 100),
        # Pct of paths where dynamic strictly beats both statics
        'dynamic_beats_both_pct': float(((dynamic_path < grid_only_path) &
                                          (dynamic_path < toll_only_path)).mean() * 100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. MARGRABE CLOSED-FORM BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def margrabe_spread_option(S1: float, S2: float,
                           sigma1: float, sigma2: float,
                           rho: float, T: float,
                           r: float = 0.0) -> dict:
    """
    Margrabe (1978) closed-form value of exchanging asset S2 for asset S1.

    Payoff: max(0, S1 - S2)   at maturity T
    Formula:
        V = S1 × N(d1) - S2 × N(d2)
        σ = sqrt(σ1² - 2ρσ1σ2 + σ2²)
        d1 = (ln(S1/S2) + σ²T/2) / (σ√T)
        d2 = d1 - σ√T

    For our use case:
        S1 = current LMP (or expected LMP at maturity)
        S2 = current toll cost = (Gas + VOM) × HR
        sigma1, sigma2 = annualized vols (already in proper units)
        rho = correlation between LMP and toll cost log-returns
        T = time to maturity in years

    NOTE: This values ONE European spark spread option with payoff at time T.
    For a strip, you'd sum/integrate over multiple T's.
    """
    if S2 <= 0 or S1 <= 0:
        return {'value': max(0, S1 - S2), 'd1': np.nan, 'd2': np.nan, 'sigma_combined': np.nan}

    sigma_combined = np.sqrt(sigma1**2 - 2 * rho * sigma1 * sigma2 + sigma2**2)
    if sigma_combined < 1e-9 or T < 1e-9:
        return {'value': max(0, S1 - S2), 'd1': np.nan, 'd2': np.nan, 'sigma_combined': sigma_combined}

    d1 = (np.log(S1 / S2) + 0.5 * sigma_combined**2 * T) / (sigma_combined * np.sqrt(T))
    d2 = d1 - sigma_combined * np.sqrt(T)
    value = S1 * norm.cdf(d1) - S2 * norm.cdf(d2)
    return {
        'value': float(value),
        'd1': float(d1), 'd2': float(d2),
        'sigma_combined': float(sigma_combined),
        'N(d1)_delta': float(norm.cdf(d1)),
        'N(d2)_exercise_prob': float(norm.cdf(d2)),
    }


def margrabe_strip_benchmark(sim, power_var: str, gas_var: str,
                              heat_rate: float, vom: float,
                              volume_mw: float = 100.0) -> pd.DataFrame:
    """
    Approximate the spark spread strip via Margrabe applied to each (hour-of-
    day, month) cell's expected values and observed volatilities.

    This is a BENCHMARK against the Monte Carlo result, not a replacement.
    Differences between MC and Margrabe arise from:
      - Mean reversion (Margrabe assumes GBM, MC uses OU)
      - Non-flat term structure (Margrabe single maturity)
      - Discrete hourly compounding vs continuous Margrabe
    Margrabe typically OVERSTATES because GBM has fatter tails than OU.
    """
    power = sim.get(power_var)
    gas   = sim.get(gas_var)
    ts    = sim.timestamps
    ym    = ts.to_period('M')

    months = sorted(set(ym))
    rows = []
    for period in months:
        mask = (ym == period)
        # Expected values across paths × hours in this month
        S1 = float(power[:, mask].mean())
        S2 = float((gas[:, mask] + vom).mean()) * heat_rate
        # Vols (log space) — across path-hour
        log_p = np.log(np.clip(power[:, mask], 1, None))
        log_t = np.log(np.clip((gas[:, mask] + vom) * heat_rate, 0.5, None))
        sigma1 = float(log_p.std())
        sigma2 = float(log_t.std())
        # Correlation (cross-sectional, flattened)
        rho = float(np.corrcoef(log_p.flatten(), log_t.flatten())[0, 1])
        # Time to maturity (mid-month) in years
        days_to_mid = (period.to_timestamp() + pd.Timedelta(days=15) -
                       sim.timestamps[0]).days
        T = max(days_to_mid / 365.25, 0.01)

        v = margrabe_spread_option(S1, S2, sigma1, sigma2, rho, T)
        # Scale to full month's hours × volume MW
        n_hours = int(mask.sum())
        rows.append({
            'month':     str(period),
            'E[LMP]':    round(S1, 2),
            'E[Toll]':   round(S2, 2),
            'sigma_LMP': round(sigma1, 3),
            'sigma_Toll':round(sigma2, 3),
            'rho':       round(rho, 3),
            'T_years':   round(T, 3),
            'margrabe_per_hr_per_MW': round(v['value'], 2),
            'margrabe_monthly_$': round(v['value'] * n_hours * volume_mw, 0),
            'delta_N(d1)':       round(v['N(d1)_delta'], 3),
            'exercise_prob_N(d2)': round(v['N(d2)_exercise_prob'], 3),
        })
    return pd.DataFrame(rows).set_index('month')


# ──────────────────────────────────────────────────────────────────────────────
# 3. INTRINSIC VS TIME VALUE
# ──────────────────────────────────────────────────────────────────────────────

def intrinsic_time_value_decomp(current_lmp: float, current_gas: float,
                                 heat_rate: float, vom: float,
                                 mc_option_value_per_hr_per_MW: float) -> dict:
    """
    Decompose the spot real option value into intrinsic + time value.
      Intrinsic value = max(0, current_spark_spread)
      Time value      = MC value − intrinsic
    """
    current_spark = current_lmp - (current_gas + vom) * heat_rate
    intrinsic = max(0.0, current_spark)
    time_val  = max(0.0, mc_option_value_per_hr_per_MW - intrinsic)
    total     = intrinsic + time_val
    return {
        'current_lmp':       current_lmp,
        'current_gas':       current_gas,
        'current_toll_cost': (current_gas + vom) * heat_rate,
        'current_spark':     current_spark,
        'intrinsic':         intrinsic,
        'time_value':        time_val,
        'total':             total,
        'pct_intrinsic':     intrinsic / total * 100 if total > 0 else 0.0,
        'pct_time_value':    time_val  / total * 100 if total > 0 else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. VOLATILITY SENSITIVITY (THE SIGNATURE REAL-OPTIONS DIAGNOSTIC)
# ──────────────────────────────────────────────────────────────────────────────

def volatility_sensitivity(model, vol_multipliers: list[float] = None,
                            n_paths: int = 3000,
                            start: pd.Timestamp = None,
                            end: pd.Timestamp = None,
                            heat_rate: float = 9.5,
                            vom: float = 3.0,
                            volume_mw: float = 100.0) -> pd.DataFrame:
    """
    Compute option value at multiple volatility scalings of the calibrated
    base model. This shows the convexity of option value in vol — the
    classic Black-Scholes-style result. Higher vol ALWAYS → higher option
    value (because the payoff is convex).

    vol_multipliers : list of floats e.g. [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
      1.0 = base calibrated vol
    """
    from monte_carlo import simulate_paths
    from copy import deepcopy

    if vol_multipliers is None:
        vol_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if start is None:
        start = pd.Timestamp('2026-06-01')
    if end is None:
        end = pd.Timestamp('2026-12-01')

    rows = []
    for mult in vol_multipliers:
        scaled = deepcopy(model)
        # Scale each OU sigma by the multiplier
        for name in scaled.var_order:
            p = getattr(scaled, name)
            p.sigma = p.sigma * mult

        sim = simulate_paths(scaled, start=start, end=end,
                              n_paths=n_paths, seed=int(42 + mult * 100))
        power = sim.get('power_houston')
        gas   = sim.get('gas_hsc')
        toll  = (gas + vom) * heat_rate
        # Real option premium (annualized)
        sim_hours = power.shape[1]
        ann = 8760.0 / sim_hours
        grid_cost = power.sum(axis=1) * volume_mw * ann
        opt_cost  = np.minimum(power, toll).sum(axis=1) * volume_mw * ann
        premium = grid_cost - opt_cost

        rows.append({
            'vol_multiplier': mult,
            'option_value_mean': float(premium.mean()),
            'option_value_p5':   float(np.quantile(premium, 0.05)),
            'option_value_p95':  float(np.quantile(premium, 0.95)),
            'exercise_rate':     float((power > toll).mean()),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 5. EMBEDDED OPTIONS INVENTORY
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDED_OPTIONS = pd.DataFrame([
    {'option': 'Operate (toll)',
     'underlying': 'HB_HOUSTON LMP − Toll cost',
     'type': 'Strip of European calls',
     'exercise': 'Hourly',
     'computed': 'YES (Exhibit 07)',
     'value_estimate': '$1.42M / 6 mo',
     'notes': 'Core spark spread option, base-case MC'},
    {'option': 'Switch location',
     'underlying': 'HB_HOUSTON − HB_WEST LMP spread',
     'type': 'Margrabe exchange option',
     'exercise': 'Hourly (for training load)',
     'computed': 'PARTIAL — analyzed but not strip-valued',
     'value_estimate': 'Significant; see Exhibit 04',
     'notes': 'Negative-LMP hours at West are free training fuel'},
    {'option': 'Time-shift training',
     'underlying': 'HB_HOUSTON LMP within day',
     'type': 'Asian/lookback (best of N hours)',
     'exercise': 'Daily (500 MWh/day must-run)',
     'computed': 'PARTIAL — diurnal pattern shown',
     'value_estimate': 'Implicit in optimal procurement',
     'notes': 'See Exhibit 03 — hours 17-20 avoid, 10-15 prefer'},
    {'option': 'Build waste heat recovery',
     'underlying': 'Heat recovery NPV vs capex',
     'type': 'American call (deferred exercise)',
     'exercise': 'One-time investment',
     'computed': 'YES (Exhibit 09)',
     'value_estimate': '$7.70M max capex (Houston)',
     'notes': 'Standard DCF NPV; not stochastic-MC'},
    {'option': 'Abandon toll',
     'underlying': 'Toll lease vs walk-away',
     'type': 'American put',
     'exercise': 'At contract review periods',
     'computed': 'NOT explicitly',
     'value_estimate': 'Small — modeled as 6mo European',
     'notes': 'Project doc says 6-mo strip, so European approx is fine'},
    {'option': 'Carbon timing',
     'underlying': 'Future carbon price',
     'type': 'Optionality to switch tech',
     'exercise': 'Multi-year horizon',
     'computed': 'PARTIAL (Exhibit 08 sensitivity)',
     'value_estimate': 'Major risk — 68% value loss @ $50/ton',
     'notes': 'Regulatory uncertainty drives this'},
])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SCRIPT — RUN ALL AND SAVE EXHIBITS
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
    HEAT_RATE = 9.5
    VOM = 3.0
    MW = 100.0

    print("=" * 70)
    print("FORMAL REAL OPTIONS ANALYSIS")
    print("=" * 70)

    print("\nLoading data and calibrating model...")
    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    print("Simulating 10,000 paths over operating period...")
    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-06-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=10_000, seed=42)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. STATIC vs DYNAMIC NPV DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("1. STATIC vs DYNAMIC NPV DECOMPOSITION")
    print("─" * 70)
    decomp = decompose_npv(sim, 'power_houston', 'gas_hsc',
                            HEAT_RATE, VOM, MW)
    print(f"\nStrategy comparison (100 MW, ANNUALIZED from 6-mo MC):")
    print(f"  {'Strategy':<30s} {'Mean':>12s} {'p5':>12s} {'p95':>12s}")
    for k, label in [('grid_only_annual', 'A: Grid-only (no flexibility)'),
                      ('toll_only_annual', 'B: Toll-only (no flexibility)'),
                      ('static_min_annual', 'min(A,B) static baseline'),
                      ('dynamic_optimal_annual', 'C: Dynamic optimal'),
                      ('real_option_premium', '⭐ REAL OPTION PREMIUM')]:
        s = decomp[k]
        print(f"  {label:<30s} ${s['mean']/1e6:>9.2f}M  "
              f"${s['p5']/1e6:>9.2f}M  ${s['p95']/1e6:>9.2f}M")
    print(f"\n  Toll dominates grid in {decomp['toll_dominates_grid_pct']:.1f}% of paths")
    print(f"  Dynamic beats both statics in {decomp['dynamic_beats_both_pct']:.1f}% of paths")



    # ═══════════════════════════════════════════════════════════════════════
    # 2. VOLATILITY SENSITIVITY — THE REAL-OPTIONS-DEFINING DIAGNOSTIC
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("2. VOLATILITY SENSITIVITY")
    print("─" * 70)
    print("Re-simulating at multiple vol scalings (this takes a minute)...")

    vol_sens = volatility_sensitivity(model,
                                       vol_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                                       n_paths=3000)
    print("\nReal option premium vs vol multiplier:")
    print(vol_sens.round(2).to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: NPV waterfall
    ax = axes[0]
    strategies = ['Grid-only\n(no flex)', 'Toll-only\n(no flex)',
                  'Dynamic\noptimal', 'Premium\n(=A-C)']
    values = [decomp['grid_only_annual']['mean'] / 1e6,
              decomp['toll_only_annual']['mean'] / 1e6,
              decomp['dynamic_optimal_annual']['mean'] / 1e6,
              decomp['real_option_premium']['mean'] / 1e6]
    colors = ['crimson', 'darkorange', 'forestgreen', 'steelblue']
    bars = ax.bar(strategies, values, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'${v:.1f}M', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Annual cost / value ($M)')
    ax.set_title('Static vs Dynamic NPV — Real Option Premium')

    # Panel B: Vol sensitivity (signature real-options chart)
    ax = axes[1]
    vm = vol_sens['vol_multiplier']
    vv = vol_sens['option_value_mean'] / 1e6
    vp5 = vol_sens['option_value_p5'] / 1e6
    vp95 = vol_sens['option_value_p95'] / 1e6
    ax.fill_between(vm, vp5, vp95, alpha=0.25, color='steelblue',
                    label='5th-95th pctile')
    ax.plot(vm, vv, 'o-', color='darkblue', lw=2, label='Mean option value')
    ax.set_xlabel('Volatility multiplier (1.0 = calibrated base)')
    ax.set_ylabel('Annual real option premium ($M, 100 MW)')
    ax.set_title('Real Option Value vs Volatility\n(Convexity: more vol → more value)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('REAL OPTIONS ANALYSIS — Houston Tolling Agreement',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EXHIBITS / '13_real_options_analysis.png', dpi=140, bbox_inches='tight')
    plt.close()

    # Save tables
    vol_sens.to_csv(EXHIBITS / 'real_options_vol_sensitivity.csv', index=False)
    EMBEDDED_OPTIONS.to_csv(EXHIBITS / 'real_options_embedded_inventory.csv', index=False)

    # Save the decomposition summary as a clean CSV
    decomp_df = pd.DataFrame({
        'strategy': ['A: Grid-only', 'B: Toll-only', 'min(A,B) static',
                     'C: Dynamic optimal', 'Real Option Premium'],
        'mean_$M': [decomp['grid_only_annual']['mean']/1e6,
                    decomp['toll_only_annual']['mean']/1e6,
                    decomp['static_min_annual']['mean']/1e6,
                    decomp['dynamic_optimal_annual']['mean']/1e6,
                    decomp['real_option_premium']['mean']/1e6],
        'p5_$M':   [decomp['grid_only_annual']['p5']/1e6,
                    decomp['toll_only_annual']['p5']/1e6,
                    decomp['static_min_annual']['p5']/1e6,
                    decomp['dynamic_optimal_annual']['p5']/1e6,
                    decomp['real_option_premium']['p5']/1e6],
        'p95_$M':  [decomp['grid_only_annual']['p95']/1e6,
                    decomp['toll_only_annual']['p95']/1e6,
                    decomp['static_min_annual']['p95']/1e6,
                    decomp['dynamic_optimal_annual']['p95']/1e6,
                    decomp['real_option_premium']['p95']/1e6],
    }).round(2)
    decomp_df.to_csv(EXHIBITS / 'real_options_npv_decomposition.csv', index=False)

    print(f"\n{'='*70}")
    print("EMBEDDED OPTIONS INVENTORY")
    print('='*70)
    print(EMBEDDED_OPTIONS.to_string(index=False))

    print(f"\nExhibit saved: {EXHIBITS / '13_real_options_analysis.png'}")
    print(f"Tables saved to {EXHIBITS}/real_options_*.csv")
