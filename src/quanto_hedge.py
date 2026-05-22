"""
EXTRA CREDIT (h) — Quanto Hedge: HDD/CDD + HRCO Linear Combination
====================================================================

OBJECTIVE
---------
Construct a linear combination of:
  (i) HDD/CDD weather options (heating/cooling degree day calls)
  (ii) Heat Rate Call Options (HRCO)
to hedge the value of physical natural gas storage CONTINGENT on
generation outages during extreme weather events.

ECONOMIC INTUITION
------------------
When extreme weather + outages co-occur (e.g., Winter Storm Uri):
  - Gas prices spike (heating demand exceeds supply post wellhead freezes)
  - Power prices spike (generators offline, scarcity pricing in ERCOT)
  - Spark spread blows up
  - Gas storage held by owner: value DEPENDS on outage state
      * If you can deliver gas (storage available + offtake operational):
        value soars
      * If you cannot (e.g., compressor outage, pipeline freeze):
        value zero or negative (you bought it; can't sell at spike)

HEDGE FRAMEWORK
---------------
Target: contingent storage payoff
    S_t = I_{extreme weather} × I_{outage_yours} × (P_spot - P_storage_cost)

Hedge: linear combination
    H_t = α × HDD_payoff + β × CDD_payoff + γ × HRCO_payoff

We solve for (α, β, γ) by minimizing:
    E[(S - H)^2]
across simulated scenarios. This is OLS regression of S on the hedge
components — the resulting (α̂, β̂, γ̂) is the minimum-variance hedge.

DATA APPROACH
-------------
Without direct access to ERCOT weather data, we proxy temperature using
the simulated power price residuals:
  - In winter (Nov-Mar) months, high power price residual → cold extreme
  - In summer (Jun-Sep) months, high power price residual → hot extreme

We define:
  T_anomaly_t = sign(month) × residual_t  (scaled to °F)
  HDD_t = max(0, 65 - T_t)
  CDD_t = max(0, T_t - 65)

Reference: Lange (2009), "Hedging Variable Annuity Guarantees..."
           applies similar quanto regression to mixed instruments.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class QuantoSetup:
    base_temp_F: float = 65.0
    summer_months: tuple = (6, 7, 8, 9)
    winter_months: tuple = (12, 1, 2, 3)
    extreme_threshold_F: float = 20.0  # degrees from base = "extreme"
    outage_threshold_pct_lmp: float = 0.95  # outage proxy = top 5% LMP events
    storage_notional_per_event: float = 100_000.0  # $/event


def proxy_temperature_from_power(power_paths: np.ndarray,
                                  timestamps: pd.DatetimeIndex,
                                  setup: QuantoSetup) -> np.ndarray:
    """
    Build a proxy temperature path from simulated power.
    HIGH power in winter → cold extreme (low T)
    HIGH power in summer → hot extreme (high T)
    LOW power in shoulder → mild

    Returns array same shape as power_paths: temperatures in °F.
    """
    # Standardize power per timestamp's month (so we compare extreme vs typical)
    months = np.array(timestamps.month)
    is_summer = np.isin(months, setup.summer_months)
    is_winter = np.isin(months, setup.winter_months)

    # Take z-score of log power across paths per timestamp
    lp = np.log(power_paths + 1)
    means = lp.mean(axis=0, keepdims=True)
    stds  = lp.std(axis=0, keepdims=True) + 1e-9
    z = (lp - means) / stds  # high z = unusually high power

    # Convert to temperature (°F)
    T = np.full_like(z, 70.0)            # mild default
    T[:, is_summer] = 80 + 10 * z[:, is_summer]    # summer: hot extreme when z high
    T[:, is_winter] = 50 - 15 * z[:, is_winter]    # winter: cold extreme when z high
    is_shoulder = ~(is_summer | is_winter)
    T[:, is_shoulder] = 65 + 5 * z[:, is_shoulder]

    return T


def hdd_cdd(temp: np.ndarray, base: float = 65.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute HDD and CDD payoffs from temperature paths."""
    hdd = np.maximum(0, base - temp)
    cdd = np.maximum(0, temp - base)
    return hdd, cdd


def aggregate_monthly(arr: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate hourly array [paths, time] to monthly sums per path."""
    ym = ts.to_period('M')
    out = {}
    for period in sorted(set(ym)):
        mask = (ym == period)
        out[str(period)] = arr[:, mask].sum(axis=1)
    return pd.DataFrame(out)  # [paths × months]


def storage_contingent_payoff(power: np.ndarray, gas: np.ndarray,
                               temp: np.ndarray, ts: pd.DatetimeIndex,
                               setup: QuantoSetup) -> np.ndarray:
    """
    Target payoff to hedge: gas storage value during weather+outage events.

    Payoff per hour:
        S = I_{extreme_T} × I_{LMP > p95} × (gas_t × markup)
        where markup represents the spot premium for stored gas at scarcity.

    Returns [paths, time] array.
    """
    extreme = (np.abs(temp - setup.base_temp_F) > setup.extreme_threshold_F)
    # Outage proxy: LMP at extreme top 5% per timestamp (across paths)
    p95 = np.quantile(power, setup.outage_threshold_pct_lmp, axis=0, keepdims=True)
    outage = (power > p95)
    # Storage value triggered: gas spot × markup
    spot_markup = gas * 3.0  # storage delivers at 3x spot in scarcity (illustrative)
    return extreme * outage * spot_markup


def fit_hedge_weights(target_monthly: pd.DataFrame,
                       hdd_monthly: pd.DataFrame,
                       cdd_monthly: pd.DataFrame,
                       hrco_monthly: pd.DataFrame) -> dict:
    """
    OLS regression of target on (HDD, CDD, HRCO).
    Stack months and paths to maximize observations.
    """
    y = target_monthly.values.flatten()
    X = np.column_stack([
        hdd_monthly.values.flatten(),
        cdd_monthly.values.flatten(),
        hrco_monthly.values.flatten(),
    ])
    # Add intercept
    X_aug = np.column_stack([np.ones(len(y)), X])
    # Solve normal equations: β = (X'X)^-1 X'y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    beta = np.linalg.solve(XtX, Xty)

    y_hat = X_aug @ beta
    resid = y - y_hat
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_total if ss_total > 0 else 0.0

    return {
        'intercept': beta[0],
        'alpha_HDD': beta[1],
        'beta_CDD':  beta[2],
        'gamma_HRCO': beta[3],
        'r_squared': r2,
        'residual_std': float(resid.std()),
        'target_std':   float(y.std()),
        'hedge_effectiveness': 1 - resid.var() / y.var() if y.var() > 0 else 0.0,
    }


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
    setup = QuantoSetup()

    print("=" * 70)
    print("PHASE 3c: QUANTO HEDGE — HDD/CDD + HRCO LINEAR COMBINATION")
    print("=" * 70)

    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    # Simulate a full year to capture both summer and winter extremes
    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-01-01'),
                         end=pd.Timestamp('2027-01-01'),
                         n_paths=5_000, seed=42)

    power = sim.get('power_houston')
    gas   = sim.get('gas_hsc')
    ts    = sim.timestamps

    # 1. Build proxy temperature
    print("\n1) Building proxy temperature paths from power price residuals...")
    temp = proxy_temperature_from_power(power, ts, setup)
    print(f"   Temperature range: {temp.min():.1f}°F to {temp.max():.1f}°F")
    print(f"   Mean by season:")
    months = np.array(ts.month)
    for season_name, mons in [('Winter', setup.winter_months),
                                ('Summer', setup.summer_months)]:
        mask = np.isin(months, mons)
        print(f"     {season_name}: mean={temp[:, mask].mean():.1f}°F, "
              f"5th pct={np.quantile(temp[:, mask], 0.05):.1f}, "
              f"95th pct={np.quantile(temp[:, mask], 0.95):.1f}")

    # 2. Compute HDD, CDD, HRCO, and target
    print("\n2) Computing HDD/CDD and HRCO instruments...")
    hdd, cdd = hdd_cdd(temp, base=setup.base_temp_F)
    hrco = np.maximum(0, power - gas * 9.5)  # HRCO at strike HR=9.5
    target = storage_contingent_payoff(power, gas, temp, ts, setup)

    # 3. Aggregate monthly per path
    print("3) Aggregating to monthly resolution per path...")
    hdd_m   = aggregate_monthly(hdd, ts)
    cdd_m   = aggregate_monthly(cdd, ts)
    hrco_m  = aggregate_monthly(hrco, ts)
    target_m = aggregate_monthly(target, ts)

    print(f"   Mean monthly HDD across all paths/months: {hdd_m.values.mean():.1f}")
    print(f"   Mean monthly CDD across all paths/months: {cdd_m.values.mean():.1f}")
    print(f"   Mean monthly HRCO payoff: ${hrco_m.values.mean():,.0f}/MWh-mo")
    print(f"   Mean monthly target (storage):  ${target_m.values.mean():,.0f}")

    # 4. Fit hedge weights
    print("\n4) Fitting hedge weights via OLS regression...")
    result = fit_hedge_weights(target_m, hdd_m, cdd_m, hrco_m)
    print("\n── HEDGE WEIGHTS (per MWh-eq) ──")
    print(f"   Intercept:         ${result['intercept']:>10,.0f}")
    print(f"   α (HDD weight):    ${result['alpha_HDD']:>10.2f} per HDD")
    print(f"   β (CDD weight):    ${result['beta_CDD']:>10.2f} per CDD")
    print(f"   γ (HRCO weight):   {result['gamma_HRCO']:>10.4f} (notional units)")
    print(f"\n── HEDGE QUALITY ──")
    print(f"   R²:                       {result['r_squared']:>10.3f}")
    print(f"   Hedge effectiveness:      {result['hedge_effectiveness']*100:>9.1f}%")
    print(f"   Target std:              ${result['target_std']:>10,.0f}")
    print(f"   Residual std (unhedged): ${result['residual_std']:>10,.0f}")
    print(f"   Variance reduction:      {(1 - result['residual_std']/result['target_std'])*100:>9.1f}%")

    # 5. Show monthly bootstrap of hedge performance
    monthly_target = target_m.mean()
    monthly_hdd = hdd_m.mean()
    monthly_cdd = cdd_m.mean()
    monthly_hrco = hrco_m.mean()
    monthly_hedge = (result['intercept'] + result['alpha_HDD'] * monthly_hdd
                     + result['beta_CDD'] * monthly_cdd
                     + result['gamma_HRCO'] * monthly_hrco)

    monthly_compare = pd.DataFrame({
        'target_mean':  monthly_target,
        'hedge_mean':   monthly_hedge,
        'residual':     monthly_target - monthly_hedge,
        'mean_HDD':     monthly_hdd,
        'mean_CDD':     monthly_cdd,
        'mean_HRCO':    monthly_hrco,
    })
    print("\n── Monthly target vs replicating hedge (means across paths) ──")
    print(monthly_compare.round(0).to_string())
    monthly_compare.to_csv(EXHIBITS / 'quanto_hedge_monthly.csv')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: monthly target vs hedge
    ax = axes[0]
    x = np.arange(len(monthly_compare))
    ax.bar(x - 0.2, monthly_compare['target_mean'] / 1000, width=0.4,
           color='crimson', alpha=0.85, label='Target (contingent storage)')
    ax.bar(x + 0.2, monthly_compare['hedge_mean'] / 1000, width=0.4,
           color='steelblue', alpha=0.85, label='Hedge replication')
    ax.set_xticks(x); ax.set_xticklabels(monthly_compare.index, rotation=45)
    ax.set_ylabel('Monthly payoff ($K, per-path mean)')
    ax.set_title(f"Quanto Hedge Replication\n(R² = {result['r_squared']:.2f}, "
                 f"Var reduction = {(1 - result['residual_std']/result['target_std'])*100:.0f}%)")
    ax.legend(fontsize=9)

    # Right: residual scatter (target vs hedge across paths/months)
    ax = axes[1]
    y_all = target_m.values.flatten()
    y_hat = (result['intercept']
             + result['alpha_HDD'] * hdd_m.values.flatten()
             + result['beta_CDD']  * cdd_m.values.flatten()
             + result['gamma_HRCO']* hrco_m.values.flatten())
    # Show a sample for clarity
    idx = np.random.default_rng(0).choice(len(y_all), size=min(3000, len(y_all)),
                                           replace=False)
    ax.scatter(y_hat[idx]/1000, y_all[idx]/1000, s=4, alpha=0.3, color='steelblue')
    lim = max(np.abs(y_all).max(), np.abs(y_hat).max()) / 1000 * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='Perfect hedge')
    ax.set_xlabel('Hedge payoff ($K)')
    ax.set_ylabel('Target payoff ($K)')
    ax.set_title('Hedge vs Target — by (path × month)')
    ax.legend()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXHIBITS / '10_quanto_hedge.png', dpi=140, bbox_inches='tight')
    plt.close()

    # Save weights
    pd.DataFrame([result]).to_csv(EXHIBITS / 'quanto_hedge_weights.csv', index=False)
    print(f"\nExhibit saved: {EXHIBITS / '10_quanto_hedge.png'}")
    print("\nMETHODOLOGY NOTE:")
    print("Temperature is proxied from power price residuals due to data access")
    print("constraints. With real ERCOT weather data (NOAA Houston/Midland HDD/CDD)")
    print("and observed gas storage values, the same regression methodology applies")
    print("with empirical observations — only the data source changes.")
