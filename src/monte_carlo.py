"""
Monte Carlo Simulation Engine
=============================
Simulates correlated hourly paths for HB_HOUSTON, HB_WEST, HSC, Waha,
Henry Hub over the 6-month operating window (June 1 – Dec 1, 2026).

Power = hourly OU (log space, shifted)
Gas   = daily OU (log space for HSC/HH, arithmetic for Waha),
        held flat within each day

Cross-correlation applied at daily innovation level (power innovations
aggregated to daily before correlating with gas).

Outputs a long-format DataFrame [path, datetime, var, value] or a
panel array of shape [paths, hours, vars].
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from calibration import JointModel, OUParams


@dataclass
class SimResult:
    """Container for simulation output."""
    timestamps:  pd.DatetimeIndex          # length T
    paths:       np.ndarray                # shape [P, T, n_vars]
    var_names:   list[str]
    params_used: JointModel

    def get(self, var_name: str) -> np.ndarray:
        """Return paths array for a single variable [P, T]."""
        idx = self.var_names.index(var_name)
        return self.paths[:, :, idx]

    def as_dataframe(self, var_name: str, paths_subset=None) -> pd.DataFrame:
        """Wide DataFrame: rows=time, cols=path index."""
        arr = self.get(var_name)
        if paths_subset is not None:
            arr = arr[:paths_subset]
        return pd.DataFrame(arr.T, index=self.timestamps,
                           columns=[f'path_{i}' for i in range(arr.shape[0])])


# ──────────────────────────────────────────────────────────────────────────────
# CORE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def _last_residual(price_series: pd.Series, params: OUParams) -> float:
    """Get the most recent residual from observed data as MC starting point."""
    # We'll start the simulation at the long-run mean of residuals (~0)
    # because residuals are mean-centered by construction. A more
    # sophisticated start would use the last observed residual.
    return 0.0


def _seasonal_value(params: OUParams, dt: pd.Timestamp) -> float:
    """Look up seasonal log-level for given timestamp."""
    return params.seasonal_table.loc[dt.hour, dt.month]


def simulate_paths(model: JointModel,
                   start: pd.Timestamp,
                   end: pd.Timestamp,
                   n_paths: int = 10_000,
                   seed: int = 42) -> SimResult:
    """
    Simulate joint hourly paths from `start` to `end` (exclusive).

    Strategy:
      1. Build hourly timeline.
      2. For each hour, advance the 2 hourly power OU processes
         + 3 daily gas OU processes (gas only advances at day boundary).
      3. Innovations are drawn from a multivariate normal with
         the calibrated correlation matrix.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start, end=end, freq='h', inclusive='left')
    T = len(timestamps)
    var_names = model.var_order
    n_var = len(var_names)

    # Pre-compute hour-of-day, month, day-index
    hours  = timestamps.hour.values
    months = timestamps.month.values
    days   = pd.Series(timestamps.normalize()).values
    day_changes = np.concatenate(([True], days[1:] != days[:-1]))

    # Cholesky of correlation matrix (positive-definite force)
    C = model.correlation.loc[var_names, var_names].values
    # symmetrize and ensure PD
    C = (C + C.T) / 2
    eig_min = np.linalg.eigvalsh(C).min()
    if eig_min < 1e-6:
        C += np.eye(n_var) * (1e-6 - eig_min)
    L = np.linalg.cholesky(C)

    # OU discrete-time params per series
    params_list = [getattr(model, n) for n in var_names]
    # phi_h = exp(-κ × Δt) per hour for each series (gas: 0 hr increment within day)
    phi_h   = np.array([np.exp(-p.kappa) if p.dt_hours == 1.0 else 1.0
                        for p in params_list])  # power: full step per hour
    phi_d   = np.array([np.exp(-p.kappa) if p.dt_hours == 24.0 else 1.0
                        for p in params_list])  # gas: full step per day
    # σ per hour (power) or per day (gas)
    sigma_h = np.array([p.sigma if p.dt_hours == 1.0 else 0.0 for p in params_list])
    sigma_d = np.array([p.sigma if p.dt_hours == 24.0 else 0.0 for p in params_list])

    # Innovation scale for AR(1):  s = sqrt((1 - exp(-2κΔt)) / (2κ)) × σ
    kappa_arr = np.array([p.kappa for p in params_list])
    dt_arr    = np.array([p.dt_hours for p in params_list])
    s_h = np.array([sigma_h[i] * np.sqrt((1 - np.exp(-2 * kappa_arr[i] * 1.0)) /
                                          (2 * kappa_arr[i])) if dt_arr[i] == 1.0
                    else 0.0 for i in range(n_var)])
    s_d = np.array([sigma_d[i] * np.sqrt((1 - np.exp(-2 * kappa_arr[i] * 24.0)) /
                                          (2 * kappa_arr[i])) if dt_arr[i] == 24.0
                    else 0.0 for i in range(n_var)])

    # State: residuals at t (in log space for log-OU, raw for arith)
    state = np.zeros((n_paths, n_var), dtype=np.float64)
    # Output (we'll store transformed PRICES, not residuals)
    out = np.zeros((n_paths, T, n_var), dtype=np.float32)

    # Pre-build seasonal lookup: array indexed by (h, m, var) → seasonal value
    season_lkp = np.zeros((24, 13, n_var), dtype=np.float64)
    for v, p in enumerate(params_list):
        if p.dt_hours == 24.0:
            # Daily series: month-only seasonality (data is at hour=0 only).
            # Broadcast monthly value across all hours.
            for m in range(1, 13):
                if m in p.seasonal_table.columns:
                    # Use mean across whatever hours are present (typically just h=0)
                    val = p.seasonal_table[m].dropna().mean()
                    if pd.notna(val):
                        season_lkp[:, m, v] = val
        else:
            # Hourly series: full hour × month seasonality
            for h in range(24):
                for m in range(1, 13):
                    if h in p.seasonal_table.index and m in p.seasonal_table.columns:
                        v_seasonal = p.seasonal_table.loc[h, m]
                        if pd.notna(v_seasonal):
                            season_lkp[h, m, v] = v_seasonal

    # Determine whether each variable steps hourly (1) or daily (0)
    is_hourly = (dt_arr == 1.0)
    is_daily  = (dt_arr == 24.0)

    for t in range(T):
        h = hours[t]
        m = months[t]

        # Draw correlated standard normals for this step
        z = rng.standard_normal((n_paths, n_var))
        z_corr = z @ L.T  # apply correlation

        # Advance state per variable
        # Hourly (power): always advance
        if is_hourly.any():
            state[:, is_hourly] = (phi_h[is_hourly] * state[:, is_hourly]
                                   + s_h[is_hourly] * z_corr[:, is_hourly])

        # Daily (gas): advance only on day boundary
        if day_changes[t] and is_daily.any():
            state[:, is_daily] = (phi_d[is_daily] * state[:, is_daily]
                                  + s_d[is_daily] * z_corr[:, is_daily])

        # Convert state → price
        seasonal = season_lkp[h, m, :]  # shape [n_var]
        log_or_arith = state + seasonal  # broadcast

        # Apply transform
        for v, p in enumerate(params_list):
            if p.transform == 'log':
                out[:, t, v] = np.exp(log_or_arith[:, v]) - p.shift
            else:
                out[:, t, v] = log_or_arith[:, v]

    return SimResult(timestamps=timestamps, paths=out,
                     var_names=var_names, params_used=model)


# ──────────────────────────────────────────────────────────────────────────────
# QUICK QC: compare simulated marginals to historical
# ──────────────────────────────────────────────────────────────────────────────

def qc_marginals(sim: SimResult, historical: pd.DataFrame) -> pd.DataFrame:
    """
    Compare simulated vs historical summary stats for each variable.
    historical : panel with columns matching var_names (or aliases).

    For a fair comparison, we compare:
      - mean across all paths × time (should match unconditional mean)
      - median path's stats (single-path vol comparable to historical)
    """
    alias = {
        'power_houston': 'hb_houston',
        'power_west':    'hb_west',
        'gas_hsc':       'gas_houston',
        'gas_waha':      'gas_west',
        'gas_hh':        'gas_hh',
    }
    rows = []
    for v in sim.var_names:
        sim_arr  = sim.get(v)  # [paths, time]
        hist_arr = historical[alias[v]].dropna().values

        # Per-path stats then average across paths (matches single-realization stats)
        per_path_std = np.std(sim_arr, axis=1)        # std along time, per path
        per_path_p95 = np.quantile(sim_arr, 0.95, axis=1)
        per_path_p99 = np.quantile(sim_arr, 0.99, axis=1)
        per_path_max = sim_arr.max(axis=1)

        rows.append({
            'var':            v,
            'sim_mean':       sim_arr.mean(),
            'hist_mean':      hist_arr.mean(),
            'sim_med_std':    per_path_std.mean(),     # avg single-path std
            'hist_std':       hist_arr.std(),
            'sim_med_p95':    per_path_p95.mean(),
            'hist_p95':       np.quantile(hist_arr, 0.95),
            'sim_med_p99':    per_path_p99.mean(),
            'hist_p99':       np.quantile(hist_arr, 0.99),
            'sim_med_max':    per_path_max.mean(),
            'hist_max':       hist_arr.max(),
        })
    return pd.DataFrame(rows).round(2)


def plot_sample_paths(sim: SimResult, n_paths_show: int = 50,
                      historical: pd.DataFrame | None = None,
                      savepath=None):
    """Plot sample simulated paths against historical envelope for QC."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    var_to_plot = ['power_houston', 'power_west', 'gas_hsc', 'gas_hh']
    fig, axes = plt.subplots(len(var_to_plot), 1, figsize=(13, 11), sharex=True)

    alias = {'power_houston': 'hb_houston', 'power_west': 'hb_west',
             'gas_hsc': 'gas_houston', 'gas_hh': 'gas_hh'}

    for ax, v in zip(axes, var_to_plot):
        arr = sim.get(v)
        # Plot subset of paths as thin grey lines
        for i in range(min(n_paths_show, arr.shape[0])):
            ax.plot(sim.timestamps, arr[i], color='grey', lw=0.3, alpha=0.4)
        # Plot quantile bands across paths
        p5  = np.quantile(arr, 0.05, axis=0)
        p50 = np.quantile(arr, 0.50, axis=0)
        p95 = np.quantile(arr, 0.95, axis=0)
        ax.fill_between(sim.timestamps, p5, p95, alpha=0.25, color='steelblue',
                        label='5th–95th pctile across paths')
        ax.plot(sim.timestamps, p50, color='steelblue', lw=1.5, label='Median path')

        # Historical band for comparison
        if historical is not None:
            hist_vals = historical[alias[v]].dropna()
            ax.axhline(hist_vals.mean(), color='red', ls='--', lw=1,
                       label=f'Hist mean = {hist_vals.mean():.1f}')

        ax.set_title(f'Simulated {v} — 2026 operating period')
        ax.set_ylabel('$/MWh' if 'power' in v else '$/MMBtu')
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from pathlib import Path

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

    print("Loading and calibrating...")
    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    # Operating period: 1 June 2026 → 1 Dec 2026
    print("\nSimulating 5,000 paths × 6 months × 5 vars...")
    sim = simulate_paths(model,
                         start=pd.Timestamp('2026-06-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=5_000, seed=42)
    print(f"Simulated array shape: {sim.paths.shape}")
    print(f"Time range: {sim.timestamps[0]} → {sim.timestamps[-1]}")
    print(f"Memory: {sim.paths.nbytes / 1e6:.1f} MB")

    print("\n── QC: per-path single-realization stats vs historical ──")
    print(qc_marginals(sim, hourly).to_string(index=False))

    plot_sample_paths(sim, n_paths_show=50, historical=hourly,
                      savepath=EXHIBITS / '06_simulated_paths.png')
    print(f"\nSample paths plotted: {EXHIBITS / '06_simulated_paths.png'}")
