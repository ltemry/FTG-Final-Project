"""
Seasonal Ornstein-Uhlenbeck Calibration
========================================

Decomposes each price series into:
    log(P_t + shift) = seasonal_component_t + residual_t

The seasonal_component is a deterministic function of (hour-of-day,
day-of-week, month-of-year). The residual is modeled as Ornstein-Uhlenbeck:

    dX = -κ X dt + σ dW       (mean-reverting toward 0)

Discrete-time analog (AR(1)):
    X_{t+1} = X_t × exp(-κ Δt) + σ × sqrt((1-exp(-2κΔt))/(2κ)) × ε_t

For DAILY-resampled gas (which has near-unit-root behavior), we still
fit OU but expect κ to be small. For HOURLY power, κ should be moderate.

Cross-correlations between residual innovations of different series
are estimated empirically.

Why no jumps: User chose this. Trade-off: model will under-estimate
extreme-event option values (e.g., Feb 2025 had $772/MWh spike that
this model cannot generate). Documented limitation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OUParams:
    """Calibrated parameters for one price series."""
    name: str
    shift: float                        # added before log
    kappa: float                        # mean-reversion speed (per Δt)
    sigma: float                        # vol of innovations
    long_run_mean_log: float            # mean of log(P+shift) in residual frame
    dt_hours: float                     # time step in hours
    seasonal_table: pd.DataFrame        # hour×month seasonal levels in log space
    transform: str = 'log'              # 'log' or 'arith'

    def __repr__(self):
        return (f"OUParams({self.name}: shift={self.shift:.2f}, "
                f"κ={self.kappa:.4f}/{self.dt_hours}h, σ={self.sigma:.4f}, "
                f"halflife≈{np.log(2)/self.kappa * self.dt_hours:.1f}h, "
                f"transform={self.transform})")


# ──────────────────────────────────────────────────────────────────────────────
# SEASONAL DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────────────

def fit_seasonality(s: pd.Series,
                    datetime_index: pd.Series,
                    transform: str = 'log',
                    shift: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """
    Fit hour-of-day × month-of-year seasonality.
    Returns (residual_series, seasonal_table).
    """
    df = pd.DataFrame({'datetime': pd.to_datetime(datetime_index), 'price': s.values})
    df = df.dropna()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month

    if transform == 'log':
        df['y'] = np.log(df['price'] + shift)
    else:
        df['y'] = df['price']

    # Seasonal: mean of y for each (hour, month) cell
    seasonal_table = df.groupby(['hour', 'month'])['y'].mean().unstack('month')

    # Subtract from each observation
    df = df.merge(
        seasonal_table.stack().rename('s').reset_index(),
        on=['hour', 'month'], how='left'
    )
    df['residual'] = df['y'] - df['s']

    return df.set_index('datetime')['residual'], seasonal_table


# ──────────────────────────────────────────────────────────────────────────────
# OU PARAMETER ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def fit_ou_ar1(residuals: pd.Series, dt_hours: float = 1.0) -> tuple[float, float]:
    """
    Fit Ornstein-Uhlenbeck via AR(1) on the residual series.

        X_{t+1} = φ × X_t + ε,     ε ~ N(0, s^2)
        → κ = -ln(φ) / Δt
        → long-run vol σ_LR = s / sqrt(1 - φ^2)
        → instantaneous vol σ = σ_LR × sqrt(2κ)
        Or equivalently σ ≈ s / sqrt((1 - exp(-2κΔt)) / (2κ))
    """
    x = residuals.dropna().values
    x0 = x[:-1]
    x1 = x[1:]
    # OLS without intercept (residuals have zero mean already by construction)
    phi = float((x0 @ x1) / (x0 @ x0))
    phi = max(min(phi, 0.9999), 1e-4)  # bound away from boundary
    eps = x1 - phi * x0
    s = float(np.std(eps, ddof=1))

    kappa = -np.log(phi) / dt_hours
    # σ from continuous-time discretization
    sigma = s * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa * dt_hours)))
    return kappa, sigma


def calibrate_series(price: pd.Series,
                     datetime: pd.Series,
                     name: str,
                     dt_hours: float = 1.0,
                     transform: str = 'log',
                     auto_shift: bool = True,
                     manual_shift: float | None = None) -> tuple[OUParams, pd.Series]:
    """
    Full calibration pipeline for a single series.
    Returns (OUParams, residual_series).
    """
    # Determine shift to ensure log-positivity
    if transform == 'log':
        if manual_shift is not None:
            shift = manual_shift
        elif auto_shift:
            p_min = price.min()
            shift = max(0.0, -p_min + 1.0) if p_min <= 0 else 0.0
        else:
            shift = 0.0
    else:
        shift = 0.0

    residuals, seasonal = fit_seasonality(price, datetime,
                                          transform=transform, shift=shift)
    kappa, sigma = fit_ou_ar1(residuals, dt_hours=dt_hours)

    params = OUParams(
        name=name, shift=shift, kappa=kappa, sigma=sigma,
        long_run_mean_log=float(residuals.mean()),
        dt_hours=dt_hours, seasonal_table=seasonal, transform=transform,
    )
    return params, residuals


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-CORRELATION OF INNOVATIONS
# ──────────────────────────────────────────────────────────────────────────────

def estimate_innovation_corr(residual_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of OU innovations.
    Innovations = ε_t = X_t - φ × X_{t-1}
    """
    innovs = {}
    for col in residual_panel.columns:
        x = residual_panel[col].dropna().values
        if len(x) < 50:
            continue
        x0, x1 = x[:-1], x[1:]
        phi = float((x0 @ x1) / (x0 @ x0))
        innovs[col] = pd.Series(x1 - phi * x0,
                                index=residual_panel[col].dropna().index[1:])
    inv_df = pd.DataFrame(innovs).dropna(how='any')
    return inv_df.corr()


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATE THE FULL JOINT MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JointModel:
    """Container holding all calibrated parameters and correlation matrix."""
    power_houston: OUParams
    power_west:    OUParams
    gas_hsc:       OUParams
    gas_waha:      OUParams
    gas_hh:        OUParams
    correlation:   pd.DataFrame
    var_order:     list = field(default_factory=lambda: [
        'power_houston', 'power_west', 'gas_hsc', 'gas_waha', 'gas_hh'
    ])

    def summary(self) -> pd.DataFrame:
        rows = []
        for nm in self.var_order:
            p = getattr(self, nm)
            rows.append({
                'series': p.name, 'transform': p.transform,
                'shift': p.shift, 'kappa': p.kappa, 'sigma': p.sigma,
                'halflife_hr': np.log(2) / p.kappa * p.dt_hours,
                'dt_hours': p.dt_hours,
            })
        return pd.DataFrame(rows).round(4)


def calibrate_joint(hourly_panel: pd.DataFrame,
                    daily_panel: pd.DataFrame) -> JointModel:
    """
    Calibrate joint model:
      - Hourly OU for HB_HOUSTON, HB_WEST (log with shift)
      - Daily OU for HSC, HH (log)
      - Daily Arithmetic OU for Waha (negative-friendly)
    Cross-correlation estimated on the *daily* aggregated innovations,
    then applied component-wise.
    """
    # POWER — hourly log-OU
    p_hou, res_hou = calibrate_series(
        hourly_panel['hb_houston'], hourly_panel['datetime'],
        name='HB_HOUSTON', dt_hours=1.0, transform='log')
    p_wst, res_wst = calibrate_series(
        hourly_panel['hb_west'], hourly_panel['datetime'],
        name='HB_WEST', dt_hours=1.0, transform='log')

    # GAS — daily, mostly log; Waha is arithmetic because of negatives
    g_hsc, res_hsc = calibrate_series(
        daily_panel['gas_houston'], daily_panel['date'],
        name='HSC', dt_hours=24.0, transform='log')
    g_hh, res_hh = calibrate_series(
        daily_panel['gas_hh'], daily_panel['date'],
        name='HenryHub', dt_hours=24.0, transform='log')
    g_waha, res_waha = calibrate_series(
        daily_panel['gas_west'], daily_panel['date'],
        name='Waha', dt_hours=24.0, transform='arith')

    # Cross-correlation: aggregate power residuals to daily, then merge
    daily_power_res = pd.DataFrame({
        'power_houston': res_hou.groupby(res_hou.index.normalize()).mean(),
        'power_west':    res_wst.groupby(res_wst.index.normalize()).mean(),
    })
    daily_power_res.index.name = 'date'

    daily_gas_res = pd.DataFrame({
        'gas_hsc':  res_hsc,
        'gas_waha': res_waha,
        'gas_hh':   res_hh,
    })
    daily_gas_res.index = pd.to_datetime(daily_gas_res.index).normalize()
    daily_gas_res.index.name = 'date'

    merged = daily_power_res.join(daily_gas_res, how='inner')
    corr = estimate_innovation_corr(merged)

    return JointModel(
        power_houston=p_hou, power_west=p_wst,
        gas_hsc=g_hsc, gas_waha=g_waha, gas_hh=g_hh,
        correlation=corr,
    )


if __name__ == '__main__':
    from data_ingest import build_hourly_panel, build_daily_panel

    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    daily  = build_daily_panel(use_texas_gas=True).dropna()
    print(f"Calibrating on: hourly={len(hourly)}, daily={len(daily)}")

    model = calibrate_joint(hourly, daily)
    print("\n── Calibrated OU parameters ──")
    print(model.summary().to_string(index=False))

    print("\n── Innovation correlation matrix ──")
    print(model.correlation.round(3))

    # Sanity check halflives
    print("\n── Half-life interpretation ──")
    for name in model.var_order:
        p = getattr(model, name)
        hl = np.log(2) / p.kappa * p.dt_hours
        unit = 'hours' if p.dt_hours == 1 else 'days'
        hl_display = hl if p.dt_hours == 1 else hl / 24
        print(f"  {p.name:12s} half-life = {hl_display:.2f} {unit}")
