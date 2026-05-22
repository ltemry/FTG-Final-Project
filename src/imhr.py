"""
Implied Market Heat Rate (IMHR) Analysis
========================================
The IMHR is the heat rate that would make the spark spread exactly zero
given observed power and gas prices.

    IMHR = (Power Price - VOM) / Gas Price       [MMBtu/MWh]

When IMHR > generator's actual heat rate (e.g., 9.5 for a peaker SCGT),
the spark spread is positive and tolling is in-the-money. The percentage
of hours where IMHR > HR is essentially the exercise frequency of the
spark spread option.

KEY INTERPRETATIONS
-------------------
- IMHR ≈ 7 :   CCGT in-the-money (HR ~6.5-7), peaker NOT in-the-money
- IMHR ≈ 9.5:  Peaker breakeven
- IMHR > 15:   Scarcity pricing — large positive spark spread
- IMHR < 0:    Negative power price OR power < VOM only — never dispatch

CONVENTIONS
-----------
- Gas in $/MMBtu, Power in $/MWh, Heat Rate in MMBtu/MWh.
- 9500 BTU/kWh = 9.5 MMBtu/MWh (the project's tolled SCGT).
- Antelope Station (Appendix Table 1): HR = 9.2 MMBtu/MWh.
- Typical CCGT: 6.5–7.0 MMBtu/MWh.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
HEAT_RATE_PEAKER    = 9.5     # SCGT (the tolled unit, project assumption)
HEAT_RATE_ANTELOPE  = 9.2     # Antelope Station, EIA Table 1
HEAT_RATE_CCGT      = 6.8     # Typical modern CCGT
HOU_VOM             = 3.0     # $/MMBtu adder, Houston toll (variable O&M)
WEST_VOM            = 0.0     # No physical generator at West


# ──────────────────────────────────────────────────────────────────────────────
# CORE CALCULATIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_imhr(power: np.ndarray | pd.Series,
                 gas:   np.ndarray | pd.Series,
                 vom:   float = 0.0) -> pd.Series:
    """
    Implied Market Heat Rate:
        IMHR = (Power - VOM_in_$_per_MWh_terms) / Gas

    NB: the project's $3/MMBtu adder is a gas-side adder, so it gets
    added to the gas price first:
        IMHR = Power / (Gas + VOM_gas_side)

    We support both conventions; default is "adder is on gas side".
    """
    if not isinstance(gas, pd.Series):
        gas = pd.Series(gas)
    if not isinstance(power, pd.Series):
        power = pd.Series(power)

    gas_eff = gas + vom
    # Guard against divide-by-zero / negative gas
    gas_safe = gas_eff.where(gas_eff > 0.01, np.nan)
    return power / gas_safe


def compute_spark_spread(power: np.ndarray | pd.Series,
                         gas:   np.ndarray | pd.Series,
                         heat_rate: float,
                         vom: float = 0.0) -> pd.Series:
    """
    Spark Spread per MWh:
        SS = Power - (Gas + VOM) × HR
    """
    if not isinstance(gas, pd.Series):
        gas = pd.Series(gas)
    if not isinstance(power, pd.Series):
        power = pd.Series(power)
    return power - (gas + vom) * heat_rate


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def imhr_summary(panel: pd.DataFrame,
                 power_col: str,
                 gas_col: str,
                 vom: float,
                 heat_rates: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Build a summary stats table for IMHR + comparison to multiple
    benchmark heat rates.
    """
    if heat_rates is None:
        heat_rates = {
            'Peaker (9.5)':   HEAT_RATE_PEAKER,
            'Antelope (9.2)': HEAT_RATE_ANTELOPE,
            'CCGT (6.8)':     HEAT_RATE_CCGT,
        }

    imhr = compute_imhr(panel[power_col], panel[gas_col], vom=vom)
    valid = imhr.dropna()

    stats = {
        'Observations':      len(valid),
        'Mean IMHR':         valid.mean(),
        'Median IMHR':       valid.median(),
        'Std Dev':           valid.std(),
        'p5':                valid.quantile(0.05),
        'p25':               valid.quantile(0.25),
        'p75':               valid.quantile(0.75),
        'p95':               valid.quantile(0.95),
        'Max':               valid.max(),
        'Min':               valid.min(),
    }
    for name, hr in heat_rates.items():
        stats[f'% hrs IMHR > {name}'] = (valid > hr).mean() * 100

    return pd.Series(stats).to_frame('value').round(3)


def plot_imhr_timeseries(panel: pd.DataFrame,
                         power_col: str,
                         gas_col: str,
                         vom: float,
                         date_col: str = 'date',
                         title: str = '',
                         heat_rates: dict[str, float] | None = None,
                         savepath: Path | str | None = None):
    """Time series of IMHR with horizontal heat rate benchmark lines."""
    if heat_rates is None:
        heat_rates = {
            'Peaker 9.5 (your toll)': HEAT_RATE_PEAKER,
            'Antelope 9.2':           HEAT_RATE_ANTELOPE,
            'CCGT 6.8':               HEAT_RATE_CCGT,
        }

    imhr = compute_imhr(panel[power_col], panel[gas_col], vom=vom)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(panel[date_col], imhr, lw=1, color='steelblue', label='IMHR (daily avg)')
    colors = ['crimson', 'darkorange', 'forestgreen']
    for (label, hr), color in zip(heat_rates.items(), colors):
        ax.axhline(hr, color=color, ls='--', lw=1.2, label=f'{label}')
    # Clip for readability
    ax.set_ylim(max(-5, imhr.quantile(0.01)), min(50, imhr.quantile(0.995)))
    ax.set_ylabel('Implied Market Heat Rate (MMBtu/MWh)')
    ax.set_title(title or 'Implied Market Heat Rate')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close()
    return fig


def plot_imhr_distribution(panel: pd.DataFrame,
                           power_col: str,
                           gas_col: str,
                           vom: float,
                           title: str = '',
                           heat_rates: dict[str, float] | None = None,
                           savepath: Path | str | None = None):
    """Histogram of IMHR with benchmark heat rate lines."""
    if heat_rates is None:
        heat_rates = {
            'Peaker 9.5': HEAT_RATE_PEAKER,
            'Antelope 9.2': HEAT_RATE_ANTELOPE,
            'CCGT 6.8': HEAT_RATE_CCGT,
        }

    imhr = compute_imhr(panel[power_col], panel[gas_col], vom=vom).dropna()
    # Trim extreme tail for visualization
    clipped = imhr.clip(lower=imhr.quantile(0.01), upper=imhr.quantile(0.99))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(clipped, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
    colors = ['crimson', 'darkorange', 'forestgreen']
    for (label, hr), color in zip(heat_rates.items(), colors):
        pct = (imhr > hr).mean() * 100
        ax.axvline(hr, color=color, ls='--', lw=1.5,
                   label=f'{label}: {pct:.1f}% of hrs ITM')
    ax.set_xlabel('Implied Market Heat Rate (MMBtu/MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title(title or 'Distribution of Implied Market Heat Rate')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close()
    return fig


if __name__ == '__main__':
    from data_ingest import build_daily_panel, build_hourly_panel

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
    EXHIBITS.mkdir(exist_ok=True)

    # ───────────────────── DAILY DATA ─────────────────────
    print("=" * 70)
    print("DAILY IMHR ANALYSIS (full 2025)")
    print("=" * 70)
    daily = build_daily_panel(use_texas_gas=True)
    print(f"Coverage: {daily['date'].min().date()} → {daily['date'].max().date()}, "
          f"{len(daily)} days\n")

    # Filter to "past 3 months" relative to most recent data
    last_date = daily['date'].max()
    cutoff = last_date - pd.Timedelta(days=90)
    recent = daily[daily['date'] >= cutoff].copy()
    print(f"Past 3 months: {recent['date'].min().date()} → "
          f"{recent['date'].max().date()}, {len(recent)} days\n")

    # ───── Houston IMHR (HSC gas + $3 VOM) ─────
    print("─" * 60)
    print("HB_HOUSTON IMHR — using HSC gas + $3/MMBtu VOM")
    print("─" * 60)
    summary = imhr_summary(recent, 'hb_houston', 'gas_houston', vom=HOU_VOM)
    print(summary)
    plot_imhr_timeseries(recent, 'hb_houston', 'gas_houston', HOU_VOM,
                         title='HB_HOUSTON IMHR — Past 3 Months (HSC + $3/MMBtu VOM)',
                         savepath=EXHIBITS / 'imhr_houston_timeseries.png')
    plot_imhr_distribution(recent, 'hb_houston', 'gas_houston', HOU_VOM,
                           title='HB_HOUSTON IMHR Distribution — Past 3 Months',
                           savepath=EXHIBITS / 'imhr_houston_distribution.png')
    print()

    # ───── Houston IMHR (HH gas + $3 VOM)  — project convention ─────
    print("─" * 60)
    print("HB_HOUSTON IMHR — using Henry Hub + $3/MMBtu VOM (project doc)")
    print("─" * 60)
    summary_hh = imhr_summary(recent, 'hb_houston', 'gas_hh', vom=HOU_VOM)
    print(summary_hh)
    plot_imhr_timeseries(recent, 'hb_houston', 'gas_hh', HOU_VOM,
                         title='HB_HOUSTON IMHR — Henry Hub Convention',
                         savepath=EXHIBITS / 'imhr_houston_HH_timeseries.png')
    print()

    # ───── West IMHR (Waha) — informational ─────
    print("─" * 60)
    print("HB_WEST IMHR — using Waha gas (no generator available)")
    print("─" * 60)
    summary_w = imhr_summary(recent, 'hb_west', 'gas_west', vom=0.0)
    print(summary_w)
    plot_imhr_timeseries(recent, 'hb_west', 'gas_west', 0.0,
                         title='HB_WEST IMHR — Past 3 Months (Waha, no VOM)',
                         savepath=EXHIBITS / 'imhr_west_timeseries.png')
    print()

    # ───────────────────── HOURLY VERSION ─────────────────────
    print("=" * 70)
    print("HOURLY IMHR ANALYSIS (past 3 months)")
    print("=" * 70)
    hourly = build_hourly_panel(use_texas_gas=True)
    hourly_recent = hourly[hourly['datetime'] >= cutoff].copy()
    print(f"Hourly observations: {len(hourly_recent)}")

    summary_hr = imhr_summary(hourly_recent, 'hb_houston', 'gas_houston', vom=HOU_VOM)
    print("\nHourly IMHR stats:")
    print(summary_hr)
