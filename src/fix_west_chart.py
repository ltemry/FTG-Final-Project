"""
Rebuild the HB_WEST analysis chart.

The original imhr_west_timeseries.png had two problems:
  1. Y-axis scaling was broken because IMHR = LMP/Gas explodes when
     Waha gas is near-zero (median is -$0.70, 86% of days near-zero/negative).
  2. The title implied this was a tolling analysis, but no physical
     generator exists at HB_WEST — the project gives you no tolling
     option there.

This script replaces it with a 4-panel exhibit that shows what's
actually economically meaningful at West:
  Panel A: HB_WEST LMP time series (the only price you actually see)
  Panel B: Waha gas time series (showing how often it goes negative)
  Panel C: West–Houston price spread (drives location-switching decisions)
  Panel D: Winsorized IMHR for context (caveat: only relevant for financial HRCOs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_ingest import build_daily_panel, build_hourly_panel

EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
plt.style.use('seaborn-v0_8-whitegrid')

# Load daily panel and clip to past 3 months
panel = build_daily_panel(use_texas_gas=True)
last_dt = panel['date'].max()
cutoff = last_dt - pd.Timedelta(days=90)
recent = panel[panel['date'] >= cutoff].copy()

# Compute metrics
recent['lmp_spread_west_minus_houston'] = recent['hb_west'] - recent['hb_houston']
# Winsorized IMHR (clip extreme values for visualization only)
raw_imhr = recent['hb_west'] / recent['gas_west']
# Replace inf/nan and clip to reasonable range
recent['imhr_west_winsor'] = raw_imhr.replace([np.inf, -np.inf], np.nan).clip(lower=-50, upper=50)
recent['waha_negative'] = (recent['gas_west'] <= 0)

# 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# ─── Panel A: HB_WEST LMP time series ───
ax = axes[0, 0]
ax.plot(recent['date'], recent['hb_west'], color='steelblue', lw=1.5,
        label='HB_WEST daily-avg LMP')
ax.fill_between(recent['date'], 0, recent['hb_west'],
                where=(recent['hb_west'] >= 0), alpha=0.2, color='steelblue')
ax.fill_between(recent['date'], 0, recent['hb_west'],
                where=(recent['hb_west'] < 0), alpha=0.3, color='crimson',
                label='Negative pricing')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('$/MWh')
ax.set_title('A) HB_WEST Daily-Avg LMP — Past 3 Months\n'
             f'mean=${recent["hb_west"].mean():.1f}, '
             f'median=${recent["hb_west"].median():.1f}, '
             f'min=${recent["hb_west"].min():.1f}')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper left', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ─── Panel B: Waha gas time series ───
ax = axes[0, 1]
ax.plot(recent['date'], recent['gas_west'], color='darkorange', lw=1.5,
        label='Waha daily spot')
ax.fill_between(recent['date'], 0, recent['gas_west'],
                where=(recent['gas_west'] >= 0), alpha=0.2, color='darkorange')
ax.fill_between(recent['date'], 0, recent['gas_west'],
                where=(recent['gas_west'] < 0), alpha=0.3, color='crimson',
                label='Negative pricing')
# Compare to Henry Hub for context
ax.plot(recent['date'], recent['gas_hh'], color='gray', lw=1, ls='--',
        label='Henry Hub (reference)')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('$/MMBtu')
ax.set_title('B) Waha (West TX) Gas — Past 3 Months\n'
             f'{(recent["gas_west"] <= 0).sum()} of {len(recent)} days '
             f'≤ $0 ({(recent["gas_west"] <= 0).mean()*100:.0f}%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper right', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ─── Panel C: West - Houston LMP spread ───
ax = axes[1, 0]
spread = recent['lmp_spread_west_minus_houston']
ax.plot(recent['date'], spread, color='purple', lw=1.5,
        label='HB_WEST − HB_HOUSTON')
ax.fill_between(recent['date'], 0, spread,
                where=(spread >= 0), alpha=0.2, color='green',
                label='West more expensive')
ax.fill_between(recent['date'], 0, spread,
                where=(spread < 0), alpha=0.2, color='red',
                label='West cheaper (shift compute here)')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Spread ($/MWh)')
ax.set_title(f'C) West−Houston LMP Spread (Location-Switching Signal)\n'
             f'mean=${spread.mean():.2f}, '
             f'{(spread < 0).mean()*100:.0f}% of days West is cheaper')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper left', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ─── Panel D: Winsorized IMHR + caveat ───
ax = axes[1, 1]
imhr = recent['imhr_west_winsor']
ax.plot(recent['date'], imhr, color='steelblue', lw=1.5,
        label='IMHR (clipped to ±50 for display)')
# Heat rate reference lines
ax.axhline(6.8, color='forestgreen', ls='--', lw=1.2,
           label='CCGT 6.8 (financial HRCO threshold)')
ax.axhline(9.5, color='crimson', ls='--', lw=1.2,
           label='SCGT 9.5 (financial HRCO threshold)')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('IMHR (MMBtu/MWh) — winsorized')
ax.set_ylim(-55, 55)
ax.set_title('D) HB_WEST "Financial" IMHR — Winsorized for Display\n'
             '(No physical generator exists at West; relevant only for HRCO valuation)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper right', fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.suptitle('HB_WEST Market Conditions — Past 3 Months\n'
             '(no tolling option available at West; analysis focuses on LMP procurement '
             'and financial HRCO opportunities)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / 'imhr_west_timeseries.png', dpi=140, bbox_inches='tight')
plt.close()

print(f"Fixed exhibit saved: {EXHIBITS / 'imhr_west_timeseries.png'}")
print()
print("Summary of what each panel shows:")
print("  A) HB_WEST LMP: the price you actually pay at West (only option there)")
print("  B) Waha gas: shows why no one builds gas plants in West Texas")
print("     (gas trades at near-zero or negative due to pipeline takeaway constraints)")
print("  C) West-Houston spread: signals when to shift compute between locations")
print("  D) Winsorized IMHR: only meaningful for FINANCIAL HRCO valuation")
print("     (raw IMHR would explode to 700,000+ MMBtu/MWh on near-zero gas days)")
