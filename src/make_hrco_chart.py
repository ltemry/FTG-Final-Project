"""
HRCO Strip Visualization (Extra Credit b)
==========================================
Reads the existing hrco_houston.csv and hrco_west.csv from exhibits/ and
builds a publication-quality exhibit showing the indicative payoffs for the
strip of Heat Rate Call Options at each location.

Payoff = max(0, P_E - F_G * HR_strike) * Volume, valued hourly and
summed to monthly strips via 10,000-path Monte Carlo.

The zero-cost-of-carry assumption (spot = 6-month forward) means the
strip is valued directly off spot-calibrated simulated paths. The fixed
cost adder (VOM/transmission/emissions) is set to zero per the project's
simplifying assumption, distinguishing the HRCO from the physical spark
spread option (which carries the $3/MMBtu VOM).

Run from the project root or from src/:  python src/make_hrco_chart.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve exhibits/ relative to this file (src/) so it works from anywhere
EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

hou = pd.read_csv(EXHIBITS / 'hrco_houston.csv')
wst = pd.read_csv(EXHIBITS / 'hrco_west.csv')

# Clean month labels
for df in (hou, wst):
    df['month_label'] = pd.to_datetime(df['month']).dt.strftime('%b')

strikes = ['HR_strike=7.0', 'HR_strike=9.5', 'HR_strike=12.0']
strike_labels = ['HR strike 7.0', 'HR strike 9.5', 'HR strike 12.0']
months = hou['month_label'].tolist()
x = np.arange(len(months))

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

# Panel A: Houston HRCO
ax = axes[0]
width = 0.26
colors = ['#9ecae1', '#4292c6', '#08519c']
for i, (s, lbl, c) in enumerate(zip(strikes, strike_labels, colors)):
    ax.bar(x + (i - 1) * width, hou[s] / 1e6, width=width,
           color=c, alpha=0.9, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(months)
ax.set_ylabel('Monthly HRCO payoff ($M, 100 MW)')
ax.set_title('A) HB_HOUSTON HRCO Strip (Henry Hub gas)\n'
             '6-mo totals: $7.3M / $5.5M / $4.2M  ->  value DECREASES with strike',
             fontsize=11)
ax.legend(title='Strike heat rate', fontsize=9)

# Panel B: West HRCO
ax = axes[1]
colors_w = ['#fdae6b', '#e6550d', '#a63603']
for i, (s, lbl, c) in enumerate(zip(strikes, strike_labels, colors_w)):
    ax.bar(x + (i - 1) * width, wst[s] / 1e6, width=width,
           color=c, alpha=0.9, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(months)
ax.set_ylabel('Monthly HRCO payoff ($M, 100 MW)')
ax.set_title('B) HB_WEST HRCO Strip (Waha gas)\n'
             '6-mo totals: $17.6M / $18.9M / $20.4M  ->  value INCREASES with strike',
             fontsize=11)
ax.legend(title='Strike heat rate', fontsize=9)

plt.suptitle('Heat Rate Call Option (HRCO) Strip Payoffs - June-November 2026\n'
             'Payoff = max(0, P$_E$ - F$_G$ x HR$_{strike}$) x Volume,  zero cost adder, '
             'spot = 6-mo forward (zero cost of carry)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / '14_hrco_strip.png', dpi=140, bbox_inches='tight')
plt.close()

# Console summary
print("Saved:", EXHIBITS / '14_hrco_strip.png')
print()
print("Houston 6-mo totals ($):")
for s in strikes:
    print(f"  {s}: ${hou[s].sum():,.0f}")
print()
print("West 6-mo totals ($):")
for s in strikes:
    print(f"  {s}: ${wst[s].sum():,.0f}")
print()
print(f"West/Houston HRCO ratio at HR=9.5: "
      f"{wst['HR_strike=9.5'].sum() / hou['HR_strike=9.5'].sum():.1f}x")
