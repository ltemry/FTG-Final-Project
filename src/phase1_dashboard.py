"""
Phase 1 Dashboard: Comprehensive IMHR + Spark Spread Analysis

Generates the core exhibits comparing:
  - Hourly IMHR distribution (HSC vs HH for Houston, Waha for West)
  - Spark spread time series with positive/negative shading
  - Tolling-vs-grid economic comparison
  - Diurnal patterns (which hours of the day are toll-favorable)
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from data_ingest import build_hourly_panel, build_daily_panel
from imhr import (compute_imhr, compute_spark_spread,
                  HEAT_RATE_PEAKER, HEAT_RATE_ANTELOPE, HEAT_RATE_CCGT,
                  HOU_VOM)

EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'
EXHIBITS.mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# Load hourly panel and filter to past 3 months
hourly = build_hourly_panel(use_texas_gas=True)
last_dt = hourly['datetime'].max()
cutoff = last_dt - pd.Timedelta(days=90)
recent = hourly[hourly['datetime'] >= cutoff].copy()
print(f"Hourly observations in past 3 months: {len(recent)}")

# Compute key metrics for all configurations
recent['imhr_hou_hsc']  = compute_imhr(recent['hb_houston'], recent['gas_houston'], vom=HOU_VOM)
recent['imhr_hou_hh']   = compute_imhr(recent['hb_houston'], recent['gas_hh'], vom=HOU_VOM)
recent['imhr_west']     = compute_imhr(recent['hb_west'],    recent['gas_west'], vom=0.0)

recent['ss_hou_hsc'] = compute_spark_spread(recent['hb_houston'], recent['gas_houston'],
                                             HEAT_RATE_PEAKER, vom=HOU_VOM)
recent['ss_hou_hh']  = compute_spark_spread(recent['hb_houston'], recent['gas_hh'],
                                             HEAT_RATE_PEAKER, vom=HOU_VOM)

recent['toll_cost_hsc'] = (recent['gas_houston'] + HOU_VOM) * HEAT_RATE_PEAKER
recent['toll_cost_hh']  = (recent['gas_hh']      + HOU_VOM) * HEAT_RATE_PEAKER
recent['hour']   = recent['datetime'].dt.hour
recent['date']   = recent['datetime'].dt.date


# ─────────────────────────────────────────────────────────────────────────────
# EXHIBIT 1: Hourly IMHR Distribution (Houston, both gas hubs)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

for ax, col, title, gascol in [
    (axes[0], 'imhr_hou_hsc', 'HSC + $3/MMBtu VOM',       'HSC'),
    (axes[1], 'imhr_hou_hh',  'Henry Hub + $3/MMBtu VOM', 'HH'),
]:
    s = recent[col].dropna()
    s_clip = s.clip(upper=s.quantile(0.99))
    ax.hist(s_clip, bins=80, color='steelblue', alpha=0.75, edgecolor='white')

    for hr, color, lbl in [(HEAT_RATE_CCGT, 'forestgreen', 'CCGT 6.8'),
                            (HEAT_RATE_ANTELOPE, 'darkorange', 'Antelope 9.2'),
                            (HEAT_RATE_PEAKER, 'crimson', 'Peaker 9.5')]:
        pct = (s > hr).mean() * 100
        ax.axvline(hr, color=color, ls='--', lw=1.4, label=f'{lbl}: {pct:.1f}% ITM')

    ax.set_title(f'HB_HOUSTON IMHR — {title}\n(median {s.median():.2f}, max {s.max():.1f})')
    ax.set_xlabel('IMHR (MMBtu/MWh)')
    ax.legend(loc='upper right', fontsize=9)

axes[0].set_ylabel('Hour count')
plt.suptitle('Hourly IMHR Distribution — Past 3 Months', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / '01_imhr_hourly_distribution.png', dpi=140, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXHIBIT 2: Tolling Cost vs RT Price — daily averages with quantile bands
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Compute daily mean + p95 of hourly LMP
daily_agg = (recent
             .groupby('date')
             .agg(
                 lmp_mean=('hb_houston', 'mean'),
                 lmp_p95=('hb_houston', lambda s: s.quantile(0.95)),
                 lmp_max=('hb_houston', 'max'),
                 toll_hsc=('toll_cost_hsc', 'mean'),
                 toll_hh=('toll_cost_hh', 'mean'),
             )
             .reset_index())
daily_agg['date'] = pd.to_datetime(daily_agg['date'])

# Panel A — HSC convention
ax = axes[0]
ax.plot(daily_agg['date'], daily_agg['lmp_mean'], color='steelblue',
        lw=1.5, label='HB_HOUSTON daily-mean LMP')
ax.fill_between(daily_agg['date'], daily_agg['lmp_mean'], daily_agg['lmp_p95'],
                alpha=0.25, color='steelblue', label='Mean → p95 hourly band')
ax.plot(daily_agg['date'], daily_agg['toll_hsc'], color='crimson',
        lw=1.5, label='Toll cost (HSC + $3 VOM × 9.5 HR)')
ax.set_ylabel('$/MWh')
ax.set_title('Houston: Grid LMP vs Toll Cost — HSC gas convention')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(top=min(300, daily_agg['lmp_p95'].max() * 1.1))

# Panel B — HH convention
ax = axes[1]
ax.plot(daily_agg['date'], daily_agg['lmp_mean'], color='steelblue',
        lw=1.5, label='HB_HOUSTON daily-mean LMP')
ax.fill_between(daily_agg['date'], daily_agg['lmp_mean'], daily_agg['lmp_p95'],
                alpha=0.25, color='steelblue', label='Mean → p95 hourly band')
ax.plot(daily_agg['date'], daily_agg['toll_hh'], color='darkorange',
        lw=1.5, label='Toll cost (HH + $3 VOM × 9.5 HR)')
ax.set_ylabel('$/MWh')
ax.set_title('Houston: Grid LMP vs Toll Cost — Henry Hub gas convention (project doc)')
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.set_ylim(top=min(300, daily_agg['lmp_p95'].max() * 1.1))

plt.suptitle('Grid Price vs Tolling Cost — Past 3 Months', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / '02_grid_vs_toll.png', dpi=140, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXHIBIT 3: Diurnal pattern — by hour of day, what % of obs is toll favored?
# ─────────────────────────────────────────────────────────────────────────────
diurnal = (recent
           .assign(toll_wins_hsc=(recent['ss_hou_hsc'] > 0),
                   toll_wins_hh =(recent['ss_hou_hh']  > 0))
           .groupby('hour')
           .agg(
               mean_lmp=('hb_houston', 'mean'),
               p95_lmp =('hb_houston', lambda s: s.quantile(0.95)),
               mean_toll_hsc=('toll_cost_hsc', 'mean'),
               mean_toll_hh =('toll_cost_hh',  'mean'),
               pct_toll_wins_hsc=('toll_wins_hsc', 'mean'),
               pct_toll_wins_hh =('toll_wins_hh',  'mean'),
           )
           .reset_index())
diurnal[['pct_toll_wins_hsc', 'pct_toll_wins_hh']] *= 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left — Mean LMP vs toll by hour
ax = axes[0]
ax.bar(diurnal['hour'] - 0.2, diurnal['mean_lmp'], width=0.4,
       color='steelblue', label='Mean LMP', alpha=0.85)
ax.bar(diurnal['hour'] + 0.2, diurnal['mean_toll_hsc'], width=0.4,
       color='crimson', label='Mean toll cost (HSC)', alpha=0.85)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('$/MWh')
ax.set_title('Mean Hourly LMP vs Tolling Cost by Hour of Day')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)

# Right — % of hours toll wins
ax = axes[1]
ax.bar(diurnal['hour'] - 0.2, diurnal['pct_toll_wins_hsc'], width=0.4,
       color='crimson', label='vs HSC gas', alpha=0.85)
ax.bar(diurnal['hour'] + 0.2, diurnal['pct_toll_wins_hh'], width=0.4,
       color='darkorange', label='vs HH gas', alpha=0.85)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('% of hours toll < grid LMP')
ax.set_title('% of Hours Tolling Beats Grid (Spark Spread > 0)')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)

plt.suptitle('Diurnal Pattern — When Does Tolling Win? (Past 3 Months)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / '03_diurnal_pattern.png', dpi=140, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXHIBIT 4: West vs Houston comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

daily_compare = (recent
                 .groupby('date')
                 .agg(hou_mean=('hb_houston', 'mean'),
                      hou_p95 =('hb_houston', lambda s: s.quantile(0.95)),
                      west_mean=('hb_west', 'mean'),
                      west_p95 =('hb_west', lambda s: s.quantile(0.95)),
                      west_min =('hb_west', 'min'))
                 .reset_index())
daily_compare['date'] = pd.to_datetime(daily_compare['date'])

ax = axes[0]
ax.plot(daily_compare['date'], daily_compare['hou_mean'], color='steelblue',
        label='HB_HOUSTON mean')
ax.fill_between(daily_compare['date'], daily_compare['hou_mean'],
                daily_compare['hou_p95'], alpha=0.2, color='steelblue', label='to p95')
ax.set_ylabel('$/MWh')
ax.set_title('HB_HOUSTON LMP — daily mean & p95')
ax.legend()

ax = axes[1]
ax.plot(daily_compare['date'], daily_compare['west_mean'], color='darkorange',
        label='HB_WEST mean')
ax.fill_between(daily_compare['date'], daily_compare['west_min'],
                daily_compare['west_p95'], alpha=0.2, color='darkorange',
                label='daily min → p95')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('$/MWh')
ax.set_title('HB_WEST LMP — note negative excursions from wind oversupply')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend()

plt.suptitle('HB_HOUSTON vs HB_WEST — Past 3 Months', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(EXHIBITS / '04_houston_vs_west.png', dpi=140, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXHIBIT 5: Summary table — print to console + save as CSV
# ─────────────────────────────────────────────────────────────────────────────

def stats_block(s, hrs):
    out = {
        'N': int(s.count()),
        'Mean': s.mean(), 'Median': s.median(),
        'p5': s.quantile(0.05), 'p25': s.quantile(0.25),
        'p75': s.quantile(0.75), 'p95': s.quantile(0.95),
        'Min': s.min(), 'Max': s.max(),
    }
    for hr_lbl, hr in hrs.items():
        out[f'% > {hr_lbl}'] = (s > hr).mean() * 100
    return out

hrs = {'CCGT (6.8)': 6.8, 'Antelope (9.2)': 9.2, 'Peaker (9.5)': 9.5}

summary = pd.DataFrame({
    'IMHR Houston (HSC+$3 VOM)': stats_block(recent['imhr_hou_hsc'].dropna(), hrs),
    'IMHR Houston (HH+$3 VOM)':  stats_block(recent['imhr_hou_hh'].dropna(), hrs),
    'IMHR West (Waha)':          stats_block(recent['imhr_west'].dropna(), hrs),
}).round(2)

print("\n" + "=" * 70)
print("HOURLY IMHR SUMMARY — PAST 3 MONTHS")
print("=" * 70)
print(summary)
summary.to_csv(EXHIBITS / 'summary_imhr_stats.csv')

# Spark spread summary
ss_summary = pd.DataFrame({
    'SS Houston (HSC+$3)': {
        'N hours':       int(recent['ss_hou_hsc'].count()),
        'Mean $/MWh':    recent['ss_hou_hsc'].mean(),
        'Median $/MWh':  recent['ss_hou_hsc'].median(),
        '% hours > 0':   (recent['ss_hou_hsc'] > 0).mean() * 100,
        'Sum positive':  recent.loc[recent['ss_hou_hsc'] > 0, 'ss_hou_hsc'].sum(),
        'Sum negative':  recent.loc[recent['ss_hou_hsc'] < 0, 'ss_hou_hsc'].sum(),
        'Net $/MWh-hr':  recent['ss_hou_hsc'].sum(),
    },
    'SS Houston (HH+$3)': {
        'N hours':       int(recent['ss_hou_hh'].count()),
        'Mean $/MWh':    recent['ss_hou_hh'].mean(),
        'Median $/MWh':  recent['ss_hou_hh'].median(),
        '% hours > 0':   (recent['ss_hou_hh'] > 0).mean() * 100,
        'Sum positive':  recent.loc[recent['ss_hou_hh'] > 0, 'ss_hou_hh'].sum(),
        'Sum negative':  recent.loc[recent['ss_hou_hh'] < 0, 'ss_hou_hh'].sum(),
        'Net $/MWh-hr':  recent['ss_hou_hh'].sum(),
    },
}).round(2)

print("\n" + "=" * 70)
print("HOURLY SPARK SPREAD SUMMARY — PAST 3 MONTHS (HR = 9.5)")
print("=" * 70)
print(ss_summary)
ss_summary.to_csv(EXHIBITS / 'summary_spark_spread.csv')

# Real-money translation
pos_sum_hsc = recent.loc[recent['ss_hou_hsc'] > 0, 'ss_hou_hsc'].sum()
print(f"\nIf we tolled 100 MW only when spark > 0 over past 3 months (HSC convention):")
print(f"  Gross option payoff: ${pos_sum_hsc * 100:,.0f}")
print(f"  (= sum of positive hourly spark × 100 MW capacity)")

print(f"\nExhibits saved to: {EXHIBITS}")
for p in sorted(EXHIBITS.glob('*')):
    print(f"   • {p.name}")
