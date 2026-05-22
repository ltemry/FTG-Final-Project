"""Quick seasonal diagnostics — informs calibration approach."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_ingest import build_hourly_panel, build_daily_panel

EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

# Hourly panel
h = build_hourly_panel(use_texas_gas=True)
h['hour']  = h['datetime'].dt.hour
h['month'] = h['datetime'].dt.month
h['dow']   = h['datetime'].dt.dayofweek

# Heatmap: hour × month price levels for HB_HOUSTON
pivot_hou = h.pivot_table(index='hour', columns='month',
                          values='hb_houston', aggfunc='mean')
pivot_west = h.pivot_table(index='hour', columns='month',
                           values='hb_west', aggfunc='mean')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, p, title in [(axes[0], pivot_hou, 'HB_HOUSTON'),
                      (axes[1], pivot_west, 'HB_WEST')]:
    im = ax.imshow(p.values, aspect='auto', origin='lower', cmap='RdYlBu_r')
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels([f"{m:02d}" for m in p.columns])
    ax.set_yticks(range(0, 24, 2))
    ax.set_xlabel('Month'); ax.set_ylabel('Hour of Day')
    ax.set_title(f'{title} mean LMP (2025) — heatmap')
    plt.colorbar(im, ax=ax, label='$/MWh')
plt.tight_layout()
plt.savefig(EXHIBITS / '05_seasonality_heatmap.png', dpi=140, bbox_inches='tight')
plt.close()

# Look at HB_HOUSTON distribution
print("HB_HOUSTON hourly LMP stats by month:")
print(h.groupby('month')['hb_houston'].agg(['mean', 'median', 'std', 'min', 'max']).round(1))
print()

# Check positivity (for log transform decision)
print("Negative or near-zero share by hub:")
for col in ['hb_houston', 'hb_west', 'gas_houston', 'gas_west', 'gas_hh']:
    s = h[col].dropna()
    neg_pct = (s <= 0).mean() * 100
    near_zero = (s <= 1).mean() * 100
    print(f"  {col:15s}: {neg_pct:5.2f}% ≤ 0, {near_zero:5.2f}% ≤ $1, min={s.min():.2f}, p1={s.quantile(0.01):.2f}")
print()

# Autocorrelation diagnostics on log-residuals
print("Log-price first-order autocorrelation (raw, no deseasonalization):")
for col in ['hb_houston', 'hb_west', 'gas_houston', 'gas_west', 'gas_hh']:
    s = h[col].dropna()
    s_shift = s + max(0, -s.min() + 1)  # shift to positive
    lp = np.log(s_shift)
    rho1 = lp.autocorr(lag=1)
    rho24 = lp.autocorr(lag=24) if 'gas' not in col else lp.autocorr(lag=7)
    print(f"  {col:15s}: ρ(1)={rho1:.3f}, ρ(24/7)={rho24:.3f}")
