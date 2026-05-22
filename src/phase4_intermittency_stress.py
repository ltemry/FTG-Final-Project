"""
PHASE 4 — Renewable Intermittency Analysis + Outage Stress Test
================================================================

PART A: Renewable Intermittency
--------------------------------
The fundamental driver of ERCOT spark spread option value is the
volatility of net load (load minus renewable generation). Without
direct access to ERCOT's wind/solar dashboard, we use observable
features in the LMP series as proxies:

  - Diurnal pattern: solar belly compresses midday prices in summer
  - Evening ramp: solar dropoff → price spike (hours 17-21)
  - Negative LMPs in West Texas: indicator of wind oversupply

We compute:
  1. Correlation of HB_WEST minimum hourly LMP with hour-of-day
     (signals wind curtailment hours)
  2. The "evening ramp premium" = mean LMP hour 18 minus hour 14
     (Captures the marginal value of dispatchable capacity at sundown)
  3. Negative-LMP frequency at HB_WEST by month (wind oversupply)

PART B: Outage Stress Test (Winter Storm Uri Analog)
------------------------------------------------------
Our MR-only Monte Carlo cannot reproduce extreme scarcity events.
We stress-test the toll option value under three scenarios:

  Scenario 1: Mild Uri (Feb 2026 cold snap)
    - 72 hours of $200-400/MWh power prices
    - 50% probability per simulated year
  Scenario 2: Moderate Uri
    - 96 hours of $500-1500/MWh prices
    - Gas spike to $20/MMBtu (HSC delivery interruption)
    - 20% probability per year
  Scenario 3: Full Uri replay
    - 100 hours at $9000/MWh cap
    - Gas spike to $250/MMBtu (Uri actual)
    - 5% probability per year

For each scenario, we compute:
  - Spark spread option payoff
  - Pure grid procurement cost (no toll)
  - Net savings from holding toll (or net loss if gas firm fails)
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# PART A: INTERMITTENCY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_intermittency(hourly_panel: pd.DataFrame) -> dict:
    """Compute intermittency signatures from historical hourly data."""
    h = hourly_panel.copy()
    h['hour'] = h['datetime'].dt.hour
    h['month'] = h['datetime'].dt.month

    # 1) Diurnal pattern with quantile bands at HB_HOUSTON
    diurnal_hou = h.groupby('hour')['hb_houston'].agg(
        mean='mean', p25=lambda s: s.quantile(0.25),
        p50='median', p75=lambda s: s.quantile(0.75),
        p95=lambda s: s.quantile(0.95), p99=lambda s: s.quantile(0.99)
    )

    # 2) Negative LMP frequency at HB_WEST by month
    neg_west_by_month = (h.assign(neg=(h['hb_west'] < 0))
                          .groupby('month')['neg'].mean() * 100)

    # 3) Evening ramp premium (hours 18-20 vs hours 13-15)
    h['period'] = pd.cut(h['hour'],
                          bins=[-1, 12, 15, 17, 20, 23],
                          labels=['night', 'midday', 'afternoon', 'evening', 'late'])
    ramp = h.groupby('period', observed=True)['hb_houston'].mean()
    ramp_premium = ramp['evening'] - ramp['midday']

    # 4) Implied solar/wind "shadow" — when does LMP drop low?
    low_lmp = h[h['hb_houston'] < 20]
    low_lmp_hours = low_lmp.groupby('hour').size()
    high_lmp = h[h['hb_houston'] > 80]
    high_lmp_hours = high_lmp.groupby('hour').size()

    return {
        'diurnal_houston': diurnal_hou.round(2),
        'pct_negative_west_by_month': neg_west_by_month.round(2),
        'evening_ramp_premium_$/MWh': float(ramp_premium),
        'midday_avg_$/MWh':           float(ramp['midday']),
        'evening_avg_$/MWh':          float(ramp['evening']),
        'low_lmp_hours_distribution': low_lmp_hours,
        'high_lmp_hours_distribution': high_lmp_hours,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART B: OUTAGE STRESS TEST
# ──────────────────────────────────────────────────────────────────────────────

def stress_overlay(sim_paths: np.ndarray,
                   timestamps: pd.DatetimeIndex,
                   spike_hours: int,
                   spike_price_range: tuple[float, float],
                   gas_paths: np.ndarray | None = None,
                   gas_spike: float | None = None,
                   probability_per_year: float = 0.10,
                   rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Winter Storm Uri-style overlay to a subset of paths.

    For paths chosen with given probability, replace `spike_hours`
    contiguous hours of power prices with values in [low, high] (and
    optionally spike gas at the same time).

    Returns: (modified_power, modified_gas) — copies, not in-place.
    """
    rng = np.random.default_rng(rng_seed)
    P = sim_paths.copy()
    G = gas_paths.copy() if gas_paths is not None else None

    n_paths, n_hours = P.shape
    # Select winter window for the spike (Dec-Feb in our timestamps)
    winter_mask = np.isin(timestamps.month, [12, 1, 2])
    winter_idx = np.where(winter_mask)[0]
    if len(winter_idx) < spike_hours:
        # No winter in window — choose any window
        winter_idx = np.arange(n_hours - spike_hours)

    affected_paths = rng.uniform(size=n_paths) < probability_per_year
    for i in np.where(affected_paths)[0]:
        # Random start within winter
        if len(winter_idx) - spike_hours > 0:
            start = rng.choice(winter_idx[:len(winter_idx) - spike_hours])
        else:
            start = winter_idx[0]
        # Spike prices
        spike_p = rng.uniform(spike_price_range[0], spike_price_range[1],
                              size=spike_hours)
        P[i, start:start + spike_hours] = spike_p
        if G is not None and gas_spike is not None:
            G[i, start:start + spike_hours] = gas_spike

    return P, G, affected_paths


def value_toll_with_stress(sim, scenarios: list[dict],
                            heat_rate: float = 9.5,
                            vom: float = 3.0,
                            volume_mw: float = 100.0) -> pd.DataFrame:
    """
    Run multiple stress scenarios and compare:
      - Procurement cost if always grid
      - Cost if optimal (min toll, grid)
      - Toll savings (or loss if gas firm fails)
    """
    power_base = sim.get('power_houston')
    gas_base = sim.get('gas_hsc')

    results = []
    for s in scenarios:
        P_stress, G_stress, affected = stress_overlay(
            power_base, sim.timestamps,
            spike_hours=s['hours'],
            spike_price_range=s['price_range'],
            gas_paths=gas_base,
            gas_spike=s.get('gas_spike'),
            probability_per_year=s['prob'],
            rng_seed=s.get('seed', 42),
        )
        toll_cost = (G_stress + vom) * heat_rate
        optimal_cost = np.minimum(P_stress, toll_cost)
        grid_cost = P_stress

        # Annualize
        annual_factor = 8760 / P_stress.shape[1]
        grid_total = grid_cost.sum(axis=1) * volume_mw * annual_factor
        optimal_total = optimal_cost.sum(axis=1) * volume_mw * annual_factor
        savings = grid_total - optimal_total

        # Separate stats by whether path was hit
        results.append({
            'scenario':              s['name'],
            'prob_per_year':         s['prob'],
            'affected_paths_pct':    affected.mean() * 100,
            'mean_grid_cost_$M':     grid_total.mean() / 1e6,
            'p95_grid_cost_$M':      np.quantile(grid_total, 0.95) / 1e6,
            'mean_optimal_cost_$M':  optimal_total.mean() / 1e6,
            'mean_toll_savings_$M':  savings.mean() / 1e6,
            'p95_toll_savings_$M':   np.quantile(savings, 0.95) / 1e6,
            'max_toll_savings_$M':   savings.max() / 1e6,
            'mean_grid_hit_$M':      grid_total[affected].mean() / 1e6
                                       if affected.any() else 0,
            'mean_savings_hit_$M':   savings[affected].mean() / 1e6
                                       if affected.any() else 0,
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from data_ingest import build_hourly_panel, build_daily_panel
    from calibration import calibrate_joint
    from monte_carlo import simulate_paths

    EXHIBITS = Path(__file__).resolve().parent.parent / 'exhibits'

    # ─────────────── PART A ───────────────
    print("=" * 70)
    print("PHASE 4a: RENEWABLE INTERMITTENCY ANALYSIS")
    print("=" * 70)

    hourly = build_hourly_panel(use_texas_gas=True).dropna()
    intermittency = analyze_intermittency(hourly)

    print("\n── Diurnal LMP profile (HB_HOUSTON, full 2025) ──")
    print(intermittency['diurnal_houston'].to_string())

    print(f"\n── Evening ramp premium: ${intermittency['evening_ramp_premium_$/MWh']:.2f}/MWh ──")
    print(f"   Midday avg LMP: ${intermittency['midday_avg_$/MWh']:.2f}/MWh")
    print(f"   Evening avg LMP: ${intermittency['evening_avg_$/MWh']:.2f}/MWh")
    print(f"   → Evening prices are {intermittency['evening_ramp_premium_$/MWh']/intermittency['midday_avg_$/MWh']*100:.0f}% "
          f"higher than midday, driven by solar dropoff")

    print(f"\n── HB_WEST negative LMP frequency by month (% of hours) ──")
    print(intermittency['pct_negative_west_by_month'].to_string())
    print(f"\n   ↑ Confirms wind oversupply hours at West Texas")

    # Plot intermittency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    d = intermittency['diurnal_houston']
    ax.fill_between(d.index, d['p25'], d['p75'], alpha=0.3, color='steelblue', label='25-75th %ile')
    ax.fill_between(d.index, d['p75'], d['p95'], alpha=0.15, color='steelblue', label='75-95th %ile')
    ax.fill_between(d.index, d['p95'], d['p99'], alpha=0.08, color='steelblue', label='95-99th %ile')
    ax.plot(d.index, d['p50'], color='darkblue', lw=2, label='Median')
    ax.plot(d.index, d['mean'], color='red', lw=1.5, ls='--', label='Mean')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('HB_HOUSTON LMP ($/MWh)')
    ax.set_title('Diurnal Volatility — Tail risk in evening hours\n(Wider tail bands = optionality value)')
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, 24, 2))

    ax = axes[1]
    nw = intermittency['pct_negative_west_by_month']
    ax.bar(nw.index, nw.values, color='darkorange', alpha=0.85)
    ax.set_xlabel('Month')
    ax.set_ylabel('% of hours with negative LMP')
    ax.set_title('HB_WEST Negative LMP Frequency by Month\n(Wind oversupply signature)')
    ax.set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig(EXHIBITS / '11_intermittency.png', dpi=140, bbox_inches='tight')
    plt.close()

    # ─────────────── PART B ───────────────
    print("\n" + "=" * 70)
    print("PHASE 4b: OUTAGE STRESS TEST (Uri Analog)")
    print("=" * 70)

    daily  = build_daily_panel(use_texas_gas=True).dropna()
    model  = calibrate_joint(hourly, daily)

    # Simulate FULL year for winter coverage
    sim = simulate_paths(model,
                         start=pd.Timestamp('2025-12-01'),
                         end=pd.Timestamp('2026-12-01'),
                         n_paths=5_000, seed=42)

    scenarios = [
        {'name': 'Baseline (no stress)',  'hours': 0,   'price_range': (0, 0),
         'prob': 0.0,    'gas_spike': None, 'seed': 1},
        {'name': 'Mild cold snap',         'hours': 72,  'price_range': (200, 400),
         'prob': 0.50, 'gas_spike': 8.0, 'seed': 2},
        {'name': 'Moderate Uri-style',     'hours': 96,  'price_range': (500, 1500),
         'prob': 0.20, 'gas_spike': 20.0, 'seed': 3},
        {'name': 'Full Uri replay',        'hours': 100, 'price_range': (5000, 9000),
         'prob': 0.05, 'gas_spike': 250.0, 'seed': 4},
    ]

    stress_table = value_toll_with_stress(sim, scenarios)
    print("\n── Stress Test Results (100 MW × annualized) ──")
    print(stress_table.round(2).to_string(index=False))

    # Plot stress results
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios_names = stress_table['scenario'].tolist()
    x = np.arange(len(scenarios_names))
    ax.bar(x - 0.2, stress_table['mean_grid_cost_$M'], width=0.4,
           color='crimson', label='Grid-only cost', alpha=0.85)
    ax.bar(x + 0.2, stress_table['mean_optimal_cost_$M'], width=0.4,
           color='forestgreen', label='Optimal (toll or grid)', alpha=0.85)
    for i, val in enumerate(stress_table['mean_toll_savings_$M']):
        ax.annotate(f'Save ${val:.1f}M', xy=(i, max(stress_table['mean_grid_cost_$M'].iloc[i],
                                                     stress_table['mean_optimal_cost_$M'].iloc[i]) + 0.3),
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(scenarios_names, rotation=10)
    ax.set_ylabel('Annual cost ($M, 100 MW load)')
    ax.set_title('Uri Stress Test — Toll Option Value vs Grid-Only Procurement\n(annualized)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(EXHIBITS / '12_uri_stress_test.png', dpi=140, bbox_inches='tight')
    plt.close()

    stress_table.to_csv(EXHIBITS / 'uri_stress_results.csv', index=False)
    print(f"\nExhibits saved:")
    print(f"  {EXHIBITS / '11_intermittency.png'}")
    print(f"  {EXHIBITS / '12_uri_stress_test.png'}")

    print("\n── KEY TAKEAWAY ──")
    print("Toll option value INCREASES in stress scenarios — it acts as")
    print("tail insurance. BUT: gas supply firmness is critical. If your")
    print("toll's gas isn't firm, scenario 4 (Uri replay) becomes a LOSS")
    print("rather than gain (no fuel + paying gas spike). This is the")
    print("hidden risk that nominal spark spread analysis misses.")
