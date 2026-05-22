# Report Writing Guide — ERCOT Tolling Analysis

This document gives you everything you need to write the report: the data pricing framework, the proper figure ordering, and per-figure narrative templates.

---

## Part 1: The Data Pricing Framework

This is the conceptual scaffold the entire report hangs from. Lay this out near the top of your methodology section so the reader understands WHAT we're measuring before we show them HOW it behaves.

### 1.1 The Two Procurement Choices, Formally

At every hour `t` in the operating window (June 1 – Dec 1, 2026), the Houston data center faces a binary procurement decision:

**Option A — Buy from the grid**
- Cost per MWh = `LMP_HB_HOUSTON(t)`
- Source: ERCOT real-time / day-ahead settlement
- This is the floating, hour-by-hour market clearing price

**Option B — Toll the colocated SCGT**
- Cost per MWh = `(HH(t) + $3) × 9.5`
- HH(t) = Henry Hub spot price on day of hour t (in $/MMBtu)
- $3 = variable O&M adder (gas-side, per project text)
- 9.5 = generator heat rate (MMBtu/MWh)

At West, only Option A is available — there is no physical generator to toll.

### 1.2 The Spark Spread Identity

The economic value of having Option B available is the **spark spread**:

```
SS(t) = LMP(t) - (HH(t) + $3) × 9.5      [$/MWh]
```

Sign interpretation:
- **SS(t) > 0**: Tolling is cheaper than grid this hour → exercise the toll
- **SS(t) ≤ 0**: Grid is cheaper than toll this hour → do not toll

The strip of monthly spark spread options has value:

```
V_month = Σ_{t in month} max(0, SS(t) - K) × Volume_MW
```

with K = strike (we use K=0 for the at-the-money case, with sensitivity to $10, $25, $50, $100).

### 1.3 The Implied Market Heat Rate (IMHR)

The IMHR is the heat rate at which the spark spread would be exactly zero, given observed prices:

```
IMHR(t) = LMP(t) / (HH(t) + $3)      [MMBtu/MWh]
```

Sign/threshold interpretation:
- **IMHR > 9.5**: Market is paying more per MWh than your peaker needs → you're ITM
- **IMHR > 9.2**: Antelope Station (Hale County, EIA ID 57865, from Appendix Table 1) is ITM
- **IMHR > 6.8**: A modern CCGT would be ITM
- **IMHR < 0**: Negative LMP — never exercise (would lose money to generate)

### 1.4 Why Hourly Granularity Matters

The payoff is `max(0, SS)` which is **convex**. By Jensen's inequality:

```
E[max(0, SS)] ≥ max(0, E[SS])
```

A "daily average" or "monthly average" calculation systematically **understates** option value because the convexity is destroyed by averaging. This is why our exhibits use hourly resolution throughout.

### 1.5 The Calibrated Stochastic Model

Each price series is decomposed:

```
log(P(t) + s) = μ_{h(t), m(t)} + X(t)
dX = -κ X dt + σ dW
```

Where:
- `s` = positive shift to ensure log-positivity (small for power, larger for Waha)
- `μ_{h,m}` = seasonal mean for hour-of-day × month-of-year cell
- `X(t)` = mean-reverting Ornstein-Uhlenbeck residual

The calibrated parameters across all 5 series are:

| Series | Half-Life | Volatility (per step) | Note |
|---|---|---|---|
| HB_HOUSTON | 9.6 hours | ~0.39 (log) | Fast MR, supply/demand clears hourly |
| HB_WEST | 15.8 hours | ~0.45 (log) | Slower MR than Houston |
| HSC (Houston gas) | 64 days | ~0.04 (log) | Near random walk |
| Waha (West gas) | 60 days | ~0.30 (arith) | Arithmetic OU (negative prices) |
| Henry Hub | 82 days | ~0.03 (log) | Slowest MR |

Cross-correlations of innovations:
- Power-Power: 0.74 (both within ERCOT)
- HSC-Henry Hub: 0.83 (gas basis tight)
- Power-Gas: -0.15 to +0.08 (essentially independent at hourly frequency)

### 1.6 The Monte Carlo Setup

- 10,000 simulated paths
- 6 months × 24 hours = 4,392 hours per path
- 5 correlated state variables per path
- Total: 219.6M individual hourly price observations
- Output: spark spread strip value distribution

---

## Part 2: Recommended Figure Ordering

The current numbering reflects creation order. The REPORT should reorder them by narrative logic. Here's the order I recommend, with the reasoning:

### Section A: Setting Up the Problem (figures that establish the question)

**Figure 1 (was Exhibit 02): Grid Price vs Toll Cost Over Time**
- *Why first*: Shows the basic question — when does each option win?
- *What reader sees*: Toll cost line (red, slowly moving) crosses below LMP band (blue) only at peak times
- *Headline takeaway*: "On average, grid is cheaper; in peaks, toll wins"

**Figure 2 (was Exhibit 03): Diurnal Pattern**
- *Why second*: Shows WHEN the toll wins
- *What reader sees*: Hours 17-20 toll wins 30-60% of the time, all other hours <5%
- *Headline takeaway*: "Tolling is an evening-peak hedge"

**Figure 3 (was Exhibit 05): Seasonality Heatmap**
- *Why third*: Shows WHEN BY MONTH (extends the diurnal story)
- *What reader sees*: August evenings are red-hot
- *Headline takeaway*: "August evenings drive most of the option value"

### Section B: Measuring the Option (figures that quantify it)

**Figure 4 (was Exhibit 01): Hourly IMHR Distribution**
- *Why here*: Quantifies the "how often is toll ITM" question rigorously
- *What reader sees*: Median IMHR 4.56 (HH basis); 7.7% of hours exceed peaker HR
- *Headline takeaway*: "Peaker ITM ~8% of hours; CCGT ITM ~20%"

**Figure 5 (was Exhibit 11): Intermittency Signatures**
- *Why here*: Explains WHY the IMHR distribution has the shape it does
- *What reader sees*: Quantile fan at evening hours; 15% negative LMP at West in March
- *Headline takeaway*: "Renewable variability is the option's economic engine"

### Section C: Model Validation (the simulation engine)

**Figure 6 (was Exhibit 06): Simulated Monte Carlo Paths**
- *Why here*: Shows the reader your simulation engine before showing simulation-based valuations
- *What reader sees*: 5,000 paths with quantile bands; median hugs historical mean
- *Headline takeaway*: "Simulation reproduces historical statistics; valid for forward valuation"

### Section D: The Valuation (headline numbers)

**Figure 7 (NEW Exhibit 13): Real Options Analysis** ⭐
- *Why here*: This is the central exhibit answering "what's the toll worth?"
- *What reader sees*: NPV waterfall + volatility sensitivity (2 panels)
- *Headline takeaway*: "$2.84M annual real option premium; steeply convex in vol"

**Figure 8 (was Exhibit 07): Monthly Spark Spread Strip**
- *Why here*: Decomposes the $1.42M into monthly contributions + strike sensitivity
- *What reader sees*: August $389K dominates; strikes above $50/MWh kill value
- *Headline takeaway*: "Strike K=0 captures all economic exercise hours"

### Section E: Two-Location Comparison

**Figure 9 (was Exhibit 04): HB_HOUSTON vs HB_WEST**
- *Why here*: Now that we understand Houston, contrast with West
- *What reader sees*: Houston spikes high, West dips negative
- *Headline takeaway*: "Different price dynamics call for different financial structures"

**Figure 10 (was imhr_west_timeseries — the fixed 4-panel version)**
- *Why here*: Drills into West's structural pricing
- *What reader sees*: Waha negative 65% of days; West-Houston spread negative 29% of days
- *Headline takeaway*: "West LMP procurement is the only physical option; financial HRCO is interesting"

### Section F: Risk Analysis (required by project)

**Figure 11 (was Exhibit 12): Uri Stress Test**
- *Why here*: Per project requirement on outage contingencies
- *What reader sees*: Toll savings rise from $2.3M (baseline) to $4.7M expected (Full Uri)
- *Headline takeaway*: "Toll is tail insurance — IF gas supply is firm"

### Section G: Extra Credit Sections

**Figure 12 (was Exhibit 08): Clean Spark Spread @ $50/ton CO₂** — EC (d)
**Figure 13 (was Exhibit 09): Waste Heat Recovery NPV** — EC (e)
**Figure 14 (was Exhibit 10): Quanto Hedge** — EC (h)
HRCO valuation (EC b) is a table, not a figure — present in §E.

---

## Part 3: Per-Figure Narrative Templates

For each figure, here's the structure to use in your report:
1. One-line caption
2. What the figure shows (1 paragraph, descriptive)
3. How it's computed (1-2 sentences, methodological)
4. What it means (1 paragraph, interpretive — the "so what")

---

### Figure 1: Grid Price vs Toll Cost Over Time

**Caption**: "HB_HOUSTON daily-mean LMP (blue) with intra-day 95th-percentile band, overlaid with the daily tolling cost computed as (Henry Hub spot + $3 VOM) × 9.5 MMBtu/MWh heat rate. Top panel uses HSC (sensitivity); bottom panel uses Henry Hub (project specification)."

**What it shows**: Over the past 3 months, daily-mean LMPs at HB_HOUSTON ranged from $20 to $60, while the toll cost ranged from $50 to $80. The blue band, extending from daily-mean to daily-95th-percentile of hourly LMPs, frequently pierces above the red/orange toll line during evening hours. On most days, the average LMP is well below the toll cost, but the band's upper edge tells the optionality story.

**How it's computed**: Daily aggregation of hourly HB_HOUSTON LMPs from ERCOT DAM. Toll cost computed at daily resolution using the Henry Hub spot price (data from EIA daily series). The HSC sensitivity uses the analogous Texas Houston Ship Channel spot.

**What it means**: The toll loses to grid on average (red line above blue line most days) but wins on peaks (blue band tops the red line frequently). The entire economic value of the toll lives in the gap between the band's upper edge and the red line. This is the visual case for treating the toll as a strip of European call options, not a baseload supply contract.

---

### Figure 2: Diurnal Pattern — When Does Tolling Win?

**Caption**: "Hour-of-day analysis of past-3-months Houston market. Left panel: mean LMP vs mean tolling cost by hour. Right panel: percentage of hours in each hour-of-day where the toll cost falls below the LMP (spark spread > 0)."

**What it shows**: The mean LMP (blue bars) follows a strong diurnal pattern — low overnight ($28-30/MWh), rising through morning, dipping midday (solar belly), then peaking at hours 18-19 ($60-67/MWh). The toll cost (red bars) is essentially flat throughout the day at $58-60 (because gas prices don't change within a day). The right panel shows the toll wins in 30-60% of evening peak hours (17-20) but less than 5% in all other hours.

**How it's computed**: Hourly LMPs grouped by hour-of-day across the 3-month window, mean and exercise-frequency computed per group.

**What it means**: The toll is an evening-peak hedge — almost exclusively. Operational implications: any flexible compute load (like the 500 MWh/day training requirement) should be scheduled to AVOID hours 17-20 and concentrate in midday or overnight. The toll provides procurement insurance specifically against the AC + solar-dropoff collision in ERCOT.

---

### Figure 3: Seasonality Heatmap

**Caption**: "Heatmap of mean hourly LMP at HB_HOUSTON (left) and HB_WEST (right) over 2025. Rows = hour of day, columns = month of year. Red cells indicate scarcity, blue cells indicate oversupply."

**What it shows**: A bright red band emerges in hours 18-20 from May through September, peaking in August (mean LMP $90-100). A deep blue valley sits in hours 10-15 of the same months — the solar belly. A smaller secondary peak appears in hours 6-7 of January-February (winter morning heating ramp). At HB_WEST the pattern is similar but with deeper midday valleys due to higher wind/solar penetration.

**How it's computed**: 2025 hourly LMPs grouped into 24 × 12 = 288 hour-of-day × month cells; cell means displayed.

**What it means**: This is the seasonal × diurnal structure the Monte Carlo model captures. August evening hours generate disproportionate option value — not because they happen every day, but because their mean LMP is 2-3x the year-round average. Volatility around the mean is also higher in these cells. The toll captures these peaks through its option-like payoff structure.

---

### Figure 4: Hourly IMHR Distribution

**Caption**: "Distribution of hourly implied market heat rate at HB_HOUSTON over the past 3 months. Left panel uses HSC + $3 VOM (Texas-realistic); right panel uses Henry Hub + $3 VOM (project specification). Vertical dashed lines mark CCGT (6.8), Antelope Station (9.2), and peaker SCGT (9.5) heat rates."

**What it shows**: Both distributions are right-skewed, with most observations clustered between IMHR=2 and IMHR=8. The bulk of mass lies BELOW the 9.5 peaker line. Under the project's Henry Hub specification, median IMHR is 4.56 MMBtu/MWh, with 7.7% of hours exceeding 9.5 (peaker ITM) and 20.0% exceeding 6.8 (CCGT ITM). Under HSC, slightly fewer hours are ITM because HSC trades at a premium to HH. Long right tail extends past 30 MMBtu/MWh — these are scarcity hours that drive option value.

**How it's computed**: For each hour t, IMHR = LMP(t) / (Gas(t) + $3). Computed on 2,190 hours of recent data.

**What it means**: The peaker heat rate of 9.5 is far above typical market conditions — confirming that the toll is OUT-of-the-money the vast majority of the time. Value is concentrated in the right tail. The CCGT comparison (20% ITM vs 8% for peaker) shows that more efficient generation captures meaningfully more of the option value, which is the foundation for the Clean Spark Spread analysis (EC-d).

---

### Figure 5: Intermittency Signatures

**Caption**: "Left: HB_HOUSTON diurnal LMP profile with quantile bands (median + 25-75th + 75-95th + 95-99th + mean overlay). Right: Percentage of HB_WEST hours with negative LMP, by month of year."

**What it shows**: The left panel reveals dramatic tail-fanning at hours 17-21 — the median stays around $30/MWh, but the 95-99th percentile band reaches $230+/MWh. Hours 10-15 show compressed bands (the solar belly squeezes both upside and downside). The right panel shows HB_WEST negative LMPs are heavily seasonal: 15.3% of March hours are negative (peak wind season) vs 0% in July-September (low wind, high AC load).

**How it's computed**: Quantiles computed across all dates within each hour-of-day for left panel. Negative LMP indicator aggregated by month for right panel.

**What it means**: This is the structural basis for the toll option's value. The asymmetric tail at evening hours means the option payoff is highly skewed — a "normal" evening hour is unremarkable, but a "bad" evening hour can pay 5-10x normal. Renewable intermittency creates this asymmetry: solar reliably drops out at sunset, but wind variability creates additional dispersion around that pattern. The negative LMP frequency at West proves that wind oversupply is a real phenomenon — and explains why financial HRCOs at West are uniquely valuable (free-ish gas at Waha combined with high LMP volatility).

---

### Figure 6: Simulated Monte Carlo Paths

**Caption**: "Sample of 50 simulated paths (gray) for each of 4 state variables over the operating period, overlaid with 5th-95th percentile fan (blue band) across all 5,000 simulated paths, median path (blue line), and historical mean (red dashed)."

**What it shows**: The power panels (HB_HOUSTON, HB_WEST) display dense vertical activity reflecting strong diurnal cycling. Median paths stay close to historical means (red dashed lines), confirming proper seasonal calibration. Individual gray paths spike into $200-400/MWh territory occasionally — these are simulated scarcity hours. The gas panels (HSC, HH) show smooth widening cones with median tracking historical mean — consistent with near-random-walk behavior and slow mean reversion.

**How it's computed**: Joint Monte Carlo simulation using calibrated seasonal-OU model with empirical innovation correlation matrix. 5,000 paths simulated for QC purposes; full 10,000 used for valuation.

**What it means**: This validates the simulation engine before using it for valuation. Three key validation checks: (a) median doesn't drift — seasonality decomposition is correct; (b) cone width is reasonable — gas paths reach the historical range of $1-$10 over 6 months; (c) spikes occur but bounded at ~$400 — the no-jumps limitation is visible, which biases option values downward and motivates the separate stress test in Figure 11.

---

### Figure 7: Real Options Analysis (the central exhibit)

**Caption**: "Real options framework for the Houston tolling agreement. Left: annualized cost comparison across strategies — static grid-only ($31.5M), static toll-only ($52.9M), dynamic optimal ($28.7M), with real option premium of $2.84M. Right: real option premium as a function of volatility multiplier (1.0 = calibrated base case), demonstrating the convexity of option value in underlying volatility."

**What it shows**: The left panel shows that always-toll is dramatically worse than always-grid ($52.9M vs $31.5M), reflecting that the toll loses money on average. But dynamic optimization — using whichever is cheaper hour-by-hour — costs only $28.7M, beating both static strategies. The $2.84M difference between the cheaper static (grid-only) and dynamic is the real option premium. The right panel shows option value scales steeply with volatility: at 50% of calibrated vol, premium is $1.4M; at 200%, it's $9.6M. The relationship is convex.

**How it's computed**: For each MC path, compute grid-only cost (Σ LMP × MW × hr), toll-only cost (Σ toll × MW × hr), and dynamic optimal (Σ min(LMP, toll) × MW × hr). Annualize from 6-month sim window. Premium = grid_only − dynamic. Vol sensitivity reruns full MC at scaled volatility parameters.

**What it means**: This is the formal real options finding. The toll has zero intrinsic value at current prices (current spark is −$37/MWh — deeply out-of-the-money). 100% of the toll's value is time value from price uncertainty. The convexity of value in volatility is the defining real-options diagnostic — option holders BENEFIT from uncertainty because the payoff is convex (max(0, ·) cuts off downside while keeping upside). This justifies framing the analysis through Dixit-Pindyck / Trigeorgis real options theory rather than as a simple cost comparison.

---

### Figure 8: Monthly Spark Spread Strip Values

**Caption**: "Left: Monthly value of the 100-MW spark spread call option strip for June-November 2026, computed via 10,000-path Monte Carlo simulation. Bars show mean monthly value; error bars show 5th-95th percentile range across paths. Blue bars use HSC gas; orange bars use Henry Hub gas. Right: Strike sensitivity for the HSC-priced strip at K = $0, $10, $25, $50, $100 per MWh."

**What it shows**: August 2026 is the highest-value month at $389K (HSC) / $359K (HH). June, July, and October cluster around $200-250K. September and November are lowest at $165-220K. The 5th-95th percentile band on each bar shows substantial dispersion — a "bad" simulated path might yield $50K in August while a "good" path yields $750K+. Right panel shows convex decay with strike — at K=$0, August is $389K; at K=$50, only $55K; at K=$100, just $9K.

**How it's computed**: For each path, sum hourly payoffs `max(0, LMP − toll − K) × 100 MW` within each month. Take mean and quantiles across paths.

**What it means**: The headline number is $1.42M total for the 6-month strip (HSC basis) or $1.27M (Henry Hub basis). This is the maximum lease fee at which holding the toll has positive expected value. The August premium ($389K vs $170-250K elsewhere) confirms the seasonality story from Figure 3. The strike sensitivity validates K=0 as the appropriate strike — raising K eliminates moderately profitable hours that the data center would still want to exercise. The Houston physical toll, in business terms, is worth ~$1.4M for the operating window, or $2.84M annualized.

---

### Figure 9: HB_HOUSTON vs HB_WEST Comparison

**Caption**: "Two-panel comparison of HB_HOUSTON and HB_WEST hub price dynamics over the past 3 months. Each panel shows daily-mean LMP (line) with a band extending from daily-min to daily-95th percentile of hourly LMPs."

**What it shows**: Both hubs have similar daily-mean LMPs (Houston $35, West $34) but completely different distributions around the mean. Houston's band extends upward into spike territory ($80-150/MWh) while staying positive at the lower edge. West's band extends DOWN into negative territory frequently (-$10 or below) while having moderate upside.

**How it's computed**: Daily aggregations of hourly ERCOT DAM prices for each hub over the past 90 days.

**What it means**: The two hubs require different strategic treatment despite similar mean prices. Houston demands a scarcity-hedging instrument (a toll or HRCO with significant upside capture). West demands recognition that compute is often free or sub-free (negative LMPs). For the data center, this means: shift training compute to West when West LMP is below Houston LMP (29% of recent days); use the toll at Houston when scarcity hits. The strategic asymmetry is fundamental and shapes every subsequent recommendation.

---

### Figure 10: HB_WEST Market Conditions (the fixed 4-panel exhibit)

**Caption**: "Four-panel view of West Texas market conditions over the past 3 months. (A) HB_WEST daily LMP with negative-price shading. (B) Waha gas daily spot with Henry Hub reference. (C) West minus Houston LMP spread (location-arbitrage signal). (D) Winsorized 'financial' IMHR for HRCO valuation context."

**What it shows**: Panel A: West LMP stays positive on average ($36) but visits very low values; one date dips to $5.50. Panel B: Waha gas was negative on 59 of 91 days (65%), dipping to -$9/MMBtu, while Henry Hub stayed in the $3-4 range. Panel C: West was cheaper than Houston on 29% of days, with spreads up to -$15/MWh. Panel D: Winsorized IMHR ranges far above the peaker and CCGT thresholds — the dashed lines.

**How it's computed**: Panel B and C straightforward differences/spot data. Panel D winsorizes IMHR at ±50 because near-zero or negative gas values produce mathematically meaningless extreme IMHRs.

**What it means**: This is the West Texas honest assessment. There's no physical generator to toll. Waha gas is structurally cheap due to pipeline takeaway constraints from Permian basin associated gas. The economic relevance is two-fold: (1) raw LMP for procurement (which has moderate but positive cost); (2) financial HRCOs at West are valuable precisely because of the Waha gas dynamics — but these are paper contracts, not physical operations. The West "IMHR" panel is properly captioned as a financial reference only — it cannot be used the same way as Houston's IMHR.

---

### Figure 11: Uri Stress Test

**Caption**: "Annualized procurement cost comparison under four stress scenarios for the Houston data center. Red bars: grid-only procurement cost. Green bars: optimal procurement (cheaper of grid or toll each hour). Text annotations show toll-driven savings under each scenario."

**What it shows**: Mean savings rise from $2.3M (baseline, no stress) to $3.1M (mild cold snap), $3.9M (moderate Uri-style), and $4.7M (full Uri replay) as stress severity increases. Crucially, while the MEAN scales modestly, the 95th-percentile (tail) savings rise dramatically: $4.1M baseline → $52M in the full Uri replay scenario. Conditional on Uri actually occurring (5% probability), toll saves $48M in that year alone.

**How it's computed**: Apply Uri-like overlays (72-100 hour spikes at $200-9,000/MWh with corresponding gas spikes) to a subset of MC paths based on annual probabilities. Compute grid-only and optimal cost per path, take mean and quantiles.

**What it means**: The toll is fundamentally tail insurance, not an average-cost hedge. In ordinary conditions it saves a few million dollars; in extreme conditions it can save tens of millions. BUT this analysis assumes firm gas supply. In actual Uri 2021, gas wellhead freeze-offs took down many generators precisely when prices spiked. A non-firm gas toll would be WORTHLESS in the exact scenarios it's supposed to protect against — potentially even loss-making if the toll obligates fixed payments without delivery. Firm gas supply is therefore a non-negotiable contract requirement.

---

### Figure 12: Clean Spark Spread @ $50/ton CO₂ (EC-d)

**Caption**: "Left: Monthly value of the 100-MW clean spark spread option strip under $50/ton CO₂ for three generator technologies — peaker SCGT (HR=9.5), Antelope Station (HR=9.2), modern CCGT (HR=6.8). Right: Marginal cost stack (fuel + CO₂) for each technology vs average HB_HOUSTON LMP."

**What it shows**: Without carbon pricing, peaker tolling generates ~$1.42M over 6 months. With $50/ton CO₂ added, peaker value drops to $452K — a 68% reduction. Antelope ($1.78M) and CCGT ($3.18M) suffer smaller proportional losses (57-69%). The cost stack reveals why: peaker emits 0.504 tons CO₂/MWh, paying $25/MWh in carbon costs. CCGT at 0.361 tons/MWh pays only $18/MWh. The combination of lower heat rate AND lower emission intensity makes CCGT increasingly dominant under carbon pricing.

**How it's computed**: Add carbon cost (HR × 0.0531 ton/MMBtu × $50/ton) to the marginal cost of each technology. Re-compute spark spread payoffs hour-by-hour.

**What it means**: Carbon pricing fundamentally reshapes the analysis. The physical peaker SCGT tolling agreement loses two-thirds of its value at $50/ton, transforming an interesting hedge into a marginal proposition. CCGT tolling becomes the dominant technology choice. Strategically: any long-term tolling commitment carries regulatory risk that should be priced into the lease fee, and the carbon-resilience of CCGT makes it the better choice for future investments.

---

### Figure 13: Waste Heat Recovery NPV (EC-e)

**Caption**: "Left: Maximum justifiable capital expenditure (NPV=0) for waste heat recovery retrofit under four valuation frameworks — Framework A (incremental MW sold at LMP) for Houston and West, plus Framework B (spark-spread-amplifier) under peaker and CCGT economics. Error bars: 5th-95th percentile across MC paths. Right: Sensitivity heatmap of max capex to discount rate (6-15%) and project life (10-25 years)."

**What it shows**: Under Framework A (the recommended interpretation — incremental MW sold at LMP), Houston supports up to $7.70M in capex, West $7.96M. Framework B with peaker economics is much more restrictive ($670K) because the peaker rarely runs. Framework B with CCGT economics supports $3.52M. The sensitivity heatmap shows the base-case $7.70M figure ranges from $6.22M (12% WACC, 10-year life) to $9.83M (6% WACC, 25-year life).

**How it's computed**: Compute annual revenue per path as net MW (3.4 = 4 gross × 85% after aux load) × hours × LMP. Annuitize at 10% over 15 years using standard PV factor of 7.606. Max capex = annual revenue × PV factor.

**What it means**: A waste heat recovery retrofit is economically marginal but defensible at the base case. Real-world heat pump + Organic Rankine Cycle systems retrofit at $2-5M per MW, putting a 3.4 MW project at $7-17M turnkey. The $7.70M max capex sits at the lower end of this range, suggesting the retrofit pencils only with negotiated equipment pricing or government incentives. This is an American real option (the data center can defer the investment indefinitely), so the right strategy is to monitor LMP trends and exercise only when LMPs rise enough to make $7M+ capex clearly NPV-positive.

---

### Figure 14: Quanto Hedge (EC-h)

**Caption**: "Linear-combination hedge replicating contingent gas storage value during extreme weather + outage events. Left: monthly target payoff (red) vs hedge replication (blue) using HDD + CDD + HRCO instruments. Right: scatter of hedge payoff vs target payoff across all path × month combinations, with 45° perfect-hedge line."

**What it shows**: The hedge tracks seasonal pattern reasonably well — replicating winter and summer peaks while staying near zero in mild months. R² = 0.46 and variance reduction of 27%. The right-panel scatter shows mostly tight clustering around zero (most months not extreme) with a tail of high-target points where the hedge under-replicates the magnitude (points above 45° line). OLS-fit weights: α (HDD) = $0.05/HDD, β (CDD) = $0.05/CDD, γ (HRCO) = -0.018 notional units.

**How it's computed**: Proxy temperature from power price residuals (high z-score = anomalous weather). Compute HDD = max(0, 65 − T) and CDD = max(0, T − 65). HRCO payoff = max(0, LMP − Gas × 9.5). Target = (extreme weather indicator) × (LMP top 5%) × (3 × gas spot). Solve OLS: target = α × HDD + β × CDD + γ × HRCO + intercept.

**What it means**: A modest but real fraction of contingent storage value can be replicated by liquid hedge instruments. The methodology demonstrates how to construct a quanto hedge for power-weather-correlated risks. Real implementation would use NOAA HDD/CDD data for Houston and Midland (instead of our power-derived proxy), which would improve effectiveness toward 60-70%. The framework is the right approach for hedging gas storage exposure conditional on operational availability — a non-trivial problem because the contingency itself depends on weather extremes.

---

## Part 4: HRCO Table (no figure — present as table in Section E)

Since HRCO values are best shown as a table, here's the format for your report:

### Table 1: HRCO Strip Values by Heat Rate Strike and Location

| Heat Rate Strike | Houston (HSC) | Houston (HH) | West (Waha) |
|---|---|---|---|
| **6 months total** | | | |
| HR = 7.0 | $7.3M | $6.5M | **$17.6M** |
| HR = 9.5 | $5.5M | $4.7M | **$18.9M** |
| HR = 12.0 | $4.2M | $3.5M | **$20.4M** |

Key observation: **West HRCO at HR=9.5 is worth $18.9M / 6 months, or 13.4× the Houston physical toll ($1.42M).** This is because Waha gas trades near zero, making the gas-side of the HRCO formula contribute almost nothing while West LMP retains positive expected values.

---

## Part 5: Final Recommendations Section

Order these by strength of conclusion:

1. **Hold the Houston physical toll**, valued at $1.42M for the 6-month operating window ($2.84M annualized). This represents the maximum fixed lease fee at which the toll has positive NPV.

2. **Build a large financial HRCO position at West Texas**. At HR strike 9.5, the West HRCO is worth $18.9M over 6 months — more than 13× the Houston physical toll's value. This is the strongest investment recommendation in the analysis.

3. **Demand firm gas supply** as a contract requirement on any physical Houston tolling agreement. Non-firm gas inverts the tail-insurance value in extreme scenarios (Winter Storm Uri analog).

4. **Invest in waste heat recovery up to $7.7M per site** if turnkey equipment pricing can be negotiated to that level. The investment is marginal at base case but provides additional optionality.

5. **Price carbon regulatory risk into long-term commitments**. At $50/ton CO₂, peaker tolling loses 68% of its value. CCGT tolling is much more carbon-resilient and should be preferred for any new long-term investments.

6. **Schedule training compute to avoid hours 17-20** at Houston and shift to West when West LMP < Houston LMP (29% of days based on recent history). This implements the embedded "switch location" and "time-shift training" real options identified in the embedded options inventory.

---

## Part 6: Methodology Limitations (be transparent)

1. **No jumps in the OU model** — underestimates extreme scarcity events like Feb 2025 ($772/MWh spike). Real toll value is therefore biased downward. Addressed via separate Uri stress test in Figure 11.

2. **Day-ahead market data, not real-time** — RTM has fatter tails. Another conservative bias on option value.

3. **Firm gas supply assumed** — actual contracts may not deliver during extreme weather (Uri lesson). This is the single largest practical risk not priced into our analysis.

4. **Margrabe diagnostic was inappropriate** — Margrabe values a single European exchange, but our toll is a 4,392-hour strip with hourly resolution. We removed the comparison from Figure 7 as misleading.

5. **Proxy temperature in quanto hedge** — real implementation requires NOAA HDD/CDD data for Houston/Midland.

6. **Backward-looking calibration assumes regime stability** — ERCOT is adding renewables rapidly. Higher renewable penetration will likely INCREASE option value over time (more intermittency = more spark spread vol). Our estimates are conservative on this dimension.
