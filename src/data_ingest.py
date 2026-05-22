"""
Data Ingestion Module
=====================
Loads and normalizes all data sources for the project:
  - ERCOT DAM Settlement Point Prices (hourly) for HB_HOUSTON, HB_WEST
  - Henry Hub daily spot prices (EIA)
  - Texas natural gas hub prices: HSC (Houston), Waha (West), Katy, Agua Dulce
  - ERCOT renewable generation (when available)

Returns clean pandas DataFrames with consistent columns:
  - 'datetime' (tz-naive UTC for hourly, date for daily)
  - 'price' ($/MWh for power, $/MMBtu for gas)
  - 'hub' / 'source' for identification
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


# ──────────────────────────────────────────────────────────────────────────────
# ERCOT POWER PRICES
# ──────────────────────────────────────────────────────────────────────────────

def load_ercot_dam_hourly(
    hubs: list[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Load ERCOT DAM Settlement Point Prices at HOURLY resolution.

    Returns long-format DataFrame: columns = [datetime, hub, price]
    Hour Ending convention: 01:00 means the hour from 00:00 to 01:00.
    We convert to Hour Beginning to align with standard conventions.

    Parameters
    ----------
    hubs : list of str
        e.g. ['HB_HOUSTON', 'HB_WEST']. None = all hubs.
    path : Path
        Path to the ERCOT xlsx file.
    """
    if path is None:
        path = DATA_DIR / 'rpt.00013060.0000000000000000.DAMLZHBSPP_2025.xlsx'
    if hubs is None:
        hubs = ['HB_HOUSTON', 'HB_WEST']

    raw = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    raw = raw[raw['Settlement Point'].isin(hubs)].copy()

    # Hour Ending is "01:00" through "24:00". Convert to Hour Beginning (0-23).
    raw['hour_ending'] = raw['Hour Ending'].astype(str).str.split(':').str[0].astype(int)
    raw['hour_beginning'] = raw['hour_ending'] - 1
    raw['date'] = pd.to_datetime(raw['Delivery Date'])
    raw['datetime'] = raw['date'] + pd.to_timedelta(raw['hour_beginning'], unit='h')

    # Handle DST "Repeated Hour Flag" — for the fall back day, hour 2 appears
    # twice; we average. Spring forward: hour 2 is missing (don't need to fix).
    if 'Repeated Hour Flag' in raw.columns:
        raw = (raw.groupby(['datetime', 'Settlement Point'], as_index=False)
                  ['Settlement Point Price'].mean())

    out = (raw[['datetime', 'Settlement Point', 'Settlement Point Price']]
           .rename(columns={'Settlement Point': 'hub', 'Settlement Point Price': 'price'})
           .sort_values(['hub', 'datetime'])
           .reset_index(drop=True))
    return out


def load_ercot_dam_daily(hubs=None, path=None) -> pd.DataFrame:
    """Daily averages by hub. Returns: columns = [date, hub, price]."""
    hourly = load_ercot_dam_hourly(hubs=hubs, path=path)
    daily = (hourly
             .groupby(['hub', pd.Grouper(key='datetime', freq='D')])['price']
             .mean()
             .reset_index()
             .rename(columns={'datetime': 'date'}))
    return daily


# ──────────────────────────────────────────────────────────────────────────────
# NATURAL GAS PRICES
# ──────────────────────────────────────────────────────────────────────────────

def load_henry_hub(path: Path | None = None) -> pd.DataFrame:
    """Henry Hub daily spot price ($/MMBtu) from EIA. Returns: [date, price]."""
    if path is None:
        path = DATA_DIR / 'HH_full.csv'
    df = pd.read_csv(path, skiprows=4)
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


def load_texas_gas(hub: str, path: Path | None = None) -> pd.DataFrame:
    """
    Texas natural gas hub daily price ($/MMBtu).

    Parameters
    ----------
    hub : 'HSC' (Houston Ship Channel), 'Waha' (West Texas),
          'Katy', 'Agua Dulce'
    """
    if path is None:
        path = DATA_DIR / 'teas_nat_gas.xlsx'
    df = pd.read_excel(path, sheet_name=hub, skiprows=6,
                       header=None, names=['Date', 'price', 'chg'])
    # Date is stored as Excel serial number
    df = df[pd.to_numeric(df['Date'], errors='coerce').notna()].copy()
    df['date'] = pd.to_datetime(df['Date'].astype(int), unit='D', origin='1899-12-30')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df[['date', 'price']].dropna().sort_values('date').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED PRICE PANEL
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_panel(
    use_texas_gas: bool = True,
    backfill_gas: bool = True,
) -> pd.DataFrame:
    """
    Build a wide daily panel for analysis. Columns:
      date | hb_houston | hb_west | gas_houston | gas_west | gas_hh

    Parameters
    ----------
    use_texas_gas : bool
        If True, gas_houston = HSC, gas_west = Waha.
        If False, both = Henry Hub (project doc convention).
    backfill_gas : bool
        Forward-fill gas prices over weekends/holidays (gas trades fewer days
        than power).
    """
    power = load_ercot_dam_daily(hubs=['HB_HOUSTON', 'HB_WEST'])
    power_wide = (power
                  .pivot(index='date', columns='hub', values='price')
                  .rename(columns={'HB_HOUSTON': 'hb_houston',
                                   'HB_WEST': 'hb_west'})
                  .reset_index())

    hh = load_henry_hub().rename(columns={'price': 'gas_hh'})

    if use_texas_gas:
        hsc = load_texas_gas('HSC').rename(columns={'price': 'gas_houston'})
        waha = load_texas_gas('Waha').rename(columns={'price': 'gas_west'})
        panel = (power_wide
                 .merge(hsc, on='date', how='left')
                 .merge(waha, on='date', how='left')
                 .merge(hh, on='date', how='left'))
    else:
        panel = (power_wide
                 .merge(hh, on='date', how='left'))
        panel['gas_houston'] = panel['gas_hh']
        panel['gas_west'] = panel['gas_hh']

    if backfill_gas:
        for c in ['gas_houston', 'gas_west', 'gas_hh']:
            if c in panel.columns:
                panel[c] = panel[c].ffill()

    return panel.sort_values('date').reset_index(drop=True)


def build_hourly_panel(use_texas_gas: bool = True) -> pd.DataFrame:
    """
    Build hourly power × daily-gas panel for option pricing.
    Columns: datetime | hb_houston | hb_west | gas_houston | gas_west | gas_hh
    Gas prices are broadcast across the day's 24 hours.
    """
    power = load_ercot_dam_hourly(hubs=['HB_HOUSTON', 'HB_WEST'])
    power_wide = (power
                  .pivot(index='datetime', columns='hub', values='price')
                  .rename(columns={'HB_HOUSTON': 'hb_houston',
                                   'HB_WEST': 'hb_west'})
                  .reset_index())
    power_wide['date'] = power_wide['datetime'].dt.normalize()

    daily_panel = build_daily_panel(use_texas_gas=use_texas_gas)
    daily_panel = daily_panel[['date', 'gas_houston', 'gas_west', 'gas_hh']]

    hourly = power_wide.merge(daily_panel, on='date', how='left')
    return hourly.drop(columns='date')


if __name__ == '__main__':
    # Quick sanity check
    panel = build_daily_panel()
    print(f"Daily panel: {panel.shape}")
    print(panel.head())
    print()
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print()
    print("Missing values per column:")
    print(panel.isna().sum())
    print()
    print("Summary stats:")
    print(panel.drop(columns='date').describe().round(2))
