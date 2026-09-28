"""
EPA AQI computation from GEOS-CF surface concentrations.

GEOS-CF gives gases as mole fraction (mol/mol) and PM as ug/m3.
EPA breakpoints expect ppm (O3, CO), ppb (NO2, SO2) and ug/m3 (PM),
each averaged over a pollutant-specific window.
"""

import numpy as np
import pandas as pd

GASES = ["O3", "NO2", "SO2", "CO"]

# (C_lo, C_hi, I_lo, I_hi)
BREAKPOINTS = {
    # PM2.5, 24-hour mean, ug/m3 (2024 revised breakpoints)
    "PM25": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    # PM10, 24-hour mean, ug/m3
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    # O3, 8-hour mean, ppm
    "O3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    # NO2, 1-hour, ppb
    "NO2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    # SO2, 1-hour, ppb (upper bins are formally 24-hour; kept for continuity)
    "SO2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    # CO, 8-hour mean, ppm
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
}

# averaging window in hours, and decimals to truncate to
SPEC = {
    "PM25": (24, 1),
    "PM10": (24, 0),
    "O3": (8, 3),
    "NO2": (1, 0),
    "SO2": (1, 0),
    "CO": (8, 1),
}


def convert_units(df):
    """mol/mol -> ppb for NO2/SO2, ppm for O3/CO. Idempotent-ish: only
    converts if the values look like raw mole fractions."""
    df = df.copy()
    if df["O3"].max() < 1e-3:  # still in mol/mol
        df["O3"] = df["O3"] * 1e6  # ppm
        df["CO"] = df["CO"] * 1e6  # ppm
        df["NO2"] = df["NO2"] * 1e9  # ppb
        df["SO2"] = df["SO2"] * 1e9  # ppb
    return df


def _truncate(x, decimals):
    f = 10.0**decimals
    return np.trunc(x * f) / f


def _subindex(conc, table):
    """Piecewise-linear interpolation onto the AQI scale."""
    out = np.full(len(conc), np.nan)
    c = np.asarray(conc, dtype=float)
    for c_lo, c_hi, i_lo, i_hi in table:
        m = (c >= c_lo) & (c <= c_hi)
        out[m] = (i_hi - i_lo) / (c_hi - c_lo) * (c[m] - c_lo) + i_lo
    # anything above the top bin is beyond-index; clamp rather than drop
    top_c = table[-1][1]
    out[c > top_c] = table[-1][3]
    return out


def add_aqi(df):
    """Append per-pollutant sub-indices and the overall AQI.

    Overall AQI is the max of the sub-indices, per EPA. The first 23 rows
    have incomplete 24-hour windows and are returned as NaN.
    """
    df = convert_units(df).copy()

    sub_cols = []
    for pol, (window, dec) in SPEC.items():
        rolled = df[pol].rolling(window, min_periods=window).mean()
        rolled = _truncate(rolled, dec)
        col = f"AQI_{pol}"
        df[col] = _subindex(rolled.to_numpy(), BREAKPOINTS[pol])
        sub_cols.append(col)

    # An overall AQI is only meaningful once every sub-index has a full
    # averaging window; otherwise the max is taken over a partial set and
    # systematically understates the first day.
    complete = df[sub_cols].notna().all(axis=1)
    df["AQI"] = df[sub_cols].max(axis=1).where(complete)
    # idxmax raises on all-NaN rows in recent pandas, which is exactly what a
    # download gap produces, so compute it on complete rows only.
    df["AQI_driver"] = pd.Series(pd.NA, index=df.index, dtype="object")
    if complete.any():
        df.loc[complete, "AQI_driver"] = (
            df.loc[complete, sub_cols].idxmax(axis=1)
            .str.replace("AQI_", "", regex=False)
        )
    return df


def category(aqi):
    """EPA category label for an AQI value."""
    edges = [50, 100, 150, 200, 300, 500]
    names = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    for e, n in zip(edges, names):
        if aqi <= e:
            return n
    return "Beyond index"


if __name__ == "__main__":
    df = pd.read_csv("data/geos_data.csv", parse_dates=["date"])
    df = add_aqi(df)
    print(df[["date", "AQI", "AQI_driver"]].tail())
    print(df["AQI"].describe())
    print(df["AQI_driver"].value_counts())