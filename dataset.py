"""
Turns the GEOS-CF hourly CSV into supervised sliding windows for
multi-horizon AQI forecasting.

Convention: given the L hours ending at time t (inclusive), predict AQI at
t+1 ... t+H. Nothing after t ever enters the input, which is what keeps the
split honest.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aqi import add_aqi

LOOKBACK = 48  # hours of history fed to the model
HORIZON = 24  # hours ahead to predict

POLLUTANTS = ["PM25", "PM10", "O3", "NO2", "SO2", "CO"]
MET = ["T", "RH", "U", "V", "PS"]
SUBINDICES = [f"AQI_{p}" for p in POLLUTANTS]


def regularize(df, max_gap=3):
    """Put the series on an exact hourly grid.

    Windows are sliced by row position, so a missing hour would make a
    48-row window silently span 49 hours and shift every target after it.
    Reindexing makes gaps explicit; short ones (<= max_gap hours) are filled
    by time interpolation, and anything longer is left as NaN so load()
    can refuse to build windows across it.
    """
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    n_missing = len(full.difference(df.index))
    df = df.reindex(full)
    df = df.interpolate(method="time", limit=max_gap, limit_area="inside")
    df.index.name = "date"
    return df.reset_index(), n_missing


def trim_warmup(df, cols):
    """Drop leading rows where rolling windows are still filling, then
    insist the remainder is gap-free."""
    ok = df[cols].notna().all(axis=1)
    first = ok.idxmax()
    df = df.loc[first:].reset_index(drop=True)
    bad = df[cols].isna().any(axis=1)
    if bad.any():
        where = df.loc[bad, "date"]
        raise ValueError(
            f"{int(bad.sum())} rows with unfillable gaps, first at "
            f"{where.iloc[0]}. Gaps longer than the interpolation limit need "
            f"handling before windowing (re-download that period, or split "
            f"the series there).")
    return df


def build_features(df):
    """Add derived predictors. Everything here is computable at time t."""
    df = add_aqi(df).copy()

    # Wind as speed + direction. Direction is circular, so sin/cos rather
    # than degrees, which would put a false discontinuity at north.
    df["wind_speed"] = np.hypot(df["U"], df["V"])
    theta = np.arctan2(df["V"], df["U"])
    df["wind_sin"] = np.sin(theta)
    df["wind_cos"] = np.cos(theta)

    # Time of day and season, also circular.
    hour = df["date"].dt.hour
    doy = df["date"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Concentrations are right-skewed; logs make the scaler behave.
    for p in ["PM25", "PM10", "NO2", "SO2"]:
        df[f"log_{p}"] = np.log1p(df[p])

    # Boundary layer height spans ~50 m at night to ~2000 m in the
    # afternoon; log it for the same reason. This is the depth pollution
    # mixes into, so it drives most of the diurnal cycle in surface AQI.
    df["log_ZPBL"] = np.log1p(df["ZPBL"])

    # Short-window deltas carry the trend information a plain level misses.
    for c in ["PM25", "O3", "AQI"]:
        df[f"d3_{c}"] = df[c] - df[c].shift(3)

    return df


FEATURES = (
    ["log_PM25", "log_PM10", "O3", "log_NO2", "log_SO2", "CO"]
    + ["T", "RH", "PS", "log_ZPBL", "wind_speed", "wind_sin", "wind_cos"]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    + SUBINDICES
    + ["AQI", "d3_PM25", "d3_O3", "d3_AQI"]
)
TARGET = "AQI"


class Standardizer:
    """Mean/std fit on training rows only, then applied everywhere."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, x):
        return (x - self.mu) / self.sd

    def inverse(self, x):
        return x * self.sd + self.mu

    @classmethod
    def identity(cls, n):
        """No-op scaler, used while computing the statistics themselves."""
        s = cls()
        s.mu = np.zeros(n)
        s.sd = np.ones(n)
        return s


class AQIWindows(Dataset):
    """Windows with residual targets.

    The model predicts AQI[t+1+h] - AQI[t] rather than the level. This makes
    persistence the all-zeros prediction, so the network starts from that
    floor instead of having to rediscover it, and it fixes the short-horizon
    deficit a level-predicting model shows. Deltas are also roughly
    zero-centred and stationary, where AQI levels drift with season.

    Returns (x, residual_target, anchor). The anchor is the raw AQI at t,
    needed to reconstruct the level at evaluation time.
    """

    def __init__(self, X, y_raw, resid_scaler, lookback=LOOKBACK, horizon=HORIZON):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_raw = np.asarray(y_raw, dtype=np.float64)
        self.rs = resid_scaler
        self.L = lookback
        self.H = horizon

    def __len__(self):
        return len(self.X) - self.L - self.H + 1

    def __getitem__(self, i):
        t = i + self.L - 1  # last observed hour
        anchor = self.y_raw[t]
        resid = self.y_raw[t + 1 : t + 1 + self.H] - anchor
        resid = (resid - self.rs.mu) / self.rs.sd
        return (
            self.X[i : t + 1],
            torch.as_tensor(resid, dtype=torch.float32),
            torch.as_tensor(anchor, dtype=torch.float32),
        )

    def residuals(self):
        """All raw (unscaled) residual targets, for fitting the scaler."""
        n = len(self)
        out = np.empty((n, self.H))
        for i in range(n):
            t = i + self.L - 1
            out[i] = self.y_raw[t + 1 : t + 1 + self.H] - self.y_raw[t]
        return out


def load(path="data/geos_data.csv", lookback=LOOKBACK, horizon=HORIZON,
         val_frac=0.15, test_frac=0.15):
    """Returns train/val/test datasets plus the target scaler and a
    persistence baseline for the test period."""
    df = pd.read_csv(path, parse_dates=["date"])
    df, n_missing = regularize(df)
    if n_missing:
        print(f"  filled {n_missing} missing hour(s) by interpolation")
    df = build_features(df)
    df = trim_warmup(df, FEATURES + [TARGET])

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)

    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    # Chronological, never shuffled. Windows that straddle a boundary are
    # dropped by slicing each segment independently.
    xs = Standardizer().fit(X[:n_train])
    Xs = xs.transform(X)

    cuts = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }

    # Fit the residual scaler on training windows only. One mean/std per
    # horizon, since spread grows with lead time.
    probe = AQIWindows(Xs[cuts["train"]], y[cuts["train"]],
                       Standardizer.identity(horizon), lookback, horizon)
    rs = Standardizer().fit(probe.residuals())

    sets = {
        k: AQIWindows(Xs[s], y[s], rs, lookback, horizon) for k, s in cuts.items()
    }

    meta = {
        "x_scaler": xs,
        "resid_scaler": rs,
        "n_features": len(FEATURES),
        "feature_names": FEATURES,
        "dates": df["date"],
        "cuts": cuts,
        "y_raw": y,
    }
    return sets, meta


def persistence_baseline(y_raw, cut, lookback=LOOKBACK, horizon=HORIZON):
    """Hold the last observed AQI flat across the horizon. Any model that
    cannot beat this has learned nothing useful."""
    seg = y_raw[cut]
    preds, trues = [], []
    for i in range(len(seg) - lookback - horizon + 1):
        t = i + lookback - 1
        preds.append(np.repeat(seg[t], horizon))
        trues.append(seg[t + 1 : t + 1 + horizon])
    return np.array(preds), np.array(trues)