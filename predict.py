"""
Forecast the next H hours of AQI from the most recent rows in the CSV.

    python3 predict.py --data data/geos_data.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch

import dataset as ds
from aqi import category
from model import AQIForecaster, AQIForecasterAttn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/geos_data.csv")
    ap.add_argument("--ckpt", default="aqi_model.pt")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    L, H = ck["lookback"], ck["horizon"]
    feats = ck["features"]

    df = pd.read_csv(args.data, parse_dates=["date"])
    df, _ = ds.regularize(df)
    df = ds.trim_warmup(ds.build_features(df), feats)
    if len(df) < L:
        raise SystemExit(f"need at least {L} clean rows, have {len(df)}")

    # Reuse the training scaler: refitting on recent data would silently
    # shift the input distribution the model was calibrated on.
    win = df[feats].to_numpy(dtype=np.float64)[-L:]
    mu, sd = ck["x_mu"], ck["x_sd"]
    x = torch.as_tensor(((win - mu) / sd)[None], dtype=torch.float32)

    qs = ck.get("quantiles", [0.5])
    Net = AQIForecasterAttn if ck["args"].get("attn") else AQIForecaster
    model = Net(len(feats), H, ck["args"]["hidden"], ck["args"]["layers"],
                ck["args"]["dropout"], n_quantiles=len(qs))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with torch.no_grad():
        resid = model(x).numpy()[0]  # (H, Q)
    # Residuals from the last observed AQI; add the anchor back for levels.
    resid = resid * ck["r_sd"][:, None] + ck["r_mu"][:, None]
    anchor = float(df["AQI"].iloc[-1])
    yq = np.clip(resid + anchor, 0, 500)

    med = yq[:, qs.index(0.5)]
    t0 = df["date"].iloc[-1]
    print(f"last observation {t0}  AQI {anchor:.0f} "
          f"({df['AQI_driver'].iloc[-1]})\n")

    out = {"valid_time": [t0 + pd.Timedelta(hours=h + 1) for h in range(H)],
           "AQI": med.round(1),
           "category": [category(v) for v in med]}
    if len(qs) > 1:
        hi = yq[:, -1]
        out[f"AQI_q{int(qs[-1]*100)}"] = hi.round(1)
        out["warning_category"] = [category(v) for v in hi]
    print(pd.DataFrame(out).to_string(index=False))

    print(f"\nmedian peak {med.max():.0f} at +{int(med.argmax())+1}h")
    if len(qs) > 1:
        print(f"q{int(qs[-1]*100)} peak    {yq[:, -1].max():.0f} at "
              f"+{int(yq[:, -1].argmax())+1}h  (plausible worst case)")


if __name__ == "__main__":
    main()