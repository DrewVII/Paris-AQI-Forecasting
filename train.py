"""
Train the AQI forecaster.

    python3 train.py --lr 3e-4 --patience 15

The model forecasts quantiles of future AQI (default: median and 90th
percentile). The median is the point forecast and is scored against
persistence exactly as before. The upper quantile is the warning forecast:
it's trained with an asymmetric loss, so it cannot hedge downward the way a
single MAE-trained forecast does during pollution episodes.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset as ds
from model import AQIForecaster, AQIForecasterAttn

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

ELEVATED = 75  # AQI threshold for "episode" diagnostics


def evaluate(model, loader, resid_scaler):
    """Returns predicted quantile levels (N, H, Q) and true levels (N, H).

    The model outputs standardized residuals from the last observed AQI.
    Standardization is a per-horizon affine map with positive scale, so it
    preserves order: inverting a quantile of the scaled residual gives the
    same quantile of the raw residual.
    """
    model.eval()
    P, T, A = [], [], []
    with torch.no_grad():
        for xb, yb, ab in loader:
            P.append(model(xb.to(DEVICE)).cpu().numpy())
            T.append(yb.numpy())
            A.append(ab.numpy())
    mu, sd = resid_scaler.mu, resid_scaler.sd
    P = np.concatenate(P) * sd[None, :, None] + mu[None, :, None]
    T = np.concatenate(T) * sd + mu
    A = np.concatenate(A).reshape(-1, 1)
    return np.clip(P + A[:, :, None], 0, 500), T + A


def pinball_np(P, T, qs):
    """Mean quantile loss in AQI units. A proper scoring rule: it's minimized
    only by forecasting the true quantiles, so it's the right thing to select
    checkpoints on when the model outputs more than a median."""
    e = T[..., None] - P
    q = np.asarray(qs)[None, None, :]
    return float(np.maximum(q * e, (q - 1) * e).mean())


def metrics(pred, true):
    err = pred - true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae_by_h": np.abs(err).mean(axis=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/geos_data.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--attn", action="store_true", help="use attention pooling")
    ap.add_argument("--hweight", choices=["equal", "linear", "sqrt"],
                    default="equal",
                    help="relative weight given to long lead times in the loss")
    ap.add_argument("--quantiles", default="0.5,0.9",
                    help="comma-separated, must include 0.5 (the point "
                         "forecast); use 0.5 alone for a plain median model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = sorted(float(q) for q in args.quantiles.split(","))
    if 0.5 not in qs:
        raise SystemExit("--quantiles must include 0.5 (the point forecast)")
    if any(not 0 < q < 1 for q in qs):
        raise SystemExit("quantiles must lie strictly between 0 and 1")
    q_med = qs.index(0.5)
    q_hi = len(qs) - 1  # highest quantile doubles as the warning forecast

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sets, meta = ds.load(args.data)
    print(f"device={DEVICE}  features={meta['n_features']}  quantiles={qs}")
    for k, v in sets.items():
        print(f"  {k:5s} {len(v):5d} windows")

    # shuffle=True is safe here: the windows themselves were carved
    # chronologically and never cross a split boundary.
    dl = {
        "train": DataLoader(sets["train"], batch_size=args.batch, shuffle=True),
        "val": DataLoader(sets["val"], batch_size=256),
        "test": DataLoader(sets["test"], batch_size=256),
    }

    Net = AQIForecasterAttn if args.attn else AQIForecaster
    model = Net(meta["n_features"], ds.HORIZON, args.hidden, args.layers,
                args.dropout, n_quantiles=len(qs)).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    # Pinball (quantile) loss. For quantile q, underprediction costs q per
    # unit and overprediction costs (1 - q): at q = 0.9 missing low is 9x
    # worse than missing high, which is exactly what stops the upper
    # forecast from hedging toward typical values. At q = 0.5 it is half
    # the absolute error, i.e. plain median regression.
    #
    # Targets are standardized per horizon, so every lead time contributes
    # equally by default; --hweight shifts emphasis toward long leads.
    h = np.arange(1, ds.HORIZON + 1, dtype=np.float32)
    w = {"equal": np.ones_like(h), "linear": h / h.mean(),
         "sqrt": np.sqrt(h) / np.sqrt(h).mean()}[args.hweight]
    w = torch.as_tensor(w, device=DEVICE)[None, :, None]
    q_t = torch.as_tensor(qs, dtype=torch.float32, device=DEVICE)[None, None, :]

    def lossf(pred, true):
        # pred (B, H, Q), true (B, H)
        e = true.unsqueeze(-1) - pred
        return (torch.maximum(q_t * e, (q_t - 1) * e) * w).mean()

    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb, _ in dl["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        tr = tot / len(sets["train"])

        P, T = evaluate(model, dl["val"], meta["resid_scaler"])
        vmae = metrics(P[..., q_med], T)["mae"]
        vpin = pinball_np(P, T, qs)
        sched.step(vpin)

        # Select on pinball, not median MAE: with several quantiles, an epoch
        # with a good median but a badly calibrated upper quantile shouldn't
        # win just because MAE ignores the upper quantile.
        if vpin < best - 1e-4:
            best, bad, best_ep = vpin, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep <= 10 or ep % 10 == 0:
            flag = "  <- best" if ep == best_ep else ""
            print(f"  ep {ep:3d}  train {tr:.4f}  val MAE {vmae:6.2f}  "
                  f"pinball {vpin:5.2f}{flag}")
        if bad >= args.patience:
            print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    print(f"  best epoch {best_ep} (val pinball {best:.2f})")
    if best_ep <= 3:
        print("  !! best epoch is very early -- lower --lr or raise "
              "--dropout; the model is memorising immediately")

    P, T = evaluate(model, dl["test"], meta["resid_scaler"])
    Pm = P[..., q_med]
    m = metrics(Pm, T)

    bp, bt = ds.persistence_baseline(meta["y_raw"], meta["cuts"]["test"])
    b = metrics(bp, bt)

    print("\n" + "=" * 46)
    print("point forecast (median)")
    print(f"{'':14s}{'MAE':>10s}{'RMSE':>10s}")
    print(f"{'persistence':14s}{b['mae']:10.2f}{b['rmse']:10.2f}")
    print(f"{'GRU':14s}{m['mae']:10.2f}{m['rmse']:10.2f}")
    print(f"{'skill score':14s}{1 - m['mae'] / b['mae']:10.1%}")

    if len(qs) > 1:
        # Calibration: the q-th quantile should sit above the truth a
        # fraction q of the time. If the 0.9 forecast covers only 70%, it's
        # overconfident and not yet a trustworthy warning level.
        print("\ncalibration (share of truth at or below each quantile)")
        for k, q in enumerate(qs):
            cov = float((T <= P[..., k]).mean())
            print(f"  q{q:.2f}  target {q:.0%}   actual {cov:.1%}")

    ep_mask = T > ELEVATED
    calm = ~ep_mask
    if ep_mask.sum() > 20:
        e_m = float(np.abs(Pm - T)[ep_mask].mean())
        e_b = float(np.abs(bp - bt)[ep_mask].mean())
        b_m = float((Pm - T)[ep_mask].mean())
        b_b = float((bp - bt)[ep_mask].mean())
        print(f"\nelevated hours only (true AQI > {ELEVATED}, n={int(ep_mask.sum())})")
        print(f"  median MAE   model {e_m:6.2f}   persistence {e_b:6.2f}   "
              f"skill {1 - e_m / e_b:.1%}")
        print(f"  median bias  model {b_m:+6.2f}   persistence {b_b:+6.2f}")

        # Detection alone is meaningless: forecasting 500 everywhere detects
        # everything. Always read the hit rate beside the false-alarm rate.
        def rates(pred):
            hit = float(((pred > ELEVATED) & ep_mask).sum() / ep_mask.sum())
            fa = float(((pred > ELEVATED) & calm).sum() / max(calm.sum(), 1))
            return hit, fa

        print(f"\n  episode warnings (forecast > {ELEVATED})")
        print(f"  {'':22s}{'hit rate':>10s}{'false alarm':>13s}")
        for name, pred in [("persistence", bp), ("GRU median", Pm)] + (
                [(f"GRU q{qs[q_hi]:.2f}", P[..., q_hi])] if len(qs) > 1 else []):
            hit, fa = rates(pred)
            print(f"  {name:22s}{hit:10.1%}{fa:13.1%}")

    print("\nMAE by lead time (median, AQI units)")
    print(f"  {'lead':>5s}{'model':>9s}{'persist':>9s}{'skill':>9s}")
    for hh in [0, 2, 5, 11, 17, 23]:
        s = 1 - m["mae_by_h"][hh] / b["mae_by_h"][hh]
        print(f"  {hh+1:4d}h{m['mae_by_h'][hh]:9.2f}{b['mae_by_h'][hh]:9.2f}"
              f"{s:8.1%}")

    torch.save(
        {"state_dict": model.state_dict(),
         "features": meta["feature_names"],
         "lookback": ds.LOOKBACK,
         "horizon": ds.HORIZON,
         "quantiles": qs,
         "x_mu": meta["x_scaler"].mu,
         "x_sd": meta["x_scaler"].sd,
         "r_mu": meta["resid_scaler"].mu,
         "r_sd": meta["resid_scaler"].sd,
         "args": vars(args)},
        "aqi_model.pt",
    )
    print("\nsaved -> aqi_model.pt")


if __name__ == "__main__":
    main()