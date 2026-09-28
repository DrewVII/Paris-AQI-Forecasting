# Paris AQI Forecasting

Hourly air-quality forecasting for Paris from NASA GEOS-CF reanalysis.

Given the last 48 hours of pollutant concentrations and meteorology, the model
predicts EPA Air Quality Index for each of the next 24 hours, as a median
(point forecast) and a 90th percentile (warning forecast).

Current skill against a persistence baseline: **~30% lower MAE**, peaking near
32% at 12–18 hours ahead. See [Results](#results) and, importantly,
[What this model actually predicts](#what-this-model-actually-predicts).

---

## Quick start

```bash
python3 -m venv tf-env && source tf-env/bin/activate
pip install earthengine-api pandas numpy torch

earthengine authenticate          # one-time, opens a browser
python3 main.py                   # download ~10k hourly rows
python3 train.py                  # train and evaluate
python3 predict.py                # 24-hour forecast from the latest data
```

Earth Engine needs a Cloud project that is both API-enabled and registered at
<https://code.earthengine.google.com/register>. Set yours in `login.py`.

---

## Layout

| File | Role |
|---|---|
| `main.py` | Entry point: fetch data, optionally train |
| `login.py` | Earth Engine authentication |
| `handle_data.py` | Earth Engine download, unit conversion, chunking |
| `aqi.py` | EPA AQI from concentrations |
| `dataset.py` | Features, sliding windows, chronological splits |
| `model.py` | GRU encoder with quantile head |
| `train.py` | Training loop, baseline comparison, diagnostics |
| `predict.py` | Inference from the most recent rows |

---

## Data

Source: [`NASA/GEOS-CF/v2/ana/tavg1hr`](https://developers.google.com/earth-engine/datasets/catalog/NASA_GEOS-CF_v2_ana_tavg1hr)
— NASA GMAO's global composition analysis, hourly, 0.25° (~27.75 km) grid.

**v2 coverage begins 2025-08-04.** `START_DATE` in `main.py` may be set
earlier; `download_range` skips empty chunks and reports the coverage it
actually got. For a longer record, `NASA/GEOS-CF/v1/rpl/tavg1hr` runs from 2018
but uses different band names (`PM25_RH35_GCC`, `U10M`, `T2M`) and a different
aerosol scheme — see [Extending the record](#extending-the-record).

12 bands are pulled: six pollutants (`PM25_RH35`, `PM10_RH35`, `O3`, `NO2`,
`SO2`, `CO`) plus `T`, `RH`, `U`, `V`, `PS`, `ZPBL`.

Gases arrive as mole fraction and are converted at ingest — ×1e6 for O₃/CO
(ppm), ×1e9 for NO₂/SO₂ (ppb). PM is already µg/m³. The CSV on disk is always
in physical units.

> Band names differ across GEOS-CF versions and collections. `download_data`
> validates the requested bands against the collection before selecting, so a
> rename fails with one readable line instead of a 200-band traceback.

---

## AQI

`aqi.py` implements the US EPA index: each pollutant is averaged over its
prescribed window, truncated to its prescribed precision, mapped onto 0–500
through a breakpoint table, and the overall AQI is the **maximum** of the
sub-indices.

| Pollutant | Window | Units |
|---|---|---|
| PM2.5, PM10 | 24 h | µg/m³ |
| O₃, CO | 8 h | ppm |
| NO₂, SO₂ | 1 h | ppb |

In the Paris series, PM2.5 drives the index ~78% of hours and ozone most of the
rest.

The rolling windows have a consequence that shapes the whole evaluation: **AQI
is autocorrelated by construction.** The 24-hour PM average at 3pm shares 23 of
24 hours with the one at 2pm, which is why persistence is very hard to beat at
short lead times and why skill must be read per horizon.

---

## Model

**Input** `(48, 27)` — 48 hours × 27 features.
**Output** `(24, 2)` — 24 horizons × {median, 90th percentile}.

Features are the six pollutants (logged where skewed), meteorology including
`log_ZPBL`, wind as speed plus sin/cos of direction, circular encodings of hour
and day-of-year, the six AQI sub-indices, and 3-hour deltas.

A 2-layer GRU (64 hidden) encodes the window; a small head emits all horizons
and quantiles in one pass.

Three design choices do most of the work:

**Residual targets.** The model predicts `AQI[t+1+h] − AQI[t]`, not the level.
Persistence therefore becomes the all-zeros prediction, so the network starts
from that floor rather than having to rediscover it. This alone took +1h MAE
from 4.35 to 1.24.

**Direct multi-horizon.** All 24 leads in one forward pass, each with its own
output weights. Autoregressive rollout compounds its own error at this data
size.

**Quantile (pinball) loss.** At q=0.9, underprediction costs 9× overprediction,
so the upper forecast cannot hedge toward typical values the way a single
MAE-trained output does during episodes. Quantiles are built monotone
(`base + cumsum(softplus(·))`) so they can never cross.

Residual targets are standardized **per horizon** — spread grows from ~2.6 AQI
at +1h to ~17 at +24h, so a single scaler would let long leads dominate the
gradient.

---

## Results

From the last full run (median-only variant, 9,885 rows, 27 features):

```
                     MAE      RMSE
persistence         9.04     15.00
GRU                 6.36     11.73
skill score        29.7%

   lead    model  persist    skill
     1h     1.11     1.38   19.6%
     6h     4.94     6.89   28.3%
    12h     6.77     9.96   32.1%
    24h     9.34    12.17   23.3%
```

**Episodes are the weak spot.** On hours where true AQI exceeds 75, MAE is
25.0 vs persistence's 30.7 — only 18.6% skill — and the median forecast
detects elevated hours no better than persistence (60.9% vs 60.0%) while being
*more* biased low (−19.6 vs −16.7). That bias is what the 90th-percentile
output is meant to address; `train.py` reports calibration and a hit rate /
false-alarm table so the tradeoff is visible.

> Read the per-horizon table, not the headline. Persistence scores 1.38 MAE at
> +1h without doing anything, because AQI is a rolling mean. Real skill lives
> at +12h and beyond.

---

## Evaluation discipline

Time-series pipelines fail quietly. These properties are enforced and were
verified numerically, not assumed:

- **Chronological splits** (70/15/15), never shuffled. Test is strictly future.
- **Scalers fit on training rows only**, saved with the checkpoint, reused
  verbatim at inference.
- **Windows never cross a split boundary** — each segment is sliced
  independently.
- **Targets are strictly `t+1 … t+H`.** No feature uses data after `t`.
- **The series is regularized to an exact hourly grid** before windowing.
  Windows are sliced by row position, so a missing hour would silently make a
  48-row window span 49 hours. Gaps ≤3 h are interpolated; longer ones raise
  with the offending timestamp.
- **A persistence baseline is computed on the same windows** and reported
  alongside every metric.

---

## What this model actually predicts

Every value in the training data is GEOS-CF output — a chemical transport
model's estimate, not a measurement. So the network has learned to forecast
**what GEOS-CF will say**, which is not the same as forecasting the air.

Three consequences worth stating plainly:

**Resolution.** A 0.25° cell is ~25 km across. The "Paris" value averages the
whole metropolitan area. Street-level NO₂ beside the Périphérique is several
times higher.

**The real benchmark isn't persistence.** GEOS-CF publishes its own 5-day
forecasts (`NASA/GEOS-CF/v2/fcst/tavg1hr`) from full atmospheric physics. The
honest test is whether this model beats *that*, not whether it beats holding
the last value flat.

**The index is American.** These are US EPA breakpoints. The European EAQI that
AirParif publishes uses different thresholds and aggregation, so the numbers
are not comparable to French public figures. Swap the tables in `aqi.py` if you
need EAQI.

Also note: the test split currently covers late July–September only. Winter PM
episodes — Paris's genuinely bad air days — appear in training but are never
evaluated.

---

## Roadmap

1. **Validate against ground stations.** OpenAQ and the EEA publish hourly
   measured PM2.5, NO₂ and O₃ for Paris. Scoring against those is what turns
   "emulates GEOS-CF" into "predicts air quality."
2. **Benchmark against the GEOS-CF forecast**, and if it wins, feed that
   forecast in as a feature and learn to correct its bias against station data
   — roughly how operational ML air-quality systems are built.
3. **Rolling-origin cross-validation** so every season serves as a test period.
4. **Satellite columns.** TEMPO (`NASA/TEMPO/NO2_L3_V4_QA`) is the natural
   extension but is geostationary over the Americas and **never observes
   Paris**. For Europe the equivalent is Sentinel-5P/TROPOMI
   (`COPERNICUS/S5P/OFFL/L3_NO2` and siblings). The `TropCol_*` bands already
   in GEOS-CF are a clean synthetic stand-in for pretraining the
   column-to-surface mapping.

### Extending the record

`NASA/GEOS-CF/v1/rpl/tavg1hr` covers 2018–2025 — roughly 60k hourly rows. It
needs its own band list and a rename map. Stitching v1 and v2 is **not**
recommended without care: NASA reports that v2 substantially changed the
representation of surface PM2.5 and SO₂, and PM2.5 drives most of this index,
so the join would introduce a step change the model would learn as real.

---

## CLI reference

```bash
python3 main.py [--download] [--train]

python3 train.py
  --lr 3e-4              # learning rate
  --hidden 64            # GRU width
  --layers 2
  --dropout 0.2
  --patience 15          # early-stopping patience
  --attn                 # attention pooling over the window
  --hweight equal|sqrt|linear    # emphasis on long lead times
  --quantiles 0.5,0.9    # must include 0.5
  --seed 0
```

With ~1,400 test windows (≈2 months), differences of a point or two between
configurations are inside the noise. Compare means over several seeds.

---

## Data citation

Keller, C. A., Knowland, K. E., Duncan, B. N., et al. (2021). Description of
the NASA GEOS Composition Forecast Modeling System GEOS-CF v1.0. *Journal of
Advances in Modeling Earth Systems*, 13(4), e2020MS002413.
<https://doi.org/10.1029/2020MS002413>
