import os
import ee
import pandas as pd

COLLECTION = "NASA/GEOS-CF/v2/ana/tavg1hr"
BANDS = ['PM25_RH35', 'PM10_RH35', 'O3', 'NO2', 'SO2', 'CO',
         'T', 'RH', 'U', 'V', 'PS', 'ZPBL']
GASES = ['O3', 'NO2', 'SO2', 'CO']
SCALE = 27750  # GEOS-CF native grid, 0.25 deg


def download_data(location, start_date, end_date, path='data/geos_data.csv'):
    aoi = ee.Geometry.Point(location)

    ic = (ee.ImageCollection(COLLECTION)
          .filterDate(start_date, end_date)
          .filterBounds(aoi))

    # An empty collection makes .first() null, and every call on it fails
    # with an unhelpful "parameter 'image' may not be null". Check first.
    if ic.size().getInfo() == 0:
        print(f"  no images in {COLLECTION} for {start_date} -> {end_date}, skipping")
        return None

    # Fail fast with a readable message instead of a 200-line EE traceback.
    available = set(ic.first().bandNames().getInfo())
    missing = [b for b in BANDS if b not in available]
    if missing:
        raise ValueError(f"Bands not in {COLLECTION}: {missing}")

    rows = ic.select(BANDS).getRegion(aoi, SCALE).getInfo()

    header, records = rows[0], rows[1:]
    df = pd.DataFrame(records, columns=header)

    df['date'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.drop(columns=['id', 'time', 'longitude', 'latitude'])
    df = df.rename(columns={'PM10_RH35': 'PM10', 'PM25_RH35': 'PM25'})

    # GEOS-CF returns gases as mole fraction. Convert once here so the CSV
    # on disk is always in physical units.
    df['O3'] = df['O3'] * 1e6    # ppm
    df['CO'] = df['CO'] * 1e6    # ppm
    df['NO2'] = df['NO2'] * 1e9  # ppb
    df['SO2'] = df['SO2'] * 1e9  # ppb

    df = df.sort_values('date').reset_index(drop=True)
    df = df[['date'] + [c for c in df.columns if c != 'date']]

    n_null = int(df.isna().sum().sum())
    if n_null:
        print(f"  warning: {n_null} null values (no valid pixel at point)")

    save_data_to_csv(df, path)
    return df


def download_range(location, start_date, end_date, path='data/geos_data.csv',
                   chunk='180D'):
    """Pull a long period in chunks. A single getRegion call over several
    years will time out or exceed the payload limit."""
    edges = pd.date_range(start_date, end_date, freq=chunk)
    edges = edges.union(pd.DatetimeIndex([start_date, end_date]))

    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        print(f"  {a.date()} -> {b.date()}")
        part = download_data(location, str(a.date()), str(b.date()), path=None)
        if part is not None and len(part):
            parts.append(part)

    if not parts:
        raise ValueError(f"{COLLECTION} has no data between {start_date} "
                         f"and {end_date}")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset='date').sort_values('date')
    df = df.reset_index(drop=True)
    print(f"  coverage: {df['date'].min().date()} -> {df['date'].max().date()} "
          f"({len(df)} hourly rows)")
    save_data_to_csv(df, path)
    return df


def save_data_to_csv(df, filename):
    if filename is None:
        return
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    df.to_csv(filename, index=False)