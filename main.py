import argparse
import os

import sys
from train import main as train_main

from handle_data import download_data, download_range
from login import logging_in

PARIS = [2.3522, 48.8566]  # Coordinates for Paris, France
START_DATE = '2023-01-01'
END_DATE = '2026-09-24'
DATA_PATH = 'data/geos_data.csv'


def get_data(force=False):
    """Download the GEOS-CF series if it isn't already on disk."""
    if os.path.exists(DATA_PATH) and not force:
        print(f"{DATA_PATH} already exists. Skipping download.")
        return

    logging_in()
    print("Downloading GEOS-CF data...")
    download_range(PARIS, START_DATE, END_DATE, DATA_PATH)
    print(f"Saved to {DATA_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true',
                    help='force a re-download even if the CSV exists')
    ap.add_argument('--train', action='store_true',
                    help='train the forecaster after fetching data')
    args = ap.parse_args()

    get_data(force=args.download)

    if args.train:
        sys.argv = [sys.argv[0]]  # let train.py parse its own defaults
        train_main()


if __name__ == "__main__":
    main()