"""Input paths and GBD export validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ["location", "year", "age", "sex", "measure", "metric"]
VALUES = ["val", "lower", "upper"]


def batch_files(directory, name):
    parquet = directory / f"{name}.parquet"
    if parquet.is_file():
        return [parquet]
    folder = directory / name
    return sorted([*folder.glob("*.csv"), *folder.glob("*.csv.gz")])


def analysis_paths():
    parser = argparse.ArgumentParser(description="Estimate retinoblastoma burden and lost welfare.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    data, output = (p.expanduser().resolve() for p in (args.data_dir, args.output_dir))
    for input_dir in (data, ROOT / "reference"):
        if input_dir == output or input_dir in output.parents or output in input_dir.parents:
            parser.error("Input and output directories must be separate and non-nested.")
    missing = [f"{n}.parquet or {n}/ CSV partitions" for n in ("R1", "R2", "POP")
               if not batch_files(data, n)]
    missing.extend(n for n in ("05-retinoblastoma.csv", "df_world2.geojson", "Table_10_Retinoblastoma.csv")
                   if not (data / n).is_file())
    if missing:
        parser.error(f"Missing inputs in {data}: {', '.join(missing)}. See data/README.md.")
    return data, output


def read_batch(directory, name, locations):
    frames = []
    for path in batch_files(directory, name):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df = df.rename(columns={f"{key}_id": key for key in [*KEY, "cause", "population_group"]
                                if key not in df.columns})
        required = KEY + VALUES + ([] if name == "POP" else ["cause"])
        missing = set(required).difference(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}. See data/README.md.")
        if name != "POP" and not df.cause.eq(1009).all():
            raise ValueError(f"{path.name}: expected retinoblastoma (cause 1009).")
        if "population_group" in df and not df.population_group.eq(1).all():
            raise ValueError(f"{path.name}: expected the all-population group.")
        frames.append(df[KEY + VALUES])
    data = pd.concat(frames, ignore_index=True)
    axes = [locations, range(1990, 2024), [1, 6], [3] if name == "R2" else [1, 2, 3],
            [44] if name == "POP" else range(1, 7), [1] if name == "POP" else [1, 3]]
    expected = pd.MultiIndex.from_product(axes, names=KEY)
    actual = pd.MultiIndex.from_frame(data[KEY])
    if not actual.is_unique or len(actual) != len(expected) or not actual.isin(expected).all():
        raise ValueError(f"{name}: expected {len(expected):,} distinct study cells. See data/README.md.")
    if (not np.isfinite(data[VALUES]).all().all() or data.lower.lt(0).any()
            or not data.val.between(data.lower, data.upper).all()):
        raise ValueError(f"{name}: expected finite, nonnegative estimates and ordered bounds.")
    return data
