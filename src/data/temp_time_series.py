# src/data/efficiency_timeseries.py
import os
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from src.utils.paths import *

# Config (internal)
WEEK_ANCHOR = "W-MON"
_BATCH_ROWS = 5_000_000  # fixed internal chunk size (not user-facing)


# =========================
#           Utils
# =========================
def iter_parquet_batches(path: str):
    """Stream a parquet file as pandas DataFrames in Arrow batches."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
        yield batch.to_pandas()


def ensure_video_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee presence of the minimal columns; fill missing with neutral defaults."""
    must_have = ["channel_id", "upload_date", "like_count", "dislike_count", "view_count", "num_comms"]
    for c in must_have:
        if c not in df.columns:
            df[c] = 0
    return df


def coerce_numeric_video_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric video columns safely."""
    for c in ["like_count", "dislike_count", "view_count", "num_comms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def add_week_bucket(df: pd.DataFrame, date_col: str = "upload_date", anchor: str = WEEK_ANCHOR) -> pd.DataFrame:
    """Add week_start column aligned to the chosen anchor."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["channel_id", date_col])
    df["week_start"] = df[date_col].dt.to_period(anchor).dt.start_time
    return df


def aggregate_weekly_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one chunk to (channel_id, week_start) weekly sums. (Temporary columns)"""
    df = df.copy()
    df["reactions"] = df["like_count"] + df["dislike_count"] + df["num_comms"]
    grp = (
        df.groupby(["channel_id", "week_start"], as_index=False)
          .agg(R_week=("reactions", "sum"),
               V_week=("view_count", "sum"))
    )
    return grp


def accumulate_weekly(acc: dict, grp: pd.DataFrame) -> None:
    """Accumulate grouped weekly sums into a dict keyed by (channel_id, week_start)."""
    for row in grp.itertuples(index=False):
        key = (row.channel_id, row.week_start)
        entry = acc.get(key)
        if entry is None:
            acc[key] = {"R_week": float(row.R_week), "V_week": float(row.V_week)}
        else:
            entry["R_week"] += float(row.R_week)
            entry["V_week"] += float(row.V_week)


def accumulator_to_weekly_df(acc: dict) -> pd.DataFrame:
    """Convert the small accumulator dict into a DataFrame."""
    if not acc:
        return pd.DataFrame(columns=["channel_id", "week_start", "R_week", "V_week"])
    data = ((cid, wk, v["R_week"], v["V_week"]) for (cid, wk), v in acc.items())
    weekly = pd.DataFrame(data, columns=["channel_id", "week_start", "R_week", "V_week"])
    return weekly


def compute_trailing_efficiency_minimal(weekly: pd.DataFrame, window_weeks: int = 4) -> pd.DataFrame:
    """
    Compute only the needed outputs:
      - channel_id
      - week_start
      - efficiency_28d (strict)
      - efficiency_28d_fill4 (forward-fill up to 4 weeks)
    """
    if weekly.empty:
        return pd.DataFrame(columns=["channel_id", "week_start", "efficiency_28d", "efficiency_28d_fill4"])

    print("→ Computing 4-week trailing efficiency (minimal outputs)...")
    weekly = weekly.sort_values(["channel_id", "week_start"], kind="mergesort").copy()

    # rolling sums (temporary, not returned)
    roll = (
        weekly.groupby("channel_id")[["R_week", "V_week"]]
              .rolling(window=window_weeks, min_periods=1)
              .sum()
              .reset_index(level=0, drop=True)
    )
    R_4w = roll["R_week"]
    V_4w = roll["V_week"]

    eff = np.where(V_4w > 0, R_4w / V_4w, np.nan)
    weekly["efficiency_28d"] = eff
    # forward-fill up to 4 weeks
    weekly["efficiency_28d_fill4"] = (
        weekly.groupby("channel_id")["efficiency_28d"].apply(lambda s: s.ffill(limit=4)).values
    )

    out = weekly[["channel_id", "week_start", "efficiency_28d", "efficiency_28d_fill4"]].copy()
    print("✓ 4-week trailing efficiency computed (minimal).")
    return out


def align_to_week(df: pd.DataFrame, dt_col: str, anchor: str = WEEK_ANCHOR) -> pd.Series:
    """Return a week_start Series for timeseries alignment."""
    return pd.to_datetime(df[dt_col], errors="coerce").dt.to_period(anchor).dt.start_time


def merge_efficiency_minimal(ts: pd.DataFrame, weekly_min: pd.DataFrame) -> pd.DataFrame:
    """Merge ONLY (week_start, efficiency_28d, efficiency_28d_fill4) into timeseries."""
    print("→ Merging efficiency columns into timeseries...")
    out = ts.merge(
        weekly_min,
        left_on=["channel", "week_start"],
        right_on=["channel_id", "week_start"],
        how="left",
    )
    if "channel_id" in out.columns:
        out = out.drop(columns=["channel_id"])
    print("✓ Merge done (minimal columns).")
    return out


# =========================
#    Main entry function
# =========================
def timeseries_preprocessing(timeseries_df: pd.DataFrame, videos_light_processed_path: str) -> pd.DataFrame:
    """
    Stream videos_light parquet (unordered OK), compute weekly channel aggregates out-of-core,
    4-week (~28d) trailing efficiency, and merge ONLY `week_start`, `efficiency_28d`,
    `efficiency_28d_fill4` into the timeseries.
    """

    print("=" * 80)
    print("STEP 5: Processing timeseries + injecting (minimal) efficiency")
    print("=" * 80)

    # --- Clean timeseries & align to weekly bucket ---
    print("→ Cleaning timeseries...")
    ts = timeseries_df.drop_duplicates(subset=["channel", "datetime"]).copy()
    ts["datetime"] = pd.to_datetime(ts["datetime"], errors="coerce")
    for col in ["delta_views", "delta_subs", "delta_videos"]:
        if col in ts.columns:
            ts[col] = ts[col].clip(lower=0)
    ts = ts.sort_values(["channel", "datetime"])
    ts["week_start"] = align_to_week(ts, "datetime", WEEK_ANCHOR)
    print(f"✓ Timeseries cleaned. Shape: {ts.shape}")

    # --- Stream videos_light and accumulate weekly aggregates ---
    print(f"→ Streaming videos_light from: {videos_light_processed_path}")
    agg = defaultdict(lambda: {"R_week": 0.0, "V_week": 0.0})

    batch_count = 0
    for batch in tqdm(iter_parquet_batches(videos_light_processed_path),
                      desc="Streaming videos_light", unit="batch"):
        batch_count += 1
        df = ensure_video_cols(batch)
        df = coerce_numeric_video_cols(df)
        df = add_week_bucket(df, "upload_date", WEEK_ANCHOR)

        grp = aggregate_weekly_chunk(df)  # has R_week, V_week (only)
        accumulate_weekly(agg, grp)

        if batch_count % 5 == 0:
            print(f"  ...processed {batch_count} batches, current agg size: {len(agg):,} channel-weeks")

        del df, grp, batch
        gc.collect()

    print(f"✓ Done streaming videos_light. Total batches: {batch_count}. Total channel-weeks (raw): {len(agg):,}")

    # --- Build weekly df and compute trailing efficiency (minimal columns) ---
    weekly = accumulator_to_weekly_df(agg)
    print(f"→ Built weekly DataFrame: {weekly.shape[0]:,} rows, "
          f"{weekly['channel_id'].nunique() if not weekly.empty else 0:,} channels")
    weekly_min = compute_trailing_efficiency_minimal(weekly, window_weeks=4)

    # --- Merge minimal efficiency columns into timeseries and save ---
    ts = merge_efficiency_minimal(ts, weekly_min)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "timeseries_with_efficiency_minimal.parquet")
    ts.to_parquet(output_path, index=False, compression="snappy")
    print(f"✓ Saved processed timeseries (minimal efficiency) to {output_path}")
    print("→ Final columns now include: ['week_start', 'efficiency_28d', 'efficiency_28d_fill4']")
    print("=" * 80)

    return ts


# =========================
#     Example (CLI)
# =========================
if __name__ == "__main__":
    VIDEOS_LIGHT_PROCESSED = os.path.join(PROCESSED_DATA_DIR, "videos_light_processed.parquet")

    def load_data(path, **kwargs):
        if path.endswith(".tsv") or path.endswith(".tsv.gz"):
            return pd.read_csv(path, sep="\t", **kwargs)
        return pd.read_parquet(path, **kwargs)

    ts_df = load_data(RAW_DATASET_TIMESERIES)
    timeseries_preprocessing(ts_df, VIDEOS_LIGHT_PROCESSED)