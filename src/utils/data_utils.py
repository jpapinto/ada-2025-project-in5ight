"""
Utility functions for data processing operations.
"""
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def align_to_weekly_period(datetime_series: pd.Series, anchor: str = "W-MON") -> pd.Series:
    """
    Align datetime series to weekly periods starting on the specified anchor day.
    
    Args:
        datetime_series: Series containing datetime values
        anchor: Week anchor (default: "W-MON" for Monday)
    
    Returns:
        Series with week start dates
    """
    return pd.to_datetime(datetime_series, errors='coerce').dt.to_period(anchor).dt.start_time


def compute_weekly_efficiency(weekly_df: pd.DataFrame, window_weeks: int = 4) -> pd.DataFrame:
    """
    Compute trailing efficiency metrics over a specified window of weeks.
    
    Args:
        weekly_df: DataFrame with columns [channel_id, week_start, reactions_week, views_week]
        window_weeks: Number of weeks for rolling window (default: 4)
    
    Returns:
        DataFrame with added efficiency columns:
        - efficiency_4week: Strict 4-week trailing efficiency
        - efficiency_4week_filled: Forward-filled efficiency (up to 4 weeks)
    """
    if weekly_df.empty:
        return pd.DataFrame(columns=['channel_id', 'week_start', 'efficiency_4week', 'efficiency_4week_filled'])
    
    # Sort by channel and week
    weekly_df = weekly_df.sort_values(['channel_id', 'week_start'], kind='mergesort').copy()
    
    # Compute rolling sums for reactions and views
    rolling_stats = (
        weekly_df.groupby('channel_id')[['reactions_week', 'views_week']]
        .rolling(window=window_weeks, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    reactions_rolling = rolling_stats['reactions_week']
    views_rolling = rolling_stats['views_week']
    
    # Calculate efficiency: reactions / views
    weekly_df['efficiency_4week'] = np.where(
        views_rolling > 0,
        reactions_rolling / views_rolling,
        np.nan
    )
    
    # Forward-fill efficiency up to 4 weeks for channels with gaps
    weekly_df['efficiency_4week_filled'] = (
        weekly_df.groupby('channel_id')['efficiency_4week']
        .apply(lambda x: x.ffill(limit=4))
        .values
    )
    
    return weekly_df[['channel_id', 'week_start', 'efficiency_4week', 'efficiency_4week_filled']]

def calculate_stagnation_rate(series):
    """
    Calculates the percentage of weeks with zero or negative growth.
    
    Note: Consistent with preprocessing, we treat <= 0 as 'Non-Growth'
    to conservatively estimate stagnation versus decline.
    
    Args:
        series (pd.Series): Time-series of deltas (subs or views).
        
    Returns:
        float: Fraction of weeks with <= 0 growth.
    """
    stagnant_weeks = (series <= 0).sum()
    total_weeks = len(series)
    return stagnant_weeks / total_weeks if total_weeks > 0 else 0.0



StrOrPath = Union[str, Path]


def add_structural_factors(
    channels_df: pd.DataFrame,
    timeseries_df: Optional[pd.DataFrame] = None,
    videos_df: Optional[pd.DataFrame] = None,
    videos_dataset: Optional[StrOrPath] = None,
    timeseries_df_path: Optional[StrOrPath] = None,
) -> pd.DataFrame:
    """
    Adds per-channel structural factors to channels_df by merging timeseries-based pace and
    consistency medians, plus optional video-duration median, keyed on 'channel'. Also adds
    log_duration computed from duration_median.

    Note: timeseries_df takes precedence over timeseries_df_path. videos_df takes precedence
    over videos_dataset. If no videos source is provided, duration_median and log_duration
    will be NaN.

    Args:
        channels_df (pd.DataFrame): Must contain a 'channel' column.
        timeseries_df (Optional[pd.DataFrame]): Timeseries data already in memory. Must include:
            'channel', 'posting_pace_8w', 'posting_consistency_8w',
            'posting_pace_4w', 'posting_consistency_4w',
            'posting_pace_12w', 'posting_consistency_12w'.
            If provided, timeseries_df_path is ignored.
        videos_df (Optional[pd.DataFrame]): Video-level data already in memory. Accepts either:
            original form: ['channel_id', 'duration'] or
            normalized form: ['channel', 'duration_sec'].
            If provided, videos_dataset is ignored.
        videos_dataset (Optional[StrOrPath]): Path to a parquet file containing video-level data.
            Expected columns: ['channel_id', 'duration'].
        timeseries_df_path (Optional[StrOrPath]): Path to a parquet file containing timeseries data.
            Only the pace/consistency columns listed above are read.

    Returns:
        pd.DataFrame: A copy of channels_df with additional columns:
            pace8_median, cons8_median,
            pace4_median, cons4_median,
            pace12_median, cons12_median,
            video_count_from_pace,
            duration_median (if videos provided),
            log_duration (log(duration_median) where duration_median > 0).
    """
    if "channel" not in channels_df.columns:
        raise ValueError("channels_df must contain a 'channel' column.")

    # Load timeseries
    if timeseries_df is None:
        if timeseries_df_path is None:
            raise ValueError("Provide either timeseries_df or timeseries_df_path.")
        pace_cols = [
            "channel",
            "posting_pace_8w", "posting_consistency_8w",
            "posting_pace_4w", "posting_consistency_4w",
            "posting_pace_12w", "posting_consistency_12w",
        ]
        timeseries_df = pd.read_parquet(Path(timeseries_df_path), columns=pace_cols).copy()
    else:
        timeseries_df = timeseries_df.copy()

    timeseries_df["channel"] = timeseries_df["channel"].astype(str)

    pace_per_ch = (
        timeseries_df.groupby("channel", as_index=False)
        .agg(
            pace8_median=("posting_pace_8w", "median"),
            cons8_median=("posting_consistency_8w", "median"),
            pace4_median=("posting_pace_4w", "median"),
            cons4_median=("posting_consistency_4w", "median"),
            pace12_median=("posting_pace_12w", "median"),
            cons12_median=("posting_consistency_12w", "median"),
            video_count_from_pace=("channel", "size"),
        )
    )

    # Load videos (optional)
    if videos_df is None and videos_dataset is None:
        dur_per_ch = pd.DataFrame({"channel": [], "duration_median": []})
    else:
        if videos_df is None:
            videos_df = pd.read_parquet(Path(videos_dataset), columns=["channel_id", "duration"]).copy()
        else:
            videos_df = videos_df.copy()

        # Normalize schema
        if "channel" not in videos_df.columns:
            if "channel_id" not in videos_df.columns:
                raise ValueError("videos_df must contain 'channel_id' (or already contain 'channel').")
            videos_df = videos_df.rename(columns={"channel_id": "channel"})

        if "duration_sec" not in videos_df.columns:
            if "duration" not in videos_df.columns:
                raise ValueError("videos_df must contain 'duration' (or already contain 'duration_sec').")
            videos_df = videos_df.rename(columns={"duration": "duration_sec"})

        videos_df["channel"] = videos_df["channel"].astype(str)
        videos_df["duration_sec"] = pd.to_numeric(videos_df["duration_sec"], errors="coerce")

        dur_per_ch = (
            videos_df.groupby("channel", as_index=False)
            .agg(duration_median=("duration_sec", "median"))
        )

    # Merge summaries, then merge onto channels
    ch_struct = pace_per_ch.merge(dur_per_ch, on="channel", how="left")

    out = channels_df.copy()
    out["channel"] = out["channel"].astype(str)
    out = out.merge(ch_struct, on="channel", how="left")

    # Add log_duration from duration_median (only if > 0)
    dur = pd.to_numeric(out["duration_median"], errors="coerce")
    out["log_duration"] = np.log(dur.where(dur > 0))

    return out


def split_channels_by_subscriber_tier_weighted(
    channels_df: pd.DataFrame,
    subs_col: str = "subscribers_cc",
    weights_col: str = "weights",
    small_quantile: float = 0.50,
    mid_upper: float = 100000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits channels into Small and Mid tiers using a weighted quantile cutoff on subscribers.

    Note: Rows with non-finite subscribers/weights or weights <= 0 are excluded from the
    weighted-quantile computation. Tier assignment is applied to rows with non-null subscribers.

    Args:
        channels_df (pd.DataFrame): Input table with subscriber and weight columns.
        subs_col (str): Column name for subscriber counts. Defaults to 'subscribers_cc'.
        weights_col (str): Column name for weights used in the weighted quantile. Defaults to 'weights'.
        small_quantile (float): Weighted quantile used as the upper cutoff for Small.
            Defaults to 0.50 (weighted median).
        mid_upper (float): Upper subscriber bound for the Mid tier (inclusive). Defaults to 100000.0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (small_df, mid_df) where:
            small_df contains rows with subs <= weighted quantile threshold,
            mid_df contains rows with subs > threshold and subs <= mid_upper.
    """
    for col in (subs_col, weights_col):
        if col not in channels_df.columns:
            raise ValueError(f"channels_df must contain '{col}' column.")

    df = channels_df.copy()
    df[subs_col] = pd.to_numeric(df[subs_col], errors="coerce")
    df[weights_col] = pd.to_numeric(df[weights_col], errors="coerce")

    subs = df[subs_col].to_numpy(dtype=float, copy=False)
    wts = df[weights_col].to_numpy(dtype=float, copy=False)

    mask = np.isfinite(subs) & np.isfinite(wts) & (wts > 0)
    if mask.sum() == 0:
        raise ValueError(f"No valid rows to compute weighted quantile (check '{subs_col}' and '{weights_col}').")

    vals = subs[mask]
    ws = wts[mask]

    order = np.argsort(vals)
    vals = vals[order]
    ws = ws[order]

    cum_w = np.cumsum(ws)
    cutoff = cum_w[-1] * float(small_quantile)
    thresh = float(np.interp(cutoff, cum_w, vals))

    valid_subs = df[subs_col].notna()
    small_df = df.loc[valid_subs & (df[subs_col] <= thresh)].copy()
    mid_df = df.loc[valid_subs & (df[subs_col] > thresh) & (df[subs_col] <= mid_upper)].copy()

    return small_df, mid_df
