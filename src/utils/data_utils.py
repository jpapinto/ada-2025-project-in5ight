"""
Utility functions for data processing operations.
"""

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
