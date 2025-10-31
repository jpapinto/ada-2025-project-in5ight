import pandas as pd
import numpy as np


def channels_preprocessing(channels_df):

    # Remove duplicates
    channels_df = channels_df.drop_duplicates(subset=['channel'])

    # Datetime conversion
    channels_df['join_date'] = pd.to_datetime(channels_df['join_date'])

    # Handle missing values
    channels_df['subscriber_rank_sb'] = channels_df['subscriber_rank_sb'].fillna(
        -1)
    channels_df['weights'] = channels_df['weights'].fillna(1.0)

    # Create additional features
    channels_df['channel_age_days'] = (
        pd.Timestamp('2019-10-27') - channels_df['join_date']
    ).dt.days

    channels_df['videos_per_day'] = (
        channels_df['videos_cc'] / channels_df['channel_age_days']
    ).replace([np.inf, -np.inf], 0)

    # Can add anything else as well

    return channels_df


def timeseries_preprocessing(timeseries_df):
    # Remove duplicates
    timeseries_df = timeseries_df.drop_duplicates(
        subset=['channel', 'datetime']
    )

    # Convert datetime
    timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'])

    # Handle negative deltas (data quality issues)
    for col in ['delta_views', 'delta_subs', 'delta_videos']:
        timeseries_df[col] = timeseries_df[col].clip(lower=0)

    # Sort by channel and date
    timeseries_df = timeseries_df.sort_values(['channel', 'datetime'])

    return timeseries_df


def videos_preprocessing(videos_df):
    # NOTE: should be valid both for .feather file which omits descriptions, titles and tags,
    # so if add some preprocessing for those columns - check if it exists

    # Remove duplicates
    videos_df = videos_df.drop_duplicates(subset=['display_id'])

    # Convert dates
    videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])

    # Handle missing values
    numeric_cols = ['view_count', 'like_count', 'dislike_count', 'duration']
    for col in numeric_cols:
        if col in videos_df.columns:
            videos_df[col] = pd.to_numeric(
                videos_df[col], errors='coerce').fillna(0)

    # Create engagement features: here we can add whatever we want actually
    if 'like_count' in videos_df.columns and 'dislike_count' in videos_df.columns:
        total_reactions = videos_df['like_count'] + videos_df['dislike_count']
        videos_df['like_ratio'] = np.where(
            total_reactions > 0,
            videos_df['like_count'] / total_reactions,
            np.nan
        )

    # This is what we selected as a metrics for engagement efficiency according to doc
    if 'view_count' in videos_df.columns and 'like_count' in videos_df.columns:
        videos_df['engagement_efficiency'] = np.where(
            videos_df['view_count'] > 0,
            (videos_df['like_count'] + videos_df['num_comms']) /
            videos_df['view_count'],
            np.nan
        )

    return videos_df


def num_comments_preprocessing(num_comments_df):

    # Remove duplicates
    num_comments_df = num_comments_df.drop_duplicates(subset=['display_id'])

    num_comments_df['num_comms'] = pd.to_numeric(
        num_comments_df['num_comms'], errors='coerce').fillna(0)

    return num_comments_df


def complete_data_preprocessing(channels_df, videos_metadata_df, timeseries_df=None, videos_num_comments_df=None, comments_df=None):

    if channels_df is not None:
        channels_df = channels_preprocessing(channels_df)

    if timeseries_df is not None:
        timeseries_df = timeseries_preprocessing(timeseries_df)

    # Merge timeseries and channels
    if timeseries_df is not None and channels_df is not None:
        print("Merging timeseries with channels data...")
        timeseries_df = timeseries_df.merge(
            channels_df, on='channel', how='left')

    # videos metadata original dataset is quite large and heavy,
    # as it includes descriptions, titles and tags

    if videos_num_comments_df is not None:
        videos_num_comments_df = num_comments_preprocessing(
            videos_num_comments_df)

    # Currently, no preprocessing for comments

    # Merge videos_metadata_df and videos_num_comments_df
    if videos_metadata_df is not None and videos_num_comments_df is not None:
        print("Merging videos metadata with number of comments...")
        videos_metadata_df = videos_metadata_df.merge(
            videos_num_comments_df, on='display_id', how='left'
        )

    # Preprocess videos metadata
    if videos_metadata_df is not None:
        print("Preprocessing videos metadata with number of comments included...")
        videos_metadata_df = videos_preprocessing(videos_metadata_df)

    return videos_metadata_df, timeseries_df
