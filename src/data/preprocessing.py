import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
from config import (
    PROCESSED_DATA_DIR,
    RAW_DATASET_CHANNELS,
    RAW_DATASET_TIMESERIES,
    RAW_DATASET_VIDEO_METADATA,
    RAW_DATASET_VIDEO_METADATA_LIGHT,
    RAW_DATASET_NUM_COMMENTS,
    DATASET_NUM_COMMENTS,
    DATASET_VIDEO_METADATA,
    DATASET_TAGS,
    DATASET_VIDEO_LIGHT,
    DATASET_CHANNELS,
    DATASET_TIMESERIES
)
from src.data.loader import iterate_videos_metadata_stream, iterate_videos_light_stream, load_data
from src.utils.data_utils import align_to_weekly_period, compute_weekly_efficiency
import os

def apply_data_processing(skip_comments=False, only_tags=True):
    """
    Main function to orchestrate the entire data processing pipeline.

    Processing order:
    1. Process num_comments first (needed for video merging)
    2. Process video_metadata (full dataset + tags) - large file, ~30 min
    3. Process video_light (feather) - saves internally
    4. Process channels using the processed videos - saves internally
    5. Process timeseries and compute 4-week efficiency metrics - saves internally

    Each preprocessing function now saves its own output to data/processed/
    """

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    num_comments_df = None
    if not skip_comments:
        print("=" * 80)
        print("STEP 1: Processing num_comments")
        num_comments_df = load_data(RAW_DATASET_NUM_COMMENTS)
        num_comments_df = num_comments_preprocessing(num_comments_df)
        print(f"  Shape: {num_comments_df.shape}")
        print("=" * 80)
        print()

    # Check and process video_light
    if os.path.exists(DATASET_VIDEO_LIGHT) and skip_comments:
        print("=" * 80)
        print("STEP 3: video_light already processed, skipping")
        print(f"  Path: {DATASET_VIDEO_LIGHT}")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("STEP 3: Processing video_light (feather)")
        video_light_path = video_light_preprocessing(num_comments_df)
        print("=" * 80)
        print()

    # Clear memory
    del num_comments_df
    gc.collect()

    # Check and process channels
    if os.path.exists(DATASET_CHANNELS):
        print("=" * 80)
        print("STEP 4: Loading existing channels (already processed)")
        channels_df = pd.read_parquet(DATASET_CHANNELS)
        print(f"  Shape: {channels_df.shape}")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("STEP 4: Processing channels")
        channels_df = load_data(RAW_DATASET_CHANNELS)
        channels_df = channels_preprocessing(channels_df, DATASET_VIDEO_LIGHT)
        print(f"  Shape: {channels_df.shape}")
        print("=" * 80)
        print()

    # Check and process timeseries
    if os.path.exists(DATASET_TIMESERIES):
        print("=" * 80)
        print("STEP 5: timeseries already processed, skipping")
        print(f"  Path: {DATASET_TIMESERIES}")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("STEP 5: Processing timeseries with efficiency metrics")
        timeseries_df = load_data(RAW_DATASET_TIMESERIES)
        timeseries_df = timeseries_preprocessing(timeseries_df, DATASET_VIDEO_LIGHT)
        print(f"  Shape: {timeseries_df.shape}")
        print("=" * 80)
        print()

    # Check and process video_metadata (full + tags)
    if os.path.exists(DATASET_VIDEO_METADATA) and os.path.exists(DATASET_TAGS):
        print("=" * 80)
        print("STEP 2: video_metadata and tags already processed, skipping")
        print(f"  Video metadata: {DATASET_VIDEO_METADATA}")
        print(f"  Tags: {DATASET_TAGS}")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("STEP 2: Processing video_metadata (full + tags)")
        print("  This may take ~30 minutes due to large file size...")
        video_metadata_preprocessing(only_tags)
        print("=" * 80)
        print()

    print("=" * 80)
    print("DATA PROCESSING COMPLETE!")
    print(f"All processed files saved to: {PROCESSED_DATA_DIR}")
    print("Processed files:")
    print(f"  - {DATASET_NUM_COMMENTS}")
    if not only_tags:
        print(f"  - {DATASET_VIDEO_METADATA}")
    print(f"  - {DATASET_TAGS}")
    print(f"  - {DATASET_VIDEO_LIGHT}")
    print(f"  - {DATASET_CHANNELS}")
    print(f"  - {DATASET_TIMESERIES}")
    print("=" * 80)
    print()



def channels_preprocessing(channels_df, videos_light_processed_path):
    """
    Preprocess channels data and aggregate video statistics.
    
    Steps:
    1. Remove duplicates
    2. Convert datetime columns
    3. Handle missing values
    4. Create new features: channel_age_days, videos_per_day
    5. Aggregate video statistics from processed videos
    """
    
    # Remove duplicates
    channels_df = channels_df.drop_duplicates(subset=['channel'])
    
    # Datetime conversion
    channels_df['join_date'] = pd.to_datetime(channels_df['join_date'])
    
    # Handle missing values
    #channels_df['subscriber_rank_sb'] = channels_df['subscriber_rank_sb'].fillna(-1)
    channels_df['weights'] = channels_df['weights'].fillna(1.0)
    
    # Create new features
    channels_df['channel_age_days'] = (
        pd.Timestamp('2019-10-27') - channels_df['join_date']
    ).dt.days
    
    channels_df['videos_per_day'] = (
        channels_df['videos_cc'] / channels_df['channel_age_days']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Aggregate video statistics from processed videos
    print("Aggregating video statistics from processed videos...")
    channel_stats = defaultdict(lambda: {
        'efficiency_sum': 0.0,
        'efficiency_count': 0,
        'views_sum': 0.0,
        'duration_sum': 0.0,
        'like_count_sum': 0.0,
        'dislike_count_sum': 0.0,
        'num_comms_sum': 0.0,
        'like_ratio_sum': 0.0,
        'like_ratio_count': 0,
        'video_count': 0
    })
    
    # Read from parquet in chunks using pandas
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(videos_light_processed_path)
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=10_000_000), desc="Processing video chunks", unit="batch", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        chunk_df = batch.to_pandas()
        
        if 'channel_id' in chunk_df.columns:
            grouped = chunk_df.groupby('channel_id').agg({
                'engagement_efficiency': ['sum', 'count'],
                'view_count': 'sum',
                'duration': 'sum',
                'like_count': 'sum',
                'dislike_count': 'sum',
                'num_comms': 'sum',
                'like_ratio': ['sum', 'count']
            })
            
            for ch_id in grouped.index:
                stats = channel_stats[ch_id]
                stats['efficiency_sum'] += grouped.loc[ch_id, ('engagement_efficiency', 'sum')]
                stats['efficiency_count'] += grouped.loc[ch_id, ('engagement_efficiency', 'count')]
                stats['views_sum'] += grouped.loc[ch_id, ('view_count', 'sum')]
                stats['duration_sum'] += grouped.loc[ch_id, ('duration', 'sum')]
                stats['like_count_sum'] += grouped.loc[ch_id, ('like_count', 'sum')]
                stats['dislike_count_sum'] += grouped.loc[ch_id, ('dislike_count', 'sum')]
                stats['num_comms_sum'] += grouped.loc[ch_id, ('num_comms', 'sum')]
                stats['like_ratio_sum'] += grouped.loc[ch_id, ('like_ratio', 'sum')]
                stats['like_ratio_count'] += grouped.loc[ch_id, ('like_ratio', 'count')]
                stats['video_count'] += grouped.loc[ch_id, ('engagement_efficiency', 'count')]
        
        del chunk_df
        gc.collect()
    
    # Convert stats dict to dataframe and merge
    stats_df = pd.DataFrame([
        {
            'channel': ch_id,
            'avg_engagement_efficiency': stats['efficiency_sum'] / stats['efficiency_count'] if stats['efficiency_count'] > 0 else 0,
            'total_views': stats['views_sum'],
            'total_duration': stats['duration_sum'],
            'total_like_count': stats['like_count_sum'],
            'total_dislike_count': stats['dislike_count_sum'],
            'total_num_comms': stats['num_comms_sum'],
            'avg_video_like_ratio': stats['like_ratio_sum'] / stats['like_ratio_count'] if stats['like_ratio_count'] > 0 else 0,
            'video_count_processed': stats['video_count']
        }
        for ch_id, stats in channel_stats.items()
    ])
    
    channels_df = channels_df.merge(stats_df, on='channel', how='left')
    
    # Fill any remaining NaN values from the merge
    channels_df['avg_engagement_efficiency'] = channels_df['avg_engagement_efficiency'].fillna(0)
    channels_df['total_views'] = channels_df['total_views'].fillna(0)
    channels_df['total_duration'] = channels_df['total_duration'].fillna(0)
    channels_df['total_like_count'] = channels_df['total_like_count'].fillna(0)
    channels_df['total_dislike_count'] = channels_df['total_dislike_count'].fillna(0)
    channels_df['total_num_comms'] = channels_df['total_num_comms'].fillna(0)
   # channels_df['avg_video_like_ratio'] = channels_df['avg_video_like_ratio'].fillna(0)
    channels_df['video_count_processed'] = channels_df['video_count_processed'].fillna(0)

    # Save to parquet in data/processed/
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    channels_df.to_parquet(DATASET_CHANNELS, index=False, compression='snappy')
    print(f"++ Saved processed channels to {DATASET_CHANNELS}")

    return channels_df






def timeseries_preprocessing(timeseries_df, videos_light_processed_path):
    """
    Preprocess timeseries data and compute channel efficiency metrics.
    
    This function processes the timeseries data and computes 4-week trailing efficiency
    by streaming through the videos_light data to calculate weekly aggregates.
    
    Updated Interpretation:
    - reactions_week: Sum of lifetime reactions (likes + dislikes + comments) for all videos uploaded in a given week.
    - views_week: Sum of lifetime views for all videos uploaded in a given week.
    
    Efficiency Metrics:
    - Reflect how engagement-efficient the videos uploaded in the last 1–4 weeks are, using their lifetime engagement totals.
    - This can help analyze recent creative output and topic shifts over time (supports RQ3).
    
    Steps:
    1. Clean timeseries data (remove duplicates, handle negative deltas)
    2. Align timeseries to weekly periods
    3. Stream videos_light data and compute weekly channel aggregates
    4. Calculate 4-week trailing efficiency metrics
    5. Merge efficiency metrics back into timeseries
    6. Save processed timeseries with efficiency columns
    
    Args:
        timeseries_df: Raw timeseries dataframe
        videos_light_processed_path: Path to processed videos_light parquet file
    
    Returns:
        pd.DataFrame: Processed timeseries with efficiency metrics
    """
    
    print("Cleaning timeseries data...")
    
    # Remove duplicates
    timeseries_df = timeseries_df.drop_duplicates(subset=['channel', 'datetime'])
    
    # Convert datetime
    timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'], errors='coerce')
    
    # Handle negative deltas (data quality issues)
    # This part is documented on the paper of YouNiverse, there cannot be
    # negative deltas, this is a limitation of the dataset.
    for col in ['delta_views', 'delta_subs', 'delta_videos']:
        if col in timeseries_df.columns:
            timeseries_df[col] = timeseries_df[col].clip(lower=0)
    
    # Sort by channel and date
    timeseries_df = timeseries_df.sort_values(['channel', 'datetime'])
    
    # Align timeseries to weekly periods (Monday anchor)
    print("Aligning timeseries to weekly periods...")
    timeseries_df['week_start'] = align_to_weekly_period(timeseries_df['datetime'], anchor='W-MON')
    timeseries_df = timeseries_df.dropna(subset=['channel', 'week_start'])
    
    print(f"++ Timeseries cleaned. Shape: {timeseries_df.shape}")
    
    # Stream videos_light and compute weekly aggregates
    print(f"Streaming videos_light from: {videos_light_processed_path}")
    
    # Accumulator for channel-week aggregates
    # Note: reactions_week = sum of lifetime reactions for videos uploaded that week
    # Note: views_week = sum of lifetime views for videos uploaded that week
    weekly_aggregates = defaultdict(lambda: {'reactions_week': 0.0, 'views_week': 0.0})
    
    parquet_file = pq.ParquetFile(videos_light_processed_path)
    batch_count = 0
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=4_000_000),
                      desc="Processing video batches", 
                      unit="batch",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        batch_count += 1
        batch_df = batch.to_pandas()
        
        # Ensure required columns exist
        required_cols = ['channel_id', 'upload_date', 'like_count', 'dislike_count', 'view_count', 'num_comms']
        for col in required_cols:
            if col not in batch_df.columns:
                batch_df[col] = 0
        
        # Coerce numeric columns
        for col in ['like_count', 'dislike_count', 'view_count', 'num_comms']:
            batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce').fillna(0)
        
        # Align videos to weekly periods
        batch_df['upload_date'] = pd.to_datetime(batch_df['upload_date'], errors='coerce')
        batch_df = batch_df.dropna(subset=['channel_id', 'upload_date'])
        batch_df['week_start'] = align_to_weekly_period(batch_df['upload_date'], anchor='W-MON')
        
        # Calculate lifetime reactions for each video (likes + dislikes + comments)
        batch_df['reactions'] = (
            batch_df['like_count'] + 
            batch_df['dislike_count'] + 
            batch_df['num_comms']
        )
        
        # Aggregate by channel and week
        # reactions_week: sum of lifetime reactions for videos uploaded in this week
        # views_week: sum of lifetime views for videos uploaded in this week
        weekly_batch = (
            batch_df.groupby(['channel_id', 'week_start'], as_index=False)
            .agg(reactions_week=('reactions', 'sum'),
                 views_week=('view_count', 'sum'))
        )
        
        # Accumulate into the aggregator
        for row in weekly_batch.itertuples(index=False):
            key = (row.channel_id, row.week_start)
            weekly_aggregates[key]['reactions_week'] += float(row.reactions_week)
            weekly_aggregates[key]['views_week'] += float(row.views_week)
        
        # Memory cleanup
        del batch_df, weekly_batch, batch
        if batch_count % 5 == 0:
            gc.collect()
    
    print(f"++ Processed {batch_count} batches. Total channel-weeks: {len(weekly_aggregates):,}")
    
    # Convert aggregator to DataFrame
    print("Building weekly aggregates DataFrame...")
    if not weekly_aggregates:
        weekly_df = pd.DataFrame(columns=['channel_id', 'week_start', 'lifetime_reactions_week', 'lifetime_views_week'])
    else:
        weekly_data = [
            {
                'channel_id': channel_id,
                'week_start': week_start,
                'lifetime_reactions_week': stats['reactions_week'],
                'lifetime_views_week': stats['views_week']
            }
            for (channel_id, week_start), stats in weekly_aggregates.items()
        ]
        weekly_df = pd.DataFrame(weekly_data)
    
    print(f"++ Weekly DataFrame: {weekly_df.shape[0]:,} rows, "
          f"{weekly_df['channel_id'].nunique() if not weekly_df.empty else 0:,} channels")
    
    # Compute 4-week trailing efficiency
    print("Computing 4-week trailing efficiency metrics...")
    # Rename columns for compatibility with compute_weekly_efficiency function
    weekly_df_for_efficiency = weekly_df.rename(columns={
        'lifetime_reactions_week': 'reactions_week',
        'lifetime_views_week': 'views_week'
    })
    efficiency_df = compute_weekly_efficiency(weekly_df_for_efficiency, window_weeks=4)
    print(f"++ Efficiency computed. Shape: {efficiency_df.shape}")
    
    # Merge efficiency metrics into timeseries
    print("Merging efficiency metrics into timeseries...")
    timeseries_df = timeseries_df.merge(
        efficiency_df,
        left_on=['channel', 'week_start'],
        right_on=['channel_id', 'week_start'],
        how='left'
    )
    
    # Drop redundant channel_id column if it exists
    if 'channel_id' in timeseries_df.columns:
        timeseries_df = timeseries_df.drop(columns=['channel_id'])
    
    #print(f"++ Merge complete. Final shape: {timeseries_df.shape}")
    
    # ========================================
    # Compute posting pace and consistency metrics (RQ3 structural factors)
    # ========================================
    #print("Computing posting pace and consistency metrics (4w, 8w, 12w windows)...")
    
    # Ensure delta_videos column exists and is numeric
    if 'delta_videos' not in timeseries_df.columns:
        print("  Warning: 'delta_videos' column not found. Setting posting metrics to 0.")
        for window in [4, 8, 12]:
            timeseries_df[f'posting_pace_{window}w'] = 0.0
            timeseries_df[f'posting_consistency_{window}w'] = 0.0
    else:
        # Convert delta_videos to float64 for calculations
        timeseries_df['delta_videos'] = pd.to_numeric(timeseries_df['delta_videos'], errors='coerce').fillna(0)
        
        # Sort by channel and week_start to ensure correct rolling window calculations
        timeseries_df = timeseries_df.sort_values(['channel', 'week_start'])
        
        # Create lagged weekly_videos (shift by 1 to avoid leakage - don't use current week's data)
        timeseries_df['wv_shift'] = timeseries_df.groupby('channel')['delta_videos'].shift(1)
        timeseries_df['wv_shift'] = timeseries_df['wv_shift'].fillna(0)
        
        # Compute rolling metrics for each window size
        for window_weeks in [4, 8, 12]:
            # Posting pace: rolling median of lagged weekly uploads
            pace_col = f'posting_pace_{window_weeks}w'
            timeseries_df[pace_col] = (
                timeseries_df.groupby('channel')['wv_shift']
                .rolling(window=window_weeks, min_periods=1)
                .quantile(0.5, interpolation='nearest')
                .reset_index(level=0, drop=True)
            )
            timeseries_df[pace_col] = timeseries_df[pace_col].fillna(0).astype('float32')
            
            # Posting consistency: 1 - (std / max(mean, 0.1)), clipped to [0, 1]
            # Rolling mean
            rolling_mean = (
                timeseries_df.groupby('channel')['wv_shift']
                .rolling(window=window_weeks, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling std
            rolling_std = (
                timeseries_df.groupby('channel')['wv_shift']
                .rolling(window=window_weeks, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            
            # Compute consistency: clip(1 - std/max(mean, 0.1), 0, 1)
            eps_cov = 0.1
            consistency_col = f'posting_consistency_{window_weeks}w'
            denominator = np.maximum(rolling_mean, eps_cov)
            timeseries_df[consistency_col] = (1.0 - (rolling_std / denominator)).clip(lower=0.0, upper=1.0)
            timeseries_df[consistency_col] = timeseries_df[consistency_col].fillna(0).astype('float32')
        
        # Drop the temporary wv_shift column
        timeseries_df = timeseries_df.drop(columns=['wv_shift'])
    
    # Save to parquet in data/processed
    print(f"Final shape: {timeseries_df.shape}")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    timeseries_df.to_parquet(DATASET_TIMESERIES, index=False, compression='snappy')
    print(f"++ Saved processed timeseries to {DATASET_TIMESERIES}")
    
    return timeseries_df







def video_metadata_preprocessing(only_tags=False):
    """
    Preprocess video metadata (full dataset with titles, descriptions, tags).
    
    This function processes the large video metadata file in chunks to avoid memory issues.
    Saves ONE or TWO datasets depending on only_tags flag:
    1. Full video_metadata_processed.parquet - cleaned and processed video data (if only_tags=False)
    2. tags_processed.parquet - lightweight version with only channel_id, display_id, tags, upload_date
    
    Args:
        only_tags: If True, only save tags_processed.parquet and skip video_metadata_processed.parquet
    
    Steps per chunk:
    1. Remove duplicates
    2. Convert dates
    3. Handle missing values for numeric columns
    4. Parse and handle text fields (titles, descriptions, tags)
    5. Create engagement features
    6. Write to parquet files incrementally
    
    Returns:
        tuple: Paths to (video_metadata_path, tags_path) or (None, tags_path) if only_tags=True
    """
    
    if only_tags:
        print("Processing video_metadata in chunks (TAGS ONLY mode - skipping full metadata)")
    else:
        print("Processing video_metadata in chunks (this may take ~30 minutes)")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Initialize variables for chunk processing
    schema = None
    writer = None
    tags_schema = None
    tags_writer = None
    total_rows = 0
    chunk_count = 0
    
    try:
        # Iterate through chunks from the JSONL file
        for chunk_df in tqdm(iterate_videos_metadata_stream(RAW_DATASET_VIDEO_METADATA, window_size=750_000, start_line=0),
            desc="Processing video metadata chunks",
            unit="chunk",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ):
            # Remove duplicates
            chunk_df = chunk_df.drop_duplicates(subset=['display_id'])
            
            # Convert dates
            chunk_df['upload_date'] = pd.to_datetime(chunk_df['upload_date'], errors='coerce')
            
            # Handle missing values for numeric columns - ensure consistent float64 type
            engagement_cols = ['view_count', 'like_count', 'dislike_count']
            for col in engagement_cols:
                if col in chunk_df.columns:
                    chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0).astype('float64')
                    
            if 'duration' in chunk_df.columns:
                chunk_df['duration'] = pd.to_numeric(chunk_df['duration'], errors='coerce')
                # Discard videos with 0 or NaN duration
                chunk_df = chunk_df[chunk_df['duration'] > 0].copy()
            
            # Handle text fields
            if 'title' in chunk_df.columns:
                chunk_df['title'] = chunk_df['title'].fillna('')
            
            if 'description' in chunk_df.columns:
                chunk_df['description'] = chunk_df['description'].fillna('')
            
            if 'tags' in chunk_df.columns:
                chunk_df['tags'] = chunk_df['tags'].fillna('[]')
            
            # Create engagement features
            if 'like_count' in chunk_df.columns and 'dislike_count' in chunk_df.columns:
                total_reactions = chunk_df['like_count'] + chunk_df['dislike_count']
                chunk_df['like_ratio'] = np.where(
                    total_reactions > 0,
                    chunk_df['like_count'] / total_reactions,
                    np.nan
                )
            
            # Only write full video_metadata if not in only_tags mode
            if not only_tags:
                # Convert full chunk to PyArrow Table and write
                table = pa.Table.from_pandas(chunk_df)
                
                # Initialize writer with schema from first chunk (video_metadata)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(DATASET_VIDEO_METADATA, schema, compression='snappy')
                else:
                    # Handle schema mismatches by aligning columns
                    current_columns = set(table.column_names)
                    expected_columns = set(schema.names)
                    
                    # Add missing columns with null values
                    for col in expected_columns - current_columns:
                        field = schema.field(col)
                        null_array = pa.nulls(len(table), type=field.type)
                        table = table.append_column(field, null_array)
                    
                    # Remove extra columns not in schema
                    for col in current_columns - expected_columns:
                        table = table.drop([col])
                    
                    # Reorder columns to match schema, then cast to ensure consistent types
                    column_order = schema.names
                    table = table.select(column_order)
                    table = table.cast(schema)
                
                # Write chunk to parquet file
                writer.write_table(table)
            
            # Create tags subset - only channel_id, display_id, tags, upload_date
            tags_chunk = chunk_df[['channel_id', 'display_id', 'tags', 'upload_date']].copy()
            tags_table = pa.Table.from_pandas(tags_chunk)
            
            # Initialize tags writer with schema from first chunk
            if tags_writer is None:
                tags_schema = tags_table.schema
                tags_writer = pq.ParquetWriter(DATASET_TAGS, tags_schema, compression='snappy')
            else:
                # Handle schema mismatches by aligning columns
                current_columns = set(tags_table.column_names)
                expected_columns = set(tags_schema.names)
                
                # Add missing columns with null values
                for col in expected_columns - current_columns:
                    field = tags_schema.field(col)
                    null_array = pa.nulls(len(tags_table), type=field.type)
                    tags_table = tags_table.append_column(field, null_array)
                
                # Remove extra columns not in schema
                for col in current_columns - expected_columns:
                    tags_table = tags_table.drop([col])
                
                # Reorder columns to match schema, then cast to ensure consistent types
                tags_column_order = tags_schema.names
                tags_table = tags_table.select(tags_column_order)
                tags_table = tags_table.cast(tags_schema)
            
            # Write tags chunk to parquet file
            tags_writer.write_table(tags_table)
            
            total_rows += len(chunk_df)
            chunk_count += 1
            
            # Clear memory
            del chunk_df
            if not only_tags:
                del table
            del tags_chunk
            del tags_table
            
            if chunk_count % 10 == 0:
                gc.collect()
        
        # Close both writers
        if writer is not None:
            writer.close()
        if tags_writer is not None:
            tags_writer.close()
        
        if only_tags:
            print(f"++ Saved processed tags to {DATASET_TAGS}")
            print(f"  Total rows processed: {total_rows:,}")
            print(f"  Total chunks: {chunk_count}")
            return None, DATASET_TAGS
        else:
            print(f"++ Saved processed video_metadata to {DATASET_VIDEO_METADATA}")
            print(f"++ Saved processed tags to {DATASET_TAGS}")
            print(f"  Total rows processed: {total_rows:,}")
            print(f"  Total chunks: {chunk_count}")
            return DATASET_VIDEO_METADATA, DATASET_TAGS
    
    except Exception as e:
        # Clean up on error
        if writer is not None:
            writer.close()
        if tags_writer is not None:
            tags_writer.close()
        print(f"-- Error processing video_metadata: {e}")
        raise






def video_light_preprocessing(num_comments_df):
    """
    Preprocess video light data (feather file without descriptions, titles, tags).
    
    Reads the feather file in batches directly from disk without loading the entire file.
    Processes in chunks of 20 million rows and writes to compressed parquet.
    
    Args:
        num_comments_df: Pre-loaded num_comments dataframe for merging
    
    Returns:
        str: Path to the saved parquet file
    """
    
    print("Processing video_light in chunks of 20 million rows...")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Initialize variables
    schema = None
    writer = None
    total_rows_processed = 0
    chunk_size = 20_000_000  # 20 million rows per chunk
    
    try:
        # Use RecordBatchStreamReader to read in batches without loading full file

        
        with pa.memory_map(RAW_DATASET_VIDEO_METADATA_LIGHT, 'r') as source:
            reader = ipc.open_file(source)
            total_rows_in_file = reader.num_record_batches
            
            print(f"Total record batches in file: {total_rows_in_file:,}")
            print(f"Processing in chunks...\n")
            
            chunk_idx = 0
            accumulated_batches = []
            accumulated_rows = 0
            
            for batch_idx in range(total_rows_in_file):
                batch = reader.get_batch(batch_idx)
                accumulated_batches.append(batch)
                accumulated_rows += len(batch)
                
                # Process when we hit chunk_size or it's the last batch
                if accumulated_rows >= chunk_size or batch_idx == total_rows_in_file - 1:
                    print(f"Processing chunk {chunk_idx + 1} ({accumulated_rows:,} rows)...")
                    
                    # Combine accumulated batches into a table
                    chunk_table = pa.Table.from_batches(accumulated_batches)
                    chunk_df = chunk_table.to_pandas()
                    
                    # Clear accumulated batches
                    accumulated_batches = []
                    accumulated_rows = 0
                    
                    # Merge with num_comments
                    chunk_df = chunk_df.merge(num_comments_df[['display_id', 'num_comms']], on='display_id', how='left')
                    
                    # Remove duplicates
                    chunk_df = chunk_df.drop_duplicates(subset=['display_id'])
                    
                    # Convert dates
                    chunk_df['upload_date'] = pd.to_datetime(chunk_df['upload_date'], errors='coerce')
                    
                    # Handle missing values for numeric columns - ensure consistent float64 type
                    engagement_cols = ['view_count', 'like_count', 'dislike_count']
                    for col in engagement_cols:
                        if col in chunk_df.columns:
                            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0).astype('float64')
                            
                    if 'duration' in chunk_df.columns:
                        chunk_df['duration'] = pd.to_numeric(chunk_df['duration'], errors='coerce')
                        # Discard videos with 0 or NaN duration
                        chunk_df = chunk_df[chunk_df['duration'] > 0].copy()
                    
                    # Ensure num_comms exists and is filled
                    if 'num_comms' not in chunk_df.columns:
                        chunk_df['num_comms'] = 0
                    else:
                        chunk_df['num_comms'] = pd.to_numeric(chunk_df['num_comms'], errors='coerce').fillna(0)
                    
                    # Create engagement features
                    if 'like_count' in chunk_df.columns and 'dislike_count' in chunk_df.columns:
                        total_reactions = chunk_df['like_count'] + chunk_df['dislike_count']
                        chunk_df['like_ratio'] = np.where(
                            total_reactions > 0,
                            chunk_df['like_count'] / total_reactions,
                            np.nan
                        )
                    
                    # Engagement efficiency metric (likes + dislikes + comments) / views
                    if 'view_count' in chunk_df.columns and 'like_count' in chunk_df.columns:
                        chunk_df['engagement_efficiency'] = np.where(
                            chunk_df['view_count'] > 0,
                            (chunk_df['like_count'] + chunk_df['dislike_count'] + chunk_df['num_comms']) / chunk_df['view_count'],
                            0.0
                        )
                    
                    # Convert to PyArrow Table
                    table = pa.Table.from_pandas(chunk_df)
                    
                    # Initialize writer with schema from first chunk
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(DATASET_VIDEO_LIGHT, schema, compression='snappy')
                    
                    # Write chunk to parquet file
                    writer.write_table(table)
                    
                    total_rows_processed += len(chunk_df)
                    print(f"  ++ Chunk {chunk_idx + 1} processed: {len(chunk_df):,} rows")
                    
                    chunk_idx += 1
                    
                    # Clear memory
                    del chunk_df
                    del chunk_table
                    del table
                    gc.collect()
        
        # Close the writer
        writer.close()
        
        print(f"\n++ Saved processed video_light to {DATASET_VIDEO_LIGHT}")
        print(f"  Total rows processed: {total_rows_processed:,}")
        
        return DATASET_VIDEO_LIGHT
    
    except Exception as e:
        # Clean up on error
        if writer is not None:
            writer.close()
        print(f"-- Error processing video_light: {e}")
        raise


def num_comments_preprocessing(num_comments_df):
    """
    Preprocess number of comments data.
    
    Steps:
    1. Remove duplicates
    2. Convert num_comms to numeric and fill NaN
    3. Save to parquet
    """
    
    # Remove duplicates
    num_comments_df = num_comments_df.drop_duplicates(subset=['display_id'])
    
    # Convert to numeric and fill NaN
    num_comments_df['num_comms'] = pd.to_numeric(
        num_comments_df['num_comms'], errors='coerce'
    ).fillna(0)
    
    # Save to parquet
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    num_comments_df.to_parquet(DATASET_NUM_COMMENTS, index=False, compression='snappy')
    print(f"++ Saved processed num_comments to {DATASET_NUM_COMMENTS}")
    
    return num_comments_df





if __name__ == "__main__":
    # Run the data processing pipeline
    processed_paths = apply_data_processing(skip_comments=False)
