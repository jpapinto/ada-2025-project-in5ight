import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.paths import *
from src.data.loader import iterate_videos_metadata_stream, iterate_videos_light_stream, load_data
import pyarrow.ipc as ipc

def apply_data_processing(skip_comments=False):
    """
    Main function to orchestrate the entire data processing pipeline.

    Processing order:
    1. Process num_comments first (needed for video merging)
    2. Process video_light (feather) - saves internally
    3. Process channels using the processed videos - saves internally
    4. Process timeseries and merge with channels - saves internally
    5. Optionally process full video_metadata (large, takes ~30 min)

    Each preprocessing function now saves its own output to data/processed/
    """

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    num_comments_df = None
    if not skip_comments:
        num_comments_path = os.path.join(PROCESSED_DATA_DIR, 'num_comments_processed.parquet')
        print("=" * 80)
        print("STEP 1: Processing num_comments")
        print("=" * 80)
        num_comments_df = load_data(RAW_DATASET_NUM_COMMENTS)
        num_comments_df = num_comments_preprocessing(num_comments_df)
        print(f"  Shape: {num_comments_df.shape}")
        print()

    # Check and process video_light
    video_light_path = os.path.join(PROCESSED_DATA_DIR, 'videos_light_processed.parquet')
    if os.path.exists(video_light_path) and skip_comments:
        print("=" * 80)
        print("STEP 2: video_light already processed, skipping")
        print("=" * 80)
        print(f"  Path: {video_light_path}")
        print()
    else:
        print("=" * 80)
        print("STEP 2: Processing video_light (feather)")
        print("=" * 80)
        video_light_path = video_light_preprocessing(num_comments_df)
        print()

    # Clear memory
    del num_comments_df
    gc.collect()

    # Check and process channels
    channels_path = os.path.join(PROCESSED_DATA_DIR, 'channels_processed.parquet')
    if os.path.exists(channels_path):
        print("=" * 80)
        print("STEP 3: Loading existing channels (already processed)")
        print("=" * 80)
        channels_df = pd.read_parquet(channels_path)
        print(f"  Shape: {channels_df.shape}")
        print()
    else:
        print("=" * 80)
        print("STEP 3: Processing channels")
        print("=" * 80)
        channels_df = load_data(RAW_DATASET_CHANNELS)
        channels_df = channels_preprocessing(channels_df, video_light_path)
        print(f"  Shape: {channels_df.shape}")
        print()

    # Check and process timeseries
    timeseries_path = os.path.join(PROCESSED_DATA_DIR, 'timeseries_processed.parquet')
    if os.path.exists(timeseries_path):
        print("=" * 80)
        print("STEP 4: timeseries already processed, skipping")
        print("=" * 80)
        print(f"  Path: {timeseries_path}")
        print()
    else:
        print("=" * 80)
        print("STEP 4: Processing timeseries")
        print("=" * 80)
        timeseries_df = load_data(RAW_DATASET_TIMESERIES)
        timeseries_df = timeseries_preprocessing(timeseries_df, channels_df)
        print(f"  Shape: {timeseries_df.shape}")
        print()

    print("=" * 80)
    print("DATA PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"All processed files saved to: {PROCESSED_DATA_DIR}")
    print()
    print("Processed files:")
    print(f"  - {num_comments_path}")
    print(f"  - {video_light_path}")
    print(f"  - {channels_path}")
    print(f"  - {timeseries_path}")
    print()
    print("Note: To process full video_metadata (with text fields), call:")
    print("  video_metadata_preprocessing()")
    print("  (This takes ~30 minutes due to large file size)")
    print()

    return {
        'channels': channels_path,
        'timeseries': timeseries_path,
        'videos_light': video_light_path,
        'num_comments': num_comments_path
    }


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
    channels_df['subscriber_rank_sb'] = channels_df['subscriber_rank_sb'].fillna(-1)
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
    channels_df['avg_video_like_ratio'] = channels_df['avg_video_like_ratio'].fillna(0)
    channels_df['video_count_processed'] = channels_df['video_count_processed'].fillna(0)

    # Save to parquet in data/processed/
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'channels_processed.parquet')
    channels_df.to_parquet(output_path, index=False, compression='snappy')
    print(f"✓ Saved processed channels to {output_path}")

    return channels_df






def timeseries_preprocessing(timeseries_df, channels_df):
    """
    Preprocess timeseries data and merge with channels.
    
    Steps:
    1. Remove duplicates
    2. Convert datetime columns
    3. Handle negative deltas (data quality issues)
    4. Sort by channel and datetime
    5. Merge with channels data
    """
    
    # Remove duplicates
    timeseries_df = timeseries_df.drop_duplicates(subset=['channel', 'datetime'])
    
    # Convert datetime
    timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'])
    
    # Handle negative deltas (data quality issues)
    for col in ['delta_views', 'delta_subs', 'delta_videos']:
        if col in timeseries_df.columns:
            timeseries_df[col] = timeseries_df[col].clip(lower=0)
    
    # Sort by channel and date
    timeseries_df = timeseries_df.sort_values(['channel', 'datetime'])
    
    # Merge with channels data
    print("Merging timeseries with channels data...")
    timeseries_df = timeseries_df.merge(channels_df, on='channel', how='left')

    # Save to parquet in data/processed
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'timeseries_processed.parquet')
    timeseries_df.to_parquet(output_path, index=False, compression='snappy')
    print(f" Saved processed timeseries to {output_path}")

    return timeseries_df






def video_metadata_preprocessing():

    CHANGES:

          SAVE INTO TWO DATASETS
            - THe original video_metadata
            - New df



    """
    Preprocess video metadata (full dataset with titles, descriptions, tags).
    
    This function processes the large video metadata file in chunks to avoid memory issues.
    It reads from the JSONL file, processes each chunk, and writes directly to a parquet file.
    
    Steps per chunk:
    1. Remove duplicates
    2. Convert dates
    3. Handle missing values for numeric columns
    4. Parse and handle text fields (titles, descriptions, tags)
    5. Merge with num_comments
    6. Create engagement features
    7. Write to parquet file incrementally
    
    Args:
        num_comments_df: Optional pre-loaded num_comments dataframe for merging
    
    Returns:
        str: Path to the saved parquet file
    """
    
    print("=" * 80)
    print("Processing video_metadata in chunks (this may take ~30 minutes)")
    print("=" * 80)
    
    # Load num_comments if not provided
    if num_comments_df is None:
        print("Loading num_comments for merging...")
        num_comments_df = load_data(RAW_DATASET_NUM_COMMENTS)
        num_comments_df = num_comments_preprocessing(num_comments_df)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'videos_metadata_processed.parquet')
    
    # Initialize variables for chunk processing
    schema = None
    writer = None
    total_rows = 0
    chunk_count = 0
    
    try:
        # Iterate through chunks from the JSONL file
        for chunk_df in tqdm(iterate_videos_metadata_stream(RAW_DATASET_VIDEO_METADATA, window_size=1_000_000, start_line=0),
            desc="Processing video metadata chunks",
            unit="chunk",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ):
            # Remove duplicates
            chunk_df = chunk_df.drop_duplicates(subset=['display_id'])
            
            # Convert dates
            chunk_df['upload_date'] = pd.to_datetime(chunk_df['upload_date'], errors='coerce')
            
            # Handle missing values for numeric columns
            numeric_cols = ['view_count', 'like_count', 'dislike_count', 'duration']
            for col in numeric_cols:
                if col in chunk_df.columns:
                    chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0)
            
            # Handle text fields
            if 'title' in chunk_df.columns:
                chunk_df['title'] = chunk_df['title'].fillna('')
            
            if 'description' in chunk_df.columns:
                chunk_df['description'] = chunk_df['description'].fillna('')
            
            if 'tags' in chunk_df.columns:
                chunk_df['tags'] = chunk_df['tags'].fillna('[]')
            
            # Merge with num_comments
            chunk_df = chunk_df.merge(num_comments_df[['display_id', 'num_comms']], on='display_id', how='left')
            
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
                    0.0
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
            
            # Initialize writer with schema from first chunk (video_metadata)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema, compression='snappy')
            
            # Write chunk to parquet file
            writer.write_table(table)
            
            total_rows += len(chunk_df)
            chunk_count += 1
            
            # Clear memory
            del chunk_df
            del table
            
            if chunk_count % 10 == 0:
                gc.collect()
        
        # Close the writer
        if writer is not None:
            writer.close()
        
        print(f"✓ Saved processed video_metadata to {output_path}")
        print(f"  Total rows processed: {total_rows:,}")
        print(f"  Total chunks: {chunk_count}")
        
        return output_path
    
    except Exception as e:
        # Clean up on error
        if writer is not None:
            writer.close()
        print(f"✗ Error processing video_metadata: {e}")
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
    output_path = os.path.join(PROCESSED_DATA_DIR, 'videos_light_processed.parquet')
    
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
                    
                    # Handle missing values for numeric columns
                    numeric_cols = ['view_count', 'like_count', 'dislike_count', 'duration']
                    for col in numeric_cols:
                        if col in chunk_df.columns:
                            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0)
                    
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
                            0.0
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
                        writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                    
                    # Write chunk to parquet file
                    writer.write_table(table)
                    
                    total_rows_processed += len(chunk_df)
                    print(f"  ✓ Chunk {chunk_idx + 1} processed: {len(chunk_df):,} rows")
                    
                    chunk_idx += 1
                    
                    # Clear memory
                    del chunk_df
                    del chunk_table
                    del table
                    gc.collect()
        
        # Close the writer
        writer.close()
        
        print(f"\n✓ Saved processed video_light to {output_path}")
        print(f"  Total rows processed: {total_rows_processed:,}")
        
        return output_path
    
    except Exception as e:
        # Clean up on error
        if writer is not None:
            writer.close()
        print(f"✗ Error processing video_light: {e}")
        raise


def create_new_tag_data(video_metadata_processing):
    """
    Makes a new dataset that has just video_id, channel_id, and tags

    :param video_metadata_processing:
    :return:
    """

    pass


def num_comments_preprocessing(num_comments_df):
    """
    Preprocess number of comments data.
    
    Steps:
    1. Remove duplicates
    2. Convert num_comms to numeric and fill NaN
    """
    
    # Remove duplicates
    num_comments_df = num_comments_df.drop_duplicates(subset=['display_id'])
    
    # Convert to numeric and fill NaN
    num_comments_df['num_comms'] = pd.to_numeric(
        num_comments_df['num_comms'], errors='coerce'
    ).fillna(0)
    return num_comments_df









if __name__ == "__main__":
    # Run the data processing pipeline
    processed_paths = apply_data_processing(skip_comments=True)
    print("Processing completed successfully!")
    print(f"Processed data paths: {processed_paths}")







