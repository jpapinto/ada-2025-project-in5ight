import pandas as pd
import gzip
import json
import pyarrow.dataset as ds
from pathlib import Path

from src.utils.paths import DATASET_CHANNELS, DATASET_TIMESERIES, DATASET_VIDEO_METADATA, DATASET_VIDEO_LIGHT
from src.utils.paths import (
    RAW_DATASET_CHANNELS,
    RAW_DATASET_TIMESERIES,
    RAW_DATASET_VIDEO_METADATA,
    RAW_DATASET_VIDEO_METADATA_LIGHT,
    RAW_DATASET_NUM_COMMENTS
)

def load_data(path, nrows=None, compression='infer'):
    """
    Load a dataset from a file into a pandas DataFrame.

    Args:
        path (str): Path to the dataset file.
        nrows (int, optional): Number of rows to read. Defaults to None (read all rows).
        compression (str, optional): Compression type. Defaults to 'infer'.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path, sep='\t', nrows=nrows,
                     engine='c', compression=compression)
    return df


def load_feather(path, nrows=None):
    """
    Load a Feather dataset into a pandas DataFrame.

    Args:
        path (str): Path to the Feather file.
        nrows (int, optional): Number of rows to read. Defaults to None (read all rows).

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not nrows:
        df = pd.read_feather(path)
        return df
    else:
        dataset = ds.dataset(path, format="feather")

        scanner = dataset.scanner(batch_size=100_000)
        batches = []
        rows_read = 0

        for batch in scanner.to_batches():
            df_batch = batch.to_pandas()
            if nrows is not None:
                remaining = nrows - rows_read
                if remaining <= 0:
                    break
                if len(df_batch) > remaining:
                    df_batch = df_batch.iloc[:remaining]
            batches.append(df_batch)
            rows_read += len(df_batch)
            if nrows is not None and rows_read >= nrows:
                break

        return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()





def load_metadata_json_chunk(path, start_line, end_line):
    """
    Load a chunk of metadata from a JSON file.

    Args:
        path (str): Path to the JSON file.
        start_line (int): Starting line number.
        end_line (int, optional): Ending line number. Defaults to None (read till end).

    Returns:
        pd.DataFrame: Loaded metadata as a DataFrame.
    """
    videos_list = []

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            try:
                video = json.loads(line)
                videos_list.append(video)
            except json.JSONDecodeError:
                continue

    if not videos_list:
        return pd.DataFrame()

    return pd.DataFrame(videos_list)



def load_metadata_json(path, nrows=None):
    """
    Load metadata from a JSON file.

    Args:
        path (str): Path to the JSON file.
        nrows (int, optional): Number of rows to read. Defaults to None (read all rows).

    Returns:
        pd.DataFrame: Loaded metadata as a DataFrame.
    """
    return load_metadata_json_chunk(path, start_line=0, end_line=nrows)



def load_video_metadata(subset_size=None, compressed_videos=False):
    """
    Load video metadata.

    Args:
        subset_size (int, optional): Number of rows to read. Defaults to None (read all rows).
        compressed_videos (bool, optional): Whether to load compressed videos. Defaults to False.

    Returns:
        pd.DataFrame: Loaded video metadata.
    """
    if compressed_videos:
        videos_df = load_feather(RAW_DATASET_VIDEO_METADATA_LIGHT)
        return videos_df
    videos_df = load_metadata_json(DATASET_VIDEO_METADATA, nrows=subset_size)
    return videos_df

def load_all_datasets(subset_size=None, compressed_videos=False):
    """
    Load all datasets.

    Args:
        subset_size (int, optional): Number of rows to read. Defaults to None (read all rows).
        compressed_videos (bool, optional): Whether to load compressed videos. Defaults to False.

    Returns:
        tuple: Tuple containing channels, videos, and timeseries DataFrames.
    """
    channels_df = load_data(RAW_DATASET_CHANNELS, nrows=subset_size)
    if compressed_videos:
        videos_df = load_feather(RAW_DATASET_VIDEO_METADATA_LIGHT)
    else:
        videos_df = load_metadata_json(
            DATASET_VIDEO_METADATA, nrows=subset_size)
    timeseries_df = load_data(DATASET_TIMESERIES, nrows=subset_size)
    return channels_df, videos_df, timeseries_df


def load_all_raw_datasets(nrows=None):
    """
    Load all raw datasets (unprocessed originals).

    Args:
        nrows (int, optional): Number of rows to read. Defaults to None (read all rows).

    Returns:
        tuple: Tuple containing channels, videos, videos_light, timeseries, and num_comments DataFrames.
    """
    channels_df = load_data(RAW_DATASET_CHANNELS, nrows=nrows)
    videos_df = load_feather(RAW_DATASET_VIDEO_METADATA_LIGHT, nrows=nrows)
    videos_light_df = load_metadata_json(RAW_DATASET_VIDEO_METADATA, nrows=nrows)
    timeseries_df = load_data(RAW_DATASET_TIMESERIES, nrows=nrows)
    num_comments_df = load_data(RAW_DATASET_NUM_COMMENTS, nrows=nrows)

    return channels_df, videos_df, videos_light_df, timeseries_df, num_comments_df



def iterate_videos_metadata_stream(path, window_size=100_000, start_line=0):
    """
    Iteratively read chunks from video metadata.

    Args:
        path (str): Path to the metadata file.
        window_size (int, optional): Number of rows per chunk. Defaults to 100,000.
        start_line (int, optional): Starting line number. Defaults to 0.

    Yields:
        pd.DataFrame: Chunk of video metadata.
    """
    videos_list = []
    total_rows = 0
    chunk_index = start_line // window_size

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        # Skip until start_line
        for _ in range(start_line):
            next(f, None)

        for i, line in enumerate(f, start=start_line):
            try:
                video = json.loads(line)
                videos_list.append(video)
            except json.JSONDecodeError:
                continue

            # Once we reach a full chunk, yield it
            if len(videos_list) >= window_size:
                df_chunk = pd.DataFrame(videos_list)
                videos_list.clear()
                total_rows = i + 1

                yield df_chunk #Returns a chunk temporarily

                print(f"++ Chunk {chunk_index:03d}: lines {i+1-window_size:,}–{i+1:,} | total={total_rows:,}")
                chunk_index += 1

        # Yield last partial chunk
        if videos_list:
            df_chunk = pd.DataFrame(videos_list)
            yield df_chunk
            print(f"++ Final chunk: lines {total_rows:,}–EOF | rows={len(df_chunk):,}")


def iterate_videos_light_stream(path, window_size, start_chunk=0):
    """
    Iteratively read chunks from a Feather file.

    Args:
        path (str): Path to the Feather file.
        window_size (int): Number of rows per chunk.
        start_chunk (int, optional): Starting chunk index. Defaults to 0.

    Yields:
        pd.DataFrame: Chunk of data.
    """
    path = Path(path)
    dataset = ds.dataset(path, format="feather")
    scanner = dataset.scanner(batch_size=window_size)

    total_chunks = 0
    for i, chunk in enumerate(scanner.to_batches()):
        if i < start_chunk:
            continue  # skip until desired chunk index

        df_chunk = chunk.to_pandas()
        total_chunks += 1
        yield df_chunk
        print(f"++ Chunk {i:03d} | rows={len(df_chunk):,} | cumulative={i * window_size + len(df_chunk):,}")



def iterate_parquet_stream(path, window_size=20_000_000, start_line=0):
    """
    Iteratively read chunks from a Parquet file as pandas DataFrames.

    Args:
        path (str): Path to the Parquet file.
        window_size (int, optional): Number of rows per chunk. Defaults to 20,000,000.
        start_line (int, optional): Starting line number. Defaults to 0.

    Yields:
        pd.DataFrame: Chunk of data.
    """
    videos_list = []
    total_rows = 0
    chunk_index = start_line // window_size

    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner()

    buffered_rows = 0

    for batch in scanner.to_batches():
        df_batch = batch.to_pandas()
        videos_list.append(df_batch)
        buffered_rows += len(df_batch)

        # Once we reach a full chunk, yield it
        if buffered_rows >= window_size:
            df_chunk = pd.concat(videos_list, ignore_index=True)
            videos_list.clear()
            total_rows += buffered_rows

            yield df_chunk  # Returns a chunk temporarily

            print(f"++ Chunk {chunk_index:03d}: rows {total_rows - buffered_rows:,}-{total_rows:,} | total={total_rows:,}")
            chunk_index += 1
            buffered_rows = 0

    # Yield last partial chunk
    if videos_list:
        df_chunk = pd.concat(videos_list, ignore_index=True)
        total_rows += buffered_rows
        yield df_chunk
        print(
            f"++ Final chunk: rows {total_rows - buffered_rows:,}-{total_rows:,} | rows={buffered_rows:,} | total={total_rows:,}")