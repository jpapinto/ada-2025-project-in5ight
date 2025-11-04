import pandas as pd
import gzip
import json
import pyarrow.dataset as ds
from pathlib import Path

from src.utils.paths import DATASET_CHANNELS, DATASET_TIMESERIES, DATASET_VIDEO_METADATA, DATASET_VIDEO_METADATA_LIGHT, DATASET_NUM_COMMENTS
from src.utils.paths import (
    RAW_DATASET_CHANNELS,
    RAW_DATASET_TIMESERIES,
    RAW_DATASET_VIDEO_METADATA,
    RAW_DATASET_VIDEO_METADATA_LIGHT,
    RAW_DATASET_COMMENTS,
    RAW_DATASET_NUM_COMMENTS
)

def load_data(path, nrows=None, compression='infer'):
    df = pd.read_csv(path, sep='\t', nrows=nrows,
                     engine='c', compression=compression)
    return df


def load_feather(path, nrows=None):

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
    return load_metadata_json_chunk(path, start_line=0, end_line=nrows)



def load_video_metadata(subset_size=None, compressed_videos=False):
    if compressed_videos:
        videos_df = load_feather(DATASET_VIDEO_METADATA_LIGHT)
        return videos_df
    videos_df = load_metadata_json(DATASET_VIDEO_METADATA, nrows=subset_size)
    return videos_df

def load_all_datasets(subset_size=None, compressed_videos=False):
    channels_df = load_data(DATASET_CHANNELS, nrows=subset_size)
    if compressed_videos:
        videos_df = load_feather(DATASET_VIDEO_METADATA_LIGHT)
    else:
        videos_df = load_metadata_json(
            DATASET_VIDEO_METADATA, nrows=subset_size)
    timeseries_df = load_data(DATASET_TIMESERIES, nrows=subset_size)
    num_comments_df = load_data(DATASET_NUM_COMMENTS, nrows=subset_size)
    return channels_df, videos_df, timeseries_df, num_comments_df


def load_all_raw_datasets(nrows=None):

    """Load all RAW datasets (unprocessed originals)."""
    channels_df = load_data(RAW_DATASET_CHANNELS, nrows=nrows)
    videos_df = load_feather(RAW_DATASET_VIDEO_METADATA_LIGHT, nrows=nrows)
    videos_light_df = load_metadata_json(RAW_DATASET_VIDEO_METADATA, nrows=nrows)
    timeseries_df = load_data(RAW_DATASET_TIMESERIES, nrows=nrows)
    num_comments_df = load_data(RAW_DATASET_NUM_COMMENTS, nrows=nrows)

    return channels_df, videos_df, videos_light_df, timeseries_df, num_comments_df



def iterate_videos_metadata_stream(path, window_size=100_000, start_line=0):
    """
    Iteratively read chunks from video_metadata.

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
