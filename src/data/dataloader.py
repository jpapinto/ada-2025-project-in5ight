import pandas as pd
import gzip
import json
import time

from src.utils.paths import DATASET_CHANNELS, DATASET_TIMESERIES, DATASET_VIDEO_METADATA, DATASET_VIDEO_METADATA_LIGHT, DATASET_NUM_COMMENTS


def load_data(path, nrows=None, compression='infer'):
    df = pd.read_csv(path, sep='\t', nrows=nrows,
                     engine='c', compression=compression)
    return df


def load_feather(path):
    df = pd.read_feather(path)
    return df


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


def load_channels_dataset(subset_size=None):
    channels_df = load_data(DATASET_CHANNELS, nrows=subset_size)
    return channels_df




def load_video_metadata(subset_size=None, compressed_videos=False):
    if compressed_videos:
        videos_df = load_feather(DATASET_VIDEO_METADATA_LIGHT)
        return videos_df
    videos_df = load_metadata_json(DATASET_VIDEO_METADATA, nrows=subset_size)
    return videos_df



def iterate_videos_metadata_stream(window_size=100_000, start_line=0):
    """
    Iteratively read chunks from video_metadata.

    """
    videos_list = []
    total_rows = 0
    chunk_index = start_line // window_size
    path = DATASET_VIDEO_METADATA

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