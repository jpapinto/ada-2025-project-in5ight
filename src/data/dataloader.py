import pandas as pd
import gzip
import json

from data.paths import DATASET_CHANNELS, DATASET_TIMESERIES, DATASET_VIDEO_METADATA, DATASET_VIDEO_METADATA_LIGHT, DATASET_NUM_COMMENTS


def load_data(path, nrows=None, compression='infer'):
    df = pd.read_csv(path, sep='\t', nrows=nrows,
                     engine='c', compression=compression)
    return df


def load_feather(path):
    df = pd.read_feather(path)
    return df


def load_metadata_json(path, nrows=None):
    videos_list = []

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            try:
                video = json.loads(line)
                videos_list.append(video)
            except json.JSONDecodeError:
                continue

    videos_df = pd.DataFrame(videos_list)

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
