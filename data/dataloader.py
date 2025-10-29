import pandas as pd
import gzip
import json

def load_data(path, nrows=None, compression=None):
    df = pd.read_csv(path, sep='\t', nrows=nrows, engine='c', compression=compression)
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

            if (i + 1) % 1000000 == 0:
                print(f"Processed {i + 1} videos...")

    videos_df = pd.DataFrame(videos_list)

    return videos_df