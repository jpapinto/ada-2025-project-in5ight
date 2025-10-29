import os

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data_files')

# Datasets we need
DATASET_CHANNELS = os.path.join(DATA_DIRECTORY, "df_channels_en.tsv.gz")
DATASET_TIMESERIES = os.path.join(DATA_DIRECTORY, "df_timeseries_en.tsv.gz")
DATASET_VIDEO_METADATA = os.path.join(DATA_DIRECTORY, "yt_metadata_en.jsonl.gz")
DATASET_VIDEO_METADATA_LIGHT = os.path.join(DATA_DIRECTORY, "yt_metadata_helper.feather")
DATASET_COMMENTS = os.path.join(DATA_DIRECTORY, "youtube_comments.tsv.gz")
DATASET_NUM_COMMENTS = os.path.join(DATA_DIRECTORY, "num_comments.tsv.gz")
DATASET_NUM_COMMENTS_AUTHORS = os.path.join(DATA_DIRECTORY, "num_comments_authors.tsv.gz")



