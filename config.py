import os
from pathlib import Path
import numpy as np

# Detect environment (script vs. notebook)
if "__file__" in globals():
    # When running as a .py script
    ROOT_DIR = Path(__file__).resolve().parent
else:
    # When running inside a Jupyter notebook or interactive shell
    ROOT_DIR = Path().resolve()
    while not (ROOT_DIR / "src").exists() and ROOT_DIR != ROOT_DIR.parent:
        ROOT_DIR = ROOT_DIR.parent


# Seed for recreation
np.random.seed(6)

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CACHED_RESUTLS_DIR = os.path.join(DATA_DIR, "cached_results")

# RAW Dataset Paths
RAW_DATASET_CHANNELS = os.path.join(RAW_DATA_DIR, "df_channels_en.tsv.gz")
RAW_DATASET_TIMESERIES = os.path.join(RAW_DATA_DIR, "df_timeseries_en.tsv.gz")
RAW_DATASET_VIDEO_METADATA = os.path.join(RAW_DATA_DIR, "yt_metadata_en.jsonl.gz")
RAW_DATASET_VIDEO_METADATA_LIGHT = os.path.join(RAW_DATA_DIR, "yt_metadata_helper.feather")
RAW_DATASET_COMMENTS = os.path.join(RAW_DATA_DIR, "youtube_comments.tsv.gz")
RAW_DATASET_NUM_COMMENTS = os.path.join(RAW_DATA_DIR, "num_comments.tsv.gz")
RAW_DATASET_NUM_COMMENTS_AUTHORS = os.path.join(RAW_DATA_DIR, "num_comments_authors.tsv.gz")

# Processed Dataset Paths (primary datasets to use in analysis)
DATASET_NUM_COMMENTS = os.path.join(PROCESSED_DATA_DIR, 'num_comments_processed.parquet')
DATASET_VIDEO_METADATA = os.path.join(PROCESSED_DATA_DIR, 'videos_metadata_processed.parquet')
DATASET_TAGS = os.path.join(PROCESSED_DATA_DIR, 'tags_processed.parquet')
DATASET_VIDEO_LIGHT = os.path.join(PROCESSED_DATA_DIR, 'videos_light_processed.parquet')
DATASET_CHANNELS = os.path.join(PROCESSED_DATA_DIR, 'channels_processed.parquet')
DATASET_TIMESERIES = os.path.join(PROCESSED_DATA_DIR, 'timeseries_processed.parquet')







