import os
from pathlib import Path


if "__file__" in locals():
    # When running as a .py script
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    # When running inside a Jupyter notebook or interactive shell
    ROOT_DIR = Path().resolve()
    while not (ROOT_DIR / "src").exists() and ROOT_DIR != ROOT_DIR.parent:
        ROOT_DIR = ROOT_DIR.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Datasets we need
DATASET_CHANNELS = os.path.join(DATA_DIR, "df_channels_en.tsv.gz")
DATASET_TIMESERIES = os.path.join(DATA_DIR, "df_timeseries_en.tsv.gz")
DATASET_VIDEO_METADATA = os.path.join(DATA_DIR, "yt_metadata_en.jsonl.gz")
DATASET_VIDEO_METADATA_LIGHT = os.path.join(DATA_DIR, "yt_metadata_helper.feather")
DATASET_COMMENTS = os.path.join(DATA_DIR, "youtube_comments.tsv.gz")
DATASET_NUM_COMMENTS = os.path.join(DATA_DIR, "num_comments.tsv.gz")
DATASET_NUM_COMMENTS_AUTHORS = os.path.join(DATA_DIR, "num_comments_authors.tsv.gz")


if __name__=="__main__":
    print(DATA_DIR)

