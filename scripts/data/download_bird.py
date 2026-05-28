"""Download BIRD (text-to-SQL) dataset into data/raw/bird/."""

import os

from beir import util
from beir.util import unzip

DATA_DIR = "data/raw/bird"
URLS = {
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}
NESTED_ZIPS = [
    "train/train_databases.zip",
    "dev_20240627/dev_databases.zip",
]


if __name__ == "__main__":
    for split, url in URLS.items():
        util.download_and_unzip(url, DATA_DIR)

    for nested in NESTED_ZIPS:
        zip_path = os.path.join(DATA_DIR, nested)
        if os.path.isfile(zip_path):
            unzip(zip_path, os.path.dirname(zip_path))
