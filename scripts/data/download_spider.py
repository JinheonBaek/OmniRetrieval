"""Download Spider (text-to-SQL) dataset into data/raw/spider/."""

import os

import gdown
from beir.util import unzip

DATA_DIR = "data/raw/spider"
GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = gdown.download(id=GDRIVE_ID, output=f"{DATA_DIR}/spider.zip")
    unzip(zip_path, DATA_DIR)
