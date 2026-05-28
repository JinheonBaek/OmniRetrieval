"""Download LC-QuAD 2.0 dataset into data/raw/lcquad2/."""

import os
import urllib.request

DATA_DIR = "data/raw/lcquad2"
BASE_URL = "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset"
FILES = ["train.json", "test.json"]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for f in FILES:
        urllib.request.urlretrieve(f"{BASE_URL}/{f}", os.path.join(DATA_DIR, f))
