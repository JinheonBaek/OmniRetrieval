"""Download SimpleQuestions (Wikidata) dataset into data/raw/simplequestions/."""

import os
import urllib.request

DATA_DIR = "data/raw/simplequestions"
BASE_URL = "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master"
FILES = [
    "annotated_wd_data_train_answerable.txt",
    "annotated_wd_data_valid_answerable.txt",
    "annotated_wd_data_test_answerable.txt",
]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for f in FILES:
        urllib.request.urlretrieve(f"{BASE_URL}/{f}", os.path.join(DATA_DIR, f))
