"""Download QALD-10 dataset into data/raw/qald10/."""

import os
import urllib.request

DATA_DIR = "data/raw/qald10"
BASE_URL = "https://raw.githubusercontent.com/KGQA/QALD_10/main/data"

# Training split = QALD-9-Plus (412 pairs), test split = QALD-10 (394 pairs)
FILES = {
    "train.json": "qald_9_plus/qald_9_plus_train_wikidata.json",
    "test.json": "qald_10/qald_10.json",
}


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for dest, src in FILES.items():
        urllib.request.urlretrieve(f"{BASE_URL}/{src}", os.path.join(DATA_DIR, dest))
