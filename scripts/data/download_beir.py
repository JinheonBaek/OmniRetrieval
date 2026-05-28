"""Download BEIR benchmark datasets into data/raw/beir/."""

from beir import util

DATA_DIR = "data/raw/beir"
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

# NQ has separate test ("nq") and train ("nq-train") downloads (beir-cellar/beir#108).
DATASETS = ["fever", "fiqa", "msmarco", "scifact", "hotpotqa", "nfcorpus", "nq", "nq-train"]


if __name__ == "__main__":
    for dataset in DATASETS:
        url = BASE_URL.format(dataset)
        util.download_and_unzip(url, DATA_DIR)
