"""
data/download_data.py
Download and preprocess PubMedQA dataset.
Saves labeled data for training/eval and artificial data for the retrieval corpus.
"""
import sys
import os
import json
import logging
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PUBMEDQA_DATASET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_SAMPLE_SIZE = 20000  # artificial examples used for knowledge base


def _flatten_record(item: dict, idx: int = 0) -> dict:
    """Flatten a PubMedQA record into a flat dict."""
    ctx = item.get("context", {})
    if isinstance(ctx, dict):
        labels = ctx.get("labels", [])
        texts  = ctx.get("contexts", [])
        meshes = ctx.get("meshes", [])
        sections = [f"[{l}] {t}" for l, t in zip(labels, texts)]
        context_str = " ".join(sections)
        mesh_str    = "; ".join(meshes)
    else:
        context_str = str(ctx)
        mesh_str    = ""

    return {
        "pubid":          str(item.get("pubid", str(idx))),
        "question":       item.get("question", ""),
        "context":        context_str,
        "meshes":         mesh_str,
        "long_answer":    item.get("long_answer", ""),
        "final_decision": item.get("final_decision", "maybe"),
    }


def download_labeled() -> list:
    """Download expert-annotated PubMedQA (1,000 examples)."""
    from datasets import load_dataset
    logger.info("Downloading pqa_labeled (1,000 expert-annotated examples)...")
    ds = load_dataset(PUBMEDQA_DATASET, "pqa_labeled", split="train")
    records = [_flatten_record(item, i) for i, item in enumerate(ds)]

    out_path = os.path.join(DATA_DIR, "labeled.json")
    with open(out_path, "w") as f:
        json.dump(records, f)
    logger.info(f"Saved {len(records)} labeled examples → {out_path}")
    return records


def download_artificial(sample_size: int = CORPUS_SAMPLE_SIZE) -> list:
    """Download pqa_artificial and sample for the retrieval corpus."""
    from datasets import load_dataset
    logger.info(f"Downloading pqa_artificial (sampling {sample_size} for corpus)...")
    ds = load_dataset(PUBMEDQA_DATASET, "pqa_artificial", split="train")

    indices = random.sample(range(len(ds)), min(sample_size, len(ds)))
    records = [_flatten_record(ds[i], i) for i in indices]

    out_path = os.path.join(DATA_DIR, "artificial_sample.json")
    with open(out_path, "w") as f:
        json.dump(records, f)
    logger.info(f"Saved {len(records)} artificial examples → {out_path}")
    return records


def create_splits(records: list, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    """Split labeled records into train / val / test and save."""
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n         = len(shuffled)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    splits = {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }
    for name, data in splits.items():
        path = os.path.join(DATA_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"  {name}: {len(data)} examples → {path}")

    return splits


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    random.seed(42)

    labeled    = download_labeled()
    splits     = create_splits(labeled)
    artificial = download_artificial()

    logger.info("=" * 50)
    logger.info("✅ Data download complete!")
    logger.info(f"   Labeled total : {len(labeled)}")
    logger.info(f"   Train / Val / Test : {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    logger.info(f"   Corpus (artificial): {len(artificial)}")


if __name__ == "__main__":
    main()
