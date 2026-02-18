"""
Data Loader for PubMedQA Dataset
Loads, preprocesses, and splits PubMedQA data for training and evaluation.
"""

import yaml
from datasets import load_dataset
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flatten_context(example: dict) -> dict:
    """Flatten nested context fields into a single string."""
    contexts = example.get("context", {})

    if isinstance(contexts, dict):
        labels = contexts.get("labels", [])
        texts = contexts.get("contexts", [])
        meshes = contexts.get("meshes", [])

        # Combine labeled sections
        sections = []
        for label, text in zip(labels, texts):
            sections.append(f"[{label}] {text}")
        context_str = " ".join(sections)
        mesh_str = ", ".join(meshes) if meshes else ""
    else:
        context_str = str(contexts)
        mesh_str = ""

    return {
        "pubid": example.get("pubid", ""),
        "question": example.get("question", ""),
        "context": context_str,
        "meshes": mesh_str,
        "long_answer": example.get("long_answer", ""),
        "final_decision": example.get("final_decision", ""),
    }


def load_pubmedqa(config: dict = None) -> Dict[str, pd.DataFrame]:
    """
    Load and preprocess PubMedQA dataset.

    Returns:
        Dictionary with 'labeled', 'artificial', 'unlabeled' DataFrames
    """
    if config is None:
        config = load_config()

    data_config = config["data"]
    dataset_name = data_config["dataset_name"]

    print("Loading PubMedQA dataset...")
    result = {}

    # Load labeled data (expert-annotated, 1K examples)
    print("  Loading pqa_labeled (1K expert-annotated)...")
    labeled = load_dataset(dataset_name, "pqa_labeled", split="train")
    labeled_processed = [flatten_context(ex) for ex in labeled]
    result["labeled"] = pd.DataFrame(labeled_processed)
    print(f"    Loaded: {len(result['labeled'])} examples")

    # Load artificial data (machine-generated, 211K examples)
    print("  Loading pqa_artificial (211K machine-generated)...")
    artificial = load_dataset(dataset_name, "pqa_artificial", split="train")
    artificial_processed = [flatten_context(ex) for ex in artificial]
    result["artificial"] = pd.DataFrame(artificial_processed)
    print(f"    Loaded: {len(result['artificial'])} examples")

    return result


def create_train_val_test_split(
    df: pd.DataFrame, config: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labeled data into train/val/test."""
    if config is None:
        config = load_config()

    seed = config["data"]["seed"]
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} examples")
    print(f"  Val:   {len(val_df)} examples")
    print(f"  Test:  {len(test_df)} examples")

    # Distribution check
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = subset["final_decision"].value_counts(normalize=True)
        print(f"  {name} label distribution: {dict(dist.round(3))}")

    return train_df, val_df, test_df


def format_for_training(row: dict) -> str:
    """Format a single example for LLM fine-tuning."""
    question = row["question"]
    context = row["context"]
    decision = row["final_decision"]
    explanation = row["long_answer"]

    prompt = f"""### Instruction: You are a medical question-answering assistant. Answer the question strictly using the provided evidence. If the evidence is insufficient, say "Insufficient evidence."

### Evidence:
{context}

### Question: {question}

### Answer: {decision}
### Explanation: {explanation}"""

    return prompt


def prepare_training_data(df: pd.DataFrame) -> List[str]:
    """Convert DataFrame to list of formatted training strings."""
    return [format_for_training(row) for _, row in df.iterrows()]


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    data = load_pubmedqa(config)

    print("\n=== Dataset Summary ===")
    for name, df in data.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Decision distribution:\n{df['final_decision'].value_counts()}")
        print(f"  Avg context length: {df['context'].str.len().mean():.0f} chars")

    # Split labeled data
    train_df, val_df, test_df = create_train_val_test_split(data["labeled"], config)

    # Show sample training format
    print("\n=== Sample Training Format ===")
    print(format_for_training(train_df.iloc[0]))
