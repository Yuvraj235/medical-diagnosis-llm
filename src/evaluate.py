"""
Evaluation Module for Medical RAG Pipeline
Evaluates retrieval quality, generation accuracy, and overall pipeline performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from src.data_loader import load_config, load_pubmedqa, create_train_val_test_split
except ImportError:
    from data_loader import load_config, load_pubmedqa, create_train_val_test_split

try:
    from src.pipeline import MedicalRAGPipeline
except ImportError:
    from pipeline import MedicalRAGPipeline


class PipelineEvaluator:
    """Evaluate the medical RAG pipeline on PubMedQA test set."""

    def __init__(self, pipeline: MedicalRAGPipeline, config: dict = None):
        if config is None:
            config = load_config()
        self.pipeline = pipeline
        self.config = config

    def evaluate_retrieval(self, test_df: pd.DataFrame, top_k: int = 5) -> Dict:
        """
        Evaluate retrieval quality.
        Metrics: Mean Reciprocal Rank (MRR), Hit Rate, Avg Similarity Score
        """
        print("Evaluating retrieval quality...")
        hit_count = 0
        total_scores = []
        reciprocal_ranks = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = row["question"]
            results = self.pipeline.retrieve_only(question, top_k=top_k)

            if results["num_results"] > 0:
                hit_count += 1
                scores = [ev["score"] for ev in results["evidence"]]
                total_scores.extend(scores)

                # MRR: 1/rank of first relevant result (score > threshold)
                for rank, score in enumerate(scores, 1):
                    if score > self.config["retriever"]["similarity_threshold"]:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    reciprocal_ranks.append(0)
            else:
                reciprocal_ranks.append(0)

        metrics = {
            "hit_rate": hit_count / len(test_df),
            "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
            "avg_similarity": np.mean(total_scores) if total_scores else 0,
            "median_similarity": np.median(total_scores) if total_scores else 0,
            "total_queries": len(test_df),
            "queries_with_results": hit_count,
        }

        print(f"\nRetrieval Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return metrics

    def evaluate_generation(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate generation quality on labeled test set.
        Compares predicted decision (yes/no/maybe) with ground truth.
        """
        print("\nEvaluating generation quality...")
        predictions = []
        ground_truths = []
        details = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = row["question"]
            true_label = row["final_decision"]

            try:
                response = self.pipeline.answer(question)
                pred_label = response.decision

                predictions.append(pred_label)
                ground_truths.append(true_label)
                details.append({
                    "question": question,
                    "true": true_label,
                    "predicted": pred_label,
                    "confidence": response.confidence,
                    "is_safe": response.is_safe,
                    "num_warnings": len(response.warnings),
                })
            except Exception as e:
                print(f"  Error on: {question[:50]}... -> {e}")
                predictions.append("maybe")
                ground_truths.append(true_label)

        # Calculate metrics
        labels = ["yes", "no", "maybe"]
        accuracy = accuracy_score(ground_truths, predictions)
        f1_macro = f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(ground_truths, predictions, labels=labels, average="weighted", zero_division=0)

        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, labels=labels, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class": {
                label: {"precision": p, "recall": r, "f1": f, "support": int(s)}
                for label, p, r, f, s in zip(labels, precision, recall, f1, support)
            },
            "total_samples": len(test_df),
            "details": pd.DataFrame(details),
        }

        print(f"\nGeneration Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(ground_truths, predictions, labels=labels, zero_division=0))

        return metrics

    def plot_confusion_matrix(self, ground_truths, predictions, save_path=None):
        """Plot confusion matrix."""
        labels = ["yes", "no", "maybe"]
        cm = confusion_matrix(ground_truths, predictions, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Medical QA - Confusion Matrix")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_confidence_distribution(self, details_df: pd.DataFrame, save_path=None):
        """Plot confidence score distribution for correct vs incorrect predictions."""
        details_df["correct"] = details_df["true"] == details_df["predicted"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution
        for label, color in [("Correct", "#2ecc71"), ("Incorrect", "#e74c3c")]:
            mask = details_df["correct"] == (label == "Correct")
            axes[0].hist(details_df.loc[mask, "confidence"], bins=20, alpha=0.6,
                        label=label, color=color)
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Confidence Distribution: Correct vs Incorrect")
        axes[0].legend()

        # Accuracy by confidence bin
        details_df["conf_bin"] = pd.cut(details_df["confidence"], bins=5)
        acc_by_conf = details_df.groupby("conf_bin")["correct"].mean()
        acc_by_conf.plot(kind="bar", ax=axes[1], color="#3498db")
        axes[1].set_xlabel("Confidence Bin")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy by Confidence Level")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_safety_analysis(self, details_df: pd.DataFrame, save_path=None):
        """Plot safety guardrail analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Safety pass rate
        safe_rate = details_df["is_safe"].mean() * 100
        axes[0].bar(["Safe", "Flagged"], [safe_rate, 100 - safe_rate],
                    color=["#2ecc71", "#e74c3c"])
        axes[0].set_ylabel("Percentage")
        axes[0].set_title(f"Safety Check Results ({safe_rate:.1f}% safe)")

        # Warnings distribution
        details_df["num_warnings"].value_counts().sort_index().plot(
            kind="bar", ax=axes[1], color="#f39c12"
        )
        axes[1].set_xlabel("Number of Warnings")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Warning Distribution")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def full_evaluation(self, test_df: pd.DataFrame, save_dir: str = ".") -> Dict:
        """Run complete evaluation and generate all plots."""
        print("=" * 60)
        print("FULL PIPELINE EVALUATION")
        print("=" * 60)

        # 1. Retrieval evaluation
        retrieval_metrics = self.evaluate_retrieval(test_df)

        # 2. Generation evaluation
        gen_metrics = self.evaluate_generation(test_df)

        # 3. Plots
        details = gen_metrics["details"]
        self.plot_confusion_matrix(
            details["true"].tolist(), details["predicted"].tolist(),
            save_path=f"{save_dir}/eval_confusion_matrix.png"
        )
        self.plot_confidence_distribution(
            details, save_path=f"{save_dir}/eval_confidence.png"
        )
        self.plot_safety_analysis(
            details, save_path=f"{save_dir}/eval_safety.png"
        )

        # Combined metrics
        return {
            "retrieval": retrieval_metrics,
            "generation": gen_metrics,
        }


# --- Main ---
if __name__ == "__main__":
    config = load_config()

    # Load test data
    data = load_pubmedqa(config)
    _, _, test_df = create_train_val_test_split(data["labeled"], config)

    # Initialize pipeline (without LLM for retrieval-only evaluation)
    pipeline = MedicalRAGPipeline(config)

    # Evaluate
    evaluator = PipelineEvaluator(pipeline, config)
    results = evaluator.full_evaluation(test_df)
