"""
run.py
Main entry point for the Medical RAG Pipeline.

Usage:
  python run.py setup          # Download data + build index
  python run.py finetune       # LoRA fine-tuning
  python run.py evaluate       # Run evaluation
  python run.py ui             # Launch Gradio UI
  python run.py query "..."    # Single query from CLI
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_setup():
    """Download PubMedQA + build embeddings + FAISS index."""
    logger.info("=" * 50)
    logger.info("STEP 1/3: Downloading PubMedQA dataset")
    logger.info("=" * 50)
    from data.download_data import main as download_main
    download_main()

    logger.info("=" * 50)
    logger.info("STEP 2/3: Building PubMedBERT embeddings")
    logger.info("=" * 50)
    from embeddings.pubmedbert_embedder import PubMedBERTEmbedder, build_corpus_embeddings
    embedder = PubMedBERTEmbedder()
    build_corpus_embeddings(embedder=embedder)

    logger.info("=" * 50)
    logger.info("STEP 3/3: Building FAISS index")
    logger.info("=" * 50)
    from retrieval.vector_store import build_faiss_index
    store = build_faiss_index()
    logger.info(f"Index ready: {store.index.ntotal} vectors")

    logger.info("✅ Setup complete! Run: python run.py ui")


def cmd_finetune():
    """Run LoRA fine-tuning."""
    logger.info("Starting LoRA fine-tuning...")
    from models.lora_finetune import train
    train()


def cmd_evaluate(n_samples: int = 100, mock: bool = False):
    """Run quantitative evaluation."""
    from evaluation.run_evaluation import run_evaluation
    result = run_evaluation(n_samples=n_samples, use_mock=mock)
    return result


def cmd_ui():
    """Launch Gradio UI."""
    from ui.app import main as ui_main
    ui_main()


def cmd_query(question: str, top_k: int = 5):
    """Run a single query and print results."""
    from pipeline.rag_pipeline import MedicalRAGPipeline
    pipeline = MedicalRAGPipeline()
    pipeline.initialize(lazy=False)
    result = pipeline.query(question, top_k=top_k)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    print(f"Predicted: {result['predicted_label'].upper()}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nChunks retrieved: {result['num_chunks_retrieved']}")
    print(f"Avg retrieval score: {result['avg_retrieval_score']:.3f}")
    print(f"Latency: {result['latency_ms']:.0f} ms")

    if result.get("safety_warnings"):
        print(f"\n⚠️ Safety warnings: {result['safety_warnings']}")

    if result.get("explainability"):
        print(f"\n{result['explainability']['attribution_summary']}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical RAG Pipeline — M.Tech Dissertation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup       Download data, build embeddings & FAISS index
  finetune    LoRA fine-tune the LLM on PubMedQA
  evaluate    Run quantitative evaluation suite
  ui          Launch Gradio web interface
  query       Run a single query

Examples:
  python run.py setup
  python run.py finetune
  python run.py evaluate --n-samples 100
  python run.py evaluate --mock          # Quick test without models
  python run.py ui
  python run.py query "Does aspirin reduce cardiovascular risk?"
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # setup
    subparsers.add_parser("setup", help="Download data and build index")

    # finetune
    subparsers.add_parser("finetune", help="LoRA fine-tuning")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--n-samples", type=int, default=100)
    eval_parser.add_argument("--mock", action="store_true", help="Use mock results")

    # ui
    subparsers.add_parser("ui", help="Launch Gradio UI")

    # query
    query_parser = subparsers.add_parser("query", help="Single query")
    query_parser.add_argument("question", type=str)
    query_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup()
    elif args.command == "finetune":
        cmd_finetune()
    elif args.command == "evaluate":
        cmd_evaluate(n_samples=args.n_samples, mock=args.mock)
    elif args.command == "ui":
        cmd_ui()
    elif args.command == "query":
        cmd_query(args.question, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
