"""
pipeline/rag_pipeline.py
Main Medical RAG Pipeline orchestrating:
  Retrieval → Generation → Safety → Explainability
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_CHUNKS, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """
    End-to-end Medical RAG Pipeline.

    Flow:
        User Query
            → Input Safety Check
            → PubMedBERT Query Encoding
            → FAISS Semantic Retrieval (Top-K chunks)
            → Context Assembly
            → LoRA Fine-tuned LLM Generation
            → Output Safety Check
            → Evidence Explainability
            → Final Response
    """

    def __init__(
        self,
        retriever=None,
        llm=None,
        guardrails=None,
        explainer=None,
        top_k: int = TOP_K_CHUNKS
    ):
        self.retriever = retriever
        self.llm = llm
        self.guardrails = guardrails
        self.explainer = explainer
        self.top_k = top_k
        self._initialized = False

    def initialize(self, lazy: bool = True):
        """
        Initialize all pipeline components.
        Set lazy=True to defer model loading until first query.
        """
        logger.info("Initializing Medical RAG Pipeline...")

        # Guardrails (lightweight, always load)
        if self.guardrails is None:
            from pipeline.guardrails import ClinicalGuardrails
            self.guardrails = ClinicalGuardrails()

        # Explainability
        if self.explainer is None:
            from pipeline.explainability import EvidenceHighlighter
            self.explainer = EvidenceHighlighter()

        if not lazy:
            # Retriever
            if self.retriever is None:
                from retrieval.retriever import MedicalRetriever
                self.retriever = MedicalRetriever()
                self.retriever.initialize()

            # LLM
            if self.llm is None:
                from models.inference import MedicalLLM
                self.llm = MedicalLLM()
                self.llm.load()

        self._initialized = True
        logger.info("✅ Pipeline initialized")

    def _ensure_retriever(self):
        if self.retriever is None:
            from retrieval.retriever import MedicalRetriever
            self.retriever = MedicalRetriever()
            self.retriever.initialize()

    def _ensure_llm(self):
        if self.llm is None:
            from models.inference import MedicalLLM
            self.llm = MedicalLLM()
            self.llm.load()
        elif not self.llm.is_loaded():
            self.llm.load()

    def query(
        self,
        question: str,
        top_k: int = None,
        return_evidence: bool = True,
        return_explainability: bool = True,
    ) -> Dict:
        """
        Process a medical question through the full RAG pipeline.

        Args:
            question: Medical question
            top_k: Override default retrieval count
            return_evidence: Include retrieved evidence in output
            return_explainability: Include explainability analysis

        Returns:
            Full response dict with answer, evidence, safety info, explainability
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        k = top_k or self.top_k

        # ── Step 1: Input Safety Check ──────────────────
        input_check = self.guardrails.check_input(question)
        if not input_check["safe"]:
            return {
                "question": question,
                "answer": input_check["reason"],
                "safe": False,
                "input_flags": input_check["flags"],
                "emergency": input_check.get("emergency", False),
                "emergency_message": self.guardrails.get_emergency_message() if input_check.get("emergency") else None,
                "latency_ms": (time.time() - start_time) * 1000
            }

        # ── Step 2: Retrieval ────────────────────────────
        self._ensure_retriever()
        retrieve_start = time.time()
        context_str, retrieved_chunks = self.retriever.retrieve_and_format(question, top_k=k)
        retrieval_time = time.time() - retrieve_start

        if not retrieved_chunks:
            context_str = "No relevant biomedical evidence found in the knowledge base."

        retrieval_scores = [r.get("score", 0.0) for r in retrieved_chunks]

        # ── Step 3: LLM Generation ───────────────────────
        self._ensure_llm()
        gen_start = time.time()
        raw_answer, prompt_used = self.llm.generate(question, context_str)
        generation_time = time.time() - gen_start

        predicted_label = self.llm.extract_decision(raw_answer)

        # ── Step 4: Output Safety Check ─────────────────
        output_check = self.guardrails.check_output(raw_answer, retrieval_scores)
        final_answer = output_check["modified_response"]

        # ── Step 5: Explainability ───────────────────────
        explainability = None
        evidence_html = None
        if return_explainability and retrieved_chunks:
            explainability = self.explainer.highlight_evidence(
                question, raw_answer, retrieved_chunks
            )
            evidence_html = self.explainer.format_evidence_html(
                explainability["evidence_analysis"]
            )

        total_time = (time.time() - start_time) * 1000

        result = {
            "question": question,
            "answer": final_answer,
            "raw_answer": raw_answer,
            "predicted_label": predicted_label,
            "safe": output_check["safe"],
            "safety_warnings": output_check["warnings"],
            "safety_scores": output_check["scores"],
            "emergency": input_check.get("emergency", False),
            "num_chunks_retrieved": len(retrieved_chunks),
            "avg_retrieval_score": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            "latency_ms": total_time,
            "retrieval_time_ms": retrieval_time * 1000,
            "generation_time_ms": generation_time * 1000,
        }

        if return_evidence:
            result["retrieved_chunks"] = retrieved_chunks
            result["context_used"] = context_str

        if return_explainability:
            result["explainability"] = explainability
            result["evidence_html"] = evidence_html

        if input_check.get("emergency"):
            result["emergency_message"] = self.guardrails.get_emergency_message()

        return result

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict]:
        """Process multiple questions."""
        results = []
        for i, q in enumerate(questions):
            logger.info(f"Processing {i+1}/{len(questions)}: {q[:60]}...")
            results.append(self.query(q, **kwargs))
        return results

    def save_result(self, result: Dict, filename: str = None):
        """Save a single result to disk."""
        fname = filename or f"result_{int(time.time())}.json"
        path = os.path.join(RESULTS_DIR, fname)
        # Remove non-serializable items
        clean = {k: v for k, v in result.items() if k not in ("evidence_html",)}
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        return path


# Singleton pipeline instance
_pipeline: Optional[MedicalRAGPipeline] = None


def get_pipeline() -> MedicalRAGPipeline:
    """Get or create singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalRAGPipeline()
        _pipeline.initialize(lazy=True)
    return _pipeline
