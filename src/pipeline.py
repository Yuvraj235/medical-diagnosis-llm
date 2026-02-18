"""
End-to-End RAG Pipeline for Medical Question Answering
Chains all components: Retriever -> Generator -> Guardrails
"""

import os
import sys
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from src.data_loader import load_config
except ImportError:
    from data_loader import load_config

try:
    from src.embeddings import PubMedBERTEmbedder
except ImportError:
    from embeddings import PubMedBERTEmbedder

try:
    from src.vector_store import MedicalVectorStore
except ImportError:
    from vector_store import MedicalVectorStore

try:
    from src.retriever import MedicalRetriever
except ImportError:
    from retriever import MedicalRetriever

try:
    from src.generator import MedicalGenerator, GeneratedAnswer
except ImportError:
    from generator import MedicalGenerator, GeneratedAnswer

try:
    from src.guardrails import ClinicalGuardrails, SafetyCheckResult
except ImportError:
    from guardrails import ClinicalGuardrails, SafetyCheckResult


@dataclass
class PipelineResponse:
    """Complete response from the medical RAG pipeline."""
    question: str
    decision: str
    explanation: str
    safe_response: str
    evidence_texts: list
    evidence_scores: list
    confidence: float
    is_safe: bool
    warnings: list
    hallucination_score: float


class MedicalRAGPipeline:
    """
    End-to-end medical question answering with:
    1. Semantic evidence retrieval (PubMedBERT + ChromaDB)
    2. Answer generation (BioMistral-7B + LoRA)
    3. Safety guardrails (hallucination check, disclaimers)
    """

    def __init__(self, config: dict = None, model=None, tokenizer=None):
        if config is None:
            config = load_config()

        self.config = config
        print("Initializing Medical RAG Pipeline...")

        # Initialize components
        print("  [1/4] Loading embedder...")
        self.embedder = PubMedBERTEmbedder(config)

        print("  [2/4] Connecting to vector store...")
        self.vector_store = MedicalVectorStore(config, self.embedder)
        self.retriever = MedicalRetriever(config, self.vector_store)

        print("  [3/4] Setting up generator...")
        self.generator = MedicalGenerator(model=model, tokenizer=tokenizer, config=config)

        print("  [4/4] Loading guardrails...")
        self.guardrails = ClinicalGuardrails(config)

        print("Pipeline ready!\n")

    def answer(self, question: str, top_k: int = None) -> PipelineResponse:
        """
        Answer a medical question using the full RAG pipeline.

        Steps:
        1. Retrieve relevant evidence from vector DB
        2. Generate answer using fine-tuned LLM
        3. Apply safety guardrails
        4. Format and return response

        Args:
            question: Medical question to answer
            top_k: Number of evidence passages to retrieve

        Returns:
            PipelineResponse with all components
        """
        print(f"Processing: {question}")

        # Step 1: Retrieve evidence
        print("  Retrieving evidence...")
        evidence = self.retriever.retrieve(question, top_k=top_k)
        print(f"  Found {len(evidence)} relevant passages")

        # Step 2: Generate answer
        print("  Generating answer...")
        if self.generator.model is not None:
            generated = self.generator.generate(question, evidence)
        else:
            # Fallback: rule-based answer when model not loaded
            generated = self._rule_based_answer(question, evidence)

        # Step 3: Safety checks
        print("  Running safety checks...")
        safety_result = self.guardrails.check(generated)

        # Step 4: Format response
        safe_response = self.guardrails.format_safe_response(generated, safety_result)

        return PipelineResponse(
            question=question,
            decision=generated.decision,
            explanation=generated.explanation,
            safe_response=safe_response,
            evidence_texts=[ev.text for ev in evidence],
            evidence_scores=[ev.score for ev in evidence],
            confidence=generated.confidence,
            is_safe=safety_result.is_safe,
            warnings=safety_result.warnings,
            hallucination_score=safety_result.hallucination_score,
        )

    def _rule_based_answer(self, question: str, evidence) -> GeneratedAnswer:
        """Fallback answer when LLM is not loaded (for testing retrieval pipeline)."""
        if not evidence:
            return GeneratedAnswer(
                question=question,
                decision="maybe",
                explanation="Insufficient evidence found to answer this question.",
                evidence_used=evidence,
                confidence=0.0,
                raw_output="No evidence retrieved.",
            )

        # Simple keyword-based heuristic
        combined = " ".join(ev.text.lower() for ev in evidence)

        positive_words = ["effective", "significant", "beneficial", "reduced", "improved", "positive", "associated with lower"]
        negative_words = ["ineffective", "no significant", "no effect", "not associated", "failed", "no benefit"]

        pos_count = sum(1 for w in positive_words if w in combined)
        neg_count = sum(1 for w in negative_words if w in combined)

        if pos_count > neg_count:
            decision = "yes"
        elif neg_count > pos_count:
            decision = "no"
        else:
            decision = "maybe"

        explanation = f"Based on {len(evidence)} retrieved evidence passages. Evidence suggests: {evidence[0].text[:200]}"
        avg_score = sum(ev.score for ev in evidence) / len(evidence)

        return GeneratedAnswer(
            question=question,
            decision=decision,
            explanation=explanation,
            evidence_used=evidence,
            confidence=avg_score,
            raw_output=explanation,
        )

    def retrieve_only(self, question: str, top_k: int = None) -> dict:
        """Retrieve evidence without generation (for testing retrieval)."""
        evidence = self.retriever.retrieve(question, top_k=top_k)
        return {
            "question": question,
            "num_results": len(evidence),
            "evidence": [
                {
                    "text": ev.text,
                    "score": ev.score,
                    "source": ev.source_id,
                    "related_question": ev.source_question,
                }
                for ev in evidence
            ],
        }


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    pipeline = MedicalRAGPipeline(config)

    test_questions = [
        "Does aspirin reduce the risk of heart attack?",
        "Is metformin effective for type 2 diabetes?",
        "Can statins prevent stroke?",
    ]

    for q in test_questions:
        print("\n" + "=" * 70)
        response = pipeline.answer(q)
        print(response.safe_response)
        print("=" * 70)
