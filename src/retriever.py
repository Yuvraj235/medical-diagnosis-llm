"""
Semantic Retriever with MMR (Maximal Marginal Relevance)
Retrieves relevant evidence passages for medical questions.
"""

import numpy as np
from typing import List, Dict, Tuple
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


@dataclass
class RetrievedEvidence:
    """A single piece of retrieved evidence."""
    text: str
    score: float
    source_id: str
    source_question: str
    meshes: str


class MedicalRetriever:
    """Retrieves relevant medical evidence using semantic search + MMR."""

    def __init__(self, config: dict = None, vector_store: MedicalVectorStore = None):
        if config is None:
            config = load_config()

        self.config = config
        ret_config = config["retriever"]

        self.top_k = ret_config["top_k"]
        self.similarity_threshold = ret_config["similarity_threshold"]
        self.use_mmr = ret_config["use_mmr"]
        self.mmr_lambda = ret_config["mmr_lambda"]

        self.vector_store = vector_store or MedicalVectorStore(config)

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedEvidence]:
        """
        Retrieve top-k relevant evidence passages for a query.

        Args:
            query: Medical question
            top_k: Number of results (defaults to config)

        Returns:
            List of RetrievedEvidence objects sorted by relevance
        """
        k = top_k or self.top_k

        # Fetch more candidates if using MMR (we'll rerank)
        fetch_k = k * 3 if self.use_mmr else k

        results = self.vector_store.query(query, top_k=fetch_k)

        evidence_list = []
        for doc, dist, meta in zip(
            results["documents"], results["distances"], results["metadatas"]
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            similarity = 1 - (dist / 2)

            if similarity >= self.similarity_threshold:
                evidence_list.append(
                    RetrievedEvidence(
                        text=doc,
                        score=similarity,
                        source_id=meta.get("pubid", ""),
                        source_question=meta.get("question", ""),
                        meshes=meta.get("meshes", ""),
                    )
                )

        if self.use_mmr and len(evidence_list) > k:
            evidence_list = self._apply_mmr(query, evidence_list, k)

        return evidence_list[:k]

    def _apply_mmr(
        self, query: str, candidates: List[RetrievedEvidence], k: int
    ) -> List[RetrievedEvidence]:
        """
        Apply Maximal Marginal Relevance to diversify results.

        MMR = lambda * sim(doc, query) - (1 - lambda) * max(sim(doc, selected))
        """
        if len(candidates) <= k:
            return candidates

        embedder = self.vector_store.embedder
        query_emb = embedder.embed_query(query)
        doc_embs = embedder.embed_texts([c.text for c in candidates], show_progress=False)

        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            mmr_scores = []
            for idx in remaining:
                # Relevance to query
                relevance = np.dot(query_emb, doc_embs[idx]) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_embs[idx])
                )

                # Diversity: max similarity to already selected
                if selected_indices:
                    sims_to_selected = [
                        np.dot(doc_embs[idx], doc_embs[sel]) / (
                            np.linalg.norm(doc_embs[idx]) * np.linalg.norm(doc_embs[sel])
                        )
                        for sel in selected_indices
                    ]
                    max_sim_selected = max(sims_to_selected)
                else:
                    max_sim_selected = 0

                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim_selected
                mmr_scores.append((idx, mmr))

            # Select highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def format_evidence_for_prompt(self, evidence_list: List[RetrievedEvidence]) -> str:
        """Format retrieved evidence into a string for the LLM prompt."""
        formatted = []
        for i, ev in enumerate(evidence_list):
            formatted.append(
                f"Evidence {i+1} (relevance: {ev.score:.2f}, source: PubMed {ev.source_id}):\n{ev.text}"
            )
        return "\n\n".join(formatted)


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    retriever = MedicalRetriever(config)

    test_queries = [
        "Does aspirin reduce the risk of heart attack?",
        "Is metformin effective for type 2 diabetes?",
        "Can statins prevent stroke in elderly patients?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        evidence = retriever.retrieve(query)
        print(f"Retrieved {len(evidence)} evidence passages:\n")

        for i, ev in enumerate(evidence):
            print(f"  [{i+1}] Score: {ev.score:.4f} | Source: PubMed {ev.source_id}")
            print(f"      {ev.text[:150]}...\n")

        print("\nFormatted for prompt:")
        print(retriever.format_evidence_for_prompt(evidence)[:500])
