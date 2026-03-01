"""
retrieval/retriever.py
Medical evidence retriever using FAISS + PubMedBERT.
"""
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_CHUNKS, SIMILARITY_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MedicalRetriever:
    """Semantic retriever: encodes query → FAISS search → ranked chunks."""

    def __init__(self, top_k: int = TOP_K_CHUNKS):
        self.top_k        = top_k
        self._embedder    = None
        self._store       = None
        self._initialized = False

    def initialize(self):
        """Load PubMedBERT embedder and FAISS index."""
        logger.info("Initialising MedicalRetriever…")
        from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
        self._embedder = PubMedBERTEmbedder()
        from retrieval.vector_store import load_faiss_index
        self._store = load_faiss_index()
        self._initialized = True
        logger.info("✅ Retriever initialised")

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a question.
        Returns list of dicts: {rank, score, chunk, metadata}.
        """
        if not self._initialized:
            self.initialize()

        k = top_k or self.top_k

        # Over-retrieve then filter
        query_emb = self._embedder.encode([question], show_progress=False)
        results   = self._store.search(query_emb[0], top_k=k * 2)

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
        if not filtered:
            filtered = results  # Keep best even if below threshold

        return filtered[:k]

    def retrieve_and_format(
        self, question: str, top_k: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve chunks and assemble a formatted context string.
        Returns: (context_str, retrieved_chunks_list)
        """
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            return "No relevant biomedical evidence found.", []

        parts = []
        for i, c in enumerate(chunks):
            pid = c["metadata"].get("pubmed_id", "N/A")
            parts.append(f"[Evidence {i+1} | PubMed:{pid}]\n{c['chunk']}")

        return "\n\n".join(parts), chunks
