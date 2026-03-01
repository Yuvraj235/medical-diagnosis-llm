"""
retrieval/vector_store.py
FAISS-based vector store for efficient semantic retrieval of biomedical literature.
"""
import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH        = os.path.join(INDEX_DIR, "corpus_metadata.json")
CHUNKS_PATH      = os.path.join(INDEX_DIR, "corpus_chunks.json")


class VectorStore:
    """FAISS-backed vector store for biomedical literature chunks."""

    def __init__(self, index, metadata: List[Dict], chunks: List[str]):
        self.index    = index          # FAISS index; ntotal == number of vectors
        self.metadata = metadata
        self.chunks   = chunks

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return top-k most similar chunks for a query embedding."""
        import faiss
        q = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append({
                "rank":     rank + 1,
                "score":    float(score),
                "chunk":    self.chunks[idx],
                "metadata": self.metadata[idx],
            })
        return results

    def save(self, path: str = FAISS_INDEX_PATH):
        import faiss
        faiss.write_index(self.index, path)
        logger.info(f"Index saved → {path}")


def build_faiss_index(
    embeddings_path: str = None,
    metadata_path:   str = None,
    chunks_path:     str = None,
) -> VectorStore:
    """
    Build a FAISS Inner Product index from pre-computed embeddings.
    Must run embeddings/pubmedbert_embedder.py (or python run.py setup) first.
    """
    import faiss

    os.makedirs(INDEX_DIR, exist_ok=True)

    emb_path  = embeddings_path or os.path.join(INDEX_DIR, "corpus_embeddings.npy")
    meta_path = metadata_path   or META_PATH
    chk_path  = chunks_path     or CHUNKS_PATH

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}\n"
            "Run: python run.py setup"
        )

    logger.info(f"Loading embeddings from {emb_path} …")
    embeddings = np.load(emb_path).astype(np.float32)

    with open(meta_path) as f: metadata = json.load(f)
    with open(chk_path)  as f: chunks   = json.load(f)

    logger.info(f"Building FAISS index for {len(embeddings)} vectors (dim={embeddings.shape[1]})…")
    faiss.normalize_L2(embeddings)   # In-place L2 normalisation

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product ≡ cosine on normalised vecs
    index.add(embeddings)

    logger.info(f"✅ FAISS index built: {index.ntotal} vectors")

    # Persist to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info(f"✅ FAISS index saved → {FAISS_INDEX_PATH}")

    return VectorStore(index=index, metadata=metadata, chunks=chunks)


def load_faiss_index() -> VectorStore:
    """Load a pre-built FAISS index from disk."""
    import faiss

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}\n"
            "Run: python run.py setup"
        )

    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH} …")
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(META_PATH)   as f: metadata = json.load(f)
    with open(CHUNKS_PATH) as f: chunks   = json.load(f)

    logger.info(f"✅ Loaded index: {index.ntotal} vectors")
    return VectorStore(index=index, metadata=metadata, chunks=chunks)
