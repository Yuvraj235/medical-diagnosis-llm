"""
embeddings/pubmedbert_embedder.py
PubMedBERT sentence encoder for medical text.
Encodes corpus documents and saves numpy embeddings for FAISS indexing.
"""
import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, BATCH_SIZE_EMBED,
    INDEX_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PubMedBERTEmbedder:
    """PubMedBERT sentence encoder using SentenceTransformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model     = None
        self._device    = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading {self.model_name} on {self._device}…")
        try:
            self._model = SentenceTransformer(self.model_name, device=self._device)
        except Exception as e:
            logger.warning(f"Could not load {self.model_name}: {e}. Using all-MiniLM fallback.")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device)
        logger.info("✅ Embedder ready")

    # ── Public API ──────────────────────────────────────────────────────────

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE_EMBED,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return L2-normalised embeddings, shape (n, dim)."""
        self._load()
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def get_embedding_dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end   = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks


# ── Corpus builder ───────────────────────────────────────────────────────────

def build_corpus_embeddings(embedder: Optional[PubMedBERTEmbedder] = None) -> None:
    """
    Chunk all corpus documents, encode with PubMedBERT, and save:
      data/index/corpus_embeddings.npy
      data/index/corpus_metadata.json
      data/index/corpus_chunks.json
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    if embedder is None:
        embedder = PubMedBERTEmbedder()

    all_records = []

    # Artificial corpus (knowledge base)
    art_path = os.path.join(DATA_DIR, "artificial_sample.json")
    if os.path.exists(art_path):
        with open(art_path) as f:
            art = json.load(f)
        all_records.extend(art)
        logger.info(f"Loaded {len(art)} artificial records")

    # Labeled data also enters the corpus
    lab_path = os.path.join(DATA_DIR, "labeled.json")
    if os.path.exists(lab_path):
        with open(lab_path) as f:
            lab = json.load(f)
        all_records.extend(lab)
        logger.info(f"Loaded {len(lab)} labeled records (added to corpus)")

    if not all_records:
        raise FileNotFoundError("No data found. Run: python run.py setup")

    logger.info(f"Chunking {len(all_records)} documents…")
    all_chunks, all_meta = [], []

    for record in all_records:
        context = record.get("context", "")
        if not context.strip():
            continue
        chunks = embedder.chunk_text(context)
        for ci, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({
                "pubmed_id":      record.get("pubid", ""),
                "question":       record.get("question", "")[:300],
                "mesh_terms":     record.get("meshes", ""),
                "final_decision": record.get("final_decision", ""),
                "chunk_idx":      ci,
            })

    logger.info(f"Total chunks: {len(all_chunks)}")

    logger.info("Encoding with PubMedBERT…")
    embeddings = embedder.encode(all_chunks, batch_size=BATCH_SIZE_EMBED)

    emb_path   = os.path.join(INDEX_DIR, "corpus_embeddings.npy")
    meta_path  = os.path.join(INDEX_DIR, "corpus_metadata.json")
    chunk_path = os.path.join(INDEX_DIR, "corpus_chunks.json")

    np.save(emb_path, embeddings)
    with open(meta_path, "w")  as f: json.dump(all_meta,   f)
    with open(chunk_path, "w") as f: json.dump(all_chunks, f)

    logger.info(f"✅ Saved embeddings  : {emb_path}  shape={embeddings.shape}")
    logger.info(f"✅ Saved metadata    : {meta_path}")
    logger.info(f"✅ Saved chunks      : {chunk_path}")


if __name__ == "__main__":
    emb = PubMedBERTEmbedder()
    build_corpus_embeddings(emb)
