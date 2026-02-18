"""
Embedding Generation using PubMedBERT
Generates domain-specific biomedical embeddings for semantic retrieval.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

try:
    from src.data_loader import load_config
except ImportError:
    from data_loader import load_config


class PubMedBERTEmbedder:
    """Generate biomedical embeddings using PubMedBERT."""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.model_name = config["embeddings"]["model_name"]
        self.batch_size = config["embeddings"]["batch_size"]
        self.chunk_size = config["embeddings"]["chunk_size"]
        self.chunk_overlap = config["embeddings"]["chunk_overlap"]

        print(f"Loading PubMedBERT: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Use MPS if available (Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA GPU")
        else:
            self.device = "cpu"
            print("Using CPU")

        self.model = self.model.to(self.device)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [text]

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode(
            [query], convert_to_numpy=True, device=self.device
        )[0]

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        test = self.model.encode(["test"], convert_to_numpy=True)
        return test.shape[1]


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    embedder = PubMedBERTEmbedder(config)

    # Test embedding
    test_texts = [
        "Does aspirin reduce the risk of heart attack?",
        "Aspirin has been shown to reduce platelet aggregation and lower cardiovascular risk.",
        "The treatment of diabetes involves insulin management and lifestyle changes.",
    ]

    embeddings = embedder.embed_texts(test_texts)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")

    # Test chunking
    long_text = " ".join(["word"] * 500)
    chunks = embedder.chunk_text(long_text)
    print(f"\nChunking {len(long_text.split())} words -> {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c.split())} words")

    # Test similarity
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print(f"\nSimilarity (Q1 vs A1): {cosine_sim(embeddings[0], embeddings[1]):.4f}")
    print(f"Similarity (Q1 vs A2): {cosine_sim(embeddings[0], embeddings[2]):.4f}")
    print("(Q1 vs A1 should be higher - both about aspirin/heart)")
