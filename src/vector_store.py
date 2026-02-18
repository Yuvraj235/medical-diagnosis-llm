"""
Vector Store using ChromaDB
Indexes PubMedQA abstracts for efficient semantic retrieval.
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

try:
    from src.data_loader import load_config, load_pubmedqa
except ImportError:
    from data_loader import load_config, load_pubmedqa

try:
    from src.embeddings import PubMedBERTEmbedder
except ImportError:
    from embeddings import PubMedBERTEmbedder


class MedicalVectorStore:
    """ChromaDB-based vector store for medical literature."""

    def __init__(self, config: dict = None, embedder: PubMedBERTEmbedder = None):
        if config is None:
            config = load_config()

        self.config = config
        db_config = config["vector_db"]

        self.persist_dir = db_config["persist_directory"]
        self.collection_name = db_config["collection_name"]

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Initialize embedder
        self.embedder = embedder or PubMedBERTEmbedder(config)

    def create_index(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Index PubMedQA data into ChromaDB.

        Args:
            df: DataFrame with columns [pubid, question, context, meshes, long_answer, final_decision]
            batch_size: Number of documents to add per batch
        """
        # Delete existing collection if exists
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Indexing {len(df)} documents into ChromaDB...")

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
            context = str(row["context"])
            chunks = self.embedder.chunk_text(context)

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{row['pubid']}_{chunk_idx}"
                all_chunks.append(chunk)
                all_ids.append(doc_id)
                all_metadatas.append({
                    "pubid": str(row["pubid"]),
                    "question": str(row["question"])[:500],
                    "meshes": str(row.get("meshes", ""))[:500],
                    "decision": str(row.get("final_decision", "")),
                    "chunk_idx": chunk_idx,
                })

        print(f"Total chunks to index: {len(all_chunks)}")

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.embed_texts(all_chunks)

        # Add to ChromaDB in batches
        print("Adding to ChromaDB...")
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
            end = min(i + batch_size, len(all_chunks))
            collection.add(
                ids=all_ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=all_chunks[i:end],
                metadatas=all_metadatas[i:end],
            )

        print(f"Indexed {collection.count()} chunks successfully!")
        return collection

    def get_collection(self):
        """Get existing collection."""
        return self.client.get_collection(self.collection_name)

    def query(
        self, query_text: str, top_k: int = 5
    ) -> Dict[str, List]:
        """
        Query the vector store.

        Returns:
            Dictionary with 'documents', 'distances', 'metadatas'
        """
        collection = self.get_collection()

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
        }


# --- Main ---
if __name__ == "__main__":
    config = load_config()

    # Load data
    data = load_pubmedqa(config)

    # Use a sample for quick testing (full indexing takes longer)
    sample_size = 5000
    sample_df = data["artificial"].sample(n=sample_size, random_state=42)
    print(f"\nUsing {sample_size} samples for quick index test")

    # Build index
    store = MedicalVectorStore(config)
    store.create_index(sample_df)

    # Test query
    print("\n=== Test Query ===")
    query = "Does aspirin reduce the risk of heart attack?"
    results = store.query(query, top_k=3)

    for i, (doc, dist, meta) in enumerate(
        zip(results["documents"], results["distances"], results["metadatas"])
    ):
        print(f"\nResult {i+1} (distance: {dist:.4f}):")
        print(f"  Source: PubMed ID {meta['pubid']}")
        print(f"  Question: {meta['question'][:100]}...")
        print(f"  Text: {doc[:200]}...")
