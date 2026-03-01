"""
pipeline/explainability.py
Evidence highlighting and explainability for RAG responses.
Shows which retrieved chunks most influenced the answer.
"""
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class EvidenceHighlighter:
    """
    Provides explainability by:
    1. Identifying key sentences in retrieved evidence
    2. Computing relevance of each evidence chunk to the final answer
    3. Highlighting supporting/contradicting evidence
    4. Generating attribution summary
    """

    def __init__(self):
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
            self._embedder = PubMedBERTEmbedder()
        return self._embedder

    def highlight_evidence(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """
        Analyze how each retrieved chunk supports the answer.

        Returns:
            {
                evidence_analysis: [...],
                top_supporting: {...},
                attribution_summary: str,
                confidence: float
            }
        """
        if not retrieved_chunks:
            return {
                "evidence_analysis": [],
                "top_supporting": None,
                "attribution_summary": "No evidence retrieved.",
                "confidence": 0.0
            }

        embedder = self._get_embedder()

        # Encode question + answer for comparison
        qa_text = f"{question} {answer}"
        qa_emb = embedder.encode([qa_text])[0]

        # Score each chunk against QA
        import numpy as np
        chunk_texts = [r["chunk"] for r in retrieved_chunks]
        chunk_embs = embedder.encode(chunk_texts)

        # Cosine similarity
        sims = np.dot(chunk_embs, qa_emb) / (
            np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(qa_emb) + 1e-9
        )

        evidence_analysis = []
        for i, (chunk_data, sim) in enumerate(zip(retrieved_chunks, sims)):
            # Extract key sentences from chunk
            key_sentences = self._extract_key_sentences(
                chunk_data["chunk"], question, top_n=2
            )

            # Determine support level
            if sim > 0.7:
                support_level = "strong"
            elif sim > 0.5:
                support_level = "moderate"
            elif sim > 0.3:
                support_level = "weak"
            else:
                support_level = "tangential"

            evidence_analysis.append({
                "rank": chunk_data.get("rank", i + 1),
                "chunk_preview": chunk_data["chunk"][:200] + "...",
                "full_chunk": chunk_data["chunk"],
                "retrieval_score": chunk_data.get("score", 0.0),
                "answer_relevance": float(sim),
                "support_level": support_level,
                "key_sentences": key_sentences,
                "pubmed_id": chunk_data.get("metadata", {}).get("pubmed_id", "N/A"),
                "mesh_terms": chunk_data.get("metadata", {}).get("mesh_terms", []),
            })

        # Sort by answer relevance
        evidence_analysis.sort(key=lambda x: x["answer_relevance"], reverse=True)
        top_supporting = evidence_analysis[0] if evidence_analysis else None

        # Overall confidence = weighted combination of retrieval & answer relevance
        avg_retrieval = np.mean([r.get("retrieval_score", 0) for r in retrieved_chunks])
        avg_relevance = np.mean(sims)
        confidence = float(0.5 * avg_retrieval + 0.5 * avg_relevance)

        attribution_summary = self._build_attribution_summary(
            answer, evidence_analysis, confidence
        )

        return {
            "evidence_analysis": evidence_analysis,
            "top_supporting": top_supporting,
            "attribution_summary": attribution_summary,
            "confidence": confidence,
            "avg_retrieval_score": float(avg_retrieval),
            "avg_answer_relevance": float(avg_relevance),
        }

    def _extract_key_sentences(
        self, text: str, question: str, top_n: int = 2
    ) -> List[str]:
        """
        Extract the most relevant sentences from a chunk using simple TF-IDF-like scoring.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= top_n:
            return sentences

        # Score sentences by question word overlap
        q_words = set(question.lower().split())
        scored = []
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(q_words & s_words) / (len(q_words) + 1)
            scored.append((overlap, sent))

        scored.sort(reverse=True)
        return [s for _, s in scored[:top_n]]

    def _build_attribution_summary(
        self,
        answer: str,
        evidence: List[Dict],
        confidence: float
    ) -> str:
        """Build a human-readable attribution summary."""
        strong = [e for e in evidence if e["support_level"] == "strong"]
        moderate = [e for e in evidence if e["support_level"] == "moderate"]

        lines = [f"📊 Evidence Attribution Summary (Confidence: {confidence:.1%})"]
        lines.append(f"Total evidence chunks used: {len(evidence)}")

        if strong:
            lines.append(f"✅ Strong support: {len(strong)} chunk(s)")
            for e in strong[:2]:
                lines.append(f"   • PubMed ID: {e['pubmed_id']} (relevance: {e['answer_relevance']:.3f})")

        if moderate:
            lines.append(f"📝 Moderate support: {len(moderate)} chunk(s)")

        if confidence < 0.4:
            lines.append("⚠️ Low confidence — answer may not be fully supported by retrieved evidence.")

        return "\n".join(lines)

    def format_evidence_html(self, evidence_analysis: List[Dict]) -> str:
        """Format evidence analysis as HTML for Gradio display."""
        if not evidence_analysis:
            return "<p>No evidence available.</p>"

        color_map = {
            "strong": "#d4edda",
            "moderate": "#fff3cd",
            "weak": "#f8d7da",
            "tangential": "#e2e3e5"
        }

        html_parts = []
        for e in evidence_analysis:
            color = color_map.get(e["support_level"], "#ffffff")
            html_parts.append(f"""
            <div style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; 
                        margin: 8px 0; background-color: {color};">
                <b>Evidence #{e['rank']}</b> — 
                PubMed: {e['pubmed_id']} | 
                Retrieval: {e['retrieval_score']:.3f} | 
                Relevance: {e['answer_relevance']:.3f} | 
                Support: <b>{e['support_level'].upper()}</b>
                <br><br>
                <em>Key sentences:</em>
                <ul>{''.join(f"<li>{s}</li>" for s in e['key_sentences'])}</ul>
                <details>
                    <summary>View full chunk</summary>
                    <p style="font-size: 0.9em;">{e['full_chunk']}</p>
                </details>
            </div>
            """)

        return "\n".join(html_parts)
