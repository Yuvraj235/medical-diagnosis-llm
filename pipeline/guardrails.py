"""
pipeline/guardrails.py
Safety and clinical guardrails for responsible medical AI.
Checks for: toxicity, bias, blocked topics, hallucination indicators.
"""
import sys
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOXICITY_THRESHOLD, BLOCKED_TOPICS, MEDICAL_DISCLAIMER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ClinicalGuardrails:
    """
    Multi-layer safety system for medical AI responses.

    Layers:
    1. Input validation (blocked topics, harmful queries)
    2. Toxicity detection (Detoxify)
    3. Bias detection (demographic bias patterns)
    4. Hallucination risk indicator (low evidence confidence)
    5. Medical disclaimer injection
    """

    def __init__(self, toxicity_threshold: float = TOXICITY_THRESHOLD):
        self.threshold = toxicity_threshold
        self._detoxify = None
        self._load_detoxify()

        # Compiled regex patterns
        self._blocked_patterns = [
            re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            for t in BLOCKED_TOPICS
        ]

        # Demographic bias trigger words
        self._bias_patterns = [
            re.compile(r"\b(race|ethnicity|gender|sex|religion|nationality)\b.*\b(inferior|superior|prone|likely|risk)\b", re.IGNORECASE),
            re.compile(r"\b(black|white|asian|hispanic|female|male)\b.*\b(more likely|less likely|inferior|superior)\b", re.IGNORECASE),
        ]

        # Emergency/urgent medical terms that should always defer to professional
        self._emergency_terms = re.compile(
            r"\b(chest pain|heart attack|stroke|overdose|emergency|suicide|seizure|anaphylaxis)\b",
            re.IGNORECASE
        )

    def _load_detoxify(self):
        """Load Detoxify model for toxicity scoring."""
        try:
            from detoxify import Detoxify
            self._detoxify = Detoxify("original")
            logger.info("✅ Detoxify toxicity model loaded")
        except ImportError:
            logger.warning("Detoxify not installed. Toxicity detection disabled.")
        except Exception as e:
            logger.warning(f"Could not load Detoxify: {e}")

    # ──────────────────────────────────────────
    # INPUT GUARDRAILS
    # ──────────────────────────────────────────

    def check_input(self, query: str) -> Dict:
        """
        Validate user query before processing.

        Returns:
            {safe: bool, reason: str, emergency: bool, flags: list}
        """
        flags = []
        is_emergency = False

        # Check blocked topics
        for pattern in self._blocked_patterns:
            if pattern.search(query):
                return {
                    "safe": False,
                    "reason": f"Query contains a restricted topic. This system is designed for evidence-based medical QA only.",
                    "emergency": False,
                    "flags": ["blocked_topic"]
                }

        # Check emergency terms
        if self._emergency_terms.search(query):
            flags.append("emergency_term")
            is_emergency = True

        # Check toxicity of input
        toxicity_score = self._get_toxicity_score(query)
        if toxicity_score > self.threshold:
            flags.append("toxic_input")
            return {
                "safe": False,
                "reason": "Query flagged for potentially harmful content.",
                "emergency": False,
                "flags": flags,
                "toxicity_score": toxicity_score
            }

        return {
            "safe": True,
            "reason": "OK",
            "emergency": is_emergency,
            "flags": flags,
            "toxicity_score": toxicity_score
        }

    # ──────────────────────────────────────────
    # OUTPUT GUARDRAILS
    # ──────────────────────────────────────────

    def check_output(self, response: str, retrieval_scores: List[float] = None) -> Dict:
        """
        Validate generated response before showing to user.

        Returns:
            {safe: bool, modified_response: str, warnings: list, scores: dict}
        """
        warnings = []
        scores = {}

        # Toxicity check
        tox_score = self._get_toxicity_score(response)
        scores["toxicity"] = tox_score
        if tox_score > self.threshold:
            warnings.append("High toxicity detected in response.")
            response = "[Response filtered due to safety concerns. Please rephrase your query.]"

        # Bias check
        for pattern in self._bias_patterns:
            if pattern.search(response):
                warnings.append("Potential demographic bias detected.")
                break

        # Low confidence / hallucination risk
        if retrieval_scores:
            avg_score = sum(retrieval_scores) / len(retrieval_scores)
            scores["avg_retrieval_score"] = avg_score
            if avg_score < 0.4:
                warnings.append(
                    "Low evidence confidence. Response may not be well-supported by retrieved literature."
                )

        # Check for false certainty in uncertain cases
        certainty_flags = self._check_false_certainty(response)
        if certainty_flags:
            warnings.extend(certainty_flags)

        # Add medical disclaimer
        safe = len([w for w in warnings if "filtered" in w]) == 0
        final_response = response + "\n\n" + MEDICAL_DISCLAIMER

        return {
            "safe": safe,
            "modified_response": final_response,
            "original_response": response,
            "warnings": warnings,
            "scores": scores
        }

    def _get_toxicity_score(self, text: str) -> float:
        """Get toxicity score (0-1) for text."""
        if self._detoxify is None or not text.strip():
            return 0.0
        try:
            results = self._detoxify.predict(text)
            return float(results.get("toxicity", 0.0))
        except Exception:
            return 0.0

    def _check_false_certainty(self, text: str) -> List[str]:
        """Detect overly certain language in uncertain medical contexts."""
        warnings = []
        certainty_phrases = [
            r"\bcure\b", r"\bguaranteed\b", r"\balways works\b",
            r"\bdefinitely will\b", r"\b100%\b"
        ]
        for phrase in certainty_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                warnings.append(f"Potentially overconfident claim detected.")
                break
        return warnings

    def get_emergency_message(self) -> str:
        return (
            "⚠️ This query may relate to a medical emergency. "
            "If you or someone else is in immediate danger, please call emergency services (911/112) "
            "or go to the nearest emergency room immediately. "
            "This AI system cannot replace emergency medical care."
        )
