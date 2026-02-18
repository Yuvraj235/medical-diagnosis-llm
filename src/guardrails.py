"""
Safety and Clinical Guardrails
Ensures responsible AI usage in healthcare contexts.
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass

try:
    from src.data_loader import load_config
except ImportError:
    from data_loader import load_config

try:
    from src.generator import GeneratedAnswer
except ImportError:
    from generator import GeneratedAnswer


@dataclass
class SafetyCheckResult:
    """Result of safety check on a generated answer."""
    is_safe: bool
    warnings: List[str]
    modified_answer: str
    hallucination_score: float
    is_medical_query: bool


class ClinicalGuardrails:
    """
    Implements safety guardrails for medical AI responses:
    1. Disclaimer injection
    2. Confidence thresholding
    3. Hallucination detection
    4. Medical scope filtering
    5. Uncertainty flagging
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        guard_config = config["guardrails"]
        self.confidence_threshold = guard_config["confidence_threshold"]
        self.hallucination_threshold = guard_config["hallucination_overlap_threshold"]
        self.medical_keywords = set(guard_config["medical_keywords"])
        self.disclaimer = guard_config["disclaimer"]

    def check(self, answer: GeneratedAnswer) -> SafetyCheckResult:
        """
        Run all safety checks on a generated answer.

        Args:
            answer: GeneratedAnswer from the generator

        Returns:
            SafetyCheckResult with safety status and modified answer
        """
        warnings = []
        modified = answer.explanation

        # 1. Check if query is medical
        is_medical = self._is_medical_query(answer.question)
        if not is_medical:
            warnings.append("NON_MEDICAL_QUERY: This question may not be medical in nature.")
            modified = "This system is designed for medical questions only. Your query appears to be outside the medical domain. " + modified

        # 2. Check confidence
        if answer.confidence < self.confidence_threshold:
            warnings.append(f"LOW_CONFIDENCE: Evidence similarity ({answer.confidence:.2f}) below threshold ({self.confidence_threshold})")
            modified = "Note: The retrieved evidence has low relevance to this question. The answer may not be reliable.\n\n" + modified

        # 3. Check hallucination
        hallucination_score = self._check_hallucination(answer)
        if hallucination_score < self.hallucination_threshold:
            warnings.append(f"POTENTIAL_HALLUCINATION: Low evidence grounding (overlap: {hallucination_score:.2f})")
            modified = "Warning: Parts of this answer may not be directly supported by the retrieved evidence.\n\n" + modified

        # 4. Flag uncertainty
        if answer.decision == "maybe":
            warnings.append("UNCERTAIN_ANSWER: The model indicates uncertainty.")
            modified = "The evidence is inconclusive. " + modified

        # 5. Add disclaimer
        modified = self.disclaimer + "\n\n" + modified

        is_safe = len([w for w in warnings if "HALLUCINATION" in w or "NON_MEDICAL" in w]) == 0

        return SafetyCheckResult(
            is_safe=is_safe,
            warnings=warnings,
            modified_answer=modified,
            hallucination_score=hallucination_score,
            is_medical_query=is_medical,
        )

    def _is_medical_query(self, question: str) -> bool:
        """Check if the question is medical in nature."""
        question_lower = question.lower()
        words = set(re.findall(r'\w+', question_lower))

        # Check keyword overlap
        overlap = words & self.medical_keywords
        if len(overlap) >= 1:
            return True

        # Check common medical question patterns
        medical_patterns = [
            r"does .+ (reduce|increase|cause|prevent|treat|cure)",
            r"is .+ (effective|safe|recommended|associated)",
            r"can .+ (help|prevent|treat|cause)",
            r"what (is|are) the (symptom|treatment|risk|cause|effect)",
            r"(diagnosis|prognosis|therapy|medication|dosage)",
        ]
        for pattern in medical_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def _check_hallucination(self, answer: GeneratedAnswer) -> float:
        """
        Check if the answer is grounded in retrieved evidence.
        Uses word overlap as a simple grounding metric.

        Returns:
            Float between 0 and 1 (1 = fully grounded, 0 = no overlap)
        """
        if not answer.evidence_used or not answer.explanation:
            return 0.0

        # Get answer words (excluding stop words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "nor", "this", "that", "these", "those", "it", "its",
        }

        answer_words = set(re.findall(r'\w+', answer.explanation.lower())) - stop_words

        # Get evidence words
        evidence_text = " ".join(ev.text for ev in answer.evidence_used)
        evidence_words = set(re.findall(r'\w+', evidence_text.lower())) - stop_words

        if not answer_words:
            return 0.0

        # Calculate overlap
        overlap = answer_words & evidence_words
        score = len(overlap) / len(answer_words)

        return score

    def format_safe_response(self, answer: GeneratedAnswer, check: SafetyCheckResult) -> str:
        """Format the final safe response with all components."""
        response = []

        # Disclaimer
        response.append(self.disclaimer)
        response.append("")

        # Decision
        decision_emoji = {"yes": "YES", "no": "NO", "maybe": "UNCERTAIN"}
        response.append(f"Decision: {decision_emoji.get(answer.decision, answer.decision.upper())}")
        response.append("")

        # Explanation
        response.append(f"Explanation: {answer.explanation}")
        response.append("")

        # Evidence sources
        response.append("Evidence Sources:")
        for i, ev in enumerate(answer.evidence_used):
            response.append(f"  [{i+1}] PubMed {ev.source_id} (relevance: {ev.score:.2f})")
            response.append(f"      {ev.text[:150]}...")
        response.append("")

        # Confidence
        response.append(f"Confidence: {answer.confidence:.2f}")

        # Warnings
        if check.warnings:
            response.append("")
            response.append("Safety Warnings:")
            for w in check.warnings:
                response.append(f"  - {w}")

        return "\n".join(response)


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    guardrails = ClinicalGuardrails(config)

    # Test medical query detection
    test_queries = [
        "Does aspirin reduce the risk of heart attack?",
        "Is metformin effective for diabetes?",
        "What is the weather today?",
        "Can exercise help with depression?",
        "What is the best pizza restaurant?",
    ]

    print("=== Medical Query Detection ===")
    for q in test_queries:
        is_med = guardrails._is_medical_query(q)
        print(f"  {'[MEDICAL]' if is_med else '[NON-MED]'} {q}")
