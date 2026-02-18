"""
RAG Answer Generator
Combines retrieved evidence with the fine-tuned LLM to generate grounded answers.
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from src.data_loader import load_config
except ImportError:
    from data_loader import load_config

try:
    from src.retriever import RetrievedEvidence
except ImportError:
    from retriever import RetrievedEvidence


@dataclass
class GeneratedAnswer:
    """Structured answer from the RAG pipeline."""
    question: str
    decision: str  # yes / no / maybe
    explanation: str
    evidence_used: List[RetrievedEvidence]
    confidence: float
    raw_output: str


class MedicalGenerator:
    """Generate evidence-grounded medical answers using fine-tuned BioMistral."""

    def __init__(self, model=None, tokenizer=None, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config
        self.llm_config = config["llm"]
        self.model = model
        self.tokenizer = tokenizer

    def compose_prompt(self, question: str, evidence_list: List[RetrievedEvidence]) -> str:
        """Compose the prompt with evidence and question."""
        evidence_text = ""
        for i, ev in enumerate(evidence_list):
            evidence_text += f"- {ev.text}\n"

        prompt = f"""### Instruction: You are a medical question-answering assistant. Answer the question strictly using the provided evidence. Do not use outside knowledge. If the evidence is insufficient, say "Insufficient evidence."

### Evidence:
{evidence_text}

### Question: {question}

### Answer:"""
        return prompt

    def generate(
        self,
        question: str,
        evidence_list: List[RetrievedEvidence],
    ) -> GeneratedAnswer:
        """
        Generate an answer using the LLM with retrieved evidence.

        Args:
            question: Medical question
            evidence_list: Retrieved evidence passages

        Returns:
            GeneratedAnswer with decision, explanation, and metadata
        """
        prompt = self.compose_prompt(question, evidence_list)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.llm_config.get("max_seq_length", 1024),
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.llm_config["max_new_tokens"],
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config["top_p"],
                repetition_penalty=self.llm_config["repetition_penalty"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer portion (after "### Answer:")
        answer_text = full_output.split("### Answer:")[-1].strip()

        # Parse decision and explanation
        decision, explanation = self._parse_answer(answer_text)

        # Calculate confidence from evidence scores
        avg_evidence_score = (
            sum(ev.score for ev in evidence_list) / len(evidence_list)
            if evidence_list
            else 0
        )

        return GeneratedAnswer(
            question=question,
            decision=decision,
            explanation=explanation,
            evidence_used=evidence_list,
            confidence=avg_evidence_score,
            raw_output=answer_text,
        )

    def _parse_answer(self, answer_text: str) -> tuple:
        """Parse the model output into decision and explanation."""
        answer_lower = answer_text.lower().strip()

        # Extract decision
        decision = "maybe"
        if answer_lower.startswith("yes"):
            decision = "yes"
        elif answer_lower.startswith("no"):
            decision = "no"
        elif answer_lower.startswith("maybe"):
            decision = "maybe"
        elif "insufficient evidence" in answer_lower:
            decision = "maybe"

        # Extract explanation
        explanation = answer_text
        for prefix in ["yes", "no", "maybe", "Yes", "No", "Maybe"]:
            if explanation.startswith(prefix):
                explanation = explanation[len(prefix):].strip()
                if explanation.startswith(",") or explanation.startswith("."):
                    explanation = explanation[1:].strip()
                break

        # Clean up explanation markers
        if "### Explanation:" in explanation:
            explanation = explanation.split("### Explanation:")[-1].strip()

        return decision, explanation

    def generate_without_evidence(self, question: str) -> str:
        """Generate answer WITHOUT evidence (for comparison/baseline)."""
        prompt = f"""### Instruction: You are a medical question-answering assistant. Answer the following medical question.

### Question: {question}

### Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.llm_config["max_new_tokens"],
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config["top_p"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output.split("### Answer:")[-1].strip()


# --- Main ---
if __name__ == "__main__":
    print("Generator module loaded. Use via pipeline.py for end-to-end inference.")
    print("Direct usage requires a loaded model. See pipeline.py or the notebook.")
