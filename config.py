"""
config.py - Central configuration for Medical RAG Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "pubmedqa")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512
BATCH_SIZE_EMBED = 32

# ─────────────────────────────────────────────
# LLM (Base model for fine-tuning)
# ─────────────────────────────────────────────
# Use a smaller open-source model suitable for medical QA
BASE_LLM_NAME = "microsoft/BioGPT-Large-PubMedQA"   # BioGPT specialized for PubMedQA
# Alternatives: "BioMistral/BioMistral-7B", "medalpaca/medalpaca-7b"
FALLBACK_LLM_NAME = "microsoft/BioGPT"               # Fallback if large model unavailable

# ─────────────────────────────────────────────
# LoRA FINE-TUNING
# ─────────────────────────────────────────────
LORA_R = 16                    # LoRA rank
LORA_ALPHA = 32                # LoRA alpha scaling
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
MAX_TRAIN_SAMPLES = 1000       # Use PQA-L expert-annotated subset

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
TOP_K_CHUNKS = 5               # Number of chunks to retrieve
CHUNK_SIZE = 300               # Words per chunk
CHUNK_OVERLAP = 50             # Word overlap between chunks
SIMILARITY_THRESHOLD = 0.5    # Minimum cosine similarity
INDEX_TYPE = "faiss"           # "faiss" or "chroma"

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.1

# ─────────────────────────────────────────────
# SAFETY / GUARDRAILS
# ─────────────────────────────────────────────
TOXICITY_THRESHOLD = 0.5
MEDICAL_DISCLAIMER = (
    "⚠️ This is an AI-generated response for research purposes only. "
    "It does not constitute medical advice. Always consult a qualified healthcare professional."
)
BLOCKED_TOPICS = [
    "suicide", "self-harm", "drug abuse instructions",
    "illegal medication procurement", "euthanasia instructions"
]

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_BATCH_SIZE = 8
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"   # pqa_labeled (1000 expert), pqa_artificial, pqa_unlabeled
PUBMEDQA_SPLIT = "train"

LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}
LABEL_NAMES = ["yes", "no", "maybe"]
