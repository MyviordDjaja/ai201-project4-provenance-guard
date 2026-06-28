"""Central configuration for Provenance Guard.

Thresholds and model names live here so the scorer (M4) and label builder (M5)
read from one source of truth that matches planning.md.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Groq / Signal 1 ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Storage ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "provenance.db")

# --- Input validation ---
# Text shorter than this is flagged low-evidence (caps confidence in M4).
MIN_TEXT_LENGTH = 40

# --- Verdict thresholds (planning.md section 3) ---
# Asymmetric on purpose: it takes more evidence to call something AI than human.
AI_THRESHOLD = 0.70      # p_ai >= this  -> verdict "ai"
HUMAN_THRESHOLD = 0.40   # p_ai <= this  -> verdict "human"
# anything in between -> "uncertain"
