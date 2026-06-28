"""Detection signals.

Signal 1 (LLM, Groq) and Signal 2 (stylometry, pure Python) both live here.
Each signal takes raw text and returns a dict whose first key is `p_ai`, a
probability in [0, 1] where 0 = looks human and 1 = looks AI.
"""

import json
import re
import statistics

from groq import Groq

import config

_client = None


def _get_client():
    """Lazily create the Groq client so importing this module never fails
    just because the key is unset (the /log route does not need Groq)."""
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set; cannot run Signal 1.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


_SYSTEM_PROMPT = (
    "You are a forensic text-attribution assistant. You estimate the probability "
    "that a passage was produced by an AI language model rather than written by a "
    "human. Be calibrated and cautious: many humans write formally, and many "
    "non-native English speakers write in ways that can resemble AI text. Only "
    "assign a high AI probability when there are strong, specific signals. "
    "Respond with JSON only."
)

_USER_TEMPLATE = (
    "Assess the following text and return JSON with exactly two keys:\n"
    '  "p_ai": a number from 0 to 1 (0 = definitely human, 1 = definitely AI),\n'
    '  "rationale": one or two sentences explaining the estimate.\n\n'
    "TEXT:\n{text}"
)


def _clamp01(value):
    """Coerce to a float in [0, 1]; fall back to 0.5 if it is not a number."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


def llm_signal(text):
    """Signal 1: ask the Groq LLM how likely the text is AI-generated.

    Returns: {"p_ai": float in [0,1], "rationale": str}
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
        ],
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return {
        "p_ai": _clamp01(data.get("p_ai")),
        "rationale": str(data.get("rationale", "")).strip(),
    }


# --- Signal 2: stylometric heuristics (pure Python) -------------------------

# Formulaic connectives that AI text over-produces (planning.md section 2).
_CONNECTIVES = [
    "moreover", "furthermore", "additionally", "in addition", "in conclusion",
    "it is important to note", "it is worth noting", "consequently",
    "as a result", "on the other hand", "in summary", "overall", "notably",
    "importantly", "ultimately", "therefore", "thus",
]

# "Rich" punctuation a human reaches for; AI tends to a flat comma-period rhythm.
_RICH_PUNCT = set(";:—–()\"'!?…")

_WORD_RE = re.compile(r"[A-Za-z']+")


def _words(text):
    return _WORD_RE.findall(text.lower())


def _linear(value, human_at, ai_at):
    """Map a raw feature onto an AI-likelihood in [0, 1] by linear interpolation:
    `human_at` -> 0.0, `ai_at` -> 1.0, clamped outside that range."""
    if human_at == ai_at:
        return 0.5
    frac = (value - human_at) / (ai_at - human_at)
    return max(0.0, min(1.0, frac))


def stylometry_signal(text):
    """Signal 2: surface statistics of the text.

    Returns: {"p_ai": float in [0,1], "features": {...}} where each feature is the
    raw measurement plus its per-feature AI-likelihood, and p_ai is their
    weighted blend.
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = _words(text)
    n_words = len(words)

    # 1. Burstiness: coefficient of variation of sentence length (in words).
    #    Human writing mixes long/short sentences (high CV); AI is uniform (low CV).
    sent_lengths = [len(_words(s)) for s in sentences]
    if len(sent_lengths) >= 2 and statistics.mean(sent_lengths) > 0:
        cv = statistics.pstdev(sent_lengths) / statistics.mean(sent_lengths)
        burst_ai = _linear(cv, human_at=0.6, ai_at=0.15)  # low CV -> AI
    else:
        cv = None
        burst_ai = 0.5  # too few sentences to tell

    # 2. Connective density: formulaic connectives per sentence.
    low = text.lower()
    conn_count = sum(low.count(c) for c in _CONNECTIVES)
    conn_ratio = conn_count / max(1, len(sentences))
    conn_ai = _linear(conn_ratio, human_at=0.0, ai_at=0.5)  # more -> AI

    # 3. Punctuation variety: distinct rich punctuation marks used.
    punct_variety = len({ch for ch in text if ch in _RICH_PUNCT})
    punct_ai = _linear(punct_variety, human_at=3, ai_at=0)  # less variety -> AI

    # 4. Lexical diversity: type-token ratio. Smoothly repetitive vocab -> AI.
    ttr = len(set(words)) / n_words if n_words else 0.0
    ttr_ai = _linear(ttr, human_at=0.7, ai_at=0.3)  # low TTR -> AI

    # 5. Repetition: fraction of repeated word bigrams.
    bigrams = list(zip(words, words[1:]))
    rep_frac = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    rep_ai = _linear(rep_frac, human_at=0.0, ai_at=0.3)  # more repeats -> AI

    # Weighted blend. Burstiness and connectives are the strongest signals;
    # TTR/repetition are noisier on short text so they carry less weight.
    weights = {"burstiness": 0.30, "connectives": 0.30, "punctuation": 0.15,
               "lexical_diversity": 0.10, "repetition": 0.15}
    p_ai = (
        weights["burstiness"] * burst_ai
        + weights["connectives"] * conn_ai
        + weights["punctuation"] * punct_ai
        + weights["lexical_diversity"] * ttr_ai
        + weights["repetition"] * rep_ai
    )

    return {
        "p_ai": round(p_ai, 3),
        "features": {
            "burstiness_cv": round(cv, 3) if cv is not None else None,
            "burstiness_ai": round(burst_ai, 3),
            "connective_ratio": round(conn_ratio, 3),
            "connective_ai": round(conn_ai, 3),
            "punctuation_variety": punct_variety,
            "punctuation_ai": round(punct_ai, 3),
            "type_token_ratio": round(ttr, 3),
            "ttr_ai": round(ttr_ai, 3),
            "repetition_frac": round(rep_frac, 3),
            "repetition_ai": round(rep_ai, 3),
        },
    }


if __name__ == "__main__":
    # Quick manual test harness: run `python detection.py` to inspect the signals
    # on a few inputs before they are wired into the endpoint. Stylometry runs
    # offline; the LLM line is commented out so this needs no network by default.
    samples = {
        "casual-human": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after."
        ),
        "formal-ai-ish": (
            "Artificial intelligence represents a transformative paradigm shift. "
            "It is important to note that while the benefits are numerous, it is "
            "equally essential to consider the ethical implications. Furthermore, "
            "stakeholders across various sectors must collaborate."
        ),
    }
    for name, sample in samples.items():
        print(f"\n=== {name} ===")
        print("stylometry:", stylometry_signal(sample))
        # print("llm:", llm_signal(sample))  # uncomment to also hit Groq
