"""Detection signals.

Signal 1 (LLM, Groq) lives here now. Signal 2 (stylometry) is added in M4.
Each signal takes raw text and returns a dict whose first key is `p_ai`, a
probability in [0, 1] where 0 = looks human and 1 = looks AI.
"""

import json

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


if __name__ == "__main__":
    # Quick manual test harness: run `python detection.py` to inspect Signal 1
    # on a few inputs before it is wired into the endpoint.
    samples = {
        "casual-human": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but way too much sodium."
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
        print(llm_signal(sample))
