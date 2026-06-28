"""Confidence scoring: combine the two signals into one calibrated result.

Implements planning.md section 3 exactly:
  - p_ai in [0,1]; 0.5 means "genuinely can't tell"
  - confidence = 2 * |p_ai - 0.5|
  - verdict bands: ai >= 0.70, human <= 0.40, else uncertain  (asymmetric)
  - disagreement between signals shrinks p_ai toward 0.5, so a confident "ai"
    verdict requires the two signals to AGREE
  - low-evidence (short) text caps how far p_ai can move from 0.5, so we never
    loudly accuse on the basis of a couple of sentences
"""

import config

# The LLM reasons about meaning and carries more weight; stylometry is the
# cheaper formal check that guards against the LLM's miscalibration.
W_LLM = 0.6
W_STYLE = 0.4

# When text is flagged low-evidence, cap |p_ai - 0.5| at this. 0.175 keeps p_ai
# inside [0.325, 0.675], below the 0.70 AI bar -> short text can never be "ai".
LOW_EVIDENCE_MAX_DEV = 0.175


def _clamp01(value):
    return max(0.0, min(1.0, value))


def score(llm_p_ai, stylometry_p_ai, low_evidence=False):
    """Combine two signal probabilities into the final result dict."""
    blend = W_LLM * llm_p_ai + W_STYLE * stylometry_p_ai

    # Disagreement -> uncertainty. spread 0 keeps the full deviation; large
    # spread pulls p_ai back toward 0.5. This is how "signals must agree for a
    # confident AI call" is enforced without a hard gate.
    spread = abs(llm_p_ai - stylometry_p_ai)
    shrink = 1.0 - min(1.0, spread)
    p_ai = 0.5 + (blend - 0.5) * shrink

    if low_evidence:
        dev = max(-LOW_EVIDENCE_MAX_DEV, min(LOW_EVIDENCE_MAX_DEV, p_ai - 0.5))
        p_ai = 0.5 + dev

    p_ai = _clamp01(p_ai)
    confidence = 2 * abs(p_ai - 0.5)

    if p_ai >= config.AI_THRESHOLD:
        verdict = "ai"
    elif p_ai <= config.HUMAN_THRESHOLD:
        verdict = "human"
    else:
        verdict = "uncertain"

    return {
        "p_ai": round(p_ai, 3),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "blend": round(blend, 3),
        "signal_spread": round(spread, 3),
        "low_evidence": low_evidence,
        "weights": {"llm": W_LLM, "stylometry": W_STYLE},
    }
