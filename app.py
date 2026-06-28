"""Provenance Guard - Flask API.

Milestone 3 scope: POST /submit runs Signal 1 (Groq LLM), writes a structured
audit entry, and returns content_id + attribution + a PLACEHOLDER confidence and
label. The real combined confidence score arrives in M4 and the real label
variants in M5; the placeholders are marked as such so they are obvious.
"""

import uuid

from flask import Flask, jsonify, request

import audit
import config
import detection

app = Flask(__name__)
audit.init_db()


def _provisional_attribution(p_ai):
    """M3 placeholder verdict from the single LLM signal.

    Replaced in M4 by the two-signal asymmetric scorer in scoring.py. Kept here
    only so /submit returns a meaningful attribution before the second signal
    exists.
    """
    if p_ai >= config.AI_THRESHOLD:
        return "likely_ai"
    if p_ai <= config.HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id", "anonymous")

    # --- input validation ---
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400

    text = text.strip()
    low_evidence = len(text) < config.MIN_TEXT_LENGTH

    content_id = uuid.uuid4().hex

    # --- Signal 1: LLM ---
    try:
        llm = detection.llm_signal(text)
    except Exception as exc:  # surface upstream/Groq failures cleanly
        return jsonify({"error": f"Detection failed: {exc}"}), 502

    attribution = _provisional_attribution(llm["p_ai"])

    # --- placeholders (real versions land in M4 confidence scoring / M5 labels) ---
    confidence = None  # placeholder until M4
    label = {
        "variant": "placeholder",
        "title": "Analysis in progress",
        "body": "Confidence scoring and the transparency label are added in later milestones.",
    }

    # --- audit log ---
    audit.log_event(
        event_type="decision",
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        p_ai=llm["p_ai"],          # provisional: single-signal until M4
        confidence=confidence,
        llm_score=llm["p_ai"],
        stylometry_score=None,     # Signal 2 added in M4
        status="classified",
        label_variant=label["variant"],
        detail={"text_excerpt": text[:280], "low_evidence": low_evidence, "llm_rationale": llm["rationale"]},
    )

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,           # placeholder (M4)
        "signals": {
            "llm": {"p_ai": llm["p_ai"], "rationale": llm["rationale"]},
        },
        "label": label,                     # placeholder (M5)
        "status": "classified",
        "low_evidence": low_evidence,
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": audit.get_log()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
