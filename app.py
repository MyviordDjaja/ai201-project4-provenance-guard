"""Provenance Guard - Flask API.

Full production layer (M5): POST /submit runs both detection signals, scores
them, builds the transparency label, rate-limits callers, and audits every
decision. POST /appeal lets a creator contest a classification, flipping the
content to 'under_review' and logging the appeal next to the original decision.
"""

import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
import config
import detection
import labels
import scoring

app = Flask(__name__)
audit.init_db()

# Rate limiting. In-memory storage is fine for a single-process local deployment;
# a real deployment would point storage_uri at Redis. Limits and reasoning are
# documented in the README (planning.md section 8).
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
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

    # --- detection pipeline: two independent signals ---
    try:
        llm = detection.llm_signal(text)
    except Exception as exc:  # surface upstream/Groq failures cleanly
        return jsonify({"error": f"Detection failed: {exc}"}), 502
    style = detection.stylometry_signal(text)

    # --- combine into the calibrated confidence score ---
    result = scoring.score(llm["p_ai"], style["p_ai"], low_evidence=low_evidence)

    # --- transparency label (reader-facing text) ---
    label = labels.build_label(result["verdict"], result["confidence"])

    # --- audit log (now records both signal scores + the combined result) ---
    audit.log_event(
        event_type="decision",
        content_id=content_id,
        creator_id=creator_id,
        attribution=result["verdict"],
        p_ai=result["p_ai"],
        confidence=result["confidence"],
        llm_score=llm["p_ai"],
        stylometry_score=style["p_ai"],
        status="classified",
        label_variant=label["variant"],
        detail={
            "text_excerpt": text[:280],
            "low_evidence": low_evidence,
            "llm_rationale": llm["rationale"],
            "stylometry_features": style["features"],
            "signal_spread": result["signal_spread"],
            "blend": result["blend"],
        },
    )

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "verdict": result["verdict"],
        "p_ai": result["p_ai"],
        "confidence": result["confidence"],
        "signals": {
            "llm": {"p_ai": llm["p_ai"], "rationale": llm["rationale"]},
            "stylometry": {"p_ai": style["p_ai"], "features": style["features"]},
        },
        "label": label,
        "status": "classified",
        "low_evidence": low_evidence,
    })


@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per minute;20 per day")
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    reasoning = data.get("creator_reasoning")

    if not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not isinstance(reasoning, str) or not reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    original = audit.get_latest_decision(content_id)
    if original is None:
        return jsonify({"error": f"No decision found for content_id '{content_id}'."}), 404

    appeal_id = uuid.uuid4().hex

    # Flip the original content to under_review, then log the appeal next to the
    # decision it contests (no automated re-classification - a human reviews it).
    audit.update_status(content_id, "under_review")
    audit.log_event(
        event_type="appeal",
        content_id=content_id,
        creator_id=original.get("creator_id"),
        attribution=original.get("attribution"),
        p_ai=original.get("p_ai"),
        confidence=original.get("confidence"),
        llm_score=original.get("llm_score"),
        stylometry_score=original.get("stylometry_score"),
        status="under_review",
        label_variant=original.get("label_variant"),
        detail={
            "appeal_id": appeal_id,
            "appeal_reasoning": reasoning.strip(),
            "original_decision_id": original.get("id"),
            "original_verdict": original.get("attribution"),
        },
    )

    return jsonify({
        "content_id": content_id,
        "appeal_id": appeal_id,
        "status": "under_review",
        "message": "Appeal received. The classification is now under review by a human.",
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": audit.get_log()})


if __name__ == "__main__":
    # use_reloader=False keeps a single process (the reloader otherwise spawns a
    # parent + child); debug stays on for readable error pages during development.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
