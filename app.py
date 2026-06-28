"""Provenance Guard - Flask API.

Milestone 4 scope: POST /submit runs both detection signals (Groq LLM +
stylometry), combines them with the asymmetric confidence scorer, and returns
content_id + verdict + a real confidence score. The transparency label is still
a PLACEHOLDER until M5, where it maps the verdict to reader-facing text.
"""

import uuid

from flask import Flask, jsonify, request

import audit
import config
import detection
import scoring

app = Flask(__name__)
audit.init_db()


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

    # --- detection pipeline: two independent signals ---
    try:
        llm = detection.llm_signal(text)
    except Exception as exc:  # surface upstream/Groq failures cleanly
        return jsonify({"error": f"Detection failed: {exc}"}), 502
    style = detection.stylometry_signal(text)

    # --- combine into the calibrated confidence score ---
    result = scoring.score(llm["p_ai"], style["p_ai"], low_evidence=low_evidence)

    # --- transparency label is still a placeholder until M5 ---
    label = {
        "variant": "placeholder",
        "title": "Analysis complete",
        "body": "The reader-facing transparency label text is added in Milestone 5.",
    }

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
        "label": label,                     # placeholder (M5)
        "status": "classified",
        "low_evidence": low_evidence,
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": audit.get_log()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
