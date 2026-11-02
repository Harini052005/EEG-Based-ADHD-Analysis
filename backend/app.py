"""
Backend Flask Application: Serves REST API and Frontend UI.
Usage:
  python backend/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory
from backend.data_service import data_service
from backend.inference import inference_service
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    target = FRONTEND_DIR / path
    if target.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/overview", methods=["GET"])
def api_overview():
    try:
        overview = data_service.get_overview()
        return jsonify({"status": "success", "data": overview})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/participants", methods=["GET"])
def api_participants():
    try:
        participants = data_service.get_participants_list()
        return jsonify({"status": "success", "count": len(participants), "data": participants})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/signal/trace", methods=["GET"])
def api_signal_trace():
    try:
        p_id = request.args.get("participant_id", default=None)
        channel = request.args.get("channel", default="Cz")
        epoch_idx = int(request.args.get("epoch_idx", default=0))
        trace = data_service.get_signal_trace(participant_id=p_id, channel=channel, epoch_idx=epoch_idx)
        return jsonify({"status": "success", "data": trace})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/channel-experiments", methods=["GET"])
def api_channel_experiments():
    try:
        metrics = data_service.get_channel_metrics()
        return jsonify({"status": "success", "data": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/noise-simulation", methods=["GET"])
def api_noise_simulation():
    try:
        metrics = data_service.get_noise_metrics()
        return jsonify({"status": "success", "data": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/participant", methods=["POST"])
def api_predict_participant():
    try:
        body = request.get_json(force=True) or {}
        p_id = body.get("participant_id")
        model = body.get("model", "svm")
        montage = body.get("montage", "full_19ch")

        if not p_id:
            return jsonify({"status": "error", "message": "Missing 'participant_id' parameter."}), 400

        result = inference_service.predict_participant(p_id, model_type=model, montage=montage)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/predict/epoch", methods=["POST"])
def api_predict_epoch():
    try:
        body = request.get_json(force=True) or {}
        epoch_id = int(body.get("epoch_id", 0))
        model = body.get("model", "svm")
        montage = body.get("montage", "full_19ch")

        result = inference_service.predict_epoch(epoch_id, model_type=model, montage=montage)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    print("=" * 70)
    print("Starting EEG ADHD Analysis Backend Server...")
    print("Serving UI at: http://127.0.0.1:5000")
    print("API at:        http://127.0.0.1:5000/api/overview")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
