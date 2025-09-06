import argparse
import time
import logging
from typing import Any, Dict, List
from pathlib import Path
from waitress import serve

from flask import Flask, request, jsonify
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ---- Prometheus metrics ----
PRED_REQ = Counter("prediction_requests_total", "Total prediction requests")
PRED_ERRORS = Counter("prediction_errors_total", "Total prediction errors")
PRED_DURATION = Histogram("prediction_duration_seconds", "Prediction duration seconds")
LAST_PRED_TS = Gauge("last_prediction_timestamp_seconds", "Unix timestamp of last prediction")
MODEL_INFO = Gauge("model_info", "Model info gauge (always 1)", ["model_uri", "model_version"])

# ---- App setup ----
app = Flask("model_server")
model_uri = "D:/smsml_ratu-chairunisa/mlartifacts/0/1d2c7fb24b4f4a05b9a51419dbcff9d6/artifacts/model"

def load_model(model_uri: str):
    """Load mlflow pyfunc model. Returns callable model object."""
    logging.info(f"Loading model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    PRED_REQ.inc()
    start = time.time()
    try:
        payload = request.get_json(force=True)
        instances = payload.get("instances", None)
        if instances is None:
            raise ValueError("JSON must contain 'instances' key")
        # model expects pandas-like input; mlflow pyfunc accepts list-of-lists or DataFrame
        preds = model.predict(instances)
        duration = time.time() - start
        PRED_DURATION.observe(duration)
        LAST_PRED_TS.set(time.time())
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        PRED_ERRORS.inc()
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 400

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Model server is running 🚀",
        "endpoints": {
            "/health": "GET - check health",
            "/predict": "POST - send JSON with 'instances'",
            "/metrics": "GET - Prometheus metrics"
        }
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=True,
                        help="MLflow model URI (e.g. 'runs:/<run_id>/model' or local path)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model_version", type=str, default="v1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    global model
    model = load_model(args.model_uri)
    # set model info gauge label for discovery
    MODEL_INFO.labels(model_uri=args.model_uri, model_version=args.model_version).set(1.0)

    serve(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()