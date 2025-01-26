from flask import Flask, request, jsonify
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import os 
import subprocess

app = Flask(__name__)

# Set the port from the environment variable or use the default (8080)
# port = int(os.environ.get("PORT", 8080))

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Start an MLflow experiment
mlflow.set_experiment('flask_model_prediction')
# Define the MLflow tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask App with MLflow!"})

@app.route('/start-mlflow', methods=['POST'])
def start_mlflow_ui():
    """Start the MLflow UI server from Flask."""
    try:
        # Start MLflow UI as a subprocess
        mlflow_ui_command = [
            "mlflow", "ui",
            "--host", "0.0.0.0",  # Expose to all network interfaces
            "--port", "5000"       # Port for the MLflow UI
        ]
        
        # Run the command to start the MLflow UI
        subprocess.Popen(mlflow_ui_command)

        return jsonify({"message": "MLflow UI started at http://127.0.0.1:5000"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status-mlflow', methods=['GET'])
def mlflow_status():
    """Check if MLflow is running."""
    try:
        # Try connecting to the MLflow server
        response = subprocess.run(
            ["curl", "-s", f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/version"],
            capture_output=True,
            text=True,
        )
        if response.returncode == 0:
            return jsonify({"message": "MLflow is running"}), 200
        else:
            return jsonify({"message": "MLflow is not running"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform and predict
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
     # Log the prediction request and metrics (example: log the prediction result)
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set the run name
    run_name = f"RandomForest_NewsCategoryPrediction_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_text", text)
        mlflow.log_param("prediction", prediction[0])
    return jsonify({"category": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)