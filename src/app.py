from flask import Flask, request, jsonify
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import os 
app = Flask(__name__)

# Set the port from the environment variable or use the default (8080)
# port = int(os.environ.get("PORT", 8080))

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Start an MLflow experiment
mlflow.set_experiment('flask_model_prediction')
@app.route('/')
def home():
    return "Hello, World!"

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