import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
import joblib
import pandas as pd
import warnings
from pathlib import Path
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Set the tracking URI to the current directory
tracking_uri = Path(os.getcwd()).as_uri()
mlflow.set_tracking_uri(f"{tracking_uri}/mlruns")

CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'sci.med',
    'rec.sport.hockey',
    'talk.politics.misc'
]

def prepare_dataset():
    dataset = fetch_20newsgroups(subset='all',
                                 categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
    data = pd.DataFrame({'text': dataset.data, 'category': dataset.target})
    data['category'] = data['category'].map(lambda x: dataset.target_names[x])
    sampled_data = data.groupby('category').apply(lambda x: x.sample(n=10, random_state=42)).reset_index(drop=True)
    return sampled_data

def split_data(data):
    train_data, test_data = pd.DataFrame(), pd.DataFrame()
    for category, group in data.groupby('category'):
        train, test = train_test_split(group, test_size=0.2, random_state=42)
        train_data = pd.concat([train_data, train])
        test_data = pd.concat([test_data, test])
    return train_data, test_data

def objective(trial):
    # Prepare dataset
    data = prepare_dataset()
    train_data, test_data = split_data(data)
    X_train, y_train = train_data['text'], train_data['category']
    X_test, y_test = test_data['text'], test_data['category']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 5, 30)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Evaluate accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # Classification metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)

    mlflow.set_experiment("News Category Prediction")
    # Log trial results to MLflow
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set the run name
    run_name = f"RandomForest_NewsCategoryTraining_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        # Log the dataset
        mlflow.log_param("dataset", "Fetch_20newsgroups")  # You can also log any other details about the dataset
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        # Log classification metrics
        for class_label, metrics_dict in metrics.items():
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    mlflow.log_metric(f"{class_label}_{metric_name}", value)
        mlflow.sklearn.log_model(model, "RandomForestClassifierModel")
    
    return  metrics["accuracy"]

def train_best_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Get best hyperparameters
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Prepare dataset
    data = prepare_dataset()
    train_data, test_data = split_data(data)
    X_train, y_train = train_data['text'], train_data['category']
    X_test, y_test = test_data['text'], test_data['category']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train final model
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train_vec, y_train)

    # Save model and vectorizer
    joblib.dump(best_model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("Best model and vectorizer saved!")

if __name__ == "__main__":
    train_best_model()
