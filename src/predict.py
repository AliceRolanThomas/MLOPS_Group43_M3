import joblib

def predict(text):
    # Load model and vectorizer
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")

    # Transform input text
    text_vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vec)
    return prediction[0]

if __name__ == "__main__":
    sample_text = "This is a sample text about graphics and design."
    category = predict(sample_text)
    print(f"Predicted category: {category}")