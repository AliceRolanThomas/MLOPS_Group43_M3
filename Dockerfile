# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy files
# COPY requirements.txt requirements.txt
# COPY model.joblib model.joblib
# COPY vectorizer.joblib vectorizer.joblib
# COPY src/app.py app.py
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask (8080) and MLflow (5000) ports
EXPOSE 8080
EXPOSE 5000

# Run Flask and MLflow commands
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.app:app"]