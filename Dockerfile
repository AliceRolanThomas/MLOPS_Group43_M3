# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y supervisor

# Expose Flask (8080) and MLflow (5000) ports
EXPOSE 8080
EXPOSE 5000

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run the Flask app
CMD ["python", "src/app.py"]