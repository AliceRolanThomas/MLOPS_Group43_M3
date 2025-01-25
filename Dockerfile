# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Expose Flask (8080) and MLflow (5000) ports
EXPOSE 8080
EXPOSE 5000

# Copy Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run Supervisor to manage Flask and MLflow
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
