FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads, results, and models
RUN mkdir -p uploads results models
RUN chmod 777 uploads results models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create a place for pre-trained models (optional)
# If you have pre-trained models, they would be stored here
RUN mkdir -p models/luts
RUN mkdir -p models/filters

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app --timeout 300