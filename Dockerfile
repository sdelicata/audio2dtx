FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy essential files
COPY PredictOnset.h5 /app/
COPY SimfilesTemplate.zip /app/
COPY main.py /app/

# Copy the new modular source code
COPY src/ /app/src/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Add the src directory to Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Set entry point
ENTRYPOINT ["python", "main.py"]