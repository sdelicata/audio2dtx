FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all project files
COPY AutoChart.ipynb /app/
COPY PredictOnset.h5 /app/
COPY SimfilesTemplate.zip /app/
COPY audio_to_chart.py /app/
COPY main.py /app/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set entry point
ENTRYPOINT ["python", "main.py"]
