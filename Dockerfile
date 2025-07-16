FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    spleeter librosa soundfile matplotlib numpy==1.24.3

WORKDIR /app
COPY main.py .

RUN mkdir input output

CMD ["python", "main.py"]
