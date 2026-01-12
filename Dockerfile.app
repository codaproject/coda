FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for Whisper, audio processing, and building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir .

# Download NLTK data needed by gilda
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"

# Pre-download Whisper model (assumes medium here)
RUN python -c "import whisper; whisper.load_model('medium')"

# Expose the web server port
EXPOSE 8000

# Run the web application
CMD ["python", "-m", "coda.app"]
