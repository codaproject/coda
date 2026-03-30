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
RUN pip install --no-cache-dir \
    "sentence-transformers" \
    "scikit-learn" \
    "openai>=1.0.0" \
    "openacme[embeddings] @ git+https://github.com/gyorilab/openacme.git"

# Download NLTK data and Gilda resources, then build sqlite db for fast startup
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')" && \
    python -m gilda.resources && \
    python -m gilda.resources.sqlite_adapter /app/grounding_terms.db

ENV GILDA_SQLITE_DB=/app/grounding_terms.db

# Pre-download Whisper model (assumes medium here)
RUN python -c "import whisper; whisper.load_model('medium')"

# Expose the web server port
EXPOSE 8000

# Run the web application
CMD ["python", "-m", "coda.app"]
