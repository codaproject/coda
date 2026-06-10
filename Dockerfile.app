FROM python:3.13-slim

# Compute device for PyTorch / Whisper. "cpu" (default) installs the CPU-only
# torch wheel, producing a much smaller image with no CUDA libraries. "cuda"
# keeps the default CUDA-enabled torch wheel for NVIDIA GPU hosts; run that
# image with `docker run --gpus all ...`.
ARG COMPUTE_DEVICE=cpu

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

# Install the package. For the CPU device, pre-install the CPU-only torch wheel
# first so the package install reuses it instead of pulling the multi-GB CUDA
# build that ships by default.
RUN if [ "$COMPUTE_DEVICE" = "cpu" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ; \
    fi && \
    pip install --no-cache-dir .

# Download NLTK data and Gilda resources, then build a namespace-filtered
# SQLite grounding db (only the namespaces CODA grounds to) for fast startup.
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')" && \
    python -m gilda.resources && \
    python -m coda.grounding.build_grounding_db /app/grounding_terms.db && \
    rm -f /root/.data/gilda/*/grounding_terms.tsv.gz

ENV GILDA_SQLITE_DB=/app/grounding_terms.db

# Device used by the app at runtime; matches the torch build above. Overridable,
# and the app falls back to CPU if CUDA is requested but unavailable.
ENV CODA_DEVICE=${COMPUTE_DEVICE}

# Pre-download Whisper model (assumes medium here)
RUN python -c "import whisper; whisper.load_model('medium')"

# Expose the web server port
EXPOSE 8000

# Run the web application
CMD ["python", "-m", "coda.app"]
