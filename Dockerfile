# Use Python 3.12 slim image
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ESSCOAI_ENV=prod \
    ESSCOAI_USE_LLM=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY WebDesign/API/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY WebDesign/ ./WebDesign/
COPY AI/ ./AI/
COPY Data/ ./Data/

# Create necessary directories
RUN mkdir -p /app/Data /app/WebDesign/static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uvicorn", "WebDesign.API.api:app", "--host", "0.0.0.0", "--port", "8000"]