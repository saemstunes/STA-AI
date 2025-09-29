
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV MODEL_NAME=microsoft/Phi-3.5-mini-instruct
ENV LOG_LEVEL=INFO
ENV RAILWAY_ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy scripts first (so model can download before code changes)
COPY scripts/download_model.py ./scripts/
RUN python scripts/download_model.py

# Copy all backend code
COPY . .

# Copy built frontend assets from Node stage
COPY --from=frontend-builder /frontend/dist ./static

# Create necessary directories
RUN mkdir -p models logs static

# Create non-root user
RUN useradd -m -u 1000 saemsai
RUN chown -R saemsai:saemsai /app
USER saemsai

# Expose port (MUST MATCH RAILWAY CONFIG)
EXPOSE 8080

# Health check (uses internal port)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start application
CMD ["python", "railway_app.py"]
