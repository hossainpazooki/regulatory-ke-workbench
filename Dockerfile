FROM python:3.11-slim

WORKDIR /app

# Install curl (healthcheck) and PostgreSQL client libs (psycopg2)
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ backend/
COPY data/ data/

# Expose port (Railway uses dynamic $PORT)
EXPOSE ${PORT:-8000}

# Health check (use $PORT for Railway compatibility)
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run with uvicorn - use shell form to expand $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
