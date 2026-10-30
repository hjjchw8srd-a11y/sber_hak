FROM python:3.11-slim

# System dependencies for LightGBM and C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, pre-trained model bundle, and sample data
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Healthcheck to verify service readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
