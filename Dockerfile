FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (Java needed for tabula)
RUN apt-get update && apt-get install -y \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app1.py .
COPY chatbot.py .
COPY data/ ./data/

# Create output directory
RUN mkdir -p /app/output

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/health')"

ENV PYTHONUNBUFFERED=1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]