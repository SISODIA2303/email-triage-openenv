# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY env/ ./env/
COPY tasks/ ./tasks/
COPY baseline/ ./baseline/
COPY openenv.yaml .

# Create a simple FastAPI app to expose the environment over HTTP
COPY app.py .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]