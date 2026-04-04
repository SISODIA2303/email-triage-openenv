FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env/ ./env/
COPY tasks/ ./tasks/
COPY baseline/ ./baseline/
COPY server/ ./server/
COPY models.py .
COPY app.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]