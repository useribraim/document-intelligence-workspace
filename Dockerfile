FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/demo ./data/demo

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn diw.api:app --host 0.0.0.0 --port ${PORT}"]
