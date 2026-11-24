FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# CPU-only PyTorch keeps the image small; the LLM and embeddings run in Ollama on the host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[ml,dev]"

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY tests ./tests
COPY evaluation ./evaluation
COPY scripts ./scripts

EXPOSE 8501
CMD ["sh", "-c", "alembic upgrade head && streamlit run app/Home.py --server.address=0.0.0.0 --server.port=8501"]
