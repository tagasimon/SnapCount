FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models
WORKDIR /srv

COPY requirements.txt .
# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app

RUN useradd --create-home snapcount && mkdir -p /models && chown snapcount /models
USER snapcount

# Model weights (~1.7 GB) download here on first start. Mount a volume at /models
# so they survive redeploys.
VOLUME /models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
