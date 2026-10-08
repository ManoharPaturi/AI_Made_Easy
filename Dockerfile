# AI Made Easy web server: browser UI + REST/WebSocket API on port 8765.
#   docker build -t ai-made-easy .
#   docker run -p 8765:8765 -e AIME_WEB_TOKEN=change-me -v aime-data:/data ai-made-easy
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    AIME_HOME=/data \
    KERAS_BACKEND=torch \
    QT_QPA_PLATFORM=offscreen

# libgomp: OpenMP runtime for LightGBM / XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY ai_made_easy ./ai_made_easy
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[web,vision,data,classic,export]"

RUN useradd --create-home --uid 1000 aime && mkdir -p /data && chown aime /data
USER aime
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/info')"
CMD ["aime", "web", "--host", "0.0.0.0", "--port", "8765"]
