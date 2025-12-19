FROM ghcr.io/astral-sh/uv:debian

RUN apt update && apt install -y libgl1 cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cd singtown-ai-trainer-classification-transformers && uv sync && uv run cache_weight.py
RUN cd rknn2 && uv sync && uv pip install ./rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl

CMD ["sh", "run.sh"]