# ---- Stage 1: install dependencies and train the model ----
FROM python:3.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /bin/
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies from uv.lock (no dev tools, no GPU library: the model runs on CPU)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-install-package nvidia-nccl-cu13

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-install-package nvidia-nccl-cu13 \
 && find .venv -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +

# Train -> models/model.bin
COPY data/support2.csv ./data/
RUN train

# ---- Stage 2: runtime image (only the venv, the code and the model) ----
FROM python:3.15-slim

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models

EXPOSE 9696
CMD ["serve"]
