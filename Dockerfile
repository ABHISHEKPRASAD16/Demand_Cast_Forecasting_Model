# Multi-stage build: the builder stage compiles/installs into a throwaway
# prefix, and only that installed tree (not pip caches, build tools, or the
# full source checkout) ends up in the final image.
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements-serving.txt pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt && \
    pip install --no-cache-dir --prefix=/install --no-deps .

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "demandcast.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
