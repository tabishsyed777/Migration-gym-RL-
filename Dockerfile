FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY migrationgym ./migrationgym
COPY artifacts/policy.npz ./artifacts/policy.npz
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home runner
USER runner
ENTRYPOINT ["migrationgym"]
CMD ["demo"]
