FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="/app/06-production:/app/02-tools-workflow:/app"
WORKDIR /app
COPY agent/learning_lab /app
RUN pip install --no-cache-dir 'fastapi>=0.110' 'uvicorn[standard]>=0.27' \
    && useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "stage06.app:app", "--host", "0.0.0.0", "--port", "8000"]
