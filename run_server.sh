#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Starting MediaToolbox Server..."
echo "Visit http://localhost:8000 to use the tools."
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload --reload-exclude "**/ENTER/**"

