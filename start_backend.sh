#!/bin/bash
echo "Starting backend..."

conda activate ml2
source .venv/bin/activate
cd backend

echo "Starting Celery scrape worker (concurrency=1)..."
celery -A celery_app worker --loglevel=info -Q scrape -c 1 -n scrape@%h &

# Translation worker: concurrency=2 for parallel GPU/CPU processing  
echo "Starting Celery translation worker (concurrency=2)..."
celery -A celery_app worker --loglevel=info -Q celery -c 2 -n translate@%h &

# FastAPI server
echo "Starting FastAPI server..."
python app.py
