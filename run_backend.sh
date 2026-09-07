#!/bin/bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
celery -A app.worker.celery_app.celery_app worker --loglevel=info &
wait
