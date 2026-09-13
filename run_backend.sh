#!/bin/bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
celery -A app.worker.celery_app.celery_app worker --pool=threads --concurrency=4 --loglevel=info &
wait
