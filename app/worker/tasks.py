from app.worker.celery_app import celery_app
from app.services.llm_service import generate_artifact
from app.core.database import SessionLocal
from app.models.models import Job
import json

@celery_app.task(bind=True)
def process_generation_task(self, job_id: str, task_type: str, document_id: str, input_text: str = None):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return

    job.status = "STARTED"
    db.commit()
    
    try:
        result = generate_artifact(task_type, document_id, input_text)
        job.status = "SUCCESS"
        job.result = json.dumps(result)
    except Exception as e:
        job.status = "FAILURE"
        job.result = json.dumps({"error": str(e)})
    finally:
        db.commit()
        db.close()
