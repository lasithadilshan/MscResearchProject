from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.models import User, Job, Document
from app.schemas.schemas import JobResponse, JobResult, ConvertTestCaseRequest, ConvertCucumberRequest, ConvertSeleniumRequest
from app.api.routers.auth import get_current_user
from app.worker.tasks import process_generation_task
import uuid
import json

router = APIRouter()

def enqueue_job(task_type: str, document_id: str, db: Session, user_id: int, input_text: str = None) -> JobResponse:
    # Verify document ownership
    doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == user_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        task_type=task_type,
        status="PENDING",
        document_id=document_id,
        owner_id=user_id
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Enqueue task
    process_generation_task.apply_async(args=[job_id, task_type, document_id, input_text])
    
    return job

@router.post("/user-stories", response_model=JobResponse)
def generate_user_stories(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return enqueue_job("user_stories", document_id, db, current_user.id)

@router.post("/test-cases", response_model=JobResponse)
def convert_to_test_cases(document_id: str, request: ConvertTestCaseRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return enqueue_job("test_cases", document_id, db, current_user.id, request.user_story_text)

@router.post("/cucumber", response_model=JobResponse)
def convert_to_cucumber(document_id: str, request: ConvertCucumberRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return enqueue_job("cucumber", document_id, db, current_user.id, request.test_case_text)

@router.post("/selenium", response_model=JobResponse)
def convert_to_selenium(document_id: str, request: ConvertSeleniumRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return enqueue_job("selenium", document_id, db, current_user.id, request.test_case_text)

@router.get("/job/{job_id}", response_model=JobResult)
def get_job_status(job_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(Job).filter(Job.id == job_id, Job.owner_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result_data = None
    if job.result:
        try:
            result_data = json.loads(job.result)
        except Exception:
            result_data = {"raw": job.result}
            
    return JobResult(status=job.status, result=result_data)
