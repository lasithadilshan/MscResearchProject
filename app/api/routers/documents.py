from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.models import User, Document
from app.schemas.schemas import DocumentResponse
from app.api.routers.auth import get_current_user
from app.services.document_service import extract_text_from_file, create_vector_store
import time
import uuid

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    try:
        file_content = await file.read()
        text = extract_text_from_file(file_content, file.filename)
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        document_id = str(uuid.uuid4())
        
        # Store in ChromaDB
        create_vector_store(document_id, text)
        
        # Save to DB
        new_doc = Document(
            id=document_id,
            filename=file.filename,
            text_length=len(text),
            owner_id=current_user.id
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        return new_doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=list[DocumentResponse])
def get_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Document).filter(Document.owner_id == current_user.id).all()
