from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    documents = relationship("Document", back_populates="owner")
    jobs = relationship("Job", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
    text_length = Column(Integer)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="documents")
    jobs = relationship("Job", back_populates="document")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, index=True) # Celery task ID
    task_type = Column(String) # user_stories, test_cases, cucumber, selenium
    status = Column(String) # PENDING, STARTED, SUCCESS, FAILURE
    result = Column(Text, nullable=True) # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document_id = Column(String, ForeignKey("documents.id"))
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    document = relationship("Document", back_populates="jobs")
    owner = relationship("User", back_populates="jobs")
