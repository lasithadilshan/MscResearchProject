from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# User Schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    
    class Config:
        from_attributes = True

# Document Schemas
class DocumentResponse(BaseModel):
    id: str
    filename: str
    upload_time: datetime
    text_length: int
    owner_id: int
    
    class Config:
        from_attributes = True

# Job Schemas
class JobResponse(BaseModel):
    id: str
    task_type: str
    status: str
    created_at: datetime
    document_id: str
    owner_id: int
    
    class Config:
        from_attributes = True

class JobResult(BaseModel):
    status: str
    result: Optional[dict] = None

# Input Schemas for generation
class ConvertTestCaseRequest(BaseModel):
    user_story_text: str

class ConvertCucumberRequest(BaseModel):
    test_case_text: str

class ConvertSeleniumRequest(BaseModel):
    test_case_text: str
