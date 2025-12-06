import json
import os
import re
import time
from io import BytesIO

import numpy as np
import pandas as pd
import pdfplumber
import pptx
from docx import Document
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(
    title="SDLC Automate API",
    description="API for BRD to User Story, Test Case, Cucumber Script, and Selenium Script generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("OPENAI_API_KEY and GOOGLE_API_KEY must be set in environment variables")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Pydantic models for request/response
class GenerateUserStoriesRequest(BaseModel):
    model: str = "Open AI GPT 4.1"


class ConvertTestCaseRequest(BaseModel):
    user_story_text: str
    model: str = "Open AI GPT 4.1"


class ConvertCucumberRequest(BaseModel):
    test_case_text: str
    model: str = "Open AI GPT 4.1"


class ConvertSeleniumRequest(BaseModel):
    test_case_text: str
    model: str = "Open AI GPT 4.1"


class QualityAssessmentResponse(BaseModel):
    confidence_score: float
    match_score: float
    overall_score: float
    confidence_level: str
    match_level: str
    overall_level: str


# Global storage for uploaded files and vector stores (in production, use a database)
uploaded_documents = {}
vector_stores = {}
qa_chains = {}


def parse_json_output(text: str):
    """Extract JSON from model output, stripping optional code fences."""
    clean = text.strip()
    fence = re.search(r"```json\s*(.*?)```", clean, re.DOTALL)
    if fence:
        clean = fence.group(1).strip()
    else:
        fence = re.search(r"```\s*(.*?)```", clean, re.DOTALL)
        if fence:
            clean = fence.group(1).strip()
    try:
        return json.loads(clean), None
    except Exception as e:
        return clean, str(e)


def get_or_create_qa_chain(document_id: str, model_selection: str) -> RetrievalQA:
    """Ensure QA chain uses the requested model; rebuild if model changed."""
    if document_id not in uploaded_documents or document_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")

    stored_model = uploaded_documents[document_id].get("model")
    if stored_model != model_selection or document_id not in qa_chains:
        llm = initialize_llm(model_selection)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_stores[document_id].as_retriever()
        )
        qa_chains[document_id] = qa_chain
        uploaded_documents[document_id]["model"] = model_selection
        print(f"[QA_CHAIN] Rebuilt for doc={document_id} model={model_selection}")
    return qa_chains[document_id]


# Function to calculate confidence level based on prompt-answer accuracy
def calculate_confidence_level(prompt: str, answer: str) -> float:
    """Calculate confidence level based on how well the answer addresses the prompt."""
    try:
        # Preprocess texts
        prompt_clean = re.sub(r'[^\w\s]', '', prompt.lower())
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        
        # Extract key terms from prompt
        prompt_keywords = set(prompt_clean.split())
        answer_words = set(answer_clean.split())
        
        # Calculate keyword overlap
        common_keywords = prompt_keywords.intersection(answer_words)
        keyword_overlap = len(common_keywords) / len(prompt_keywords) if prompt_keywords else 0
        
        # Use TF-IDF for semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([prompt_clean, answer_clean])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Combined confidence score (weighted average)
        confidence_score = (keyword_overlap * 0.3 + similarity_score * 0.7) * 100
        
        return min(confidence_score, 100)
    except Exception as e:
        print(f"Error calculating confidence: {str(e)}")
        return 0


# Function to calculate match percentage with source document
def calculate_match_percentage(answer: str, source_text: str) -> float:
    """Calculate how well the answer matches the source document content."""
    try:
        # Preprocess texts
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        source_clean = re.sub(r'[^\w\s]', '', source_text.lower())
        
        # Use TF-IDF for document similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([source_clean, answer_clean])
        match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage
        match_percentage = match_score * 100
        
        return min(match_percentage, 100)
    except Exception as e:
        print(f"Error calculating match percentage: {str(e)}")
        return 0


# Function to get confidence level category
def get_confidence_category(percentage: float) -> tuple[str, str]:
    """Convert percentage to confidence level category."""
    if percentage >= 70:
        return "High", "🟢"
    elif percentage >= 30:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"


# Function to extract text from various file types
def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extracts text based on file type."""
    text = ""
    file_ext = os.path.splitext(filename)[1].lower()

    # Handle PDF files
    if file_ext == ".pdf":
        # Use pdfplumber to capture text and tables; fallback to PyPDF2 on errors
        try:
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    tables = page.extract_tables()
                    for table in tables or []:
                        # Flatten table rows into TSV-style lines for embedding
                        rows = ["\t".join(cell if cell is not None else "" for cell in row) for row in table]
                        text += "\n".join(rows) + "\n"
        except Exception as e:
            print(f"pdfplumber failed, falling back to PyPDF2: {e}")
            pdf_reader = PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text()

    # Handle Word (.docx) files
    elif file_ext == ".docx":
        doc = Document(BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text + "\n"

    # Handle text (.txt) files
    elif file_ext == ".txt":
        text = file_content.decode("utf-8")

    # Handle Excel files (.xlsx, .xls)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(BytesIO(file_content))
        text = df.to_string()

    # Handle PowerPoint files (.pptx, .ppt)
    elif file_ext in [".pptx", ".ppt"]:
        ppt = pptx.Presentation(BytesIO(file_content))
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

    return text


# Function to create vector store from extracted text
def create_vector_store(text: str) -> FAISS:
    """Create and return a FAISS vector store from text."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)


# Initialize LLM based on model selection
def initialize_llm(model_selection: str):
    """Initialize and return the appropriate LLM."""
    if model_selection == "Open AI GPT 4.1":
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0.5,
        )
    elif model_selection == "Google Gemini 2.0 Flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
        )
    else:
        raise ValueError("Invalid model selection. Choose 'Open AI GPT 4.1' or 'Google Gemini 2.0 Flash'")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SDLC Automate API",
        "version": "1.0.0",
        "description": "API for BRD to User Story, Test Case, Cucumber Script, and Selenium Script generation",
        "endpoints": {
            "upload_document": "POST /upload-document",
            "generate_user_stories": "POST /generate-user-stories",
            "convert_to_test_cases": "POST /convert-to-test-cases",
            "convert_to_cucumber": "POST /convert-to-cucumber",
            "convert_to_selenium": "POST /convert-to-selenium",
            "quality_assessment": "POST /quality-assessment"
        }
    }


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), model: str = "Open AI GPT 4.1"):
    """Upload a BRD document and initialize the QA chain."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text from file
        text = extract_text_from_file(file_content, file.filename)
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Store document
        document_id = file.filename.replace(".", "_").replace(" ", "_")
        uploaded_documents[document_id] = {
            "filename": file.filename,
            "text": text,
            "upload_time": time.time(),
            "model": model
        }
        
        # Create vector store
        vector_store = create_vector_store(text)
        vector_stores[document_id] = vector_store
        
        # Initialize LLM and QA chain
        llm = initialize_llm(model)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        qa_chains[document_id] = qa_chain
        
        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "text_length": len(text),
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading document: {str(e)}")


@app.post("/generate-user-stories")
async def generate_user_stories(request: GenerateUserStoriesRequest, document_id: str):
    """Generate user stories from uploaded BRD document."""
    try:
        if document_id not in qa_chains:
            raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")
        text = uploaded_documents[document_id]["text"]
        qa_chain = get_or_create_qa_chain(document_id, request.model)
        print(f"[MODEL] generate_user_stories doc={document_id} model={request.model}")
        
        prompt_message = """
You are an Expert Business Analyst with 20+ years of experience in requirements engineering and Agile transformation.

CRITICAL INSTRUCTION: Extract EVERY POSSIBLE user story from the BRD below. No requirement should be missed.

## DOCUMENT TO ANALYZE:
{document_text}

## EXTRACTION METHODOLOGY:

### PHASE 1: Comprehensive Requirement Mining
1. **Functional Requirements**: Extract ALL features, capabilities, and system behaviors mentioned
2. **User Interactions**: Identify EVERY user action, input, output, and workflow step
3. **Business Rules**: Capture ALL validation rules, constraints, and business logic
4. **Data Requirements**: Extract ALL data fields, entities, relationships, and transformations
5. **Integration Points**: Identify ALL system interfaces, APIs, and external dependencies
6. **Non-Functional Requirements**: Include performance, security, usability, accessibility needs
7. **Reporting & Analytics**: Extract ALL reporting, monitoring, and analytical capabilities
8. **Administrative Functions**: Capture ALL configuration, setup, and maintenance features
9. **Error Scenarios**: Include ALL error handling, validation, and recovery scenarios
10. **Compliance & Audit**: Extract ALL regulatory, compliance, and audit trail requirements

### PHASE 2: User Story Generation Rules

**MANDATORY FORMAT**: "As a [specific role], I want [specific feature/action] so that [measurable business value]"

**Story Categorization** (Generate stories for EACH category where applicable):
- **Core Features**: Primary business functions
- **CRUD Operations**: Create, Read, Update, Delete for each entity
- **Search & Filter**: All search, filter, sort capabilities
- **Validation & Rules**: Input validation, business rule enforcement
- **Workflow & Process**: Multi-step processes, approvals, state transitions
- **Notifications & Alerts**: Email, SMS, in-app notifications
- **Reports & Exports**: All reporting and data export features
- **Security & Access**: Authentication, authorization, role management
- **Integration**: External system interactions, API calls
- **Configuration**: System settings, preferences, customization
- **Audit & Compliance**: Logging, tracking, compliance features
- **Error Handling**: Error recovery, rollback, exception scenarios
- **Performance**: Load handling, response time, scalability
- **Mobile/Responsive**: Device-specific features
- **Accessibility**: Support for users with disabilities

**Acceptance Criteria Requirements**:
- Minimum 3-5 criteria per story
- Use strict Gherkin format: Given [context], When [action], Then [outcome]
- Include: Happy path, Error scenarios, Boundary conditions, Business rules
- Reference specific data fields, values, and thresholds from the BRD

**Priority Assignment Logic**:
- "Critical": Core business functions, regulatory requirements, security
- "High": Primary user workflows, key features
- "Medium": Secondary features, enhancements
- "Low": Nice-to-have, future considerations

**Story Sizing Guidance**:
- Break complex features into multiple smaller stories
- Each story should be completable in 1-3 days
- Use vertical slicing (end-to-end functionality)

### PHASE 3: Quality Checks

**Ensure EVERY story has**:
1. Unique sequential ID (US_001, US_002, ...)
2. Clear, specific, searchable title
3. Complete user story statement with role, feature, and value
4. 3-5 detailed acceptance criteria covering multiple scenarios
5. Realistic priority based on business impact
6. Relevant technical and business notes

**Extraction Completeness Verification**:
- Every paragraph in the BRD should generate at least one user story
- Every user role mentioned should appear in multiple stories
- Every data field should have CRUD stories
- Every business rule should have validation stories
- Every integration point should have connection stories

### OUTPUT REQUIREMENTS:

Return ONLY valid JSON (no markdown, no explanations):

{
  "user_stories": [
    {
      "id": "US_001",
      "title": "[Specific, searchable title from BRD content]",
      "story": "As a [specific role from BRD], I want [specific feature from BRD] so that [specific value from BRD]",
      "acceptance_criteria": [
        "Given [specific context from BRD], when [specific action], then [specific outcome with data/thresholds]",
        "Given [error scenario], when [invalid action], then [error handling from BRD]",
        "Given [edge case], when [boundary condition], then [expected behavior]",
        "Given [business rule from BRD], when [rule trigger], then [rule enforcement]",
        "Given [performance requirement], when [load condition], then [performance metric]"
      ],
      "priority": "[Critical/High/Medium/Low]",
      "story_points": [1-13],
      "category": "[category_name]",
      "notes": [
        "Affected users: [specific roles from BRD]",
        "Related module: [specific module/component from BRD]",
        "Dependencies: [specific systems/features from BRD]",
        "Data entities: [specific entities from BRD]",
        "Business rules: [specific rules from BRD]"
      ]
    }
  ]
}

IMPORTANT RULES:
1. Generate AT LEAST 25-30 stories for a typical BRD
2. Use EXACT terminology, field names, and values from the BRD
3. NO generic placeholders - use specific BRD content
4. NO trailing commas in JSON
5. EVERY requirement in the BRD must be covered
6. Include negative scenarios and edge cases
7. Ensure technical accuracy and business relevance

BEGIN EXTRACTION NOW - BE EXHAUSTIVE!
"""
        
        start_time = time.time()
        response = qa_chain.invoke({"query": prompt_message})
        processing_time = time.time() - start_time
        
        # Calculate metrics
        confidence_score = calculate_confidence_level(prompt_message, response['result'])
        match_score = calculate_match_percentage(response['result'], text)
        overall_score = (confidence_score + match_score) / 2
        
        conf_level, _ = get_confidence_category(confidence_score)
        match_level, _ = get_confidence_category(match_score)
        overall_level, _ = get_confidence_category(overall_score)
        parsed, parse_error = parse_json_output(response['result'])

        return {
            "user_stories": parsed,
            "parse_error": parse_error,
            "quality_assessment": {
                "confidence_score": round(confidence_score, 2),
                "match_score": round(match_score, 2),
                "overall_score": round(overall_score, 2),
                "confidence_level": conf_level,
                "match_level": match_level,
                "overall_level": overall_level
            },
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating user stories: {str(e)}")


@app.post("/convert-to-test-cases")
async def convert_to_test_cases(request: ConvertTestCaseRequest, document_id: str):
    """Convert user story to test cases."""
    try:
        if document_id not in qa_chains:
            raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")
        
        if not request.user_story_text.strip():
            raise HTTPException(status_code=400, detail="User story text is required")
        qa_chain = get_or_create_qa_chain(document_id, request.model)
        print(f"[MODEL] convert_to_test_cases doc={document_id} model={request.model}")
        
        test_case_prompt = """
You are a highly experienced Senior QA Engineer with over 15 years of expertise in software testing and quality assurance.

Your responsibility is to design a comprehensive test suite for the following user story:

""" + request.user_story_text + """

Provide professional, detailed, and well-structured test cases based on the following functional and non-functional requirements:

### Scope of Test Cases:
- Include **positive**, **negative**, **edge**, **database related where applicable**, and **alternative** scenarios.
- Address **input validation**, **error handling**, **security**, **usability**, **performance**, **exploratory**, **exceptional**, and **compatibility** (where applicable).
- Ensure all test cases are **independent**, **clear**, and **suitable for automation**.
- Use **realistic and meaningful** test data.

### Output Format:
Respond in **valid JSON only** using the following structure.
IMPORTANT: Do NOT include trailing commas before closing brackets or braces.

{
  "test_cases": [
    {
      "id": "TC_001",
      "title": "Generate a descriptive title",
      "preconditions": ["Precondition 1", "Precondition 2"],
      "test_data": ["data_field_1: value_1"],
      "test_steps": ["1. Step description"],
      "expected_results": ["Expected result"],
      "priority": "High",
      "attachments": []
    }
  ]
}
"""
        
        start_time = time.time()
        response = qa_chain.invoke({"query": test_case_prompt})
        processing_time = time.time() - start_time
        
        # Calculate metrics
        confidence_score = calculate_confidence_level(test_case_prompt, response['result'])
        match_score = calculate_match_percentage(response['result'], request.user_story_text)
        overall_score = (confidence_score + match_score) / 2
        
        conf_level, _ = get_confidence_category(confidence_score)
        match_level, _ = get_confidence_category(match_score)
        overall_level, _ = get_confidence_category(overall_score)
        parsed, parse_error = parse_json_output(response['result'])

        return {
            "test_cases": parsed,
            "parse_error": parse_error,
            "quality_assessment": {
                "confidence_score": round(confidence_score, 2),
                "match_score": round(match_score, 2),
                "overall_score": round(overall_score, 2),
                "confidence_level": conf_level,
                "match_level": match_level,
                "overall_level": overall_level
            },
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to test cases: {str(e)}")


@app.post("/convert-to-cucumber")
async def convert_to_cucumber(request: ConvertCucumberRequest, document_id: str):
    """Convert test case to Cucumber script."""
    try:
        if document_id not in qa_chains:
            raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")
        
        if not request.test_case_text.strip():
            raise HTTPException(status_code=400, detail="Test case text is required")
        qa_chain = get_or_create_qa_chain(document_id, request.model)
        print(f"[MODEL] convert_to_cucumber doc={document_id} model={request.model}")
        
        cucumber_prompt = """You are a BDD expert. Convert the test case into professional Cucumber Gherkin format.

TEST CASE:
""" + request.test_case_text + """

INSTRUCTIONS:
1. Start with 'Feature:' for business capability
2. Add 'Scenario:' for each test case
3. Use Given/When/Then format
4. Use 'And' for additional steps
5. Add @tags for @automated, @regression, @smoke
6. Include realistic test data
7. Cover happy path and error scenarios

FORMAT:
Feature: [Business capability]
  Scenario: [Test scenario name]
    Given [precondition]
      And [more preconditions]
    When [user action]
      And [more actions]
    Then [expected result]
      And [assertions]

Generate ONLY Gherkin code, no explanations."""
        
        start_time = time.time()
        response = qa_chain.invoke({"query": cucumber_prompt})
        processing_time = time.time() - start_time
        
        # Calculate metrics
        confidence_score = calculate_confidence_level(cucumber_prompt, response['result'])
        match_score = calculate_match_percentage(response['result'], request.test_case_text)
        overall_score = (confidence_score + match_score) / 2
        
        conf_level, _ = get_confidence_category(confidence_score)
        match_level, _ = get_confidence_category(match_score)
        overall_level, _ = get_confidence_category(overall_score)
        
        return {
            "cucumber_script": response['result'],
            "quality_assessment": {
                "confidence_score": round(confidence_score, 2),
                "match_score": round(match_score, 2),
                "overall_score": round(overall_score, 2),
                "confidence_level": conf_level,
                "match_level": match_level,
                "overall_level": overall_level
            },
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to Cucumber: {str(e)}")


@app.post("/convert-to-selenium")
async def convert_to_selenium(request: ConvertSeleniumRequest, document_id: str):
    """Convert test case to Selenium script."""
    try:
        if document_id not in qa_chains:
            raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")
        
        if not request.test_case_text.strip():
            raise HTTPException(status_code=400, detail="Test case text is required")
        qa_chain = get_or_create_qa_chain(document_id, request.model)
        print(f"[MODEL] convert_to_selenium doc={document_id} model={request.model}")
        
        selenium_prompt = """You are a Senior Test Automation Engineer specializing in Selenium and Python. Convert the following test case into a robust, production-ready Selenium WebDriver script in Python.

INSTRUCTIONS:
- Use best practices for maintainability, reliability, and readability.
- Include all necessary imports, setup, and teardown logic.
- Use explicit waits (WebDriverWait) for element interactions, not time.sleep.
- Add comments for each major step.
- Validate all expected outcomes with assert statements.
- Handle exceptions gracefully and log errors.
- Use Page Object Model if the scenario is complex.
- Ensure the script is ready to run as a standalone test.
- Use realistic locators (id, name, xpath, css selector) based on the test case.
- If data is required, use sample values from the test case.
- If login or setup is needed, include those steps.

Test Case:
""" + request.test_case_text + """

Return ONLY the complete Python code, no explanations, no markdown."""
        
        start_time = time.time()
        response = qa_chain.invoke({"query": selenium_prompt})
        processing_time = time.time() - start_time
        
        # Calculate metrics
        confidence_score = calculate_confidence_level(selenium_prompt, response['result'])
        match_score = calculate_match_percentage(response['result'], request.test_case_text)
        overall_score = (confidence_score + match_score) / 2
        
        conf_level, _ = get_confidence_category(confidence_score)
        match_level, _ = get_confidence_category(match_score)
        overall_level, _ = get_confidence_category(overall_score)
        
        return {
            "selenium_script": response['result'],
            "quality_assessment": {
                "confidence_score": round(confidence_score, 2),
                "match_score": round(match_score, 2),
                "overall_score": round(overall_score, 2),
                "confidence_level": conf_level,
                "match_level": match_level,
                "overall_level": overall_level
            },
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to Selenium: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return {
        "documents": [
            {
                "document_id": doc_id,
                "filename": info["filename"],
                "model": info["model"],
                "text_length": len(info["text"]),
                "upload_time": info["upload_time"]
            }
            for doc_id, info in uploaded_documents.items()
        ],
        "total_documents": len(uploaded_documents)
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete an uploaded document and its associated data."""
    try:
        if document_id not in uploaded_documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Clean up resources
        del uploaded_documents[document_id]
        if document_id in vector_stores:
            del vector_stores[document_id]
        if document_id in qa_chains:
            del qa_chains[document_id]
        
        return {
            "message": f"Document {document_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
