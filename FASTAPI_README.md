# SDLC Automate FastAPI Application

A FastAPI-based REST API for BRD (Business Requirements Document) to User Story, Test Case, Cucumber Script, and Selenium Script generation using AI models.

## Features

- **Document Upload**: Upload BRD documents (PDF, DOCX, TXT, XLSX, PPTX)
- **User Story Generation**: Automatically generate comprehensive user stories from BRD
- **Test Case Generation**: Convert user stories to detailed test cases
- **Cucumber Script Generation**: Generate Gherkin-format Cucumber test scripts
- **Selenium Script Generation**: Create production-ready Selenium WebDriver scripts in Python
- **Quality Assessment**: Automatic confidence and match scoring for generated content
- **Multiple AI Models**: Support for OpenAI GPT-4.1 and Google Gemini 2.0 Flash

## Prerequisites

- Python 3.11+
- API Keys for:
  - OpenAI (GPT-4.1)
  - Google Gemini (2.0 Flash)

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

Required packages:
- fastapi
- uvicorn
- langchain
- langchain-openai
- langchain-google-genai
- langchain-huggingface
- pydantic
- python-multipart
- pydantic-settings

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-gemini-api-key"
```

## Running the Application

### Using uvicorn directly

```bash
uvicorn fast_api_app:app --reload --host 0.0.0.0 --port 8000
```

### Using Python

```bash
python fast_api_app.py
```

## API Documentation

Once the app is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Upload Document
```
POST /upload-document
```

**Parameters:**
- `file`: Document file (PDF, DOCX, TXT, XLSX, PPTX)
- `model`: AI model to use (default: "Open AI GPT 4.1")

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "document_id": "file_name_pdf",
  "filename": "file_name.pdf",
  "text_length": 5000,
  "model": "Open AI GPT 4.1"
}
```

### 3. Generate User Stories
```
POST /generate-user-stories?document_id=<document_id>
```

**Request Body:**
```json
{
  "model": "Open AI GPT 4.1"
}
```

**Response:**
```json
{
  "user_stories": "[JSON formatted user stories]",
  "quality_assessment": {
    "confidence_score": 85.5,
    "match_score": 78.3,
    "overall_score": 81.9,
    "confidence_level": "High",
    "match_level": "High",
    "overall_level": "High"
  },
  "processing_time_seconds": 12.5
}
```

### 4. Convert to Test Cases
```
POST /convert-to-test-cases?document_id=<document_id>
```

**Request Body:**
```json
{
  "user_story_text": "As a user, I want to login...",
  "model": "Open AI GPT 4.1"
}
```

**Response:**
```json
{
  "test_cases": "[JSON formatted test cases]",
  "quality_assessment": {
    "confidence_score": 87.2,
    "match_score": 82.1,
    "overall_score": 84.6,
    "confidence_level": "High",
    "match_level": "High",
    "overall_level": "High"
  },
  "processing_time_seconds": 10.3
}
```

### 5. Convert to Cucumber Script
```
POST /convert-to-cucumber?document_id=<document_id>
```

**Request Body:**
```json
{
  "test_case_text": "Test Case: User Login\n- Given the user is on the login page...",
  "model": "Open AI GPT 4.1"
}
```

**Response:**
```json
{
  "cucumber_script": "Feature: User Authentication\n  Scenario: Valid Login\n    Given the user is on the login page...",
  "quality_assessment": {
    "confidence_score": 89.5,
    "match_score": 85.2,
    "overall_score": 87.4,
    "confidence_level": "High",
    "match_level": "High",
    "overall_level": "High"
  },
  "processing_time_seconds": 8.5
}
```

### 6. Convert to Selenium Script
```
POST /convert-to-selenium?document_id=<document_id>
```

**Request Body:**
```json
{
  "test_case_text": "Test Case: User Login\n- Given the user is on the login page...",
  "model": "Open AI GPT 4.1"
}
```

**Response:**
```json
{
  "selenium_script": "from selenium import webdriver\nfrom selenium.webdriver.common.by import By...",
  "quality_assessment": {
    "confidence_score": 88.3,
    "match_score": 84.1,
    "overall_score": 86.2,
    "confidence_level": "High",
    "match_level": "High",
    "overall_level": "High"
  },
  "processing_time_seconds": 9.2
}
```

### 7. List Documents
```
GET /documents
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "brd_pdf",
      "filename": "brd.pdf",
      "model": "Open AI GPT 4.1",
      "text_length": 5000,
      "upload_time": 1701660000.0
    }
  ],
  "total_documents": 1
}
```

### 8. Delete Document
```
DELETE /documents/{document_id}
```

**Response:**
```json
{
  "message": "Document document_id deleted successfully"
}
```

## Workflow Example

### 1. Upload Document
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@brd.pdf" \
  -F "model=Open AI GPT 4.1"
```

Response:
```json
{
  "document_id": "brd_pdf",
  "filename": "brd.pdf",
  "text_length": 5000,
  "message": "Document uploaded successfully"
}
```

### 2. Generate User Stories
```bash
curl -X POST "http://localhost:8000/generate-user-stories?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{"model": "Open AI GPT 4.1"}'
```

### 3. Create Test Cases from User Story
```bash
curl -X POST "http://localhost:8000/convert-to-test-cases?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "user_story_text": "As a user, I want to reset my password so that I can regain access to my account",
    "model": "Open AI GPT 4.1"
  }'
```

### 4. Generate Cucumber Script
```bash
curl -X POST "http://localhost:8000/convert-to-cucumber?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: Password Reset\nGiven user clicks forgot password\nWhen user enters email\nThen user receives reset link",
    "model": "Open AI GPT 4.1"
  }'
```

### 5. Generate Selenium Script
```bash
curl -X POST "http://localhost:8000/convert-to-selenium?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: Password Reset\nGiven user clicks forgot password\nWhen user enters email\nThen user receives reset link",
    "model": "Open AI GPT 4.1"
  }'
```

## Quality Assessment Metrics

All generation endpoints return quality metrics:

- **Confidence Score**: How well the AI response addresses the prompt (0-100)
- **Match Score**: How well the response aligns with source content (0-100)
- **Overall Score**: Average of confidence and match scores
- **Confidence Level**: High (≥70), Medium (30-69), Low (<30)

## Error Handling

Common HTTP status codes:
- `200`: Successful request
- `400`: Bad request (missing/invalid parameters)
- `404`: Document not found
- `500`: Server error (internal processing error)

Error response example:
```json
{
  "detail": "Error message describing the issue"
}
```

## Configuration

### AI Model Selection

Two models are currently supported:

1. **Open AI GPT 4.1**
   - Temperature: 0.5
   - Better for precise, structured output

2. **Google Gemini 2.0 Flash**
   - Temperature: 0.7
   - Better for creative, detailed responses

Change the temperature in `fast_api_app.py` `initialize_llm()` function if needed.

## Advanced Features

### Vector Store & Embeddings

- Uses `HuggingFaceEmbeddings` for document embeddings
- FAISS for vector similarity search
- `RecursiveCharacterTextSplitter` for intelligent chunking (800 chars, 50 overlap)

### Document Persistence

Currently, documents are stored in memory. For production:
1. Use a database (PostgreSQL, MongoDB)
2. Implement persistent vector store (Pinecone, Weaviate)
3. Add authentication and authorization

## Deployment

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fast_api_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t sdlc-automate-api .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  sdlc-automate-api
```

### Using Cloud Platforms

**AWS EC2/ECS**:
```bash
uvicorn fast_api_app:app --host 0.0.0.0 --port 8000
```

**Google Cloud Run**:
```bash
gcloud run deploy sdlc-automate-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="key",GOOGLE_API_KEY="key"
```

**Azure App Service**:
```bash
az webapp up --resource-group mygroup --name sdlc-automate-api
```

## Rate Limiting (Optional)

Add rate limiting for production:

```bash
pip install slowapi
```

## Performance Considerations

- Document chunking: 800 chars with 50 char overlap
- Vector similarity search: Top 3 results
- Processing time: 8-15 seconds per request
- Memory: ~500MB for typical BRD documents

## Troubleshooting

### "API key not found"
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Set them if missing
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### "Document not found"
Ensure you use the correct `document_id` returned from the upload endpoint.

### "Port 8000 already in use"
```bash
# Use a different port
uvicorn fast_api_app:app --port 8001
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - See LICENSE file

## Support

For issues and feature requests, please create a GitHub issue.

## Version History

- **v1.0.0** (Current): Initial FastAPI implementation with support for user stories, test cases, Cucumber, and Selenium script generation
