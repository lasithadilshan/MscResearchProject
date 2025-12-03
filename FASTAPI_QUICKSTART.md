# FastAPI Quick Start Guide

## 1. Installation & Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /Users/lasithadilshan/Documents/Thesis/Project/MscResearchProject
pip install -r fastapi_requirements.txt
```

### Step 2: Set Environment Variables

**On macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
export GOOGLE_API_KEY="AIzaxxxxxxxxxxxxxxx"
```

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxx"
$env:GOOGLE_API_KEY="AIzaxxxxxxxxxxxxxxx"
```

### Step 3: Start the Server
```bash
python fast_api_app.py
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

## 2. Access the API (URLs)

- **Main API**: http://localhost:8000/
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## 3. Basic Usage

### Using Swagger UI (Easiest)

1. Go to http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Fill in the required parameters
4. Click "Execute"

### Using cURL

#### Example 1: Upload a Document
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@/path/to/your/brd.pdf"
```

Response:
```json
{
  "message": "Document uploaded successfully",
  "document_id": "brd_pdf",
  "filename": "brd.pdf",
  "text_length": 5234,
  "model": "Open AI GPT 4.1"
}
```

**Copy the `document_id` for the next steps!**

#### Example 2: Generate User Stories
```bash
curl -X POST "http://localhost:8000/generate-user-stories?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{"model": "Open AI GPT 4.1"}'
```

#### Example 3: Convert User Story to Test Cases
```bash
curl -X POST "http://localhost:8000/convert-to-test-cases?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "user_story_text": "As a customer, I want to reset my password so that I can regain access to my account",
    "model": "Open AI GPT 4.1"
  }'
```

#### Example 4: Generate Cucumber Script
```bash
curl -X POST "http://localhost:8000/convert-to-cucumber?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: Password Reset\nGiven user is on login page\nWhen user clicks forgot password\nAnd user enters email\nThen password reset email is sent",
    "model": "Open AI GPT 4.1"
  }'
```

#### Example 5: Generate Selenium Script
```bash
curl -X POST "http://localhost:8000/convert-to-selenium?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: Password Reset\nGiven user is on login page\nWhen user clicks forgot password\nAnd user enters email\nThen password reset email is sent",
    "model": "Open AI GPT 4.1"
  }'
```

### Using Python Requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload document
with open("brd.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{BASE_URL}/upload-document",
        files=files,
        data={"model": "Open AI GPT 4.1"}
    )
    document_id = response.json()["document_id"]
    print(f"Document ID: {document_id}")

# Generate user stories
response = requests.post(
    f"{BASE_URL}/generate-user-stories?document_id={document_id}",
    json={"model": "Open AI GPT 4.1"}
)
user_stories = response.json()["user_stories"]
print(user_stories)

# Convert to test cases
response = requests.post(
    f"{BASE_URL}/convert-to-test-cases?document_id={document_id}",
    json={
        "user_story_text": "As a user, I want to login...",
        "model": "Open AI GPT 4.1"
    }
)
test_cases = response.json()["test_cases"]
print(test_cases)
```

## 4. Workflow Example (Step by Step)

### Step 1: Upload BRD Document
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@brd.pdf" \
  -F "model=Open AI GPT 4.1"
```

**Save the `document_id` from response** (e.g., `brd_pdf`)

### Step 2: Generate User Stories
```bash
curl -X POST "http://localhost:8000/generate-user-stories?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{"model": "Open AI GPT 4.1"}'
```

### Step 3: Copy a Generated User Story

From the output, select one user story. Example:
```
"As a user, I want to view my account profile so that I can see my personal information"
```

### Step 4: Generate Test Cases
```bash
curl -X POST "http://localhost:8000/convert-to-test-cases?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "user_story_text": "As a user, I want to view my account profile so that I can see my personal information",
    "model": "Open AI GPT 4.1"
  }'
```

### Step 5: Copy a Test Case

From the output, select one test case. Example:
```
Test Case: User Views Account Profile
- Given user is logged in
- When user clicks on profile link
- Then user sees profile page with personal information
```

### Step 6: Generate Cucumber Script
```bash
curl -X POST "http://localhost:8000/convert-to-cucumber?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: User Views Account Profile\n- Given user is logged in\n- When user clicks on profile link\n- Then user sees profile page with personal information",
    "model": "Open AI GPT 4.1"
  }'
```

### Step 7: Generate Selenium Script
```bash
curl -X POST "http://localhost:8000/convert-to-selenium?document_id=brd_pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "test_case_text": "Test Case: User Views Account Profile\n- Given user is logged in\n- When user clicks on profile link\n- Then user sees profile page with personal information",
    "model": "Open AI GPT 4.1"
  }'
```

## 5. API Response Explanation

All generation endpoints return:

```json
{
  "user_stories": "JSON string with generated content",
  "quality_assessment": {
    "confidence_score": 85.5,           // 0-100: How well prompt was addressed
    "match_score": 78.3,                // 0-100: How well aligned with source
    "overall_score": 81.9,              // Average of confidence and match
    "confidence_level": "High",         // High, Medium, or Low
    "match_level": "High",
    "overall_level": "High"
  },
  "processing_time_seconds": 12.5       // Time taken to generate
}
```

**Score Interpretation:**
- **High (≥70)**: High quality output, good confidence
- **Medium (30-69)**: Decent quality, consider reviewing
- **Low (<30)**: Low confidence, may need revision

## 6. List and Delete Documents

### List All Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

### Delete a Document
```bash
curl -X DELETE "http://localhost:8000/documents/brd_pdf"
```

## 7. Troubleshooting

### Issue: "API key not found"
```bash
# Check if environment variables are set
printenv | grep API_KEY

# If not, set them again
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Restart the app
python fast_api_app.py
```

### Issue: "Port 8000 is already in use"
```bash
# Find and kill the process
lsof -i :8000
kill -9 <PID>

# Or use a different port
uvicorn fast_api_app:app --port 8001
```

### Issue: "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r fastapi_requirements.txt

# Or use uv
uv pip install -r fastapi_requirements.txt
```

### Issue: "Slow response times"
- First request will be slower (model initialization)
- Check your internet connection
- Verify API rate limits (OpenAI/Google)

## 8. Advanced Configuration

### Change Model Temperature
Edit `fast_api_app.py` in `initialize_llm()`:

```python
def initialize_llm(model_selection: str):
    if model_selection == "Open AI GPT 4.1":
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0.3,  # Lower = more deterministic, Higher = more creative
        )
```

### Change Vector Store Settings
Edit `create_vector_store()`:

```python
def create_vector_store(text: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,  # Increase for larger chunks
        chunk_overlap=100,  # Increase for more overlap
        length_function=len
    )
```

## 9. Next Steps

1. ✅ Test with Swagger UI (easier)
2. ✅ Test with cURL (for automation)
3. ✅ Integrate into your application
4. ✅ Deploy to production

## 10. Performance Tips

- **First request**: ~15 seconds (model loading)
- **Subsequent requests**: ~8-12 seconds
- **Document size**: Works well with 5,000-50,000 character documents
- **Recommended batch size**: Process one document at a time

## Support

For detailed API documentation, see `FASTAPI_README.md`

For the original Streamlit version, see the main `README.md`
