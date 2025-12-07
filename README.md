# BRD to User Story, Test Case, Cucumber Script, and Selenium Script Generator

## Overview

This application is a FastAPI backend with Streamlit frontend designed to assist users in generating user stories, test cases, Cucumber scripts, and Selenium scripts from uploaded Business Requirement Document (BRD) files. It uses the LangChain framework with multiple AI models (OpenAI GPT-4 and Google Gemini) to automate the extraction and generation of actionable development and testing artifacts.

---

## Architecture

- **Backend**: FastAPI REST API serving all AI processing and document handling
- **Frontend**: Streamlit web interface for user interactions
- **AI Models**: Support for both OpenAI GPT-4.1 and Google Gemini 2.0 Flash
- **Vector Store**: FAISS with HuggingFace embeddings for document retrieval
- **Document Processing**: Enhanced PDF extraction with pdfplumber (including table support)

---

## Features

### 1. **Multi-Model AI Support**

- Switch between OpenAI GPT-4.1 and Google Gemini 2.0 Flash
- Model-specific QA chain management for optimal performance

### 2. **File Upload and Text Extraction**

- Supports file formats: `.pdf`, `.docx`, `.txt`, `.xlsx`, and `.pptx`
- Advanced PDF extraction with table support using pdfplumber
- Fallback to PyPDF2 for robust PDF handling

### 3. **User Story Generation**

- Converts BRD content into comprehensive user stories
- Generates 25-30+ stories for typical BRDs
- Includes acceptance criteria in Gherkin format
- Quality assessment with confidence and match scoring

### 4. **User Story to Test Case Conversion**

- Generates comprehensive test cases covering positive, negative, and edge scenarios
- Includes preconditions, test data, steps, and expected results
- Quality metrics for each generation

### 5. **Test Case to Cucumber Script Conversion**

- Converts test cases into Gherkin syntax scripts
- Includes tags for automation, regression, and smoke testing

### 6. **Test Case to Selenium Script Conversion**

- Generates production-ready Python Selenium scripts
- Includes explicit waits, error handling, and best practices

---

## Installation

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended for faster dependency management)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/lasithadilshan/MscResearchProject.git
   cd MscResearchProject
   ```

2. Install dependencies using uv:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Set up your API keys in `.streamlit/secrets.toml`:
   - Create the file `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

4. Run the FastAPI backend:
   ```bash
   .uv/bin/python3 -m uvicorn fast_api_app:app --reload
   ```

5. In a new terminal, run the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

6. Access the application:
   - Frontend: `http://localhost:8501`
   - Backend API: `http://127.0.0.1:8000`
   - API Documentation: `http://127.0.0.1:8000/docs`

---

## Usage

### Select AI Model

1. In the sidebar, select your preferred AI model:
   - **Open AI GPT 4.1** - More precise and consistent
   - **Google Gemini 2.0 Flash** - Faster and more creative

### Upload a BRD Document

1. Upload a file via the sidebar
2. Supported formats: `.pdf`, `.docx`, `.txt`, `.xlsx`, `.pptx`
3. The document is processed and ready for all conversions

### Generate User Stories

1. Navigate to the **User Story Generation** tab
2. Click **Generate User Stories**
3. View generated stories with:
   - Story ID, title, description, and acceptance criteria
   - Priority and story points
   - Quality assessment metrics (confidence and match scores)
   - Processing time

### Convert User Stories to Test Cases

1. Navigate to the **User Story to Test Case** tab
2. Enter or paste the user story text
3. Click **Generate Test Cases**
4. View comprehensive test cases with preconditions, steps, and expected results

### Convert Test Cases to Cucumber Scripts

1. Navigate to the **Test Case to Cucumber Script** tab
2. Enter the test case text
3. Click **Generate Cucumber Script**
4. Get Gherkin-formatted scenarios ready for automation

### Convert Test Cases to Selenium Scripts

1. Navigate to the **Test Case to Selenium Script** tab
2. Enter the test case text
3. Click **Generate Selenium Script**
4. Get production-ready Python Selenium code

---

## Key Dependencies

| Package                   | Version  | Purpose                              |
| ------------------------- | -------- | ------------------------------------ |
| fastapi                   | 0.104.1  | Backend REST API framework           |
| uvicorn[standard]         | 0.24.0   | ASGI server for FastAPI              |
| streamlit                 | 1.38.0   | Frontend web interface               |
| langchain                 | 0.3.0    | LLM framework and chains             |
| langchain-openai          | 0.3.12   | OpenAI integration                   |
| langchain-google-genai    | 2.1.2    | Google Gemini integration            |
| langchain-huggingface     | 0.1.2    | HuggingFace embeddings               |
| faiss-cpu                 | 1.8.0    | Vector store for document retrieval  |
| pdfplumber                | 0.7.0    | Advanced PDF extraction with tables  |
| PyPDF2                    | 3.0.1    | PDF extraction fallback              |
| python-docx               | 0.8.11   | Word document processing             |
| python-pptx               | 0.6.23   | PowerPoint processing                |
| pandas                    | 2.1.3    | Data handling                        |
| scikit-learn              | 1.6.1    | Quality scoring algorithms           |
| toml                      | 0.10.2   | Secrets file parsing                 |

---

## Project Structure

```
.
├── fast_api_app.py           # FastAPI backend with all AI logic
├── app.py                    # Streamlit frontend interface
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys configuration
├── images/
│   └── favicon.png           # Application icon
├── README.md                 # Project documentation
└── .uv/                      # UV Python environment
```

---

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /upload-document` - Upload and process BRD documents
- `POST /generate-user-stories` - Generate user stories from BRD
- `POST /convert-to-test-cases` - Convert user stories to test cases
- `POST /convert-to-cucumber` - Convert test cases to Cucumber scripts
- `POST /convert-to-selenium` - Convert test cases to Selenium scripts
- `GET /documents` - List all uploaded documents
- `DELETE /documents/{document_id}` - Delete a document

Full API documentation available at `http://127.0.0.1:8000/docs` when backend is running.

---

## Performance Optimization

- **Session State Caching**: Streamlit session state prevents redundant document uploads
- **Model-Specific QA Chains**: Cached per document and model selection
- **Vector Store**: FAISS enables fast similarity search
- **HuggingFace Embeddings**: Efficient local embeddings without API calls

---

## Configuration

### API Keys Setup

Both API keys must be configured in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-proj-..."
GOOGLE_API_KEY = "AIza..."
```

The backend automatically loads these keys on startup.

### Text Splitting Parameters

In `fast_api_app.py`, you can adjust:
- `chunk_size`: Default 800 (controls document chunk size)
- `chunk_overlap`: Default 50 (overlap between chunks)

---

## Troubleshooting

### Common Issues

1. **Port 8000 already in use**:
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Missing or invalid API Keys**:
   - Verify `.streamlit/secrets.toml` exists and has valid keys
   - Check backend startup logs for key validation errors

3. **Import errors**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Frontend can't connect to backend**:
   - Ensure FastAPI is running on `http://127.0.0.1:8000`
   - Check terminal for backend errors

5. **PDF extraction issues**:
   - The app uses pdfplumber first, then falls back to PyPDF2
   - Check if PDF is password-protected or corrupted

6. **Model switching not working**:
   - The app rebuilds QA chains when model changes
   - Wait for processing to complete before switching again

---

## Quality Assessment

Each generation includes quality metrics:

- **Confidence Score**: How well the output addresses the input (0-100%)
- **Match Score**: Similarity to source document content (0-100%)
- **Overall Score**: Combined quality metric
- **Processing Time**: Time taken for generation

Scores are categorized as:
- 🟢 High (70%+)
- 🟡 Medium (30-70%)
- 🔴 Low (<30%)

---

## Future Enhancements

- Database persistence for documents and results
- User authentication and multi-tenancy
- Batch processing for multiple documents
- Export to various formats (Word, Excel, JSON)
- Integration with JIRA/Azure DevOps
- Support for additional LLM models (Claude, Llama)
- Real-time collaboration features
- Custom prompt templates

---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contributors

- Lasitha Thilakarathna - Developer
