# SDLC Mate - Backend Application (FastAPI)

SDLC Mate Backend is a high-performance, asynchronous REST API built with FastAPI, LangChain, and Celery. It powers the automated transformation of Business Requirement Documents (BRDs) into structured User Stories, Test Cases, Cucumber (Gherkin) BDD features, and Selenium automation code using state-of-the-art LLMs (OpenAI GPT-4 and Google Gemini).

---

## Architecture Overview

- **REST API Framework**: FastAPI with Pydantic v2 validation and OpenAPI Swagger documentation.
- **Authentication**: JWT (JSON Web Tokens) with secure bcrypt password hashing and user role management.
- **Database & ORM**: SQLAlchemy with SQLite (default: `app.db`) or PostgreSQL support.
- **Task Queue & Workers**: Celery with Redis for asynchronous background processing of intensive document parsing and LLM generation jobs.
- **Vector Store & RAG**: ChromaDB persistent vector database with chunking via LangChain's `RecursiveCharacterTextSplitter`.
- **Document Processing**: Comprehensive multi-format text extraction supporting `.pdf` (pdfplumber with table extraction + PyPDF2 fallback), `.docx`, `.xlsx`, `.pptx`, and `.txt`.
- **Structured LLM Generation**: LangChain integration utilizing `with_structured_output()` and strict Pydantic models for predictable, schema-validated outputs.
- **Code Quality & SAST**: SonarQube LTS verified with a **PASSED** Quality Gate (0 vulnerabilities, 0 security hotspots, 0 bugs, 0 code smells, 0.0% code duplication).

---

## Project Structure

```
MscResearchProject/
├── app/
│   ├── api/
│   │   └── routers/
│   │       ├── auth.py              # User authentication (register, login, profile)
│   │       ├── documents.py         # Document upload, storage, and retrieval
│   │       └── generation.py        # Asynchronous artifact generation & job polling
│   ├── core/
│   │   ├── config.py                # Environment configuration (.env)
│   │   ├── database.py              # SQLAlchemy database engine and session
│   │   └── security.py              # JWT token handling and bcrypt hashing
│   ├── models/
│   │   └── models.py                # Database models (User, Document, Job)
│   ├── schemas/
│   │   └── schemas.py               # Pydantic request and response schemas
│   ├── services/
│   │   ├── document_service.py      # Multi-format document parser & ChromaDB indexing
│   │   └── llm_service.py           # LLM chains, Pydantic structured output, TF-IDF scoring
│   ├── worker/
│   │   ├── celery_app.py            # Celery worker configuration
│   │   └── tasks.py                 # Background generation tasks
│   └── main.py                      # FastAPI application entrypoint with CORS & routers
├── docker-compose.yml               # Redis service for Celery background tasks
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata and dependencies
├── run_backend.sh                   # Script to run FastAPI server
├── run.sh                           # Script to start Redis and FastAPI
├── sonar-project.properties         # SonarQube SAST configuration
└── .env                             # Environment variables & API keys (not in git)
```

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **uv** (recommended) or `pip`
- **Docker** (optional, for Redis background worker)

### Installation

1. Navigate to the backend directory:
   ```bash
   cd MscResearchProject
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or with uv:
   # uv pip install -r requirements.txt
   ```

4. Configure Environment Variables:
   Create a `.env` file in the root directory:
   ```env
   # Database & Security
   DATABASE_URL=sqlite:///./app.db
   SECRET_KEY=your_secure_jwt_secret_key_here
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=1440

   # Redis / Celery (Optional for background workers)
   REDIS_URL=redis://localhost:6379/0

   # AI Model API Keys
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## Running the Application

### 1. Start Redis (for Celery background workers)

```bash
docker-compose up -d
```

### 2. Start Celery Worker (Optional for background processing)

```bash
celery -A app.worker.celery_app.celery worker --loglevel=info
```

### 3. Start the FastAPI Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or using the helper script:
```bash
./run_backend.sh
```

- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## API Endpoints

### Authentication (`/api/auth`)
- `POST /api/auth/register` — Register a new user account.
- `POST /api/auth/token` — Authenticate and receive a JWT Bearer token.
- `GET /api/auth/me` — Retrieve current user profile and role.

### Document Management (`/api/documents`)
- `POST /api/documents/upload` — Upload BRD file (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`) and index into ChromaDB.
- `GET /api/documents` — List all documents belonging to the authenticated user.
- `GET /api/documents/{document_id}` — Retrieve document metadata and status.

### Artifact Generation (`/api/generate`)
- `POST /api/generate/user-stories` — Enqueue asynchronous user story generation.
- `POST /api/generate/test-cases` — Enqueue asynchronous test case derivation.
- `POST /api/generate/cucumber` — Enqueue Cucumber (Gherkin) BDD script conversion.
- `POST /api/generate/selenium` — Enqueue Python Selenium automation script generation.
- `GET /api/generate/job/{job_id}` — Poll execution status and retrieve generation results.

---

## SonarQube SAST Code Analysis

The backend repository includes complete SonarQube static analysis configuration ([`sonar-project.properties`](file:///Users/lasithadilshan/Documents/Thesis/Project/MscResearchProject/sonar-project.properties)).

### Running the Scanner

With SonarQube running locally on port 9000:

```bash
npx sonarqube-scanner \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=admin \
  -Dsonar.password=admin1
```

### Quality Gate Results

- **Vulnerabilities**: **0 (Grade A)**
- **Security Hotspots**: **0 (100% Reviewed)**
- **Bugs**: **0 (Grade A)**
- **Code Smells**: **0 (Grade A)**
- **Duplicated Lines**: **0.0%**
- **Quality Gate Status**: **PASSED (OK)**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

