from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import auth, documents, generation
from app.core.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SDLC Automate API",
    description="Commercial-ready API for BRD to SDLC artifacts generation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    # In production, replace with specific frontend origins
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(generation.router, prefix="/generate", tags=["Generation"])

@app.get("/")
def read_root():
    return {"message": "SDLC Automate API is running"}
