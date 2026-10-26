"""
Main FastAPI Application for Hidani AI Job Assistant.
Provides RESTful APIs for resume upload, multi-agent conversational chat,
and session state inspection.
"""

import logging
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.orchestrator import get_session_state, run_orchestrator
from app.agents.parser_agent import parse_resume_with_llm
from app.config import (
    APP_TITLE,
    CORS_ORIGINS,
    ENVIRONMENT,
    is_openai_configured,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStateResponse,
    JobApplication,
    UploadResumeResponse,
)
from app.services.db_service import (
    load_applications,
    load_jobs,
    search_jobs,
)
from app.services.document_parser import (
    DocumentParsingError,
    UnsupportedFileTypeError,
    parse_resume_document,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("App")

app = FastAPI(
    title="Hidani AI Job Assistant API",
    description="Multi-agent AI Job Recommendation and Application Engine for Hidani Tech Assessment",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """System diagnostic health check endpoint."""
    jobs_count = len(load_jobs())
    apps_count = len(load_applications())
    return {
        "status": "online",
        "environment": ENVIRONMENT,
        "openai_configured": is_openai_configured(),
        "database": {
            "total_jobs": jobs_count,
            "total_applications": apps_count
        }
    }


@app.post(
    "/api/upload-resume",
    response_model=UploadResumeResponse,
    summary="Upload and parse resume (PDF or DOCX)",
    tags=["Resume"]
)
@app.post("/api/upload", response_model=UploadResumeResponse, include_in_schema=False)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume file (.pdf or .docx).
    The document parser extracts raw text, and Sub-Agent 1 (Parser Agent)
    extracts structured candidate profile data using LLM with heuristic fallback.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was selected.")

    try:
        content_bytes = await file.read()
        if len(content_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Document parsing
        extracted_text = parse_resume_document(file.filename, content_bytes)
        
        # Sub-Agent 1: Parse candidate profile with LLM
        candidate_profile = parse_resume_with_llm(extracted_text)

        logger.info(f"Resume parsed successfully for candidate '{candidate_profile.name}' ({len(candidate_profile.skills)} skills found)")

        return UploadResumeResponse(
            status="success",
            message=f"Resume parsed successfully for {candidate_profile.name}.",
            profile=candidate_profile
        )

    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    except DocumentParsingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error handling resume upload: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(exc)}")


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Interact with the Multi-Agent Orchestrator",
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Send a message to the Multi-Agent Chat Assistant.
    The Orchestrator agent evaluates user intent using OpenAI Tool Calling
    and delegates tasks to the Matcher Agent or Applier Agent.
    """
    clean_message = (request.message or "").strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        reply, actions = run_orchestrator(
            message=clean_message,
            session_id=request.session_id,
            profile=request.profile
        )

        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            actions=actions,
            profile=request.profile
        )

    except Exception as exc:
        logger.error(f"Error in chat processing: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat assistant error: {str(exc)}")


@app.get(
    "/api/state/{session_id}",
    response_model=ChatStateResponse,
    summary="Get chat session state, applications, and profile",
    tags=["State"]
)
@app.get("/api/state", response_model=ChatStateResponse, include_in_schema=False)
async def get_state(session_id: str = "default-session"):
    """
    Retrieve operational and diagnostic state for a chat session:
    parsed profile, application count, and message activity.
    """
    session_data = get_session_state(session_id)
    raw_apps = load_applications()
    apps = [JobApplication(**a) for a in raw_apps]
    total_jobs = len(load_jobs())

    return ChatStateResponse(
        session_id=session_id,
        profile=session_data.get("profile"),
        total_jobs_available=total_jobs,
        applications=apps,
        applied_count=len(apps),
        message_count=session_data.get("message_count", 0),
        openai_configured=is_openai_configured()
    )


@app.get("/api/jobs", summary="Browse available jobs", tags=["Jobs"])
async def get_jobs_list(
    query: Optional[str] = Query(None, description="Optional search keyword"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Retrieve paginated jobs from the local 1,600+ job database."""
    all_jobs = search_jobs(query=query, limit=limit + offset) if query else load_jobs()
    paged = all_jobs[offset : offset + limit]
    return {
        "total": len(all_jobs),
        "limit": limit,
        "offset": offset,
        "jobs": paged
    }


@app.get("/api/applications", summary="List all submitted applications", tags=["Applications"])
async def get_applications_list():
    """List all recorded job applications from applications.json."""
    return load_applications()


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
