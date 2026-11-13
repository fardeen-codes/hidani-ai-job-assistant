"""
Main FastAPI Application for Hidani AI Job Assistant.
Provides RESTful APIs for resume upload, multi-agent conversational chat,
and session state inspection.
"""

import csv
import io
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.agents.matcher_agent import analyze_skill_gap
from app.agents.orchestrator import (
    clear_session_state,
    get_session_state,
    run_orchestrator,
)
from app.agents.parser_agent import parse_resume_with_llm
from app.config import (
    APP_TITLE,
    CORS_ORIGINS,
    ENVIRONMENT,
    is_openai_configured,
)
from app.schemas import (
    ApplicationCreateRequest,
    ApplicationNotesUpdateRequest,
    ApplicationStatsResponse,
    ApplicationStatusUpdateRequest,
    ChatRequest,
    ChatResponse,
    ChatStateResponse,
    JobApplication,
    JobBookmark,
    JobStatsResponse,
    SearchHistoryEntry,
    SkillGapAnalysisResponse,
    UploadResumeResponse,
)
from app.services.db_service import (
    apply_to_job,
    clear_search_history,
    get_application_stats,
    get_application_status,
    get_bookmarked_jobs,
    get_job_by_id,
    get_job_stats,
    get_recent_searches,
    load_applications,
    load_jobs,
    record_search_query,
    remove_job_bookmark,
    save_job_bookmark,
    search_jobs,
    update_application_notes,
    update_application_status,
    withdraw_application,
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


@app.post("/api/chat/reset", summary="Reset session conversation history", tags=["Chat"])
@app.post("/api/chat/reset/{session_id}", include_in_schema=False)
async def reset_chat_session(session_id: str = "default-session"):
    """Reset conversational history and recommendation context for a session."""
    cleared = clear_session_state(session_id)
    return {
        "status": "success",
        "message": f"Session '{session_id}' conversation history has been reset.",
        "session_id": session_id,
        "cleared": cleared
    }


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
    location: Optional[str] = Query(None, description="Optional location filter (e.g. Remote, Berlin)"),
    experience_level: Optional[str] = Query(None, description="Optional experience tier filter (Entry-Level, Mid-Level, Senior)"),
    company: Optional[str] = Query(None, description="Optional company name filter"),
    sort_by: Optional[str] = Query(None, description="Optional field to sort by (title, company, experience, location)"),
    order: Optional[str] = Query("asc", description="Sort direction (asc, desc)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Retrieve paginated jobs from the local 1,600+ job database with multi-field filtering and sorting."""
    clean_query = (query or "").strip()
    clean_loc = (location or "").strip()
    clean_exp = (experience_level or "").strip()
    clean_company = (company or "").strip()
    clean_sort = (sort_by or "").strip().lower()
    is_desc = (order or "asc").strip().lower() == "desc"

    if clean_query or clean_loc or clean_exp or clean_company or clean_sort:
        all_jobs = search_jobs(
            query=clean_query or None,
            location=clean_loc or None,
            experience_level=clean_exp or None,
            company=clean_company or None,
            sort_by=clean_sort or None,
            order="desc" if is_desc else "asc",
            limit=None
        )
        if clean_query or clean_loc or clean_exp:
            record_search_query(
                query=clean_query or None,
                location=clean_loc or None,
                experience=clean_exp or None,
                results_count=len(all_jobs)
            )
    else:
        all_jobs = load_jobs()

    paged = all_jobs[offset : offset + limit]
    return {
        "total": len(all_jobs),
        "limit": limit,
        "offset": offset,
        "jobs": paged
    }


@app.get("/api/searches/recent", response_model=List[SearchHistoryEntry], summary="Get recent job searches", tags=["Searches"])
async def get_recent_searches_endpoint(limit: int = Query(10, ge=1, le=50)):
    """Retrieve candidate's recent search queries and active filter history."""
    return get_recent_searches(limit=limit)


@app.delete("/api/searches/recent", summary="Clear search history", tags=["Searches"])
async def clear_recent_searches_endpoint():
    """Clear candidate's recorded search history."""
    clear_search_history()
    return {"status": "success", "message": "Search history cleared."}


@app.get("/api/jobs/stats", response_model=JobStatsResponse, summary="Get job database statistics and market insights", tags=["Jobs"])
async def get_jobs_statistics():
    """Retrieve aggregated job market statistics, top demanded skills, and remote job distribution."""
    return get_job_stats()


@app.get("/api/jobs/{job_id}", summary="Get job details by ID", tags=["Jobs"])
async def get_job_details_by_id(job_id: str):
    """Retrieve full details for a single job posting by ID."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{clean_id}' was not found in the database."
        )
    return job


@app.get(
    "/api/jobs/{job_id}/match",
    response_model=SkillGapAnalysisResponse,
    summary="Evaluate candidate skill gap for a specific job",
    tags=["Jobs"]
)
async def evaluate_job_skill_gap(
    job_id: str,
    skills: Optional[str] = Query(None, description="Comma-separated candidate skills (e.g. 'Python,FastAPI')"),
    session_id: Optional[str] = Query(None, description="Optional session ID to pull candidate profile skills")
):
    """Assess skill readiness, matching skills, missing skills, and career recommendations for a job."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")

    candidate_skills: List[str] = []
    if skills:
        candidate_skills = [s.strip() for s in skills.split(",") if s.strip()]
    elif session_id:
        sess_state = get_session_state(session_id)
        if sess_state.get("profile"):
            candidate_skills = sess_state["profile"].get("skills", [])

    analysis = analyze_skill_gap(clean_id, candidate_skills=candidate_skills)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{clean_id}' was not found in the database."
        )
    return analysis


@app.get("/api/applications", summary="List all submitted applications", tags=["Applications"])
async def get_applications_list(
    status: Optional[str] = Query(None, description="Optional status filter (e.g. Applied, Under Review)"),
    search: Optional[str] = Query(None, description="Optional search keyword matching job title or company")
):
    """List all recorded job applications from applications.json with optional status and search filters."""
    apps = load_applications()
    if status:
        stat_clean = status.strip().lower()
        apps = [a for a in apps if (a.get("status") or "").strip().lower() == stat_clean]
    if search:
        search_clean = search.strip().lower()
        apps = [
            a for a in apps
            if search_clean in (a.get("job_title") or "").lower()
            or search_clean in (a.get("company") or "").lower()
            or search_clean in (a.get("job_id") or "").lower()
        ]
    return apps


@app.post(
    "/api/applications",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a job application",
    tags=["Applications"]
)
async def submit_application(payload: ApplicationCreateRequest):
    """
    Submit a direct job application with deduplication and validation.
    Returns 201 Created on success, 409 Conflict if already applied, or 404 if job ID is not found.
    """
    clean_id = (payload.job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )

    success, message, app_data = apply_to_job(clean_id, notes=payload.notes)
    if not success:
        if app_data is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )

    return {
        "status": "success",
        "message": message,
        "application": app_data
    }


@app.get(
    "/api/applications/stats",
    response_model=ApplicationStatsResponse,
    summary="Get application statistics and status breakdown",
    tags=["Applications"]
)
async def get_applications_statistics():
    """Retrieve aggregated statistics across submitted applications including status breakdown and top companies."""
    return get_application_stats()


@app.get(
    "/api/applications/export",
    summary="Export applications as JSON or CSV",
    tags=["Applications"]
)
async def export_applications(
    format: str = Query("json", description="Export format: json or csv")
):
    """Export all submitted job applications in JSON or CSV format for spreadsheet tracking."""
    fmt = (format or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format. Supported export formats are 'json' and 'csv'."
        )

    apps = load_applications()
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Job ID", "Job Title", "Company", "Location", "Status", "Applied Date", "Notes"])
        for app in apps:
            writer.writerow([
                app.get("job_id", ""),
                app.get("job_title", ""),
                app.get("company", ""),
                app.get("location", ""),
                app.get("status", "Applied"),
                app.get("applied_date", ""),
                app.get("notes") or ""
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=job_applications.csv"}
        )

    return JSONResponse(
        content=apps,
        headers={"Content-Disposition": "attachment; filename=job_applications.json"}
    )


@app.get(
    "/api/applications/{job_id}",
    summary="Get application details by job ID",
    tags=["Applications"]
)
async def get_application_by_id(job_id: str):
    """Retrieve application status and details for a specific job ID."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    apps = get_application_status(clean_id)
    if not apps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No application found for Job ID '{clean_id}'."
        )
    return apps[0]


@app.patch(
    "/api/applications/{job_id}",
    summary="Update application status",
    tags=["Applications"]
)
async def update_application_status_endpoint(job_id: str, payload: ApplicationStatusUpdateRequest):
    """Update the tracking status of an existing job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = update_application_status(clean_id, payload.status)
    if not success:
        if "Invalid status" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.patch(
    "/api/applications/{job_id}/notes",
    summary="Update candidate notes for an application",
    tags=["Applications"]
)
async def update_application_notes_endpoint(job_id: str, payload: ApplicationNotesUpdateRequest):
    """Update personal applicant notes or interview comments for a submitted job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = update_application_notes(clean_id, payload.notes)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.delete("/api/applications/{job_id}", summary="Withdraw an active job application", tags=["Applications"])
async def withdraw_application_endpoint(job_id: str):
    """Withdraw/cancel a previously submitted job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message = withdraw_application(clean_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    return {
        "status": "success",
        "message": message,
        "job_id": clean_id
    }


# Bookmarks Endpoints
@app.get("/api/bookmarks", response_model=list, summary="List all saved job bookmarks", tags=["Bookmarks"])
async def get_bookmarks_list():
    """Retrieve all bookmarked / saved jobs."""
    raw_bms = get_bookmarked_jobs()
    return [JobBookmark(**b).model_dump() for b in raw_bms]


@app.post("/api/bookmarks/{job_id}", status_code=status.HTTP_201_CREATED, summary="Bookmark a job posting", tags=["Bookmarks"])
async def bookmark_job_endpoint(job_id: str):
    """Save a job posting to bookmarks for later review."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )

    success, message, bm_data = save_job_bookmark(clean_id)
    if not success:
        if bm_data is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )

    return {
        "status": "success",
        "message": message,
        "bookmark": bm_data
    }


@app.delete("/api/bookmarks/{job_id}", summary="Remove a job from bookmarks", tags=["Bookmarks"])
async def remove_bookmark_endpoint(job_id: str):
    """Remove a bookmarked job."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )

    success, message = remove_job_bookmark(clean_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )

    return {
        "status": "success",
        "message": message,
        "job_id": clean_id
    }


# Static Frontend Delivery
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the conversational UI single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index_file)
    return {"message": "Hidani AI Job Assistant API is running. Visit /docs for Swagger UI."}


@app.get("/app.js", include_in_schema=False)
async def serve_app_js():
    """Serve the frontend client script."""
    js_file = FRONTEND_DIR / "app.js"
    if js_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Frontend script not found.")


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
