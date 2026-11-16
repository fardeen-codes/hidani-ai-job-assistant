"""
Main FastAPI Application for Hidani AI Job Assistant.
Provides RESTful APIs for resume upload, multi-agent conversational chat,
and session state inspection.
"""

import csv
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.agents.matcher_agent import (
    analyze_skill_gap,
    batch_score_jobs,
    calculate_salary_expectation_fit,
    calculate_weighted_skill_match,
    estimate_interview_difficulty,
    evaluate_skill_proficiency_gaps,
    generate_learning_roadmap,
    normalize_job_title_semantics,
    predict_company_interview_questions,
    score_career_transition,
)
from app.agents.orchestrator import (
    clear_session_state,
    coach_star_behavioral_answer,
    evaluate_intent_confidence_and_clarification,
    export_session_transcript,
    generate_cold_outreach_message,
    generate_post_interview_thank_you_email,
    generate_recruiter_followup_cadence,
    generate_salary_negotiation_script,
    get_session_metrics,
    get_session_preferences,
    get_session_state,
    run_orchestrator,
    update_session_preferences,
)
from app.agents.parser_agent import parse_resume_with_llm
from app.config import (
    APP_TITLE,
    CORS_ORIGINS,
    ENVIRONMENT,
    is_openai_configured,
)
from app.schemas import (
    AdvancedJobFilterRequest,
    AdvancedJobFilterResponse,
    ApplicationArchiveUpdateRequest,
    ApplicationAttachmentCreateRequest,
    ApplicationChannelAnalyticsResponse,
    ApplicationChannelMetric,
    ApplicationCreateRequest,
    ApplicationDeadlinesUpdateRequest,
    ATSScoreRequest,
    ATSScoreResponse,
    ApplicationFunnelResponse,
    ApplicationInterviewCreateRequest,
    InterviewStageUpdateRequest,
    InterviewQuestionCreateRequest,
    InterviewQuestionEntry,
    InterviewQuestionsResponse,
    ApplicationNotesUpdateRequest,
    ApplicationOutcomeUpdateRequest,
    ApplicationPortfolioItem,
    ApplicationPortfolioResponse,
    ApplicationPriorityUpdateRequest,
    ApplicationRejectionAnalyticsResponse,
    ApplicationRollbackResponse,
    ApplicationReminderUpdateRequest,
    ApplicationSourceUpdateRequest,
    ApplicationStageDurationMetric,
    ApplicationStageDurationResponse,
    ApplicationStatsResponse,
    ApplicationStatusUpdateRequest,
    ApplicationTagsUpdateRequest,
    ApplicationTimelineResponse,
    ApplicationVelocityResponse,
    AutoFollowupRequest,
    AutoFollowupResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    BookmarkCategoriesResponse,
    BookmarkUpdateRequest,
    BulkApplicationStatusUpdateRequest,
    BulkApplicationStatusUpdateResponse,
    BulkBookmarkDeleteRequest,
    BulkBookmarkDeleteResponse,
    CandidateLanguageItem,
    CandidateLanguagesResponse,
    CandidateLanguagesUpdateRequest,
    CandidatePortfolioLinkItem,
    CandidatePortfolioLinksResponse,
    PortfolioLinkCreateRequest,
    CareerTransitionRequest,
    CareerTransitionResponse,
    CertificationAddRequest,
    CertificationItem,
    CertificationsResponse,
    ChatRequest,
    ChatResponse,
    ChatStateResponse,
    ColdOutreachMessageRequest,
    ColdOutreachMessageResponse,
    CollectionChecksumItem,
    CompanyDirectoryResponse,
    CompanyInterviewPredictionResponse,
    CompanyHiringSurgeItem,
    CompanyHiringSurgeResponse,
    CompanyProfile,
    DatabaseAuditLogResponse,
    DatabaseBackupSnapshot,
    DatabaseBenchmarkResponse,
    DatabaseChecksumResponse,
    DatabaseCompactionResponse,
    DatabaseHealthResponse,
    DatabaseIntegrityIssue,
    DatabaseIntegrityReportResponse,
    DatabaseRepairResponse,
    DatabaseRestoreResponse,
    DatabaseStorageMetricsResponse,
    DatabaseVacuumCollectionResult,
    DatabaseVacuumResponse,
    DisambiguationResponse,
    ExperienceTierBreakdownResponse,
    ExperienceTierMetric,
    FollowupTemplateResponse,
    FollowupCadenceStep,
    FollowupCadenceRecommendationResponse,
    InterviewDifficultyResponse,
    InterviewDifficultyFeedbackRequest,
    InterviewDifficultyFeedbackResponse,
    InterviewQuestionPredictionItem,
    JobApplication,
    LearningRoadmapRequest,
    LearningRoadmapResponse,
    JobBookmark,
    JobFreshnessResponse,
    JobStatsResponse,
    JobTagCloudResponse,
    LocationPreferenceResponse,
    LocationPreferenceUpdateRequest,
    OfferComparisonRequest,
    OfferComparisonResponse,
    PostInterviewThankYouRequest,
    PostInterviewThankYouResponse,
    ProfileCompletenessResponse,
    RejectionReasonLogRequest,
    RejectionReasonLogResponse,
    RemoteWorkBreakdownResponse,
    RemoteWorkPolicyMetric,
    SalaryInfo,
    SalaryNegotiationScriptRequest,
    SalaryNegotiationScriptResponse,
    SalaryPreferenceResponse,
    SalaryExpectationMatchRequest,
    SalaryExpectationMatchResponse,
    SalaryPreferenceUpdateRequest,
    SavedSearch,
    SavedSearchCreateRequest,
    SavedSearchExecutionResponse,
    SavedSearchUpdateRequest,
    SearchHistoryEntry,
    SearchSuggestionsResponse,
    SessionMetricsResponse,
    SessionPreferencesResponse,
    SessionPreferencesUpdateRequest,
    SkillCoOccurrenceItem,
    SkillCorrelationResponse,
    SkillGapAnalysisResponse,
    SkillProficiencyGapRequest,
    SkillProficiencyGapResponse,
    SkillSalaryInsightsResponse,
    SoftSkillsResponse,
    SoftSkillsUpdateRequest,
    StaleJobItem,
    StaleJobsResponse,
    StarAnswerCoachRequest,
    StarAnswerCoachResponse,
    StorageSpeedBenchmark,
    SystemDiagnosticsResponse,
    TierSalaryInsightsResponse,
    TitleNormalizationRequest,
    TitleNormalizationResponse,
    UploadResumeResponse,
    WeightedSkillScoreRequest,
    WeightedSkillScoreResponse,
    WorkPreferencesResponse,
    WorkPreferencesUpdateRequest,
)
from app.services.db_service import (
    add_application_attachment,
    add_application_interview,
    add_candidate_certification,
    add_candidate_portfolio_link,
    add_interview_question,
    advanced_filter_jobs,
    analyze_resume_ats_readability,
    apply_to_job,
    archive_application,
    benchmark_database_performance,
    bulk_remove_bookmarks,
    bulk_update_application_status,
    calculate_and_set_auto_followup,
    calculate_profile_completeness,
    check_database_health,
    clear_search_history,
    compact_database,
    compare_offer_packages,
    compute_database_checksums,
    create_database_backup,
    create_saved_search,
    delete_candidate_certification,
    delete_candidate_portfolio_link,
    delete_saved_search,
    execute_saved_search,
    export_application_portfolio,
    export_jobs_to_csv,
    generate_database_integrity_report,
    generate_followup_template,
    get_application_channel_analytics,
    get_application_funnel_metrics,
    get_application_stage_durations,
    get_application_stats,
    get_application_status,
    get_application_timeline,
    get_application_velocity_metrics,
    get_bookmark_categories,
    get_bookmarked_jobs,
    get_candidate_certifications,
    get_candidate_languages,
    get_candidate_location_preferences,
    get_candidate_portfolio_links,
    get_candidate_salary_preferences,
    get_candidate_soft_skills,
    get_candidate_work_preferences,
    get_company_hiring_surges,
    get_company_profile,
    get_database_audit_log,
    get_database_storage_metrics,
    get_experience_tier_breakdown,
    get_interview_questions,
    get_job_by_id,
    get_job_freshness_details,
    get_job_stats,
    get_job_tag_cloud,
    get_recent_searches,
    get_rejection_analytics,
    get_remote_work_breakdown,
    get_search_suggestions,
    get_similar_jobs,
    get_skill_correlations,
    get_skill_salary_insights,
    get_stale_jobs,
    get_tier_salary_insights,
    get_upcoming_reminders,
    has_rollback_snapshot,
    list_hiring_companies,
    load_applications,
    log_application_rejection_reason,
    load_candidate_profile,
    load_jobs,
    load_saved_searches,
    record_interview_difficulty_feedback,
    record_search_query,
    remove_job_bookmark,
    repair_database_records,
    restore_database_backup,
    rollback_application_state,
    run_system_storage_diagnostics,
    save_candidate_profile,
    save_job_bookmark,
    search_jobs,
    update_application_deadlines,
    update_application_interview,
    update_application_notes,
    update_application_outcome,
    update_application_priority,
    update_application_reminder,
    update_application_source,
    update_application_status,
    update_application_tags,
    update_candidate_languages,
    update_candidate_location_preferences,
    update_candidate_salary_preferences,
    update_candidate_soft_skills,
    update_candidate_work_preferences,
    update_job_bookmark,
    update_saved_search,
    vacuum_database_collections,
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
        save_candidate_profile(candidate_profile.model_dump())

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
    "/api/resume/ats-score",
    response_model=ATSScoreResponse,
    summary="Analyze resume ATS compatibility and keyword density",
    tags=["Resume"]
)
async def analyze_ats_score_endpoint(payload: ATSScoreRequest):
    """Analyze resume plain text for ATS keyword density, formatting, and structural completeness."""
    return analyze_resume_ats_readability(payload.resume_text, target_category=payload.target_category)


@app.get(
    "/api/profile",
    response_model=Dict[str, Any],
    summary="Get active candidate profile",
    tags=["Resume"]
)
async def get_candidate_profile_endpoint():
    """Retrieve currently stored candidate resume profile."""
    return load_candidate_profile()


@app.post(
    "/api/profile",
    response_model=Dict[str, Any],
    summary="Save or update candidate profile",
    tags=["Resume"]
)
async def save_candidate_profile_endpoint(profile: Dict[str, Any]):
    """Update and persist candidate profile data."""
    save_candidate_profile(profile)
    return {"status": "success", "message": "Candidate profile saved successfully.", "profile": profile}


@app.get(
    "/api/profile/completeness",
    response_model=ProfileCompletenessResponse,
    summary="Audit candidate profile completeness and strength",
    tags=["Resume"]
)
async def get_profile_completeness_endpoint():
    """Evaluate profile completeness score (0-100) and identify missing sections for currently loaded profile."""
    profile = load_candidate_profile()
    return calculate_profile_completeness(profile)


@app.post(
    "/api/profile/completeness",
    response_model=ProfileCompletenessResponse,
    summary="Audit completeness of provided profile payload",
    tags=["Resume"]
)
async def evaluate_profile_completeness_endpoint(profile: Dict[str, Any]):
    """Evaluate profile completeness score and identify missing sections for an arbitrary profile."""
    return calculate_profile_completeness(profile)


@app.get(
    "/api/profile/salary-preferences",
    response_model=SalaryPreferenceResponse,
    summary="Get candidate target salary preferences",
    tags=["Resume"]
)
async def get_salary_preferences_endpoint():
    """Retrieve candidate target salary range, currency, and payment frequency."""
    return get_candidate_salary_preferences()


@app.patch(
    "/api/profile/salary-preferences",
    response_model=SalaryPreferenceResponse,
    summary="Update candidate target salary preferences",
    tags=["Resume"]
)
async def update_salary_preferences_endpoint(payload: SalaryPreferenceUpdateRequest):
    """Update candidate target salary min/max, currency, or period."""
    success, message, prefs = update_candidate_salary_preferences(
        target_min_salary=payload.target_min_salary,
        target_max_salary=payload.target_max_salary,
        currency=payload.currency or "USD",
        period=payload.period or "year"
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return {
        "status": "success",
        "message": message,
        "target_min_salary": prefs.get("target_min_salary"),
        "target_max_salary": prefs.get("target_max_salary"),
        "currency": prefs.get("currency", "USD"),
        "period": prefs.get("period", "year")
    }


@app.get(
    "/api/profile/work-preferences",
    response_model=WorkPreferencesResponse,
    summary="Get candidate target work environment preferences",
    tags=["Resume"]
)
async def get_work_preferences_endpoint():
    """Retrieve candidate preferred company sizes, industry sectors, and remote flexibility."""
    return get_candidate_work_preferences()


@app.patch(
    "/api/profile/work-preferences",
    response_model=WorkPreferencesResponse,
    summary="Update candidate target work environment preferences",
    tags=["Resume"]
)
async def update_work_preferences_endpoint(payload: WorkPreferencesUpdateRequest):
    """Update candidate target company stages, industries, or remote preference."""
    success, message, prefs = update_candidate_work_preferences(
        company_sizes=payload.company_sizes,
        preferred_industries=payload.preferred_industries,
        remote_preference=payload.remote_preference or "any"
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return {
        "status": "success",
        "message": message,
        "company_sizes": prefs.get("company_sizes", []),
        "preferred_industries": prefs.get("preferred_industries", []),
        "remote_preference": prefs.get("remote_preference", "any")
    }


@app.get(
    "/api/profile/certifications",
    response_model=CertificationsResponse,
    summary="List candidate certifications and credentials",
    tags=["Resume"]
)
async def list_certifications_endpoint():
    """Retrieve all verified professional certifications on candidate profile."""
    certs = get_candidate_certifications()
    return CertificationsResponse(
        status="success",
        total=len(certs),
        certifications=certs
    )


@app.post(
    "/api/profile/certifications",
    response_model=CertificationItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a professional certification",
    tags=["Resume"]
)
async def add_certification_endpoint(payload: CertificationAddRequest):
    """Add a new credential or certification to candidate profile."""
    success, message, new_cert = add_candidate_certification(
        name=payload.name,
        issuer=payload.issuer,
        issue_date=payload.issue_date,
        expiration_date=payload.expiration_date,
        credential_url=payload.credential_url
    )
    if not success or not new_cert:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return new_cert


@app.delete(
    "/api/profile/certifications/{cert_id}",
    summary="Delete a candidate certification",
    tags=["Resume"]
)
async def delete_certification_endpoint(cert_id: str):
    """Delete a professional certification by its unique cert_id."""
    clean_id = (cert_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Certification ID cannot be empty.")
    success, message = delete_candidate_certification(clean_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {"status": "success", "message": message}


@app.get(
    "/api/profile/languages",
    response_model=CandidateLanguagesResponse,
    summary="Get candidate language proficiencies",
    tags=["Resume"]
)
async def get_candidate_languages_endpoint():
    """Retrieve all language proficiencies recorded on candidate profile."""
    langs = get_candidate_languages()
    return CandidateLanguagesResponse(
        status="success",
        total_languages=len(langs),
        languages=langs
    )


@app.put(
    "/api/profile/languages",
    response_model=CandidateLanguagesResponse,
    summary="Update candidate language proficiencies",
    tags=["Resume"]
)
async def update_candidate_languages_endpoint(payload: CandidateLanguagesUpdateRequest):
    """Set and persist language proficiencies on candidate profile."""
    res = update_candidate_languages([item.model_dump() for item in payload.languages])
    return CandidateLanguagesResponse(**res)


@app.get(
    "/api/profile/location-preferences",
    response_model=LocationPreferenceResponse,
    summary="Get candidate location and relocation preferences",
    tags=["Resume"]
)
async def get_location_preferences_endpoint():
    """Retrieve geographic, workplace type, and relocation willingness preferences."""
    prefs = get_candidate_location_preferences()
    return LocationPreferenceResponse(**prefs)


@app.patch(
    "/api/profile/location-preferences",
    response_model=LocationPreferenceResponse,
    summary="Update candidate location and relocation preferences",
    tags=["Resume"]
)
async def update_location_preferences_endpoint(payload: LocationPreferenceUpdateRequest):
    """Update preferred locations, relocation destinations, and workplace model."""
    res = update_candidate_location_preferences(
        preferred_locations=payload.preferred_locations,
        open_to_relocation=payload.open_to_relocation,
        relocation_destinations=payload.relocation_destinations,
        workplace_type=payload.workplace_type
    )
    return LocationPreferenceResponse(**res)


@app.get(
    "/api/profile/links",
    response_model=CandidatePortfolioLinksResponse,
    summary="List candidate portfolio, GitHub, and project links",
    tags=["Resume"]
)
async def list_portfolio_links_endpoint():
    """Retrieve all external project links, repositories, and portfolios on candidate profile."""
    links = get_candidate_portfolio_links()
    return CandidatePortfolioLinksResponse(
        status="success",
        total_links=len(links),
        links=links
    )


@app.post(
    "/api/profile/links",
    response_model=CandidatePortfolioLinkItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add an external portfolio or project link",
    tags=["Resume"]
)
async def add_portfolio_link_endpoint(payload: PortfolioLinkCreateRequest):
    """Add a GitHub, portfolio, or live project link to candidate profile."""
    success, message, link_item = add_candidate_portfolio_link(
        label=payload.label,
        url=payload.url,
        description=payload.description
    )
    if not success or not link_item:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return link_item


@app.delete(
    "/api/profile/links/{link_id}",
    summary="Delete a candidate portfolio link",
    tags=["Resume"]
)
async def delete_portfolio_link_endpoint(link_id: str):
    """Remove a portfolio or project link by its link_id."""
    clean_id = (link_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Link ID cannot be empty.")
    success, message = delete_candidate_portfolio_link(clean_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {"status": "success", "message": message}


@app.get(
    "/api/profile/soft-skills",
    response_model=SoftSkillsResponse,
    summary="Get candidate soft skills and interpersonal strengths",
    tags=["Resume"]
)
async def get_soft_skills_endpoint():
    """Retrieve all candidate recorded soft skills and collaboration strengths."""
    skills = get_candidate_soft_skills()
    return SoftSkillsResponse(
        status="success",
        total_skills=len(skills),
        soft_skills=skills
    )


@app.put(
    "/api/profile/soft-skills",
    response_model=SoftSkillsResponse,
    summary="Update candidate soft skills and strengths",
    tags=["Resume"]
)
async def update_soft_skills_endpoint(payload: SoftSkillsUpdateRequest):
    """Update and persist candidate interpersonal and leadership strengths."""
    res = update_candidate_soft_skills(payload.soft_skills)
    return SoftSkillsResponse(**res)






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


@app.post(
    "/api/chat/clarification",
    response_model=DisambiguationResponse,
    summary="Evaluate intent confidence and suggest clarification options",
    tags=["Chat"]
)
async def evaluate_chat_clarification_endpoint(payload: ChatRequest):
    """Detect if conversational request is ambiguous and generate quick-reply clarification prompts."""
    return evaluate_intent_confidence_and_clarification(payload.message)


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


@app.post(
    "/api/chat/tools/negotiation-script",
    response_model=SalaryNegotiationScriptResponse,
    summary="Generate personalized salary counter-offer script",
    tags=["Chat"]
)
async def generate_negotiation_script_endpoint(payload: SalaryNegotiationScriptRequest):
    """Generate professional counter-offer email template and live phone negotiation talking points."""
    return generate_salary_negotiation_script(
        company=payload.company,
        role_title=payload.role_title,
        initial_offer=payload.initial_offer,
        target_counter=payload.target_counter,
        leverage_points=payload.key_leverage_points
    )


@app.post(
    "/api/chat/tools/cold-outreach",
    response_model=ColdOutreachMessageResponse,
    summary="Generate tailored cold outreach message for recruiters or hiring managers",
    tags=["Chat"]
)
async def generate_cold_outreach_endpoint(payload: ColdOutreachMessageRequest):
    """Generate concise LinkedIn connection request notes or professional cold outreach emails."""
    return generate_cold_outreach_message(
        company=payload.company,
        role_title=payload.role_title,
        recipient_type=payload.recipient_type,
        candidate_background=payload.candidate_background,
        channel=payload.channel
    )


@app.post(
    "/api/chat/coach/star",
    response_model=StarAnswerCoachResponse,
    summary="Structure behavioral interview response into STAR format",
    tags=["Chat"]
)
async def coach_star_answer_endpoint(payload: StarAnswerCoachRequest):
    """Structure unstructured candidate experiences into Situation, Task, Action, Result framework."""
    return coach_star_behavioral_answer(
        question=payload.question,
        raw_notes=payload.raw_notes,
        target_role=payload.target_role
    )


@app.get(
    "/api/chat/tools/followup-cadence",
    response_model=FollowupCadenceRecommendationResponse,
    summary="Get optimal recruiter follow-up cadence schedule and drafts",
    tags=["Chat"]
)
async def get_followup_cadence_endpoint(
    company: str = Query(..., description="Target company"),
    role_title: str = Query(..., description="Target role title"),
    current_stage: str = Query("applied", description="Current stage: applied or interview")
):
    """Generate multi-stage recruiter follow-up schedule and pre-written message templates."""
    return generate_recruiter_followup_cadence(
        company=company,
        role_title=role_title,
        current_stage=current_stage
    )


@app.post(
    "/api/chat/tools/thank-you-email",
    response_model=PostInterviewThankYouResponse,
    summary="Generate personalized post-interview thank you email",
    tags=["Chat"]
)
async def generate_thank_you_email_endpoint(payload: PostInterviewThankYouRequest):
    """Generate professional, personalized thank you email mentioning key interview discussions."""
    return generate_post_interview_thank_you_email(
        company=payload.company,
        role_title=payload.role_title,
        interviewer_name=payload.interviewer_name,
        key_discussion_points=payload.key_discussion_points,
        custom_gratitude_note=payload.custom_gratitude_note
    )


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


@app.get(
    "/api/chat/session/{session_id}/metrics",
    response_model=SessionMetricsResponse,
    summary="Get conversation session activity and turn metrics",
    tags=["Chat"]
)
async def get_session_metrics_endpoint(session_id: str):
    """Retrieve detailed interaction metrics, turns count, and activity timestamps for a chat session."""
    return get_session_metrics(session_id)


@app.get(
    "/api/chat/session/{session_id}/export",
    summary="Export chat conversation transcript",
    tags=["Chat"]
)
async def export_chat_transcript_endpoint(
    session_id: str,
    format: str = Query("markdown", description="Format: 'markdown' or 'text'")
):
    """Export formatted conversation transcript for candidate review or offline storage."""
    clean_id = (session_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID cannot be empty.")
    success, message, transcript = export_session_transcript(clean_id, format=format)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    ext = "md" if format == "markdown" else "txt"
    media_type = "text/markdown" if format == "markdown" else "text/plain"
    return Response(
        content=transcript,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="transcript_{clean_id}.{ext}"'}
    )


@app.get(
    "/api/chat/session/{session_id}/preferences",
    response_model=SessionPreferencesResponse,
    summary="Get retained chat session preferences",
    tags=["Chat"]
)
async def get_session_preferences_endpoint(session_id: str):
    """Retrieve candidate context and preferences retained in current conversation session."""
    clean_id = (session_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID cannot be empty.")
    prefs = get_session_preferences(clean_id)
    return SessionPreferencesResponse(status="success", session_id=clean_id, preferences=prefs)


@app.patch(
    "/api/chat/session/{session_id}/preferences",
    response_model=SessionPreferencesResponse,
    summary="Update retained chat session preferences",
    tags=["Chat"]
)
async def update_session_preferences_endpoint(session_id: str, payload: SessionPreferencesUpdateRequest):
    """Update contextual preferences (target role, location, min salary) in session memory."""
    clean_id = (session_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID cannot be empty.")
    prefs = update_session_preferences(
        session_id=clean_id,
        target_title=payload.target_title,
        location_preference=payload.location_preference,
        min_salary=payload.min_salary
    )
    return SessionPreferencesResponse(status="success", session_id=clean_id, preferences=prefs)


@app.get("/api/jobs", summary="Browse available jobs", tags=["Jobs"])
async def get_jobs_list(
    query: Optional[str] = Query(None, description="Optional search keyword"),
    location: Optional[str] = Query(None, description="Optional location filter (e.g. Remote, Berlin)"),
    experience_level: Optional[str] = Query(None, description="Optional experience tier filter (Entry-Level, Mid-Level, Senior)"),
    company: Optional[str] = Query(None, description="Optional company name filter"),
    category: Optional[str] = Query(None, description="Optional industry domain filter (Engineering, Data & AI, Product & Design, Finance & Operations, Marketing & Sales)"),
    freshness: Optional[str] = Query(None, description="Optional freshness filter (new, recent, archive)"),
    work_mode: Optional[str] = Query(None, description="Optional work mode filter (Remote, Hybrid, Onsite)"),
    tier: Optional[str] = Query(None, description="Optional standardized career tier filter (Entry-Level, Mid-Level, Senior, Lead)"),
    has_salary: Optional[bool] = Query(None, description="Optional filter for jobs with disclosed compensation"),
    min_salary: Optional[float] = Query(None, ge=0, description="Optional minimum annual salary filter"),
    salary_currency: Optional[str] = Query(None, description="Optional currency filter (USD, EUR, GBP)"),
    sort_by: Optional[str] = Query(None, description="Optional field to sort by (title, company, experience, location, salary)"),
    order: Optional[str] = Query("asc", description="Sort direction (asc, desc)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Retrieve paginated jobs from the local 1,600+ job database with multi-field filtering and sorting."""
    clean_query = (query or "").strip()
    clean_loc = (location or "").strip()
    clean_exp = (experience_level or "").strip()
    clean_company = (company or "").strip()
    clean_category = (category or "").strip()
    clean_freshness = (freshness or "").strip()
    clean_mode = (work_mode or "").strip()
    clean_tier = (tier or "").strip()
    clean_sort = (sort_by or "").strip().lower()
    is_desc = (order or "asc").strip().lower() == "desc"

    if clean_query or clean_loc or clean_exp or clean_company or clean_category or clean_freshness or clean_mode or clean_tier or clean_sort or has_salary is not None or min_salary is not None or salary_currency:
        all_jobs = search_jobs(
            query=clean_query or None,
            location=clean_loc or None,
            experience_level=clean_exp or None,
            company=clean_company or None,
            category=clean_category or None,
            freshness=clean_freshness or None,
            work_mode=clean_mode or None,
            tier=clean_tier or None,
            has_salary=has_salary,
            min_salary=min_salary,
            salary_currency=salary_currency,
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


@app.post(
    "/api/jobs/filter",
    response_model=AdvancedJobFilterResponse,
    summary="Advanced multi-criteria job filtering",
    tags=["Jobs"]
)
async def advanced_job_filter_endpoint(payload: AdvancedJobFilterRequest):
    """Perform complex multi-criteria boolean filtering across jobs with AND/OR skills matching and compensation floor."""
    return advanced_filter_jobs(
        keywords=payload.keywords,
        skills=payload.skills,
        match_all_skills=payload.match_all_skills,
        work_modes=payload.work_modes,
        experience_tiers=payload.experience_tiers,
        min_salary=payload.min_salary,
        category=payload.category,
        limit=payload.limit,
        offset=payload.offset
    )


@app.get(
    "/api/companies",
    response_model=CompanyDirectoryResponse,
    summary="List hiring companies and hiring insights",
    tags=["Companies"]
)
async def list_companies_endpoint(limit: int = Query(50, ge=1, le=100)):
    """Retrieve directory of hiring companies ranked by active job listings with skill insights."""
    companies = list_hiring_companies(limit=limit)
    return {
        "total_companies": len(companies),
        "companies": companies
    }


@app.get(
    "/api/companies/{company_name}",
    response_model=CompanyProfile,
    summary="Get detailed company hiring profile",
    tags=["Companies"]
)
async def get_company_profile_endpoint(company_name: str):
    """Retrieve in-depth hiring profile, top demanded skills, and sample open roles for a company."""
    profile = get_company_profile(company_name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No active job listings found for company '{company_name}'.")
    return profile


@app.get("/api/searches/recent", response_model=List[SearchHistoryEntry], summary="Get recent job searches", tags=["Searches"])
async def get_recent_searches_endpoint(limit: int = Query(10, ge=1, le=50)):
    """Retrieve candidate's recent search queries and active filter history."""
    return get_recent_searches(limit=limit)


@app.delete("/api/searches/recent", summary="Clear search history", tags=["Searches"])
async def clear_recent_searches_endpoint():
    """Clear candidate's recorded search history."""
    clear_search_history()
    return {"status": "success", "message": "Search history cleared."}


@app.post(
    "/api/searches/saved",
    response_model=SavedSearch,
    status_code=status.HTTP_201_CREATED,
    summary="Save a search query subscription / job alert",
    tags=["Searches"]
)
async def create_saved_search_endpoint(payload: SavedSearchCreateRequest):
    """Save a search criteria subscription for quick candidate re-runs."""
    success, message, new_search = create_saved_search(
        title=payload.title,
        query=payload.query,
        location=payload.location,
        experience_level=payload.experience_level,
        category=payload.category
    )
    if not success:
        if new_search:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return new_search


@app.get(
    "/api/searches/saved",
    response_model=List[SavedSearch],
    summary="List saved search subscriptions",
    tags=["Searches"]
)
async def list_saved_searches_endpoint():
    """Retrieve all saved search subscriptions."""
    return load_saved_searches()


@app.patch(
    "/api/searches/saved/{search_id}",
    response_model=SavedSearch,
    summary="Update saved search criteria",
    tags=["Searches"]
)
async def update_saved_search_endpoint(search_id: str, payload: SavedSearchUpdateRequest):
    """Update saved search subscription title or filter criteria."""
    clean_id = (search_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search ID cannot be empty.")
    success, message, updated_s = update_saved_search(
        clean_id,
        title=payload.title,
        query=payload.query,
        location=payload.location,
        experience_level=payload.experience_level,
        category=payload.category
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return updated_s


@app.delete(
    "/api/searches/saved/{search_id}",
    summary="Delete a saved search subscription",
    tags=["Searches"]
)
async def delete_saved_search_endpoint(search_id: str):
    """Delete a saved search subscription by its search_id."""
    clean_id = (search_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search ID cannot be empty.")
    success, message = delete_saved_search(clean_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {"status": "success", "message": message}


@app.post(
    "/api/searches/saved/{search_id}/run",
    response_model=SavedSearchExecutionResponse,
    summary="Execute a saved search subscription query",
    tags=["Searches"]
)
async def execute_saved_search_endpoint(
    search_id: str,
    limit: int = Query(15, ge=1, le=50, description="Max job matches to return")
):
    """Execute a saved search criteria subscription and update match counts and execution timestamps."""
    clean_id = (search_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search ID cannot be empty.")
    success, message, result = execute_saved_search(clean_id, limit=limit)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return result


@app.get("/api/jobs/stats", response_model=JobStatsResponse, summary="Get job database statistics and market insights", tags=["Jobs"])
async def get_jobs_statistics():
    """Retrieve aggregated job market statistics, top demanded skills, and remote job distribution."""
    return get_job_stats()


@app.get(
    "/api/jobs/suggestions",
    response_model=SearchSuggestionsResponse,
    summary="Get search query autocomplete suggestions",
    tags=["Jobs"]
)
async def get_search_suggestions_endpoint(
    prefix: str = Query(..., min_length=1, description="Query prefix to search for"),
    limit: int = Query(8, ge=1, le=25, description="Max suggestions to return")
):
    """Retrieve fast typeahead suggestions across job titles, skills, and hiring companies."""
    clean_prefix = (prefix or "").strip()
    suggestions = get_search_suggestions(clean_prefix, limit=limit)
    return SearchSuggestionsResponse(
        prefix=clean_prefix,
        suggestions=suggestions
    )


@app.get(
    "/api/jobs/analytics/skill-salaries",
    response_model=SkillSalaryInsightsResponse,
    summary="Get skill-to-compensation benchmarks",
    tags=["Jobs"]
)
async def get_skill_salary_insights_endpoint(
    limit: int = Query(15, ge=1, le=50, description="Max skills to evaluate")
):
    """Retrieve average compensation benchmarks associated with in-demand technical skills."""
    return get_skill_salary_insights(limit=limit)


@app.get(
    "/api/jobs/analytics/tag-cloud",
    response_model=JobTagCloudResponse,
    summary="Get aggregated technical tag cloud across job postings",
    tags=["Jobs"]
)
async def get_job_tag_cloud_endpoint(
    category: Optional[str] = Query(None, description="Optional category filter"),
    limit: int = Query(30, ge=1, le=100, description="Max tags to return")
):
    """Retrieve top technical keywords and skills tag cloud with occurrence counts."""
    return get_job_tag_cloud(category=category, limit=limit)


@app.get(
    "/api/jobs/analytics/tier-salaries",
    response_model=TierSalaryInsightsResponse,
    summary="Get compensation benchmarks across career tiers",
    tags=["Jobs"]
)
async def get_tier_salary_insights_endpoint():
    """Retrieve compensation benchmarks and distributions across career tiers (Entry-Level, Mid-Level, Senior, Lead)."""
    return get_tier_salary_insights()


@app.get(
    "/api/jobs/analytics/hiring-surges",
    response_model=CompanyHiringSurgeResponse,
    summary="Detect companies with rapid hiring surges",
    tags=["Jobs"]
)
async def get_company_hiring_surges_endpoint(limit: int = Query(10, ge=1, le=50)):
    """Analyze company hiring velocity, department expansion patterns, and recent job surges."""
    return get_company_hiring_surges(limit=limit)


@app.get(
    "/api/jobs/analytics/stale",
    response_model=StaleJobsResponse,
    summary="Detect stale and aging job postings",
    tags=["Jobs"]
)
async def get_stale_jobs_endpoint(
    threshold_days: int = Query(45, ge=7, le=180, description="Minimum age in days to consider a posting stale"),
    limit: int = Query(50, ge=1, le=100, description="Max stale jobs to return")
):
    """Scan the job database to surface older listings that may be expired or no longer actively hiring."""
    return get_stale_jobs(threshold_days=threshold_days, limit=limit)


@app.get(
    "/api/jobs/analytics/skill-correlations",
    response_model=SkillCorrelationResponse,
    summary="Get companion skill co-occurrence correlations",
    tags=["Jobs"]
)
async def get_skill_correlations_endpoint(
    skill: str = Query(..., min_length=1, description="Anchor skill to evaluate companion co-occurrences for"),
    limit: int = Query(10, ge=1, le=50, description="Max correlated companion skills to return")
):
    """Identify companion skills most frequently demanded alongside a target technical skill."""
    return get_skill_correlations(skill=skill, limit=limit)


@app.get(
    "/api/jobs/analytics/remote-breakdown",
    response_model=RemoteWorkBreakdownResponse,
    summary="Get remote vs hybrid vs onsite market breakdown",
    tags=["Jobs"]
)
async def get_remote_work_breakdown_endpoint():
    """Analyze the distribution of remote, hybrid, and onsite job listings with average salary benchmarks."""
    return get_remote_work_breakdown()


@app.get(
    "/api/jobs/analytics/experience-tiers",
    response_model=ExperienceTierBreakdownResponse,
    summary="Get experience tier distribution analytics and salary ranges",
    tags=["Jobs"]
)
async def get_experience_tier_breakdown_endpoint():
    """Analyze the distribution of job listings across seniority tiers with top categories and compensation benchmarks."""
    return get_experience_tier_breakdown()



@app.get("/api/jobs/export/csv", summary="Export jobs as CSV", tags=["Jobs"])
async def export_jobs_csv_endpoint(
    query: Optional[str] = Query(None, description="Search term filter"),
    location: Optional[str] = Query(None, description="Location filter"),
    experience_level: Optional[str] = Query(None, description="Experience level filter"),
    category: Optional[str] = Query(None, description="Domain category filter")
):
    """Export job listings matching current filters as a downloadable CSV spreadsheet."""
    csv_content = export_jobs_to_csv(query=query, location=location, experience_level=experience_level, category=category)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="jobs_export.csv"'}
    )


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
    "/api/jobs/{job_id}/freshness",
    response_model=JobFreshnessResponse,
    summary="Get job posting freshness tier and age in days",
    tags=["Jobs"]
)
async def get_job_freshness_endpoint(job_id: str):
    """Retrieve job posting age, freshness tier (New, Recent, Established), and urgency status."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    freshness = get_job_freshness_details(clean_id)
    if not freshness:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{clean_id}' was not found.")
    return freshness



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


@app.get(
    "/api/jobs/{job_id}/interview-difficulty",
    response_model=InterviewDifficultyResponse,
    summary="Estimate technical interview difficulty score and complexity breakdown",
    tags=["Jobs"]
)
async def estimate_job_interview_difficulty_endpoint(job_id: str):
    """Calculate predicted interview difficulty rating (1.0 to 5.0) and expected stage curriculum."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    result = estimate_interview_difficulty(clean_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{clean_id}' was not found in the database."
        )
    return result



@app.post(
    "/api/matcher/batch-score",
    response_model=BatchScoreResponse,
    summary="Batch evaluate candidate compatibility against multiple jobs",
    tags=["Matcher"]
)
async def batch_score_jobs_endpoint(payload: BatchScoreRequest):
    """Score candidate compatibility, matching skills, and missing skills across multiple job postings."""
    results = batch_score_jobs(
        profile=payload.profile,
        candidate_skills=payload.candidate_skills,
        job_ids=payload.job_ids
    )
    return {
        "total_evaluated": len(results),
        "results": results
    }


@app.post(
    "/api/matcher/learning-roadmap",
    response_model=LearningRoadmapResponse,
    summary="Generate personalized study roadmap and hands-on projects for missing job skills",
    tags=["Matcher"]
)
async def generate_learning_roadmap_endpoint(payload: LearningRoadmapRequest):
    """Generate structured study curriculum, estimated effort, and project recommendations for target missing skills."""
    return generate_learning_roadmap(
        missing_skills=payload.missing_skills,
        target_role=payload.target_role
    )


@app.post(
    "/api/matcher/weighted-score",
    response_model=WeightedSkillScoreResponse,
    summary="Evaluate job match with custom candidate-weighted skill multipliers",
    tags=["Matcher"]
)
async def calculate_weighted_skill_match_endpoint(payload: WeightedSkillScoreRequest):
    """Calculate customized job compatibility score applying candidate-defined skill weights."""
    result = calculate_weighted_skill_match(
        job_id=payload.job_id,
        candidate_skills=payload.candidate_skills,
        skill_weights=payload.skill_weights
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{payload.job_id}' was not found in the database."
        )
    return result


@app.post(
    "/api/matcher/proficiency-gap",
    response_model=SkillProficiencyGapResponse,
    summary="Evaluate candidate skill proficiency level gaps against job expectations",
    tags=["Matcher"]
)
async def evaluate_skill_proficiency_gaps_endpoint(payload: SkillProficiencyGapRequest):
    """Assess whether candidate skill proficiencies meet, exceed, or fall short of role requirements."""
    return evaluate_skill_proficiency_gaps(
        candidate_proficiencies=payload.candidate_proficiencies,
        job_id=payload.job_id,
        job_required_skills=payload.job_required_skills
    )


@app.post(
    "/api/matcher/career-transition-score",
    response_model=CareerTransitionResponse,
    summary="Evaluate career trajectory leap and bridge skills between roles",
    tags=["Matcher"]
)
async def score_career_transition_endpoint(payload: CareerTransitionRequest):
    """Assess transition difficulty, similarity score, and bridge competencies between current and target titles."""
    return score_career_transition(
        current_title=payload.current_title,
        target_title=payload.target_title,
        current_experience_years=payload.current_experience_years or 3.0
    )


@app.get(
    "/api/matcher/predict-questions",
    response_model=CompanyInterviewPredictionResponse,
    summary="Predict company-tailored interview questions and evaluation loops",
    tags=["Matcher"]
)
async def predict_company_questions_endpoint(
    company: str = Query(..., description="Target company name"),
    role_title: str = Query(..., description="Target job title"),
    skills: Optional[List[str]] = Query(None, description="Key technical skills evaluated")
):
    """Predict high-probability technical, behavioral, and system design questions for a company and role."""
    return predict_company_interview_questions(
        company=company,
        role_title=role_title,
        skills=skills
    )


@app.post(
    "/api/matcher/salary-fit",
    response_model=SalaryExpectationMatchResponse,
    summary="Evaluate candidate compensation expectations against job salary",
    tags=["Matcher"]
)
async def evaluate_salary_fit_endpoint(payload: SalaryExpectationMatchRequest):
    """Calculate compensation compatibility, range overlap percentage, and negotiation headroom."""
    return calculate_salary_expectation_fit(
        candidate_min=payload.candidate_min_salary,
        candidate_max=payload.candidate_max_salary,
        job_id=payload.job_id,
        job_min=payload.job_min_salary,
        job_max=payload.job_max_salary
    )


@app.post(
    "/api/matcher/normalize-title",
    response_model=TitleNormalizationResponse,
    summary="Normalize job title to canonical tech family, seniority, and competencies",
    tags=["Matcher"]
)
async def normalize_title_endpoint(payload: TitleNormalizationRequest):
    """Classify and normalize arbitrary tech role titles into standard engineering disciplines."""
    return normalize_job_title_semantics(payload.raw_title)





@app.get(
    "/api/jobs/{job_id}/similar",
    summary="Get similar and related job recommendations",
    tags=["Jobs"]
)
async def get_similar_jobs_endpoint(
    job_id: str,
    limit: int = Query(5, ge=1, le=20, description="Max similar jobs to retrieve")
):
    """Recommend related jobs based on skills overlap, domain category, experience, and location."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job with ID '{clean_id}' was not found.")
    similar = get_similar_jobs(clean_id, limit=limit)
    return {
        "job_id": clean_id,
        "job_title": job.get("title"),
        "total_similar": len(similar),
        "similar_jobs": similar
    }


@app.get("/api/applications", summary="List all submitted applications", tags=["Applications"])
async def get_applications_list(
    status: Optional[str] = Query(None, description="Optional status filter (e.g. Applied, Under Review)"),
    search: Optional[str] = Query(None, description="Optional search keyword matching job title or company"),
    tag: Optional[str] = Query(None, description="Optional tag filter"),
    priority: Optional[str] = Query(None, description="Optional priority filter (Low, Medium, High, Urgent)"),
    is_favorite: Optional[bool] = Query(None, description="Optional filter for starred/favorite applications"),
    include_archived: bool = Query(False, description="Whether to include archived applications")
):
    """List all recorded job applications from applications.json with optional status, search, and tag filters."""
    apps = load_applications()
    if not include_archived:
        apps = [a for a in apps if not a.get("is_archived", False)]
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
    if tag:
        tag_clean = tag.strip().lower()
        apps = [
            a for a in apps
            if any(tag_clean == (t or "").strip().lower() for t in a.get("tags", []))
        ]
    if priority:
        prio_clean = priority.strip().lower()
        apps = [a for a in apps if (a.get("priority") or "medium").strip().lower() == prio_clean]
    if is_favorite is not None:
        apps = [a for a in apps if bool(a.get("is_favorite", False)) == is_favorite]
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
    "/api/applications/analytics/funnel",
    response_model=ApplicationFunnelResponse,
    summary="Get hiring pipeline conversion funnel metrics",
    tags=["Applications"]
)
async def get_applications_funnel_endpoint():
    """Calculate conversion rates across candidate application pipeline stages."""
    return get_application_funnel_metrics()


@app.get(
    "/api/applications/analytics/velocity",
    response_model=ApplicationVelocityResponse,
    summary="Get application submission rate and velocity metrics",
    tags=["Applications"]
)
async def get_applications_velocity_endpoint():
    """Retrieve application submission rate across 7-day and 30-day windows."""
    return get_application_velocity_metrics()


@app.get(
    "/api/applications/analytics/rejections",
    response_model=ApplicationRejectionAnalyticsResponse,
    summary="Get rejection reasons and debrief feedback analytics",
    tags=["Applications"]
)
async def get_applications_rejections_endpoint():
    """Analyze rejection codes and qualitative feedback across candidate applications."""
    return get_rejection_analytics()


@app.get(
    "/api/applications/analytics/channels",
    response_model=ApplicationChannelAnalyticsResponse,
    summary="Get application conversion rates by submission source channel",
    tags=["Applications"]
)
async def get_application_channel_analytics_endpoint():
    """Analyze application interview and offer conversion performance grouped by channel."""
    return get_application_channel_analytics()


@app.get(
    "/api/applications/analytics/stage-durations",
    response_model=ApplicationStageDurationResponse,
    summary="Get interview stage duration and time-in-stage analytics",
    tags=["Applications"]
)
async def get_application_stage_durations_endpoint():
    """Calculate average dwell time and duration metrics across application pipeline stages."""
    return get_application_stage_durations()


@app.post(
    "/api/applications/compare-offers",
    response_model=OfferComparisonResponse,
    summary="Compare job offer compensation packages side-by-side",
    tags=["Applications"]
)
async def compare_offers_endpoint(payload: OfferComparisonRequest):
    """Compare multiple job offers evaluating base salary, equity, bonus, and total first-year compensation."""
    offers_dicts = [o.model_dump() for o in payload.offers]
    return compare_offer_packages(offers_dicts)


@app.get(
    "/api/applications/export/portfolio",
    response_model=ApplicationPortfolioResponse,
    summary="Export candidate application history and portfolio archive",
    tags=["Applications"]
)
async def export_application_portfolio_endpoint():
    """Generate structured portfolio summary of all applications, interview counts, and timeline activity."""
    return export_application_portfolio()


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
    "/api/applications/reminders",
    response_model=List[JobApplication],
    summary="List pending application follow-up reminders",
    tags=["Applications"]
)
async def get_application_reminders_endpoint():
    """Retrieve all submitted job applications with pending follow-up reminders, sorted chronologically."""
    return get_upcoming_reminders()


@app.patch(
    "/api/applications/bulk/status",
    response_model=BulkApplicationStatusUpdateResponse,
    summary="Bulk update status across multiple applications",
    tags=["Applications"]
)
async def bulk_update_applications_status_endpoint(payload: BulkApplicationStatusUpdateRequest):
    """Update tracking status across multiple job applications simultaneously."""
    success, message, count, updated_ids = bulk_update_application_status(payload.job_ids, payload.status)
    if not success:
        if "Invalid status" in message or "cannot be empty" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)

    return BulkApplicationStatusUpdateResponse(
        updated_count=count,
        updated_job_ids=updated_ids,
        status=payload.status,
        message=message
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


@app.get(
    "/api/applications/{job_id}/timeline",
    response_model=ApplicationTimelineResponse,
    summary="Get application audit event history and timeline",
    tags=["Applications"]
)
async def get_application_timeline_endpoint(job_id: str):
    """Retrieve the chronological lifecycle audit trail of events for an application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")

    timeline = get_application_timeline(clean_id)
    if timeline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No application found for Job ID '{clean_id}'."
        )
    return ApplicationTimelineResponse(
        job_id=clean_id,
        total_events=len(timeline),
        timeline=timeline
    )


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


@app.post(
    "/api/applications/{job_id}/rejection-reason",
    response_model=RejectionReasonLogResponse,
    summary="Log rejection feedback and category for an application",
    tags=["Applications"]
)
async def log_application_rejection_reason_endpoint(job_id: str, payload: RejectionReasonLogRequest):
    """Log structured feedback and reason categorization for a rejected job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = log_application_rejection_reason(
        clean_id, payload.reason, payload.category or "other"
    )
    if not success or not updated_app:
        if "cannot be empty" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    return {
        "job_id": updated_app.get("job_id", clean_id),
        "status": updated_app.get("status", "Rejected"),
        "rejection_reason": updated_app.get("rejection_reason", payload.reason),
        "rejection_category": updated_app.get("rejection_category", payload.category or "other"),
        "logged_at": updated_app.get("rejection_logged_at", ""),
        "message": message
    }


@app.post(
    "/api/applications/{job_id}/interview-feedback",
    response_model=InterviewDifficultyFeedbackResponse,
    summary="Log interview difficulty score and notes for an application",
    tags=["Applications"]
)
async def log_interview_difficulty_feedback_endpoint(job_id: str, payload: InterviewDifficultyFeedbackRequest):
    """Log interview difficulty score (1-5) and qualitative candidate feedback notes."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = record_interview_difficulty_feedback(
        clean_id, payload.difficulty_score, payload.feedback_notes
    )
    if not success or not updated_app:
        if "between 1 and 5" in message or "cannot be empty" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    return {
        "job_id": updated_app.get("job_id", clean_id),
        "difficulty_score": updated_app.get("interview_difficulty_score", payload.difficulty_score),
        "difficulty_tier": updated_app.get("interview_difficulty_tier", "Moderate"),
        "feedback_notes": updated_app.get("interview_feedback_notes", payload.feedback_notes),
        "logged_at": updated_app.get("interview_feedback_logged_at", ""),
        "message": message
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


@app.put(
    "/api/applications/{job_id}/tags",
    summary="Update application custom tags",
    tags=["Applications"]
)
async def update_application_tags_endpoint(job_id: str, payload: ApplicationTagsUpdateRequest):
    """Update custom organization tags and labels for a submitted application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = update_application_tags(clean_id, payload.tags)
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


@app.patch(
    "/api/applications/{job_id}/reminder",
    summary="Set or clear application follow-up reminder",
    tags=["Applications"]
)
async def update_application_reminder_endpoint(job_id: str, payload: ApplicationReminderUpdateRequest):
    """Set or clear a follow-up reminder date or deadline for an application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID cannot be empty."
        )
    success, message, updated_app = update_application_reminder(clean_id, payload.reminder_date)
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


@app.post(
    "/api/applications/{job_id}/interviews",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule or log an interview stage for an application",
    tags=["Applications"]
)
async def add_application_interview_endpoint(job_id: str, payload: ApplicationInterviewCreateRequest):
    """Schedule or log a new interview stage or round for a submitted job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, stage_entry = add_application_interview(
        job_id=clean_id,
        round_name=payload.round_name,
        scheduled_at=payload.scheduled_at,
        interviewer=payload.interviewer,
        meeting_link=payload.meeting_link,
        notes=payload.notes,
        status=payload.status or "Scheduled"
    )
    if not success:
        if "cannot be empty" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "interview": stage_entry
    }


@app.patch(
    "/api/applications/{job_id}/interviews/{stage_id}",
    response_model=dict,
    summary="Update interview round status or reschedule time",
    tags=["Applications"]
)
async def update_application_interview_endpoint(
    job_id: str,
    stage_id: str,
    payload: InterviewStageUpdateRequest
):
    """Update status, rescheduled date/time, or debrief notes for an interview stage."""
    clean_id = (job_id or "").strip()
    clean_stage = (stage_id or "").strip()
    if not clean_id or not clean_stage:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID and Stage ID cannot be empty.")
    success, message, stage_entry = update_application_interview(
        job_id=clean_id,
        stage_id=clean_stage,
        status=payload.status,
        scheduled_at=payload.scheduled_at,
        notes=payload.notes
    )
    if not success:
        if "Invalid interview status" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "interview": stage_entry
    }


@app.post(
    "/api/applications/{job_id}/interview-questions",
    response_model=InterviewQuestionEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Log an interview preparation question",
    tags=["Applications"]
)
async def add_interview_question_endpoint(job_id: str, payload: InterviewQuestionCreateRequest):
    """Record an interview question, answer outline, and round difficulty."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, entry = add_interview_question(
        job_id=clean_id,
        question=payload.question,
        answer_notes=payload.answer_notes,
        difficulty=payload.difficulty or "Medium",
        round_type=payload.round_type or "Technical"
    )
    if not success or not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return entry


@app.get(
    "/api/applications/{job_id}/interview-questions",
    response_model=InterviewQuestionsResponse,
    summary="Get logged interview questions for an application",
    tags=["Applications"]
)
async def get_interview_questions_endpoint(job_id: str):
    """Retrieve all interview questions logged for this application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, questions = get_interview_questions(clean_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return InterviewQuestionsResponse(
        status="success",
        job_id=clean_id,
        total=len(questions),
        questions=questions
    )


@app.patch(
    "/api/applications/{job_id}/archive",
    response_model=dict,
    summary="Archive or restore an application",
    tags=["Applications"]
)
async def archive_application_endpoint(job_id: str, payload: ApplicationArchiveUpdateRequest):
    """Move an application to archive or restore it to the active pipeline."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_app = archive_application(clean_id, archived=payload.archived)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.patch(
    "/api/applications/{job_id}/priority",
    response_model=dict,
    summary="Update application priority level or star status",
    tags=["Applications"]
)
async def update_application_priority_endpoint(job_id: str, payload: ApplicationPriorityUpdateRequest):
    """Set application priority rating (Low, Medium, High, Urgent) or toggle starred status."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_app = update_application_priority(
        clean_id,
        priority=payload.priority,
        is_favorite=payload.is_favorite
    )
    if not success:
        if "Invalid priority" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.patch(
    "/api/applications/{job_id}/deadlines",
    response_model=dict,
    summary="Update application decision deadlines and expected feedback date",
    tags=["Applications"]
)
async def update_application_deadlines_endpoint(job_id: str, payload: ApplicationDeadlinesUpdateRequest):
    """Set or clear target decision deadline or expected response milestone for an application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_app = update_application_deadlines(
        clean_id,
        decision_deadline=payload.decision_deadline,
        expected_feedback_date=payload.expected_feedback_date
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.patch(
    "/api/applications/{job_id}/source",
    response_model=dict,
    summary="Update application submission channel",
    tags=["Applications"]
)
async def update_application_source_endpoint(job_id: str, payload: ApplicationSourceUpdateRequest):
    """Update application submission channel or referral source."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_app = update_application_source(clean_id, payload.source)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.post(
    "/api/applications/{job_id}/auto-followup",
    response_model=AutoFollowupResponse,
    summary="Automatically schedule optimal application follow-up date",
    tags=["Applications"]
)
async def auto_followup_endpoint(job_id: str, payload: AutoFollowupRequest):
    """Compute future business day milestone and update application reminder deadline."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, date_str, target_app = calculate_and_set_auto_followup(
        job_id=clean_id,
        business_days_ahead=payload.business_days_ahead or 5,
        reason=payload.reason
    )
    if not success or not target_app or not date_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return AutoFollowupResponse(
        status="success",
        message=message,
        job_id=clean_id,
        calculated_followup_date=date_str,
        application=target_app
    )


@app.post(
    "/api/applications/{job_id}/attachments",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Add document attachment metadata to an application",
    tags=["Applications"]
)
async def add_application_attachment_endpoint(job_id: str, payload: ApplicationAttachmentCreateRequest):
    """Associate document attachment metadata with an existing job application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, att_entry = add_application_attachment(
        job_id=clean_id,
        file_name=payload.file_name,
        file_type=payload.file_type,
        file_size_bytes=payload.file_size_bytes,
        notes=payload.notes
    )
    if not success:
        if "cannot be empty" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "attachment": att_entry
    }


@app.post(
    "/api/applications/{job_id}/outcome",
    response_model=dict,
    summary="Record interview outcome or rejection reason",
    tags=["Applications"]
)
async def update_application_outcome_endpoint(job_id: str, payload: ApplicationOutcomeUpdateRequest):
    """Record final application conclusion outcome and debrief notes."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_app = update_application_outcome(
        job_id=clean_id,
        outcome=payload.outcome,
        reason_code=payload.reason_code,
        feedback_notes=payload.feedback_notes
    )
    if not success:
        if "cannot be empty" in message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "application": updated_app
    }


@app.get(
    "/api/applications/{job_id}/followup-template",
    response_model=FollowupTemplateResponse,
    summary="Generate personalized follow-up reminder email draft",
    tags=["Applications"]
)
async def get_application_followup_template_endpoint(
    job_id: str,
    template_type: str = Query("status_inquiry", description="Template style: status_inquiry, interview_followup, or thank_you")
):
    """Generate a polite, tailored email template for following up with hiring teams."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, draft = generate_followup_template(clean_id, template_type=template_type)
    if not success or not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return draft


@app.get(
    "/api/applications/{job_id}/rollback-available",
    summary="Check if a rollback snapshot is available for an application",
    tags=["Applications"]
)
async def check_rollback_available_endpoint(job_id: str):
    """Check whether an undo/rollback snapshot exists in the transaction buffer for this application."""
    clean_id = (job_id or "").strip()
    return {
        "status": "success",
        "job_id": clean_id,
        "rollback_available": has_rollback_snapshot(clean_id)
    }


@app.post(
    "/api/applications/{job_id}/rollback",
    response_model=ApplicationRollbackResponse,
    summary="Rollback application state to previous snapshot",
    tags=["Applications"]
)
async def rollback_application_endpoint(job_id: str):
    """Revert the most recent status or notes mutation from the in-memory rollback buffer."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, restored = rollback_application_state(clean_id)
    if not success or not restored:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return {
        "status": "success",
        "job_id": clean_id,
        "message": message,
        "restored_status": restored.get("status"),
        "restored_notes": restored.get("notes"),
        "snapshot_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
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


@app.post(
    "/api/bookmarks/bulk/delete",
    response_model=BulkBookmarkDeleteResponse,
    summary="Bulk remove multiple job bookmarks",
    tags=["Bookmarks"]
)
async def bulk_delete_bookmarks_endpoint(payload: BulkBookmarkDeleteRequest):
    """Remove multiple saved bookmarks simultaneously."""
    success, message, deleted_cnt, rem_cnt = bulk_remove_bookmarks(payload.job_ids)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return BulkBookmarkDeleteResponse(
        status="success",
        deleted_count=deleted_cnt,
        remaining_count=rem_cnt,
        message=message
    )


@app.get(
    "/api/bookmarks/export",
    summary="Export bookmarks as JSON or CSV",
    tags=["Bookmarks"]
)
async def export_bookmarks_endpoint(
    format: str = Query("json", description="Export format: json or csv")
):
    """Export all saved job bookmarks in JSON or CSV format for spreadsheet tracking."""
    fmt = (format or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format. Supported export formats are 'json' and 'csv'."
        )

    bookmarks = get_bookmarked_jobs()
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Job ID", "Job Title", "Company", "Location", "Category", "Saved Date", "Notes"])
        for bm in bookmarks:
            writer.writerow([
                bm.get("job_id", ""),
                bm.get("job_title", ""),
                bm.get("company", ""),
                bm.get("location", ""),
                bm.get("category", "") or "",
                bm.get("saved_date", ""),
                bm.get("notes") or ""
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=job_bookmarks.csv"}
        )

    return JSONResponse(
        content=bookmarks,
        headers={"Content-Disposition": "attachment; filename=job_bookmarks.json"}
    )


@app.get(
    "/api/bookmarks/categories",
    response_model=BookmarkCategoriesResponse,
    summary="Get bookmark categories and folder counts",
    tags=["Bookmarks"]
)
async def get_bookmark_categories_endpoint():
    """Retrieve user bookmark categories with counts of saved jobs per folder."""
    return get_bookmark_categories()


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


@app.patch(
    "/api/bookmarks/{job_id}",
    response_model=dict,
    summary="Update notes or category on a saved bookmark",
    tags=["Bookmarks"]
)
async def update_job_bookmark_endpoint(job_id: str, payload: BookmarkUpdateRequest):
    """Update personal candidate notes or category folder for a bookmarked job posting."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job ID cannot be empty.")
    success, message, updated_bm = update_job_bookmark(clean_id, notes=payload.notes, category=payload.category)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return {
        "status": "success",
        "message": message,
        "bookmark": updated_bm
    }


@app.get(
    "/api/admin/backup",
    response_model=DatabaseBackupSnapshot,
    summary="Export a complete database snapshot backup",
    tags=["Database Admin"]
)
async def create_database_backup_endpoint():
    """Export a complete structured backup snapshot across all collections."""
    return create_database_backup()


@app.post(
    "/api/admin/restore",
    response_model=DatabaseRestoreResponse,
    summary="Restore database collections from a snapshot backup",
    tags=["Database Admin"]
)
async def restore_database_backup_endpoint(payload: DatabaseBackupSnapshot):
    """Restore database collections atomically from an uploaded backup snapshot."""
    success, message, counts = restore_database_backup(payload.model_dump())
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return DatabaseRestoreResponse(
        status="success",
        message=message,
        restored_counts=counts
    )


@app.get(
    "/api/admin/db/health",
    response_model=DatabaseHealthResponse,
    summary="Check database integrity and collection health",
    tags=["Database Admin"]
)
async def check_database_health_endpoint():
    """Perform integrity diagnostic check across all local JSON collections."""
    return check_database_health()


@app.post(
    "/api/admin/db/compact",
    response_model=DatabaseCompactionResponse,
    summary="Compact database and purge orphan references",
    tags=["Database Admin"]
)
@app.post("/api/db/compact", response_model=DatabaseCompactionResponse, include_in_schema=False)
async def compact_database_endpoint():
    """Purge orphaned bookmarks and deduplicate search history."""
    return compact_database()


@app.get(
    "/api/admin/db/checksums",
    response_model=DatabaseChecksumResponse,
    summary="Compute cryptographic SHA-256 integrity checksums for all database collections",
    tags=["Database Admin"]
)
async def compute_database_checksums_endpoint():
    """Generate SHA-256 integrity manifest and storage statistics across persistent storage files."""
    return compute_database_checksums()


@app.get(
    "/api/admin/db/integrity/report",
    response_model=DatabaseIntegrityReportResponse,
    summary="Generate deep database integrity and structural validation report",
    tags=["Database Admin"]
)
@app.get("/api/db/integrity/report", response_model=DatabaseIntegrityReportResponse, include_in_schema=False)
async def generate_database_integrity_report_endpoint():
    """Perform comprehensive structural validation, foreign key checks, and schema integrity scan."""
    return generate_database_integrity_report()


@app.post(
    "/api/admin/db/vacuum",
    response_model=DatabaseVacuumResponse,
    summary="Vacuum and canonicalize JSON storage collections",
    tags=["Database Admin"]
)
async def vacuum_database_endpoint():
    """Perform recursive JSON key ordering, whitespace trimming, and space reclaiming vacuum on database files."""
    return vacuum_database_collections()


@app.get(
    "/api/admin/system/diagnostics",
    response_model=SystemDiagnosticsResponse,
    summary="Run environment diagnostics and storage IOPS throughput profiler",
    tags=["Database Admin"]
)
@app.get("/api/db/diagnostics", response_model=SystemDiagnosticsResponse, include_in_schema=False)
async def run_system_diagnostics_endpoint():
    """Execute live system runtime diagnostics, process environment inspection, and storage I/O speed benchmark."""
    return run_system_storage_diagnostics()





@app.post(
    "/api/admin/db/repair",
    response_model=DatabaseRepairResponse,
    summary="Repair database records and enforce missing schema defaults",
    tags=["Database Admin"]
)
async def repair_database_endpoint():
    """Verify and patch missing default attributes across applications and bookmarks collections."""
    return repair_database_records()


@app.get(
    "/api/admin/db/audit-log",
    response_model=DatabaseAuditLogResponse,
    summary="Retrieve database operations audit trail log",
    tags=["Database Admin"]
)
async def get_database_audit_log_endpoint(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of audit events to return"),
    entity: Optional[str] = Query(None, description="Optional filter by entity (application, bookmark, system)")
):
    """Retrieve chronologically ordered audit logs of transactions and multi-entity operations."""
    return get_database_audit_log(limit=limit, entity=entity)


@app.get(
    "/api/admin/db/storage",
    response_model=DatabaseStorageMetricsResponse,
    summary="Get database storage footprint and collection metrics",
    tags=["Database Admin"]
)
async def get_database_storage_metrics_endpoint():
    """Compute disk storage size, record counts, and density metrics across all collections."""
    return get_database_storage_metrics()


@app.get(
    "/api/admin/db/benchmark",
    response_model=DatabaseBenchmarkResponse,
    summary="Benchmark database read latencies and collection performance",
    tags=["Database Admin"]
)
async def benchmark_database_endpoint():
    """Execute high-precision read latency benchmarks across all JSON database collections."""
    return benchmark_database_performance()


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
