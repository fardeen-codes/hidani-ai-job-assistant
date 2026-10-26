"""
Sub-Agent 3: Applier Agent.
Handles the execution and confirmation of job applications,
ensuring non-duplicate applications, database synchronization,
and status inquiry reporting.
"""

import logging
from typing import Any, Dict, List, Optional

from app.schemas import JobApplication
from app.services.db_service import apply_to_job, get_application_status, get_job_by_id

logger = logging.getLogger("ApplierAgent")


def apply_for_job(job_id: str) -> Dict[str, Any]:
    """
    Sub-Agent 3 execution for submitting an application:
    Validates job ID, checks deduplication, records application in applications.json.
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return {
            "success": False,
            "message": "Please provide a valid Job ID (e.g., job-00001) to apply.",
            "application": None
        }

    # Verify existence
    job = get_job_by_id(clean_id)
    if not job:
        return {
            "success": False,
            "message": f"Job ID '{clean_id}' does not exist in the database. Please verify the ID from the recommended jobs.",
            "application": None
        }

    success, message, app_data = apply_to_job(clean_id)

    return {
        "success": success,
        "message": message,
        "application": app_data,
        "job": {
            "job_id": job.get("job_id"),
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "source_link": job.get("source_link")
        } if job else None
    }


def get_applications_summary(job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sub-Agent 3 execution for querying application statuses:
    Returns full list of applications or targeted status for a specific job.
    """
    apps = get_application_status(job_id)

    if job_id and not apps:
        job = get_job_by_id(job_id)
        job_name = f"for '{job.get('title')}'" if job else f"for Job ID '{job_id}'"
        return {
            "count": 0,
            "applications": [],
            "message": f"You haven't submitted an application {job_name} yet."
        }

    if not apps:
        return {
            "count": 0,
            "applications": [],
            "message": "You haven't submitted any job applications yet. Tell me what kind of roles you are interested in and I'll find matching jobs for you!"
        }

    return {
        "count": len(apps),
        "applications": apps,
        "message": f"Found {len(apps)} submitted application(s)."
    }
