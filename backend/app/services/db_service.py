"""
Database Service for Local JSON Storage.
Handles reading and writing jobs.json and applications.json with deduplication,
thread-safe atomic writes, and query filtering.
"""

import json
import logging
import os
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import APPLICATIONS_FILE, JOBS_FILE

logger = logging.getLogger("DBService")
_db_lock = threading.RLock()


def _atomic_write_json(file_path: Path, data: Any) -> None:
    """Atomically write JSON data to file using a temporary file to avoid corruption."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = file_path.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name
        os.replace(temp_name, file_path)
    except OSError as exc:
        logger.warning(f"Atomic file replace failed ({exc}), falling back to direct write")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from jobs.json with error recovery."""
    with _db_lock:
        if not JOBS_FILE.exists():
            return []
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as exc:
            logger.error(f"Error loading jobs database: {exc}")
            return []


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Find a specific job by job_id (case-insensitive trim)."""
    clean_id = (job_id or "").strip().lower()
    for job in load_jobs():
        if job.get("job_id", "").strip().lower() == clean_id:
            return job
    return None


def search_jobs(
    skills: Optional[List[str]] = None,
    designation: Optional[str] = None,
    query: Optional[str] = None,
    experience_level: Optional[str] = None,
    location: Optional[str] = None,
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search and filter jobs by designation, skills, query keyword, location, and experience level.
    Returns matched job dictionaries.
    """
    all_jobs = load_jobs()
    if not all_jobs:
        return []

    skills_set = {s.strip().lower() for s in (skills or []) if s.strip()}
    query_clean = (query or "").strip().lower()
    desig_clean = (designation or "").strip().lower()
    exp_clean = (experience_level or "").strip().lower()
    loc_clean = (location or "").strip().lower()

    scored_jobs = []

    for job in all_jobs:
        job_loc = job.get("location", "").lower()
        title_lower = job.get("title", "").lower()
        desc_lower = job.get("description", "").lower()
        company_lower = job.get("company", "").lower()
        job_exp = job.get("experience", "").lower()
        job_skills = {s.lower() for s in job.get("skills", [])}

        # Strict location filter if specified (e.g. "remote")
        if loc_clean:
            if loc_clean == "remote" and "remote" not in job_loc:
                continue
            elif loc_clean != "remote" and loc_clean not in job_loc:
                continue

        score = 0.0

        # Designation match (e.g. "Full Stack Developer", "Backend", "AI")
        if desig_clean:
            if desig_clean in title_lower:
                score += 35.0
            elif any(part in title_lower for part in desig_clean.split() if len(part) > 2):
                score += 20.0

        # Skill matching score (Jaccard overlap)
        if skills_set:
            overlap = len(skills_set.intersection(job_skills))
            if overlap > 0:
                score += (overlap / len(skills_set)) * 40.0
            # Also check if any candidate skill appears in description
            desc_matches = sum(1 for s in skills_set if s in desc_lower)
            score += min(desc_matches * 3.0, 15.0)

        # Query / keyword matching (e.g., "React", "Python", "Docker")
        if query_clean:
            if query_clean in title_lower:
                score += 30.0
            elif query_clean in job_skills:
                score += 25.0
            elif query_clean in desc_lower or query_clean in company_lower:
                score += 15.0

        # Experience level match
        if exp_clean:
            if exp_clean in job_exp or job_exp in exp_clean:
                score += 15.0
            elif "mid" in exp_clean and "entry" in job_exp:
                score += 5.0

        # Baseline score
        if not skills_set and not query_clean and not desig_clean and not exp_clean:
            score = 1.0

        if score > 0:
            job_copy = dict(job)
            job_copy["internal_score"] = score
            scored_jobs.append(job_copy)

    # Sort descending by calculated score
    scored_jobs.sort(key=lambda x: x.get("internal_score", 0), reverse=True)
    return scored_jobs[:limit] if limit is not None else scored_jobs


def load_applications() -> List[Dict[str, Any]]:
    """Read all submitted applications from applications.json."""
    with _db_lock:
        if not APPLICATIONS_FILE.exists():
            return []
        try:
            with open(APPLICATIONS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as exc:
            logger.error(f"Error loading applications: {exc}")
            return []


def apply_to_job(job_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Apply to a job ID with deduplication and validation.
    Returns: (success: bool, message: str, application_data: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    # Verify job exists in jobs database
    job = get_job_by_id(clean_id)
    if not job:
        return False, f"Job with ID '{clean_id}' was not found in the database. Please verify the job ID and try again.", None

    with _db_lock:
        applications = load_applications()

        # Check for duplicate application
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                applied_date = app.get("applied_date", "earlier")
                status = app.get("status", "Applied")
                return (
                    False,
                    f"You have already applied for '{job.get('title')}' at '{job.get('company')}' on {applied_date} (Current status: {status}).",
                    app
                )

        # Create new application record
        new_application = {
            "job_id": job.get("job_id"),
            "job_title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "applied_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "Applied"
        }

        applications.append(new_application)
        _atomic_write_json(APPLICATIONS_FILE, applications)

    logger.info(f"Successfully recorded application for job {clean_id} ({job.get('title')})")
    return (
        True,
        f"Application successfully submitted for '{job.get('title')}' at {job.get('company')}! Your application ID is reference {clean_id}.",
        new_application
    )


def get_application_status(job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all applications or filter for a specific job ID."""
    apps = load_applications()
    clean_id = (job_id or "").strip().lower()
    if not clean_id:
        return apps

    return [a for a in apps if a.get("job_id", "").strip().lower() == clean_id]


def withdraw_application(job_id: str) -> Tuple[bool, str]:
    """
    Withdraw / cancel an existing job application by job_id.
    Thread-safe and atomically updates applications.json.
    Returns: (success: bool, message: str)
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty."

    with _db_lock:
        applications = load_applications()
        initial_len = len(applications)
        remaining = [
            a for a in applications
            if a.get("job_id", "").strip().lower() != clean_id.lower()
        ]

        if len(remaining) == initial_len:
            return False, f"No existing application found for Job ID '{clean_id}'."

        _atomic_write_json(APPLICATIONS_FILE, remaining)

    logger.info(f"Successfully withdrew application for job {clean_id}")
    return True, f"Application for Job ID '{clean_id}' has been successfully withdrawn."


def get_job_stats() -> Dict[str, Any]:
    """
    Compute aggregated market statistics across all available jobs,
    including remote count, top required skills, experience level distribution,
    and top locations.
    """
    jobs = load_jobs()
    total_jobs = len(jobs)
    remote_count = 0
    skills_counter: Counter = Counter()
    locations_counter: Counter = Counter()
    exp_counter: Counter = Counter()

    for job in jobs:
        loc = (job.get("location") or "").strip()
        if "remote" in loc.lower():
            remote_count += 1
        if loc:
            locations_counter[loc] += 1

        exp = (job.get("experience") or "").strip()
        if exp:
            exp_counter[exp] += 1

        for skill in job.get("skills", []):
            clean_skill = str(skill).strip()
            if clean_skill:
                skills_counter[clean_skill] += 1

    top_skills = [
        {"skill": skill, "count": count}
        for skill, count in skills_counter.most_common(10)
    ]
    top_locations = [
        {"location": loc, "count": count}
        for loc, count in locations_counter.most_common(5)
    ]

    apps_count = len(load_applications())

    return {
        "total_jobs": total_jobs,
        "remote_jobs": remote_count,
        "top_locations": top_locations,
        "top_skills": top_skills,
        "experience_breakdown": dict(exp_counter),
        "total_applications": apps_count,
    }

