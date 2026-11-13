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

from app.config import APPLICATIONS_FILE, BOOKMARKS_FILE, JOBS_FILE, SEARCH_HISTORY_FILE

logger = logging.getLogger("DBService")
_db_lock = threading.RLock()

# In-memory mtime cache and fast lookup index for static jobs database
_jobs_cache: Optional[List[Dict[str, Any]]] = None
_jobs_by_id_cache: Dict[str, Dict[str, Any]] = {}
_jobs_cache_mtime: Optional[float] = None

# In-memory mtime cache for applications database
_applications_cache: Optional[List[Dict[str, Any]]] = None
_applications_cache_mtime: Optional[float] = None

# In-memory mtime cache for bookmarks database
_bookmarks_cache: Optional[List[Dict[str, Any]]] = None
_bookmarks_cache_mtime: Optional[float] = None

# In-memory mtime cache for search history
_search_history_cache: Optional[List[Dict[str, Any]]] = None
_search_history_cache_mtime: Optional[float] = None


def invalidate_jobs_cache() -> None:
    """Clear in-memory jobs cache to force reload on next access."""
    global _jobs_cache, _jobs_by_id_cache, _jobs_cache_mtime
    with _db_lock:
        _jobs_cache = None
        _jobs_by_id_cache = {}
        _jobs_cache_mtime = None


def invalidate_applications_cache() -> None:
    """Clear in-memory applications cache to force reload on next access."""
    global _applications_cache, _applications_cache_mtime
    with _db_lock:
        _applications_cache = None
        _applications_cache_mtime = None


def invalidate_bookmarks_cache() -> None:
    """Clear in-memory bookmarks cache to force reload on next access."""
    global _bookmarks_cache, _bookmarks_cache_mtime
    with _db_lock:
        _bookmarks_cache = None
        _bookmarks_cache_mtime = None


def _atomic_write_json(file_path: Path, data: Any) -> None:
    """Atomically write JSON data to file using a temporary file to avoid corruption."""
    temp_name: Optional[str] = None
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = file_path.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            temp_name = tf.name
            json.dump(data, tf, indent=2, ensure_ascii=False)
        os.replace(temp_name, file_path)
    except OSError as exc:
        logger.warning(f"Atomic file replace failed ({exc}), falling back to direct write")
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from jobs.json with error recovery and mtime-based in-memory caching."""
    global _jobs_cache, _jobs_by_id_cache, _jobs_cache_mtime
    with _db_lock:
        if not JOBS_FILE.exists():
            _jobs_cache = []
            _jobs_by_id_cache = {}
            _jobs_cache_mtime = None
            return []
        try:
            current_mtime = JOBS_FILE.stat().st_mtime
            if _jobs_cache is not None and _jobs_cache_mtime == current_mtime:
                return _jobs_cache

            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _jobs_cache = []
                    _jobs_by_id_cache = {}
                    _jobs_cache_mtime = current_mtime
                    return []
                jobs = json.loads(content)
                _jobs_cache = jobs
                _jobs_by_id_cache = {
                    j.get("job_id", "").strip().lower(): j
                    for j in jobs
                    if j.get("job_id")
                }
                _jobs_cache_mtime = current_mtime
                return _jobs_cache
        except Exception as exc:
            logger.error(f"Error loading jobs database: {exc}")
            return _jobs_cache if _jobs_cache is not None else []


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Find a specific job by job_id with O(1) indexed lookup."""
    clean_id = (job_id or "").strip().lower()
    if not clean_id:
        return None

    load_jobs()
    with _db_lock:
        return _jobs_by_id_cache.get(clean_id)


def search_jobs(
    skills: Optional[List[str]] = None,
    designation: Optional[str] = None,
    query: Optional[str] = None,
    experience_level: Optional[str] = None,
    location: Optional[str] = None,
    company: Optional[str] = None,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search and filter jobs by designation, skills, query keyword, location, company, and experience level.
    Supports sorting by title, company, experience, or relevance score.
    Returns matched job dictionaries.
    """
    all_jobs = load_jobs()
    if not all_jobs:
        return []

    skills_set = {str(s).strip().lower() for s in (skills or []) if s and str(s).strip()}
    query_clean = (query or "").strip().lower()
    desig_clean = (designation or "").strip().lower()
    exp_clean = (experience_level or "").strip().lower()
    loc_clean = (location or "").strip().lower()
    company_clean = (company or "").strip().lower()

    scored_jobs = []

    for job in all_jobs:
        job_loc = (job.get("location") or "").lower()
        title_lower = (job.get("title") or "").lower()
        desc_lower = (job.get("description") or "").lower()
        company_lower = (job.get("company") or "").lower()
        job_exp = (job.get("experience") or "").lower()
        job_skills = {str(s).strip().lower() for s in (job.get("skills") or []) if s and str(s).strip()}

        # Strict location filter if specified (e.g. "remote")
        if loc_clean:
            if loc_clean == "remote" and "remote" not in job_loc:
                continue
            elif loc_clean != "remote" and loc_clean not in job_loc:
                continue

        # Strict company filter if specified
        if company_clean and company_clean not in company_lower:
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

    # Ordering / Sorting
    sort_clean = (sort_by or "").strip().lower()
    is_desc = (order or "asc").strip().lower() == "desc"
    if sort_clean in {"title", "company", "experience", "location"}:
        scored_jobs.sort(key=lambda x: (x.get(sort_clean) or "").lower(), reverse=is_desc)
    else:
        # Default: descending by calculated score
        scored_jobs.sort(key=lambda x: x.get("internal_score", 0), reverse=True)

    return scored_jobs[:limit] if limit is not None else scored_jobs


def _save_applications(applications: List[Dict[str, Any]]) -> None:
    """Persist applications atomically and synchronize the in-memory cache."""
    global _applications_cache, _applications_cache_mtime
    _atomic_write_json(APPLICATIONS_FILE, applications)
    _applications_cache = [dict(a) for a in applications]
    try:
        _applications_cache_mtime = APPLICATIONS_FILE.stat().st_mtime if APPLICATIONS_FILE.exists() else None
    except OSError:
        _applications_cache_mtime = None


def load_applications() -> List[Dict[str, Any]]:
    """Read all submitted applications from applications.json with mtime-based caching."""
    global _applications_cache, _applications_cache_mtime
    with _db_lock:
        if not APPLICATIONS_FILE.exists():
            _applications_cache = []
            _applications_cache_mtime = None
            return []
        try:
            current_mtime = APPLICATIONS_FILE.stat().st_mtime
            if _applications_cache is not None and _applications_cache_mtime == current_mtime:
                return [dict(a) for a in _applications_cache]

            with open(APPLICATIONS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _applications_cache = []
                    _applications_cache_mtime = current_mtime
                    return []
                apps = json.loads(content)
                _applications_cache = [dict(a) for a in apps]
                _applications_cache_mtime = current_mtime
                return [dict(a) for a in _applications_cache]
        except Exception as exc:
            logger.error(f"Error loading applications: {exc}")
            return [dict(a) for a in _applications_cache] if _applications_cache is not None else []


def apply_to_job(job_id: str, notes: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Apply to a job ID with deduplication and validation.
    Accepts an optional applicant note / cover letter remark.
    Returns: (success: bool, message: str, application_data: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    clean_notes = (notes or "").strip() or None
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
            "status": "Applied",
            "notes": clean_notes
        }

        applications.append(new_application)
        _save_applications(applications)

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

        _save_applications(remaining)

    logger.info(f"Successfully withdrew application for job {clean_id}")
    return True, f"Application for Job ID '{clean_id}' has been successfully withdrawn."


VALID_APPLICATION_STATUSES = {
    "applied": "Applied",
    "under review": "Under Review",
    "interview": "Interview",
    "offer": "Offer",
    "rejected": "Rejected",
}


def update_application_status(job_id: str, new_status: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update the status of an existing job application.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    status_key = (new_status or "").strip().lower()
    canonical_status = VALID_APPLICATION_STATUSES.get(status_key)
    if not canonical_status:
        allowed = ", ".join(VALID_APPLICATION_STATUSES.values())
        return False, f"Invalid status '{new_status}'. Allowed values: {allowed}.", None

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["status"] = canonical_status
                app["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated application status for job {clean_id} to '{canonical_status}'")
    return True, f"Application status for Job ID '{clean_id}' updated to '{canonical_status}'.", updated_app


def update_application_notes(job_id: str, notes: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update personal applicant notes or interview remarks for a submitted application.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["notes"] = (notes or "").strip() or None
                app["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated application notes for job {clean_id}")
    return True, f"Application notes for Job ID '{clean_id}' have been updated.", updated_app


def update_application_reminder(job_id: str, reminder_date: Optional[str]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Set or clear a follow-up reminder date for an application.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    clean_rem = (reminder_date or "").strip() or None

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["reminder_date"] = clean_rem
                app["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    if clean_rem:
        msg = f"Follow-up reminder for Job ID '{clean_id}' set for {clean_rem}."
    else:
        msg = f"Follow-up reminder for Job ID '{clean_id}' has been cleared."
    logger.info(msg)
    return True, msg, updated_app


def get_upcoming_reminders() -> List[Dict[str, Any]]:
    """Retrieve all submitted applications that have pending follow-up reminders, sorted chronologically."""
    apps = load_applications()
    reminders = [a for a in apps if a.get("reminder_date")]
    reminders.sort(key=lambda x: str(x.get("reminder_date", "")))
    return reminders


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


def get_application_stats() -> Dict[str, Any]:
    """
    Compute summary analytics for submitted job applications,
    including status distribution, top applied companies, and recent submissions.
    """
    apps = load_applications()
    status_counter: Counter = Counter()
    company_counter: Counter = Counter()

    for app in apps:
        status_val = (app.get("status") or "Applied").strip()
        status_counter[status_val] += 1

        company_val = (app.get("company") or "").strip()
        if company_val:
            company_counter[company_val] += 1

    top_companies = [
        {"company": company, "count": count}
        for company, count in company_counter.most_common(5)
    ]

    recent_apps = sorted(
        apps,
        key=lambda a: a.get("applied_date") or "",
        reverse=True
    )[:5]

    return {
        "total_applications": len(apps),
        "status_breakdown": dict(status_counter),
        "top_companies": top_companies,
        "recent_applications": recent_apps,
    }


def _save_bookmarks(bookmarks: List[Dict[str, Any]]) -> None:
    """Persist bookmarks atomically and synchronize the in-memory cache."""
    global _bookmarks_cache, _bookmarks_cache_mtime
    _atomic_write_json(BOOKMARKS_FILE, bookmarks)
    _bookmarks_cache = [dict(b) for b in bookmarks]
    try:
        _bookmarks_cache_mtime = BOOKMARKS_FILE.stat().st_mtime if BOOKMARKS_FILE.exists() else None
    except OSError:
        _bookmarks_cache_mtime = None


def load_bookmarks() -> List[Dict[str, Any]]:
    """Read all bookmarked jobs with mtime-based in-memory caching."""
    global _bookmarks_cache, _bookmarks_cache_mtime
    with _db_lock:
        if not BOOKMARKS_FILE.exists():
            _bookmarks_cache = []
            _bookmarks_cache_mtime = None
            return []
        try:
            current_mtime = BOOKMARKS_FILE.stat().st_mtime
            if _bookmarks_cache is not None and _bookmarks_cache_mtime == current_mtime:
                return [dict(b) for b in _bookmarks_cache]

            with open(BOOKMARKS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _bookmarks_cache = []
                    _bookmarks_cache_mtime = current_mtime
                    return []
                bms = json.loads(content)
                _bookmarks_cache = [dict(b) for b in bms]
                _bookmarks_cache_mtime = current_mtime
                return [dict(b) for b in _bookmarks_cache]
        except Exception as exc:
            logger.error(f"Error loading bookmarks: {exc}")
            return [dict(b) for b in _bookmarks_cache] if _bookmarks_cache is not None else []


def save_job_bookmark(job_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Bookmark / save a job posting for later review.
    Returns: (success: bool, message: str, bookmark_record: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    job = get_job_by_id(clean_id)
    if not job:
        return False, f"Job with ID '{clean_id}' was not found in the database.", None

    with _db_lock:
        bookmarks = load_bookmarks()
        for b in bookmarks:
            if b.get("job_id", "").strip().lower() == clean_id.lower():
                return False, f"Job '{job.get('title')}' (ID: {clean_id}) is already in your saved bookmarks.", b

        new_bm = {
            "job_id": job.get("job_id"),
            "job_title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "saved_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        bookmarks.append(new_bm)
        _save_bookmarks(bookmarks)

    logger.info(f"Bookmarked job {clean_id} ({job.get('title')})")
    return True, f"Job '{job.get('title')}' at {job.get('company')} has been saved to your bookmarks.", new_bm


def get_bookmarked_jobs() -> List[Dict[str, Any]]:
    """Retrieve all saved job bookmarks."""
    return load_bookmarks()


def remove_job_bookmark(job_id: str) -> Tuple[bool, str]:
    """
    Remove / unsave a job from bookmarks by Job ID.
    Returns: (success: bool, message: str)
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty."

    with _db_lock:
        bookmarks = load_bookmarks()
        initial_len = len(bookmarks)
        remaining = [b for b in bookmarks if b.get("job_id", "").strip().lower() != clean_id.lower()]

        if len(remaining) == initial_len:
            return False, f"No bookmark found for Job ID '{clean_id}'."

        _save_bookmarks(remaining)

    logger.info(f"Removed bookmark for job {clean_id}")
    return True, f"Job ID '{clean_id}' removed from your saved bookmarks."


def _save_search_history(history: List[Dict[str, Any]]) -> None:
    """Persist search history atomically and synchronize in-memory cache."""
    global _search_history_cache, _search_history_cache_mtime
    _atomic_write_json(SEARCH_HISTORY_FILE, history)
    _search_history_cache = [dict(h) for h in history]
    try:
        _search_history_cache_mtime = SEARCH_HISTORY_FILE.stat().st_mtime if SEARCH_HISTORY_FILE.exists() else None
    except OSError:
        _search_history_cache_mtime = None


def load_search_history() -> List[Dict[str, Any]]:
    """Read search history with mtime-based in-memory caching."""
    global _search_history_cache, _search_history_cache_mtime
    with _db_lock:
        if not SEARCH_HISTORY_FILE.exists():
            _search_history_cache = []
            _search_history_cache_mtime = None
            return []
        try:
            current_mtime = SEARCH_HISTORY_FILE.stat().st_mtime
            if _search_history_cache is not None and _search_history_cache_mtime == current_mtime:
                return [dict(h) for h in _search_history_cache]

            with open(SEARCH_HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _search_history_cache = []
                    _search_history_cache_mtime = current_mtime
                    return []
                entries = json.loads(content)
                _search_history_cache = [dict(h) for h in entries]
                _search_history_cache_mtime = current_mtime
                return [dict(h) for h in _search_history_cache]
        except Exception as exc:
            logger.error(f"Error loading search history: {exc}")
            return [dict(h) for h in _search_history_cache] if _search_history_cache is not None else []


def record_search_query(
    query: Optional[str] = None,
    location: Optional[str] = None,
    experience: Optional[str] = None,
    skills: Optional[List[str]] = None,
    results_count: int = 0
) -> None:
    """Record a non-empty search query into candidate search history (keeps latest 50)."""
    clean_q = (query or "").strip()
    clean_loc = (location or "").strip()
    clean_exp = (experience or "").strip()
    clean_skills = [s.strip() for s in (skills or []) if s.strip()]

    # If completely empty search query and filters, do not record
    if not clean_q and not clean_loc and not clean_exp and not clean_skills:
        return

    entry = {
        "query": clean_q or None,
        "location": clean_loc or None,
        "experience": clean_exp or None,
        "skills": clean_skills,
        "results_count": max(0, results_count),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with _db_lock:
        history = load_search_history()
        # Avoid duplicate if identical to latest entry
        if history:
            latest = history[0]
            if (
                latest.get("query") == entry["query"]
                and latest.get("location") == entry["location"]
                and latest.get("experience") == entry["experience"]
                and latest.get("skills") == entry["skills"]
            ):
                return
        history.insert(0, entry)
        if len(history) > 50:
            history = history[:50]
        _save_search_history(history)


def get_recent_searches(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve the most recent candidate searches up to limit."""
    history = load_search_history()
    limit = max(1, min(limit, 50))
    return history[:limit]


def clear_search_history() -> bool:
    """Clear all recorded search history."""
    with _db_lock:
        _save_search_history([])
    return True

