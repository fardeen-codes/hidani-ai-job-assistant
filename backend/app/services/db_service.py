"""
Database Service for Local JSON Storage.
Handles reading and writing jobs.json and applications.json with deduplication,
thread-safe atomic writes, and query filtering.
"""

import json
import logging
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    APPLICATIONS_FILE,
    BOOKMARKS_FILE,
    JOBS_FILE,
    SAVED_SEARCHES_FILE,
    SEARCH_HISTORY_FILE,
)

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

# In-memory mtime cache for saved searches
_saved_searches_cache: Optional[List[Dict[str, Any]]] = None
_saved_searches_cache_mtime: Optional[float] = None


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


JOB_CATEGORIES: Dict[str, List[str]] = {
    "Engineering": ["engineer", "developer", "backend", "frontend", "full stack", "fullstack", "devops", "software", "architect", "embedded", "qa", "quality", "programmer", "tech lead", "cloud", "security", "infrastructure"],
    "Data & AI": ["data", "machine learning", "ai", "llm", "deep learning", "analytics", "bi", "scientist", "nlp", "computer vision", "quantization"],
    "Product & Design": ["product manager", "head of product", "product owner", "ux", "ui", "designer", "design", "graphic"],
    "Finance & Operations": ["finance", "financial", "accounting", "accountant", "controller", "operations", "audit", "tax", "legal", "hr", "recruiting", "talent", "payroll"],
    "Marketing & Sales": ["marketing", "sales", "business development", "bdr", "sdr", "account executive", "customer success", "sem", "seo", "growth", "media buyer", "advertising"],
}


def classify_job_category(title: str, skills: Optional[List[str]] = None) -> str:
    """Classify a job posting into an industry domain based on its title and required skills."""
    text = (title or "").lower()
    skills_text = " ".join([str(s).lower() for s in (skills or [])])
    combined = f"{text} {skills_text}"

    for cat in ["Data & AI", "Product & Design", "Finance & Operations", "Marketing & Sales", "Engineering"]:
        for kw in JOB_CATEGORIES[cat]:
            if kw in combined:
                return cat
    return "Other"


SALARY_RANGE_REGEX = re.compile(
    r"(?P<curr1>[\$\€\£]|USD|EUR|GBP)?\s*"
    r"(?P<min>\d{1,3}(?:,\d{3})+|\d+)\s*(?P<mink>[kK])?\s*"
    r"(?:-|–|—|to)\s*"
    r"(?P<curr2>[\$\€\£]|USD|EUR|GBP)?\s*"
    r"(?P<max>\d{1,3}(?:,\d{3})+|\d+)\s*(?P<maxk>[kK])?\s*"
    r"(?P<curr3>USD|EUR|GBP)?\s*"
    r"(?P<period>(?:/\s*(?:year|yr|annum|month|mo|hour|hr)|per\s+(?:year|annum|month|hour)|hourly|annually|monthly))?",
    re.IGNORECASE
)

SALARY_SINGLE_REGEX = re.compile(
    r"(?:salary|compensation|pay|rate|offering)?[:\s]*"
    r"(?P<curr>[\$\€\£]|USD|EUR|GBP)\s*"
    r"(?P<val>\d{1,3}(?:,\d{3})+|\d+)\s*(?P<k>[kK])?\s*"
    r"(?P<period>(?:/\s*(?:year|yr|annum|month|mo|hour|hr)|per\s+(?:year|annum|month|hour)|hourly|annually|monthly))?",
    re.IGNORECASE
)

CURRENCY_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
}


def extract_salary_info(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract structured salary details (currency, range, period) from job descriptions or titles."""
    if not text:
        return None

    clean_text = str(text)
    m = SALARY_RANGE_REGEX.search(clean_text)
    if m:
        curr = m.group("curr1") or m.group("curr2") or m.group("curr3")
        if curr:
            norm_curr = CURRENCY_MAP.get(curr, curr.upper())
            min_val = float(m.group("min").replace(",", ""))
            max_val = float(m.group("max").replace(",", ""))
            if m.group("mink") or (m.group("maxk") and min_val < 1000):
                min_val *= 1000.0
            if m.group("maxk"):
                max_val *= 1000.0
            p_text = (m.group("period") or "").lower()
            period = "hour" if any(w in p_text for w in ["hour", "hr"]) else ("month" if any(w in p_text for w in ["month", "mo"]) else "year")
            return {
                "currency": norm_curr,
                "min_salary": min_val,
                "max_salary": max_val,
                "period": period,
                "raw_text": m.group(0).strip()
            }

    m2 = SALARY_SINGLE_REGEX.search(clean_text)
    if m2:
        curr = m2.group("curr")
        norm_curr = CURRENCY_MAP.get(curr, curr.upper())
        val = float(m2.group("val").replace(",", ""))
        if m2.group("k") or val < 1000:
            if m2.group("k"):
                val *= 1000.0
        p_text = (m2.group("period") or "").lower()
        period = "hour" if any(w in p_text for w in ["hour", "hr"]) else ("month" if any(w in p_text for w in ["month", "mo"]) else "year")
        return {
            "currency": norm_curr,
            "min_salary": val,
            "max_salary": val,
            "period": period,
            "raw_text": m2.group(0).strip()
        }

    return None


def compute_job_freshness(job: Dict[str, Any]) -> str:
    """Determine freshness category (New, Recent, Archive) based on posting date or deterministic hash."""
    posted = job.get("posted_date") or job.get("created_at") or job.get("date")
    if posted:
        try:
            clean_date = str(posted).split("T")[0].strip()
            dt = datetime.strptime(clean_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (datetime.now(timezone.utc) - dt).days
            if delta_days <= 7:
                return "New"
            elif delta_days <= 30:
                return "Recent"
            else:
                return "Archive"
        except Exception:
            pass

    jid = str(job.get("job_id", ""))
    num = sum(ord(c) for c in jid)
    mod = num % 10
    if mod <= 2:
        return "New"
    elif mod <= 7:
        return "Recent"
    return "Archive"


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
                for j in jobs:
                    if "category" not in j:
                        j["category"] = classify_job_category(j.get("title", ""), j.get("skills", []))
                    if "salary" not in j:
                        j["salary"] = extract_salary_info(f"{j.get('description', '')} {j.get('title', '')}")
                    if "freshness" not in j:
                        j["freshness"] = compute_job_freshness(j)
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
    category: Optional[str] = None,
    freshness: Optional[str] = None,
    has_salary: Optional[bool] = None,
    min_salary: Optional[float] = None,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search and filter jobs by designation, skills, query keyword, location, company, category, and experience level.
    Supports sorting by title, company, experience, salary, or relevance score.
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
    cat_clean = (category or "").strip().lower()
    fresh_clean = (freshness or "").strip().lower()

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

        # Strict category filter if specified
        if cat_clean:
            job_cat = (job.get("category") or classify_job_category(job.get("title", ""), job.get("skills", []))).lower()
            if cat_clean not in job_cat:
                continue

        # Strict freshness filter if specified
        if fresh_clean:
            job_fresh = (job.get("freshness") or compute_job_freshness(job)).lower()
            if fresh_clean != job_fresh:
                continue

        # Strict salary presence filter
        if has_salary is not None:
            job_sal = job.get("salary")
            if has_salary and not job_sal:
                continue
            if not has_salary and job_sal:
                continue

        # Strict minimum salary filter
        if min_salary is not None:
            job_sal = job.get("salary")
            if not job_sal:
                continue
            eff_sal = job_sal.get("max_salary") or job_sal.get("min_salary") or 0.0
            p = job_sal.get("period", "year")
            if p == "hour":
                eff_sal *= 2000.0
            elif p == "month":
                eff_sal *= 12.0
            if eff_sal < min_salary:
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
    elif sort_clean == "salary":
        def get_sal_val(x):
            s = x.get("salary")
            if not s:
                return -float("inf") if is_desc else float("inf")
            val = s.get("max_salary") or s.get("min_salary") or 0.0
            p = s.get("period", "year")
            if p == "hour":
                val *= 2000.0
            elif p == "month":
                val *= 12.0
            return val
        scored_jobs.sort(key=get_sal_val, reverse=is_desc)
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


def _record_app_event(app: Dict[str, Any], event_type: str, details: str) -> None:
    """Append a timestamped audit event to an application's lifecycle timeline."""
    if "timeline" not in app or not isinstance(app["timeline"], list):
        app["timeline"] = []
    app["timeline"].append({
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "details": details
    })


def apply_to_job(job_id: str, notes: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Apply to a job ID with deduplication and validation.
    Accepts an optional applicant note / cover letter remark.
    Records initial submission event in application timeline.
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
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        new_application = {
            "job_id": job.get("job_id"),
            "job_title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "applied_date": now_ts,
            "status": "Applied",
            "notes": clean_notes,
            "timeline": [
                {
                    "event_type": "SUBMITTED",
                    "timestamp": now_ts,
                    "details": "Application submitted" + (f" with cover note: {clean_notes[:40]}..." if clean_notes else "")
                }
            ]
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
                old_status = app.get("status", "Applied")
                app["status"] = canonical_status
                app["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                _record_app_event(app, "STATUS_CHANGED", f"Status changed from '{old_status}' to '{canonical_status}'")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated application status for job {clean_id} to '{canonical_status}'")
    return True, f"Application status for Job ID '{clean_id}' updated to '{canonical_status}'.", updated_app


def bulk_update_application_status(
    job_ids: List[str],
    new_status: str
) -> Tuple[bool, str, int, List[str]]:
    """
    Update tracking status for multiple applications atomically.
    Returns: (success: bool, message: str, updated_count: int, updated_job_ids: List[str])
    """
    if not job_ids:
        return False, "Job IDs list cannot be empty.", 0, []

    status_key = (new_status or "").strip().lower()
    canonical_status = VALID_APPLICATION_STATUSES.get(status_key)
    if not canonical_status:
        allowed = ", ".join(VALID_APPLICATION_STATUSES.values())
        return False, f"Invalid status '{new_status}'. Allowed values: {allowed}.", 0, []

    target_ids = {str(jid).strip().lower() for jid in job_ids if str(jid).strip()}
    updated_ids: List[str] = []

    with _db_lock:
        applications = load_applications()
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        for app in applications:
            curr_id = app.get("job_id", "").strip().lower()
            if curr_id in target_ids:
                old_status = app.get("status", "Applied")
                app["status"] = canonical_status
                app["updated_date"] = now_ts
                _record_app_event(app, "STATUS_CHANGED", f"Status updated from '{old_status}' to '{canonical_status}' via bulk action")
                updated_ids.append(app.get("job_id", ""))

        if updated_ids:
            _save_applications(applications)

    if not updated_ids:
        return False, "None of the specified Job IDs matched existing applications.", 0, []

    msg = f"Successfully updated status to '{canonical_status}' for {len(updated_ids)} application(s)."
    logger.info(msg)
    return True, msg, len(updated_ids), updated_ids


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
                _record_app_event(app, "NOTES_UPDATED", "Candidate personal notes updated")
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
                rem_detail = f"Follow-up reminder set for {clean_rem}" if clean_rem else "Follow-up reminder cleared"
                _record_app_event(app, "REMINDER_UPDATED", rem_detail)
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


def get_application_timeline(job_id: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieve the chronological audit event timeline for an application."""
    clean_id = (job_id or "").strip().lower()
    if not clean_id:
        return None
    apps = get_application_status(clean_id)
    if not apps:
        return None
    return apps[0].get("timeline") or []


def update_application_tags(
    job_id: str,
    tags: List[str]
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update organizational tags/labels on an existing application.
    Deduplicates and normalizes tags, and records event in audit timeline.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    seen = set()
    clean_tags = []
    for t in (tags or []):
        norm = str(t).strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            clean_tags.append(norm)

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["tags"] = clean_tags
                app["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                tags_desc = f"Tags updated to: {', '.join(clean_tags)}" if clean_tags else "Tags cleared"
                _record_app_event(app, "TAGS_UPDATED", tags_desc)
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    msg = f"Application tags for Job ID '{clean_id}' updated ({len(clean_tags)} tag(s))."
    logger.info(msg)
    return True, msg, updated_app


def add_application_interview(
    job_id: str,
    round_name: str,
    scheduled_at: Optional[str] = None,
    interviewer: Optional[str] = None,
    meeting_link: Optional[str] = None,
    notes: Optional[str] = None,
    status: str = "Scheduled"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Log or schedule an interview round for an existing application.
    Generates stage_id, records timeline audit event, and atomically saves record.
    Returns: (success: bool, message: str, stage_record: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    clean_round = (round_name or "").strip()
    if not clean_round:
        return False, "Interview round name cannot be empty.", None

    stage_id = f"stage_{int(datetime.now(timezone.utc).timestamp())}_{os.urandom(3).hex()}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    stage_entry = {
        "stage_id": stage_id,
        "round_name": clean_round,
        "scheduled_at": (scheduled_at or "").strip() or None,
        "interviewer": (interviewer or "").strip() or None,
        "meeting_link": (meeting_link or "").strip() or None,
        "notes": (notes or "").strip() or None,
        "status": (status or "Scheduled").strip(),
        "created_at": now_str
    }

    with _db_lock:
        applications = load_applications()
        matched = False
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                if "interviews" not in app:
                    app["interviews"] = []
                app["interviews"].append(stage_entry)
                app["updated_date"] = now_str
                event_detail = f"Interview '{clean_round}' scheduled" + (f" for {scheduled_at}" if scheduled_at else "")
                _record_app_event(app, "INTERVIEW_SCHEDULED", event_detail)
                matched = True
                break

        if not matched:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Logged interview stage '{clean_round}' for application {clean_id}")
    return True, f"Interview stage '{clean_round}' has been scheduled.", stage_entry


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
    category_counter: Counter = Counter()

    for job in jobs:
        loc = (job.get("location") or "").strip()
        if "remote" in loc.lower():
            remote_count += 1
        if loc:
            locations_counter[loc] += 1

        exp = (job.get("experience") or "").strip()
        if exp:
            exp_counter[exp] += 1

        cat = job.get("category") or classify_job_category(job.get("title", ""), job.get("skills", []))
        if cat:
            category_counter[cat] += 1

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
        "category_breakdown": dict(category_counter),
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


def _save_saved_searches(searches: List[Dict[str, Any]]) -> None:
    """Persist saved searches atomically and synchronize in-memory cache."""
    global _saved_searches_cache, _saved_searches_cache_mtime
    _atomic_write_json(SAVED_SEARCHES_FILE, searches)
    _saved_searches_cache = [dict(s) for s in searches]
    try:
        _saved_searches_cache_mtime = SAVED_SEARCHES_FILE.stat().st_mtime if SAVED_SEARCHES_FILE.exists() else None
    except OSError:
        _saved_searches_cache_mtime = None


def load_saved_searches() -> List[Dict[str, Any]]:
    """Read saved searches with mtime-based in-memory caching."""
    global _saved_searches_cache, _saved_searches_cache_mtime
    with _db_lock:
        if not SAVED_SEARCHES_FILE.exists():
            _saved_searches_cache = []
            _saved_searches_cache_mtime = None
            return []
        try:
            current_mtime = SAVED_SEARCHES_FILE.stat().st_mtime
            if _saved_searches_cache is not None and _saved_searches_cache_mtime == current_mtime:
                return [dict(s) for s in _saved_searches_cache]

            with open(SAVED_SEARCHES_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _saved_searches_cache = []
                    _saved_searches_cache_mtime = current_mtime
                    return []
                searches = json.loads(content)
                _saved_searches_cache = [dict(s) for s in searches]
                _saved_searches_cache_mtime = current_mtime
                return [dict(s) for s in _saved_searches_cache]
        except Exception as exc:
            logger.error(f"Error loading saved searches: {exc}")
            return [dict(s) for s in _saved_searches_cache] if _saved_searches_cache is not None else []


def create_saved_search(
    title: str,
    query: Optional[str] = None,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    category: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Create and persist a new saved search criteria / job alert."""
    clean_title = (title or "").strip()
    if not clean_title:
        return False, "Title cannot be empty.", None

    clean_query = (query or "").strip() or None
    clean_loc = (location or "").strip() or None
    clean_exp = (experience_level or "").strip() or None
    clean_cat = (category or "").strip() or None

    search_id = f"search_{int(datetime.now(timezone.utc).timestamp())}_{os.urandom(3).hex()}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    new_search = {
        "search_id": search_id,
        "title": clean_title,
        "query": clean_query,
        "location": clean_loc,
        "experience_level": clean_exp,
        "category": clean_cat,
        "created_date": now_str
    }

    with _db_lock:
        searches = load_saved_searches()
        searches.append(new_search)
        _save_saved_searches(searches)

    logger.info(f"Created saved search '{clean_title}' ({search_id})")
    return True, f"Saved search '{clean_title}' created successfully.", new_search


def delete_saved_search(search_id: str) -> Tuple[bool, str]:
    """Remove a saved search subscription by its unique search_id."""
    clean_id = (search_id or "").strip()
    if not clean_id:
        return False, "Search ID cannot be empty."

    with _db_lock:
        searches = load_saved_searches()
        initial_len = len(searches)
        remaining = [s for s in searches if s.get("search_id") != clean_id]

        if len(remaining) == initial_len:
            return False, f"Saved search with ID '{clean_id}' was not found."

        _save_saved_searches(remaining)

    logger.info(f"Deleted saved search {clean_id}")
    return True, f"Saved search with ID '{clean_id}' has been deleted."


def create_database_backup() -> Dict[str, Any]:
    """
    Generate a full structured database backup snapshot of all writable collections:
    applications, bookmarks, saved_searches, and search_history.
    """
    with _db_lock:
        apps = load_applications()
        bookmarks = load_bookmarks()
        saved_searches = load_saved_searches()
        search_history = load_search_history()

        now_str = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "timestamp": now_str,
            "version": "1.0",
            "applications": [dict(a) for a in apps],
            "bookmarks": [dict(b) for b in bookmarks],
            "saved_searches": [dict(s) for s in saved_searches],
            "search_history": [dict(h) for h in search_history],
        }
        logger.info(
            f"Created database backup snapshot: {len(apps)} apps, "
            f"{len(bookmarks)} bookmarks, {len(saved_searches)} saved searches, "
            f"{len(search_history)} history entries."
        )
        return snapshot


def restore_database_backup(snapshot: Dict[str, Any]) -> Tuple[bool, str, Dict[str, int]]:
    """
    Restore database collections from a structured backup snapshot.
    Validates payload format and atomically updates storage files and in-memory caches.
    Returns: (success: bool, message: str, restored_counts: Dict[str, int])
    """
    if not isinstance(snapshot, dict):
        return False, "Invalid snapshot: root must be a JSON object.", {}

    if "applications" not in snapshot or not isinstance(snapshot["applications"], list):
        return False, "Invalid snapshot: missing or invalid 'applications' list.", {}

    if "bookmarks" not in snapshot or not isinstance(snapshot["bookmarks"], list):
        return False, "Invalid snapshot: missing or invalid 'bookmarks' list.", {}

    if "saved_searches" not in snapshot or not isinstance(snapshot["saved_searches"], list):
        return False, "Invalid snapshot: missing or invalid 'saved_searches' list.", {}

    if "search_history" not in snapshot or not isinstance(snapshot["search_history"], list):
        return False, "Invalid snapshot: missing or invalid 'search_history' list.", {}

    apps = [dict(a) for a in snapshot["applications"]]
    bookmarks = [dict(b) for b in snapshot["bookmarks"]]
    saved_searches = [dict(s) for s in snapshot["saved_searches"]]
    search_history = [dict(h) for h in snapshot["search_history"]]

    with _db_lock:
        _save_applications(apps)
        _save_bookmarks(bookmarks)
        _save_saved_searches(saved_searches)
        _save_search_history(search_history)

    counts = {
        "applications": len(apps),
        "bookmarks": len(bookmarks),
        "saved_searches": len(saved_searches),
        "search_history": len(search_history),
    }

    logger.info(f"Database successfully restored from snapshot ({counts})")
    return True, "Database successfully restored from backup snapshot.", counts


def check_database_health() -> Dict[str, Any]:
    """
    Perform an integrity check on all JSON storage collections:
    jobs, applications, bookmarks, saved_searches, and search_history.
    Returns status summary with record counts and file metadata.
    """
    monitored_files = [
        ("jobs", JOBS_FILE),
        ("applications", APPLICATIONS_FILE),
        ("bookmarks", BOOKMARKS_FILE),
        ("saved_searches", SAVED_SEARCHES_FILE),
        ("search_history", SEARCH_HISTORY_FILE),
    ]

    collection_statuses = []
    healthy_count = 0

    with _db_lock:
        for name, file_path in monitored_files:
            if not file_path.exists():
                collection_statuses.append({
                    "collection": name,
                    "exists": False,
                    "record_count": 0,
                    "file_size_bytes": 0,
                    "status": "missing",
                    "last_modified": None
                })
                continue

            try:
                stat = file_path.stat()
                size_bytes = stat.st_size
                mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        data = []
                    else:
                        data = json.loads(content)
                record_count = len(data) if isinstance(data, list) else 1
                collection_statuses.append({
                    "collection": name,
                    "exists": True,
                    "record_count": record_count,
                    "file_size_bytes": size_bytes,
                    "status": "healthy",
                    "last_modified": mtime_iso
                })
                healthy_count += 1
            except Exception as exc:
                logger.error(f"Integrity check failed for collection {name}: {exc}")
                collection_statuses.append({
                    "collection": name,
                    "exists": True,
                    "record_count": 0,
                    "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                    "status": "corrupted",
                    "last_modified": None
                })

    overall_status = "healthy" if healthy_count == len(monitored_files) else ("degraded" if healthy_count > 0 else "unhealthy")
    return {
        "status": overall_status,
        "total_collections": len(monitored_files),
        "healthy_collections": healthy_count,
        "collections": collection_statuses
    }

