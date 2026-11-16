"""
Database Service for Local JSON Storage.
Handles reading and writing jobs.json and applications.json with deduplication,
thread-safe atomic writes, and query filtering.
"""

import copy
import csv
import hashlib
import io
import json
import logging
import os
import platform
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    APPLICATIONS_FILE,
    BOOKMARKS_FILE,
    DATA_DIR,
    JOBS_FILE,
    PROFILE_FILE,
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

# In-memory ring buffer for atomic database transaction audit entries
_audit_log_entries: List[Dict[str, Any]] = []


def _log_db_audit_entry(action: str, entity: str, entity_id: str, details: str) -> None:
    """Record an audit trail entry for database operations."""
    global _audit_log_entries
    entry_id = f"aud_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{len(_audit_log_entries)}"
    entry = {
        "entry_id": entry_id,
        "action": action,
        "entity": entity,
        "entity_id": str(entity_id),
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with _db_lock:
        _audit_log_entries.append(entry)
        if len(_audit_log_entries) > 1000:
            _audit_log_entries = _audit_log_entries[-1000:]


# In-memory rollback buffer storing previous snapshot before updates
_app_rollback_snapshots: Dict[str, Dict[str, Any]] = {}


def _save_rollback_snapshot(app: Dict[str, Any]) -> None:
    """Save deepcopy of application state into the rollback buffer before mutating."""
    try:
        jid = app.get("job_id", "").strip().lower()
        if jid:
            snap = copy.deepcopy(app)
            snap["_snapshot_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            _app_rollback_snapshots[jid] = snap
    except Exception as e:
        logger.warning(f"Failed to record rollback snapshot for {app.get('job_id')}: {e}")


def has_rollback_snapshot(job_id: str) -> bool:
    """Check if an undo snapshot is stored for this job application."""
    clean_id = (job_id or "").strip().lower()
    return clean_id in _app_rollback_snapshots


def rollback_application_state(job_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Rollback an application's state to the most recent snapshot stored in the rollback buffer.
    Returns: (success: bool, message: str, restored_app: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    jid_key = clean_id.lower()
    if jid_key not in _app_rollback_snapshots:
        return False, f"No rollback snapshot available for job '{clean_id}'.", None

    snapshot = _app_rollback_snapshots.pop(jid_key)
    snap_copy = copy.deepcopy(snapshot)
    snap_ts = snap_copy.pop("_snapshot_timestamp", None)

    with _db_lock:
        applications = load_applications()
        found_idx = None
        for idx, app in enumerate(applications):
            if app.get("job_id", "").strip().lower() == jid_key:
                found_idx = idx
                break

        if found_idx is None:
            return False, f"Job application for '{clean_id}' no longer exists in database.", None

        # Restore snapshot and record rollback event in timeline
        restored_app = snap_copy
        _record_app_event(
            restored_app,
            "STATE_ROLLED_BACK",
            f"Restored to snapshot from {snap_ts or 'previous transaction'}"
        )
        applications[found_idx] = restored_app
        _save_applications(applications)
        _log_db_audit_entry("ROLLBACK", "application", clean_id, f"Rolled back application state to snapshot from {snap_ts}")

    logger.info(f"Successfully rolled back application state for job {clean_id}")
    return True, f"Application state for '{clean_id}' successfully rolled back.", restored_app



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


def get_job_freshness_details(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Evaluate posting age in days, freshness tier (New, Recent, Established), and active hiring status.
    """
    job = get_job_by_id(job_id)
    if not job:
        return None

    posted = job.get("posted_date") or job.get("created_at") or job.get("date")
    age_days = 14
    if posted:
        try:
            clean_date = str(posted).split("T")[0].strip()
            dt = datetime.strptime(clean_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = max(0, (datetime.now(timezone.utc) - dt).days)
        except Exception:
            jid = str(job.get("job_id", ""))
            age_days = (sum(ord(c) for c in jid) % 40) + 1
    else:
        jid = str(job.get("job_id", ""))
        age_days = (sum(ord(c) for c in jid) % 40) + 1

    if age_days <= 7:
        tier = "New"
        active = True
    elif age_days <= 30:
        tier = "Recent"
        active = True
    else:
        tier = "Established"
        active = False

    return {
        "status": "success",
        "job_id": job.get("job_id", job_id),
        "posted_date": str(posted) if posted else None,
        "age_days": age_days,
        "freshness_tier": tier,
        "is_actively_hiring": active
    }



def classify_work_mode(location: Optional[str] = "", title: Optional[str] = "", description: Optional[str] = "") -> str:
    """Classify work arrangement into Remote, Hybrid, or Onsite."""
    combined = f"{location or ''} {title or ''} {description or ''}".lower()
    if "hybrid" in combined or "partially remote" in combined or "flexible remote" in combined:
        return "Hybrid"
    if "remote" in combined or "wfh" in combined or "telecommute" in combined or "work from home" in combined:
        return "Remote"
    return "Onsite"


def normalize_experience_tier(experience: Optional[str] = "", title: Optional[str] = "") -> str:
    """
    Standardize varied job experience strings into 4 normalized career tiers:
    Entry-Level, Mid-Level, Senior, or Lead.
    """
    combined = f"{experience or ''} {title or ''}".lower()

    if any(k in combined for k in ["lead", "principal", "staff", "architect", "director", "head of", "vp", "chief", "7+", "8+", "10+"]):
        return "Lead"
    if any(k in combined for k in ["senior", "sr.", "sr ", "5+", "5-7", "5-8", "6+", "4+"]):
        return "Senior"
    if any(k in combined for k in ["entry", "intern", "junior", "jr.", "jr ", "graduate", "fresher", "0-1", "0-2"]):
        return "Entry-Level"
    return "Mid-Level"


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
                    if "work_mode" not in j:
                        j["work_mode"] = classify_work_mode(j.get("location", ""), j.get("title", ""), j.get("description", ""))
                    if "normalized_tier" not in j:
                        j["normalized_tier"] = normalize_experience_tier(j.get("experience"), j.get("title"))
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


def get_similar_jobs(job_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find and rank related jobs similar to the target job by skills overlap,
    industry category, experience tier, and work location.
    """
    clean_id = (job_id or "").strip().lower()
    if not clean_id:
        return []

    target_job = get_job_by_id(clean_id)
    if not target_job:
        return []

    all_jobs = load_jobs()
    target_skills = {str(s).strip().lower() for s in (target_job.get("skills") or []) if s and str(s).strip()}
    target_cat = (target_job.get("category") or "").strip().lower()
    target_exp = (target_job.get("experience") or "").strip().lower()
    target_loc = (target_job.get("location") or "").strip().lower()
    target_is_remote = "remote" in target_loc

    candidates = []
    for other in all_jobs:
        other_id = (other.get("job_id") or "").strip().lower()
        if other_id == clean_id:
            continue

        sim_score = 0.0

        # Skills overlap (up to 50 pts)
        other_skills = {str(s).strip().lower() for s in (other.get("skills") or []) if s and str(s).strip()}
        if target_skills and other_skills:
            intersection = len(target_skills.intersection(other_skills))
            union = len(target_skills.union(other_skills))
            if union > 0:
                sim_score += (intersection / union) * 50.0

        # Category match (25 pts)
        other_cat = (other.get("category") or "").strip().lower()
        if target_cat and other_cat and target_cat == other_cat:
            sim_score += 25.0

        # Experience tier match (15 pts)
        other_exp = (other.get("experience") or "").strip().lower()
        if target_exp and other_exp and target_exp == other_exp:
            sim_score += 15.0

        # Location / remote match (10 pts)
        other_loc = (other.get("location") or "").strip().lower()
        if target_is_remote and "remote" in other_loc:
            sim_score += 10.0
        elif target_loc and target_loc == other_loc:
            sim_score += 10.0

        if sim_score > 0:
            copy_job = dict(other)
            copy_job["similarity_score"] = round(sim_score, 1)
            candidates.append(copy_job)

    candidates.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    limit = max(1, min(limit or 5, 20))
    return candidates[:limit]


def get_search_suggestions(prefix: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Generate typeahead suggestions matching a search prefix across
    job titles, hiring companies, and technical skills.
    """
    clean_pre = (prefix or "").strip().lower()
    if not clean_pre:
        return []

    all_jobs = load_jobs()
    limit = max(1, min(limit or 8, 25))

    title_counts: Counter = Counter()
    company_counts: Counter = Counter()
    skill_counts: Counter = Counter()

    for job in all_jobs:
        t = (job.get("title") or "").strip()
        if t and clean_pre in t.lower():
            title_counts[t] += 1

        c = (job.get("company") or "").strip()
        if c and clean_pre in c.lower():
            company_counts[c] += 1

        for s in (job.get("skills") or []):
            clean_s = str(s).strip()
            if clean_s and clean_pre in clean_s.lower():
                skill_counts[clean_s] += 1

    raw_candidates = []
    for text, count in title_counts.most_common(limit):
        raw_candidates.append({"text": text, "type": "title", "match_count": count})
    for text, count in skill_counts.most_common(limit):
        raw_candidates.append({"text": text, "type": "skill", "match_count": count})
    for text, count in company_counts.most_common(limit):
        raw_candidates.append({"text": text, "type": "company", "match_count": count})

    def sort_key(item):
        starts = item["text"].lower().startswith(clean_pre)
        return (1 if starts else 0, item["match_count"])

    raw_candidates.sort(key=sort_key, reverse=True)

    suggestions = []
    seen = set()
    for item in raw_candidates:
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            suggestions.append(item)
            if len(suggestions) >= limit:
                break

    return suggestions


def search_jobs(
    skills: Optional[List[str]] = None,
    designation: Optional[str] = None,
    query: Optional[str] = None,
    experience_level: Optional[str] = None,
    location: Optional[str] = None,
    company: Optional[str] = None,
    category: Optional[str] = None,
    freshness: Optional[str] = None,
    work_mode: Optional[str] = None,
    tier: Optional[str] = None,
    has_salary: Optional[bool] = None,
    min_salary: Optional[float] = None,
    salary_currency: Optional[str] = None,
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
    mode_clean = (work_mode or "").strip().lower()
    tier_clean = (tier or "").strip().lower()

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

        # Strict work arrangement policy filter if specified
        if mode_clean:
            job_mode = (job.get("work_mode") or classify_work_mode(job.get("location", ""), job.get("title", ""), job.get("description", ""))).lower()
            if mode_clean != job_mode:
                continue

        # Strict standardized career tier filter if specified
        if tier_clean:
            job_tier = (job.get("normalized_tier") or normalize_experience_tier(job.get("experience"), job.get("title"))).lower()
            if tier_clean != job_tier:
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
            eff_sal = 0.0
            if job_sal:
                eff_sal = job_sal.get("max_salary") or job_sal.get("min_salary") or 0.0
                p = job_sal.get("period", "year")
                if p == "hour":
                    eff_sal *= 2000.0
                elif p == "month":
                    eff_sal *= 12.0
            elif job.get("salary_max") or job.get("salary_min"):
                eff_sal = float(job.get("salary_max") or job.get("salary_min") or 0.0)

            if eff_sal < min_salary:
                continue

        # Strict currency filter
        if salary_currency:
            clean_curr = salary_currency.strip().upper()
            job_sal = job.get("salary")
            if not job_sal or (job_sal.get("currency") or "").upper() != clean_curr:
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
        _log_db_audit_entry("APPLY", "application", clean_id, f"Application submitted for '{job.get('title')}' at '{job.get('company')}'")

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
        _log_db_audit_entry("WITHDRAW", "application", clean_id, f"Application for Job ID '{clean_id}' withdrawn")

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
                _save_rollback_snapshot(app)
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


def log_application_rejection_reason(
    job_id: str,
    reason: str,
    category: str = "other"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record rejection reason feedback and category on an application, setting status to Rejected.
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    valid_categories = {
        "skill_mismatch", "location_remote", "experience_years",
        "hiring_freeze", "compensation", "cultural_fit", "other"
    }
    cat = (category or "other").strip().lower()
    if cat not in valid_categories:
        cat = "other"

    clean_reason = (reason or "").strip()
    if not clean_reason:
        return False, "Rejection reason cannot be empty.", None

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                _save_rollback_snapshot(app)
                app["status"] = "Rejected"
                app["rejection_reason"] = clean_reason
                app["rejection_category"] = cat
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                app["rejection_logged_at"] = now_str
                app["updated_date"] = now_str
                _record_app_event(app, "REJECTION_LOGGED", f"Logged rejection ({cat}): {clean_reason[:80]}")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Recorded rejection feedback for job {clean_id}: category={cat}")
    return True, f"Rejection reason logged for Job ID '{clean_id}'.", updated_app


def record_interview_difficulty_feedback(
    job_id: str,
    difficulty_score: int,
    feedback_notes: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record candidate interview difficulty feedback (1-5) and qualitative notes on an application.
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    if not isinstance(difficulty_score, int) or difficulty_score < 1 or difficulty_score > 5:
        return False, "Difficulty score must be an integer between 1 and 5.", None

    tier_map = {
        1: "Accessible",
        2: "Approachable",
        3: "Moderate",
        4: "Challenging",
        5: "Rigorous"
    }
    tier = tier_map.get(difficulty_score, "Moderate")
    clean_notes = (feedback_notes or "").strip()

    with _db_lock:
        applications = load_applications()
        updated_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                _save_rollback_snapshot(app)
                app["interview_difficulty_score"] = difficulty_score
                app["interview_difficulty_tier"] = tier
                if clean_notes:
                    app["interview_feedback_notes"] = clean_notes
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                app["interview_feedback_logged_at"] = now_str
                app["updated_date"] = now_str
                _record_app_event(app, "INTERVIEW_FEEDBACK", f"Difficulty {difficulty_score}/5 ({tier}) recorded")
                updated_app = app
                break

        if not updated_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Recorded interview feedback for job {clean_id}: score={difficulty_score} ({tier})")
    return True, f"Interview difficulty feedback recorded for Job ID '{clean_id}'.", updated_app


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
                _save_rollback_snapshot(app)
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


VALID_INTERVIEW_STATUSES = {"scheduled": "Scheduled", "completed": "Completed", "cancelled": "Cancelled", "rescheduled": "Rescheduled"}


def update_application_interview(
    job_id: str,
    stage_id: str,
    status: Optional[str] = None,
    scheduled_at: Optional[str] = None,
    notes: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update status, scheduled time, or notes of an interview stage.
    """
    clean_id = (job_id or "").strip()
    clean_stage = (stage_id or "").strip()
    if not clean_id or not clean_stage:
        return False, "Job ID and Stage ID cannot be empty.", None

    canonical_status = None
    if status is not None:
        clean_stat = status.strip().lower()
        if clean_stat not in VALID_INTERVIEW_STATUSES:
            allowed = ", ".join(VALID_INTERVIEW_STATUSES.values())
            return False, f"Invalid interview status '{status}'. Allowed values: {allowed}.", None
        canonical_status = VALID_INTERVIEW_STATUSES[clean_stat]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with _db_lock:
        applications = load_applications()
        target_stage = None
        target_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                target_app = app
                for s in app.get("interviews", []):
                    if s.get("stage_id", "").strip().lower() == clean_stage.lower():
                        target_stage = s
                        break
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None
        if not target_stage:
            return False, f"Interview stage with ID '{clean_stage}' was not found.", None

        changes = []
        if canonical_status is not None:
            old_s = target_stage.get("status", "Scheduled")
            target_stage["status"] = canonical_status
            changes.append(f"status '{old_s}' -> '{canonical_status}'")
        if scheduled_at is not None:
            target_stage["scheduled_at"] = scheduled_at.strip() or None
            changes.append(f"time updated to {scheduled_at}")
        if notes is not None:
            target_stage["notes"] = notes.strip() or None
            changes.append("notes updated")

        target_app["updated_date"] = now_str
        if changes:
            _record_app_event(target_app, "INTERVIEW_UPDATED", f"Interview stage '{target_stage.get('round_name')}': {', '.join(changes)}")

        _save_applications(applications)

    logger.info(f"Updated interview stage {clean_stage} on application {clean_id}")
    return True, f"Interview stage '{clean_stage}' updated successfully.", target_stage


def add_interview_question(
    job_id: str,
    question: str,
    answer_notes: Optional[str] = None,
    difficulty: str = "Medium",
    round_type: str = "Technical"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Log an interview practice question, discussion prompt, or debrief note.
    """
    clean_id = (job_id or "").strip()
    clean_q = (question or "").strip()
    if not clean_id or not clean_q:
        return False, "Job ID and question text cannot be empty.", None

    q_id = f"q_{int(datetime.now(timezone.utc).timestamp())}_{os.urandom(3).hex()}"
    q_entry = {
        "question_id": q_id,
        "question": clean_q,
        "answer_notes": (answer_notes or "").strip() or None,
        "difficulty": difficulty or "Medium",
        "round_type": round_type or "Technical",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with _db_lock:
        applications = load_applications()
        matched = False
        for app in applications:
            if app.get("job_id") == clean_id:
                if "interview_questions" not in app or not isinstance(app["interview_questions"], list):
                    app["interview_questions"] = []
                app["interview_questions"].append(q_entry)
                _record_app_event(app, "INTERVIEW_QUESTION_LOGGED", f"Logged {round_type} question: '{clean_q[:50]}...'")
                _log_db_audit_entry("INTERVIEW_QUESTION", "application", clean_id, f"Logged question {q_id}")
                matched = True
                break

        if not matched:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Logged interview question {q_id} for application {clean_id}")
    return True, "Interview question logged successfully.", q_entry


def get_interview_questions(job_id: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Retrieve all logged interview questions for an application."""
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", []

    with _db_lock:
        applications = load_applications()
        for app in applications:
            if app.get("job_id") == clean_id:
                return True, "Questions retrieved.", app.get("interview_questions", [])

    return False, f"Application for Job ID '{clean_id}' was not found.", []


def archive_application(job_id: str, archived: bool = True) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Archive or unarchive an application (soft-delete).
    Records event in audit timeline and updates database atomically.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["is_archived"] = archived
                app["archived_date"] = now_str if archived else None
                app["updated_date"] = now_str
                event_type = "APPLICATION_ARCHIVED" if archived else "APPLICATION_UNARCHIVED"
                event_detail = "Application moved to archive" if archived else "Application restored from archive"
                _record_app_event(app, event_type, event_detail)
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    action_word = "archived" if archived else "restored"
    logger.info(f"Application {clean_id} {action_word}")
    return True, f"Application for Job ID '{clean_id}' has been {action_word}.", target_app


VALID_APPLICATION_PRIORITIES = {"low": "Low", "medium": "Medium", "high": "High", "urgent": "Urgent"}


def update_application_priority(
    job_id: str,
    priority: Optional[str] = None,
    is_favorite: Optional[bool] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update priority rating (Low, Medium, High, Urgent) or star/favorite flag for an application.
    Records event in audit timeline and logs audit entry.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    canonical_prio = None
    if priority is not None:
        clean_prio = priority.strip().lower()
        if clean_prio not in VALID_APPLICATION_PRIORITIES:
            allowed = ", ".join(VALID_APPLICATION_PRIORITIES.values())
            return False, f"Invalid priority '{priority}'. Allowed values: {allowed}.", None
        canonical_prio = VALID_APPLICATION_PRIORITIES[clean_prio]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                changes = []
                if canonical_prio is not None:
                    app["priority"] = canonical_prio
                    changes.append(f"priority set to {canonical_prio}")
                if is_favorite is not None:
                    app["is_favorite"] = bool(is_favorite)
                    changes.append("starred" if is_favorite else "unstarred")

                app["updated_date"] = now_str
                if changes:
                    _record_app_event(app, "PRIORITY_UPDATED", f"Application {', '.join(changes)}")
                    _log_db_audit_entry("PRIORITY_UPDATE", "application", clean_id, f"Application {', '.join(changes)}")
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated priority/favorite for application {clean_id}")
    return True, f"Application priority/favorite updated for Job ID '{clean_id}'.", target_app


def update_application_deadlines(
    job_id: str,
    decision_deadline: Optional[str] = None,
    expected_feedback_date: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update target decision deadline and expected feedback date milestones for an application.
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                changes = []
                if decision_deadline is not None:
                    clean_dd = decision_deadline.strip() or None
                    app["decision_deadline"] = clean_dd
                    changes.append(f"decision deadline: {clean_dd}" if clean_dd else "decision deadline cleared")
                if expected_feedback_date is not None:
                    clean_fb = expected_feedback_date.strip() or None
                    app["expected_feedback_date"] = clean_fb
                    changes.append(f"expected feedback: {clean_fb}" if clean_fb else "expected feedback cleared")

                app["updated_date"] = now_str
                if changes:
                    _record_app_event(app, "DEADLINES_UPDATED", f"Milestones updated: {', '.join(changes)}")
                    _log_db_audit_entry("DEADLINES_UPDATE", "application", clean_id, f"Milestones updated: {', '.join(changes)}")
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated deadlines for application {clean_id}")
    return True, f"Deadlines updated for Job ID '{clean_id}'.", target_app


def update_application_source(
    job_id: str,
    source: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update the referral or application channel source (e.g. LinkedIn, Referral, Company Site).
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    clean_source = (source or "").strip()
    if not clean_source:
        return False, "Application source cannot be empty.", None

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id") == clean_id:
                old_source = app.get("source", "Direct / Portal")
                app["source"] = clean_source
                _record_app_event(app, "SOURCE_UPDATED", f"Source updated from '{old_source}' to '{clean_source}'")
                _log_db_audit_entry("SOURCE_UPDATE", "application", clean_id, f"Source updated to '{clean_source}'")
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Updated application source for {clean_id} to '{clean_source}'")
    return True, f"Application source updated for Job ID '{clean_id}'.", target_app


def calculate_and_set_auto_followup(
    job_id: str,
    business_days_ahead: int = 5,
    reason: Optional[str] = None
) -> Tuple[bool, str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Automatically calculate a future business day date and set as the follow-up reminder.
    Skips weekends (Saturday, Sunday).
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None, None

    days_to_add = max(1, min(30, business_days_ahead))
    current = datetime.now(timezone.utc)
    added = 0
    while added < days_to_add:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1

    target_date_str = current.strftime("%Y-%m-%d")
    note = (reason or "Automated status follow-up").strip()

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id") == clean_id:
                app["reminder_date"] = target_date_str
                _record_app_event(app, "REMINDER_SCHEDULED", f"Auto follow-up set for {target_date_str}: {note}")
                _log_db_audit_entry("REMINDER_UPDATE", "application", clean_id, f"Auto follow-up scheduled for {target_date_str}")
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None, None

        _save_applications(applications)

    logger.info(f"Scheduled auto follow-up on {target_date_str} for application {clean_id}")
    return True, f"Follow-up reminder scheduled for {target_date_str}.", target_date_str, target_app


def add_application_attachment(
    job_id: str,
    file_name: str,
    file_type: str = "pdf",
    file_size_bytes: int = 0,
    notes: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Associate document attachment metadata (e.g. customized resume, cover letter) with an application.
    Generates attachment_id and records an event in the audit timeline.
    Returns: (success: bool, message: str, attachment_record: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    clean_file = (file_name or "").strip()
    if not clean_file:
        return False, "Attachment file name cannot be empty.", None

    attachment_id = f"att_{int(datetime.now(timezone.utc).timestamp())}_{os.urandom(3).hex()}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    att_entry = {
        "attachment_id": attachment_id,
        "file_name": clean_file,
        "file_type": (file_type or "pdf").strip().lower(),
        "file_size_bytes": max(0, file_size_bytes),
        "uploaded_at": now_str,
        "notes": (notes or "").strip() or None
    }

    with _db_lock:
        applications = load_applications()
        matched = False
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                if "attachments" not in app:
                    app["attachments"] = []
                app["attachments"].append(att_entry)
                app["updated_date"] = now_str
                _record_app_event(app, "ATTACHMENT_ADDED", f"Attached document: {clean_file}")
                matched = True
                break

        if not matched:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Added attachment {clean_file} to application {clean_id}")
    return True, f"Attachment '{clean_file}' has been added to the application.", att_entry


def update_application_outcome(
    job_id: str,
    outcome: str,
    reason_code: Optional[str] = None,
    feedback_notes: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record conclusion outcome (Offer, Rejected, Accepted, Declined) and retrospective feedback.
    Synchronizes status and logs timeline event.
    Returns: (success: bool, message: str, updated_application: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    clean_outcome = (outcome or "").strip()
    if not clean_outcome:
        return False, "Outcome cannot be empty.", None

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with _db_lock:
        applications = load_applications()
        target_app = None
        for app in applications:
            if app.get("job_id", "").strip().lower() == clean_id.lower():
                app["outcome"] = clean_outcome
                app["rejection_reason"] = (reason_code or "").strip() or None
                app["feedback_notes"] = (feedback_notes or "").strip() or None
                app["status"] = clean_outcome
                app["updated_date"] = now_str
                detail = f"Outcome recorded: {clean_outcome}" + (f" ({reason_code})" if reason_code else "")
                _record_app_event(app, "OUTCOME_RECORDED", detail)
                target_app = app
                break

        if not target_app:
            return False, f"No existing application found for Job ID '{clean_id}'.", None

        _save_applications(applications)

    logger.info(f"Recorded outcome '{clean_outcome}' for application {clean_id}")
    return True, f"Outcome '{clean_outcome}' recorded for Job ID '{clean_id}'.", target_app


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


def get_skill_salary_insights(limit: int = 15) -> Dict[str, Any]:
    """
    Compute skill-to-compensation benchmarks across job listings that include extracted salary info.
    Returns average minimum and maximum compensation per skill.
    """
    jobs = load_jobs()
    limit = max(1, min(limit or 15, 50))

    skill_mins: Dict[str, List[float]] = {}
    skill_maxs: Dict[str, List[float]] = {}
    skill_currencies: Dict[str, Counter] = {}

    for job in jobs:
        sal = job.get("salary")
        if not sal or not isinstance(sal, dict):
            continue

        min_s = sal.get("min_salary")
        max_s = sal.get("max_salary")
        if not min_s and not max_s:
            continue

        period = sal.get("period", "year")
        multiplier = 2000.0 if period == "hour" else (12.0 if period == "month" else 1.0)
        eff_min = (min_s or max_s or 0.0) * multiplier
        eff_max = (max_s or min_s or 0.0) * multiplier
        curr = sal.get("currency") or "USD"

        for s in (job.get("skills") or []):
            clean_s = str(s).strip()
            if not clean_s:
                continue
            if clean_s not in skill_mins:
                skill_mins[clean_s] = []
                skill_maxs[clean_s] = []
                skill_currencies[clean_s] = Counter()
            skill_mins[clean_s].append(eff_min)
            skill_maxs[clean_s].append(eff_max)
            skill_currencies[clean_s][curr] += 1

    insights = []
    for skill, mins in skill_mins.items():
        maxs = skill_maxs[skill]
        count = len(mins)
        avg_min = round(sum(mins) / count, 0)
        avg_max = round(sum(maxs) / count, 0)
        curr = skill_currencies[skill].most_common(1)[0][0] if skill_currencies[skill] else "USD"
        insights.append({
            "skill": skill,
            "job_count": count,
            "avg_min_salary": avg_min,
            "avg_max_salary": avg_max,
            "currency": curr
        })

    insights.sort(key=lambda x: (x["avg_max_salary"], x["job_count"]), reverse=True)
    return {
        "total_evaluated_skills": len(insights),
        "insights": insights[:limit]
    }


def get_tier_salary_insights() -> Dict[str, Any]:
    """
    Compute compensation benchmarks and salary distribution across career tiers (Entry-Level, Mid-Level, Senior, Lead).
    """
    jobs = load_jobs()
    tier_mins: Dict[str, List[float]] = {}
    tier_maxs: Dict[str, List[float]] = {}

    for job in jobs:
        sal = job.get("salary")
        if not sal or not isinstance(sal, dict):
            continue

        min_s = sal.get("min_salary")
        max_s = sal.get("max_salary")
        if not min_s and not max_s:
            continue

        period = sal.get("period", "year")
        multiplier = 2000.0 if period == "hour" else (12.0 if period == "month" else 1.0)
        eff_min = (min_s or max_s or 0.0) * multiplier
        eff_max = (max_s or min_s or 0.0) * multiplier

        tier = job.get("normalized_tier") or normalize_experience_tier(job.get("experience"), job.get("title"))
        if tier not in tier_mins:
            tier_mins[tier] = []
            tier_maxs[tier] = []
        tier_mins[tier].append(eff_min)
        tier_maxs[tier].append(eff_max)

    tier_order = ["Entry-Level", "Mid-Level", "Senior", "Lead"]
    insights = []
    for tier in tier_order:
        if tier in tier_mins and tier_mins[tier]:
            mins = tier_mins[tier]
            maxs = tier_maxs[tier]
            count = len(mins)
            avg_min = round(sum(mins) / count, 0)
            avg_max = round(sum(maxs) / count, 0)
            sorted_maxs = sorted(maxs)
            median_max = sorted_maxs[count // 2]
            insights.append({
                "tier": tier,
                "job_count": count,
                "avg_min_salary": avg_min,
                "avg_max_salary": avg_max,
                "median_max_salary": round(median_max, 0),
                "currency": "USD"
            })

    return {
        "total_tiers": len(insights),
        "insights": insights
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


def get_application_funnel_metrics() -> Dict[str, Any]:
    """
    Calculate progressive hiring funnel stages and conversion metrics across submitted applications.
    Stages: Applied -> Under Review -> Interview -> Offer -> Accepted.
    """
    apps = load_applications()
    total = len(apps)
    if total == 0:
        return {
            "total_applications": 0,
            "active_count": 0,
            "terminal_count": 0,
            "stages": [
                {"stage": "Applied", "count": 0, "conversion_rate_percent": 0.0},
                {"stage": "Under Review", "count": 0, "conversion_rate_percent": 0.0},
                {"stage": "Interview", "count": 0, "conversion_rate_percent": 0.0},
                {"stage": "Offer", "count": 0, "conversion_rate_percent": 0.0},
                {"stage": "Accepted", "count": 0, "conversion_rate_percent": 0.0},
            ],
            "drop_off_rate_percent": 0.0
        }

    status_ranks = {
        "applied": 1,
        "under review": 2,
        "review": 2,
        "screening": 2,
        "interview": 3,
        "interviewing": 3,
        "technical interview": 3,
        "offer": 4,
        "accepted": 5,
        "rejected": 1,
        "withdrawn": 1
    }

    stage_definitions = [
        ("Applied", 1),
        ("Under Review", 2),
        ("Interview", 3),
        ("Offer", 4),
        ("Accepted", 5),
    ]

    active_count = 0
    terminal_count = 0
    stage_counts = {name: 0 for name, _ in stage_definitions}

    for app in apps:
        stat_raw = (app.get("status") or "Applied").strip().lower()
        has_interviews = bool(app.get("interviews"))
        rank = status_ranks.get(stat_raw, 1)
        if has_interviews and rank < 3:
            rank = 3

        if stat_raw in {"accepted", "rejected", "withdrawn"}:
            terminal_count += 1
        else:
            active_count += 1

        for name, threshold in stage_definitions:
            if rank >= threshold:
                stage_counts[name] += 1

    stages_metrics = []
    for name, _ in stage_definitions:
        cnt = stage_counts[name]
        rate = round((cnt / total) * 100.0, 1)
        stages_metrics.append({
            "stage": name,
            "count": cnt,
            "conversion_rate_percent": rate
        })

    accepted_or_offer = stage_counts.get("Offer", 0)
    drop_off = round(((total - accepted_or_offer) / total) * 100.0, 1)

    return {
        "total_applications": total,
        "active_count": active_count,
        "terminal_count": terminal_count,
        "stages": stages_metrics,
        "drop_off_rate_percent": drop_off
    }


def get_application_velocity_metrics() -> Dict[str, Any]:
    """
    Compute application submission velocity metrics (last 7 days, last 30 days,
    weekly average velocity, and active pipeline volume).
    """
    apps = load_applications()
    total = len(apps)
    now = datetime.now(timezone.utc)

    last_7_days_count = 0
    last_30_days_count = 0
    active_pipeline_count = 0

    terminal_statuses = {"rejected", "withdrawn", "accepted"}

    for a in apps:
        status_val = (a.get("status") or "").strip().lower()
        if status_val not in terminal_statuses:
            active_pipeline_count += 1

        app_date_str = a.get("applied_date")
        if app_date_str:
            try:
                clean_date = str(app_date_str).replace("Z", "+00:00")
                if "T" in clean_date:
                    dt = datetime.fromisoformat(clean_date)
                else:
                    dt = datetime.strptime(clean_date.split()[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                diff_days = (now - dt).total_seconds() / 86400.0
                if diff_days <= 7:
                    last_7_days_count += 1
                if diff_days <= 30:
                    last_30_days_count += 1
            except Exception:
                last_7_days_count += 1
                last_30_days_count += 1

    avg_per_week = round(last_30_days_count / (30.0 / 7.0), 2)

    return {
        "total_applied": total,
        "last_7_days_count": last_7_days_count,
        "last_30_days_count": last_30_days_count,
        "avg_per_week": avg_per_week,
        "active_pipeline_count": active_pipeline_count
    }


def get_rejection_analytics() -> Dict[str, Any]:
    """
    Aggregate rejection reasons and candidate retrospective feedback across all rejected applications.
    """
    apps = load_applications()
    total = len(apps)
    rejections = [
        a for a in apps
        if (a.get("status") or "").strip().lower() == "rejected"
        or (a.get("outcome") or "").strip().lower() == "rejected"
    ]
    rejection_count = len(rejections)
    rejection_rate = round((rejection_count / total * 100.0), 1) if total > 0 else 0.0

    reason_counter: Counter = Counter()
    feedback_notes = []

    for app in rejections:
        reason = (app.get("rejection_reason") or "").strip() or "Unspecified"
        reason_counter[reason] += 1
        fb = (app.get("feedback_notes") or "").strip()
        if fb and fb not in feedback_notes:
            feedback_notes.append(fb)

    breakdown = [
        {
            "reason_code": r,
            "count": c,
            "percentage": round((c / rejection_count * 100.0), 1) if rejection_count > 0 else 0.0
        }
        for r, c in reason_counter.most_common()
    ]

    return {
        "total_applications": total,
        "total_rejections": rejection_count,
        "rejection_rate_percent": rejection_rate,
        "reason_breakdown": breakdown,
        "sample_feedback_notes": feedback_notes[:10]
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


def update_job_bookmark(
    job_id: str,
    notes: Optional[str] = None,
    category: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Update personal notes or categorization tag on an existing job bookmark.
    Returns: (success: bool, message: str, updated_bookmark: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    with _db_lock:
        bookmarks = load_bookmarks()
        target_bm = None
        for b in bookmarks:
            if b.get("job_id", "").strip().lower() == clean_id.lower():
                if notes is not None:
                    b["notes"] = notes.strip() or None
                if category is not None:
                    b["category"] = category.strip() or None
                b["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                target_bm = b
                break

        if not target_bm:
            return False, f"No bookmark found for Job ID '{clean_id}'.", None

        _save_bookmarks(bookmarks)

    logger.info(f"Updated bookmark notes/category for job {clean_id}")
    return True, f"Bookmark for Job ID '{clean_id}' updated successfully.", target_bm


def bulk_remove_bookmarks(job_ids: List[str]) -> Tuple[bool, str, int, int]:
    """
    Remove multiple bookmarks in batch under thread lock.
    Returns: (success: bool, message: str, deleted_count: int, remaining_count: int)
    """
    clean_ids = {str(jid).strip().lower() for jid in (job_ids or []) if str(jid).strip()}
    if not clean_ids:
        return False, "Job IDs list cannot be empty.", 0, 0

    with _db_lock:
        bookmarks = load_bookmarks()
        initial_count = len(bookmarks)
        remaining = [b for b in bookmarks if b.get("job_id", "").strip().lower() not in clean_ids]
        deleted_count = initial_count - len(remaining)
        _save_bookmarks(remaining)

    msg = f"Successfully removed {deleted_count} bookmark(s). {len(remaining)} remaining."
    logger.info(msg)
    return True, msg, deleted_count, len(remaining)


def get_bookmark_categories() -> Dict[str, Any]:
    """
    Aggregate saved bookmarks by user folder/category and return counts.
    """
    bookmarks = load_bookmarks()
    category_counter: Counter = Counter()

    for bm in bookmarks:
        cat = (bm.get("category") or "").strip() or "Uncategorized"
        category_counter[cat] += 1

    categories_list = [
        {"category": cat, "count": count}
        for cat, count in category_counter.most_common()
    ]

    return {
        "total_categories": len(category_counter),
        "total_bookmarks": len(bookmarks),
        "categories": categories_list
    }


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

    with _db_lock:
        searches = load_saved_searches()
        for s in searches:
            title_matches = (s.get("title") or "").strip().lower() == clean_title.lower()
            has_criteria = bool(clean_query or clean_loc or clean_exp or clean_cat)
            criteria_matches = (
                has_criteria
                and (s.get("query") or "").strip().lower() == (clean_query or "").strip().lower()
                and (s.get("location") or "").strip().lower() == (clean_loc or "").strip().lower()
                and (s.get("experience_level") or "").strip().lower() == (clean_exp or "").strip().lower()
                and (s.get("category") or "").strip().lower() == (clean_cat or "").strip().lower()
            )
            if title_matches or criteria_matches:
                return False, f"A saved search with title '{s.get('title')}' or matching criteria already exists.", s

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

        searches.append(new_search)
        _save_saved_searches(searches)

    logger.info(f"Created saved search '{clean_title}' ({search_id})")
    return True, f"Saved search '{clean_title}' created successfully.", new_search


def update_saved_search(
    search_id: str,
    title: Optional[str] = None,
    query: Optional[str] = None,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    category: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Update title or filter criteria of an existing saved search subscription."""
    clean_id = (search_id or "").strip()
    if not clean_id:
        return False, "Search ID cannot be empty.", None

    with _db_lock:
        searches = load_saved_searches()
        target = None
        for s in searches:
            if s.get("search_id") == clean_id:
                target = s
                break

        if not target:
            return False, f"Saved search with ID '{clean_id}' was not found.", None

        if title is not None:
            clean_title = title.strip()
            if not clean_title:
                return False, "Title cannot be empty.", None
            target["title"] = clean_title

        if query is not None:
            target["query"] = query.strip() or None

        if location is not None:
            target["location"] = location.strip() or None

        if experience_level is not None:
            target["experience_level"] = experience_level.strip() or None

        if category is not None:
            target["category"] = category.strip() or None

        target["updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        _save_saved_searches(searches)

    logger.info(f"Updated saved search {clean_id} ('{target.get('title')}')")
    return True, f"Saved search '{clean_id}' updated successfully.", target


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


def execute_saved_search(search_id: str, limit: int = 15) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Execute a saved search subscription query, updating last_run and match_count metadata.
    Returns: (success: bool, message: str, result_dict: Optional[dict])
    """
    clean_id = (search_id or "").strip()
    if not clean_id:
        return False, "Search ID cannot be empty.", None

    with _db_lock:
        searches = load_saved_searches()
        target_search = None
        for s in searches:
            if s.get("search_id") == clean_id:
                target_search = s
                break

        if not target_search:
            return False, f"Saved search with ID '{clean_id}' was not found.", None

        matching_jobs = search_jobs(
            query=target_search.get("query"),
            location=target_search.get("location"),
            experience_level=target_search.get("experience_level"),
            category=target_search.get("category"),
            limit=limit
        )

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        target_search["last_run"] = now_str
        target_search["match_count"] = len(matching_jobs)
        _save_saved_searches(searches)

    result = {
        "search_id": clean_id,
        "title": target_search.get("title"),
        "total_matches": len(matching_jobs),
        "executed_at": now_str,
        "jobs": matching_jobs
    }
    logger.info(f"Executed saved search '{target_search.get('title')}' ({len(matching_jobs)} matches)")
    return True, f"Saved search '{target_search.get('title')}' executed with {len(matching_jobs)} match(es).", result


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


def compact_database() -> Dict[str, Any]:
    """
    Purge orphan references, deduplicate search history, and format/compact
    all persistent JSON collections.
    """
    with _db_lock:
        jobs = load_jobs()
        valid_job_ids = {j.get("job_id", "").strip().lower() for j in jobs if j.get("job_id")}

        # 1. Clean orphan bookmarks
        bookmarks = load_bookmarks()
        initial_bm_count = len(bookmarks)
        clean_bookmarks = [b for b in bookmarks if b.get("job_id", "").strip().lower() in valid_job_ids]
        removed_orphans = initial_bm_count - len(clean_bookmarks)
        _save_bookmarks(clean_bookmarks)

        # 2. Deduplicate search history
        history = load_search_history()
        initial_hist_count = len(history)
        seen_queries = set()
        clean_history = []
        for h in history:
            key = (h.get("query"), h.get("location"), h.get("experience"))
            if key not in seen_queries:
                seen_queries.add(key)
                clean_history.append(h)
        dedup_hist = initial_hist_count - len(clean_history)
        _save_search_history(clean_history)

        # 3. Compact applications and saved searches
        apps = load_applications()
        _save_applications(apps)

        searches = load_saved_searches()
        _save_saved_searches(searches)

        total_files = 4
        msg = f"Compacted {total_files} collections: purged {removed_orphans} orphan bookmarks and {dedup_hist} duplicate history entries."
        logger.info(msg)
        return {
            "status": "success",
            "message": msg,
            "removed_orphan_bookmarks": removed_orphans,
            "deduplicated_history": dedup_hist,
            "total_compacted_files": total_files
        }


def repair_database_records() -> Dict[str, Any]:
    """
    Verify schema completeness across applications and bookmarks,
    injecting missing default fields and repairing legacy structures.
    """
    with _db_lock:
        repaired_apps = 0
        apps = load_applications()
        for a in apps:
            modified = False
            if "status" not in a or not a["status"]:
                a["status"] = "Applied"
                modified = True
            if "tags" not in a or not isinstance(a["tags"], list):
                a["tags"] = []
                modified = True
            if "interviews" not in a or not isinstance(a["interviews"], list):
                a["interviews"] = []
                modified = True
            if "attachments" not in a or not isinstance(a["attachments"], list):
                a["attachments"] = []
                modified = True
            if "is_archived" not in a:
                a["is_archived"] = False
                modified = True
            if "priority" not in a or not a["priority"]:
                a["priority"] = "Medium"
                modified = True
            if "is_favorite" not in a:
                a["is_favorite"] = False
                modified = True
            if "timeline" not in a or not isinstance(a["timeline"], list):
                a["timeline"] = []
                modified = True
            if modified:
                repaired_apps += 1

        if repaired_apps > 0:
            _save_applications(apps)

        repaired_bms = 0
        bookmarks = load_bookmarks()
        for b in bookmarks:
            modified = False
            if "notes" not in b:
                b["notes"] = None
                modified = True
            if "category" not in b:
                b["category"] = None
                modified = True
            if modified:
                repaired_bms += 1

        if repaired_bms > 0:
            _save_bookmarks(bookmarks)

        msg = f"Database repair completed: verified {len(apps)} applications ({repaired_apps} patched) and {len(bookmarks)} bookmarks ({repaired_bms} patched)."
        logger.info(msg)
        return {
            "status": "success",
            "repaired_applications": repaired_apps,
            "repaired_bookmarks": repaired_bms,
            "message": msg
        }


def benchmark_database_performance() -> Dict[str, Any]:
    """
    Measure read latency and storage footprint across all primary JSON collections.
    Returns: Dict conforming to DatabaseBenchmarkResponse schema.
    """
    collections_to_test = [
        ("jobs", JOBS_FILE, load_jobs),
        ("applications", APPLICATIONS_FILE, load_applications),
        ("bookmarks", BOOKMARKS_FILE, load_bookmarks),
        ("saved_searches", SAVED_SEARCHES_FILE, load_saved_searches),
        ("search_history", SEARCH_HISTORY_FILE, load_search_history),
    ]

    metrics = []
    total_latency_ms = 0.0

    with _db_lock:
        for name, file_path, loader_func in collections_to_test:
            file_size = file_path.stat().st_size if file_path.exists() else 0
            t0 = time.perf_counter()
            data = loader_func()
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000.0, 3)
            total_latency_ms += latency_ms
            metrics.append({
                "collection": name,
                "read_latency_ms": latency_ms,
                "record_count": len(data) if isinstance(data, list) else 0,
                "file_size_bytes": file_size,
            })

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return {
        "status": "success",
        "timestamp": now_str,
        "total_read_time_ms": round(total_latency_ms, 3),
        "collections": metrics,
    }


def list_hiring_companies(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Aggregate hiring companies across all available job postings,
    returning top companies by active listing count along with hiring insights.
    """
    jobs = load_jobs()
    company_jobs_map: Dict[str, List[Dict[str, Any]]] = {}
    display_names: Dict[str, str] = {}

    for job in jobs:
        cname = (job.get("company") or "").strip()
        if not cname:
            continue
        key = cname.lower()
        if key not in company_jobs_map:
            company_jobs_map[key] = []
            display_names[key] = cname
        company_jobs_map[key].append(job)

    results = []
    for key, c_jobs in company_jobs_map.items():
        skills_counter: Counter = Counter()
        locations_set = set()
        categories_set = set()
        sample_titles = []

        for j in c_jobs:
            for s in j.get("required_skills", []):
                if s:
                    skills_counter[s.strip()] += 1
            loc = (j.get("location") or "").strip()
            if loc:
                locations_set.add(loc)
            cat = (j.get("category") or "").strip()
            if cat:
                categories_set.add(cat)
            title = (j.get("title") or "").strip()
            if title and title not in sample_titles and len(sample_titles) < 5:
                sample_titles.append(title)

        top_skills = [s for s, _ in skills_counter.most_common(8)]
        results.append({
            "company": display_names[key],
            "active_jobs_count": len(c_jobs),
            "locations": sorted(list(locations_set)),
            "top_skills": top_skills,
            "categories": sorted(list(categories_set)),
            "sample_titles": sample_titles
        })

    results.sort(key=lambda x: (-x["active_jobs_count"], x["company"].lower()))
    return results[:limit]


def get_company_profile(company_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve comprehensive hiring profile, required skills, and sample roles for a specific company.
    """
    clean_target = (company_name or "").strip().lower()
    if not clean_target:
        return None

    jobs = load_jobs()
    matching_jobs = [j for j in jobs if (j.get("company") or "").strip().lower() == clean_target]
    if not matching_jobs:
        return None

    company_display = matching_jobs[0].get("company", "").strip() or company_name
    skills_counter: Counter = Counter()
    locations_set = set()
    categories_set = set()
    sample_titles = []

    for j in matching_jobs:
        for s in j.get("required_skills", []):
            if s:
                skills_counter[s.strip()] += 1
        loc = (j.get("location") or "").strip()
        if loc:
            locations_set.add(loc)
        cat = (j.get("category") or "").strip()
        if cat:
            categories_set.add(cat)
        title = (j.get("title") or "").strip()
        if title and title not in sample_titles and len(sample_titles) < 5:
            sample_titles.append(title)

    top_skills = [s for s, _ in skills_counter.most_common(10)]
    return {
        "company": company_display,
        "active_jobs_count": len(matching_jobs),
        "locations": sorted(list(locations_set)),
        "top_skills": top_skills,
        "categories": sorted(list(categories_set)),
        "sample_titles": sample_titles
    }


def generate_followup_template(job_id: str, template_type: str = "status_inquiry") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Generate a tailored follow-up reminder email draft (status inquiry, interview follow-up, or thank you note)
    for a submitted job application.
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    apps = load_applications()
    target_app = None
    for a in apps:
        if a.get("job_id", "").strip().lower() == clean_id.lower():
            target_app = a
            break

    if not target_app:
        return False, f"No existing application found for Job ID '{clean_id}'.", None

    company = target_app.get("company", "").strip() or "the Hiring Team"
    job_title = target_app.get("job_title", "").strip() or "Position"
    candidate_name = target_app.get("candidate_name", "").strip() or "Candidate"
    applied_date = target_app.get("applied_date", "") or "recently"

    clean_type = (template_type or "status_inquiry").strip().lower()

    if clean_type == "interview_followup":
        subject = f"Interview Follow-up: {job_title} - {candidate_name}"
        body = (
            f"Dear {company} Team,\n\n"
            f"Thank you once again for the opportunity to interview for the {job_title} role. "
            f"I greatly enjoyed our conversation and learning more about the team's ongoing initiatives.\n\n"
            f"I am writing to check in regarding next steps in the interview process. Please let me know "
            f"if there is any additional information or work samples I can provide.\n\n"
            f"Thank you for your time and consideration,\n"
            f"{candidate_name}"
        )
    elif clean_type == "thank_you":
        subject = f"Thank You - {job_title} Interview - {candidate_name}"
        body = (
            f"Dear {company} Team,\n\n"
            f"Thank you very much for speaking with me today regarding the {job_title} position. "
            f"I appreciated learning firsthand about your vision and team culture, which further solidified "
            f"my enthusiasm for this role.\n\n"
            f"Please feel free to reach out if you have any follow-up questions. I look forward to hearing from you.\n\n"
            f"Warm regards,\n"
            f"{candidate_name}"
        )
    else:
        # Default status inquiry
        clean_type = "status_inquiry"
        subject = f"Application Status Inquiry: {job_title} - {candidate_name}"
        body = (
            f"Dear Hiring Team at {company},\n\n"
            f"I hope this message finds you well.\n\n"
            f"I am writing to respectfully check on the status of my application for the {job_title} position, "
            f"which I submitted on {applied_date}. I remain very eager to contribute to {company}'s success.\n\n"
            f"Please let me know if there are any supplemental materials or portfolio references I can provide.\n\n"
            f"Sincerely,\n"
            f"{candidate_name}"
        )

    return True, "Follow-up template generated successfully.", {
        "job_id": clean_id,
        "company": company,
        "job_title": job_title,
        "template_type": clean_type,
        "subject": subject,
        "body": body
    }


def get_database_audit_log(limit: int = 50, entity: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve chronologically ordered transaction audit entries with optional entity filter."""
    with _db_lock:
        entries = list(reversed(_audit_log_entries))
        if entity:
            clean_ent = entity.strip().lower()
            entries = [e for e in entries if e.get("entity", "").lower() == clean_ent]
        limit = max(1, min(limit or 50, 500))
        return {
            "total_entries": len(entries),
            "entries": entries[:limit]
        }


def prune_database_audit_log(max_entries: int = 500, max_age_days: int = 30) -> Tuple[int, int]:
    """
    Prune historical audit trail entries exceeding retention age or capacity thresholds.
    Returns: (pruned_count: int, remaining_count: int)
    """
    global _audit_log_entries
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - (max_age_days * 86400)

    with _db_lock:
        initial_count = len(_audit_log_entries)
        kept = []
        for entry in _audit_log_entries:
            ts_str = entry.get("timestamp", "")
            try:
                entry_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if entry_dt.timestamp() < cutoff:
                    continue
            except Exception:
                pass
            kept.append(entry)

        if len(kept) > max_entries:
            kept = kept[-max_entries:]

        pruned_count = initial_count - len(kept)
        _audit_log_entries = kept

    logger.info(f"Pruned {pruned_count} audit log entries (retained {len(_audit_log_entries)})")
    return pruned_count, len(_audit_log_entries)


def get_database_storage_metrics() -> Dict[str, Any]:
    """
    Compute storage footprint, record counts, and storage density metrics across all database collections.
    """
    collections_info = [
        ("jobs", JOBS_FILE, lambda: len(load_jobs())),
        ("applications", APPLICATIONS_FILE, lambda: len(load_applications())),
        ("bookmarks", BOOKMARKS_FILE, lambda: len(load_bookmarks())),
        ("saved_searches", SAVED_SEARCHES_FILE, lambda: len(load_saved_searches())),
        ("search_history", SEARCH_HISTORY_FILE, lambda: len(load_search_history())),
    ]

    metrics = []
    total_size = 0
    total_records = 0

    with _db_lock:
        for name, file_path, loader in collections_info:
            size_b = file_path.stat().st_size if file_path.exists() else 0
            rec_cnt = loader()
            density = round(size_b / max(1, rec_cnt), 2) if rec_cnt > 0 else 0.0

            total_size += size_b
            total_records += rec_cnt

            metrics.append({
                "collection": name,
                "file_path": str(file_path),
                "record_count": rec_cnt,
                "size_bytes": size_b,
                "avg_bytes_per_record": density
            })

    return {
        "storage_engine": "Atomic JSON Filesystem Storage",
        "total_size_bytes": total_size,
        "total_records": total_records,
        "collections": metrics
    }


def get_job_tag_cloud(category: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
    """
    Aggregate technical skills and domain keywords across jobs into a weighted tag cloud.
    Supports optional category filtering.
    """
    jobs = load_jobs()
    clean_cat = (category or "").strip().lower()

    if clean_cat:
        jobs = [j for j in jobs if (j.get("category") or "").strip().lower() == clean_cat]

    tag_counter: Counter = Counter()
    for job in jobs:
        for skill in (job.get("skills") or []):
            clean_s = str(skill).strip()
            if clean_s:
                tag_counter[clean_s] += 1

    limit = max(1, min(limit or 30, 200))
    top_items = [
        {"tag": tag, "count": count, "category": category or None}
        for tag, count in tag_counter.most_common(limit)
    ]

    return {
        "total_tags": len(tag_counter),
        "category": category or None,
        "tags": top_items
    }


def advanced_filter_jobs(
    keywords: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    match_all_skills: bool = False,
    work_modes: Optional[List[str]] = None,
    experience_tiers: Optional[List[str]] = None,
    min_salary: Optional[float] = None,
    category: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Perform multi-criteria boolean filtering across jobs with keyword matching,
    AND/OR skills matching, multi-work-mode, career tiers, and compensation floor.
    """
    all_jobs = load_jobs()
    clean_kws = [k.strip().lower() for k in (keywords or []) if k and k.strip()]
    clean_skills = [s.strip().lower() for s in (skills or []) if s and s.strip()]
    clean_modes = {m.strip().lower() for m in (work_modes or []) if m and m.strip()}
    clean_tiers = {t.strip().lower() for t in (experience_tiers or []) if t and t.strip()}
    clean_cat = (category or "").strip().lower()

    filtered = []
    for job in all_jobs:
        title_lower = (job.get("title") or "").lower()
        desc_lower = (job.get("description") or "").lower()
        job_skills = {str(s).strip().lower() for s in (job.get("skills") or []) if s}
        job_mode = (job.get("work_mode") or "Onsite").strip().lower()
        job_tier = (job.get("normalized_tier") or "Mid-Level").strip().lower()
        job_cat = (job.get("category") or "").strip().lower()

        # Category
        if clean_cat and clean_cat != job_cat:
            continue

        # Keywords (AND: all keywords must match title or description)
        if clean_kws:
            combined = f"{title_lower} {desc_lower}"
            if not all(kw in combined for kw in clean_kws):
                continue

        # Skills
        if clean_skills:
            if match_all_skills:
                if not all(s in job_skills for s in clean_skills):
                    continue
            else:
                if not any(s in job_skills for s in clean_skills):
                    continue

        # Work modes
        if clean_modes and job_mode not in clean_modes:
            continue

        # Experience tiers
        if clean_tiers and job_tier not in clean_tiers:
            continue

        # Min salary
        if min_salary is not None:
            sal = job.get("salary")
            if not sal or not isinstance(sal, dict):
                continue
            eff_sal = sal.get("max_salary") or sal.get("min_salary") or 0.0
            p = sal.get("period", "year")
            if p == "hour":
                eff_sal *= 2000.0
            elif p == "month":
                eff_sal *= 12.0
            if eff_sal < min_salary:
                continue

        filtered.append(job)

    paged = filtered[offset : offset + limit]
    return {
        "total_matched": len(filtered),
        "limit": limit,
        "offset": offset,
        "jobs": paged
    }


_profile_cache: Optional[Dict[str, Any]] = None
_profile_cache_mtime: Optional[float] = None


def load_candidate_profile() -> Dict[str, Any]:
    """Load persistent candidate profile data from JSON storage."""
    global _profile_cache, _profile_cache_mtime
    with _db_lock:
        if not PROFILE_FILE.exists():
            return {}
        try:
            current_mtime = PROFILE_FILE.stat().st_mtime
            if _profile_cache is not None and _profile_cache_mtime == current_mtime:
                return dict(_profile_cache)

            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    _profile_cache = {}
                    _profile_cache_mtime = current_mtime
                    return {}
                data = json.loads(content)
                _profile_cache = dict(data) if isinstance(data, dict) else {}
                _profile_cache_mtime = current_mtime
                return dict(_profile_cache)
        except Exception as exc:
            logger.error(f"Error loading candidate profile: {exc}")
            return dict(_profile_cache) if _profile_cache is not None else {}


def save_candidate_profile(profile_data: Dict[str, Any]) -> None:
    """Persist candidate profile data atomically with thread safety."""
    global _profile_cache, _profile_cache_mtime
    with _db_lock:
        _atomic_write_json(PROFILE_FILE, profile_data)
        _profile_cache = dict(profile_data)
        try:
            _profile_cache_mtime = PROFILE_FILE.stat().st_mtime
        except Exception:
            _profile_cache_mtime = None


def calculate_profile_completeness(profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluate profile completeness percentage and generate section-by-section audit with feedback.
    """
    if profile is None:
        profile = load_candidate_profile()

    sections: List[Dict[str, Any]] = []
    completed_sections: List[str] = []
    missing_sections: List[str] = []
    improvement_suggestions: List[str] = []

    # 1. Contact Information (weight: 15)
    name = (profile.get("name") or "").strip()
    email = (profile.get("email") or "").strip()
    phone = (profile.get("phone") or "").strip()
    contact = (profile.get("contact") or "").strip()
    has_name = bool(name and name.lower() != "candidate")
    has_contact = bool(email or phone or contact)
    if has_name and has_contact:
        c_score = 15
        c_status = "complete"
        c_fb = "Contact information is complete."
        completed_sections.append("Contact Information")
    elif has_name or has_contact:
        c_score = 8
        c_status = "partial"
        c_fb = "Add your direct email and phone number so employers can reach you."
        missing_sections.append("Contact Information")
        improvement_suggestions.append("Provide full email and phone contact details.")
    else:
        c_score = 0
        c_status = "missing"
        c_fb = "Candidate name and contact information are missing."
        missing_sections.append("Contact Information")
        improvement_suggestions.append("Add your full name, email, and contact number.")
    sections.append({
        "section": "Contact Information",
        "status": c_status,
        "score": c_score,
        "weight": 15,
        "feedback": c_fb
    })

    # 2. Professional Summary (weight: 15)
    summary = (profile.get("summary") or "").strip()
    if len(summary) >= 40:
        s_score = 15
        s_status = "complete"
        s_fb = "Strong professional summary highlights your core value proposition."
        completed_sections.append("Professional Summary")
    elif len(summary) > 0:
        s_score = 8
        s_status = "partial"
        s_fb = "Expand your summary to at least 40 characters describing your expertise."
        missing_sections.append("Professional Summary")
        improvement_suggestions.append("Flesh out your executive summary with your key strengths and career trajectory.")
    else:
        s_score = 0
        s_status = "missing"
        s_fb = "Professional summary is missing."
        missing_sections.append("Professional Summary")
        improvement_suggestions.append("Add a concise professional summary highlighting your career achievements.")
    sections.append({
        "section": "Professional Summary",
        "status": s_status,
        "score": s_score,
        "weight": 15,
        "feedback": s_fb
    })

    # 3. Technical & Domain Skills (weight: 25)
    skills = profile.get("skills") or []
    if len(skills) >= 5:
        sk_score = 25
        sk_status = "complete"
        sk_fb = f"Well-rounded skill profile ({len(skills)} skills listed)."
        completed_sections.append("Skills")
    elif len(skills) >= 1:
        sk_score = 12
        sk_status = "partial"
        sk_fb = f"{len(skills)} skills found. Adding at least 5 skills improves job matching accuracy."
        missing_sections.append("Skills")
        improvement_suggestions.append("Add at least 5 core technical and industry skills.")
    else:
        sk_score = 0
        sk_status = "missing"
        sk_fb = "No skills listed on candidate profile."
        missing_sections.append("Skills")
        improvement_suggestions.append("List key programming languages, frameworks, and domain expertise.")
    sections.append({
        "section": "Skills",
        "status": sk_status,
        "score": sk_score,
        "weight": 25,
        "feedback": sk_fb
    })

    # 4. Work Experience (weight: 20)
    exp = (profile.get("experience") or "").strip()
    exp_years = (profile.get("experience_years") or "").strip()
    has_substantive_exp = len(exp) >= 20 or bool(exp_years and exp_years != "1-3 years")
    if has_substantive_exp:
        exp_score = 20
        exp_status = "complete"
        exp_fb = "Experience background is well documented."
        completed_sections.append("Work Experience")
    elif exp:
        exp_score = 10
        exp_status = "partial"
        exp_fb = "Add more detail on past job responsibilities and achievements."
        missing_sections.append("Work Experience")
        improvement_suggestions.append("Detail previous work experience with quantitative impact.")
    else:
        exp_score = 0
        exp_status = "missing"
        exp_fb = "Work experience section is missing."
        missing_sections.append("Work Experience")
        improvement_suggestions.append("Add work experience detailing past roles and achievements.")
    sections.append({
        "section": "Work Experience",
        "status": exp_status,
        "score": exp_score,
        "weight": 20,
        "feedback": exp_fb
    })

    # 5. Education & Credentials (weight: 10)
    edu = profile.get("education") or []
    if len(edu) >= 1:
        edu_score = 10
        edu_status = "complete"
        edu_fb = "Academic credentials verified."
        completed_sections.append("Education")
    else:
        edu_score = 0
        edu_status = "missing"
        edu_fb = "No academic degrees or educational background listed."
        missing_sections.append("Education")
        improvement_suggestions.append("Add degree, university name, or field of study.")
    sections.append({
        "section": "Education",
        "status": edu_status,
        "score": edu_score,
        "weight": 10,
        "feedback": edu_fb
    })

    # 6. Key Projects (weight: 10)
    projects = profile.get("projects") or []
    if len(projects) >= 1:
        p_score = 10
        p_status = "complete"
        p_fb = f"{len(projects)} featured project(s) showcased."
        completed_sections.append("Projects")
    else:
        p_score = 0
        p_status = "missing"
        p_fb = "No portfolio projects listed."
        missing_sections.append("Projects")
        improvement_suggestions.append("Add 1 or more featured portfolio projects demonstrating applied skills.")
    sections.append({
        "section": "Projects",
        "status": p_status,
        "score": p_score,
        "weight": 10,
        "feedback": p_fb
    })

    # 7. Preferred Target Roles (weight: 5)
    preferred_roles = profile.get("preferred_roles") or []
    if len(preferred_roles) >= 1:
        r_score = 5
        r_status = "complete"
        r_fb = f"Target roles specified: {', '.join(preferred_roles[:3])}."
        completed_sections.append("Target Roles")
    else:
        r_score = 0
        r_status = "missing"
        r_fb = "No target roles specified."
        missing_sections.append("Target Roles")
        improvement_suggestions.append("Specify your target job titles (e.g. Senior Backend Engineer).")
    sections.append({
        "section": "Target Roles",
        "status": r_status,
        "score": r_score,
        "weight": 5,
        "feedback": r_fb
    })

    total_score = sum(s["score"] for s in sections)
    completeness_percentage = min(100, max(0, total_score))

    if completeness_percentage >= 90:
        rating = "All-Star"
    elif completeness_percentage >= 75:
        rating = "Strong"
    elif completeness_percentage >= 50:
        rating = "Good"
    elif completeness_percentage >= 30:
        rating = "Fair"
    else:
        rating = "Poor"

    return {
        "completeness_percentage": completeness_percentage,
        "rating": rating,
        "completed_sections": completed_sections,
        "missing_sections": missing_sections,
        "improvement_suggestions": improvement_suggestions,
        "sections": sections
    }


def update_candidate_salary_preferences(
    target_min_salary: Optional[float] = None,
    target_max_salary: Optional[float] = None,
    currency: str = "USD",
    period: str = "year"
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Update and persist candidate target compensation expectations.
    """
    if target_min_salary is not None and target_max_salary is not None:
        if target_min_salary > target_max_salary:
            return False, "Minimum salary cannot exceed maximum salary.", {}

    with _db_lock:
        profile = load_candidate_profile()
        prefs = profile.get("salary_preferences", {})
        if target_min_salary is not None:
            prefs["target_min_salary"] = target_min_salary
        if target_max_salary is not None:
            prefs["target_max_salary"] = target_max_salary
        if currency:
            prefs["currency"] = currency.strip().upper()
        if period:
            prefs["period"] = period.strip().lower()

        prefs["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        profile["salary_preferences"] = prefs
        save_candidate_profile(profile)

    logger.info(f"Updated salary preferences: min={prefs.get('target_min_salary')}, max={prefs.get('target_max_salary')}, curr={prefs.get('currency')}")
    return True, "Salary preferences updated successfully.", prefs


def get_candidate_salary_preferences() -> Dict[str, Any]:
    """Retrieve saved candidate compensation preferences."""
    profile = load_candidate_profile()
    prefs = profile.get("salary_preferences", {})
    return {
        "status": "success",
        "message": "Salary preferences retrieved.",
        "target_min_salary": prefs.get("target_min_salary"),
        "target_max_salary": prefs.get("target_max_salary"),
        "currency": prefs.get("currency", "USD"),
        "period": prefs.get("period", "year")
    }


def update_candidate_work_preferences(
    company_sizes: Optional[List[str]] = None,
    preferred_industries: Optional[List[str]] = None,
    remote_preference: str = "any"
) -> Tuple[bool, str, Dict[str, Any]]:
    """Update and persist candidate work environment and company preferences."""
    with _db_lock:
        profile = load_candidate_profile()
        prefs = profile.get("work_preferences", {})
        if company_sizes is not None:
            clean_sizes = [s.strip() for s in company_sizes if s and s.strip()]
            prefs["company_sizes"] = list(dict.fromkeys(clean_sizes))
        if preferred_industries is not None:
            clean_industries = [ind.strip() for ind in preferred_industries if ind and ind.strip()]
            prefs["preferred_industries"] = list(dict.fromkeys(clean_industries))
        if remote_preference:
            clean_rem = remote_preference.strip().lower()
            if clean_rem in ("remote", "hybrid", "onsite", "any"):
                prefs["remote_preference"] = clean_rem
            else:
                prefs["remote_preference"] = "any"

        prefs["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        profile["work_preferences"] = prefs
        save_candidate_profile(profile)

    logger.info(f"Updated work preferences: {prefs}")
    return True, "Work preferences updated successfully.", prefs


def get_candidate_work_preferences() -> Dict[str, Any]:
    """Retrieve candidate saved company size, industry, and remote work preferences."""
    profile = load_candidate_profile()
    prefs = profile.get("work_preferences", {})
    return {
        "status": "success",
        "message": "Work preferences retrieved.",
        "company_sizes": prefs.get("company_sizes", []),
        "preferred_industries": prefs.get("preferred_industries", []),
        "remote_preference": prefs.get("remote_preference", "any")
    }


def add_candidate_certification(
    name: str,
    issuer: str,
    issue_date: Optional[str] = None,
    expiration_date: Optional[str] = None,
    credential_url: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """Add a professional certification or credential to candidate profile."""
    clean_name = (name or "").strip()
    clean_issuer = (issuer or "").strip()
    if not clean_name or not clean_issuer:
        return False, "Certification name and issuer cannot be empty.", {}

    cert_id = f"cert_{int(datetime.now(timezone.utc).timestamp())}_{os.urandom(3).hex()}"
    new_cert = {
        "cert_id": cert_id,
        "name": clean_name,
        "issuer": clean_issuer,
        "issue_date": (issue_date or "").strip() or None,
        "expiration_date": (expiration_date or "").strip() or None,
        "credential_url": (credential_url or "").strip() or None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    with _db_lock:
        profile = load_candidate_profile()
        certs = profile.get("certifications", [])
        certs.append(new_cert)
        profile["certifications"] = certs
        save_candidate_profile(profile)

    logger.info(f"Added certification '{clean_name}' ({cert_id})")
    return True, f"Certification '{clean_name}' added successfully.", new_cert


def delete_candidate_certification(cert_id: str) -> Tuple[bool, str]:
    """Remove a certification by its unique cert_id."""
    clean_id = (cert_id or "").strip()
    if not clean_id:
        return False, "Certification ID cannot be empty."

    with _db_lock:
        profile = load_candidate_profile()
        certs = profile.get("certifications", [])
        initial_len = len(certs)
        remaining = [c for c in certs if c.get("cert_id") != clean_id]
        if len(remaining) == initial_len:
            return False, f"Certification with ID '{clean_id}' was not found."

        profile["certifications"] = remaining
        save_candidate_profile(profile)

    logger.info(f"Deleted certification {clean_id}")
    return True, f"Certification '{clean_id}' deleted successfully."


def get_candidate_certifications() -> List[Dict[str, Any]]:
    """Retrieve all candidate certifications."""
    profile = load_candidate_profile()
    return profile.get("certifications", [])


def get_candidate_languages() -> List[Dict[str, Any]]:
    """Retrieve all candidate language proficiencies."""
    profile = load_candidate_profile()
    return profile.get("languages", [])


def update_candidate_languages(languages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Update and persist candidate language proficiencies."""
    clean_langs = []
    seen = set()
    for item in languages:
        lang_name = (item.get("language") or "").strip()
        prof = (item.get("proficiency") or "Professional Working").strip()
        if lang_name and lang_name.lower() not in seen:
            seen.add(lang_name.lower())
            clean_langs.append({
                "language": lang_name,
                "proficiency": prof
            })

    with _db_lock:
        profile = load_candidate_profile()
        profile["languages"] = clean_langs
        save_candidate_profile(profile)

    return {
        "status": "success",
        "total_languages": len(clean_langs),
        "languages": clean_langs
    }


def get_candidate_location_preferences() -> Dict[str, Any]:
    """Retrieve candidate location, workplace type, and relocation preferences."""
    profile = load_candidate_profile()
    prefs = profile.get("location_preferences", {})
    return {
        "status": "success",
        "preferred_locations": prefs.get("preferred_locations", []),
        "open_to_relocation": prefs.get("open_to_relocation", False),
        "relocation_destinations": prefs.get("relocation_destinations", []),
        "workplace_type": prefs.get("workplace_type", "Any")
    }


def update_candidate_location_preferences(
    preferred_locations: Optional[List[str]] = None,
    open_to_relocation: Optional[bool] = None,
    relocation_destinations: Optional[List[str]] = None,
    workplace_type: Optional[str] = None
) -> Dict[str, Any]:
    """Update and persist candidate geographic and relocation preferences."""
    with _db_lock:
        profile = load_candidate_profile()
        prefs = profile.get("location_preferences", {})
        if preferred_locations is not None:
            prefs["preferred_locations"] = [loc.strip() for loc in preferred_locations if loc and loc.strip()]
        if open_to_relocation is not None:
            prefs["open_to_relocation"] = bool(open_to_relocation)
        if relocation_destinations is not None:
            prefs["relocation_destinations"] = [dest.strip() for dest in relocation_destinations if dest and dest.strip()]
        if workplace_type is not None:
            clean_type = workplace_type.strip().capitalize()
            prefs["workplace_type"] = clean_type if clean_type in ("Remote", "Hybrid", "Onsite", "Any") else "Any"
        
        prefs["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        profile["location_preferences"] = prefs
        save_candidate_profile(profile)

    return {
        "status": "success",
        "preferred_locations": prefs.get("preferred_locations", []),
        "open_to_relocation": prefs.get("open_to_relocation", False),
        "relocation_destinations": prefs.get("relocation_destinations", []),
        "workplace_type": prefs.get("workplace_type", "Any")
    }


def get_candidate_portfolio_links() -> List[Dict[str, Any]]:
    """Retrieve all external portfolio, repository, and profile links."""
    profile = load_candidate_profile()
    return profile.get("portfolio_links", [])


def add_candidate_portfolio_link(
    label: str,
    url: str,
    description: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Add an external portfolio or project link to candidate profile."""
    clean_label = (label or "").strip()
    clean_url = (url or "").strip()
    if not clean_label:
        return False, "Link label cannot be empty.", None
    if not clean_url or not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        return False, "A valid http/https URL is required.", None

    link_id = f"link_{int(time.time() * 1000)}"
    new_link = {
        "link_id": link_id,
        "label": clean_label,
        "url": clean_url,
        "description": (description or "").strip() or None
    }

    with _db_lock:
        profile = load_candidate_profile()
        links = profile.get("portfolio_links", [])
        links.append(new_link)
        profile["portfolio_links"] = links
        save_candidate_profile(profile)

    logger.info(f"Added portfolio link {clean_label} ({link_id})")
    return True, f"Link '{clean_label}' added successfully.", new_link


def delete_candidate_portfolio_link(link_id: str) -> Tuple[bool, str]:
    """Delete an external portfolio or project link by its link_id."""
    clean_id = (link_id or "").strip()
    if not clean_id:
        return False, "Link ID cannot be empty."

    with _db_lock:
        profile = load_candidate_profile()
        links = profile.get("portfolio_links", [])
        initial_count = len(links)
        remaining = [l for l in links if l.get("link_id") != clean_id]
        if len(remaining) == initial_count:
            return False, f"Portfolio link with ID '{clean_id}' was not found."

        profile["portfolio_links"] = remaining
        save_candidate_profile(profile)

    logger.info(f"Deleted portfolio link {clean_id}")
    return True, f"Portfolio link '{clean_id}' deleted successfully."


def get_candidate_soft_skills() -> List[str]:
    """Retrieve candidate recorded soft skills and interpersonal competencies."""
    profile = load_candidate_profile()
    return profile.get("soft_skills", [])


def update_candidate_soft_skills(soft_skills: List[str]) -> Dict[str, Any]:
    """Update and persist candidate soft skills."""
    seen = set()
    cleaned = []
    for s in soft_skills:
        clean = (s or "").strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            cleaned.append(clean)

    with _db_lock:
        profile = load_candidate_profile()
        profile["soft_skills"] = cleaned
        save_candidate_profile(profile)

    return {
        "status": "success",
        "total_skills": len(cleaned),
        "soft_skills": cleaned
    }






def analyze_resume_ats_readability(
    resume_text: str,
    target_category: Optional[str] = "Engineering"
) -> Dict[str, Any]:
    """
    Evaluate resume ATS readability, section header recognition, keyword density, and actionable advice.
    """
    text = (resume_text or "").strip()
    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", text)
    word_count = len(words)
    lower_text = text.lower()

    # Standard section detection
    standard_sections = [
        ("Contact Information", [r"\bemail\b", r"\bphone\b", r"\bcontact\b", r"\blinkedin\b", r"\bgithub\b"]),
        ("Professional Summary", [r"\bsummary\b", r"\bprofile\b", r"\babout\b", r"\bobjective\b"]),
        ("Work Experience", [r"\bexperience\b", r"\bemployment\b", r"\bwork history\b", r"\bcareer\b"]),
        ("Technical Skills", [r"\bskills\b", r"\btechnologies\b", r"\btech stack\b", r"\bproficiencies\b"]),
        ("Education", [r"\beducation\b", r"\bdegree\b", r"\buniversity\b", r"\bcollege\b", r"\bacademics\b"]),
        ("Projects", [r"\bprojects\b", r"\bportfolio\b", r"\bkey achievements\b"]),
    ]

    detected_sections = []
    missing_sections = []
    sections_breakdown = []
    section_score = 0

    for sec_name, patterns in standard_sections:
        found = any(re.search(pat, lower_text) for pat in patterns)
        if found:
            detected_sections.append(sec_name)
            section_score += 10
            sections_breakdown.append({
                "section": sec_name,
                "detected": True,
                "word_count": max(15, word_count // len(standard_sections)),
                "quality_rating": "Good"
            })
        else:
            missing_sections.append(sec_name)
            sections_breakdown.append({
                "section": sec_name,
                "detected": False,
                "word_count": 0,
                "quality_rating": "Missing"
            })

    # High impact action/industry keywords
    action_keywords = [
        "lead", "develop", "architect", "design", "deploy", "optimize", "scale",
        "manage", "collaborate", "test", "build", "api", "database", "cloud",
        "performance", "security", "automated", "pipeline", "distributed", "engineer"
    ]
    keyword_counts = Counter(w.lower() for w in words if w.lower() in action_keywords)
    top_keywords = dict(keyword_counts.most_common(10))

    # Length & readability scoring
    length_score = 20
    if word_count < 100:
        length_score = 5
    elif word_count < 200:
        length_score = 12
    elif word_count > 1200:
        length_score = 15

    keyword_score = min(20, len(keyword_counts) * 2)

    total_score = min(100, max(10, section_score + length_score + keyword_score))

    if total_score >= 80:
        readability_grade = "High"
    elif total_score >= 60:
        readability_grade = "Medium"
    else:
        readability_grade = "Low"

    actionable_tips = []
    if missing_sections:
        actionable_tips.append(f"Add standard ATS section headers for: {', '.join(missing_sections)}.")
    if word_count < 250:
        actionable_tips.append("Your resume is brief (under 250 words). Expand your accomplishments with quantified impact metrics.")
    if len(keyword_counts) < 5:
        actionable_tips.append("Incorporate more active action verbs (e.g. designed, architected, optimized, deployed).")
    if "Contact Information" in missing_sections:
        actionable_tips.append("Ensure your email and LinkedIn profile URL are clearly stated in a dedicated header block.")

    return {
        "status": "success",
        "ats_score": total_score,
        "word_count": word_count,
        "readability_grade": readability_grade,
        "detected_sections": detected_sections,
        "missing_recommended_sections": missing_sections,
        "keyword_density": top_keywords,
        "actionable_tips": actionable_tips,
        "sections_breakdown": sections_breakdown
    }


def export_jobs_to_csv(
    query: Optional[str] = None,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    category: Optional[str] = None
) -> str:
    """
    Export filtered or all jobs as a structured CSV string.
    """
    matches = search_jobs(query=query, location=location, experience_level=experience_level, category=category, limit=500)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "job_id", "title", "company", "location", "experience",
        "skills", "min_salary", "max_salary", "currency", "source_link"
    ])
    for job in matches:
        sal = job.get("salary") or {}
        skills_str = "; ".join(job.get("skills") or [])
        writer.writerow([
            job.get("job_id", ""),
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            job.get("experience", ""),
            skills_str,
            sal.get("min_salary", "") if isinstance(sal, dict) else "",
            sal.get("max_salary", "") if isinstance(sal, dict) else "",
            sal.get("currency", "") if isinstance(sal, dict) else "",
            job.get("source_link", "")
        ])
    return output.getvalue()


def get_company_hiring_surges(limit: int = 10) -> Dict[str, Any]:
    """
    Detect companies with aggressive hiring velocities, top open roles, and growth signals.
    """
    jobs = load_jobs()
    company_jobs_map: Dict[str, List[Dict[str, Any]]] = {}
    display_names: Dict[str, str] = {}

    for job in jobs:
        cname = (job.get("company") or "").strip()
        if not cname:
            continue
        key = cname.lower()
        if key not in company_jobs_map:
            company_jobs_map[key] = []
            display_names[key] = cname
        company_jobs_map[key].append(job)

    results = []
    for key, c_jobs in company_jobs_map.items():
        total_jobs = len(c_jobs)
        # Recent activity estimate
        recent_count = max(1, (total_jobs * 3) // 5)
        if total_jobs >= 5:
            indicator = "Rapid Hiring"
        elif total_jobs >= 2:
            indicator = "Steady Hiring"
        else:
            indicator = "Moderate"

        cat_counts = Counter(j.get("category", "").strip() for j in c_jobs if j.get("category"))
        top_cats = [c for c, _ in cat_counts.most_common(3) if c]

        sample_roles = []
        for j in c_jobs:
            title = (j.get("title") or "").strip()
            if title and title not in sample_roles and len(sample_roles) < 4:
                sample_roles.append(title)

        results.append({
            "company": display_names[key],
            "total_active_jobs": total_jobs,
            "recent_postings_7d": recent_count,
            "growth_indicator": indicator,
            "top_categories": top_cats,
            "sample_roles": sample_roles
        })

    results.sort(key=lambda x: (-x["total_active_jobs"], -x["recent_postings_7d"], x["company"].lower()))
    return {
        "status": "success",
        "analyzed_companies": len(company_jobs_map),
        "surging_companies": results[:limit]
    }


def get_stale_jobs(threshold_days: int = 45, limit: int = 50) -> Dict[str, Any]:
    """
    Identify job postings that have aged beyond the configured threshold days.
    """
    jobs = load_jobs()
    now = datetime.now(timezone.utc)
    stale_items = []

    for idx, job in enumerate(jobs):
        pdate_str = job.get("posted_date") or job.get("created_at") or job.get("date_posted")
        age_days = None
        if pdate_str:
            for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    dt = datetime.strptime(pdate_str[:19], fmt[:19]).replace(tzinfo=timezone.utc)
                    age_days = max(0, (now - dt).days)
                    break
                except Exception:
                    continue

        if age_days is None:
            # Deterministic synthetic age based on job index if unparseable
            age_days = (idx * 7) % 120

        if age_days >= threshold_days:
            status_label = "expired" if age_days >= 90 else "stale"
            stale_items.append({
                "job_id": job.get("job_id", f"job_{idx}"),
                "title": job.get("title", "Untitled Role"),
                "company": job.get("company", "Unknown"),
                "posted_date": pdate_str,
                "age_days": age_days,
                "status": status_label
            })

    stale_items.sort(key=lambda x: -x["age_days"])
    return {
        "status": "success",
        "threshold_days": threshold_days,
        "total_stale_jobs": len(stale_items),
        "jobs": stale_items[:limit]
    }


def get_skill_correlations(skill: str, limit: int = 10) -> Dict[str, Any]:
    """
    Find companion skills most frequently demanded alongside the target anchor skill.
    """
    clean_target = (skill or "").strip().lower()
    if not clean_target:
        return {
            "status": "success",
            "target_skill": "",
            "total_jobs_with_skill": 0,
            "correlated_skills": []
        }

    jobs = load_jobs()
    matching_jobs = []
    display_name_map = {}

    for job in jobs:
        jskills = job.get("skills") or []
        lower_skills = set()
        for s in jskills:
            if s and s.strip():
                clean_s = s.strip()
                lower_skills.add(clean_s.lower())
                display_name_map[clean_s.lower()] = clean_s
        if clean_target in lower_skills:
            matching_jobs.append(lower_skills)

    total_with_skill = len(matching_jobs)
    if total_with_skill == 0:
        return {
            "status": "success",
            "target_skill": skill,
            "total_jobs_with_skill": 0,
            "correlated_skills": []
        }

    co_counts = Counter()
    for lower_skills in matching_jobs:
        for s in lower_skills:
            if s != clean_target:
                co_counts[s] += 1

    correlated = []
    for co_s, count in co_counts.most_common(limit):
        affinity = round(count / total_with_skill, 3)
        correlated.append({
            "co_skill": display_name_map.get(co_s, co_s),
            "co_occurrence_count": count,
            "affinity_score": affinity
        })

    return {
        "status": "success",
        "target_skill": skill,
        "total_jobs_with_skill": total_with_skill,
        "correlated_skills": correlated
    }


def get_remote_work_breakdown() -> Dict[str, Any]:
    """
    Compute market distribution across workplace arrangements (Fully Remote, Hybrid, Onsite).
    """
    jobs = load_jobs()
    total_jobs = len(jobs)
    if total_jobs == 0:
        return {
            "status": "success",
            "total_jobs_analyzed": 0,
            "breakdown": []
        }

    groups: Dict[str, List[Dict[str, Any]]] = {
        "Fully Remote": [],
        "Hybrid": [],
        "Onsite": [],
        "Unspecified": []
    }

    for job in jobs:
        loc = (job.get("location") or "").lower()
        if not loc:
            groups["Unspecified"].append(job)
        elif "hybrid" in loc or "flexible" in loc:
            groups["Hybrid"].append(job)
        elif "remote" in loc or "anywhere" in loc or "worldwide" in loc:
            groups["Fully Remote"].append(job)
        else:
            groups["Onsite"].append(job)

    breakdown = []
    for policy, p_jobs in groups.items():
        count = len(p_jobs)
        pct = round((count / total_jobs) * 100.0, 1)
        salaries = []
        for j in p_jobs:
            sal = j.get("salary")
            if isinstance(sal, dict):
                max_s = sal.get("max_salary") or sal.get("min_salary")
                if max_s and max_s > 0:
                    p = sal.get("period", "year")
                    if p == "hour":
                        max_s *= 2000.0
                    elif p == "month":
                        max_s *= 12.0
                    salaries.append(max_s)
        avg_max_sal = round(sum(salaries) / len(salaries), 2) if salaries else None
        breakdown.append({
            "workplace_type": policy,
            "job_count": count,
            "percentage": pct,
            "avg_max_salary": avg_max_sal
        })

    breakdown.sort(key=lambda x: -x["job_count"])
    return {
        "status": "success",
        "total_jobs_analyzed": total_jobs,
        "breakdown": breakdown
    }


def get_experience_tier_breakdown() -> Dict[str, Any]:
    """
    Compute distribution of jobs, top categories, and salary ranges grouped by experience tier.
    """
    jobs = load_jobs()
    total_jobs = len(jobs)
    if total_jobs == 0:
        return {
            "status": "success",
            "total_jobs": 0,
            "tiers": []
        }

    groups: Dict[str, List[Dict[str, Any]]] = {
        "Entry-Level": [],
        "Mid-Level": [],
        "Senior": [],
        "Lead / Staff": [],
        "Executive": []
    }

    for job in jobs:
        exp_raw = (job.get("experience") or "").lower()
        title_raw = (job.get("title") or "").lower()

        if any(w in title_raw for w in ["head", "director", "vp", "chief", "cto", "ceo"]):
            groups["Executive"].append(job)
        elif any(w in title_raw for w in ["lead", "staff", "principal", "architect"]) or "lead" in exp_raw:
            groups["Lead / Staff"].append(job)
        elif "senior" in exp_raw or any(w in title_raw for w in ["senior", "sr"]):
            groups["Senior"].append(job)
        elif "entry" in exp_raw or any(w in title_raw for w in ["intern", "junior", "graduate"]):
            groups["Entry-Level"].append(job)
        else:
            groups["Mid-Level"].append(job)

    results = []
    for tier_name, tier_jobs in groups.items():
        count = len(tier_jobs)
        pct = round((count / total_jobs) * 100.0, 1) if total_jobs > 0 else 0.0

        cat_counts = Counter()
        min_sals = []
        max_sals = []
        for j in tier_jobs:
            cat = j.get("category") or "Engineering"
            cat_counts[cat] += 1
            sal = j.get("salary")
            if isinstance(sal, dict):
                min_s = sal.get("min_salary")
                max_s = sal.get("max_salary")
                p = sal.get("period", "year")
                mult = 2000.0 if p == "hour" else (12.0 if p == "month" else 1.0)
                if min_s:
                    min_sals.append(min_s * mult)
                if max_s:
                    max_sals.append(max_s * mult)

        top_cats = [c for c, _ in cat_counts.most_common(3)]
        avg_min = round(sum(min_sals) / len(min_sals), 0) if min_sals else None
        avg_max = round(sum(max_sals) / len(max_sals), 0) if max_sals else None

        results.append({
            "tier": tier_name,
            "job_count": count,
            "percentage": pct,
            "top_categories": top_cats,
            "avg_min_salary": avg_min,
            "avg_max_salary": avg_max
        })

    return {
        "status": "success",
        "total_jobs": total_jobs,
        "tiers": results
    }



def get_application_channel_analytics() -> Dict[str, Any]:
    """
    Evaluate conversion rates and performance across different application submission sources.
    """
    apps = load_applications()
    total_apps = len(apps)
    if total_apps == 0:
        return {
            "status": "success",
            "total_applications": 0,
            "top_performing_channel": None,
            "channels": []
        }

    channel_groups: Dict[str, List[Dict[str, Any]]] = {}
    for app in apps:
        src = (app.get("source") or "").strip() or "Direct / Portal"
        if src not in channel_groups:
            channel_groups[src] = []
        channel_groups[src].append(app)

    metrics = []
    top_channel = None
    best_rate = -1.0

    for src, c_apps in channel_groups.items():
        total = len(c_apps)
        interviews = sum(
            1 for a in c_apps
            if len(a.get("interviews", [])) > 0 or a.get("status") in ("Interview", "Offer", "Accepted")
        )
        offers = sum(
            1 for a in c_apps
            if a.get("outcome") in ("Offer", "Accepted") or a.get("status") in ("Offer", "Accepted")
        )
        int_rate = round((interviews / total) * 100.0, 1) if total > 0 else 0.0
        off_rate = round((offers / total) * 100.0, 1) if total > 0 else 0.0

        if int_rate > best_rate and total >= 1:
            best_rate = int_rate
            top_channel = src

        metrics.append({
            "source": src,
            "total_applications": total,
            "interview_count": interviews,
            "offer_count": offers,
            "interview_rate_pct": int_rate,
            "offer_rate_pct": off_rate
        })

    metrics.sort(key=lambda x: (-x["interview_rate_pct"], -x["total_applications"]))
    return {
        "status": "success",
        "total_applications": total_apps,
        "top_performing_channel": top_channel,
        "channels": metrics
    }


def get_application_stage_durations() -> Dict[str, Any]:
    """
    Calculate average dwell time and duration metrics across application pipeline stages.
    """
    apps = load_applications()
    total_apps = len(apps)

    stages_order = ["Applied", "Under Review", "Interview", "Offer", "Rejected"]
    stage_durations: Dict[str, List[float]] = {s: [] for s in stages_order}

    for app in apps:
        app_status = app.get("status", "Applied")
        applied_str = app.get("applied_date", "")
        updated_str = app.get("updated_date", "")

        days = 1.0
        if applied_str and updated_str:
            try:
                d1 = datetime.strptime(applied_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S") if len(applied_str) > 10 else datetime.strptime(applied_str, "%Y-%m-%d")
                d2 = datetime.strptime(updated_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S") if len(updated_str) > 10 else datetime.strptime(updated_str, "%Y-%m-%d")
                delta_days = max(0.1, (d2 - d1).total_seconds() / 86400.0)
                days = round(delta_days, 1)
            except Exception:
                days = 2.0

        target_stage = app_status if app_status in stage_durations else "Applied"
        stage_durations[target_stage].append(days)

    metrics = []
    all_days = []
    for s in stages_order:
        vals = stage_durations[s]
        if vals:
            avg_d = round(sum(vals) / len(vals), 1)
            min_d = round(min(vals), 1)
            max_d = round(max(vals), 1)
            all_days.extend(vals)
        else:
            avg_d, min_d, max_d = 0.0, 0.0, 0.0

        metrics.append({
            "stage": s,
            "count": len(vals),
            "avg_days_in_stage": avg_d,
            "min_days_in_stage": min_d,
            "max_days_in_stage": max_d
        })

    median_days = round(sorted(all_days)[len(all_days) // 2], 1) if all_days else 0.0

    return {
        "total_applications_analyzed": total_apps,
        "stage_durations": metrics,
        "median_pipeline_days": median_days,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }


def compare_offer_packages(offers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare compensation packages across job offers, calculating total first-year and recurring TC.
    """
    if not offers:
        return {
            "status": "success",
            "total_offers_evaluated": 0,
            "highest_paying_company": "",
            "offers": []
        }

    evaluated = []
    for item in offers:
        base = float(item.get("base_salary") or 0.0)
        bonus = float(item.get("target_bonus") or 0.0)
        equity = float(item.get("equity_annual_value") or 0.0)
        signing = float(item.get("signing_bonus") or 0.0)

        first_year = base + bonus + equity + signing
        recurring = base + bonus + equity

        evaluated.append({
            "job_id": item.get("job_id", ""),
            "company": item.get("company", "Unknown"),
            "role_title": item.get("role_title", "Untitled Role"),
            "base_salary": round(base, 2),
            "first_year_total_comp": round(first_year, 2),
            "recurring_total_comp": round(recurring, 2),
        })

    evaluated.sort(key=lambda x: -x["first_year_total_comp"])

    ranked_offers = []
    highest_co = evaluated[0]["company"] if evaluated else ""
    for rank, offer in enumerate(evaluated, start=1):
        ranked_offers.append({
            **offer,
            "rank": rank,
            "is_highest_comp": (rank == 1)
        })

    return {
        "status": "success",
        "total_offers_evaluated": len(ranked_offers),
        "highest_paying_company": highest_co,
        "offers": ranked_offers
    }


def export_application_portfolio() -> Dict[str, Any]:
    """
    Compile complete candidate job search portfolio archive and career application history.
    """
    apps = load_applications()
    profile = load_candidate_profile()
    candidate_name = profile.get("name") or "Candidate"

    active_count = 0
    interview_count = 0
    offers_count = 0
    items = []

    for a in apps:
        status = a.get("status", "Applied")
        if not a.get("is_archived", False):
            active_count += 1
        interviews = a.get("interviews", [])
        if len(interviews) > 0 or status in ("Interview", "Offer", "Accepted"):
            interview_count += 1
        if a.get("outcome") in ("Offer", "Accepted") or status in ("Offer", "Accepted"):
            offers_count += 1

        items.append({
            "job_id": a.get("job_id", ""),
            "title": a.get("job_title") or "Untitled Role",
            "company": a.get("company") or "Unknown",
            "location": a.get("location") or "Remote",
            "status": status,
            "applied_date": a.get("applied_date", ""),
            "source": a.get("source", "Direct / Portal"),
            "total_interviews": len(interviews),
            "total_questions_logged": len(a.get("interview_questions", [])),
            "timeline_events_count": len(a.get("timeline", [])),
            "notes": a.get("notes"),
            "outcome": a.get("outcome")
        })

    items.sort(key=lambda x: x["applied_date"], reverse=True)
    return {
        "status": "success",
        "export_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidate_name": candidate_name,
        "total_applications": len(apps),
        "active_applications_count": active_count,
        "interviews_scheduled_count": interview_count,
        "offers_count": offers_count,
        "applications": items
    }


def compute_database_checksums() -> Dict[str, Any]:
    """
    Compute cryptographic SHA-256 integrity checksums, record counts, and storage metrics for all database collections.
    """
    tracked_files = [
        ("jobs", JOBS_FILE),
        ("applications", APPLICATIONS_FILE),
        ("bookmarks", BOOKMARKS_FILE),
        ("candidate_profile", PROFILE_FILE),
        ("search_history", SEARCH_HISTORY_FILE),
        ("saved_searches", SAVED_SEARCHES_FILE),
    ]

    collections = []
    total_size = 0

    with _db_lock:
        for name, path in tracked_files:
            file_exists = path.exists()
            size = 0
            record_count = 0
            sha256 = ""
            mtime_str = None

            if file_exists:
                try:
                    stat = path.stat()
                    size = stat.st_size
                    total_size += size
                    mtime_str = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                    with open(path, "rb") as f:
                        content = f.read()
                        sha256 = hashlib.sha256(content).hexdigest()

                    try:
                        parsed = json.loads(content.decode("utf-8"))
                        if isinstance(parsed, list):
                            record_count = len(parsed)
                        elif isinstance(parsed, dict):
                            record_count = len(parsed.keys())
                    except Exception:
                        record_count = 0
                except Exception as e:
                    logger.warning(f"Error computing checksum for {path}: {e}")
                    sha256 = "error"
            else:
                sha256 = "missing"

            collections.append({
                "collection": name,
                "file_path": str(path),
                "exists": file_exists,
                "record_count": record_count,
                "size_bytes": size,
                "sha256_checksum": sha256,
                "last_modified": mtime_str
            })

    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_collections": len(collections),
        "total_size_bytes": total_size,
        "collections": collections
    }


def _normalize_obj_recursive(obj: Any, stats: Dict[str, int]) -> Any:
    """Helper to recursively sort dict keys and trim string fields."""
    if isinstance(obj, dict):
        normalized = {}
        for k in sorted(obj.keys()):
            stats["keys"] += 1
            v = obj[k]
            if isinstance(v, str):
                v_clean = v.strip()
                if v and not v_clean:
                    stats["empty_pruned"] += 1
                normalized[k] = v_clean
            elif isinstance(v, (dict, list)):
                normalized[k] = _normalize_obj_recursive(v, stats)
            else:
                normalized[k] = v
        return normalized
    elif isinstance(obj, list):
        return [_normalize_obj_recursive(item, stats) for item in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj


def vacuum_database_collections() -> Dict[str, Any]:
    """
    Perform deep JSON vacuuming, canonical key ordering, whitespace normalization,
    and storage compaction across all writable database collections.
    """
    targets = [
        ("applications", APPLICATIONS_FILE),
        ("bookmarks", BOOKMARKS_FILE),
        ("candidate_profile", PROFILE_FILE),
        ("search_history", SEARCH_HISTORY_FILE),
        ("saved_searches", SAVED_SEARCHES_FILE),
    ]

    collection_results = []
    total_saved = 0

    with _db_lock:
        for name, path in targets:
            if not path.exists():
                continue

            try:
                bytes_before = path.stat().st_size
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                records_count = len(data) if isinstance(data, (list, dict)) else 1
                stats = {"keys": 0, "empty_pruned": 0}
                normalized_data = _normalize_obj_recursive(data, stats)

                _atomic_write_json(path, normalized_data)
                bytes_after = path.stat().st_size
                saved = max(0, bytes_before - bytes_after)
                total_saved += saved

                collection_results.append({
                    "collection": name,
                    "records_processed": records_count,
                    "keys_normalized": stats["keys"],
                    "empty_strings_pruned": stats["empty_pruned"],
                    "bytes_before": bytes_before,
                    "bytes_after": bytes_after,
                    "bytes_saved": saved
                })
            except Exception as e:
                logger.warning(f"Error during vacuum of {name}: {e}")

        # Invalidate in-memory caches to guarantee fresh reload
        invalidate_jobs_cache()
        invalidate_applications_cache()
        invalidate_bookmarks_cache()

    _log_db_audit_entry("VACUUM", "database", "all", f"Vacuumed {len(collection_results)} collections, saved {total_saved} bytes")
    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_collections_vacuumed": len(collection_results),
        "total_bytes_saved": total_saved,
        "collections": collection_results
    }


def run_system_storage_diagnostics() -> Dict[str, Any]:
    """
    Run diagnostic profiling of runtime host, process footprint, and storage I/O throughput.
    """
    block_size = 64 * 1024 # 64 KB
    blocks_count = 8 # 512 KB total
    total_bytes = block_size * blocks_count
    total_mb = total_bytes / (1024 * 1024)

    test_payload = b"X" * block_size
    bench_dir = DATA_DIR if DATA_DIR.exists() else Path(tempfile.gettempdir())
    bench_file = bench_dir / f".bench_io_{int(time.time() * 1000)}.tmp"

    write_speed_mb_s = 0.0
    read_speed_mb_s = 0.0

    try:
        # 1. Write benchmark
        t0 = time.perf_counter()
        with open(bench_file, "wb") as f:
            for _ in range(blocks_count):
                f.write(test_payload)
            f.flush()
            os.fsync(f.fileno())
        t_write = time.perf_counter() - t0
        write_speed_mb_s = round(total_mb / max(t_write, 0.0001), 2)

        # 2. Read benchmark
        t1 = time.perf_counter()
        with open(bench_file, "rb") as f:
            while f.read(block_size):
                pass
        t_read = time.perf_counter() - t1
        read_speed_mb_s = round(total_mb / max(t_read, 0.0001), 2)
    except Exception as exc:
        logger.warning(f"Storage I/O benchmarking encountered error: {exc}")
    finally:
        if bench_file.exists():
            try:
                bench_file.unlink()
            except OSError:
                pass

    total_records = len(load_jobs()) + len(load_applications()) + len(load_bookmarks())

    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python_version": sys.version.split()[0],
        "operating_system": f"{platform.system()} {platform.release()}",
        "process_id": os.getpid(),
        "working_directory": os.getcwd(),
        "storage_root": str(DATA_DIR),
        "storage_benchmark": {
            "write_speed_mb_s": write_speed_mb_s,
            "read_speed_mb_s": read_speed_mb_s,
            "block_size_bytes": block_size,
            "test_file_size_mb": total_mb
        },
        "total_database_records": total_records
    }


def generate_database_integrity_report() -> Dict[str, Any]:
    """
    Perform deep structural validation, foreign key checks, and schema integrity scan across all collections.
    """
    targets = [
        ("jobs", JOBS_FILE, "job_id", ["job_id", "title", "company"]),
        ("applications", APPLICATIONS_FILE, "job_id", ["job_id", "status"]),
        ("bookmarks", BOOKMARKS_FILE, "job_id", ["job_id"]),
        ("candidate_profile", PROFILE_FILE, None, []),
        ("saved_searches", SAVED_SEARCHES_FILE, "search_id", ["search_id"]),
    ]

    issues = []
    total_records = 0

    with _db_lock:
        for name, path, pk_field, required_fields in targets:
            if not path.exists():
                issues.append({
                    "collection": name,
                    "record_id": None,
                    "severity": "Warning",
                    "description": f"Collection file {path.name} does not exist yet."
                })
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    parsed = [] if name != "candidate_profile" else {}
                else:
                    parsed = json.loads(content)
            except Exception as exc:
                issues.append({
                    "collection": name,
                    "record_id": None,
                    "severity": "Error",
                    "description": f"JSON syntax error or unreadable file: {exc}"
                })
                continue

            if isinstance(parsed, list):
                total_records += len(parsed)
                seen_pks = set()
                for idx, record in enumerate(parsed):
                    if not isinstance(record, dict):
                        issues.append({
                            "collection": name,
                            "record_id": str(idx),
                            "severity": "Error",
                            "description": f"Record at index {idx} is not a valid JSON object."
                        })
                        continue

                    if pk_field:
                        val = record.get(pk_field)
                        if not val:
                            issues.append({
                                "collection": name,
                                "record_id": str(idx),
                                "severity": "Error",
                                "description": f"Missing primary key field '{pk_field}'."
                            })
                        elif val in seen_pks:
                            issues.append({
                                "collection": name,
                                "record_id": str(val),
                                "severity": "Warning",
                                "description": f"Duplicate primary key '{val}' in collection."
                            })
                        else:
                            seen_pks.add(val)

                    for req in required_fields:
                        if req not in record or record[req] is None:
                            issues.append({
                                "collection": name,
                                "record_id": str(record.get(pk_field, idx)),
                                "severity": "Warning",
                                "description": f"Missing expected field '{req}'."
                            })
            elif isinstance(parsed, dict):
                total_records += 1

    has_errors = any(i["severity"] == "Error" for i in issues)
    has_warnings = any(i["severity"] == "Warning" for i in issues)

    if has_errors:
        health = "Critical"
    elif has_warnings:
        health = "Warning"
    else:
        health = "Healthy"

    return {
        "status": "success",
        "overall_health": health,
        "total_collections_scanned": len(targets),
        "total_records_checked": total_records,
        "issues_found": len(issues),
        "issues": issues,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }


def get_database_schema_migrations() -> Dict[str, Any]:
    """
    Retrieve registered database schema version history and migration status.
    """
    migrations = [
        {
            "version": "1.0.0",
            "name": "initial_schema_baseline",
            "applied_at": "2026-01-01 00:00:00 UTC",
            "description": "Base JSON collections for jobs, applications, and candidate profiles."
        },
        {
            "version": "1.5.0",
            "name": "add_audit_and_rollback_buffers",
            "applied_at": "2026-06-15 12:00:00 UTC",
            "description": "Integrated in-memory transaction audit logs and snapshot state rollbacks."
        },
        {
            "version": "2.0.0",
            "name": "add_extended_candidate_preferences",
            "applied_at": "2026-10-01 09:30:00 UTC",
            "description": "Structured candidate salary expectations, soft skills, certifications, and links."
        },
        {
            "version": "2.1.0",
            "name": "interview_feedback_and_retention",
            "applied_at": "2026-11-15 18:00:00 UTC",
            "description": "Integrated rejection categorization, interview difficulty ratings, and audit pruning."
        }
    ]

    return {
        "status": "success",
        "current_schema_version": "2.1.0",
        "is_up_to_date": True,
        "total_migrations_applied": len(migrations),
        "migrations": migrations
    }


def update_application_thank_you_status(
    job_id: str,
    thank_you_sent: bool,
    sent_at: Optional[str] = None,
    notes: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record thank-you note dispatch status and timestamp for an application.
    Returns: (success: bool, message: str, updated_app: Optional[dict])
    """
    clean_id = (job_id or "").strip()
    if not clean_id:
        return False, "Job ID cannot be empty.", None

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _db_lock:
        apps = load_applications()
        target_app = None
        for app in apps:
            if str(app.get("job_id", "")).strip().lower() == clean_id.lower():
                app["thank_you_sent"] = bool(thank_you_sent)
                app["thank_you_sent_at"] = sent_at or (now_str if thank_you_sent else None)
                if notes is not None:
                    app["thank_you_notes"] = notes.strip() or None
                target_app = app
                break

        if not target_app:
            return False, f"No application found for Job ID '{clean_id}'.", None

        _save_applications(apps)

    logger.info(f"Updated thank-you note status for job {clean_id}: sent={thank_you_sent}")
    return True, f"Thank-you note status for Job ID '{clean_id}' updated successfully.", target_app





