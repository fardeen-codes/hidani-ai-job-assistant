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

