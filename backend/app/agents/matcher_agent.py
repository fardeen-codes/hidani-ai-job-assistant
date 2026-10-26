"""
Sub-Agent 2: Matcher Agent.
Implements a hybrid matching architecture:
1. Lexical and semantic skill-overlap scoring to filter candidates from the 1600+ job database.
2. LLM-based evaluation to generate personalized 'match_reason' and calibrated 'match_score'.
Includes heuristic fallback for instantaneous offline execution.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, is_openai_configured
from app.schemas import Job
from app.services.db_service import load_jobs, search_jobs

logger = logging.getLogger("MatcherAgent")


def _calculate_lexical_score(job: Dict[str, Any], candidate_skills: List[str], candidate_level: str, query: str = "") -> float:
    """
    Calculate a normalized match score (0-100) based on skill overlap,
    experience level alignment, and title keyword match.
    """
    score = 40.0 # Base relevance
    
    cand_skills_lower = {s.strip().lower() for s in candidate_skills if s.strip()}
    job_skills_lower = {s.strip().lower() for s in job.get("skills", []) if s.strip()}
    
    # 1. Skill Overlap (up to 40 points)
    if cand_skills_lower and job_skills_lower:
        overlap = cand_skills_lower.intersection(job_skills_lower)
        overlap_ratio = len(overlap) / max(len(job_skills_lower), 1)
        score += min(overlap_ratio * 40.0, 40.0)
    elif cand_skills_lower:
        # Check if candidate skills appear in job description
        desc_lower = job.get("description", "").lower()
        matches = sum(1 for s in cand_skills_lower if s in desc_lower)
        score += min(matches * 6.0, 30.0)

    # 2. Experience Level Alignment (up to 10 points)
    job_exp = job.get("experience", "").lower()
    cand_exp = (candidate_level or "mid-level").lower()
    if cand_exp in job_exp or job_exp in cand_exp:
        score += 10.0
    elif "mid" in cand_exp and "senior" in job_exp:
        score += 5.0

    # 3. Query / Title relevance (up to 10 points)
    if query:
        q_clean = query.lower()
        title_lower = job.get("title", "").lower()
        if q_clean in title_lower:
            score += 10.0
        elif any(word in title_lower for word in q_clean.split()):
            score += 5.0

    return min(round(score, 1), 99.0)


def _generate_heuristic_reason(job: Dict[str, Any], candidate_skills: List[str], score: float) -> str:
    """Generate clear, personalized explanation without LLM."""
    cand_skills_lower = {s.strip().lower(): s for s in candidate_skills if s.strip()}
    job_skills_lower = {s.strip().lower() for s in job.get("skills", []) if s.strip()}
    
    overlapping = [cand_skills_lower[k] for k in cand_skills_lower if k in job_skills_lower]
    
    if overlapping:
        return (
            f"Strong fit with your stack: overlaps with {len(overlapping)} of your core skills "
            f"({', '.join(overlapping[:3])}) and aligns with your {job.get('experience', 'tech')} experience."
        )
    return (
        f"Matches your background in software engineering with relevant scope in {job.get('title')} "
        f"at {job.get('company')} ({job.get('location')})."
    )


def match_jobs_for_candidate(
    candidate_skills: Optional[List[str]] = None,
    candidate_level: Optional[str] = None,
    query: Optional[str] = None,
    top_k: int = 5
) -> List[Job]:
    """
    Sub-Agent 2 execution:
    Finds top matching jobs using hybrid text scoring, then enriches them
    with calibrated scores and personalized match reasons.
    """
    candidate_skills = candidate_skills or []
    candidate_level = candidate_level or "Mid-Level"

    # Step 1: Query database using db_service candidate retrieval
    candidates = search_jobs(
        skills=candidate_skills,
        query=query,
        experience_level=candidate_level,
        limit=20
    )

    if not candidates:
        # Fallback to load default jobs if filters returned none
        candidates = load_jobs()[:20]

    if not candidates:
        logger.warning("No jobs available in database for matching.")
        return []

    # Step 2: Score candidates
    scored_candidates = []
    for j in candidates:
        score = _calculate_lexical_score(j, candidate_skills, candidate_level, query or "")
        scored_candidates.append((score, j))

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:top_k]

    # Step 3: LLM reasoning or fallback reasoning
    if not is_openai_configured():
        results = []
        for score, j in top_candidates:
            reason = _generate_heuristic_reason(j, candidate_skills, score)
            results.append(Job(
                job_id=j.get("job_id"),
                title=j.get("title"),
                company=j.get("company"),
                location=j.get("location"),
                experience=j.get("experience"),
                skills=j.get("skills", []),
                description=j.get("description"),
                source_link=j.get("source_link"),
                match_score=score,
                match_reason=reason
            ))
        return results

    # LLM-enhanced match reasons
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        jobs_summary = [
            {
                "index": i,
                "job_id": j.get("job_id"),
                "title": j.get("title"),
                "company": j.get("company"),
                "skills": j.get("skills", []),
                "experience": j.get("experience")
            }
            for i, (_, j) in enumerate(top_candidates)
        ]

        system_prompt = (
            "You are an AI Talent Matcher sub-agent. "
            "Given candidate skills and a list of job listings, generate a concise, compelling 1-2 sentence 'match_reason' "
            "for each job explaining why this candidate is a strong fit. "
            "Output JSON with a list under 'matches', where each item has 'job_id', 'match_score' (number 70-98), and 'match_reason'."
        )

        user_prompt = (
            f"Candidate Skills: {', '.join(candidate_skills)}\n"
            f"Candidate Level: {candidate_level}\n"
            f"Jobs to evaluate:\n{json.dumps(jobs_summary, indent=2)}"
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=600
        )

        parsed = json.loads(response.choices[0].message.content or "{}")
        llm_matches = {m.get("job_id"): m for m in parsed.get("matches", []) if m.get("job_id")}

        results = []
        for lexical_score, j in top_candidates:
            jid = j.get("job_id")
            llm_info = llm_matches.get(jid, {})
            score = float(llm_info.get("match_score", lexical_score))
            reason = llm_info.get("match_reason") or _generate_heuristic_reason(j, candidate_skills, score)

            results.append(Job(
                job_id=j.get("job_id"),
                title=j.get("title"),
                company=j.get("company"),
                location=j.get("location"),
                experience=j.get("experience"),
                skills=j.get("skills", []),
                description=j.get("description"),
                source_link=j.get("source_link"),
                match_score=round(score, 1),
                match_reason=reason
            ))
        return results

    except Exception as exc:
        logger.error(f"LLM match reasoning failed: {exc}. Using fallback reasoning.")
        results = []
        for score, j in top_candidates:
            reason = _generate_heuristic_reason(j, candidate_skills, score)
            results.append(Job(
                job_id=j.get("job_id"),
                title=j.get("title"),
                company=j.get("company"),
                location=j.get("location"),
                experience=j.get("experience"),
                skills=j.get("skills", []),
                description=j.get("description"),
                source_link=j.get("source_link"),
                match_score=score,
                match_reason=reason
            ))
        return results
