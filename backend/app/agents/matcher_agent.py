"""
Sub-Agent 2: Matcher Agent.
Implements a hybrid matching architecture:
1. Lexical and semantic skill-overlap scoring to filter candidates from the 1600+ job database.
2. LLM-based evaluation to generate personalized 'match_reason' and calibrated 'match_score'.
Includes heuristic fallback for instantaneous offline execution.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, is_openai_configured
from app.schemas import Job
from app.services.db_service import get_job_by_id, load_jobs, search_jobs

logger = logging.getLogger("MatcherAgent")

_SKILL_ALIASES = {
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "nextjs": "next.js",
    "vuejs": "vue.js",
    "golang": "go",
    "postgres": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "py": "python",
}


def _normalize_skill(skill: str) -> str:
    """Normalize skill string to canonical lowercase form using alias map."""
    clean = (skill or "").strip().lower()
    return _SKILL_ALIASES.get(clean, clean)


def _calculate_lexical_score(job: Dict[str, Any], candidate_skills: List[str], candidate_level: str, query: str = "") -> float:
    """
    Calculate a normalized match score (0-100) based on skill overlap,
    experience level alignment, and title keyword match.
    """
    score = 40.0 # Base relevance
    
    cand_skills_lower = {_normalize_skill(s) for s in (candidate_skills or []) if s and s.strip()}
    job_skills_lower = {_normalize_skill(s) for s in (job.get("skills") or []) if s and s.strip()}
    
    # 1. Skill Overlap (up to 40 points)
    if cand_skills_lower and job_skills_lower:
        overlap = cand_skills_lower.intersection(job_skills_lower)
        overlap_ratio = len(overlap) / max(len(job_skills_lower), 1)
        score += min(overlap_ratio * 40.0, 40.0)
    elif cand_skills_lower:
        # Check if candidate skills appear in job description
        desc_lower = (job.get("description") or "").lower()
        matches = sum(1 for s in cand_skills_lower if s in desc_lower)
        score += min(matches * 6.0, 30.0)

    # 2. Experience Level Alignment (up to 10 points)
    job_exp = (job.get("experience") or "").lower()
    cand_exp = (candidate_level or "mid-level").lower()
    if cand_exp in job_exp or job_exp in cand_exp:
        score += 10.0
    elif "mid" in cand_exp and "senior" in job_exp:
        score += 5.0

    # 3. Query / Title relevance (up to 10 points)
    if query:
        q_clean = query.strip().lower()
        title_lower = (job.get("title") or "").lower()
        if q_clean in title_lower:
            score += 10.0
        elif any(word in title_lower for word in q_clean.split()):
            score += 5.0

    return min(round(score, 1), 99.0)


def _generate_heuristic_reason(job: Dict[str, Any], candidate_skills: List[str], candidate_designation: str = "", score: float = 85.0) -> str:
    """Generate clear, personalized explanation without LLM."""
    cand_skills_lower = {_normalize_skill(s): s for s in (candidate_skills or []) if s and s.strip()}
    job_skills_lower = {_normalize_skill(s) for s in (job.get("skills") or []) if s and s.strip()}
    
    overlapping = [cand_skills_lower[k] for k in cand_skills_lower if k in job_skills_lower]
    
    role_mention = f"Direct alignment with your {candidate_designation} profile: " if candidate_designation else ""
    
    if overlapping:
        return (
            f"{role_mention}Matches {len(overlapping)} of your core skills "
            f"({', '.join(overlapping[:3])}) and fits your {job.get('experience') or 'engineering'} experience tier."
        )
    return (
        f"{role_mention}Aligns with your engineering background with impactful responsibilities in {job.get('title') or 'Software Engineering'} "
        f"at {job.get('company') or 'the company'} ({job.get('location') or 'Remote'})."
    )


def match_jobs_for_candidate(
    candidate_skills: Optional[List[str]] = None,
    candidate_designation: Optional[str] = None,
    candidate_level: Optional[str] = None,
    query: Optional[str] = None,
    location: Optional[str] = None,
    top_k: int = 5
) -> List[Job]:
    """
    Sub-Agent 2 execution:
    Finds top matching jobs using hybrid text scoring matching designation, skills, and experience,
    then enriches them with calibrated scores and personalized match reasons.
    """
    candidate_skills = candidate_skills or []
    candidate_level = candidate_level or "Mid-Level"
    candidate_designation = candidate_designation or "Software Engineer"

    # Step 1: Query database using candidate retrieval
    candidates = search_jobs(
        skills=candidate_skills,
        designation=candidate_designation,
        query=query,
        experience_level=candidate_level,
        location=location,
        limit=25
    )

    if not candidates:
        # Fallback to general query if strict filters returned empty
        candidates = search_jobs(
            skills=candidate_skills,
            query=query,
            experience_level=candidate_level,
            limit=25
        )

    if not candidates:
        candidates = load_jobs()[:25]

    if not candidates:
        logger.warning("No jobs available in database for matching.")
        return []

    # Step 2: Score candidates
    scored_candidates = []
    for j in candidates:
        score = _calculate_lexical_score(j, candidate_skills, candidate_level, query or "")
        # Boost if designation matches job title
        if candidate_designation and candidate_designation.lower() in j.get("title", "").lower():
            score = min(score + 8.0, 99.0)
        scored_candidates.append((score, j))

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:top_k]

    # Step 3: LLM reasoning or fallback reasoning
    if not is_openai_configured():
        results = []
        for score, j in top_candidates:
            reason = _generate_heuristic_reason(j, candidate_skills, candidate_designation, score)
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
            reason = llm_info.get("match_reason") or _generate_heuristic_reason(j, candidate_skills, candidate_designation, score)

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
            reason = _generate_heuristic_reason(j, candidate_skills, candidate_designation, score)
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


def analyze_skill_gap(
    job_id: str,
    candidate_skills: Optional[List[str]] = None,
    candidate_level: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Sub-Agent 2 execution: Analyze skill gap and readiness for a specific job posting.
    Computes matching skills, missing skills, match score, readiness category, and advice.
    """
    from app.services.db_service import get_job_by_id

    clean_id = (job_id or "").strip()
    if not clean_id:
        return None

    job = get_job_by_id(clean_id)
    if not job:
        return None

    candidate_skills = candidate_skills or []
    cand_norm_map = {_normalize_skill(s): s for s in candidate_skills if s and s.strip()}

    job_skills = job.get("skills") or []
    matching_skills: List[str] = []
    missing_skills: List[str] = []

    for s in job_skills:
        clean_s = s.strip()
        norm_s = _normalize_skill(clean_s)
        if norm_s in cand_norm_map:
            matching_skills.append(clean_s)
        else:
            missing_skills.append(clean_s)

    score = _calculate_lexical_score(job, candidate_skills, candidate_level or "Mid-Level")

    if score >= 80 or (job_skills and len(missing_skills) == 0):
        readiness = "High Readiness"
        rec = f"Strong fit! You meet {len(matching_skills)} of {len(job_skills)} required skills. You are well-positioned to apply."
    elif score >= 55 or len(matching_skills) > 0:
        readiness = "Moderate Readiness"
        missing_preview = ", ".join(missing_skills[:3]) if missing_skills else "advanced concepts"
        rec = f"Promising match. Consider highlighting your {', '.join(matching_skills[:2]) if matching_skills else 'experience'} and brushing up on {missing_preview}."
    else:
        readiness = "Needs Preparation"
        missing_preview = ", ".join(missing_skills[:3]) if missing_skills else "core technologies"
        rec = f"This position requires skills not currently in your primary profile ({missing_preview}). Additional coursework or projects are recommended."

    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("title"),
        "company": job.get("company"),
        "match_score": score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "readiness": readiness,
        "recommendation": rec
    }


def batch_score_jobs(
    profile: Optional[Dict[str, Any]] = None,
    candidate_skills: Optional[List[str]] = None,
    job_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate candidate compatibility, match scores, and skill overlap across multiple job IDs in batch.
    """
    effective_skills = list(candidate_skills or [])
    level = "Mid-Level"
    if profile:
        if not effective_skills and profile.get("skills"):
            effective_skills = profile.get("skills", [])
        level = profile.get("experience_level") or profile.get("experience") or "Mid-Level"

    cand_norm_map = {_normalize_skill(s): s for s in effective_skills if s and s.strip()}

    results = []
    for jid in (job_ids or []):
        job = get_job_by_id(jid)
        if not job:
            continue

        job_skills = job.get("skills") or []
        matching = []
        missing = []

        for s in job_skills:
            clean_s = str(s).strip()
            norm_s = _normalize_skill(clean_s)
            if norm_s in cand_norm_map:
                matching.append(clean_s)
            else:
                missing.append(clean_s)

        score = _calculate_lexical_score(job, effective_skills, level)

        results.append({
            "job_id": job.get("job_id"),
            "job_title": job.get("title", ""),
            "company": job.get("company", ""),
            "match_score": round(score, 1),
            "matching_skills": matching,
            "missing_skills": missing
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


_SKILL_CURRICULUM_TEMPLATES = {
    "kubernetes": {
        "topics": ["Pod Lifecycle & Deployments", "Services & Ingress", "StatefulSets & Helm", "CRDs & Operators"],
        "hours": 30,
        "priority": "High",
        "project": "Deploy a multi-tier microservice architecture to local Minikube using Helm charts."
    },
    "docker": {
        "topics": ["Dockerfile Multi-stage Builds", "Compose Orchestration", "Networking & Volumes", "Image Minimization"],
        "hours": 15,
        "priority": "High",
        "project": "Containerize a polyglot application with optimized multi-stage build caching."
    },
    "aws": {
        "topics": ["IAM & VPC Security", "ECS / Fargate", "S3 & Lambda Serverless", "CloudWatch Observability"],
        "hours": 35,
        "priority": "High",
        "project": "Provision an automated serverless data processing pipeline using Terraform and AWS Lambda."
    },
    "graphql": {
        "topics": ["Schema Definition Language (SDL)", "Resolvers & DataLoader (N+1)", "Subscriptions", "Apollo Federation"],
        "hours": 20,
        "priority": "Medium",
        "project": "Build a unified GraphQL gateway with batching and caching over downstream REST services."
    },
    "kafka": {
        "topics": ["Producers, Consumers & Offsets", "Partitioning & Consumer Groups", "Kafka Streams", "Schema Registry & Avro"],
        "hours": 25,
        "priority": "High",
        "project": "Implement real-time event streaming and dead-letter queue recovery mechanism."
    },
    "redis": {
        "topics": ["Data Structures & TTLs", "Pub/Sub & Streams", "Caching Strategies & Cache Invalidation", "Redis Cluster & Persistence"],
        "hours": 15,
        "priority": "Medium",
        "project": "Implement a distributed rate limiter and session cache with sliding window logic."
    },
    "typescript": {
        "topics": ["Generics & Conditional Types", "Utility Types & Mapped Types", "Strict Null Checking", "Compiler Options & Tooling"],
        "hours": 18,
        "priority": "High",
        "project": "Refactor a dynamic JavaScript library to fully typed TypeScript with strict compiler validation."
    },
    "python": {
        "topics": ["Asyncio & Concurrency", "Metaclasses & Descriptors", "Type Annotations & Pydantic", "Profiling & Memory Optimization"],
        "hours": 25,
        "priority": "High",
        "project": "Develop a high-throughput asynchronous web crawler using asyncio and aiohttp."
    },
    "postgresql": {
        "topics": ["Indexing Strategies & EXPLAIN ANALYZE", "Transactions & Isolation Levels", "Partitioning & Vacuuming", "Connection Pooling"],
        "hours": 20,
        "priority": "High",
        "project": "Optimize slow multi-join query performance with composite B-Tree and GIN indexes."
    },
    "react": {
        "topics": ["Hooks & Custom Hooks", "Server Components & Suspense", "State Management & Context", "Performance Profiling & Memoization"],
        "hours": 25,
        "priority": "High",
        "project": "Build an interactive dashboard with optimistic UI updates and virtualization."
    }
}


def generate_learning_roadmap(missing_skills: List[str], target_role: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate an actionable technical learning roadmap with study topics, estimated effort, and project ideas.
    """
    seen = set()
    cleaned_skills = []
    for s in missing_skills:
        clean = (s or "").strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            cleaned_skills.append(clean)

    roadmap = []
    total_hours = 0

    for skill in cleaned_skills:
        norm = _normalize_skill(skill)
        template = _SKILL_CURRICULUM_TEMPLATES.get(norm)
        if not template:
            # Check if any template key is a substring
            for k, val in _SKILL_CURRICULUM_TEMPLATES.items():
                if k in norm:
                    template = val
                    break

        if template:
            hours = template["hours"]
            item = {
                "skill": skill,
                "priority": template["priority"],
                "suggested_topics": list(template["topics"]),
                "estimated_hours": hours,
                "recommended_project_idea": template["project"]
            }
        else:
            role_ctx = f" for {target_role}" if target_role else ""
            hours = 20
            item = {
                "skill": skill,
                "priority": "Medium",
                "suggested_topics": [
                    f"Core Fundamentals & Architecture of {skill}",
                    f"Best Practices & Design Patterns for {skill}{role_ctx}",
                    f"Integration & Testing with {skill}",
                    f"Production Deployment & Monitoring for {skill}"
                ],
                "estimated_hours": hours,
                "recommended_project_idea": f"Design and implement a standalone proof-of-concept system showcasing {skill} in a production-ready application."
            }

        roadmap.append(item)
        total_hours += hours

    return {
        "status": "success",
        "total_skills": len(roadmap),
        "total_estimated_hours": total_hours,
        "roadmap": roadmap
    }


def calculate_weighted_skill_match(
    job_id: str,
    candidate_skills: List[str],
    skill_weights: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate custom weighted skill match score where candidate prioritizes specific technologies.
    """
    job = get_job_by_id(job_id)
    if not job:
        return None

    raw_weights = skill_weights or {}
    norm_weights = {}
    for k, v in raw_weights.items():
        if k:
            norm_weights[_normalize_skill(str(k))] = max(float(v), 0.1)

    cand_skills = [s.strip() for s in (candidate_skills or []) if s and s.strip()]
    cand_norm_set = {_normalize_skill(s) for s in cand_skills}

    job_skills = job.get("skills") or []
    matching_skills = []
    missing_skills = []

    total_weight = 0.0
    matched_weight = 0.0
    applied_weights = {}

    for s in job_skills:
        clean_s = str(s).strip()
        norm_s = _normalize_skill(clean_s)
        w = norm_weights.get(norm_s, 1.0)
        applied_weights[clean_s] = w
        total_weight += w

        if norm_s in cand_norm_set:
            matching_skills.append(clean_s)
            matched_weight += w
        else:
            missing_skills.append(clean_s)

    # Standard unweighted lexical score
    standard_score = _calculate_lexical_score(job, cand_skills, job.get("experience", "Mid-Level"))

    # Weighted match calculation
    if total_weight > 0:
        weighted_ratio = matched_weight / total_weight
        weighted_score = min(round(35.0 + (weighted_ratio * 60.0), 1), 99.0)
    else:
        weighted_score = standard_score

    return {
        "status": "success",
        "job_id": job.get("job_id", job_id),
        "job_title": job.get("title", ""),
        "company": job.get("company", ""),
        "standard_match_score": standard_score,
        "weighted_match_score": weighted_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "skill_weights_applied": applied_weights
    }


def _get_seniority_level(title: str) -> int:
    """Helper to detect approximate seniority tier from title."""
    t = title.lower()
    if any(k in t for k in ["vp", "director", "head of", "chief", "manager"]):
        return 5
    if any(k in t for k in ["staff", "principal", "architect", "lead"]):
        return 4
    if any(k in t for k in ["senior", "sr.", "sr "]):
        return 3
    if any(k in t for k in ["intern", "graduate", "junior", "jr.", "entry"]):
        return 1
    return 2 # Mid-level default


def score_career_transition(
    current_title: str,
    target_title: str,
    current_experience_years: float = 3.0
) -> Dict[str, Any]:
    """
    Evaluate career transition feasibility, trajectory difficulty, and recommend bridge competencies.
    """
    cur_clean = (current_title or "").strip()
    tgt_clean = (target_title or "").strip()

    cur_words = set(re.findall(r"[a-zA-Z]+", cur_clean.lower()))
    tgt_words = set(re.findall(r"[a-zA-Z]+", tgt_clean.lower()))

    # Ignore generic filler words
    stopwords = {"and", "of", "the", "in", "for", "with", "at", "to", "engineer", "specialist"}
    cur_core = cur_words - stopwords or cur_words
    tgt_core = tgt_words - stopwords or tgt_words

    intersection = cur_core & tgt_core
    union = cur_core | tgt_core
    jaccard = len(intersection) / max(len(union), 1)

    cur_seniority = _get_seniority_level(cur_clean)
    tgt_seniority = _get_seniority_level(tgt_clean)
    seniority_delta = tgt_seniority - cur_seniority

    base_similarity = round(jaccard * 50.0 + 35.0, 1)

    if seniority_delta > 0:
        if jaccard >= 0.3:
            transition_type = "Seniority Promotion"
            if seniority_delta == 1:
                difficulty = "Moderate"
                base_similarity = min(base_similarity + 10.0, 92.0)
                bridge_skills = ["System Architecture Design", "Technical Mentorship", "Cross-Functional Project Ownership"]
                advice = f"Your background strongly positions you for a step up to {tgt_clean}. Emphasize projects where you drove technical decisions and mentored teammates."
            else:
                difficulty = "High"
                base_similarity = max(base_similarity - 10.0, 45.0)
                bridge_skills = ["Large-Scale Distributed Systems", "Staff-Level Architectural Strategy", "Executive Stakeholder Alignment"]
                advice = f"Leaping multiple seniority tiers to {tgt_clean} requires demonstrating enterprise-scale system ownership and organization-wide influence."
        else:
            transition_type = "Domain Shift"
            difficulty = "High"
            bridge_skills = ["Core Domain Tooling", "Cross-Domain Translation", "Proof of Concept Portfolio"]
            advice = f"Transitioning from {cur_clean} to {tgt_clean} spans both a domain pivot and higher seniority. Build high-visibility portfolio projects in the new domain."
    elif jaccard >= 0.4:
        transition_type = "Lateral Move"
        difficulty = "Low"
        base_similarity = min(base_similarity + 15.0, 95.0)
        bridge_skills = ["Tech Stack Translation", "Domain-Specific Paradigms", "Agile Execution Adaptability"]
        advice = f"Direct lateral alignment. Your core skillset translates naturally to {tgt_clean}."
    elif jaccard >= 0.15:
        transition_type = "Domain Shift"
        difficulty = "Moderate"
        base_similarity = min(max(base_similarity, 50.0), 75.0)
        bridge_skills = ["Target Domain Foundations", "Hands-On Integration Projects", "Domain Terminology & RFCs"]
        advice = f"A solid adjacent transition. Highlight foundational engineering strengths while closing specific {tgt_clean} domain nuances."
    else:
        transition_type = "Radical Pivot"
        difficulty = "Very High"
        base_similarity = min(max(base_similarity - 15.0, 20.0), 45.0)
        bridge_skills = ["Foundational Domain Bootcamp", "End-to-End Capstone Projects", "Open-Source Contributions in New Field"]
        advice = f"This represents a substantial career pivot. Dedicate focused study to bridge skills and develop public proof-of-work repositories."

    return {
        "status": "success",
        "current_title": cur_clean,
        "target_title": tgt_clean,
        "similarity_score": round(base_similarity, 1),
        "transition_type": transition_type,
        "estimated_difficulty": difficulty,
        "recommended_bridge_skills": bridge_skills,
        "key_advice": advice
    }


def estimate_interview_difficulty(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Estimate technical interview difficulty score (1.0 to 5.0) and complexity breakdown.
    """
    job = get_job_by_id(job_id)
    if not job:
        return None

    title = job.get("title", "")
    exp = (job.get("experience") or "").lower()
    skills = [str(s).lower() for s in (job.get("skills") or [])]
    desc = (job.get("description") or "").lower()

    # Determine seniority baseline
    sen = _get_seniority_level(title)
    if sen >= 4 or "lead" in exp or "staff" in exp or "principal" in exp:
        score = 4.4
        tier = "Expert" if sen >= 5 else "Rigorous"
    elif sen == 3 or "senior" in exp or "sr" in exp:
        score = 3.8
        tier = "Challenging"
    elif sen == 1 or "junior" in exp or "entry" in exp or "intern" in exp:
        score = 2.0
        tier = "Introductory"
    else:
        score = 2.9
        tier = "Standard"

    # Evaluate specialized technologies
    complex_tech = {"kubernetes", "distributed", "concurrency", "microservices", "kafka", "rust", "c++", "security", "ml", "machine learning"}
    found_complex = [s for s in skills if any(c in s for c in complex_tech)]
    if found_complex:
        score = min(score + 0.3, 5.0)

    # Recalibrate tier based on adjusted score
    if score >= 4.5:
        tier = "Expert"
    elif score >= 3.8:
        tier = "Rigorous"
    elif score >= 3.0:
        tier = "Challenging"
    elif score >= 2.2:
        tier = "Standard"
    else:
        tier = "Introductory"

    # Stages and factors
    factors = []
    if sen >= 3:
        factors.append(f"Seniority expectations for {title} require proven architectural depth and mentoring capability")
    if found_complex:
        factors.append(f"Requires distributed/systems complexity expertise ({', '.join(found_complex[:3])})")
    if len(skills) > 5:
        factors.append(f"Extensive breadth across {len(skills)} declared technical requirements")
    if not factors:
        factors.append("Standard software engineering core data structures, algorithms, and application design")

    if score >= 4.0:
        stages = [
            "1. Technical Recruiter Screen (30m)",
            "2. Low-level Coding & Concurrency Screen (60m)",
            "3. Distributed System Architecture & Scalability Design (60m)",
            "4. Engineering Leadership & Cross-Functional Alignment (45m)",
            "5. Executive / Bar Raiser Cultural Evaluation (45m)"
        ]
        prep = [
            "High-throughput data partitioning and consistency models (CAP / PACELC)",
            "Distributed tracing, failure recovery, and idempotency patterns",
            "Staff-level technical influence stories using the STAR method"
        ]
    elif score >= 3.0:
        stages = [
            "1. Recruiter Screen (30m)",
            "2. Live Algorithm & Data Structures Coding (60m)",
            "3. Practical System Design & API Architecture (60m)",
            "4. Past Project Deep-Dive & Behavioral Fit (45m)"
        ]
        prep = [
            "Microservice boundaries, RESTful API contracts, and caching tiers",
            "Relational schema indexing and query performance optimization",
            "Clean code, test coverage, and modular design patterns"
        ]
    else:
        stages = [
            "1. Recruiter Screen (30m)",
            "2. Practical Coding & Debugging Exercise (45m)",
            "3. Core CS Fundamentals & Team Collaboration Interview (45m)"
        ]
        prep = [
            "Core language syntax, standard library idioms, and unit tests",
            "Basic HTTP protocols, CRUD APIs, and database fundamentals",
            "Clear communication and problem-solving explanation while coding"
        ]

    return {
        "status": "success",
        "job_id": job.get("job_id", job_id),
        "job_title": title,
        "company": job.get("company", ""),
        "difficulty_score": round(score, 1),
        "difficulty_tier": tier,
        "expected_interview_stages": stages,
        "key_complexity_factors": factors,
        "recommended_prep_focus": prep
    }


def evaluate_skill_proficiency_gaps(
    candidate_proficiencies: Dict[str, str],
    job_id: Optional[str] = None,
    job_required_skills: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Weight skill proficiencies (Beginner=1, Intermediate=2, Expert=3) against required skills.
    """
    level_weights = {
        "beginner": 1,
        "intermediate": 2,
        "expert": 3,
        "advanced": 3,
        "proficient": 2,
        "familiar": 1
    }

    req_skills = []
    if job_id:
        job = get_job_by_id(job_id)
        if job and job.get("skills"):
            req_skills.extend(job.get("skills", []))
    if job_required_skills:
        req_skills.extend(job_required_skills)

    normalized_req = []
    seen = set()
    for s in req_skills:
        norm = _normalize_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            normalized_req.append(s)

    if not normalized_req:
        normalized_req = ["Python", "SQL", "Git"]

    cand_norm = {}
    for k, v in candidate_proficiencies.items():
        cand_norm[_normalize_skill(k)] = (v or "Intermediate").strip().lower()

    breakdown = []
    total_weights = 0.0
    matching_count = 0
    gaps_count = 0
    missing_count = 0

    for req in normalized_req:
        norm_r = _normalize_skill(req)
        req_level = "Intermediate"
        req_val = 2

        if norm_r in cand_norm:
            cand_lvl_str = cand_norm[norm_r]
            cand_val = level_weights.get(cand_lvl_str, 2)
            cand_display = cand_lvl_str.capitalize()
            if cand_val >= req_val:
                status = "Exceeds" if cand_val > req_val else "Matches"
                weight = 1.0
                matching_count += 1
            else:
                status = "Minor Gap"
                weight = 0.7
                gaps_count += 1
        else:
            cand_display = None
            status = "Missing"
            weight = 0.0
            missing_count += 1

        total_weights += weight
        breakdown.append({
            "skill": req,
            "candidate_level": cand_display,
            "required_level": req_level,
            "weight_score": round(weight, 2),
            "gap_status": status
        })

    overall_score = round((total_weights / len(normalized_req)) * 100.0, 1) if normalized_req else 100.0

    return {
        "overall_proficiency_score": overall_score,
        "matching_skills_count": matching_count,
        "gaps_count": gaps_count,
        "missing_count": missing_count,
        "breakdown": breakdown
    }


def predict_company_interview_questions(
    company: str,
    role_title: str,
    skills: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate tailored interview question predictions based on target company tier and role domain.
    """
    clean_company = (company or "Tech Company").strip()
    clean_role = (role_title or "Software Engineer").strip()
    skills_list = [s.strip() for s in (skills or []) if s and s.strip()]

    title_lower = clean_role.lower()
    is_senior = any(w in title_lower for w in ["senior", "lead", "staff", "principal", "architect"])
    is_backend = any(w in title_lower for w in ["backend", "distributed", "infra", "platform", "cloud", "data"])

    questions = []

    if is_backend or not skills_list:
        questions.append({
            "question": f"How would you design a distributed idempotency mechanism for high-throughput transactions at {clean_company}?",
            "category": "System Design",
            "difficulty": "Hard" if is_senior else "Medium",
            "rationale": f"Core systems at {clean_company} demand strict fault tolerance and deduplication under network partitions."
        })
        questions.append({
            "question": "Explain how you handle database connection pooling and failovers during an unexpected traffic spike.",
            "category": "Domain Architecture",
            "difficulty": "Medium",
            "rationale": "Evaluates practical operational reliability and resilience under stress."
        })

    if skills_list:
        top_skill = skills_list[0]
        questions.append({
            "question": f"In {top_skill}, what is your approach to profiling memory leaks and optimizing concurrency bottlenecks?",
            "category": "Coding / Algorithms",
            "difficulty": "Hard" if is_senior else "Medium",
            "rationale": f"Tests deep mastery of {top_skill} internals beyond surface-level syntax."
        })

    questions.append({
        "question": "Describe a scenario where you strongly disagreed with an engineering decision at your previous company. How did you resolve it?",
        "category": "Behavioral & Culture",
        "difficulty": "Medium",
        "rationale": f"Validates communication maturity, cross-functional empathy, and alignment with {clean_company} values."
    })

    focus = "Distributed System Design & Concurrency" if is_backend else "Full-Stack Architecture & User Latency"
    loops = 5 if is_senior else 4

    return {
        "status": "success",
        "company": clean_company,
        "role_title": clean_role,
        "predicted_questions": questions,
        "primary_interview_focus": focus,
        "estimated_interview_loops": loops
    }




