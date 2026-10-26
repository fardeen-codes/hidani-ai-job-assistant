"""
Sub-Agent 1: Parser Agent.
Uses OpenAI LLM to parse extracted resume text into structured candidate profile data
(Candidate name, email, phone, technical skills, years of experience, level, and summary).
Includes an intelligent heuristic fallback parser for offline/demo reliability.
"""

import json
import logging
import re
from typing import Any, Dict

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, is_openai_configured
from app.schemas import ResumeData

logger = logging.getLogger("ParserAgent")


def _heuristic_resume_parser(raw_text: str) -> Dict[str, Any]:
    """
    Heuristic rule-based fallback parser when LLM is unavailable.
    Uses regex patterns and keyword matching to extract profile fields.
    """
    logger.info("Using heuristic rule-based resume parser fallback.")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_text)
    email = email_match.group(0) if email_match else None

    # Extract Phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}", raw_text)
    phone = phone_match.group(0) if phone_match else None

    # Guess Candidate Name from first non-contact line
    candidate_name = "Candidate"
    for line in lines[:5]:
        if email and email in line:
            continue
        if any(char.isdigit() for char in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            candidate_name = line
            break

    # Extract Skills from keyword dictionary
    tech_keywords = [
        "Python", "JavaScript", "TypeScript", "React", "Node.js", "FastAPI", "Django",
        "Flask", "Go", "Golang", "Java", "Spring Boot", "C++", "C#", "Rust", "SQL",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "AWS", "Azure", "GCP", "Docker",
        "Kubernetes", "Git", "REST APIs", "GraphQL", "Machine Learning", "PyTorch",
        "TensorFlow", "Pandas", "Next.js", "Vue.js", "Tailwind CSS", "Linux", "DevOps"
    ]
    
    found_skills = set()
    raw_text_lower = f" {raw_text.lower()} "
    for kw in tech_keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, raw_text_lower):
            found_skills.add(kw)

    skills_list = sorted(list(found_skills))
    if not skills_list:
        skills_list = ["Python", "FastAPI", "Software Engineering", "Git"]

    # Extract Experience Years
    exp_match = re.search(r"(\d{1,2})\+?\s*(years?|yrs?)\s*(of)?\s*(experience)?", raw_text, re.IGNORECASE)
    if exp_match:
        years_num = int(exp_match.group(1))
        experience_years = f"{years_num}+ years"
        if years_num <= 1:
            experience_level = "Entry-Level"
        elif years_num <= 4:
            experience_level = "Mid-Level"
        elif years_num <= 8:
            experience_level = "Senior"
        else:
            experience_level = "Lead"
    else:
        experience_years = "2-4 years"
        experience_level = "Mid-Level"

    # Preferred roles
    preferred_roles = []
    for role_name in ["Software Engineer", "Backend Developer", "Full Stack Developer", "Frontend Developer", "Machine Learning Engineer", "DevOps Engineer"]:
        if role_name.lower() in raw_text_lower:
            preferred_roles.append(role_name)
    if not preferred_roles:
        preferred_roles = ["Software Engineer"]

    return {
        "name": candidate_name,
        "email": email,
        "phone": phone,
        "skills": skills_list,
        "experience_years": experience_years,
        "experience_level": experience_level,
        "preferred_roles": preferred_roles,
        "summary": f"Software engineering candidate with focus on {', '.join(skills_list[:4])}."
    }


def parse_resume_with_llm(raw_text: str) -> ResumeData:
    """
    Sub-Agent 1 execution:
    Uses OpenAI LLM to parse raw resume text into structured ResumeData.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("Empty text passed to parse_resume_with_llm.")
        return ResumeData()

    if not is_openai_configured():
        logger.info("OpenAI API key not configured. Using heuristic parsing fallback.")
        data = _heuristic_resume_parser(raw_text)
        return ResumeData(**data)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Truncate to first 6500 characters to cover header, experience, and skills
        prompt_text = raw_text[:6500]

        system_instruction = (
            "You are an expert HR resume parsing AI agent. "
            "Your task is to analyze the provided resume text and extract structured candidate profile data in valid JSON. "
            "Output JSON with the following exact keys:\n"
            "- name: string (Candidate's full name, or 'Candidate' if unstated)\n"
            "- email: string or null\n"
            "- phone: string or null\n"
            "- skills: array of strings (Normalized technical and domain skills, max 15)\n"
            "- experience_years: string (e.g. '3 years', 'Entry-Level', '5+ years')\n"
            "- experience_level: string (One of: 'Entry-Level', 'Mid-Level', 'Senior', 'Lead')\n"
            "- preferred_roles: array of strings (Target or current job titles, e.g. ['Backend Developer'])\n"
            "- summary: string (2-3 sentence executive professional summary)\n"
            "Return ONLY the JSON object."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Resume Text:\n\n{prompt_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600
        )

        response_content = response.choices[0].message.content or "{}"
        parsed_json = json.loads(response_content)
        logger.info(f"Successfully parsed resume with LLM for {parsed_json.get('name')}")
        return ResumeData(**parsed_json)

    except Exception as exc:
        logger.error(f"LLM resume parsing failed: {exc}. Falling back to heuristic parser.")
        fallback_data = _heuristic_resume_parser(raw_text)
        return ResumeData(**fallback_data)
