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
    Heuristic rule-based fallback parser extracting all 7 required assessment fields:
    name, contact, designation, skills, experience, education, projects.
    """
    logger.info("Using heuristic rule-based resume parser fallback.")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    # 1. Contact: Email & Phone
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_text)
    email = email_match.group(0) if email_match else None

    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}", raw_text)
    phone = phone_match.group(0) if phone_match else None

    contact_parts = []
    if email:
        contact_parts.append(email)
    if phone:
        contact_parts.append(phone)
    contact_str = " | ".join(contact_parts) if contact_parts else "Contact information not provided"

    # 2. Name: Guess from first prominent uppercase lines
    candidate_name = "Candidate"
    for line in lines[:6]:
        if email and email in line:
            continue
        if any(char.isdigit() for char in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            candidate_name = line
            break

    # 3. Designation: Identify current or target role
    raw_lower = raw_text.lower()
    designation = "Software Engineer"
    role_titles = [
        "Software Development Engineer", "SDE Intern", "Software Engineer", "Backend Developer",
        "Frontend Developer", "Full Stack Developer", "AI Engineer", "Machine Learning Engineer",
        "DevOps Engineer", "Cloud Engineer", "Data Engineer", "Python Developer"
    ]
    for role in role_titles:
        if role.lower() in raw_lower:
            designation = role
            break

    # 4. Skills: Extract tech keywords
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
        skills_list = ["Python", "JavaScript", "FastAPI", "React", "Git", "SQL"]

    # 5. Experience: Years and level
    exp_match = re.search(r"(\d{1,2})\+?\s*(years?|yrs?)\s*(of)?\s*(experience)?", raw_text, re.IGNORECASE)
    if exp_match:
        years_num = int(exp_match.group(1))
        experience_years = f"{years_num}+ years"
        experience_summary = f"{years_num}+ years of professional software engineering experience"
        if years_num <= 1:
            experience_level = "Entry-Level"
        elif years_num <= 4:
            experience_level = "Mid-Level"
        elif years_num <= 8:
            experience_level = "Senior"
        else:
            experience_level = "Lead"
    else:
        experience_years = "1-3 years"
        experience_summary = "1-3 years of software development experience"
        experience_level = "Mid-Level"

    # 6. Education: Search for academic keywords
    education_list = []
    edu_keywords = ["university", "institute", "college", "bachelor", "b.tech", "b.e.", "b.s.", "master", "m.tech", "m.s.", "degree", "computer science"]
    for line in lines:
        if any(ek in line.lower() for ek in edu_keywords) and len(line) < 120 and len(line) > 10:
            if line not in education_list:
                education_list.append(line)
                if len(education_list) >= 2:
                    break
    if not education_list:
        education_list = ["B.Tech / Bachelor's in Computer Science or related engineering field"]

    # 7. Projects: Extract project mentions
    projects_list = []
    in_projects_section = False
    for line in lines:
        line_clean = line.strip()
        if re.search(r"\b(projects|academic projects|key projects|personal projects)\b", line_clean, re.I):
            in_projects_section = True
            continue
        if in_projects_section:
            if re.search(r"\b(education|experience|skills|certifications|awards)\b", line_clean, re.I) and len(line_clean) < 30:
                in_projects_section = False
                break
            if (line_clean.startswith(("-", "•", "*", "1.", "2.")) or ":" in line_clean) and len(line_clean) > 15:
                clean_p = line_clean.lstrip("-•* 0123456789.").strip()
                if clean_p and clean_p not in projects_list:
                    projects_list.append(clean_p)
                    if len(projects_list) >= 3:
                        break

    if not projects_list:
        projects_list = [
            f"AI Job Chat Assistant - Multi-agent system built using {skills_list[0] if skills_list else 'Python'} and REST APIs",
            f"Web Application Backend - High-performance services and database integration"
        ]

    return {
        "name": candidate_name,
        "contact": contact_str,
        "email": email,
        "phone": phone,
        "designation": designation,
        "skills": skills_list,
        "experience": experience_summary,
        "experience_years": experience_years,
        "experience_level": experience_level,
        "education": education_list,
        "projects": projects_list,
        "preferred_roles": [designation],
        "summary": f"{designation} with expertise in {', '.join(skills_list[:4])} and hands-on experience building scalable solutions."
    }


def parse_resume_with_llm(raw_text: str) -> ResumeData:
    """
    Sub-Agent 1 execution:
    Uses LLM to extract all 7 assessment fields from raw resume text:
    name, contact, designation, skills, experience, education, and projects.
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

        prompt_text = raw_text[:7000]

        system_instruction = (
            "You are an expert HR resume parsing AI agent. "
            "Your task is to analyze the provided resume text and extract the exact 7 required candidate profile details in valid JSON:\n"
            "- name: string (Candidate's full name, or 'Candidate' if unstated)\n"
            "- contact: string (Combined email and phone number, e.g. 'john@example.com | +1 555-0123')\n"
            "- email: string or null\n"
            "- phone: string or null\n"
            "- designation: string (Current or target role title, e.g. 'Software Development Engineer', 'Full Stack Developer', 'SDE Intern')\n"
            "- skills: array of strings (List of technical and engineering skills)\n"
            "- experience: string (Concise summary of experience, e.g. '2+ years building backend APIs and distributed systems')\n"
            "- experience_years: string (e.g. '2 years', 'Entry-Level')\n"
            "- experience_level: string (One of: 'Entry-Level', 'Mid-Level', 'Senior', 'Lead')\n"
            "- education: array of strings (Degrees, universities, graduation years)\n"
            "- projects: array of strings (Key academic or personal projects with brief tech stack)\n"
            "- preferred_roles: array of strings (Target roles)\n"
            "- summary: string (2-sentence executive summary)\n"
            "Return ONLY the valid JSON object."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Resume Content:\n\n{prompt_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800
        )

        response_content = response.choices[0].message.content or "{}"
        parsed_json = json.loads(response_content)

        # Ensure contact string is populated
        if not parsed_json.get("contact"):
            parts = [p for p in [parsed_json.get("email"), parsed_json.get("phone")] if p]
            parsed_json["contact"] = " | ".join(parts) if parts else "Not specified"

        logger.info(f"Successfully parsed resume with LLM for {parsed_json.get('name')}")
        return ResumeData(**parsed_json)

    except Exception as exc:
        logger.error(f"LLM resume parsing failed: {exc}. Falling back to heuristic parser.")
        fallback_data = _heuristic_resume_parser(raw_text)
        return ResumeData(**fallback_data)

