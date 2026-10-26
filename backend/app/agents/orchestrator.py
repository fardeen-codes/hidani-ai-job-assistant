"""
Master Orchestrator Agent.
Coordinates conversation history, interprets candidate intents via OpenAI Function / Tool Calling,
and routes tasks to Sub-Agent 1 (Parser), Sub-Agent 2 (Matcher), or Sub-Agent 3 (Applier).
Includes an intent-routing fallback for seamless offline or local evaluation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.applier_agent import apply_for_job, get_applications_summary
from app.agents.matcher_agent import match_jobs_for_candidate
from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, is_openai_configured
from app.schemas import ChatAction
from app.services.db_service import get_job_by_id

logger = logging.getLogger("OrchestratorAgent")

# In-memory session store
_session_histories: Dict[str, List[Dict[str, Any]]] = {}
_session_profiles: Dict[str, Dict[str, Any]] = {}
_session_last_recommendations: Dict[str, List[Dict[str, Any]]] = {}

# Tool Definitions for OpenAI Function Calling
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search the 1600+ job database for positions matching candidate designation, skills, query keyword, location, or experience level. Sub-Agent 2 (Matcher) executes this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "designation": {
                        "type": "string",
                        "description": "Candidate designation (e.g. 'Software Engineer', 'Backend Developer')"
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant technical skills to match (e.g. ['Python', 'React', 'FastAPI'])"
                    },
                    "query": {
                        "type": "string",
                        "description": "Job title or technology keyword (e.g. 'React', 'AI', 'Backend', 'Data')"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location filter such as 'Remote' or specific city"
                    },
                    "experience_level": {
                        "type": "string",
                        "enum": ["Entry-Level", "Mid-Level", "Senior", "Lead", "Staff"],
                        "description": "Candidate experience tier"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of job recommendations to retrieve (default 4, max 10)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_to_job",
            "description": "Apply the candidate to a specific job ID or recommendation number (e.g. 'job-00042' or '3' for the 3rd recommended job). Sub-Agent 3 (Applier) handles deduplication, validation, and storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact ID of the job (e.g. 'job-00001') or the recommendation index (e.g. '1', '2', '3')"
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_application_status",
            "description": "Check the status of submitted job applications. Sub-Agent 3 (Applier) returns application timestamps and statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Optional specific Job ID to check. If omitted, returns all applications."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Fetch the full description, responsibilities, and details for a given job ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Job ID to inspect"
                    }
                },
                "required": ["job_id"]
            }
        }
    }
]


def _resolve_job_id_from_input(input_val: str, session_id: str) -> Optional[str]:
    """
    Resolve job ID from either a direct ID (e.g. 'job-00042') or a conversational
    reference (e.g. '3', 'job 3', 'job #3', '#3', 'the third job').
    """
    if not input_val:
        return None
        
    clean = input_val.strip().lower()
    
    # Direct match for job-XXXXX format
    direct_match = re.search(r"job-[a-zA-Z0-9]{5,8}", clean)
    if direct_match:
        return direct_match.group(0)

    last_recs = _session_last_recommendations.get(session_id, [])

    # Numeric match: "apply to job 3", "job 3", "3", "#3"
    num_match = re.search(r"\b(?:job\s*#?|#)?(\d{1,2})\b", clean)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(last_recs):
            return last_recs[idx].get("job_id")

    # Word ordinal matches
    word_map = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4
    }
    for word, idx in word_map.items():
        if re.search(r"\b" + word + r"\b", clean):
            if 0 <= idx < len(last_recs):
                return last_recs[idx].get("job_id")

    return clean if clean.startswith("job-") else None


def _build_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build the dynamic system prompt with active candidate context."""
    base_prompt = (
        "You are the 'Hidani AI Job Assistant', an intelligent, proactive career copilot.\n"
        "You have access to specialized sub-agents via tools:\n"
        "1. Matcher Agent (`search_jobs`): Finds matching jobs and computes match reasoning based on candidate skills, designation, and experience.\n"
        "2. Applier Agent (`apply_to_job`): Submits candidate applications without duplicates.\n"
        "3. Status Agent (`get_application_status`): Reviews applied jobs.\n"
        "4. Details Agent (`get_job_details`): Inspects full job descriptions.\n\n"
        "RULES FOR CONVERSATION:\n"
        "- When recommending jobs, list them clearly with numbered items: 1., 2., 3., 4.\n"
        "- When the user asks to apply (e.g. 'apply to job 3' or 'apply to #2'), map it to the 3rd or 2nd job from your recent recommendations and call `apply_to_job`.\n"
        "- Allow the user to naturally refine results (e.g., 'show only remote roles', 'jobs needing React', 'junior python jobs').\n"
        "- Be conversational, encouraging, and clear."
    )

    if profile:
        skills = profile.get("skills", [])
        name = profile.get("name", "Candidate")
        desig = profile.get("designation", "Software Engineer")
        exp = profile.get("experience") or profile.get("experience_years") or "1-3 years"
        edu = ", ".join(profile.get("education", [])) if profile.get("education") else "Engineering"
        base_prompt += (
            f"\n\nCURRENT CANDIDATE PROFILE:\n"
            f"- Name: {name}\n"
            f"- Designation: {desig}\n"
            f"- Skills: {', '.join(skills) if skills else 'Not specified yet'}\n"
            f"- Experience: {exp}\n"
            f"- Education: {edu}\n"
            "Use this profile context proactively when searching and matching jobs!"
        )
    return base_prompt


def _execute_tool_call(name: str, args: Dict[str, Any], profile: Optional[Dict[str, Any]] = None, session_id: str = "default-session") -> Tuple[str, Optional[ChatAction]]:
    """Execute the sub-agent corresponding to the tool call."""
    logger.info(f"Orchestrator routing to sub-agent tool '{name}' with args: {args}")

    if name == "search_jobs":
        skills = args.get("skills")
        if not skills and profile:
            skills = profile.get("skills", [])
        desig = args.get("designation")
        if not desig and profile:
            desig = profile.get("designation", "")
        level = args.get("experience_level")
        if not level and profile:
            level = profile.get("experience_level", "Mid-Level")
        query = args.get("query", "")
        location = args.get("location", "")
        limit = min(int(args.get("limit", 4)), 10)

        matched_jobs = match_jobs_for_candidate(
            candidate_skills=skills,
            candidate_designation=desig,
            candidate_level=level,
            query=query,
            location=location,
            top_k=limit
        )

        jobs_dicts = [j.model_dump() for j in matched_jobs]
        _session_last_recommendations[session_id] = jobs_dicts
        action = ChatAction(type="RECOMMEND_JOBS", data=jobs_dicts)

        summary_lines = [f"Found {len(matched_jobs)} matching job positions:"]
        for idx, j in enumerate(matched_jobs, 1):
            summary_lines.append(
                f"{idx}. **{j.title}** at {j.company} (ID: `{j.job_id}`) - Match: {j.match_score}%\n   *Reason*: {j.match_reason}"
            )
        return "\n".join(summary_lines), action

    elif name == "apply_to_job":
        raw_id = args.get("job_id", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        res = apply_for_job(resolved_id)
        action = ChatAction(type="APPLICATION_SUBMITTED", data=res)
        return json.dumps(res), action

    elif name == "get_application_status":
        job_id = args.get("job_id")
        if job_id:
            job_id = _resolve_job_id_from_input(str(job_id), session_id) or str(job_id)
        res = get_applications_summary(job_id)
        action = ChatAction(type="APPLICATIONS_LIST", data=res)
        return json.dumps(res), action

    elif name == "get_job_details":
        job_id = args.get("job_id", "")
        resolved_id = _resolve_job_id_from_input(str(job_id), session_id) or str(job_id)
        job = get_job_by_id(resolved_id)
        if not job:
            return f"Job with ID '{job_id}' not found.", None
        return json.dumps(job), ChatAction(type="JOB_DETAILS", data=job)

    return f"Unknown tool '{name}'", None


def _fallback_intent_orchestrator(message: str, profile: Optional[Dict[str, Any]] = None, session_id: str = "default-session") -> Tuple[str, List[ChatAction]]:
    """
    Intelligent heuristic fallback orchestrator when OPENAI_API_KEY is not configured.
    Detects user intent (search, apply, status, info) and handles natural refinements.
    """
    msg_lower = message.lower().strip()
    actions: List[ChatAction] = []
    
    # 1. Check for Job Application Intent: "apply to job 3", "apply to job-00001", "apply 2", etc.
    if "apply" in msg_lower:
        resolved_job_id = _resolve_job_id_from_input(message, session_id)
        if resolved_job_id:
            res = apply_for_job(resolved_job_id)
            actions.append(ChatAction(type="APPLICATION_SUBMITTED", data=res))
            if res.get("success"):
                return (
                    f"### 🎉 Application Successfully Submitted!\n\n"
                    f"{res.get('message')}\n\n"
                    f"- **Job ID**: `{resolved_job_id}`\n"
                    f"- **Status**: `Applied`\n"
                    f"- **Company**: {res.get('job', {}).get('company')}\n"
                    f"- **Role**: {res.get('job', {}).get('title')}\n\n"
                    f"You can ask me anytime: *'Show my applied jobs'* to track your progress.",
                    actions
                )
            else:
                return f"⚠️ {res.get('message')}", actions
        else:
            return (
                "Please specify which job you'd like to apply for (e.g., `apply to job 1`, `apply to job 3`, or `apply to job-00001`).",
                actions
            )

    # 2. Check for Application Status Intent: "status", "applied", "applications", "tracking"
    if any(k in msg_lower for k in ["status", "applied", "my application", "applications", "track"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        res = get_applications_summary(target_id)
        actions.append(ChatAction(type="APPLICATIONS_LIST", data=res))
        
        apps = res.get("applications", [])
        if not apps:
            return (
                "📋 **Application Status**: You have not applied to any jobs yet.\n\n"
                "Ask me *'Recommend jobs for me'* or upload your resume to get started!",
                actions
            )
        
        lines = [f"### 📋 Your Applied Jobs ({len(apps)} Total)\n"]
        for a in apps:
            lines.append(f"- **{a.get('job_title', 'Software Role')}** at **{a.get('company', 'Tech')}** (ID: `{a.get('job_id')}`) — Status: `{a.get('status')}` on *{a.get('applied_date')}*")
        lines.append("\nWould you like me to find more opportunities or check another role?")
        return "\n".join(lines), actions

    # 3. Check for Job Recommendation & Natural Refinement Intent
    skills = profile.get("skills", []) if profile else []
    level = profile.get("experience_level", "Mid-Level") if profile else "Mid-Level"
    desig = profile.get("designation", "Software Engineer") if profile else "Software Engineer"
    
    # Location refinement: "show only remote roles", "remote jobs", "remote"
    location = "remote" if "remote" in msg_lower else None

    # Experience refinement: "junior", "intern", "entry level", "senior"
    if any(k in msg_lower for k in ["junior", "intern", "entry", "entry-level"]):
        level = "Entry-Level"
    elif any(k in msg_lower for k in ["senior", "sr", "lead"]):
        level = "Senior"

    # Skill or Domain query: e.g. "jobs needing React", "python", "ai", "fastapi"
    query = ""
    for kw in ["react", "python", "javascript", "typescript", "fastapi", "django", "node", "ai", "machine learning", "devops", "go", "golang", "backend", "frontend", "full stack"]:
        if kw in msg_lower:
            query = kw
            break

    matched_jobs = match_jobs_for_candidate(
        candidate_skills=skills,
        candidate_designation=desig,
        candidate_level=level,
        query=query,
        location=location,
        top_k=4
    )

    jobs_dicts = [j.model_dump() for j in matched_jobs]
    _session_last_recommendations[session_id] = jobs_dicts
    actions.append(ChatAction(type="RECOMMEND_JOBS", data=jobs_dicts))

    greeting = f"Hello {profile.get('name', 'there')}! " if profile else "Hello! "
    refinement_note = ""
    if location:
        refinement_note += " (filtered for **Remote** roles)"
    if query:
        refinement_note += f" (matching **{query.title()}**)"

    intro = (
        f"{greeting}Here are **{len(matched_jobs)} matching job positions** from our database{refinement_note}:\n\n"
    )

    cards = []
    for idx, j in enumerate(matched_jobs, 1):
        cards.append(
            f"### {idx}. 💼 {j.title} — {j.company}\n"
            f"- **Location**: {j.location} | **Level**: {j.experience}\n"
            f"- **Job ID**: `{j.job_id}` | **Match Score**: `{j.match_score}%`\n"
            f"- **Skills Required**: {', '.join(j.skills[:5])}\n"
            f"- **Match Reason**: {j.match_reason}\n"
            f"- [View Original Posting]({j.source_link})\n"
        )

    closing = (
        f"\n💡 *You can apply right now by saying **'apply to job 1'**, **'apply to job {len(matched_jobs)}'**, "
        f"or by clicking the **Quick Apply** button on any card below!*"
    )
    return intro + "\n---\n".join(cards) + closing, actions


def run_orchestrator(
    message: str,
    session_id: str = "default-session",
    profile: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[ChatAction]]:
    """
    Main entry point for the Multi-Agent Orchestrator.
    Manages session context, tool-calling loop with OpenAI, and delegates to sub-agents.
    """
    if profile:
        _session_profiles[session_id] = profile
    active_profile = _session_profiles.get(session_id) or profile

    # Initialize session conversation if new
    if session_id not in _session_histories:
        _session_histories[session_id] = [
            {"role": "system", "content": _build_system_prompt(active_profile)}
        ]
    else:
        # Keep system prompt updated with profile changes
        _session_histories[session_id][0] = {
            "role": "system",
            "content": _build_system_prompt(active_profile)
        }

    # If OpenAI API is not configured, run deterministic intent orchestrator
    if not is_openai_configured():
        logger.info("OPENAI_API_KEY is not configured. Running fallback intent orchestrator.")
        reply, actions = _fallback_intent_orchestrator(message, active_profile, session_id=session_id)
        _session_histories[session_id].append({"role": "user", "content": message})
        _session_histories[session_id].append({"role": "assistant", "content": reply})
        return reply, actions

    # Full OpenAI Tool Calling Execution
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Append incoming user prompt
        _session_histories[session_id].append({"role": "user", "content": message})

        # Limit history to last 12 messages to prevent excessive token growth
        recent_messages = [_session_histories[session_id][0]] + _session_histories[session_id][-11:]

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=recent_messages,
            tools=ORCHESTRATOR_TOOLS,
            tool_choice="auto",
            temperature=0.3
        )

        response_msg = response.choices[0].message
        collected_actions: List[ChatAction] = []

        # If model decided to invoke one or more tools
        if response_msg.tool_calls:
            # Append assistant's call message
            _session_histories[session_id].append(response_msg)

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    func_args = {}

                tool_result_str, action_obj = _execute_tool_call(func_name, func_args, active_profile, session_id=session_id)

                if action_obj:
                    collected_actions.append(action_obj)

                _session_histories[session_id].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result_str
                })

            # Get final synthesized response from model after tools returned
            second_response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=_session_histories[session_id][-14:],
                temperature=0.4
            )
            final_reply = second_response.choices[0].message.content or "I have processed your request."
            _session_histories[session_id].append({"role": "assistant", "content": final_reply})
            return final_reply, collected_actions

        else:
            final_reply = response_msg.content or "How can I assist you with your job search today?"
            _session_histories[session_id].append({"role": "assistant", "content": final_reply})
            return final_reply, collected_actions

    except Exception as exc:
        logger.error(f"Orchestrator LLM execution error: {exc}. Using fallback router.")
        reply, actions = _fallback_intent_orchestrator(message, active_profile)
        return reply, actions


def get_session_state(session_id: str) -> Dict[str, Any]:
    """Inspect the internal state and stats of an active chat session."""
    history = _session_histories.get(session_id, [])
    profile = _session_profiles.get(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "has_profile": profile is not None,
        "profile": profile
    }
