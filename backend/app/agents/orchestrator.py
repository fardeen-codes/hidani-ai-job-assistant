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
from app.config import OPENAI_API_KEY, OPENAI_MODEL, is_openai_configured
from app.schemas import ChatAction
from app.services.db_service import get_job_by_id

logger = logging.getLogger("OrchestratorAgent")

# In-memory session store: session_id -> list of OpenAI messages
_session_histories: Dict[str, List[Dict[str, Any]]] = {}
_session_profiles: Dict[str, Dict[str, Any]] = {}

# Tool Definitions for OpenAI Function Calling
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search the 1600+ job database for positions matching candidate skills, title query, or experience level. Sub-Agent 2 (Matcher) executes this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant technical skills to match (e.g. ['Python', 'FastAPI', 'Docker'])"
                    },
                    "query": {
                        "type": "string",
                        "description": "Job title keyword or domain (e.g. 'Software Engineer', 'Backend', 'Data')"
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
            "description": "Apply the candidate to a specific job ID. Sub-Agent 3 (Applier) handles deduplication, validation, and database storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact ID of the job to apply to (e.g. 'job-00001')"
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


def _build_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build the dynamic system prompt with active candidate context."""
    base_prompt = (
        "You are the 'Hidani AI Job Assistant', an intelligent, proactive talent advisor and job application agent.\n"
        "You have access to specialized sub-agents via tools:\n"
        "1. Matcher Agent (`search_jobs`): Finds matching jobs and computes match reasoning.\n"
        "2. Applier Agent (`apply_to_job`): Submits candidate applications without duplicates.\n"
        "3. Status Agent (`get_application_status`): Reviews applied jobs.\n"
        "4. Details Agent (`get_job_details`): Inspects full job descriptions.\n\n"
        "Always use your tools when the user expresses intent to find jobs, apply, or check status.\n"
        "Format job recommendations cleanly with markdown: include Title, Company, Location, Job ID, Match Score, and Reason.\n"
        "Encourage the user to apply by stating the Job ID or clicking the quick apply button."
    )

    if profile:
        skills = profile.get("skills", [])
        name = profile.get("name", "Candidate")
        exp = profile.get("experience_years") or profile.get("experience_level", "Mid-Level")
        base_prompt += (
            f"\n\nCURRENT CANDIDATE PROFILE:\n"
            f"- Name: {name}\n"
            f"- Skills: {', '.join(skills) if skills else 'Not specified yet'}\n"
            f"- Experience: {exp}\n"
            f"- Target Roles: {', '.join(profile.get('preferred_roles', []))}\n"
            "Use this profile context proactively when searching and matching jobs!"
        )
    return base_prompt


def _execute_tool_call(name: str, args: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[ChatAction]]:
    """Execute the sub-agent corresponding to the tool call."""
    logger.info(f"Orchestrator routing to sub-agent tool '{name}' with args: {args}")

    if name == "search_jobs":
        skills = args.get("skills")
        if not skills and profile:
            skills = profile.get("skills", [])
        level = args.get("experience_level")
        if not level and profile:
            level = profile.get("experience_level", "Mid-Level")
        query = args.get("query", "")
        limit = min(int(args.get("limit", 4)), 10)

        matched_jobs = match_jobs_for_candidate(
            candidate_skills=skills,
            candidate_level=level,
            query=query,
            top_k=limit
        )

        jobs_dicts = [j.model_dump() for j in matched_jobs]
        action = ChatAction(type="RECOMMEND_JOBS", data=jobs_dicts)
        summary = (
            f"Found {len(matched_jobs)} matching job positions:\n" +
            "\n".join([
                f"- **{j.title}** at {j.company} (ID: `{j.job_id}`) - Match: {j.match_score}%\n  *Reason*: {j.match_reason}"
                for j in matched_jobs
            ])
        )
        return summary, action

    elif name == "apply_to_job":
        job_id = args.get("job_id", "")
        res = apply_for_job(job_id)
        action = ChatAction(type="APPLICATION_SUBMITTED", data=res)
        return json.dumps(res), action

    elif name == "get_application_status":
        job_id = args.get("job_id")
        res = get_applications_summary(job_id)
        action = ChatAction(type="APPLICATIONS_LIST", data=res)
        return json.dumps(res), action

    elif name == "get_job_details":
        job_id = args.get("job_id", "")
        job = get_job_by_id(job_id)
        if not job:
            return f"Job with ID '{job_id}' not found.", None
        return json.dumps(job), ChatAction(type="JOB_DETAILS", data=job)

    return f"Unknown tool '{name}'", None


def _fallback_intent_orchestrator(message: str, profile: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ChatAction]]:
    """
    Intelligent heuristic fallback orchestrator when OPENAI_API_KEY is not configured.
    Detects user intent (search, apply, status, info) and routes to sub-agents.
    """
    msg_lower = message.lower().strip()
    actions: List[ChatAction] = []
    
    # Check for Job Application Intent: "apply", "apply to job-XXXXX", "apply job-XXXXX"
    job_id_match = re.search(r"job-[a-zA-Z0-9]{5,8}", message, re.IGNORECASE)
    if "apply" in msg_lower or job_id_match:
        if job_id_match:
            target_job_id = job_id_match.group(0).lower()
            res = apply_for_job(target_job_id)
            actions.append(ChatAction(type="APPLICATION_SUBMITTED", data=res))
            if res.get("success"):
                return (
                    f"### 🎉 Application Successfully Submitted!\n\n"
                    f"{res.get('message')}\n\n"
                    f"- **Job ID**: `{target_job_id}`\n"
                    f"- **Status**: `Applied`\n"
                    f"- **Company**: {res.get('job', {}).get('company')}\n"
                    f"- **Role**: {res.get('job', {}).get('title')}\n\n"
                    f"You can ask me anytime: *'Check my application status'* to track your progress.",
                    actions
                )
            else:
                return f"⚠️ {res.get('message')}", actions
        else:
            return (
                "Please specify the **Job ID** you'd like to apply for (e.g., `apply to job-00001`). "
                "Or tell me to *'find matching jobs'* to see available recommendations!",
                actions
            )

    # Check for Application Status Intent: "status", "applied", "applications", "tracking"
    if any(k in msg_lower for k in ["status", "applied", "my application", "applications", "track"]):
        target_id = job_id_match.group(0) if job_id_match else None
        res = get_applications_summary(target_id)
        actions.append(ChatAction(type="APPLICATIONS_LIST", data=res))
        
        apps = res.get("applications", [])
        if not apps:
            return (
                "📋 **Application Status**: You have not applied to any jobs yet.\n\n"
                "Upload your resume or ask me *'Recommend jobs for me'* to get started!",
                actions
            )
        
        lines = [f"### 📋 Your Job Applications ({len(apps)} Total)\n"]
        for a in apps:
            lines.append(f"- **{a.get('job_title', 'Software Role')}** at **{a.get('company', 'Tech')}** (ID: `{a.get('job_id')}`) — Status: `{a.get('status')}` on *{a.get('applied_date')}*")
        lines.append("\nWould you like me to find more opportunities or check another role?")
        return "\n".join(lines), actions

    # Check for Job Recommendation / Search Intent
    skills = profile.get("skills", []) if profile else []
    level = profile.get("experience_level", "Mid-Level") if profile else "Mid-Level"
    
    # Extract possible keywords from message
    query = ""
    for kw in ["python", "react", "backend", "frontend", "full stack", "ai", "machine learning", "devops", "go", "golang"]:
        if kw in msg_lower:
            query = kw
            break

    matched_jobs = match_jobs_for_candidate(
        candidate_skills=skills,
        candidate_level=level,
        query=query,
        top_k=4
    )

    jobs_dicts = [j.model_dump() for j in matched_jobs]
    actions.append(ChatAction(type="RECOMMEND_JOBS", data=jobs_dicts))

    greeting = f"Hello {profile.get('name', 'there')}! " if profile else "Hello! "
    intro = (
        f"{greeting}Based on your profile and query, here are **{len(matched_jobs)} top job matches** "
        f"from our database of 1,600+ tech positions:\n\n"
    )

    cards = []
    for j in matched_jobs:
        cards.append(
            f"#### 💼 {j.title} — {j.company}\n"
            f"- **Location**: {j.location} | **Level**: {j.experience}\n"
            f"- **Job ID**: `{j.job_id}` | **Match Score**: `{j.match_score}%`\n"
            f"- **Skills Required**: {', '.join(j.skills[:5])}\n"
            f"- **Why it matches**: {j.match_reason}\n"
            f"- [View Original Posting]({j.source_link})\n"
        )

    closing = (
        "\n💡 *To apply, simply type `apply to {job_id}` or click the Quick Apply button on any card below!*"
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
        reply, actions = _fallback_intent_orchestrator(message, active_profile)
        _session_histories[session_id].append({"role": "user", "content": message})
        _session_histories[session_id].append({"role": "assistant", "content": reply})
        return reply, actions

    # Full OpenAI Tool Calling Execution
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

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

                tool_result_str, action_obj = _execute_tool_call(func_name, func_args, active_profile)
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
