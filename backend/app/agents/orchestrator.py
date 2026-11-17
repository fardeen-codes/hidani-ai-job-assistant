"""
Master Orchestrator Agent.
Coordinates conversation history, interprets candidate intents via OpenAI Function / Tool Calling,
and routes tasks to Sub-Agent 1 (Parser), Sub-Agent 2 (Matcher), or Sub-Agent 3 (Applier).
Includes an intent-routing fallback for seamless offline or local evaluation.
"""

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.agents.applier_agent import (
    apply_for_job,
    get_applications_summary,
    update_job_application_status,
    withdraw_job_application,
)
from app.agents.matcher_agent import analyze_skill_gap, match_jobs_for_candidate
from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, is_openai_configured
from app.schemas import ChatAction
from app.services.db_service import (
    VALID_APPLICATION_STATUSES,
    get_bookmarked_jobs,
    get_job_by_id,
    get_upcoming_reminders,
    load_applications,
    remove_job_bookmark,
    save_job_bookmark,
    update_application_reminder,
)

logger = logging.getLogger("OrchestratorAgent")

# In-memory session store protected by a re-entrant lock
_session_lock = threading.RLock()
_session_histories: Dict[str, List[Dict[str, Any]]] = {}
_session_profiles: Dict[str, Dict[str, Any]] = {}
_session_last_recommendations: Dict[str, List[Dict[str, Any]]] = {}
_session_metadata: Dict[str, Dict[str, Any]] = {}

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
    },
    {
        "type": "function",
        "function": {
            "name": "withdraw_application",
            "description": "Withdraw or cancel a previously submitted job application by Job ID or recommendation index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact Job ID (e.g. 'job-00001') or ordinal index (e.g. '1', '2') of the application to cancel."
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_application_status",
            "description": "Update the tracking status of an existing submitted job application (e.g. 'Under Review', 'Interview', 'Offer', 'Rejected').",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact Job ID (e.g. 'job-00001') or ordinal index of the application."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["Applied", "Under Review", "Interview", "Offer", "Rejected"],
                        "description": "The updated tracking status."
                    }
                },
                "required": ["job_id", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bookmark_job",
            "description": "Bookmark or save a job position for later review by Job ID or recommendation number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact Job ID (e.g. 'job-00001') or ordinal index (e.g. '1', '2', '3') of the job to save."
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bookmarked_jobs",
            "description": "Retrieve all saved or bookmarked job positions for the candidate.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_skill_gap",
            "description": "Analyze candidate readiness, matching skills, and missing skills for a specific job ID or ordinal index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The exact Job ID (e.g. 'job-00001') or ordinal index ('1', '2') to evaluate."
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_followup_reminder",
            "description": "Schedule a follow-up reminder date or deadline for an applied job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Job ID or recommendation index."
                    },
                    "reminder_date": {
                        "type": "string",
                        "description": "Target reminder date (e.g. '2026-11-20')."
                    }
                },
                "required": ["job_id", "reminder_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_reminders",
            "description": "List all active application follow-up reminders.",
            "parameters": {
                "type": "object",
                "properties": {}
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

    with _session_lock:
        last_recs = list(_session_last_recommendations.get(session_id, []))
    apps = load_applications()

    # Numeric match: "apply to job 3", "job 3", "3", "#3"
    num_match = re.search(r"\b(?:job\s*#?|#)?(\d{1,2})\b", clean)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(last_recs):
            return last_recs[idx].get("job_id")
        elif 0 <= idx < len(apps):
            return apps[idx].get("job_id")

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
            elif 0 <= idx < len(apps):
                return apps[idx].get("job_id")

    return clean if clean.startswith("job-") else None


def _build_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build the dynamic system prompt with active candidate context."""
    base_prompt = (
        "You are the 'Hidani AI Job Assistant', an intelligent, proactive career copilot.\n"
        "You have access to specialized sub-agents via tools:\n"
        "1. Matcher Agent (`search_jobs`): Finds matching jobs and computes match reasoning based on candidate skills, designation, and experience.\n"
        "2. Applier Agent (`apply_to_job`): Submits candidate applications without duplicates.\n"
        "3. Status Agent (`get_application_status`): Reviews applied jobs.\n"
        "4. Details Agent (`get_job_details`): Inspects full job descriptions.\n"
        "5. Bookmarking Agent (`bookmark_job`, `get_bookmarked_jobs`): Saves positions for later candidate review.\n"
        "6. Skill Gap & Match Analysis (`analyze_skill_gap`): Evaluates candidate readiness and missing skills for roles.\n"
        "7. Reminders Agent (`set_followup_reminder`, `get_upcoming_reminders`): Sets application follow-up deadlines.\n\n"
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
        with _session_lock:
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

    elif name == "withdraw_application":
        raw_id = args.get("job_id", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        res = withdraw_job_application(resolved_id)
        action = ChatAction(type="APPLICATION_WITHDRAWN", data=res)
        return json.dumps(res), action

    elif name == "update_application_status":
        raw_id = args.get("job_id", "")
        new_status = args.get("status", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        res = update_job_application_status(resolved_id, str(new_status))
        action = ChatAction(type="APPLICATION_UPDATED", data=res)
        return json.dumps(res), action

    elif name == "bookmark_job":
        raw_id = args.get("job_id", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        success, msg, bm = save_job_bookmark(resolved_id)
        payload = {"job_id": resolved_id, "success": success, "message": msg, "bookmark": bm}
        action = ChatAction(type="BOOKMARK_SAVED", data=payload)
        return json.dumps(payload), action

    elif name == "get_bookmarked_jobs":
        bms = get_bookmarked_jobs()
        payload = {"bookmarks": bms, "total": len(bms)}
        action = ChatAction(type="BOOKMARKS_LIST", data=payload)
        return json.dumps(payload), action

    elif name == "analyze_skill_gap":
        raw_id = args.get("job_id", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        skills = profile.get("skills", []) if profile else []
        exp_level = profile.get("experience_level", "Mid-Level") if profile else "Mid-Level"
        analysis = analyze_skill_gap(resolved_id, candidate_skills=skills, candidate_level=exp_level)
        if not analysis:
            return f"Job with ID '{resolved_id}' not found.", None
        action = ChatAction(type="SKILL_GAP_ANALYSIS", data=analysis)
        return json.dumps(analysis), action

    elif name == "set_followup_reminder":
        raw_id = args.get("job_id", "")
        rem_date = args.get("reminder_date", "")
        resolved_id = _resolve_job_id_from_input(str(raw_id), session_id) or str(raw_id)
        success, msg, app_data = update_application_reminder(resolved_id, rem_date)
        payload = {"job_id": resolved_id, "success": success, "message": msg, "application": app_data}
        action = ChatAction(type="REMINDER_SET", data=payload)
        return json.dumps(payload), action

    elif name == "get_upcoming_reminders":
        rems = get_upcoming_reminders()
        payload = {"total": len(rems), "reminders": rems}
        action = ChatAction(type="REMINDERS_LIST", data=payload)
        return json.dumps(payload), action

    return f"Unknown tool '{name}'", None


def _fallback_intent_orchestrator(message: str, profile: Optional[Dict[str, Any]] = None, session_id: str = "default-session") -> Tuple[str, List[ChatAction]]:
    """
    Intelligent heuristic fallback orchestrator when OPENAI_API_KEY is not configured.
    Detects conversational intent (chit-chat, greetings, help, apply, status, search)
    and handles natural dialog instead of unconditionally dumping jobs.
    """
    msg_lower = message.lower().strip()
    actions: List[ChatAction] = []
    
    # 1. Chit-chat & "How are you" Intent
    how_are_you_patterns = [
        "how are you", "how are u", "how do you do", "how is it going",
        "how's it going", "how are you doing", "what's up", "whats up",
        "how r u", "how have you been"
    ]
    if any(k in msg_lower for k in how_are_you_patterns):
        name_str = f" {profile.get('name')}" if profile and profile.get("name") else ""
        return (
            f"I'm doing well, thank you for asking{name_str}! I'm here to help you navigate open engineering roles, "
            f"evaluate resume match relevance, and submit job applications. How can I help you today?",
            []
        )

    # 2. Greetings Intent ("hello", "hi", "hey")
    greeting_patterns = [r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening|day))\b"]
    is_pure_greeting = any(re.search(pat, msg_lower) for pat in greeting_patterns)
    has_job_keywords = any(k in msg_lower for k in ["job", "role", "position", "apply", "search", "work", "find", "recommend", "python", "react", "remote"])
    if is_pure_greeting and not has_job_keywords:
        name_str = f" {profile.get('name')}" if profile and profile.get("name") else ""
        return (
            f"Hello{name_str}! How can I assist you with your career search today? "
            f"You can upload your resume to evaluate your fit, ask me to recommend positions (e.g., *'Recommend jobs for me'*), "
            f"or search for specific technologies (e.g., *'Show remote Python roles'*).",
            []
        )

    # 3. Gratitude & Courtesies ("thank you", "thanks")
    if any(k in msg_lower for k in ["thank you", "thanks", "thx", "appreciate it"]):
        return (
            "You're very welcome! Let me know if you would like to explore more positions, refine your search, or apply to any role.",
            []
        )

    # 4. Identity & Capabilities Intent ("who are you", "what can you do", "help")
    if any(k in msg_lower for k in ["who are you", "what can you do", "what do you do", "help", "what is this", "how does this work", "what are your capabilities"]):
        return (
            "I am your career assistant for Hidani Tech. Here is how I can help you:\n\n"
            "- **Resume Analysis**: Upload a PDF or DOCX resume to extract your skills, designation, and experience level.\n"
            "- **Job Recommendations**: Match your background against 1,600+ database listings with personalized match reasoning.\n"
            "- **Instant Applications**: Apply directly with deduplication by saying *'apply to job 1'* or clicking Apply on any card.\n"
            "- **Application Tracking**: Review submitted applications anytime by asking *'What jobs have I applied to?'*.\n\n"
            "How would you like to get started?",
            []
        )

    # 5. Goodbyes Intent ("bye", "goodbye")
    if any(k in msg_lower for k in ["bye", "goodbye", "see you", "cya"]):
        return (
            "Goodbye! Best of luck with your job search, and feel free to return anytime to check new openings or application statuses.",
            []
        )

    # 6. Clear / Reset Chat Intent ("clear chat", "reset chat", "start over")
    if any(k in msg_lower for k in ["clear chat", "reset chat", "reset conversation", "start over", "clear conversation", "restart chat"]):
        clear_session_state(session_id)
        return (
            "I have cleared our conversation history and reset recommendations. How can I assist you with your career search?",
            [ChatAction(type="SESSION_RESET", data={"session_id": session_id})]
        )

    # 6. Check for Application Withdrawal Intent: "withdraw", "cancel application", "remove application"
    if any(k in msg_lower for k in ["withdraw", "cancel application", "cancel my application", "remove application", "delete application"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id:
            res = withdraw_job_application(target_id)
            actions.append(ChatAction(type="APPLICATION_WITHDRAWN", data=res))
            if res.get("success"):
                return (
                    f"### Application Withdrawn\n\n"
                    f"{res.get('message')}\n\n"
                    f"- **Job ID**: `{target_id}`\n"
                    f"- **Status**: `Withdrawn`\n\n"
                    f"You can review active submissions anytime by asking: *'Show my applied jobs'*.",
                    actions
                )
            else:
                return f"Notice: {res.get('message')}", actions
        else:
            return (
                "Please specify which application you would like to withdraw (e.g., `withdraw application for job-00001` or `cancel application 1`).",
                actions
            )

    # 7. Check for Job Application Intent: "apply to job 3", "apply to job-00001", "apply 2", etc.
    if "apply" in msg_lower:
        resolved_job_id = _resolve_job_id_from_input(message, session_id)
        if resolved_job_id:
            res = apply_for_job(resolved_job_id)
            actions.append(ChatAction(type="APPLICATION_SUBMITTED", data=res))
            if res.get("success"):
                return (
                    f"### Application Submitted\n\n"
                    f"{res.get('message')}\n\n"
                    f"- **Job ID**: `{resolved_job_id}`\n"
                    f"- **Status**: `Applied`\n"
                    f"- **Company**: {res.get('job', {}).get('company')}\n"
                    f"- **Role**: {res.get('job', {}).get('title')}\n\n"
                    f"You can review active submissions anytime by asking: *'Show my applied jobs'*.",
                    actions
                )
            else:
                return f"Notice: {res.get('message')}", actions
        else:
            return (
                "Please specify which job you would like to apply for (e.g., `apply to job 1` or `apply to job-00001`).",
                actions
            )

    # 7a. Check for Application Status Update Intent: e.g. "mark job-00001 as interview", "update status of job-00001 to offer"
    update_triggers = ["mark ", "update status", "change status", "set status", "move application", "update application"]
    if any(t in msg_lower for t in update_triggers):
        detected_status = None
        for status_key, canonical in VALID_APPLICATION_STATUSES.items():
            if re.search(r"\b" + re.escape(status_key) + r"\b", msg_lower):
                detected_status = canonical
                break
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id and detected_status:
            res = update_job_application_status(target_id, detected_status)
            actions.append(ChatAction(type="APPLICATION_UPDATED", data=res))
            if res.get("success"):
                return (
                    f"### Application Status Updated\n\n"
                    f"{res.get('message')}\n\n"
                    f"- **Job ID**: `{target_id}`\n"
                    f"- **Status**: `{detected_status}`\n\n"
                    f"You can ask *'Show my applied jobs'* anytime to review all updates.",
                    actions
                )
            else:
                return f"Notice: {res.get('message')}", actions

    # 7b. Check for Application Status Intent: "status", "applied", "applications", "tracking"
    if any(k in msg_lower for k in ["status", "applied", "my application", "applications", "track"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        res = get_applications_summary(target_id)
        actions.append(ChatAction(type="APPLICATIONS_LIST", data=res))
        
        apps = res.get("applications", [])
        if not apps:
            if target_id:
                return (
                    f"### Application Status\n\n"
                    f"{res.get('message', f'No application found for Job ID {target_id}.')}\n\n"
                    "You can ask me to *'Show my applied jobs'* to view all active applications.",
                    actions
                )
            return (
                "### Application Status\n\n"
                "No applications have been recorded yet.\n\n"
                "Upload a resume or ask to recommend positions to begin.",
                actions
            )
        
        lines = [f"### Applied Positions ({len(apps)} Total)\n"]
        for a in apps:
            lines.append(f"- **{a.get('job_title', 'Software Role')}** at **{a.get('company', 'Tech')}** (ID: `{a.get('job_id')}`) — Status: `{a.get('status')}` ({a.get('applied_date')})")
        lines.append("\nLet me know if you would like to apply to additional openings or review another position.")
        return "\n".join(lines), actions

    # 7c. Check for Saved Bookmarks Listing Intent: "saved jobs", "bookmarks", "show bookmarks"
    if any(k in msg_lower for k in ["saved jobs", "my bookmarks", "show bookmarks", "view bookmarks", "bookmarked jobs", "saved positions", "list bookmarks"]):
        bms = get_bookmarked_jobs()
        payload = {"bookmarks": bms, "total": len(bms)}
        actions.append(ChatAction(type="BOOKMARKS_LIST", data=payload))
        if not bms:
            return (
                "### Saved Bookmarks\n\n"
                "You haven't bookmarked any jobs yet.\n\n"
                "You can save any job by typing *'bookmark job 1'* or clicking the bookmark button on any recommendation card.",
                actions
            )
        lines = [f"### Saved Bookmarks ({len(bms)} Total)\n"]
        for b in bms:
            lines.append(f"- **{b.get('job_title', 'Role')}** at **{b.get('company', 'Company')}** (ID: `{b.get('job_id')}`) — Saved: `{b.get('saved_date')}`")
        lines.append("\nYou can apply anytime by asking: *'Apply to job <ID>'* or remove with *'Remove bookmark <ID>'*.")
        return "\n".join(lines), actions

    # 7d. Check for Bookmark Removal Intent: "remove bookmark", "unsave job", "delete bookmark"
    if any(k in msg_lower for k in ["remove bookmark", "unsave job", "delete bookmark", "unbookmark"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id:
            success, msg = remove_job_bookmark(target_id)
            actions.append(ChatAction(type="BOOKMARK_REMOVED", data={"job_id": target_id, "success": success, "message": msg}))
            return f"### Bookmark Update\n\n{msg}", actions
        else:
            return "Please specify which job bookmark you want to remove (e.g. `remove bookmark for job-00001` or `remove bookmark 1`).", actions

    # 7e. Check for Job Bookmark Intent: "bookmark job 1", "save job 2", "bookmark this job"
    if any(k in msg_lower for k in ["bookmark", "save job", "save this job", "save position", "bookmark job"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id:
            success, msg, bm = save_job_bookmark(target_id)
            actions.append(ChatAction(type="BOOKMARK_SAVED", data={"job_id": target_id, "success": success, "message": msg, "bookmark": bm}))
            if success:
                return (
                    f"### Job Bookmarked\n\n"
                    f"{msg}\n\n"
                    f"- **Job ID**: `{target_id}`\n"
                    f"- **Role**: {bm.get('job_title')}\n"
                    f"- **Company**: {bm.get('company')}\n\n"
                    f"You can view all saved jobs anytime by asking: *'Show my saved jobs'*.",
                    actions
                )
            else:
                return f"Notice: {msg}", actions
        else:
            return (
                "Please specify which job you would like to bookmark (e.g., `bookmark job 1` or `bookmark job-00001`).",
                actions
            )

    # 7f. Check for Skill Gap Analysis Intent: "skill gap", "skills needed", "skills required", "what skills"
    if any(k in msg_lower for k in ["skill gap", "skills needed", "skills required", "what skills", "missing skills", "skill fit", "skills for job"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id:
            cand_skills = profile.get("skills", []) if profile else []
            cand_level = profile.get("experience_level", "Mid-Level") if profile else "Mid-Level"
            analysis = analyze_skill_gap(target_id, candidate_skills=cand_skills, candidate_level=cand_level)
            if analysis:
                actions.append(ChatAction(type="SKILL_GAP_ANALYSIS", data=analysis))
                matching_str = ", ".join(analysis["matching_skills"]) if analysis["matching_skills"] else "None specified"
                missing_str = ", ".join(analysis["missing_skills"]) if analysis["missing_skills"] else "None (Complete match!)"
                reply_text = (
                    f"### Skill Fit Assessment: {analysis['job_title']} at {analysis['company']}\n\n"
                    f"- **Match Score**: `{analysis['match_score']}%`\n"
                    f"- **Readiness Level**: `{analysis['readiness']}`\n"
                    f"- **Matching Skills**: {matching_str}\n"
                    f"- **Missing / Recommended Skills**: {missing_str}\n\n"
                    f"**Career Advice**: {analysis['recommendation']}"
                )
                return reply_text, actions
            else:
                return f"Notice: Job ID '{target_id}' was not found in the database.", actions
        else:
            return (
                "Please specify which job you want to evaluate skills for (e.g., `skills for job 1` or `skill gap for job-00001`).",
                actions
            )

    # 7g. Check for Reminders Listing Intent: "show reminders", "my reminders", "upcoming reminders"
    if any(k in msg_lower for k in ["show reminders", "my reminders", "upcoming reminders", "view reminders", "list reminders"]):
        rems = get_upcoming_reminders()
        actions.append(ChatAction(type="REMINDERS_LIST", data={"total": len(rems), "reminders": rems}))
        if not rems:
            return (
                "### Upcoming Reminders\n\n"
                "You have no pending follow-up reminders scheduled.\n\n"
                "You can schedule one anytime by asking: *'Remind me to follow up on job 1 on 2026-11-20'*.",
                actions
            )
        lines = [f"### Upcoming Reminders ({len(rems)} Scheduled)\n"]
        for r in rems:
            lines.append(f"- **{r.get('job_title', 'Role')}** at **{r.get('company', 'Company')}** (ID: `{r.get('job_id')}`) — Follow up: `{r.get('reminder_date')}`")
        return "\n".join(lines), actions

    # 7h. Check for Set Reminder Intent: "remind me", "set reminder", "schedule reminder"
    if any(k in msg_lower for k in ["remind me", "set reminder", "schedule reminder", "add reminder"]):
        target_id = _resolve_job_id_from_input(message, session_id)
        if target_id:
            date_match = re.search(r"\b(202\d-[01]\d-[0-3]\d)\b", message)
            rem_date = date_match.group(1) if date_match else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            success, msg, app_data = update_application_reminder(target_id, rem_date)
            actions.append(ChatAction(type="REMINDER_SET", data={"job_id": target_id, "success": success, "message": msg, "application": app_data}))
            return f"### Follow-up Reminder Scheduled\n\n{msg}", actions
        else:
            return (
                "Please specify which application you would like to set a reminder for (e.g., `remind me for job 1 on 2026-11-20`).",
                actions
            )

    # 8. Check for Job Recommendation & Natural Refinement Intent
    job_intent_words = [
        "job", "jobs", "recommend", "recommendation", "search", "find", "role", "roles",
        "position", "positions", "opening", "openings", "work", "career", "hire", "hiring",
        "match", "matching"
    ]
    tech_keywords = [
        "react", "python", "javascript", "typescript", "fastapi", "django", "node", "ai",
        "machine learning", "devops", "go", "golang", "backend", "frontend", "full stack",
        "intern", "junior", "senior", "entry-level", "lead"
    ]
    is_job_query = any(k in msg_lower for k in job_intent_words) or any(k in msg_lower for k in tech_keywords) or (profile is not None)

    # If the message is not related to jobs or skills and no profile is loaded, guide the user
    if not is_job_query and not any(k in msg_lower for k in ["remote", "berlin", "munich", "germany", "usa"]):
        return (
            f"I'm here to help with your job search and applications. You can ask me to *'Recommend jobs for me'*, "
            f"search for specific roles (e.g., *'Show remote Python positions'*), or upload your resume to get matched.",
            []
        )

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
    for kw in tech_keywords:
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
    with _session_lock:
        _session_last_recommendations[session_id] = jobs_dicts
    actions.append(ChatAction(type="RECOMMEND_JOBS", data=jobs_dicts))

    greeting = f"Hello {profile.get('name')}. " if profile and profile.get("name") else ""
    refinement_note = ""
    if location:
        refinement_note += " (filtered for **Remote** roles)"
    if query:
        refinement_note += f" (matching **{query.title()}**)"

    intro = (
        f"{greeting}Here are **{len(matched_jobs)} matching positions** from the database{refinement_note}:\n\n"
    )

    cards = []
    for idx, j in enumerate(matched_jobs, 1):
        cards.append(
            f"### {idx}. {j.title} — {j.company}\n"
            f"- **Location**: {j.location} | **Level**: {j.experience}\n"
            f"- **Job ID**: `{j.job_id}` | **Match Score**: `{j.match_score}%`\n"
            f"- **Skills Required**: {', '.join(j.skills[:5])}\n"
            f"- **Match Assessment**: {j.match_reason}\n"
            f"- [View Original Posting]({j.source_link})\n"
        )

    closing = (
        f"\nYou can apply directly by typing **'apply to job 1'**, **'apply to job {len(matched_jobs)}'**, "
        f"or by selecting **Apply** on any job card below."
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
    with _session_lock:
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

        now_ts = datetime.now(timezone.utc).isoformat()
        if session_id not in _session_metadata:
            _session_metadata[session_id] = {
                "created_at": now_ts,
                "turns_count": 0,
                "last_active": now_ts
            }
        _session_metadata[session_id]["turns_count"] += 1
        _session_metadata[session_id]["last_active"] = now_ts

    # If OpenAI API is not configured, run deterministic intent orchestrator
    if not is_openai_configured():
        logger.info("OPENAI_API_KEY is not configured. Running fallback intent orchestrator.")
        reply, actions = _fallback_intent_orchestrator(message, active_profile, session_id=session_id)
        with _session_lock:
            if session_id in _session_histories:
                _session_histories[session_id].append({"role": "user", "content": message})
                _session_histories[session_id].append({"role": "assistant", "content": reply})
        return reply, actions

    # Full OpenAI Tool Calling Execution
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        with _session_lock:
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
            with _session_lock:
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

                with _session_lock:
                    _session_histories[session_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result_str
                    })

            with _session_lock:
                synthesis_messages = list(_session_histories[session_id][-14:])

            # Get final synthesized response from model after tools returned
            second_response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=synthesis_messages,
                temperature=0.4
            )
            final_reply = second_response.choices[0].message.content or "I have processed your request."
            with _session_lock:
                _session_histories[session_id].append({"role": "assistant", "content": final_reply})
            return final_reply, collected_actions

        else:
            final_reply = response_msg.content or "How can I assist you with your job search today?"
            with _session_lock:
                _session_histories[session_id].append({"role": "assistant", "content": final_reply})
            return final_reply, collected_actions

    except Exception as exc:
        logger.error(f"Orchestrator LLM execution error: {exc}. Using fallback router.")
        reply, actions = _fallback_intent_orchestrator(message, active_profile, session_id=session_id)
        return reply, actions


def get_session_state(session_id: str) -> Dict[str, Any]:
    """Inspect the internal state and stats of an active chat session."""
    with _session_lock:
        history = list(_session_histories.get(session_id, []))
        profile = _session_profiles.get(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "has_profile": profile is not None,
        "profile": profile
    }


def get_session_metrics(session_id: str) -> Dict[str, Any]:
    """Retrieve detailed interaction and engagement metrics for a chat session."""
    with _session_lock:
        history = list(_session_histories.get(session_id, []))
        profile = _session_profiles.get(session_id)
        recs = list(_session_last_recommendations.get(session_id, []))
        meta = _session_metadata.get(session_id, {})
        now_ts = datetime.now(timezone.utc).isoformat()
        created_at = meta.get("created_at", now_ts)
        last_active = meta.get("last_active", now_ts)
        turns = meta.get("turns_count", 0)

    return {
        "session_id": session_id,
        "message_count": len(history),
        "turns_count": turns,
        "last_active": last_active,
        "has_active_profile": profile is not None,
        "cached_recommendations_count": len(recs),
        "created_at": created_at
    }


def clear_session_state(session_id: str) -> bool:
    """Clear conversation history, cached recommendations, and metadata for a given session."""
    with _session_lock:
        had_state = (
            session_id in _session_histories
            or session_id in _session_last_recommendations
            or session_id in _session_metadata
        )
        _session_histories.pop(session_id, None)
        _session_last_recommendations.pop(session_id, None)
        _session_metadata.pop(session_id, None)
        return had_state


def export_session_transcript(session_id: str, format: str = "markdown") -> Tuple[bool, str, str]:
    """
    Format and export conversation history for a chat session in markdown or plain text.
    """
    clean_id = (session_id or "").strip()
    if not clean_id:
        return False, "Session ID cannot be empty.", ""

    with _session_lock:
        history = list(_session_histories.get(clean_id, []))
        meta = _session_metadata.get(clean_id, {})

    if not history:
        return False, f"No conversation history found for session '{clean_id}'.", ""

    fmt = (format or "markdown").strip().lower()
    created_at = meta.get("created_at", "N/A")

    def _extract_msg(m: Any) -> Tuple[str, str]:
        if isinstance(m, dict):
            r = m.get("role") or "assistant"
            c = m.get("content") or ""
        else:
            r = getattr(m, "role", "assistant") or "assistant"
            c = getattr(m, "content", "") or ""
        return str(r), str(c)

    if fmt == "markdown":
        lines = [
            f"# Chat Transcript — Session `{clean_id}`",
            f"*Created:* {created_at} | *Total Messages:* {len(history)}",
            "",
            "---",
            ""
        ]
        for msg in history:
            role, content = _extract_msg(msg)
            if not content.strip():
                continue
            lines.append(f"### **{role.capitalize()}**")
            lines.append(f"{content}")
            lines.append("")
        return True, "Transcript formatted successfully.", "\n".join(lines)
    else:
        lines = [
            f"=== Chat Transcript: {clean_id} ===",
            f"Created: {created_at} | Messages: {len(history)}",
            "=====================================",
            ""
        ]
        for msg in history:
            role, content = _extract_msg(msg)
            if not content.strip():
                continue
            lines.append(f"[{role.upper()}]: {content}\n")
        return True, "Transcript formatted successfully.", "\n".join(lines)


def evaluate_intent_confidence_and_clarification(user_message: str) -> Dict[str, Any]:
    """
    Score intent classification confidence and generate clarification suggestions when ambiguous.
    """
    text = (user_message or "").strip().lower()
    if not text:
        return {
            "status": "success",
            "needs_clarification": True,
            "top_intent": None,
            "confidence": 0.0,
            "clarification_message": "How can I assist you with your job search today?",
            "quick_replies": [
                {"intent_label": "job_search", "confidence": 0.3, "suggested_prompt": "Find remote Python jobs"},
                {"intent_label": "view_applications", "confidence": 0.3, "suggested_prompt": "Show my applied jobs"},
                {"intent_label": "resume_analysis", "confidence": 0.3, "suggested_prompt": "Review my resume ATS score"}
            ]
        }

    intent_keywords = {
        "job_search": ["job", "jobs", "find", "search", "role", "roles", "hiring", "developer", "engineer", "position", "remote"],
        "view_bookmarks": ["bookmark", "bookmarks", "saved", "favorite", "saved jobs", "shortlist"],
        "view_applications": ["applied", "application", "applications", "interview", "interviews", "status", "pipeline", "offer"],
        "resume_analysis": ["resume", "cv", "ats", "readability", "score", "skills gap", "critique", "profile"],
        "stats_overview": ["stats", "market", "salary", "compensation", "trends", "analytics", "companies"]
    }

    intent_prompts = {
        "job_search": "Find matching job openings",
        "view_bookmarks": "View saved bookmarked jobs",
        "view_applications": "Check job application status",
        "resume_analysis": "Audit resume completeness and ATS score",
        "stats_overview": "Browse job market salary trends"
    }

    scores = {}
    tokens = set(re.findall(r"\b\w+\b", text))
    for intent, kws in intent_keywords.items():
        match_count = sum(1 for kw in kws if kw in text or kw in tokens)
        score = min(1.0, match_count / max(1, len(tokens) * 0.5))
        scores[intent] = round(score, 2)

    sorted_intents = sorted(scores.items(), key=lambda x: -x[1])
    top_intent, top_score = sorted_intents[0]
    second_intent, second_score = sorted_intents[1]

    # Needs clarification if top score is weak, or top two are close and ambiguous
    is_ambiguous = (top_score < 0.35) or (top_score < 0.6 and (top_score - second_score) < 0.15)

    quick_replies = []
    for intent, sc in sorted_intents[:3]:
        quick_replies.append({
            "intent_label": intent,
            "confidence": max(0.2, sc),
            "suggested_prompt": intent_prompts.get(intent, intent)
        })

    clarification_msg = None
    if is_ambiguous:
        clarification_msg = f"I want to make sure I understand. Did you mean to {intent_prompts.get(top_intent, 'search')}, or would you prefer something else?"

    return {
        "status": "success",
        "needs_clarification": is_ambiguous,
        "top_intent": top_intent if not is_ambiguous else None,
        "confidence": top_score,
        "clarification_message": clarification_msg,
        "quick_replies": quick_replies
    }


def update_session_preferences(
    session_id: str,
    target_title: Optional[str] = None,
    location_preference: Optional[str] = None,
    min_salary: Optional[float] = None
) -> Dict[str, Any]:
    """Store conversational memory preferences for a chat session."""
    clean_id = (session_id or "default-session").strip()
    with _session_lock:
        if clean_id not in _session_metadata:
            _session_metadata[clean_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "turns_count": 0,
                "preferences": {}
            }
        prefs = _session_metadata[clean_id].setdefault("preferences", {})
        if target_title is not None:
            prefs["target_title"] = target_title.strip() or None
        if location_preference is not None:
            prefs["location_preference"] = location_preference.strip() or None
        if min_salary is not None:
            prefs["min_salary"] = min_salary
        _session_metadata[clean_id]["last_active"] = datetime.now(timezone.utc).isoformat()
    return dict(prefs)


def get_session_preferences(session_id: str) -> Dict[str, Any]:
    """Retrieve in-memory preferences stored for a session."""
    clean_id = (session_id or "default-session").strip()
    with _session_lock:
        meta = _session_metadata.get(clean_id, {})
        prefs = meta.get("preferences", {})
    return dict(prefs)


def generate_salary_negotiation_script(
    company: str,
    role_title: str,
    initial_offer: float,
    target_counter: float,
    leverage_points: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate professional email counter-offer template and live call talking points.
    """
    clean_company = (company or "the team").strip()
    clean_role = (role_title or "this position").strip()
    delta = round(max(0.0, target_counter - initial_offer), 2)
    delta_pct = round((delta / initial_offer) * 100.0, 1) if initial_offer > 0 else 0.0

    points = [p.strip() for p in (leverage_points or []) if p and p.strip()]
    if points:
        points_text = "\n" + "\n".join(f"- {p}" for p in points) + "\n"
    else:
        points_text = "- Alignment with current market benchmarks for staff-level contributions\n- Demonstrated track record of delivering high-impact system reliability and scale\n"

    email_subject = f"Offer Follow-up: {clean_role} — {clean_company}"
    email_body = f"""Dear Hiring Team,

Thank you very much for extending the offer for the {clean_role} position at {clean_company}. I am genuinely excited about the team's mission and the opportunity to make meaningful contributions to your roadmap.

After careful consideration of the overall package and market data for this tier of responsibility:
{points_text}
I would be delighted to formally accept an updated base compensation of ${target_counter:,.0f}.

I am confident that my technical leadership and domain expertise will deliver tremendous value to {clean_company}. Please let me know if we can schedule a quick brief sync to finalize details.

Warm regards,
Candidate"""

    phone_points = [
        f"Express genuine gratitude and excitement for the offer at {clean_company}.",
        f"Reiterate core value proposition and key competencies: {', '.join(points[:2]) if points else 'Proven architecture and delivery record'}.",
        f"State counter-offer clearly: 'Based on my target scope and market rates, I am looking for ${target_counter:,.0f} base salary.'",
        "Keep the tone collaborative: 'If we can align on this figure, I am ready to sign and get started.'",
        "Be comfortable pausing and letting the hiring manager respond before speaking again."
    ]

    return {
        "status": "success",
        "email_subject": email_subject,
        "email_body": email_body,
        "phone_script_bullet_points": phone_points,
        "counter_delta": delta,
        "counter_delta_pct": delta_pct
    }


def generate_cold_outreach_message(
    company: str,
    role_title: str,
    recipient_type: str = "Recruiter",
    candidate_background: Optional[str] = None,
    channel: str = "LinkedIn"
) -> Dict[str, Any]:
    """
    Generate tailored cold outreach copy optimized for LinkedIn notes (<= 300 chars) or email inquiries.
    """
    clean_company = (company or "").strip()
    clean_role = (role_title or "").strip()
    clean_channel = (channel or "LinkedIn").strip().capitalize()
    rec_label = recipient_type.strip() if recipient_type else "Hiring Team"
    bg = candidate_background.strip() if candidate_background else ""

    if clean_channel == "Linkedin":
        clean_channel = "LinkedIn"

    if clean_channel == "LinkedIn":
        # Character limit on LinkedIn connection invitation is 300 characters
        if bg:
            # Keep it tight so it fits in 300 chars
            base_msg = f"Hi! Saw {clean_company}'s {clean_role} opening. With my background in {bg[:80]}, I'd love to connect and share how I can support your roadmap!"
        else:
            base_msg = f"Hi! I saw the {clean_role} role at {clean_company} and would love to connect. My engineering background aligns closely with the team's mission!"
        
        # Enforce hard 300 char clamp if needed
        if len(base_msg) > 300:
            base_msg = base_msg[:297] + "..."

        return {
            "status": "success",
            "channel": "LinkedIn",
            "subject": None,
            "message": base_msg,
            "character_count": len(base_msg),
            "is_within_channel_limit": len(base_msg) <= 300
        }
    else:
        subject = f"Inquiry: {clean_role} — {clean_company}"
        bg_section = (
            f"With experience across {bg}, I have delivered robust systems and solved high-impact technical problems."
            if bg
            else f"I have a track record of building performant systems and driving high-impact technical deliverables."
        )
        body = f"""Hi {rec_label},

I noticed the {clean_role} position at {clean_company} and wanted to reach out directly.

{bg_section}

I would welcome the opportunity to connect for 10-15 minutes to share how my experience directly aligns with {clean_company}'s engineering goals.

Best regards,
Candidate"""
        return {
            "status": "success",
            "channel": "Email",
            "subject": subject,
            "message": body,
            "character_count": len(body),
            "is_within_channel_limit": len(body) <= 3000
        }


def coach_star_behavioral_answer(
    question: str,
    raw_notes: str,
    target_role: Optional[str] = None
) -> Dict[str, Any]:
    """
    Coach and format raw behavioral interview experience into a structured STAR response.
    """
    clean_q = (question or "").strip()
    clean_notes = (raw_notes or "").strip()
    role_ctx = f" for the {target_role.strip()} role" if target_role and target_role.strip() else ""

    situation = f"In my previous engineering tenure, our team faced a pivotal operational challenge: {clean_notes[:180]}."
    task = f"My objective{role_ctx} was to diagnose the root bottlenecks, maintain system stability, and establish resilient workflows without customer interruption."
    action = "To achieve this, I took initiative to design a comprehensive mitigation roadmap: decomposing the problem into isolated components, leading technical communication, and iteratively deploying validated fixes."
    result = "As a direct outcome, we restored normal operations with zero data loss, improved pipeline throughput by 35%, and codified post-mortem lessons into automated safeguards."

    structured_answer = f"**Situation:** {situation}\n\n**Task:** {task}\n\n**Action:** {action}\n\n**Result:** {result}"

    tips = [
        "Focus at least 60% of your delivery time on the Action and Result pillars.",
        "Quantify your results with concrete percentages, latency drops, or time saved.",
        "Emphasize personal actions using 'I decided' and 'I implemented' rather than generic 'we'.",
        "Keep the entire verbal response within 90 to 120 seconds."
    ]

    return {
        "status": "success",
        "question": clean_q,
        "situation": situation,
        "task": task,
        "action": action,
        "result": result,
        "coaching_tips": tips,
        "full_structured_answer": structured_answer
    }


def generate_recruiter_followup_cadence(
    company: str,
    role_title: str,
    current_stage: str = "applied"
) -> Dict[str, Any]:
    """
    Generate an optimal multi-step recruiter follow-up schedule and tailored email templates.
    """
    clean_company = (company or "Target Company").strip()
    clean_role = (role_title or "Software Engineer").strip()
    stage = (current_stage or "applied").strip().lower()

    if stage in ("interview", "interviews"):
        steps = [
            {
                "step_number": 1,
                "days_after_event": 1,
                "step_name": "Post-Interview Thank You Note",
                "recommended_channel": "Email",
                "subject_line": f"Thank You — {clean_role} Interview at {clean_company}",
                "template_body": f"Hi Team,\n\nThank you for the thoughtful conversation today regarding the {clean_role} role at {clean_company}. I enjoyed learning more about your technical roadmap and team goals.\n\nLooking forward to hearing about the next steps!\n\nBest regards,\nCandidate"
            },
            {
                "step_number": 2,
                "days_after_event": 7,
                "step_name": "Stage Status Check-in",
                "recommended_channel": "Email",
                "subject_line": f"Checking In — {clean_role} Application at {clean_company}",
                "template_body": f"Hi Team,\n\nI hope your week is going well! I am writing to check in on the status of my {clean_role} application following our recent discussion. Please let me know if there are any additional materials or code samples I can provide.\n\nWarm regards,\nCandidate"
            },
            {
                "step_number": 3,
                "days_after_event": 14,
                "step_name": "Final Gentle Ping & Timeline Sync",
                "recommended_channel": "Email",
                "subject_line": f"Following Up — {clean_role} Role at {clean_company}",
                "template_body": f"Hi Team,\n\nI wanted to gently follow up on our interview for {clean_role}. I remain very enthusiastic about {clean_company}'s engineering work. If the team has moved in another direction, I completely understand and appreciate your feedback.\n\nBest regards,\nCandidate"
            }
        ]
        advisory = "For interview follow-ups, send Step 1 within 24 hours while discussions are fresh. Avoid messaging more frequently than once every 5-7 business days."
    else:
        steps = [
            {
                "step_number": 1,
                "days_after_event": 5,
                "step_name": "Initial Application Confirmation",
                "recommended_channel": "Email / LinkedIn",
                "subject_line": f"Application Follow-up: {clean_role} — {clean_company}",
                "template_body": f"Hi Recruiting Team,\n\nI recently submitted my application for the {clean_role} position at {clean_company} and wanted to briefly reiterate my excitement for the role. With my background in high-scale systems, I would love the chance to connect.\n\nBest regards,\nCandidate"
            },
            {
                "step_number": 2,
                "days_after_event": 12,
                "step_name": "Mid-Cadence Inbound Check",
                "recommended_channel": "Email",
                "subject_line": f"Checking In: {clean_role} Opening at {clean_company}",
                "template_body": f"Hi Team,\n\nI am writing to inquire if you are still actively reviewing candidates for the {clean_role} vacancy. I am happy to answer any questions or provide a quick overview of recent project deliverables.\n\nWarm regards,\nCandidate"
            },
            {
                "step_number": 3,
                "days_after_event": 21,
                "step_name": "Final Polite Close",
                "recommended_channel": "Email",
                "subject_line": f"Status Update on {clean_role} Application — {clean_company}",
                "template_body": f"Hi Team,\n\nI wanted to check one last time regarding my submission for {clean_role}. If the position has been filled, thank you for your consideration, and I hope we can stay in touch for future openings.\n\nBest,\nCandidate"
            }
        ]
        advisory = "For submitted applications, allow recruiters 4-5 business days before the first touchpoint. Always keep check-ins concise, polite, and value-oriented."

    return {
        "status": "success",
        "company": clean_company,
        "role_title": clean_role,
        "current_stage": stage,
        "total_steps": len(steps),
        "recommended_steps": steps,
        "advisory_note": advisory
    }


def generate_post_interview_thank_you_email(
    company: str,
    role_title: str,
    interviewer_name: Optional[str] = None,
    key_discussion_points: Optional[List[str]] = None,
    custom_gratitude_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a tailored, high-converting thank you email incorporating specific interview takeaways.
    """
    clean_company = (company or "the team").strip()
    clean_role = (role_title or "this role").strip()
    clean_interviewer = (interviewer_name or "Hiring Team").strip()

    points = [p.strip() for p in (key_discussion_points or []) if p and p.strip()]
    if points:
        takeaway_text = "I particularly enjoyed our discussion around " + ", ".join(points) + ". It reinforced my excitement about how my background in distributed systems can help accelerate your milestones."
    else:
        takeaway_text = "I particularly enjoyed learning more about your technical architecture and the team's upcoming roadmap priorities."

    personal_remark = f"\n\n{custom_gratitude_note.strip()}" if custom_gratitude_note and custom_gratitude_note.strip() else ""

    subject = f"Thank you — {clean_role} Interview ({clean_interviewer})"
    body = f"""Hi {clean_interviewer},

Thank you very much for your time and engaging conversation today about the {clean_role} opportunity at {clean_company}.

{takeaway_text}{personal_remark}

Please feel free to reach out if you need any additional code repositories, design documents, or references. I look forward to hearing about the next steps!

Warm regards,
Candidate"""

    return {
        "status": "success",
        "company": clean_company,
        "role_title": clean_role,
        "interviewer_name": clean_interviewer,
        "subject_line": subject,
        "email_body": body,
        "character_count": len(body),
        "delivery_tip": "Send within 24 hours of your interview, ideally before end of business day."
    }


def generate_elevator_pitch(
    job_title: str,
    target_skills: Optional[List[str]] = None,
    candidate_summary: Optional[str] = None,
    candidate_skills: Optional[List[str]] = None,
    tone: Optional[str] = "confident"
) -> Dict[str, Any]:
    """
    Generate tailored 30-second and 60-second verbal elevator pitches highlighting skill alignment.
    """
    clean_title = (job_title or "Software Engineer").strip()
    c_skills = [s.strip() for s in (candidate_skills or []) if s and s.strip()]
    t_skills = [s.strip() for s in (target_skills or []) if s and s.strip()]

    overlap = [s for s in c_skills if any(s.lower() == t.lower() for t in t_skills)]
    featured_skills = overlap[:3] if overlap else (c_skills[:3] or t_skills[:3] or ["modern software engineering", "problem solving"])
    skills_phrase = ", ".join(featured_skills)

    summary_phrase = (candidate_summary or "experienced engineering professional with a strong track record of delivery").strip()
    if summary_phrase.endswith('.'):
        summary_phrase = summary_phrase[:-1]

    pitch_30s = (
        f"I'm an {summary_phrase}, specializing in {skills_phrase}. "
        f"I focus on designing reliable, high-performance systems and am excited about the {clean_title} role because it directly leverages my strengths in shipping impactful features."
    )

    pitch_60s = (
        f"Throughout my career as an {summary_phrase}, I've prioritized building scalable solutions using {skills_phrase}. "
        f"In my recent projects, I drove measurable improvements in reliability, performance, and developer velocity. "
        f"I'm targeting this {clean_title} position because your team's mission aligns with how I like to work: tackling complex domain challenges while maintaining engineering excellence. "
        f"I'm eager to bring that same rigor and velocity to help accelerate your milestones."
    )

    key_points = [
        f"Direct alignment with {clean_title} technical requirements",
        f"Core expertise in {skills_phrase}",
        "Demonstrated ownership from architectural design to production monitoring"
    ]

    return {
        "job_title": clean_title,
        "pitch_30s": pitch_30s,
        "pitch_60s": pitch_60s,
        "key_selling_points": key_points,
        "call_to_action": f"I'd love to learn more about the team's key priorities for the {clean_title} role this quarter."
    }


def generate_technical_interview_cheatsheet(
    technology: str,
    seniority: Optional[str] = "Senior",
    focus_areas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate high-yield technical mechanics, pitfalls, and interview Q&As for a specific tool or tech stack.
    """
    clean_tech = (technology or "Python").strip()
    clean_sen = (seniority or "Senior").strip()
    t_lower = clean_tech.lower()

    if "python" in t_lower:
        mechanics = [
            "GIL & Concurrency: Understand asyncio event loop vs multiprocessing for CPU-bound tasks.",
            "Memory Management: Reference counting combined with generational cyclic GC.",
            "Data Model & Dunder: __enter__/__exit__ protocols, generators yield from, __slots__ for memory optimization."
        ]
        pitfalls = [
            "Mutable default arguments (def f(arg=[]) evaluated once at definition time).",
            "Blocking calls inside async coroutines starving the asyncio event loop.",
            "Misunderstanding late binding in Python closures inside loops."
        ]
        questions = [
            {"q": "How does asyncio differ from threading in Python?", "a": "Asyncio uses single-threaded cooperative multitasking with an event loop, avoiding OS thread overhead and race conditions on GIL, ideal for high-concurrency I/O."},
            {"q": "When should you use generators over list comprehensions?", "a": "Generators provide lazy stream evaluation yielding one item at a time, keeping memory footprint O(1) across massive datasets."}
        ]
        tips = [
            "Always profile memory allocations with tracemalloc and CPU with cProfile before optimizing.",
            "Use Pydantic v2 or dataclasses for strict runtime data validation contracts."
        ]
    elif any(k in t_lower for k in ["k8s", "kube", "kubernetes"]):
        mechanics = [
            "Control Plane Reconciliation: API Server, etcd quorum, Kube-Controller-Manager reconcilers.",
            "Networking: CNI overlays, kube-proxy iptables/IPVS mode, Ingress routing.",
            "Pod Lifecycle: initContainers, preStop lifecycle hooks, liveness vs readiness vs startup probes."
        ]
        pitfalls = [
            "Setting CPU limits triggering aggressive CFS kernel throttling.",
            "Missing readiness probes causing traffic black-holing during rolling deployments.",
            "Hardcoding pod IPs rather than relying on CoreDNS ClusterIP services."
        ]
        questions = [
            {"q": "Explain the difference between Liveness and Readiness probes.", "a": "Liveness probes restart a deadlocked container; Readiness probes temporarily remove the pod from Service endpoints until it can serve traffic."},
            {"q": "How does etcd ensure consensus during network partitions?", "a": "etcd uses the Raft protocol requiring a strict majority quorum ((N/2) + 1) to commit state transitions."}
        ]
        tips = [
            "Always implement PodDisruptionBudgets (PDB) and topology spread constraints for zero-downtime nodes draining.",
            "Leverage Horizontal Pod Autoscaler (HPA) using custom Prometheus metrics over raw CPU/RAM."
        ]
    elif any(k in t_lower for k in ["react", "next", "frontend"]):
        mechanics = [
            "Fiber Reconciler: Work-in-progress trees, double buffering, and prioritization queues.",
            "State & Effects: Batching updates, dependency arrays, cleanup functions.",
            "SSR / Server Components: Streaming HTML hydration, selective boundary Suspense."
        ]
        pitfalls = [
            "Unstable object references passed as props triggering cascading child re-renders.",
            "Mutating state directly instead of creating new immutable references.",
            "Stale closures capturing outdated state inside useEffect or useCallback callbacks."
        ]
        questions = [
            {"q": "How does React 18 Concurrent Mode handle high-priority user input?", "a": "It uses startTransition to interrupt non-urgent renders, keeping typing and clicking responsive."},
            {"q": "What is the benefit of React Server Components (RSC)?", "a": "RSCs execute solely on the server, zero client bundle size impact, direct DB access without client API boilerplate."}
        ]
        tips = [
            "Keep state as local as possible; avoid premature global state libraries.",
            "Use React DevTools Profiler to identify commit phase bottlenecks."
        ]
    else:
        mechanics = [
            f"Core Architecture: Fundamental execution model, state handling, and lifecycle of {clean_tech}.",
            f"Concurrency & Scalability: Horizontal scaling boundaries and resource constraints in {clean_tech}.",
            "Interface Contracts: Strong boundary typing, versioning, and backwards compatibility."
        ]
        pitfalls = [
            f"Premature optimization without profiling real {clean_tech} bottlenecks in staging.",
            "Ignoring failure domains and distributed split-brain scenarios.",
            "Lack of observability, telemetry, and structured tracing."
        ]
        questions = [
            {"q": f"How do you design for high availability in {clean_tech}?", "a": "Decouple services using asynchronous messaging queues, stateless service replicas, and idempotent retry policies."},
            {"q": f"What is the most critical operational metric when monitoring {clean_tech}?", "a": "The 4 Golden Signals: Latency (p99), Traffic, Error Rate, and Saturation."}
        ]
        tips = [
            f"Follow 12-factor application methodology when shipping {clean_tech} services to production.",
            "Implement automated regression testing and canary rollouts."
        ]

    return {
        "technology": clean_tech,
        "seniority": clean_sen,
        "core_mechanics": mechanics,
        "frequent_pitfalls": pitfalls,
        "high_probability_questions": questions,
        "architecture_tips": tips
    }


def generate_project_talking_points(
    project_name: str,
    tech_stack: Optional[List[str]] = None,
    project_summary: Optional[str] = None,
    challenges: Optional[List[str]] = None,
    target_role: Optional[str] = "Senior Backend Engineer"
) -> Dict[str, Any]:
    """
    Generate structured, high-impact interview talking points for technical portfolio projects.
    """
    p_name = (project_name or "Architecture Showcase").strip()
    role = (target_role or "Senior Backend Engineer").strip()
    stack = [s.strip() for s in (tech_stack or []) if s and s.strip()]
    stack_phrase = ", ".join(stack) if stack else "modern distributed backend services"
    summary = (project_summary or f"A high-throughput application designed with {stack_phrase}").strip()

    highlights = [
        f"Designed core architecture of {p_name} utilizing {stack_phrase}.",
        "Emphasized modular service decoupling, explicit API boundaries, and low latency.",
        "Implemented end-to-end telemetry and automated regression test harnesses."
    ]

    tradeoffs = [
        f"Selected {stack[0] if stack else 'current framework'} for developer ergonomics and concurrency model over lower-level alternatives.",
        "Prioritized eventual consistency across service replicas to maximize write availability under peak load.",
        "Favor explicit data schemas and strict validation over dynamic schema-less serialization."
    ]

    metrics = [
        "Maintained sub-50ms p99 response times under simulated concurrency spikes.",
        "Reduced resource saturation by 35% via intelligent connection pooling and in-memory caching.",
        "Zero data corruption or state divergence during simulated node failures."
    ]

    opening = (
        f"In my project {p_name}, I set out to solve {summary.lower() if summary.lower().startswith('a ') else summary}. "
        f"Leveraging {stack_phrase}, I led the architectural decisions from initial schema design to production hardening, "
        f"with a keen focus on the scalability requirements relevant to this {role} role."
    )

    return {
        "project_name": p_name,
        "target_role": role,
        "architectural_highlights": highlights,
        "technical_tradeoffs": tradeoffs,
        "scale_and_performance_metrics": metrics,
        "suggested_opening_pitch": opening
    }


def generate_system_design_scenario(
    topic: Optional[str] = "Distributed Rate Limiter",
    scale_target: Optional[str] = "100M Daily Active Users",
    seniority: Optional[str] = "Senior"
) -> Dict[str, Any]:
    """
    Generate high-yield mock system design interview problem with requirements, sizing, and architecture components.
    """
    clean_topic = (topic or "Distributed Rate Limiter").strip()
    clean_scale = (scale_target or "100M Daily Active Users").strip()
    clean_sen = (seniority or "Senior").strip()

    t_lower = clean_topic.lower()

    if "rate" in t_lower or "limit" in t_lower:
        func = [
            "Limit client requests based on IP, User ID, or API Token tier.",
            "Return HTTP 429 Too Many Requests with Retry-After header when threshold exceeded.",
            "Support configurable sliding window and token bucket policies per endpoint."
        ]
        non_func = [
            "Ultra-low latency overhead (< 2ms per request evaluation).",
            "High availability across global multi-region deployments (fail-open policy).",
            "Accuracy within 99% under distributed concurrency spikes."
        ]
        sizing = [
            "100M DAU * 50 requests/day = 5 Billion requests/day = ~58,000 QPS average, 120,000 QPS peak.",
            "Token state memory: 100M keys * 32 bytes = 3.2 GB RAM (easily fits in Redis cluster)."
        ]
        components = [
            "API Gateway / Reverse Proxy with embedded Envoy rate limiter filter",
            "Distributed In-Memory Store (Redis Cluster running Lua scripts for atomic token evaluation)",
            "Configuration Management Service (Dynamic rule propagation via ZooKeeper/Consul)"
        ]
        deep_dives = [
            "Token Bucket vs Leaky Bucket vs Sliding Window Log algorithms",
            "Handling race conditions between concurrent requests (Redis EVAL Lua script vs Redis Cell)",
            "Local in-memory caching with batch sync to reduce Redis network roundtrips"
        ]
    elif "cache" in t_lower:
        func = [
            "Key-value storage with fast get(k) and set(k, v) operations.",
            "Configurable TTL eviction and LRU / LFU cache eviction policies.",
            "Persistence snapshotting (RDB / AOF equivalent)."
        ]
        non_func = [
            "Sub-millisecond read/write latency.",
            "Strong high availability with master-replica failover.",
            "Horizontal sharding across memory nodes."
        ]
        sizing = [
            "Read-heavy workload: 100:1 read-to-write ratio.",
            "30TB total cached dataset across 100 nodes with 512GB RAM each."
        ]
        components = [
            "Consistent Hashing client ring",
            "In-memory storage engine with hash table + doubly linked list for LRU",
            "Consensus coordinator for cluster membership"
        ]
        deep_dives = [
            "Cache stampede (thundering herd) mitigation using singleflight mutexes",
            "Consistent hashing with virtual nodes for uniform key distribution",
            "Cache invalidation patterns (Write-through vs Write-behind vs Cache-aside)"
        ]
    else:
        func = [
            f"Core transactional workflows for {clean_topic}.",
            "Idempotent API endpoints for state mutations.",
            "Search and query retrieval capabilities."
        ]
        non_func = [
            f"Scale handling for {clean_scale}.",
            "P99 latency under 100ms under load.",
            "99.99% uptime availability SLA."
        ]
        sizing = [
            f"Estimated peak throughput: 50,000+ QPS to support {clean_scale}.",
            "Data storage growth: 500GB - 2TB new transactional data per month."
        ]
        components = [
            "Load Balancers (Layer 4 / Layer 7)",
            "Stateless Microservice Replicas",
            "Distributed Database (Primary-Replica with read replicas) + Caching Layer",
            "Async Message Queue (Kafka/RabbitMQ) for deferred background tasks"
        ]
        deep_dives = [
            "Database sharding and partition key selection strategies",
            "Event-driven architecture and asynchronous decoupled processing",
            "Zero-downtime database schema migration strategies"
        ]

    return {
        "topic": clean_topic,
        "seniority": clean_sen,
        "functional_requirements": func,
        "non_functional_requirements": non_func,
        "back_of_envelope_calculations": sizing,
        "key_architectural_components": components,
        "deep_dive_topics": deep_dives
    }


def generate_resignation_letter(
    company_name: str,
    manager_name: Optional[str] = "Manager",
    last_working_day: Optional[str] = "two weeks from today",
    tenure_years: Optional[float] = 2.0,
    tone: Optional[str] = "Professional and Warm",
    handover_focus: Optional[str] = "System documentation and knowledge transfer"
) -> Dict[str, Any]:
    """Generate a high-empathy, professional resignation letter and offboarding transition plan."""
    company = (company_name or "the Company").strip()
    mgr = (manager_name or "Manager").strip()
    lwd = (last_working_day or "two weeks from today").strip()
    tenure = float(tenure_years or 2.0)
    focus = (handover_focus or "System documentation and knowledge transfer").strip()

    subject = f"Resignation Notice - [Your Name] - Final Day: {lwd}"
    tenure_text = f"over the past {int(tenure)} years" if tenure >= 1 else "during my tenure"

    body = (
        f"Dear {mgr},\n\n"
        f"Please accept this letter as formal notification that I am resigning from my position at {company}. "
        f"My proposed last working day will be {lwd}.\n\n"
        f"I want to express my sincere gratitude for the opportunities I have had {tenure_text}. "
        f"I have truly enjoyed collaborating with the team, tackling technical challenges, and contributing to our engineering milestones. "
        f"Thank you for your guidance, mentorship, and support throughout my time here.\n\n"
        f"During this notice window, my highest priority will be ensuring an orderly handover. Specifically, I will focus on {focus.lower()}, "
        f"wrapping up active tickets, and transferring contextual knowledge to teammates to ensure uninterrupted delivery.\n\n"
        f"I wish {company} and the entire team continued momentum and success. Please let me know how else I can assist during this transition.\n\n"
        f"Warm regards,\n[Your Name]"
    )

    checklist = [
        f"Schedule 1:1 conversation with {mgr} prior to sending formal written notice.",
        "Prepare centralized handover document detailing services, runbooks, and credential locations.",
        "Coordinate redistribution of on-call rotation schedules and Jira ticket backlogs.",
        "Transfer admin permissions and document critical API tokens with team leads.",
        "Archive personal documents from work workstation and review post-employment IP obligations."
    ]

    tips = [
        "Deliver the resignation verbally first during a private sync before transmitting the written copy to HR.",
        "Maintain positive neutrality regarding future endeavors and avoid venting legacy grievances.",
        "Keep daily commit hygiene high through your final week to solidify your professional reputation."
    ]

    return {
        "company_name": company,
        "manager_name": mgr,
        "last_working_day": lwd,
        "subject_line": subject,
        "letter_body": body,
        "transition_checklist": checklist,
        "etiquette_tips": tips
    }







