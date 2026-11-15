"""
Unit and integration test suite for Hidani AI Job Assistant backend.
Tests health check, job retrieval/filtering, application submission/deduplication/withdrawal,
and multi-agent orchestrator intent routing.
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.services.db_service import (
    apply_to_job,
    get_application_status,
    get_job_by_id,
    load_jobs,
    search_jobs,
    withdraw_application,
)
from app.agents.orchestrator import _fallback_intent_orchestrator, run_orchestrator


class BackendTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_job_id = "job-00001"

    def tearDown(self):
        # Clean up any test application submitted for job-00001
        withdraw_application(self.test_job_id)

    def test_health_check(self):
        """Test system diagnostic health check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("database", data)
        self.assertGreater(data["database"]["total_jobs"], 0)

    def test_jobs_listing_and_filtering(self):
        """Test paginated jobs listing and multi-field query filters."""
        # 1. Base pagination
        res = self.client.get("/api/jobs?limit=5&offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["jobs"]), 5)
        self.assertGreaterEqual(data["total"], 5)

        # 2. Query filter
        res_query = self.client.get("/api/jobs?query=python&limit=3")
        self.assertEqual(res_query.status_code, 200)
        data_query = res_query.json()
        self.assertLessEqual(len(data_query["jobs"]), 3)

        # 3. Location filter
        res_loc = self.client.get("/api/jobs?location=remote&limit=3")
        self.assertEqual(res_loc.status_code, 200)
        data_loc = res_loc.json()
        for j in data_loc["jobs"]:
            self.assertIn("remote", j["location"].lower())

    def test_get_job_details_by_id(self):
        """Test retrieving specific job details by ID."""
        # Valid ID
        res = self.client.get(f"/api/jobs/{self.test_job_id}")
        self.assertEqual(res.status_code, 200)
        job = res.json()
        self.assertEqual(job["job_id"], self.test_job_id)
        self.assertIn("title", job)
        self.assertIn("company", job)

        # Invalid ID -> 404
        res_404 = self.client.get("/api/jobs/non-existent-job-99999")
        self.assertEqual(res_404.status_code, 404)

    def test_application_lifecycle_and_withdrawal(self):
        """Test application submission, duplicate prevention, and withdrawal."""
        # Ensure clean state
        withdraw_application(self.test_job_id)

        # 1. Apply to job
        success, msg, app_data = apply_to_job(self.test_job_id)
        self.assertTrue(success)
        self.assertIsNotNone(app_data)
        self.assertEqual(app_data["job_id"], self.test_job_id)

        # 2. Duplicate prevention
        dup_success, dup_msg, _ = apply_to_job(self.test_job_id)
        self.assertFalse(dup_success)
        self.assertIn("already applied", dup_msg.lower())

        # 3. Verify in applications list
        res_apps = self.client.get("/api/applications")
        self.assertEqual(res_apps.status_code, 200)
        applied_ids = [a["job_id"] for a in res_apps.json()]
        self.assertIn(self.test_job_id, applied_ids)

        # 4. Withdraw application via DELETE endpoint
        res_del = self.client.delete(f"/api/applications/{self.test_job_id}")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.json()["status"], "success")

        # 5. Verify withdrawal reflected in list
        res_apps_after = self.client.get("/api/applications")
        remaining_ids = [a["job_id"] for a in res_apps_after.json()]
        self.assertNotIn(self.test_job_id, remaining_ids)

        # 6. Withdrawing again yields 404
        res_del_404 = self.client.delete(f"/api/applications/{self.test_job_id}")
        self.assertEqual(res_del_404.status_code, 404)

    def test_orchestrator_intents_and_withdrawal(self):
        """Test orchestrator fallback routing for greetings, search, apply, and withdrawal."""
        session_id = "test-session-suite"

        # 1. Greetings intent
        reply, actions = _fallback_intent_orchestrator("Hello there!", session_id=session_id)
        self.assertIn("Hello", reply)

        # 2. Capabilities intent
        reply, actions = _fallback_intent_orchestrator("What can you do?", session_id=session_id)
        self.assertIn("career assistant", reply.lower())

        # 3. Search intent
        reply, actions = _fallback_intent_orchestrator("Recommend python jobs", session_id=session_id)
        self.assertTrue(any(a.type == "RECOMMEND_JOBS" for a in actions))

        # 4. Apply intent for ordinal job 1
        reply_apply, actions_apply = _fallback_intent_orchestrator("apply to job 1", session_id=session_id)
        self.assertTrue(any(a.type == "APPLICATION_SUBMITTED" for a in actions_apply))

        # 5. Withdraw intent
        reply_withdraw, actions_withdraw = _fallback_intent_orchestrator("withdraw application 1", session_id=session_id)
        self.assertTrue(any(a.type == "APPLICATION_WITHDRAWN" for a in actions_withdraw))

    def test_session_reset(self):
        """Test chat session reset endpoint and natural intent."""
        session_id = "test-reset-session"

        # Initialize session state with a message
        self.client.post("/api/chat", json={"message": "hi", "session_id": session_id})
        state_before = self.client.get(f"/api/state/{session_id}").json()
        self.assertGreater(state_before["message_count"], 0)

        # Reset session
        res_reset = self.client.post(f"/api/chat/reset/{session_id}")
        self.assertEqual(res_reset.status_code, 200)
        self.assertEqual(res_reset.json()["status"], "success")

        # Verify state is cleared
        state_after = self.client.get(f"/api/state/{session_id}").json()
        self.assertEqual(state_after["message_count"], 0)

        # Test reset intent
        reply, actions = _fallback_intent_orchestrator("reset chat", session_id=session_id)
        self.assertTrue(any(a.type == "SESSION_RESET" for a in actions))

    def test_job_stats_endpoint(self):
        """Test aggregated market stats endpoint /api/jobs/stats."""
        res = self.client.get("/api/jobs/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total_jobs"], 1600)
        self.assertGreater(data["remote_jobs"], 0)
        self.assertIsInstance(data["top_locations"], list)
        self.assertGreater(len(data["top_locations"]), 0)
        self.assertIsInstance(data["top_skills"], list)
        self.assertGreater(len(data["top_skills"]), 0)
        self.assertIn("Mid-Level", data["experience_breakdown"])

    def test_direct_application_endpoints(self):
        """Test RESTful POST /api/applications and GET /api/applications/{job_id}."""
        # Clean up any leftover test application
        withdraw_application(self.test_job_id)

        # 1. Successful direct application
        res_post = self.client.post("/api/applications", json={"job_id": self.test_job_id})
        self.assertEqual(res_post.status_code, 201)
        data_post = res_post.json()
        self.assertEqual(data_post["status"], "success")
        self.assertEqual(data_post["application"]["job_id"], self.test_job_id)

        # 2. Retrieve application details by job ID
        res_get = self.client.get(f"/api/applications/{self.test_job_id}")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["job_id"], self.test_job_id)
        self.assertEqual(data_get["status"], "Applied")

        # 3. Duplicate application yields 409 Conflict
        res_dup = self.client.post("/api/applications", json={"job_id": self.test_job_id})
        self.assertEqual(res_dup.status_code, 409)

        # 4. Non-existent job application yields 404 Not Found
        res_missing = self.client.get("/api/applications/non-existent-job-xyz")
        self.assertEqual(res_missing.status_code, 404)

        # 5. Apply to invalid job yields 404 Not Found
        res_invalid = self.client.post("/api/applications", json={"job_id": "non-existent-job-xyz"})
        self.assertEqual(res_invalid.status_code, 404)

        # 6. Withdraw application and verify it no longer exists
        res_del = self.client.delete(f"/api/applications/{self.test_job_id}")
        self.assertEqual(res_del.status_code, 200)

        res_after = self.client.get(f"/api/applications/{self.test_job_id}")
        self.assertEqual(res_after.status_code, 404)

    def test_applications_query_filters(self):
        """Test GET /api/applications with status and search query parameters."""
        withdraw_application(self.test_job_id)
        # Apply to test job
        self.client.post("/api/applications", json={"job_id": self.test_job_id})

        # 1. Filter by status
        res_status = self.client.get("/api/applications?status=Applied")
        self.assertEqual(res_status.status_code, 200)
        self.assertTrue(any(a["job_id"] == self.test_job_id for a in res_status.json()))

        # Non-matching status
        res_none = self.client.get("/api/applications?status=Rejected")
        self.assertEqual(res_none.status_code, 200)
        self.assertFalse(any(a["job_id"] == self.test_job_id for a in res_none.json()))

        # 2. Filter by search keyword
        res_search = self.client.get(f"/api/applications?search={self.test_job_id}")
        self.assertEqual(res_search.status_code, 200)
        self.assertEqual(len(res_search.json()), 1)
        self.assertEqual(res_search.json()[0]["job_id"], self.test_job_id)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_heuristic_reason_clean_designation(self):
        """Verify heuristic matcher reason produces clean text with designation and skills."""
        from app.agents.matcher_agent import _generate_heuristic_reason
        job = {"title": "Backend Engineer", "company": "Hidani", "skills": ["Python", "FastAPI"], "experience": "Mid-Level"}
        reason = _generate_heuristic_reason(job, candidate_skills=["Python"], candidate_designation="Senior Python Developer", score=90.0)
        self.assertIn("Senior Python Developer", reason)
        self.assertNotIn("90.0 profile", reason)

    def test_applier_agent_edge_cases_and_status(self):
        """Test Applier Agent validation and targeted status querying."""
        from app.agents.applier_agent import apply_for_job, get_applications_summary, withdraw_job_application

        # Empty job ID apply
        res_empty = apply_for_job("")
        self.assertFalse(res_empty["success"])
        self.assertIsNone(res_empty["application"])
        self.assertIsNone(res_empty["job"])

        # Non-existent job ID apply
        res_none = apply_for_job("job-does-not-exist-000")
        self.assertFalse(res_none["success"])
        self.assertIsNone(res_none["job"])

        # Empty withdraw
        res_withdraw_empty = withdraw_job_application("")
        self.assertFalse(res_withdraw_empty["success"])
        self.assertIsNone(res_withdraw_empty["job_id"])

        # Targeted status for job not applied to
        summary = get_applications_summary("job-00001")
        self.assertEqual(summary["count"], 0)
        self.assertIn("haven't submitted an application", summary["message"])

        # Status intent in orchestrator for targeted non-applied job
        reply, actions = _fallback_intent_orchestrator("status of job-00001", session_id="test-targeted-status")
        self.assertIn("haven't submitted an application", reply.lower())
        self.assertTrue(any(a.type == "APPLICATIONS_LIST" for a in actions))

        # Whitespace handling in get_application_status
        all_apps = get_application_status()
        ws_apps = get_application_status("   ")
        self.assertEqual(len(all_apps), len(ws_apps))

    def test_endpoint_input_sanitization_and_empty_id_validation(self):
        """Test endpoint 400 validation on whitespace or empty job IDs."""
        # 1. GET /api/jobs with whitespace id
        res_job_empty = self.client.get("/api/jobs/%20%20")
        self.assertEqual(res_job_empty.status_code, 400)
        self.assertIn("cannot be empty", res_job_empty.json()["detail"].lower())

        # 2. GET /api/applications with whitespace id
        res_app_empty = self.client.get("/api/applications/%20%20")
        self.assertEqual(res_app_empty.status_code, 400)
        self.assertIn("cannot be empty", res_app_empty.json()["detail"].lower())

        # 3. DELETE /api/applications with whitespace id
        res_del_empty = self.client.delete("/api/applications/%20%20")
        self.assertEqual(res_del_empty.status_code, 400)
        self.assertIn("cannot be empty", res_del_empty.json()["detail"].lower())

        # 4. get_job_by_id with empty or whitespace returns None immediately
        self.assertIsNone(get_job_by_id(""))
        self.assertIsNone(get_job_by_id("   "))

        # 5. GET /api/jobs with whitespace query falls back cleanly
        res_ws_query = self.client.get("/api/jobs?query=%20%20&limit=5")
        self.assertEqual(res_ws_query.status_code, 200)
        self.assertEqual(len(res_ws_query.json()["jobs"]), 5)

    def test_db_cache_and_invalidation(self):
        """Test in-memory jobs cache and cache invalidation in db_service."""
        from app.services.db_service import invalidate_jobs_cache

        # Clear cache and reload
        invalidate_jobs_cache()
        jobs1 = load_jobs()
        self.assertGreater(len(jobs1), 0)

        # Subsequent load returns identical cached reference
        jobs2 = load_jobs()
        self.assertIs(jobs1, jobs2)

        # O(1) indexed lookup finds existing job
        job = get_job_by_id(self.test_job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["job_id"], self.test_job_id)

        # Cache invalidation forces fresh list
        invalidate_jobs_cache()
        jobs3 = load_jobs()
        self.assertEqual(len(jobs3), len(jobs1))

    def test_frontend_assets_served(self):
        """Test serving single-page application index.html and app.js."""
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)

        res_js = self.client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("javascript", res_js.headers.get("content-type", "").lower())
        self.assertIn("closeAllModals", res_js.text)
        self.assertIn("updateApplicationStatusFromModal", res_js.text)

    def test_heuristic_resume_parser_skills_and_fields(self):
        """Test heuristic resume parser extracts non-alphanumeric skills like C++ and C# cleanly."""
        from app.agents.parser_agent import _heuristic_resume_parser

        sample_resume = (
            "Jane Doe\n"
            "jane.doe@example.com | +1 555-234-5678\n"
            "Site Reliability Engineer with 6+ years of experience\n"
            "Skills: C++, C#, Python, Docker, Kubernetes, Terraform, CI/CD\n"
            "Education: B.S. in Computer Science, State University\n"
        )
        parsed = _heuristic_resume_parser(sample_resume)
        self.assertEqual(parsed["name"], "Jane Doe")
        self.assertEqual(parsed["email"], "jane.doe@example.com")
        self.assertEqual(parsed["designation"], "Site Reliability Engineer")
        self.assertEqual(parsed["experience_level"], "Senior")
        self.assertIn("C++", parsed["skills"])
        self.assertIn("C#", parsed["skills"])
        self.assertIn("Terraform", parsed["skills"])
        self.assertIn("CI/CD", parsed["skills"])

    def test_document_parser_validations(self):
        """Test document parser validations for filenames, types, empty content, and size limits."""
        from app.services.document_parser import (
            DocumentParsingError,
            MAX_RESUME_SIZE_BYTES,
            UnsupportedFileTypeError,
            parse_resume_document,
        )

        # Empty or whitespace filename
        with self.assertRaises(UnsupportedFileTypeError):
            parse_resume_document("   ", b"data")

        # Unsupported extension
        with self.assertRaises(UnsupportedFileTypeError):
            parse_resume_document("resume.txt", b"data")

        # Empty file bytes
        with self.assertRaises(DocumentParsingError):
            parse_resume_document("resume.pdf", b"")

        # Oversized payload (> 10MB)
        with self.assertRaises(DocumentParsingError):
            parse_resume_document("resume.pdf", b"x" * (MAX_RESUME_SIZE_BYTES + 1))

        # Whitespace-padded extension supported -> hits PDF parser error on invalid bytes
        with self.assertRaises(DocumentParsingError):
            parse_resume_document("  resume.PDF  ", b"not-a-real-pdf")

    def test_matcher_skill_normalization_and_null_safety(self):
        """Test matcher skill alias normalization and null-safe lexical scoring."""
        from app.agents.matcher_agent import _calculate_lexical_score, _generate_heuristic_reason

        job = {
            "title": "Backend Engineer",
            "company": "Hidani",
            "skills": ["React", "Go", "PostgreSQL"],
            "experience": "Mid-Level",
            "description": None,
        }
        # Alias variations: React.js -> React, Golang -> Go, Postgres -> PostgreSQL
        score = _calculate_lexical_score(
            job,
            candidate_skills=["React.js", "Golang", "Postgres"],
            candidate_level="Mid-Level",
        )
        # Base 40 + 40 (full skill overlap) + 10 (experience) = 90.0
        self.assertEqual(score, 90.0)

        reason = _generate_heuristic_reason(
            job,
            candidate_skills=["React.js", "Golang"],
            candidate_designation="Full Stack Developer",
        )
        self.assertIn("Matches 2 of your core skills", reason)

    def test_update_application_status_endpoint(self):
        """Test PATCH /api/applications/{job_id} status transitions and validation."""
        withdraw_application(self.test_job_id)

        # Updating missing application yields 404
        res_404 = self.client.patch(f"/api/applications/{self.test_job_id}", json={"status": "Interview"})
        self.assertEqual(res_404.status_code, 404)

        # Apply to job first
        self.client.post("/api/applications", json={"job_id": self.test_job_id})

        # Valid status update (case-insensitive normalization)
        res_patch = self.client.patch(f"/api/applications/{self.test_job_id}", json={"status": "interview"})
        self.assertEqual(res_patch.status_code, 200)
        data = res_patch.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["application"]["status"], "Interview")
        self.assertIn("updated_date", data["application"])

        # Invalid status yields 400 Bad Request
        res_invalid = self.client.patch(f"/api/applications/{self.test_job_id}", json={"status": "UnknownStatus"})
        self.assertEqual(res_invalid.status_code, 400)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_applications_cache_consistency(self):
        """Test mtime cache synchronization and shallow-copy isolation for applications."""
        from app.services.db_service import invalidate_applications_cache, load_applications

        withdraw_application(self.test_job_id)
        invalidate_applications_cache()

        apps_initial = load_applications()
        self.assertFalse(any(a["job_id"] == self.test_job_id for a in apps_initial))

        # Apply updates cache immediately
        apply_to_job(self.test_job_id)
        apps_after_apply = load_applications()
        self.assertTrue(any(a["job_id"] == self.test_job_id for a in apps_after_apply))

        # Mutating returned dict does not pollute internal cache
        for a in apps_after_apply:
            if a["job_id"] == self.test_job_id:
                a["status"] = "MUTATED_LOCALLY"
        reloaded = get_application_status(self.test_job_id)
        self.assertEqual(reloaded[0]["status"], "Applied")

        # Withdraw updates cache immediately
        withdraw_application(self.test_job_id)
        apps_after_withdraw = load_applications()
        self.assertFalse(any(a["job_id"] == self.test_job_id for a in apps_after_withdraw))

    def test_db_atomic_write_and_null_safe_search(self):
        """Test atomic JSON writer fallback cleanup and null-safe search_jobs handling."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app.services.db_service import _atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "out.json"
            with patch("app.services.db_service.os.replace", side_effect=OSError("replace failed")):
                _atomic_write_json(target, [{"ok": True}])

            # Fallback direct write succeeded and no leftover temp files remain
            files = list(Path(tmpdir).iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "out.json")

        # Null-safe search_jobs with None fields in mocked job records
        mock_jobs = [
            {
                "job_id": "job-null-1",
                "title": "Python Engineer",
                "company": None,
                "location": None,
                "description": None,
                "experience": None,
                "skills": [None, "Python", " "],
            }
        ]
        with patch("app.services.db_service.load_jobs", return_value=mock_jobs):
            results = search_jobs(skills=[None, "Python", ""], query="Python", limit=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["job_id"], "job-null-1")

    def test_application_stats_endpoint(self):
        """Test GET /api/applications/stats summary breakdown."""
        withdraw_application(self.test_job_id)
        self.client.post("/api/applications", json={"job_id": self.test_job_id})
        self.client.patch(f"/api/applications/{self.test_job_id}", json={"status": "Offer"})

        res = self.client.get("/api/applications/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total_applications"], 1)
        self.assertIn("Offer", data["status_breakdown"])
        self.assertIsInstance(data["top_companies"], list)
        self.assertIsInstance(data["recent_applications"], list)
        self.assertTrue(any(a["job_id"] == self.test_job_id for a in data["recent_applications"]))

        withdraw_application(self.test_job_id)

    def test_jobs_company_filtering(self):
        """Test GET /api/jobs and search_jobs filtering by company name."""
        from urllib.parse import quote

        sample_job = get_job_by_id(self.test_job_id)
        company_name = sample_job["company"]

        res = self.client.get(f"/api/jobs?company={quote(company_name)}&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total"], 1)
        for job in data["jobs"]:
            self.assertIn(company_name.lower(), job["company"].lower())

        # Non-matching company returns 0
        res_empty = self.client.get("/api/jobs?company=CompanyThatDoesNotExistXYZ999")
        self.assertEqual(res_empty.status_code, 200)
        self.assertEqual(res_empty.json()["total"], 0)

    def test_orchestrator_status_update_intent_and_applier(self):
        """Test Applier Agent status update helper and Orchestrator natural status-update intent."""
        from app.agents.applier_agent import update_job_application_status
        from app.agents.orchestrator import _execute_tool_call

        withdraw_application(self.test_job_id)

        # Empty job_id validation
        res_empty = update_job_application_status("", "Interview")
        self.assertFalse(res_empty["success"])
        self.assertIsNone(res_empty["application"])

        # Apply then update via natural intent
        apply_to_job(self.test_job_id)
        reply, actions = _fallback_intent_orchestrator(
            f"mark {self.test_job_id} as interview",
            session_id="test-status-update-session",
        )
        self.assertIn("Application Status Updated", reply)
        self.assertTrue(any(a.type == "APPLICATION_UPDATED" for a in actions))
        self.assertEqual(get_application_status(self.test_job_id)[0]["status"], "Interview")

        # Update via tool call executor
        import json
        tool_out, tool_action = _execute_tool_call(
            "update_application_status",
            {"job_id": self.test_job_id, "status": "Offer"},
            profile=None,
            session_id="test-status-update-session",
        )
        self.assertEqual(tool_action.type, "APPLICATION_UPDATED")
        self.assertEqual(json.loads(tool_out)["application"]["status"], "Offer")

        withdraw_application(self.test_job_id)

    def test_jobs_sorting_and_ordering(self):
        """Test GET /api/jobs sort_by and order parameters."""
        res_asc = self.client.get("/api/jobs?sort_by=title&order=asc&limit=10")
        self.assertEqual(res_asc.status_code, 200)
        jobs_asc = res_asc.json()["jobs"]
        titles_asc = [j["title"].lower() for j in jobs_asc]
        self.assertEqual(titles_asc, sorted(titles_asc))

        res_desc = self.client.get("/api/jobs?sort_by=title&order=desc&limit=10")
        self.assertEqual(res_desc.status_code, 200)
        jobs_desc = res_desc.json()["jobs"]
        titles_desc = [j["title"].lower() for j in jobs_desc]
        self.assertEqual(titles_desc, sorted(titles_desc, reverse=True))

    def test_application_with_notes(self):
        """Test submitting an application with custom candidate notes/cover remarks."""
        withdraw_application(self.test_job_id)

        res = self.client.post("/api/applications", json={
            "job_id": self.test_job_id,
            "notes": "Excited about this opportunity; 4 years Python experience."
        })
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["application"]["notes"], "Excited about this opportunity; 4 years Python experience.")

        # Verify retrieval retains notes
        app_details = self.client.get(f"/api/applications/{self.test_job_id}").json()
        self.assertEqual(app_details["notes"], "Excited about this opportunity; 4 years Python experience.")

        withdraw_application(self.test_job_id)

    def test_application_export_json_and_csv(self):
        """Test GET /api/applications/export for JSON and CSV file downloads."""
        withdraw_application(self.test_job_id)
        self.client.post("/api/applications", json={
            "job_id": self.test_job_id,
            "notes": "Export test candidate"
        })

        # 1. JSON Export
        res_json = self.client.get("/api/applications/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("job_applications.json", res_json.headers.get("content-disposition", ""))
        self.assertIsInstance(res_json.json(), list)
        self.assertTrue(any(a["job_id"] == self.test_job_id for a in res_json.json()))

        # 2. CSV Export
        res_csv = self.client.get("/api/applications/export?format=csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("job_applications.csv", res_csv.headers.get("content-disposition", ""))
        self.assertIn("Job ID,Job Title,Company", res_csv.text)
        self.assertIn(self.test_job_id, res_csv.text)

        # 3. Invalid format yields 400
        res_bad = self.client.get("/api/applications/export?format=xml")
        self.assertEqual(res_bad.status_code, 400)

        withdraw_application(self.test_job_id)

    def test_job_bookmarks_lifecycle(self):
        """Test bookmarking a job, duplicate prevention, listing, and removal."""
        from app.services.db_service import remove_job_bookmark
        remove_job_bookmark(self.test_job_id)

        # 1. Bookmark valid job -> 201 Created
        res_post = self.client.post(f"/api/bookmarks/{self.test_job_id}")
        self.assertEqual(res_post.status_code, 201)
        data = res_post.json()
        self.assertEqual(data["bookmark"]["job_id"], self.test_job_id)
        self.assertIn("saved_date", data["bookmark"])

        # 2. Duplicate bookmark -> 409 Conflict
        res_dup = self.client.post(f"/api/bookmarks/{self.test_job_id}")
        self.assertEqual(res_dup.status_code, 409)

        # 3. Retrieve list
        res_list = self.client.get("/api/bookmarks")
        self.assertEqual(res_list.status_code, 200)
        bm_ids = [b["job_id"] for b in res_list.json()]
        self.assertIn(self.test_job_id, bm_ids)

        # 4. Remove bookmark -> 200 OK
        res_del = self.client.delete(f"/api/bookmarks/{self.test_job_id}")
        self.assertEqual(res_del.status_code, 200)

        # 5. Remove non-existent -> 404
        res_del_404 = self.client.delete(f"/api/bookmarks/{self.test_job_id}")
        self.assertEqual(res_del_404.status_code, 404)

        # 6. Bookmark non-existent job -> 404
        res_missing = self.client.post("/api/bookmarks/job-does-not-exist-xyz")
        self.assertEqual(res_missing.status_code, 404)

    def test_orchestrator_bookmarking_intent_and_tools(self):
        """Test Orchestrator tool calls and conversational intents for saving and listing bookmarks."""
        from app.agents.orchestrator import _execute_tool_call, _fallback_intent_orchestrator
        from app.services.db_service import remove_job_bookmark
        session_id = "test-orchestrator-bookmark-session"
        remove_job_bookmark(self.test_job_id)

        # 1. Tool execution: bookmark_job
        tool_res, action = _execute_tool_call("bookmark_job", {"job_id": self.test_job_id}, session_id=session_id)
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "BOOKMARK_SAVED")
        self.assertTrue(action.data.get("success"))

        # 2. Tool execution: get_bookmarked_jobs
        list_res, list_action = _execute_tool_call("get_bookmarked_jobs", {}, session_id=session_id)
        self.assertIsNotNone(list_action)
        self.assertEqual(list_action.type, "BOOKMARKS_LIST")
        self.assertGreaterEqual(list_action.data.get("total", 0), 1)

        # 3. Conversational intent: show saved jobs
        reply_list, actions_list = _fallback_intent_orchestrator("show my saved jobs", session_id=session_id)
        self.assertIn("Saved Bookmarks", reply_list)
        self.assertTrue(any(a.type == "BOOKMARKS_LIST" for a in actions_list))

        # 4. Conversational intent: remove bookmark
        reply_del, actions_del = _fallback_intent_orchestrator(f"remove bookmark {self.test_job_id}", session_id=session_id)
        self.assertIn("Bookmark Update", reply_del)
        self.assertTrue(any(a.type == "BOOKMARK_REMOVED" for a in actions_del))

        # 5. Conversational intent: bookmark job via recommendation ordinal
        _fallback_intent_orchestrator("recommend python jobs", session_id=session_id)
        reply_bm, actions_bm = _fallback_intent_orchestrator("bookmark job 1", session_id=session_id)
        self.assertIn("Job Bookmarked", reply_bm)
        self.assertTrue(any(a.type == "BOOKMARK_SAVED" for a in actions_bm))
        for a in actions_bm:
            if a.data and a.data.get("job_id"):
                remove_job_bookmark(a.data["job_id"])

        # Clean up
        remove_job_bookmark(self.test_job_id)

    def test_application_candidate_notes_lifecycle(self):
        """Test candidate notes update endpoint and helper."""
        from app.agents.applier_agent import update_job_application_notes
        withdraw_application(self.test_job_id)

        # 1. Update notes on non-existent application -> 404
        res_404 = self.client.patch(
            f"/api/applications/{self.test_job_id}/notes",
            json={"notes": "Interview scheduled"}
        )
        self.assertEqual(res_404.status_code, 404)

        # 2. Apply first, then update notes -> 200 OK
        apply_to_job(self.test_job_id, notes="Initial cover note")
        res_patch = self.client.patch(
            f"/api/applications/{self.test_job_id}/notes",
            json={"notes": "First round with lead recruiter on Thursday at 2 PM"}
        )
        self.assertEqual(res_patch.status_code, 200)
        data = res_patch.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["application"]["notes"], "First round with lead recruiter on Thursday at 2 PM")
        self.assertIn("updated_date", data["application"])

        # 3. Direct agent call
        agent_res = update_job_application_notes(self.test_job_id, "Completed technical assessment")
        self.assertTrue(agent_res["success"])
        self.assertEqual(agent_res["application"]["notes"], "Completed technical assessment")

        # 4. Clean up
        withdraw_application(self.test_job_id)

    def test_search_history_lifecycle(self):
        """Test search query recording, retrieval, and clearing."""
        from app.services.db_service import clear_search_history, get_recent_searches

        # 1. Clear search history
        res_del = self.client.delete("/api/searches/recent")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(len(get_recent_searches()), 0)

        # 2. Trigger search via /api/jobs
        res_search = self.client.get("/api/jobs?query=python&location=remote")
        self.assertEqual(res_search.status_code, 200)

        # 3. Retrieve recent searches endpoint
        res_history = self.client.get("/api/searches/recent")
        self.assertEqual(res_history.status_code, 200)
        entries = res_history.json()
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["query"], "python")
        self.assertEqual(entries[0]["location"], "remote")
        self.assertGreater(entries[0]["results_count"], 0)
        self.assertIn("timestamp", entries[0])

        # 4. Duplicate search does not duplicate consecutive entries
        self.client.get("/api/jobs?query=python&location=remote")
        entries_after = self.client.get("/api/searches/recent").json()
        self.assertEqual(len(entries_after), len(entries))

        # 5. Clean up
        clear_search_history()

    def test_job_skill_gap_analysis(self):
        """Test skill gap analysis agent helper and GET /api/jobs/{job_id}/match endpoint."""
        from app.agents.matcher_agent import analyze_skill_gap

        # 1. Non-existent job endpoint -> 404
        res_404 = self.client.get("/api/jobs/job-does-not-exist-xyz/match")
        self.assertEqual(res_404.status_code, 404)

        # 2. Existing job match endpoint with matching skills
        # self.test_job_id (job-00001) has skill "Finance"
        res_match = self.client.get(f"/api/jobs/{self.test_job_id}/match?skills=Finance,Excel,Python")
        self.assertEqual(res_match.status_code, 200)
        data = res_match.json()
        self.assertEqual(data["job_id"], self.test_job_id)
        self.assertIn("Finance", data["matching_skills"])
        self.assertGreater(data["match_score"], 0)
        self.assertIn(data["readiness"], ["High Readiness", "Moderate Readiness", "Needs Preparation"])
        self.assertIn("recommendation", data)

        # 3. Direct agent call
        analysis = analyze_skill_gap(self.test_job_id, candidate_skills=["Finance"])
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["job_id"], self.test_job_id)
        self.assertEqual(analysis["readiness"], "High Readiness")

        # 4. Direct agent call with empty or invalid id
        self.assertIsNone(analyze_skill_gap(""))
        self.assertIsNone(analyze_skill_gap("invalid-job-id-9999"))

    def test_application_followup_reminders(self):
        """Test setting, querying, and clearing application follow-up reminders."""
        from app.agents.applier_agent import get_upcoming_reminders_agent, set_application_reminder_agent
        withdraw_application(self.test_job_id)

        # 1. Update reminder on non-existent application -> 404
        res_404 = self.client.patch(
            f"/api/applications/{self.test_job_id}/reminder",
            json={"reminder_date": "2026-11-20"}
        )
        self.assertEqual(res_404.status_code, 404)

        # 2. Submit application
        apply_to_job(self.test_job_id)

        # 3. Set reminder -> 200 OK
        res_rem = self.client.patch(
            f"/api/applications/{self.test_job_id}/reminder",
            json={"reminder_date": "2026-11-20"}
        )
        self.assertEqual(res_rem.status_code, 200)
        self.assertEqual(res_rem.json()["application"]["reminder_date"], "2026-11-20")

        # 4. Query reminders list endpoint
        res_list = self.client.get("/api/applications/reminders")
        self.assertEqual(res_list.status_code, 200)
        reminders = res_list.json()
        self.assertTrue(any(r["job_id"] == self.test_job_id for r in reminders))

        # 5. Direct agent helpers
        agent_res = set_application_reminder_agent(self.test_job_id, "2026-11-25")
        self.assertTrue(agent_res["success"])
        self.assertEqual(agent_res["application"]["reminder_date"], "2026-11-25")

        summary = get_upcoming_reminders_agent()
        self.assertGreaterEqual(summary["total"], 1)

        # 6. Clear reminder
        res_clear = self.client.patch(
            f"/api/applications/{self.test_job_id}/reminder",
            json={"reminder_date": None}
        )
        self.assertEqual(res_clear.status_code, 200)
        self.assertIsNone(res_clear.json()["application"]["reminder_date"])

        # 7. Clean up
        withdraw_application(self.test_job_id)

    def test_orchestrator_skill_gap_and_reminders_intents(self):
        """Test Orchestrator tool routing and fallback intents for skill gap evaluation and follow-up reminders."""
        from app.agents.orchestrator import _execute_tool_call, _fallback_intent_orchestrator
        session_id = "test-orchestrator-advanced-suite"
        apply_to_job(self.test_job_id)

        # 1. Tool execution: analyze_skill_gap
        res_gap, action_gap = _execute_tool_call("analyze_skill_gap", {"job_id": self.test_job_id}, session_id=session_id)
        self.assertIsNotNone(action_gap)
        self.assertEqual(action_gap.type, "SKILL_GAP_ANALYSIS")
        self.assertEqual(action_gap.data["job_id"], self.test_job_id)

        # 2. Tool execution: set_followup_reminder
        res_rem, action_rem = _execute_tool_call("set_followup_reminder", {"job_id": self.test_job_id, "reminder_date": "2026-11-22"}, session_id=session_id)
        self.assertIsNotNone(action_rem)
        self.assertEqual(action_rem.type, "REMINDER_SET")
        self.assertTrue(action_rem.data["success"])

        # 3. Tool execution: get_upcoming_reminders
        res_list, action_list = _execute_tool_call("get_upcoming_reminders", {}, session_id=session_id)
        self.assertIsNotNone(action_list)
        self.assertEqual(action_list.type, "REMINDERS_LIST")
        self.assertGreaterEqual(action_list.data["total"], 1)

        # 4. Fallback intent: skills needed for job
        _fallback_intent_orchestrator("recommend python jobs", session_id=session_id)
        reply_gap, actions_gap = _fallback_intent_orchestrator("skills needed for job 1", session_id=session_id)
        self.assertIn("Skill Fit Assessment", reply_gap)
        self.assertTrue(any(a.type == "SKILL_GAP_ANALYSIS" for a in actions_gap))

        # 5. Fallback intent: set reminder
        reply_rem, actions_rem = _fallback_intent_orchestrator(f"remind me for {self.test_job_id} on 2026-11-30", session_id=session_id)
        self.assertIn("Follow-up Reminder Scheduled", reply_rem)
        self.assertTrue(any(a.type == "REMINDER_SET" for a in actions_rem))

        # 6. Fallback intent: show upcoming reminders
        reply_show, actions_show = _fallback_intent_orchestrator("show upcoming reminders", session_id=session_id)
        self.assertIn("Upcoming Reminders", reply_show)
        self.assertTrue(any(a.type == "REMINDERS_LIST" for a in actions_show))

        # Clean up
        withdraw_application(self.test_job_id)

    def test_job_category_classification_and_filtering(self):
        """Test industry domain classification, stats breakdown, and category filter."""
        from app.services.db_service import classify_job_category

        # 1. Classification helper verification
        self.assertEqual(classify_job_category("Senior Backend Engineer"), "Engineering")
        self.assertEqual(classify_job_category("Working Student - LLM Quantization & Inference"), "Data & AI")
        self.assertEqual(classify_job_category("Head of Product"), "Product & Design")
        self.assertEqual(classify_job_category("Finance Intern"), "Finance & Operations")
        self.assertEqual(classify_job_category("SEM Manager"), "Marketing & Sales")

        # 2. Filter endpoint /api/jobs?category=Engineering
        res_eng = self.client.get("/api/jobs?category=Engineering&limit=10")
        self.assertEqual(res_eng.status_code, 200)
        jobs_eng = res_eng.json()["jobs"]
        self.assertGreater(len(jobs_eng), 0)
        for j in jobs_eng:
            self.assertEqual(j["category"], "Engineering")

        # 3. Market stats category breakdown
        res_stats = self.client.get("/api/jobs/stats")
        self.assertEqual(res_stats.status_code, 200)
        data = res_stats.json()
        self.assertIn("category_breakdown", data)
        cats = data["category_breakdown"]
        self.assertIn("Engineering", cats)
        self.assertGreater(cats["Engineering"], 0)

    def test_bulk_application_status_update(self):
        """Test bulk status update endpoint and validation."""
        other_id = "job-00002"
        withdraw_application(self.test_job_id)
        withdraw_application(other_id)

        # 1. Apply to two jobs
        apply_to_job(self.test_job_id)
        apply_to_job(other_id)

        # 2. Bulk update both to "Interview"
        res_bulk = self.client.patch(
            "/api/applications/bulk/status",
            json={"job_ids": [self.test_job_id, other_id], "status": "Interview"}
        )
        self.assertEqual(res_bulk.status_code, 200)
        data = res_bulk.json()
        self.assertEqual(data["updated_count"], 2)
        self.assertIn(self.test_job_id, data["updated_job_ids"])
        self.assertIn(other_id, data["updated_job_ids"])

        # 3. Verify status in database
        app1 = self.client.get(f"/api/applications/{self.test_job_id}").json()
        app2 = self.client.get(f"/api/applications/{other_id}").json()
        self.assertEqual(app1["status"], "Interview")
        self.assertEqual(app2["status"], "Interview")

        # 4. Invalid status -> 400
        res_bad = self.client.patch(
            "/api/applications/bulk/status",
            json={"job_ids": [self.test_job_id], "status": "NotAValidStatus123"}
        )
        self.assertEqual(res_bad.status_code, 400)

        # 5. Non-existent jobs -> 404
        res_404 = self.client.patch(
            "/api/applications/bulk/status",
            json={"job_ids": ["non-existent-job-abc-xyz"], "status": "Offer"}
        )
        self.assertEqual(res_404.status_code, 404)

        # Clean up
        withdraw_application(self.test_job_id)
        withdraw_application(other_id)

    def test_application_timeline_audit_log(self):
        """Test application event audit log and timeline tracking across its lifecycle."""
        withdraw_application(self.test_job_id)

        # 1. Non-existent application -> 404
        res_404 = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_404.status_code, 404)

        # 2. Submit application -> initial SUBMITTED event
        apply_to_job(self.test_job_id, notes="Interested in this role")

        # 3. Update status -> STATUS_CHANGED event
        self.client.patch(
            f"/api/applications/{self.test_job_id}",
            json={"status": "Under Review"}
        )

        # 4. Update notes -> NOTES_UPDATED event
        self.client.patch(
            f"/api/applications/{self.test_job_id}/notes",
            json={"notes": "Recruiter phone screen passed"}
        )

        # 5. Set reminder -> REMINDER_UPDATED event
        self.client.patch(
            f"/api/applications/{self.test_job_id}/reminder",
            json={"reminder_date": "2026-11-28"}
        )

        # 6. Retrieve timeline endpoint
        res_timeline = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_timeline.status_code, 200)
        data = res_timeline.json()
        self.assertEqual(data["job_id"], self.test_job_id)
        self.assertEqual(data["total_events"], 4)
        events = data["timeline"]
        self.assertEqual(events[0]["event_type"], "SUBMITTED")
        self.assertEqual(events[1]["event_type"], "STATUS_CHANGED")
        self.assertEqual(events[2]["event_type"], "NOTES_UPDATED")
        self.assertEqual(events[3]["event_type"], "REMINDER_UPDATED")
        self.assertIn("2026-11-28", events[3]["details"])

        # Clean up
        withdraw_application(self.test_job_id)

    def test_saved_searches_lifecycle(self):
        """Test creating, listing, and deleting saved search subscriptions."""
        # 1. Create saved search -> 201 Created
        payload = {
            "title": "Remote Python Roles",
            "query": "python",
            "location": "remote",
            "experience_level": "Mid-Level",
            "category": "Engineering"
        }
        res_create = self.client.post("/api/searches/saved", json=payload)
        self.assertEqual(res_create.status_code, 201)
        created = res_create.json()
        search_id = created["search_id"]
        self.assertEqual(created["title"], "Remote Python Roles")
        self.assertEqual(created["query"], "python")
        self.assertIn("created_date", created)

        # 2. List saved searches -> includes created
        res_list = self.client.get("/api/searches/saved")
        self.assertEqual(res_list.status_code, 200)
        items = res_list.json()
        self.assertTrue(any(s["search_id"] == search_id for s in items))

        # 3. Delete saved search -> 200 OK
        res_del = self.client.delete(f"/api/searches/saved/{search_id}")
        self.assertEqual(res_del.status_code, 200)

        # 4. Delete non-existent search -> 404
        res_del_404 = self.client.delete(f"/api/searches/saved/{search_id}")
        self.assertEqual(res_del_404.status_code, 404)

    def test_saved_search_update_and_deduplication(self):
        """Test duplicate detection and updating filter criteria on saved searches."""
        # 1. Create original saved search
        payload = {
            "title": "Unique AI Researcher Alert",
            "query": "deep learning",
            "location": "remote",
            "experience_level": "Mid-Level",
            "category": "Data & Analytics"
        }
        res_create = self.client.post("/api/searches/saved", json=payload)
        self.assertEqual(res_create.status_code, 201)
        search_id = res_create.json()["search_id"]

        try:
            # 2. Duplicate title check -> 409 Conflict
            res_dup_title = self.client.post("/api/searches/saved", json={
                "title": "Unique AI Researcher Alert",
                "query": "natural language processing"
            })
            self.assertEqual(res_dup_title.status_code, 409)

            # 3. Duplicate criteria check -> 409 Conflict
            res_dup_crit = self.client.post("/api/searches/saved", json={
                "title": "Another Alert Name",
                "query": "deep learning",
                "location": "remote",
                "experience_level": "Mid-Level",
                "category": "Data & Analytics"
            })
            self.assertEqual(res_dup_crit.status_code, 409)

            # 4. Update saved search criteria -> 200 OK
            update_payload = {
                "title": "Updated Senior AI Alert",
                "experience_level": "Senior",
                "category": "AI Research"
            }
            res_update = self.client.patch(f"/api/searches/saved/{search_id}", json=update_payload)
            self.assertEqual(res_update.status_code, 200)
            updated_data = res_update.json()
            self.assertEqual(updated_data["title"], "Updated Senior AI Alert")
            self.assertEqual(updated_data["experience_level"], "Senior")
            self.assertEqual(updated_data["category"], "AI Research")
            self.assertEqual(updated_data["query"], "deep learning")  # Unchanged
            self.assertIsNotNone(updated_data.get("updated_date"))

            # 5. Non-existent search update -> 404
            res_404 = self.client.patch("/api/searches/saved/search_nonexistent_999", json={"title": "Test"})
            self.assertEqual(res_404.status_code, 404)
        finally:
            self.client.delete(f"/api/searches/saved/{search_id}")

    def test_application_custom_tags_lifecycle(self):
        """Test adding, deduplicating, filtering, and clearing custom organization tags on job applications."""
        withdraw_application(self.test_job_id)
        apply_to_job(self.test_job_id)

        # 1. Update tags with case duplicate
        payload = {"tags": ["High Priority", "Referral", "high priority", "Tier 1"]}
        res = self.client.put(f"/api/applications/{self.test_job_id}/tags", json=payload)
        self.assertEqual(res.status_code, 200)
        app_data = res.json()["application"]
        self.assertEqual(app_data["tags"], ["High Priority", "Referral", "Tier 1"])

        # 2. Filter applications by tag (case-insensitive)
        res_tag = self.client.get("/api/applications?tag=referral")
        self.assertEqual(res_tag.status_code, 200)
        matched = [a for a in res_tag.json() if a["job_id"] == self.test_job_id]
        self.assertEqual(len(matched), 1)

        # 3. Filter with non-matching tag
        res_none = self.client.get("/api/applications?tag=nonexistent-tag")
        self.assertEqual(res_none.status_code, 200)
        matched_none = [a for a in res_none.json() if a["job_id"] == self.test_job_id]
        self.assertEqual(len(matched_none), 0)

        # 4. Timeline records TAGS_UPDATED event
        res_timeline = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_timeline.status_code, 200)
        events = res_timeline.json()["timeline"]
        self.assertTrue(any(e["event_type"] == "TAGS_UPDATED" for e in events))

        # 5. Invalid job ID -> 404
        res_404 = self.client.put("/api/applications/non-existent-job-xyz/tags", json={"tags": ["Test"]})
        self.assertEqual(res_404.status_code, 404)

        # 6. Clear tags
        res_clear = self.client.put(f"/api/applications/{self.test_job_id}/tags", json={"tags": []})
        self.assertEqual(res_clear.status_code, 200)
        self.assertEqual(res_clear.json()["application"]["tags"], [])

        # Clean up
        withdraw_application(self.test_job_id)

    def test_job_salary_extraction_and_filtering(self):
        """Test regex extraction of salary details and API filtering by salary disclosures."""
        from unittest.mock import patch
        from app.services.db_service import extract_salary_info, search_jobs

        # 1. Regex unit extraction tests
        sal1 = extract_salary_info("We offer $120,000 - $160,000 / year plus equity")
        self.assertIsNotNone(sal1)
        self.assertEqual(sal1["currency"], "USD")
        self.assertEqual(sal1["min_salary"], 120000.0)
        self.assertEqual(sal1["max_salary"], 160000.0)
        self.assertEqual(sal1["period"], "year")

        sal2 = extract_salary_info("Competitive pay: €75k - €95k annually")
        self.assertIsNotNone(sal2)
        self.assertEqual(sal2["currency"], "EUR")
        self.assertEqual(sal2["min_salary"], 75000.0)
        self.assertEqual(sal2["max_salary"], 95000.0)

        sal3 = extract_salary_info("Contract rate: £50 - £75 / hour")
        self.assertIsNotNone(sal3)
        self.assertEqual(sal3["currency"], "GBP")
        self.assertEqual(sal3["period"], "hour")

        sal_none = extract_salary_info("Great company culture with 2020-2024 experience.")
        self.assertIsNone(sal_none)

        # 2. Filtering and sorting with mock data
        mock_jobs = [
            {
                "job_id": "sal-job-1",
                "title": "Senior Backend Dev",
                "company": "Alpha Corp",
                "location": "Remote",
                "description": "Salary: $150,000 / year",
                "salary": {"currency": "USD", "min_salary": 150000.0, "max_salary": 150000.0, "period": "year"}
            },
            {
                "job_id": "sal-job-2",
                "title": "Junior Python Dev",
                "company": "Beta Inc",
                "location": "Remote",
                "description": "Salary: $70,000 / year",
                "salary": {"currency": "USD", "min_salary": 70000.0, "max_salary": 70000.0, "period": "year"}
            },
            {
                "job_id": "sal-job-3",
                "title": "Intern Dev",
                "company": "Gamma LLC",
                "location": "Remote",
                "description": "No salary listed",
                "salary": None
            }
        ]

        with patch("app.services.db_service.load_jobs", return_value=mock_jobs):
            # has_salary=True
            with_sal = search_jobs(has_salary=True, limit=10)
            self.assertEqual(len(with_sal), 2)
            self.assertTrue(all(j["salary"] is not None for j in with_sal))

            # has_salary=False
            without_sal = search_jobs(has_salary=False, limit=10)
            self.assertEqual(len(without_sal), 1)
            self.assertEqual(without_sal[0]["job_id"], "sal-job-3")

            # min_salary=100000
            high_sal = search_jobs(min_salary=100000, limit=10)
            self.assertEqual(len(high_sal), 1)
            self.assertEqual(high_sal[0]["job_id"], "sal-job-1")

            # sort_by="salary" desc
            sorted_desc = search_jobs(sort_by="salary", order="desc", limit=10)
            self.assertEqual(sorted_desc[0]["job_id"], "sal-job-1")
            self.assertEqual(sorted_desc[1]["job_id"], "sal-job-2")

        # 3. API endpoint integration
        res = self.client.get("/api/jobs?limit=5")
        self.assertEqual(res.status_code, 200)
        jobs_data = res.json()["jobs"]
        self.assertTrue(len(jobs_data) > 0)
        # Verify schema serialization has salary key
        self.assertIn("salary", jobs_data[0])

    def test_database_backup_and_restore_lifecycle(self):
        """Test creating database backup snapshot, modifying collections, and restoring previous state."""
        # 1. Generate full backup snapshot
        res_backup = self.client.get("/api/admin/backup")
        self.assertEqual(res_backup.status_code, 200)
        snapshot = res_backup.json()
        self.assertEqual(snapshot["version"], "1.0")
        self.assertIn("timestamp", snapshot)
        self.assertIn("applications", snapshot)
        self.assertIn("bookmarks", snapshot)
        self.assertIn("saved_searches", snapshot)
        self.assertIn("search_history", snapshot)

        # 2. Modify database collections
        self.client.post(f"/api/bookmarks/{self.test_job_id}")
        res_search = self.client.post("/api/searches/saved", json={"title": "Snapshot Test Query"})
        self.assertEqual(res_search.status_code, 201)
        created_search_id = res_search.json()["search_id"]

        # Verify items exist
        bms_before = self.client.get("/api/bookmarks").json()
        self.assertTrue(any(b["job_id"] == self.test_job_id for b in bms_before))
        searches_before = self.client.get("/api/searches/saved").json()
        self.assertTrue(any(s["search_id"] == created_search_id for s in searches_before))

        # 3. Restore from backup snapshot
        res_restore = self.client.post("/api/admin/restore", json=snapshot)
        self.assertEqual(res_restore.status_code, 200)
        restore_data = res_restore.json()
        self.assertEqual(restore_data["status"], "success")
        self.assertIn("restored_counts", restore_data)

        # 4. Verify test items are restored away
        bms_after = self.client.get("/api/bookmarks").json()
        self.assertEqual(len(bms_after), len(snapshot["bookmarks"]))
        searches_after = self.client.get("/api/searches/saved").json()
        self.assertEqual(len(searches_after), len(snapshot["saved_searches"]))

        # 5. Invalid restore payload -> 422 Unprocessable Entity
        res_invalid = self.client.post("/api/admin/restore", json={"bad_field": 123})
        self.assertEqual(res_invalid.status_code, 422)

    def test_database_health_diagnostics(self):
        """Test database integrity check and health diagnostics API."""
        res = self.client.get("/api/admin/db/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertEqual(data["total_collections"], 5)
        self.assertTrue(data["healthy_collections"] >= 1)
        names = [c["collection"] for c in data["collections"]]
        self.assertIn("jobs", names)
        self.assertIn("applications", names)
        self.assertIn("bookmarks", names)
        self.assertIn("saved_searches", names)
        self.assertIn("search_history", names)

    def test_job_freshness_classification_and_filtering(self):
        """Test computing job freshness recency tiers and filtering via search API."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch
        from app.services.db_service import compute_job_freshness, search_jobs

        now = datetime.now(timezone.utc)
        recent_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        older_date = (now - timedelta(days=20)).strftime("%Y-%m-%d")
        archived_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")

        self.assertEqual(compute_job_freshness({"posted_date": recent_date}), "New")
        self.assertEqual(compute_job_freshness({"posted_date": older_date}), "Recent")
        self.assertEqual(compute_job_freshness({"posted_date": archived_date}), "Archive")

        mock_jobs = [
            {"job_id": "j-new", "title": "New Engineer", "company": "Co", "location": "Remote", "freshness": "New"},
            {"job_id": "j-old", "title": "Old Engineer", "company": "Co", "location": "Remote", "freshness": "Archive"}
        ]
        with patch("app.services.db_service.load_jobs", return_value=mock_jobs):
            new_jobs = search_jobs(freshness="new")
            self.assertEqual(len(new_jobs), 1)
            self.assertEqual(new_jobs[0]["job_id"], "j-new")

        # Integration with GET /api/jobs
        res = self.client.get("/api/jobs?limit=5")
        self.assertEqual(res.status_code, 200)
        jobs = res.json()["jobs"]
        if jobs:
            self.assertIn(jobs[0].get("freshness"), ["New", "Recent", "Archive"])

    def test_application_interview_stages_lifecycle(self):
        """Test scheduling interview rounds, retrieving stages, and timeline event recording."""
        withdraw_application(self.test_job_id)
        apply_to_job(self.test_job_id)

        # 1. Schedule technical interview stage
        payload = {
            "round_name": "Technical System Design",
            "scheduled_at": "2026-11-25 14:00:00 UTC",
            "interviewer": "Jane Doe, Staff Eng",
            "meeting_link": "https://meet.google.com/xyz-abc",
            "notes": "Review distributed caching and microservices architecture",
            "status": "Scheduled"
        }
        res = self.client.post(f"/api/applications/{self.test_job_id}/interviews", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "success")
        stage = data["interview"]
        self.assertEqual(stage["round_name"], "Technical System Design")
        self.assertEqual(stage["interviewer"], "Jane Doe, Staff Eng")
        self.assertIn("stage_id", stage)

        # 2. Verify interview appears on application record
        app_res = self.client.get(f"/api/applications/{self.test_job_id}")
        self.assertEqual(app_res.status_code, 200)
        interviews = app_res.json().get("interviews", [])
        self.assertEqual(len(interviews), 1)
        self.assertEqual(interviews[0]["round_name"], "Technical System Design")

        # 3. Verify timeline event logged
        timeline_res = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(timeline_res.status_code, 200)
        events = timeline_res.json()["timeline"]
        self.assertTrue(any(e["event_type"] == "INTERVIEW_SCHEDULED" for e in events))

        # 4. 404 for invalid job_id
        res_404 = self.client.post("/api/applications/invalid-job-id-999/interviews", json=payload)
        self.assertEqual(res_404.status_code, 404)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_bookmark_notes_and_category_update(self):
        """Test adding notes and folder categorization to saved bookmarks."""
        # Ensure bookmark exists
        self.client.delete(f"/api/bookmarks/{self.test_job_id}")
        self.client.post(f"/api/bookmarks/{self.test_job_id}")

        # Update bookmark with notes and category
        payload = {
            "notes": "Spoke with hiring manager on LinkedIn; great culture fit.",
            "category": "Priority Outreach"
        }
        res = self.client.patch(f"/api/bookmarks/{self.test_job_id}", json=payload)
        self.assertEqual(res.status_code, 200)
        bm_data = res.json()["bookmark"]
        self.assertEqual(bm_data["notes"], payload["notes"])
        self.assertEqual(bm_data["category"], "Priority Outreach")

        # 404 for non-existent bookmark
        res_404 = self.client.patch("/api/bookmarks/non-existent-bookmark-123", json=payload)
        self.assertEqual(res_404.status_code, 404)

        # Clean up
        self.client.delete(f"/api/bookmarks/{self.test_job_id}")

    def test_application_funnel_metrics(self):
        """Test application hiring pipeline conversion funnel calculations."""
        res = self.client.get("/api/applications/analytics/funnel")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_applications", data)
        self.assertIn("stages", data)
        self.assertIn("drop_off_rate_percent", data)
        stage_names = [s["stage"] for s in data["stages"]]
        self.assertIn("Applied", stage_names)
        self.assertIn("Under Review", stage_names)
        self.assertIn("Interview", stage_names)

    def test_similar_jobs_recommendation(self):
        """Test recommending similar jobs based on skills overlap and metadata matching."""
        res = self.client.get(f"/api/jobs/{self.test_job_id}/similar?limit=3")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["job_id"], self.test_job_id)
        self.assertIn("similar_jobs", data)
        self.assertTrue(len(data["similar_jobs"]) <= 3)
        if data["similar_jobs"]:
            first_sim = data["similar_jobs"][0]
            self.assertIn("similarity_score", first_sim)
            self.assertNotEqual(first_sim["job_id"], self.test_job_id)

        # 404 for invalid job_id
        res_404 = self.client.get("/api/jobs/invalid-job-id-99999/similar")
        self.assertEqual(res_404.status_code, 404)

    def test_application_archive_and_filtering(self):
        """Test archiving an application, exclusion from active list, and restoration."""
        withdraw_application(self.test_job_id)
        apply_to_job(self.test_job_id)

        # 1. Archive application
        res_arch = self.client.patch(f"/api/applications/{self.test_job_id}/archive", json={"archived": True})
        self.assertEqual(res_arch.status_code, 200)
        app_data = res_arch.json()["application"]
        self.assertTrue(app_data["is_archived"])
        self.assertIsNotNone(app_data["archived_date"])

        # 2. Excluded by default from active list
        active_res = self.client.get("/api/applications")
        self.assertEqual(active_res.status_code, 200)
        active_ids = [a["job_id"] for a in active_res.json()]
        self.assertNotIn(self.test_job_id, active_ids)

        # 3. Included when include_archived=true
        all_res = self.client.get("/api/applications?include_archived=true")
        self.assertEqual(all_res.status_code, 200)
        all_ids = [a["job_id"] for a in all_res.json()]
        self.assertIn(self.test_job_id, all_ids)

        # 4. Unarchive
        res_unarch = self.client.patch(f"/api/applications/{self.test_job_id}/archive", json={"archived": False})
        self.assertEqual(res_unarch.status_code, 200)
        self.assertFalse(res_unarch.json()["application"]["is_archived"])

        # Clean up
        withdraw_application(self.test_job_id)

    def test_search_query_autocomplete_suggestions(self):
        """Test typeahead search query suggestions across titles, skills, and companies."""
        res = self.client.get("/api/jobs/suggestions?prefix=eng&limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["prefix"], "eng")
        self.assertIn("suggestions", data)
        self.assertTrue(len(data["suggestions"]) <= 5)
        for s in data["suggestions"]:
            self.assertIn(s["type"], ["title", "skill", "company"])
            self.assertTrue("eng" in s["text"].lower())

        # Empty prefix returns 422
        res_empty = self.client.get("/api/jobs/suggestions?prefix=")
        self.assertEqual(res_empty.status_code, 422)

    def test_database_compaction_and_orphan_cleanup(self):
        """Test database compaction endpoint and orphan pruning."""
        res = self.client.post("/api/admin/db/compact")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("removed_orphan_bookmarks", data)
        self.assertIn("deduplicated_history", data)
        self.assertIn("total_compacted_files", data)

    def test_application_attachments_lifecycle(self):
        """Test attaching document metadata to application and timeline logging."""
        withdraw_application(self.test_job_id)
        apply_to_job(self.test_job_id)

        # 1. Attach customized resume
        payload = {
            "file_name": "Tailored_Backend_Resume_V2.pdf",
            "file_type": "pdf",
            "file_size_bytes": 1048576,
            "notes": "Customized for microservices emphasis"
        }
        res = self.client.post(f"/api/applications/{self.test_job_id}/attachments", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "success")
        att = data["attachment"]
        self.assertEqual(att["file_name"], payload["file_name"])
        self.assertIn("attachment_id", att)

        # 2. Verify attachment appears in application model
        app_res = self.client.get(f"/api/applications/{self.test_job_id}")
        self.assertEqual(app_res.status_code, 200)
        attachments = app_res.json().get("attachments", [])
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["file_name"], payload["file_name"])

        # 3. Timeline event logged
        timeline_res = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(timeline_res.status_code, 200)
        events = timeline_res.json()["timeline"]
        self.assertTrue(any(e["event_type"] == "ATTACHMENT_ADDED" for e in events))

        # 4. Clean up
        withdraw_application(self.test_job_id)

    def test_job_work_mode_classification_and_filtering(self):
        """Test work mode arrangement classification (Remote, Hybrid, Onsite) and API filtering."""
        from unittest.mock import patch
        from app.services.db_service import classify_work_mode, search_jobs

        self.assertEqual(classify_work_mode(location="Remote - Anywhere"), "Remote")
        self.assertEqual(classify_work_mode(location="Berlin, Germany (Hybrid)"), "Hybrid")
        self.assertEqual(classify_work_mode(location="Munich, Germany"), "Onsite")

        mock_jobs = [
            {"job_id": "mode-1", "title": "Dev", "company": "Co", "location": "Remote", "work_mode": "Remote"},
            {"job_id": "mode-2", "title": "Dev", "company": "Co", "location": "Berlin", "work_mode": "Hybrid"},
            {"job_id": "mode-3", "title": "Dev", "company": "Co", "location": "Munich", "work_mode": "Onsite"}
        ]
        with patch("app.services.db_service.load_jobs", return_value=mock_jobs):
            res_remote = search_jobs(work_mode="remote")
            self.assertEqual(len(res_remote), 1)
            self.assertEqual(res_remote[0]["job_id"], "mode-1")

            res_hybrid = search_jobs(work_mode="hybrid")
            self.assertEqual(len(res_hybrid), 1)
            self.assertEqual(res_hybrid[0]["job_id"], "mode-2")

        # Integration test
        res = self.client.get("/api/jobs?work_mode=remote&limit=5")
        self.assertEqual(res.status_code, 200)
        jobs = res.json()["jobs"]
        for j in jobs:
            self.assertEqual(j.get("work_mode"), "Remote")

    def test_skill_salary_analytics(self):
        """Test skill demand and salary correlation statistics calculation."""
        res = self.client.get("/api/jobs/analytics/skill-salaries?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_evaluated_skills", data)
        self.assertIn("insights", data)
        self.assertTrue(len(data["insights"]) <= 5)
        for item in data["insights"]:
            self.assertIn("skill", item)
            self.assertIn("avg_min_salary", item)
            self.assertIn("avg_max_salary", item)
            self.assertTrue(item["avg_max_salary"] >= item["avg_min_salary"])


    def test_bulk_remove_bookmarks(self):
        """Test bulk removal of bookmarks with single batch operation."""
        # Ensure test bookmark exists
        self.client.post(f"/api/bookmarks/{self.test_job_id}")

        res = self.client.post("/api/bookmarks/bulk/delete", json={"job_ids": [self.test_job_id, "non-existent-id"]})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["deleted_count"] >= 1)

        # Empty job_ids -> 422
        res_empty = self.client.post("/api/bookmarks/bulk/delete", json={"job_ids": []})
        self.assertEqual(res_empty.status_code, 422)

    def test_application_outcome_tracking(self):
        """Test recording interview outcome and rejection reasons with audit logging."""
        withdraw_application(self.test_job_id)
        apply_to_job(self.test_job_id)

        payload = {
            "outcome": "Offer",
            "reason_code": "Selected Final Candidate",
            "feedback_notes": "Passed system design and cultural interviews with strong positive signals."
        }
        res = self.client.post(f"/api/applications/{self.test_job_id}/outcome", json=payload)
        self.assertEqual(res.status_code, 200)
        app_data = res.json()["application"]
        self.assertEqual(app_data["outcome"], "Offer")
        self.assertEqual(app_data["status"], "Offer")
        self.assertEqual(app_data["rejection_reason"], "Selected Final Candidate")

        # Timeline event recorded
        timeline_res = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(timeline_res.status_code, 200)
        events = timeline_res.json()["timeline"]
        self.assertTrue(any(e["event_type"] == "OUTCOME_RECORDED" for e in events))

        # Clean up
        withdraw_application(self.test_job_id)

    def test_execute_saved_search(self):
        """Test running a saved search subscription query and updating match metadata."""
        # 1. Create saved search
        payload = {
            "title": "Python Engineering Remote",
            "query": "python",
            "location": "remote",
            "category": "Engineering"
        }
        res_create = self.client.post("/api/searches/saved", json=payload)
        self.assertEqual(res_create.status_code, 201)
        search_id = res_create.json()["search_id"]

        # 2. Run saved search
        res_run = self.client.post(f"/api/searches/saved/{search_id}/run?limit=5")
        self.assertEqual(res_run.status_code, 200)
        data = res_run.json()
        self.assertEqual(data["search_id"], search_id)
        self.assertIn("total_matches", data)
        self.assertIn("executed_at", data)
        self.assertIn("jobs", data)

        # 3. 404 for invalid search_id
        res_404 = self.client.post("/api/searches/saved/invalid-search-9999/run")
        self.assertEqual(res_404.status_code, 404)

        # Clean up
        self.client.delete(f"/api/searches/saved/{search_id}")

    def test_company_directory_and_profile(self):
        """Test listing hiring companies and retrieving detailed company profile."""
        # 1. Directory
        res_dir = self.client.get("/api/companies?limit=10")
        self.assertEqual(res_dir.status_code, 200)
        data = res_dir.json()
        self.assertIn("total_companies", data)
        self.assertIn("companies", data)
        self.assertGreater(data["total_companies"], 0)
        top_company = data["companies"][0]
        self.assertIn("company", top_company)
        self.assertGreater(top_company["active_jobs_count"], 0)
        self.assertIsInstance(top_company["locations"], list)
        self.assertIsInstance(top_company["top_skills"], list)

        # 2. Specific company profile
        target_company = top_company["company"]
        res_prof = self.client.get(f"/api/companies/{target_company}")
        self.assertEqual(res_prof.status_code, 200)
        prof = res_prof.json()
        self.assertEqual(prof["company"], target_company)
        self.assertEqual(prof["active_jobs_count"], top_company["active_jobs_count"])
        self.assertIsInstance(prof["sample_titles"], list)

        # 3. Non-existent company 404
        res_404 = self.client.get("/api/companies/NonExistentFictionalCo999")
        self.assertEqual(res_404.status_code, 404)

    def test_application_followup_template_generation(self):
        """Test generating customized follow-up reminder email templates."""
        # Ensure application exists
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Test notes"})

        # 1. Status inquiry template (default)
        res_default = self.client.get(f"/api/applications/{self.test_job_id}/followup-template")
        self.assertEqual(res_default.status_code, 200)
        data = res_default.json()
        self.assertEqual(data["job_id"], self.test_job_id)
        self.assertEqual(data["template_type"], "status_inquiry")
        self.assertIn("subject", data)
        self.assertIn("body", data)
        self.assertIn("status of my application", data["body"].lower())

        # 2. Interview follow-up template
        res_iv = self.client.get(f"/api/applications/{self.test_job_id}/followup-template?template_type=interview_followup")
        self.assertEqual(res_iv.status_code, 200)
        self.assertEqual(res_iv.json()["template_type"], "interview_followup")
        self.assertIn("interview", res_iv.json()["subject"].lower())

        # 3. Thank you template
        res_ty = self.client.get(f"/api/applications/{self.test_job_id}/followup-template?template_type=thank_you")
        self.assertEqual(res_ty.status_code, 200)
        self.assertEqual(res_ty.json()["template_type"], "thank_you")
        self.assertIn("thank you", res_ty.json()["subject"].lower())

        # 4. 404 for unapplied job
        res_404 = self.client.get("/api/applications/non-existent-9999/followup-template")
        self.assertEqual(res_404.status_code, 404)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_application_velocity_metrics(self):
        """Test application submission velocity tracking and metrics."""
        # Ensure at least one application submitted
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Velocity test"})

        res = self.client.get("/api/applications/analytics/velocity")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_applied", data)
        self.assertIn("last_7_days_count", data)
        self.assertIn("last_30_days_count", data)
        self.assertIn("avg_per_week", data)
        self.assertIn("active_pipeline_count", data)
        self.assertGreaterEqual(data["total_applied"], 1)
        self.assertGreaterEqual(data["last_7_days_count"], 1)
        self.assertGreaterEqual(data["last_30_days_count"], 1)
        self.assertGreaterEqual(data["active_pipeline_count"], 1)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_experience_tier_normalization_and_filtering(self):
        """Test standardized career tier classification and tier filtering."""
        # 1. Fetch senior tier jobs
        res_sr = self.client.get("/api/jobs?tier=Senior&limit=5")
        self.assertEqual(res_sr.status_code, 200)
        sr_jobs = res_sr.json()["jobs"]
        self.assertGreater(len(sr_jobs), 0)
        for j in sr_jobs:
            self.assertEqual(j["normalized_tier"], "Senior")

        # 2. Fetch entry-level tier jobs
        res_entry = self.client.get("/api/jobs?tier=Entry-Level&limit=5")
        self.assertEqual(res_entry.status_code, 200)
        entry_jobs = res_entry.json()["jobs"]
        for j in entry_jobs:
            self.assertEqual(j["normalized_tier"], "Entry-Level")

        # 3. Fetch lead tier jobs
        res_lead = self.client.get("/api/jobs?tier=Lead&limit=5")
        self.assertEqual(res_lead.status_code, 200)
        lead_jobs = res_lead.json()["jobs"]
        for j in lead_jobs:
            self.assertEqual(j["normalized_tier"], "Lead")

    def test_database_audit_log_and_transaction_tracking(self):
        """Test recording and retrieving transaction audit trail log."""
        # 1. Trigger an audited action (apply to job)
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Audit test"})

        # 2. Query audit log
        res = self.client.get("/api/admin/db/audit-log?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_entries", data)
        self.assertIn("entries", data)
        self.assertGreater(data["total_entries"], 0)
        recent_entry = data["entries"][0]
        self.assertIn("entry_id", recent_entry)
        self.assertIn("action", recent_entry)
        self.assertIn("entity", recent_entry)
        self.assertIn("timestamp", recent_entry)

        # 3. Filter by entity
        res_filter = self.client.get("/api/admin/db/audit-log?entity=application")
        self.assertEqual(res_filter.status_code, 200)
        for entry in res_filter.json()["entries"]:
            self.assertEqual(entry["entity"], "application")

        # Clean up
        withdraw_application(self.test_job_id)

    def test_export_bookmarks_csv_and_json(self):
        """Test exporting bookmarks in both CSV and JSON formats."""
        # Ensure a bookmark exists
        self.client.post(f"/api/bookmarks/{self.test_job_id}")

        # 1. Export JSON
        res_json = self.client.get("/api/bookmarks/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("application/json", res_json.headers.get("content-type", ""))
        bm_list = res_json.json()
        self.assertIsInstance(bm_list, list)
        self.assertTrue(any(b["job_id"] == self.test_job_id for b in bm_list))

        # 2. Export CSV
        res_csv = self.client.get("/api/bookmarks/export?format=csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("Job ID,Job Title,Company", res_csv.text)
        self.assertIn(self.test_job_id, res_csv.text)

        # 3. Invalid format 400
        res_inv = self.client.get("/api/bookmarks/export?format=xml")
        self.assertEqual(res_inv.status_code, 400)

        # Clean up
        self.client.delete(f"/api/bookmarks/{self.test_job_id}")

    def test_application_priority_and_favorite_lifecycle(self):
        """Test setting application priority level, starring as favorite, and filtering."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Priority test"})

        # 2. Update priority to High and star as favorite
        res_prio = self.client.patch(
            f"/api/applications/{self.test_job_id}/priority",
            json={"priority": "High", "is_favorite": True}
        )
        self.assertEqual(res_prio.status_code, 200)
        app_data = res_prio.json()["application"]
        self.assertEqual(app_data["priority"], "High")
        self.assertTrue(app_data["is_favorite"])

        # 3. Filter applications by priority and favorite
        res_filter = self.client.get("/api/applications?priority=High&is_favorite=true")
        self.assertEqual(res_filter.status_code, 200)
        apps = res_filter.json()
        self.assertTrue(any(a["job_id"] == self.test_job_id for a in apps))

        # 4. Invalid priority returns 400
        res_inv = self.client.patch(
            f"/api/applications/{self.test_job_id}/priority",
            json={"priority": "SuperCriticalMega"}
        )
        self.assertEqual(res_inv.status_code, 400)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_database_storage_metrics(self):
        """Test database storage metrics endpoint and collection statistics."""
        res = self.client.get("/api/admin/db/storage")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("storage_engine", data)
        self.assertIn("total_size_bytes", data)
        self.assertIn("total_records", data)
        self.assertIn("collections", data)
        self.assertGreater(data["total_size_bytes"], 0)
        self.assertGreater(data["total_records"], 0)
        self.assertEqual(len(data["collections"]), 5)
        for col in data["collections"]:
            self.assertIn("collection", col)
            self.assertIn("file_path", col)
            self.assertIn("record_count", col)
            self.assertIn("size_bytes", col)
            self.assertIn("avg_bytes_per_record", col)

    def test_job_tag_cloud_aggregation(self):
        """Test tag cloud keyword aggregation and category filtering."""
        res = self.client.get("/api/jobs/analytics/tag-cloud?limit=20")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_tags", data)
        self.assertIn("tags", data)
        self.assertGreater(data["total_tags"], 0)
        self.assertTrue(len(data["tags"]) <= 20)
        top_tag = data["tags"][0]
        self.assertIn("tag", top_tag)
        self.assertIn("count", top_tag)
        self.assertGreater(top_tag["count"], 0)

        # Filtered by category
        res_cat = self.client.get("/api/jobs/analytics/tag-cloud?category=Engineering&limit=10")
        self.assertEqual(res_cat.status_code, 200)
        data_cat = res_cat.json()
        self.assertEqual(data_cat["category"], "Engineering")
        self.assertTrue(len(data_cat["tags"]) <= 10)

    def test_application_rejection_analytics(self):
        """Test aggregation of rejection codes, rates, and candidate debrief notes."""
        # 1. Apply to job and record rejection outcome
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Rejection test"})
        self.client.post(
            f"/api/applications/{self.test_job_id}/outcome",
            json={
                "outcome": "Rejected",
                "reason_code": "Skill Gap",
                "feedback_notes": "Needs deeper experience with distributed caches."
            }
        )

        # 2. Get rejection analytics
        res = self.client.get("/api/applications/analytics/rejections")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_applications", data)
        self.assertIn("total_rejections", data)
        self.assertIn("rejection_rate_percent", data)
        self.assertIn("reason_breakdown", data)
        self.assertIn("sample_feedback_notes", data)
        self.assertGreater(data["total_rejections"], 0)
        self.assertTrue(any(r["reason_code"] == "Skill Gap" for r in data["reason_breakdown"]))
        self.assertTrue(any("distributed caches" in note for note in data["sample_feedback_notes"]))

        # Clean up
        withdraw_application(self.test_job_id)

    def test_session_metrics_endpoint(self):
        """Test retrieving conversational session activity, message counts, and turns metrics."""
        test_sess = "metrics-test-session"
        # 1. Trigger conversation turn
        chat_res = self.client.post("/api/chat", json={"message": "hello assistant", "session_id": test_sess})
        self.assertEqual(chat_res.status_code, 200)

        # 2. Query session metrics
        res = self.client.get(f"/api/chat/session/{test_sess}/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["session_id"], test_sess)
        self.assertGreaterEqual(data["turns_count"], 1)
        self.assertGreaterEqual(data["message_count"], 2)  # system + user + assistant
        self.assertIn("last_active", data)
        self.assertIn("created_at", data)
        self.assertIn("cached_recommendations_count", data)

        # Clean up session
        self.client.post(f"/api/chat/reset/{test_sess}")

    def test_advanced_job_filtering(self):
        """Test multi-criteria complex job filtering with AND/OR skills and work mode."""
        payload = {
            "keywords": ["engineer"],
            "skills": ["python"],
            "match_all_skills": False,
            "work_modes": ["Remote", "Hybrid"],
            "experience_tiers": ["Senior", "Mid-Level"],
            "limit": 10,
            "offset": 0
        }
        res = self.client.post("/api/jobs/filter", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_matched", data)
        self.assertIn("jobs", data)
        self.assertGreater(data["total_matched"], 0)
        for job in data["jobs"]:
            self.assertIn(job["work_mode"], ["Remote", "Hybrid"])
            self.assertIn(job["normalized_tier"], ["Senior", "Mid-Level"])

    def test_update_application_interview_stage(self):
        """Test scheduling an interview and then transitioning its status and rescheduling."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Interview lifecycle test"})

        # 2. Add interview stage
        res_add = self.client.post(
            f"/api/applications/{self.test_job_id}/interviews",
            json={"round_name": "Technical Screen", "scheduled_at": "2026-11-20 10:00"}
        )
        self.assertEqual(res_add.status_code, 201)
        stage_id = res_add.json()["interview"]["stage_id"]

        # 3. Update interview stage (reschedule and complete)
        res_update = self.client.patch(
            f"/api/applications/{self.test_job_id}/interviews/{stage_id}",
            json={"status": "Completed", "notes": "Candidate passed technical coding challenge."}
        )
        self.assertEqual(res_update.status_code, 200)
        updated_stage = res_update.json()["interview"]
        self.assertEqual(updated_stage["status"], "Completed")
        self.assertIn("coding challenge", updated_stage["notes"])

        # 4. Invalid status returns 400
        res_inv = self.client.patch(
            f"/api/applications/{self.test_job_id}/interviews/{stage_id}",
            json={"status": "InvalidStageStatus99"}
        )
        self.assertEqual(res_inv.status_code, 400)

        # Clean up
        withdraw_application(self.test_job_id)

    def test_batch_job_scoring(self):
        """Test scoring multiple job postings in batch against candidate profile."""
        payload = {
            "candidate_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
            "job_ids": [self.test_job_id]
        }
        res = self.client.post("/api/matcher/batch-score", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_evaluated", data)
        self.assertIn("results", data)
        self.assertEqual(data["total_evaluated"], 1)
        item = data["results"][0]
        self.assertEqual(item["job_id"], self.test_job_id)
        self.assertIn("match_score", item)
        self.assertIn("matching_skills", item)
        self.assertIn("missing_skills", item)
        self.assertIsInstance(item["matching_skills"], list)

    def test_bookmark_categories_aggregation(self):
        """Test bookmark folder categories and counts aggregation."""
        # 1. Bookmark a job and assign category
        self.client.post(f"/api/bookmarks/{self.test_job_id}")
        self.client.patch(f"/api/bookmarks/{self.test_job_id}", json={"category": "Dream Roles"})

        # 2. Get categories
        res = self.client.get("/api/bookmarks/categories")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_categories", data)
        self.assertIn("total_bookmarks", data)
        self.assertIn("categories", data)
        self.assertGreater(data["total_bookmarks"], 0)
        self.assertTrue(any(c["category"] == "Dream Roles" for c in data["categories"]))

        # Clean up
        self.client.delete(f"/api/bookmarks/{self.test_job_id}")

    def test_database_integrity_repair(self):
        """Test database integrity repair endpoint and schema verification."""
        res = self.client.post("/api/admin/db/repair")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("repaired_applications", data)
        self.assertIn("repaired_bookmarks", data)
        self.assertIn("message", data)

    def test_application_deadlines_tracking(self):
        """Test setting and updating target decision deadline and expected feedback date milestones."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id, "notes": "Milestone test"})

        # 2. Update deadlines
        res = self.client.patch(
            f"/api/applications/{self.test_job_id}/deadlines",
            json={
                "decision_deadline": "2026-12-15",
                "expected_feedback_date": "2026-11-30"
            }
        )
        self.assertEqual(res.status_code, 200)
        app_data = res.json()["application"]
        self.assertEqual(app_data["decision_deadline"], "2026-12-15")
        self.assertEqual(app_data["expected_feedback_date"], "2026-11-30")

        # 3. Verify in timeline
        res_time = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_time.status_code, 200)
        self.assertTrue(any(e["event_type"] == "DEADLINES_UPDATED" for e in res_time.json()["timeline"]))

        # Clean up
        withdraw_application(self.test_job_id)

    def test_tier_salary_analytics(self):
        """Test tier-level compensation metrics (min, max, median) across career tiers."""
        res = self.client.get("/api/jobs/analytics/tier-salaries")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_tiers", data)
        self.assertIn("insights", data)
        self.assertGreater(data["total_tiers"], 0)
        for insight in data["insights"]:
            self.assertIn("tier", insight)
            self.assertIn("job_count", insight)
            self.assertIn("avg_min_salary", insight)
            self.assertIn("avg_max_salary", insight)
            self.assertIn("median_max_salary", insight)
            self.assertGreater(insight["avg_max_salary"], 0)

    def test_database_benchmark_profiler(self):
        """Test database read latency benchmarking across JSON collections."""
        res = self.client.get("/api/admin/db/benchmark")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("timestamp", data)
        self.assertIn("total_read_time_ms", data)
        self.assertGreaterEqual(data["total_read_time_ms"], 0.0)
        self.assertIsInstance(data["collections"], list)
        self.assertEqual(len(data["collections"]), 5)
        collection_names = [c["collection"] for c in data["collections"]]
        self.assertIn("jobs", collection_names)
        self.assertIn("applications", collection_names)
        self.assertIn("bookmarks", collection_names)
        self.assertIn("saved_searches", collection_names)
        self.assertIn("search_history", collection_names)
        for metric in data["collections"]:
            self.assertIn("read_latency_ms", metric)
            self.assertIn("record_count", metric)
            self.assertIn("file_size_bytes", metric)
            self.assertGreaterEqual(metric["read_latency_ms"], 0.0)

    def test_candidate_profile_completeness_and_scoring(self):
        """Test candidate profile completeness auditing, rating tiers, and improvement suggestions."""
        # 1. Evaluate empty profile -> Poor rating
        res_empty = self.client.post("/api/profile/completeness", json={})
        self.assertEqual(res_empty.status_code, 200)
        data_empty = res_empty.json()
        self.assertEqual(data_empty["completeness_percentage"], 0)
        self.assertEqual(data_empty["rating"], "Poor")
        self.assertTrue(len(data_empty["missing_sections"]) >= 5)
        self.assertTrue(len(data_empty["improvement_suggestions"]) >= 5)

        # 2. Evaluate comprehensive profile -> All-Star rating
        full_profile = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1-555-0199",
            "summary": "Experienced full-stack software engineer with a track record of scaling high-throughput distributed architectures.",
            "skills": ["Python", "FastAPI", "React", "Docker", "PostgreSQL", "AWS"],
            "experience": "Senior Software Engineer at TechCorp for 5 years developing cloud services.",
            "experience_years": "5+ years",
            "education": ["B.S. in Computer Science, University of California, 2019"],
            "projects": ["Distributed Job Crawler in Go and Redis"],
            "preferred_roles": ["Staff Backend Engineer", "Lead Developer"]
        }
        res_full = self.client.post("/api/profile/completeness", json=full_profile)
        self.assertEqual(res_full.status_code, 200)
        data_full = res_full.json()
        self.assertEqual(data_full["completeness_percentage"], 100)
        self.assertEqual(data_full["rating"], "All-Star")
        self.assertEqual(len(data_full["missing_sections"]), 0)

        # 3. Save profile and retrieve via GET
        res_save = self.client.post("/api/profile", json=full_profile)
        self.assertEqual(res_save.status_code, 200)

        res_get = self.client.get("/api/profile")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json().get("name"), "Jane Doe")

        res_comp_get = self.client.get("/api/profile/completeness")
        self.assertEqual(res_comp_get.status_code, 200)
        self.assertEqual(res_comp_get.json()["rating"], "All-Star")

    def test_candidate_salary_preferences_lifecycle(self):
        """Test setting, validating, and retrieving candidate target salary preferences."""
        # 1. Valid salary preferences update -> 200 OK
        payload = {
            "target_min_salary": 130000.0,
            "target_max_salary": 170000.0,
            "currency": "EUR",
            "period": "year"
        }
        res_patch = self.client.patch("/api/profile/salary-preferences", json=payload)
        self.assertEqual(res_patch.status_code, 200)
        data = res_patch.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["target_min_salary"], 130000.0)
        self.assertEqual(data["target_max_salary"], 170000.0)
        self.assertEqual(data["currency"], "EUR")

        # 2. Retrieve preferences via GET -> 200 OK
        res_get = self.client.get("/api/profile/salary-preferences")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["target_min_salary"], 130000.0)
        self.assertEqual(data_get["currency"], "EUR")

        # 3. Invalid salary range (min > max) -> 400 Bad Request
        res_invalid = self.client.patch("/api/profile/salary-preferences", json={
            "target_min_salary": 200000.0,
            "target_max_salary": 100000.0
        })
        self.assertEqual(res_invalid.status_code, 400)

    def test_candidate_work_preferences_lifecycle(self):
        """Test setting and retrieving company stage, industry, and remote work preferences."""
        # 1. Update preferences -> 200 OK
        payload = {
            "company_sizes": ["Startup", "Growth"],
            "preferred_industries": ["Fintech", "AI/ML", "DevTools"],
            "remote_preference": "remote"
        }
        res_patch = self.client.patch("/api/profile/work-preferences", json=payload)
        self.assertEqual(res_patch.status_code, 200)
        data = res_patch.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["remote_preference"], "remote")
        self.assertIn("Fintech", data["preferred_industries"])
        self.assertIn("Startup", data["company_sizes"])

        # 2. Retrieve preferences via GET -> 200 OK
        res_get = self.client.get("/api/profile/work-preferences")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["remote_preference"], "remote")
        self.assertEqual(len(data_get["preferred_industries"]), 3)

    def test_candidate_certifications_management(self):
        """Test adding, listing, and deleting candidate professional certifications."""
        # 1. Add certification -> 201 Created
        payload = {
            "name": "AWS Certified Solutions Architect",
            "issuer": "Amazon Web Services",
            "issue_date": "2024-05",
            "expiration_date": "2027-05",
            "credential_url": "https://aws.amazon.com/verify/12345"
        }
        res_post = self.client.post("/api/profile/certifications", json=payload)
        self.assertEqual(res_post.status_code, 201)
        cert = res_post.json()
        cert_id = cert["cert_id"]
        self.assertEqual(cert["name"], "AWS Certified Solutions Architect")
        self.assertEqual(cert["issuer"], "Amazon Web Services")

        # 2. List certifications -> includes new cert
        res_list = self.client.get("/api/profile/certifications")
        self.assertEqual(res_list.status_code, 200)
        list_data = res_list.json()
        self.assertGreaterEqual(list_data["total"], 1)
        self.assertTrue(any(c["cert_id"] == cert_id for c in list_data["certifications"]))

        # 3. Delete certification -> 200 OK
        res_del = self.client.delete(f"/api/profile/certifications/{cert_id}")
        self.assertEqual(res_del.status_code, 200)

        # 4. Delete non-existent cert -> 404 Not Found
        res_del_404 = self.client.delete(f"/api/profile/certifications/{cert_id}")
        self.assertEqual(res_del_404.status_code, 404)

    def test_resume_ats_readability_analysis(self):
        """Test resume ATS scoring, section recognition, keyword density, and advice generation."""
        # 1. Short text validation error -> 422
        res_short = self.client.post("/api/resume/ats-score", json={"resume_text": "Too brief"})
        self.assertEqual(res_short.status_code, 422)

        # 2. Comprehensive resume evaluation -> 200 OK
        sample_resume = """
        Alex Chen - Senior Distributed Systems Engineer
        Email: alex.chen@example.com | Phone: +1-555-0182 | LinkedIn: linkedin.com/in/alexchen

        Summary:
        Accomplished software engineer with 6+ years of experience designing, deploying, and scaling
        cloud-native microservices, automated CI/CD pipelines, and high-performance database architectures.

        Work Experience:
        Senior Software Engineer - CloudScale Systems (2021 - Present)
        - Architected and developed event-driven message broker handling 50k requests per second.
        - Optimized distributed database query performance, decreasing p99 latency by 45%.
        - Led cross-functional team of 6 engineers to deploy secure Kubernetes infrastructure.

        Technical Skills:
        Python, Go, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, AWS, System Architecture, Testing.

        Education:
        B.S. in Computer Science - Stanford University, 2018

        Projects:
        Open Source Async Query Engine: Built high-throughput data streamer in Go.
        """
        res = self.client.post("/api/resume/ats-score", json={"resume_text": sample_resume})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["ats_score"], 70)
        self.assertIn(data["readability_grade"], ["Medium", "High"])
        self.assertGreater(data["word_count"], 100)
        self.assertIn("Work Experience", data["detected_sections"])
        self.assertIn("Technical Skills", data["detected_sections"])
        self.assertTrue(len(data["keyword_density"]) >= 3)
        self.assertTrue(len(data["sections_breakdown"]) >= 5)

    def test_jobs_export_to_csv(self):
        """Test exporting active jobs as a formatted CSV spreadsheet."""
        # 1. Export all jobs
        res = self.client.get("/api/jobs/export/csv")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.headers["content-type"])
        self.assertIn('filename="jobs_export.csv"', res.headers.get("content-disposition", ""))
        content = res.text
        lines = content.strip().split("\n")
        self.assertGreater(len(lines), 1)  # Header + at least 1 job
        header = lines[0]
        self.assertIn("job_id", header)
        self.assertIn("title", header)
        self.assertIn("company", header)
        self.assertIn("location", header)

        # 2. Filtered export
        res_filtered = self.client.get("/api/jobs/export/csv?query=software")
        self.assertEqual(res_filtered.status_code, 200)
        self.assertGreater(len(res_filtered.text.strip().split("\n")), 1)

    def test_company_hiring_surge_analytics(self):
        """Test detection of companies with aggressive hiring velocities and department surges."""
        res = self.client.get("/api/jobs/analytics/hiring-surges?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["analyzed_companies"], 0)
        self.assertIsInstance(data["surging_companies"], list)
        self.assertTrue(len(data["surging_companies"]) > 0)
        for item in data["surging_companies"]:
            self.assertIn("company", item)
            self.assertGreaterEqual(item["total_active_jobs"], 1)
            self.assertIn(item["growth_indicator"], ["Rapid Hiring", "Steady Hiring", "Moderate"])
            self.assertIsInstance(item["sample_roles"], list)

    def test_stale_jobs_detection(self):
        """Test scanning job database for aging/stale postings exceeding threshold."""
        res = self.client.get("/api/jobs/analytics/stale?threshold_days=30&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["threshold_days"], 30)
        self.assertIn("total_stale_jobs", data)
        self.assertIsInstance(data["jobs"], list)
        for item in data["jobs"]:
            self.assertIn("job_id", item)
            self.assertGreaterEqual(item["age_days"], 30)
            self.assertIn(item["status"], ["stale", "expired"])

    def test_skill_correlations_analytics(self):
        """Test companion skill co-occurrence and affinity ratio analytics."""
        res = self.client.get("/api/jobs/analytics/skill-correlations?skill=python&limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["target_skill"], "python")
        self.assertGreater(data["total_jobs_with_skill"], 0)
        self.assertIsInstance(data["correlated_skills"], list)
        for item in data["correlated_skills"]:
            self.assertIn("co_skill", item)
            self.assertGreaterEqual(item["co_occurrence_count"], 1)
            self.assertGreater(item["affinity_score"], 0.0)
            self.assertLessEqual(item["affinity_score"], 1.0)

    def test_remote_work_breakdown_analytics(self):
        """Test market breakdown across remote, hybrid, and onsite workplace arrangements."""
        res = self.client.get("/api/jobs/analytics/remote-breakdown")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["total_jobs_analyzed"], 0)
        self.assertIsInstance(data["breakdown"], list)
        self.assertTrue(len(data["breakdown"]) >= 3)
        types = [b["workplace_type"] for b in data["breakdown"]]
        self.assertIn("Fully Remote", types)
        self.assertIn("Onsite", types)
        for item in data["breakdown"]:
            self.assertIn("job_count", item)
            self.assertIn("percentage", item)

    def test_application_source_tracking(self):
        """Test recording and updating application submission channel / referral source."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id})

        # 2. Update source
        res_patch = self.client.patch(
            f"/api/applications/{self.test_job_id}/source",
            json={"source": "LinkedIn Referral"}
        )
        self.assertEqual(res_patch.status_code, 200)
        app_data = res_patch.json()["application"]
        self.assertEqual(app_data["source"], "LinkedIn Referral")

        # 3. Check timeline
        res_time = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_time.status_code, 200)
        events = res_time.json()["timeline"]
        self.assertTrue(any(e["event_type"] == "SOURCE_UPDATED" for e in events))

        # 4. Clean up
        withdraw_application(self.test_job_id)

    def test_application_interview_question_bank(self):
        """Test logging and retrieving interview questions and answers per application."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id})

        # 2. Log question -> 201 Created
        payload = {
            "question": "How do you design an idempotent payment processing webhook?",
            "answer_notes": "Use unique idempotency keys in Redis with short TTL and database unique constraints.",
            "difficulty": "Hard",
            "round_type": "System Design"
        }
        res_post = self.client.post(f"/api/applications/{self.test_job_id}/interview-questions", json=payload)
        self.assertEqual(res_post.status_code, 201)
        q_data = res_post.json()
        self.assertIn("question_id", q_data)
        self.assertEqual(q_data["difficulty"], "Hard")
        self.assertEqual(q_data["round_type"], "System Design")

        # 3. Retrieve question bank -> 200 OK
        res_get = self.client.get(f"/api/applications/{self.test_job_id}/interview-questions")
        self.assertEqual(res_get.status_code, 200)
        list_data = res_get.json()
        self.assertEqual(list_data["total"], 1)
        self.assertEqual(list_data["questions"][0]["question_id"], q_data["question_id"])

        # 4. Verify in timeline
        res_time = self.client.get(f"/api/applications/{self.test_job_id}/timeline")
        self.assertEqual(res_time.status_code, 200)
        self.assertTrue(any(e["event_type"] == "INTERVIEW_QUESTION_LOGGED" for e in res_time.json()["timeline"]))

        # 5. Clean up
        withdraw_application(self.test_job_id)

    def test_application_channel_analytics(self):
        """Test application submission channel effectiveness and conversion analytics."""
        # 1. Apply to job
        self.client.post("/api/applications", json={"job_id": self.test_job_id})
        self.client.patch(f"/api/applications/{self.test_job_id}/source", json={"source": "Direct Referral"})

        # 2. Query channel analytics -> 200 OK
        res = self.client.get("/api/applications/analytics/channels")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["total_applications"], 0)
        self.assertIsInstance(data["channels"], list)
        self.assertTrue(any(c["source"] == "Direct Referral" for c in data["channels"]))

        # 3. Clean up
        withdraw_application(self.test_job_id)

    def test_application_offer_packages_comparison(self):
        """Test comparing multiple job offers, ranking by total first-year compensation."""
        # 1. Validation failure for < 2 offers -> 422
        res_single = self.client.post("/api/applications/compare-offers", json={"offers": [
            {"job_id": "job_1", "company": "Co A", "role_title": "SWE", "base_salary": 120000.0}
        ]})
        self.assertEqual(res_single.status_code, 422)

        # 2. Compare 2 offers -> 200 OK
        payload = {
            "offers": [
                {
                    "job_id": "job_a",
                    "company": "Alpha Tech",
                    "role_title": "Senior Engineer",
                    "base_salary": 140000.0,
                    "target_bonus": 15000.0,
                    "equity_annual_value": 20000.0,
                    "signing_bonus": 10000.0
                },
                {
                    "job_id": "job_b",
                    "company": "Beta Cloud",
                    "role_title": "Staff Engineer",
                    "base_salary": 160000.0,
                    "target_bonus": 25000.0,
                    "equity_annual_value": 35000.0,
                    "signing_bonus": 20000.0
                }
            ]
        }
        res = self.client.post("/api/applications/compare-offers", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_offers_evaluated"], 2)
        self.assertEqual(data["highest_paying_company"], "Beta Cloud")
        self.assertEqual(data["offers"][0]["company"], "Beta Cloud")
        self.assertEqual(data["offers"][0]["rank"], 1)
        self.assertTrue(data["offers"][0]["is_highest_comp"])
        self.assertEqual(data["offers"][1]["rank"], 2)


if __name__ == "__main__":
    unittest.main()

