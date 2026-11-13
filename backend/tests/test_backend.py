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


if __name__ == "__main__":
    unittest.main()

