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


if __name__ == "__main__":
    unittest.main()

