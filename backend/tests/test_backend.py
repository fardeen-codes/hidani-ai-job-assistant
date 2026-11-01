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


if __name__ == "__main__":
    unittest.main()
