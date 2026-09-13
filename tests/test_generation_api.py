import pytest
from unittest.mock import patch
from fastapi import status
from app.models.models import Job
import json

@pytest.fixture
def sample_document(client, auth_headers):
    files = {"file": ("requirements.txt", b"System requires login and payment features.", "text/plain")}
    res = client.post("/documents/upload", headers=auth_headers, files=files)
    return res.json()

def test_generate_user_stories_missing_doc(client, auth_headers):
    res = client.post("/generate/user-stories?document_id=non-existent-uuid", headers=auth_headers)
    assert res.status_code == status.HTTP_404_NOT_FOUND
    assert res.json()["detail"] == "Document not found"

def test_generate_user_stories_cross_tenant_rejected(client, sample_document, secondary_auth_headers):
    # User B tries to generate user stories from User A's document
    res = client.post(f"/generate/user-stories?document_id={sample_document['id']}", headers=secondary_auth_headers)
    assert res.status_code == status.HTTP_404_NOT_FOUND

def test_generate_user_stories_success(client, sample_document, auth_headers):
    with patch("app.api.routers.generation.process_generation_task.apply_async") as mock_celery:
        res = client.post(f"/generate/user-stories?document_id={sample_document['id']}", headers=auth_headers)
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert "id" in data
        assert data["task_type"] == "user_stories"
        assert data["status"] == "PENDING"
        assert data["document_id"] == sample_document["id"]
        mock_celery.assert_called_once()

def test_convert_test_cases_success(client, sample_document, auth_headers):
    with patch("app.api.routers.generation.process_generation_task.apply_async") as mock_celery:
        res = client.post(
            f"/generate/test-cases?document_id={sample_document['id']}",
            headers=auth_headers,
            json={"user_story_text": "As a user, I want to login so that I can access my dashboard."}
        )
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["task_type"] == "test_cases"
        assert data["status"] == "PENDING"
        mock_celery.assert_called_once()

def test_convert_cucumber_success(client, sample_document, auth_headers):
    with patch("app.api.routers.generation.process_generation_task.apply_async") as mock_celery:
        res = client.post(
            f"/generate/cucumber?document_id={sample_document['id']}",
            headers=auth_headers,
            json={"test_case_text": "TC_001: Login with valid credentials"}
        )
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["task_type"] == "cucumber"
        mock_celery.assert_called_once()

def test_convert_selenium_success(client, sample_document, auth_headers):
    with patch("app.api.routers.generation.process_generation_task.apply_async") as mock_celery:
        res = client.post(
            f"/generate/selenium?document_id={sample_document['id']}",
            headers=auth_headers,
            json={"test_case_text": "TC_001: Login with valid credentials"}
        )
        assert res.status_code == status.HTTP_200_OK
        data = res.json()
        assert data["task_type"] == "selenium"
        mock_celery.assert_called_once()

def test_get_job_status_pending_and_success(client, db, test_user, auth_headers):
    # Insert a job directly into DB
    job = Job(
        id="job-test-12345",
        task_type="user_stories",
        status="SUCCESS",
        document_id="doc-123",
        owner_id=test_user.id,
        result=json.dumps({"user_stories": [{"id": "US_001", "title": "Test Story"}]})
    )
    db.add(job)
    db.commit()

    res = client.get(f"/generate/job/{job.id}", headers=auth_headers)
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "user_stories" in data["result"]
    assert data["result"]["user_stories"][0]["id"] == "US_001"

def test_get_job_status_not_found(client, auth_headers):
    res = client.get("/generate/job/nonexistent-job-id", headers=auth_headers)
    assert res.status_code == status.HTTP_404_NOT_FOUND

def test_get_job_status_cross_tenant_isolation(client, db, test_user, secondary_auth_headers):
    job = Job(
        id="user-a-private-job",
        task_type="selenium",
        status="PENDING",
        document_id="doc-123",
        owner_id=test_user.id
    )
    db.add(job)
    db.commit()

    res = client.get(f"/generate/job/{job.id}", headers=secondary_auth_headers)
    assert res.status_code == status.HTTP_404_NOT_FOUND
