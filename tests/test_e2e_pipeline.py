import pytest
import json
from fastapi import status
from app.models.models import Job

def test_full_sdlc_generation_pipeline(client, db):
    # Step 1: User Registration & Authentication
    reg_res = client.post("/auth/register", json={
        "email": "e2e_tester@sdlc.io",
        "password": "E2ETestPassword123!"
    })
    assert reg_res.status_code == status.HTTP_200_OK
    user_id = reg_res.json()["id"]

    login_res = client.post("/auth/login", data={
        "username": "e2e_tester@sdlc.io",
        "password": "E2ETestPassword123!"
    })
    assert login_res.status_code == status.HTTP_200_OK
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Step 2: Upload BRD Document
    brd_content = (
        b"PROJECT SPECIFICATION: User Profile Management\n"
        b"1. User can view profile information including email, avatar, and contact number.\n"
        b"2. User can update their contact phone number.\n"
        b"3. System must validate that phone number contains 10 numeric digits.\n"
        b"4. System sends an SMS verification code to verify the updated phone number.\n"
    )
    files = {"file": ("user_profile_brd.txt", brd_content, "text/plain")}
    doc_res = client.post("/documents/upload", headers=headers, files=files)
    assert doc_res.status_code == status.HTTP_200_OK
    doc_data = doc_res.json()
    doc_id = doc_data["id"]
    assert doc_data["owner_id"] == user_id

    # Step 3: Trigger User Story Generation
    us_job_res = client.post(f"/generate/user-stories?document_id={doc_id}", headers=headers)
    assert us_job_res.status_code == status.HTTP_200_OK
    us_job_id = us_job_res.json()["id"]

    # Simulate Celery Worker completion for User Stories
    us_job = db.query(Job).filter(Job.id == us_job_id).first()
    us_mock_output = {
        "user_stories": [
            {
                "id": "US_001",
                "title": "Update Phone Number",
                "story": "As a user, I want to update my phone number so that I can receive SMS notifications.",
                "acceptance_criteria": [
                    "Given user is on profile page, When valid 10-digit number is entered, Then update button is enabled",
                    "Given invalid phone number, When user clicks update, Then validation error is shown"
                ],
                "priority": "High",
                "story_points": 3,
                "category": "Profile",
                "notes": ["Integrates with SMS Gateway"]
            }
        ]
    }
    us_job.status = "SUCCESS"
    us_job.result = json.dumps(us_mock_output)
    db.commit()

    # Verify Job Status Polling
    poll_us = client.get(f"/generate/job/{us_job_id}", headers=headers)
    assert poll_us.status_code == status.HTTP_200_OK
    assert poll_us.json()["status"] == "SUCCESS"
    extracted_stories = poll_us.json()["result"]["user_stories"]
    assert len(extracted_stories) == 1
    assert extracted_stories[0]["id"] == "US_001"

    # Step 4: Convert User Story to Test Cases
    tc_input_text = (
        f"Title: {extracted_stories[0]['title']}\n"
        f"Story: {extracted_stories[0]['story']}\n"
        f"Acceptance Criteria:\n- " + "\n- ".join(extracted_stories[0]["acceptance_criteria"])
    )
    tc_job_res = client.post(
        f"/generate/test-cases?document_id={doc_id}",
        headers=headers,
        json={"user_story_text": tc_input_text}
    )
    assert tc_job_res.status_code == status.HTTP_200_OK
    tc_job_id = tc_job_res.json()["id"]

    # Simulate Celery Worker completion for Test Cases
    tc_job = db.query(Job).filter(Job.id == tc_job_id).first()
    tc_mock_output = {
        "test_cases": [
            {
                "id": "TC_001",
                "title": "Validate 10-digit phone number entry",
                "preconditions": ["User is logged in", "Profile page is loaded"],
                "test_data": ["Phone: 0771234567"],
                "test_steps": ["Navigate to profile", "Enter 0771234567 in phone field", "Click Save"],
                "expected_results": ["Verification code modal is displayed"],
                "priority": "High",
                "attachments": []
            }
        ]
    }
    tc_job.status = "SUCCESS"
    tc_job.result = json.dumps(tc_mock_output)
    db.commit()

    poll_tc = client.get(f"/generate/job/{tc_job_id}", headers=headers)
    assert poll_tc.status_code == status.HTTP_200_OK
    assert poll_tc.json()["status"] == "SUCCESS"
    test_cases = poll_tc.json()["result"]["test_cases"]
    assert len(test_cases) == 1
    assert test_cases[0]["id"] == "TC_001"

    # Step 5: Convert Test Case to Cucumber BDD
    tc_text = f"TC_001: {test_cases[0]['title']}\nSteps: " + "; ".join(test_cases[0]["test_steps"])
    cuc_job_res = client.post(
        f"/generate/cucumber?document_id={doc_id}",
        headers=headers,
        json={"test_case_text": tc_text}
    )
    assert cuc_job_res.status_code == status.HTTP_200_OK
    cuc_job_id = cuc_job_res.json()["id"]

    cuc_job = db.query(Job).filter(Job.id == cuc_job_id).first()
    cuc_mock_script = (
        "**FEATURE FILE (Profile.feature):**\n"
        "```gherkin\n"
        "Feature: Profile Phone Validation\n"
        "  Scenario: Valid phone number\n"
        "    Given user is on profile page\n"
        "    When user enters \"0771234567\" in phone field\n"
        "    Then verification modal is displayed\n"
        "```\n"
        "**STEP DEFINITIONS (ProfileSteps.java):**\n"
        "```java\n"
        "package steps;\n"
        "public class ProfileSteps {}\n"
        "```"
    )
    cuc_job.status = "SUCCESS"
    cuc_job.result = json.dumps({"cucumber_script": cuc_mock_script})
    db.commit()

    poll_cuc = client.get(f"/generate/job/{cuc_job_id}", headers=headers)
    assert poll_cuc.status_code == status.HTTP_200_OK
    assert poll_cuc.json()["status"] == "SUCCESS"

    # Step 6: Convert Test Case to Selenium Script
    sel_job_res = client.post(
        f"/generate/selenium?document_id={doc_id}",
        headers=headers,
        json={"test_case_text": tc_text}
    )
    assert sel_job_res.status_code == status.HTTP_200_OK
    sel_job_id = sel_job_res.json()["id"]

    sel_job = db.query(Job).filter(Job.id == sel_job_id).first()
    sel_mock_script = (
        "from selenium import webdriver\n"
        "from selenium.webdriver.common.by import By\n"
        "def test_phone_update():\n"
        "    driver = webdriver.Chrome()\n"
        "    driver.get('http://localhost:4200/profile')\n"
        "    driver.quit()\n"
    )
    sel_job.status = "SUCCESS"
    sel_job.result = json.dumps({"selenium_script": sel_mock_script})
    db.commit()

    poll_sel = client.get(f"/generate/job/{sel_job_id}", headers=headers)
    assert poll_sel.status_code == status.HTTP_200_OK
    assert poll_sel.json()["status"] == "SUCCESS"
    assert "selenium" in poll_sel.json()["result"]["selenium_script"]
