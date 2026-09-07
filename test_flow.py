import requests
import time

BASE_URL = "http://localhost:8000"

print("1. Logging in...")
res = requests.post(f"{BASE_URL}/auth/login", data={"username": "test@example.com", "password": "testpassword"})
res.raise_for_status()
token = res.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("Token acquired.")

print("2. Uploading document...")
files = {'file': ('test_brd.txt', 'This is a test BRD document for a user login feature. The system should allow a user to enter their email and password. If the credentials match, they should be redirected to the dashboard. If they do not match, an error message "Invalid credentials" should be shown.', 'text/plain')}
res = requests.post(f"{BASE_URL}/documents/upload", headers=headers, files=files)
res.raise_for_status()
doc_id = res.json()["id"]
print(f"Document uploaded. ID: {doc_id}")

print("3. Starting generation job for User Stories...")
res = requests.post(f"{BASE_URL}/generate/user-stories?document_id={doc_id}", headers=headers)
res.raise_for_status()
job_id = res.json()["id"]
print(f"Job started. ID: {job_id}")

print("4. Polling for job status...")
while True:
    res = requests.get(f"{BASE_URL}/generate/job/{job_id}", headers=headers)
    res.raise_for_status()
    status = res.json()["status"]
    print(f"Status: {status}")
    if status in ["SUCCESS", "FAILURE"]:
        if status == "SUCCESS":
            print("Result:", res.json().get("result"))
        else:
            print("Job failed.")
        break
    time.sleep(2)
