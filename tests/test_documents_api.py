import pytest
from io import BytesIO
from fastapi import status

def test_upload_document_success(client, auth_headers):
    file_content = b"This is a Business Requirements Document for testing."
    files = {"file": ("test_brd.txt", file_content, "text/plain")}
    
    response = client.post("/documents/upload", headers=auth_headers, files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test_brd.txt"
    assert data["text_length"] == len(file_content)

def test_upload_empty_document_returns_400(client, auth_headers):
    # Empty file has no extractable text, should return 400 Bad Request
    files = {"file": ("empty.txt", b"", "text/plain")}
    
    response = client.post("/documents/upload", headers=auth_headers, files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Could not extract text from file" in response.json()["detail"]

def test_upload_unsupported_file_returns_400(client, auth_headers):
    # .bin file returns empty string from extract_text_from_file
    files = {"file": ("binary.bin", b"\x00\x01\x02\x03", "application/octet-stream")}
    
    response = client.post("/documents/upload", headers=auth_headers, files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_get_documents_list(client, auth_headers):
    # Upload one doc first
    files = {"file": ("doc_list_test.txt", b"Document content here", "text/plain")}
    client.post("/documents/upload", headers=auth_headers, files=files)

    response = client.get("/documents/", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    docs = response.json()
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert any(d["filename"] == "doc_list_test.txt" for d in docs)

def test_multi_tenant_document_isolation(client, auth_headers, secondary_auth_headers):
    # User A uploads a document
    files_a = {"file": ("confidential_a.txt", b"User A confidential requirements", "text/plain")}
    res_a = client.post("/documents/upload", headers=auth_headers, files=files_a)
    doc_a_id = res_a.json()["id"]

    # User B queries their documents
    res_b = client.get("/documents/", headers=secondary_auth_headers)
    docs_b = res_b.json()

    # User A's document must NOT appear in User B's list
    assert not any(d["id"] == doc_a_id for d in docs_b)
