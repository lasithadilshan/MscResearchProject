import pytest
from fastapi import status

def test_register_success(client):
    response = client.post("/auth/register", json={
        "email": "newuser@example.com",
        "password": "ValidPassword123!"
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data
    assert "password" not in data
    assert "hashed_password" not in data

def test_register_duplicate_email(client, test_user):
    response = client.post("/auth/register", json={
        "email": test_user.email,
        "password": "AnotherPassword123!"
    })
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Email already registered"

def test_login_success(client, test_user):
    response = client.post("/auth/login", data={
        "username": test_user.email,
        "password": "StrongPassword123!"
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 20

def test_login_invalid_password(client, test_user):
    response = client.post("/auth/login", data={
        "username": test_user.email,
        "password": "WrongPasswordHere"
    })
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Incorrect email or password"

def test_login_nonexistent_user(client):
    response = client.post("/auth/login", data={
        "username": "nonexistent@example.com",
        "password": "SomePassword"
    })
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Incorrect email or password"

def test_unauthorized_access_documents(client):
    response = client.get("/documents/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_invalid_token_format(client):
    response = client.get("/documents/", headers={"Authorization": "Bearer invalid.token.value"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
