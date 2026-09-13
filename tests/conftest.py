import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import Base, get_db
from app.core.security import get_password_hash, create_access_token
from app.models.models import User

# In-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def test_user(db):
    user = User(
        email="qa_test_user@example.com",
        hashed_password=get_password_hash("StrongPassword123!")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def auth_headers(test_user):
    token = create_access_token(data={"email": test_user.email})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def secondary_user(db):
    user = User(
        email="secondary_user@example.com",
        hashed_password=get_password_hash("Password12345!")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def secondary_auth_headers(secondary_user):
    token = create_access_token(data={"email": secondary_user.email})
    return {"Authorization": f"Bearer {token}"}
