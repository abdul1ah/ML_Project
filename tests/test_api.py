import pytest
from fastapi.testclient import TestClient
from Script.fastapi.backend import app

# The 'with' statement triggers the @asynccontextmanager lifespan (loading models)
def test_health():
    with TestClient(app) as client:
        # FIX: Changed from "/" to "/health"
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

def test_search_functional():
    with TestClient(app) as client:
        response = client.get("/search?query=Toy")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

def test_recommendation_structure():
    with TestClient(app) as client:
        response = client.get("/recommend?user_id=1&n=5")
        assert response.status_code == 200
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            assert "movie_id" in data[0]
            assert "predicted_rating" in data[0]

def test_user_history_exists():
    with TestClient(app) as client:
        response = client.get("/user/history?user_id=1")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

def test_invalid_user_id():
    with TestClient(app) as client:
        response = client.get("/recommend?user_id=999999")
        assert response.status_code == 200
        assert response.json() == []

def test_admin_stats_protection():
    with TestClient(app) as client:
        response = client.get("/admin/stats")
        assert response.status_code == 403

def test_admin_stats_authorized():
    with TestClient(app) as client:
        response = client.get("/admin/stats?username=admin")
        assert response.status_code == 200
        assert "total_users" in response.json()