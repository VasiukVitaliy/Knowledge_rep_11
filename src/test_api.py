import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.update(API_KEY_PUBLIC="p" * 20, API_KEY_ANALYST="a" * 20, RATE_LIMIT_PER_MIN="5")

import pytest

CHUNKS = [
    {"chunk_id": "pub#1", "doc_id": "pub", "title": "T", "section": "S", "type": "нормативний",
     "date": "2026-01-01", "sensitivity": "public", "context": "публічний   текст"},
    {"chunk_id": "res#1", "doc_id": "res", "title": "T", "section": "S", "type": "новина",
     "date": "2026-01-01", "sensitivity": "restricted", "context": "закритий текст"},
]


class FakeChat:
    def __init__(self, user_id, strict_mode=("public",), use_graph=True):
        self.strict, self._history, self.last, self.use_graph = list(strict_mode), [], {}, use_graph

    def call_llm(self, msg):
        ids = [c["chunk_id"] for c in CHUNKS if c["sensitivity"] in self.strict]
        self.last = {"sources": ids, "tokens_in": 10, "tokens_out": 5, "graph_text": ""}
        return "відповідь " + " ".join(f"[{i}]" for i in ids)


fake = types.ModuleType("llm_pipe")
fake.UserChat, fake.G = FakeChat, None
fake.R = types.SimpleNamespace(chunks=CHUNKS)
sys.modules["llm_pipe"] = fake

from fastapi.testclient import TestClient
from api import main

client = TestClient(main.app)
PUB, ANA = {"X-API-Key": "p" * 20}, {"X-API-Key": "a" * 20}


@pytest.fixture(autouse=True)
def reset_limits():
    main._hits.clear()


def test_health_no_key():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_requires_key():
    assert client.post("/ask", json={"prompt": "питання"}).status_code == 401
    assert client.post("/ask", json={"prompt": "питання"}, headers={"X-API-Key": "x" * 20}).status_code == 401


def test_two_roles_see_different_sources():
    pub = client.post("/ask", json={"prompt": "питання"}, headers=PUB).json()
    ana = client.post("/ask", json={"prompt": "питання"}, headers=ANA).json()
    assert [s["chunk_id"] for s in pub["sources"]] == ["pub#1"] and pub["role"] == "public"
    assert [s["chunk_id"] for s in ana["sources"]] == ["pub#1", "res#1"] and ana["role"] == "analyst"


def test_sources_endpoint_rights_and_404():
    assert client.get("/sources/res%231", headers=PUB).status_code == 403
    r = client.get("/sources/res%231", headers=ANA)
    assert r.status_code == 200 and r.json()["text"] == "закритий текст"
    assert client.get("/sources/nope", headers=ANA).status_code == 404
    assert client.get("/sources/pub%231").status_code == 401


@pytest.mark.parametrize("body", [{"prompt": "ab"}, {"prompt": "x" * 1001}, {"prompt": "   "},
                                  {"prompt": "питання", "role": "analyst"},     # роль від клієнта заборонена
                                  {"prompt": "питання", "user_id": "not-uuid"},
                                  {"prompt": 123}, {}])
def test_validation_422(body):
    assert client.post("/ask", json=body, headers=PUB).status_code == 422


def test_rate_limit_429_with_retry_after():
    codes = [client.post("/ask", json={"prompt": "питання"}, headers=PUB).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    r = client.post("/ask", json={"prompt": "питання"}, headers=PUB)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.post("/ask", json={"prompt": "питання"}, headers=ANA).status_code == 200   # ліміт на ключ


def test_session_bound_to_role():
    uid = "11111111-1111-1111-1111-111111111111"
    a = client.post("/ask", json={"prompt": "питання", "user_id": uid}, headers=PUB).json()
    b = client.post("/ask", json={"prompt": "питання", "user_id": uid}, headers=ANA).json()
    assert a["role"] == "public" and b["role"] == "analyst"
