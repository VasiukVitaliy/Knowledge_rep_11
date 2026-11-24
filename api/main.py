import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi import Path as PathParam
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator

import llm_pipe                      # завантажує моделі й пошук при імпорті
from llm_pipe import UserChat
from rag import ROLES                # роль -> дозволені sensitivity

log = logging.getLogger("api")

KEY_ENV = {"public": "API_KEY_PUBLIC", "analyst": "API_KEY_ANALYST"}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
MAX_SESSIONS = 200
MAX_HISTORY_MSGS = 6                 
LOG_PATH = Path("logs/api_log.jsonl")

API_KEYS = {role: os.getenv(env) for role, env in KEY_ENV.items() if os.getenv(env)}
if not API_KEYS:
    raise RuntimeError("Задайте API_KEY_PUBLIC та/або API_KEY_ANALYST у .env")
if any(len(k) < 16 for k in API_KEYS.values()):
    raise RuntimeError("API-ключі мають бути не коротші за 16 символів")

CHUNKS = {c["chunk_id"]: c for c in llm_pipe.R.chunks}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")      
    user_id: Optional[UUID] = Field(None, description="UUID сесії діалогу; якщо не вказано, створюється новий")
    prompt: StrictStr = Field(..., min_length=3, max_length=1000, description="Питання користувача")
    use_graph: StrictBool = Field(True, description="false = режим «лише текст», true = «текст + підграф»")

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("питання занадто коротке")
        return v


class SourceInfo(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    type: Optional[str] = None
    date: Optional[str] = None
    sensitivity: str


class GraphInfo(BaseModel):
    used: bool
    mode: Optional[str] = None
    entities: list[str] = []
    facts: str = Field("", description="Факти підграфа з id виду [g1]")


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0


class AskResponse(BaseModel):
    user_id: UUID
    role: str
    answer: str
    sources: list[SourceInfo]
    graph: GraphInfo
    usage: Usage
    latency_ms: float
    citations_valid: bool = Field(True, description="false, якщо модель цитувала id поза наданим контекстом (їх прибрано)")
    invalid_citations: list[str] = []


class SourceDetail(SourceInfo):
    text: str


class ErrorOut(BaseModel):
    detail: str


ERRORS = {
    401: {"model": ErrorOut, "description": "Немає або невірний API-ключ"},
    422: {"description": "Невалідний вхід (Pydantic): порожнє/задовге питання, зайві поля, не-UUID"},
    403: {"model": ErrorOut, "description": "Роль не має доступу до цього ресурсу"},
    429: {"model": ErrorOut, "description": "Перевищено ліміт запитів (заголовок Retry-After)"},
}


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def current_role(key: Optional[str] = Security(api_key_header)) -> str:
    found = None
    if key:
        for role, expected in API_KEYS.items():      
            if secrets.compare_digest(key.encode(), expected.encode()):
                found = role
    if found is None:
        raise HTTPException(401, "Невірний або відсутній API-ключ",
                            headers={"WWW-Authenticate": "ApiKey"})
    return found


_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def rate_limit(role: str = Depends(current_role)) -> str:
    """Ковзне вікно 60 с на ключ (роль)."""
    now = time.monotonic()
    with _hits_lock:
        q = _hits[role]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            retry = int(60 - (now - q[0])) + 1
            raise HTTPException(429, f"Перевищено ліміт {RATE_LIMIT} запитів на хвилину",
                                headers={"Retry-After": str(retry)})
        q.append(now)
    return role


sessions: "OrderedDict[tuple, UserChat]" = OrderedDict()
sessions_lock = threading.Lock()
llm_lock = threading.Lock()          # одна модель на GPU: генерації виконуються по черзі


def get_chat(role: str, user_id: UUID) -> UserChat:
    key = (role, user_id)            # сесія прив'язана до ролі: чужу сесію з іншим ключем не отримати
    with sessions_lock:
        chat = sessions.get(key)
        if chat is None:
            chat = UserChat(str(user_id), strict_mode=ROLES[role])
            sessions[key] = chat
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        else:
            sessions.move_to_end(key)
        return chat


def visible(role: str, chunk: dict) -> bool:
    return chunk["sensitivity"] in ROLES[role]


def source_info(c: dict) -> SourceInfo:
    return SourceInfo(chunk_id=c["chunk_id"], doc_id=c.get("doc_id"), title=c.get("title"),
                      section=c.get("section"), type=c.get("type"), date=c.get("date"),
                      sensitivity=c["sensitivity"])


def write_log(row: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


app = FastAPI(
    title="Domain Knowledge Hub: RAG-чат",
    version="1.0",
    description="RAG-чат за корпусом (BM25 + dense + reranker) і графом знань Neo4j. "
                "Автентифікація: заголовок X-API-Key; роль (public/analyst) визначається ключем.",
)


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok", "graph_available": llm_pipe.G is not None, "chunks": len(CHUNKS)}


@app.post("/ask", response_model=AskResponse, tags=["chat"], responses=ERRORS)
def ask(body: AskRequest, role: str = Depends(rate_limit)):
    user_id = body.user_id or uuid4()
    chat = get_chat(role, user_id)
    t0 = time.perf_counter()
    try:
        with llm_lock:
            chat.use_graph = body.use_graph
            chat.last = {}
            answer = chat.call_llm(body.prompt)
            last = dict(chat.last)
            del chat._history[:-MAX_HISTORY_MSGS]
    except Exception:
        log.exception("збій генерації")
        raise HTTPException(500, "Не вдалося згенерувати відповідь")
    latency = round((time.perf_counter() - t0) * 1000, 1)

    chunks = [CHUNKS[i] for i in last.get("sources", []) if i in CHUNKS]
    if any(not visible(role, c) for c in chunks):     # друга лінія захисту
        log.error("порушення політики доступу: роль %s, джерела %s", role, last.get("sources"))
        raise HTTPException(500, "Порушення політики доступу")

    resp = AskResponse(
        user_id=user_id, role=role, answer=answer,
        sources=[source_info(c) for c in chunks],
        graph=GraphInfo(used=bool(last.get("graph_text")), mode=last.get("graph_mode"),
                        entities=last.get("graph_entities", []), facts=last.get("graph_text", "")),
        usage=Usage(tokens_in=last.get("tokens_in", 0), tokens_out=last.get("tokens_out", 0)),
        latency_ms=latency,
        citations_valid=last.get("citations_valid", True),
        invalid_citations=last.get("invalid_citations", []),
    )
    write_log({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "role": role,
               "user_id": str(user_id), "question": body.prompt, "use_graph": body.use_graph,
               "sources": last.get("sources", []), "answer": answer,
               "tokens_in": resp.usage.tokens_in, "tokens_out": resp.usage.tokens_out,
               "graph_error": last.get("graph_error", ""), "invalid_citations": resp.invalid_citations,
               "latency_ms": latency})
    return resp


@app.get("/sources/{source_id}", response_model=SourceDetail, tags=["chat"],
         responses={**ERRORS, 404: {"model": ErrorOut, "description": "Джерела з таким id немає"}})
def get_source(source_id: str = PathParam(..., min_length=1, max_length=128),
               role: str = Depends(rate_limit)):
    c = CHUNKS.get(source_id)
    if c is None:
        raise HTTPException(404, "Джерело не знайдено")
    if not visible(role, c):
        raise HTTPException(403, "Недостатньо прав для цього джерела")
    return SourceDetail(**source_info(c).model_dump(), text=" ".join(c["context"].split()))