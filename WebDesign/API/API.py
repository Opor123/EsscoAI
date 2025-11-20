# API.py - professional, self-contained rewrite for EsscoAI
from __future__ import annotations
import os
import secrets
import json
import logging
import sys
import time
import re
import hashlib
import base64
import asyncio
from time import monotonic
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pydantic import BaseModel, Field

# Optional imports, handled gracefully
try:
    import redis.asyncio as redis
    from fastapi_limiter import FastAPILimiter
    from fastapi_limiter.depends import RateLimiter
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("esscoapi")



# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------
# Path / project discovery
# -----------------------
# Expect layout:
# EsscoAI/
#   WebDesign/
#     API/
#       API.py   <-- this file
#     static/
#       main.html, style.css, script.js
HERE = Path(__file__).resolve().parent.parent   # WebDesign/
PROJECT_ROOT = HERE.parent                      # EsscoAI/
STATIC_DIR = HERE / "static"                    # WebDesign/static
DATA_DIR = PROJECT_ROOT / "Data"

# Ensure module search path includes project root (helps optional imports)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------
# Config
# -----------------------
ENV = os.getenv("ESSCOAI_ENV", "dev")  # 'dev'|'prod'
FORCE_HTTPS = os.getenv("ESSCOAI_FORCE_HTTPS", "0") == "1"
# Only enforce HTTPS in prod by default
ENFORCE_HTTPS = FORCE_HTTPS and ENV == "prod"

SHOW_ADMIN_LINK = os.getenv("ESSCOAI_SHOW_ADMIN_LINK", "0") == "1"


CORS_ORIGINS = [o.strip() for o in os.getenv("ESSCOAI_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
RATE_LIMIT_DISABLE = os.getenv("ESSCOAI_DISABLE_RATE_LIMIT", "1") == "1"  # default disabled for dev
REDIS_URL = os.getenv("ESSCOAI_REDIS_URL", "redis://localhost:6379/0")
ADMIN_USERS_ENV = os.getenv("ESSCOAI_ADMIN_USERS", "admin")
ADMIN_USERS: Dict[str, str] = {}
for pair in [p.strip() for p in ADMIN_USERS_ENV.split(",") if p.strip()]:
    if ":" in pair:
        u, p = pair.split(":", 1)
        ADMIN_USERS[u.strip()] = p.strip()

logger.info(f"SHOW_ADMIN_LINK={SHOW_ADMIN_LINK}")

# -----------------------
# Basic telemetry (optional)
# -----------------------
if METRICS_AVAILABLE:
    QUERY_COUNT = Counter("qa_queries_total", "Total Queries Processed")
    QUERY_DURATION = Histogram("qa_query_duration_seconds", "Query processing time")
    ERROR_COUNT = Counter("qa_errors_total", "Total errors", ["error_type"])
else:
    # Dummy placeholders
    class _Dummy:
        def inc(self, *a, **k): pass
        def time(self): 
            class _Ctxt:
                def __enter__(self): pass
                def __exit__(self, *a): pass
            return _Ctxt()
    QUERY_COUNT = QUERY_DURATION = ERROR_COUNT = _Dummy()

# -----------------------
# Optional components (LLM, QARetriever, FeedbackStore)
# They may live in AI.* or root. Import failures are handled.
# -----------------------
QARetriever = None
QARetrieverSQL = None
get_project_root = None
FeedbackStore = None
get_llm_service = None

try:
    from AI.Model import QARetriever as _QAR, get_project_root as _gpr
    from AI.Model import QARetrieverSQL as _QARSQL
    QARetriever = _QAR
    QARetrieverSQL = _QARSQL
    get_project_root = _gpr
except Exception:
    try:
        from Model import QARetriever as _QAR, get_project_root as _gpr
        from Model import QARetrieverSQL as _QARSQL
        QARetriever = _QAR
        QARetrieverSQL = _QARSQL
        get_project_root = _gpr
    except Exception:
        logger.warning("QARetriever not importable — Q&A endpoints will be disabled")

try:
    from AI.feedback_store import FeedbackStore as _FS
    FeedbackStore = _FS
except Exception:
    try:
        from feedback_store import FeedbackStore as _FS
        FeedbackStore = _FS
    except Exception:
        logger.warning("FeedbackStore not importable — feedback will be appended manually")

try:
    from AI.llm_service import get_llm_service as _get_llm
    get_llm_service = _get_llm
except Exception:
    try:
        from llm_service import get_llm_service as _get_llm
        get_llm_service = _get_llm
    except Exception:
        logger.warning("LLM service not available")

# -----------------------
# Pydantic models
# -----------------------
class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    category: Optional[str] = None

class Feedback(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model_answer: Optional[str] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    user_comment: Optional[str] = Field(None, max_length=2000)
    correct_answer: Optional[str] = None
    label: str = Field("neutral", pattern=r"^(up|down|neutral)$")
    user_name: Optional[str] = None
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        pattern=r"^[A-Za-z0-9_\-:.]+$"
    )

    model_config = {
        "protected_namespaces": ()
    }

class Query(BaseModel):
    query: str
    top_k: Optional[int] = Field(3, ge=1, le=10)
    user_profile: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    query_time_ms: float
    has_high: bool
    confidence: str = Field(..., description="'high'|'medium'|'low'")

class SystemStatus(BaseModel):
    status: str
    qa_system_loaded: bool
    total_items: int
    total_qa_pairs: Optional[int] = None
    data_path: str
    feedback_path: str
    env: str
    admin_link_enabled: bool = False

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(3, ge=1, le=10)
    session_id: Optional[str] = Field(None, max_length=100, pattern=r'^[A-Za-z0-9_\-:.]+$')

class ChatResponse(BaseModel):
    message: str
    answer: Optional[str] = None
    conf: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None

class ConversationSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    last_message_time: float
    preview: str

class ConversationDetails(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int

class AdminDashboardSummary(BaseModel):
    # Traffic / usage
    total_conversations: int
    total_messages: int
    last_24h_messages: int
    first_message_time: Optional[float] = None
    last_message_time: Optional[float] = None

    # Retrieval / data status
    qa_system_loaded: bool
    total_qa_pairs: Optional[int] = None

    # LLM status
    llm_available: bool
    llm_enabled: bool
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None

    # Feedback
    feedback_total: int
    feedback_up: int
    feedback_down: int
    feedback_neutral: int

    # Environment info
    env: str


# -----------------------
# Utility helpers
# -----------------------
SECRET_PATS = [
    re.compile(r"(?i)(sk-[a-z0-9]{16,})"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*[a-z0-9\-_]{12,})")
]

def mask_secrets(s: str) -> str:
    t = s or ""
    for pat in SECRET_PATS:
        t = pat.sub("[REDACTED]", t)
    return t

CHAT_LOG_FILE = DATA_DIR / "conversation.jsonl"

def append_conversation(entry: dict) -> None:
    entry = dict(entry)
    entry["user_message"] = mask_secrets(entry.get("user_message") or "")
    entry["bot_answer"] = mask_secrets(entry.get("bot_answer") or "")
    CHAT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_confidence_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"

# -----------------------
# API Application factory
# -----------------------
class API:
    def __init__(self):
        self.app = FastAPI(title="EsscoAI Integrated API", version="1.0.0")
        self._setup_middlewares()
        self._setup_state()
        self._setup_static_files()
        self._setup_routes()
        # start background tasks if needed
        if not RATE_LIMIT_DISABLE and REDIS_AVAILABLE:
            # attempt to init redis limiter on startup via event
            @self.app.on_event("startup")
            async def init_redis_limiter():
                try:
                    self._redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
                    await FastAPILimiter.init(self._redis)
                    logger.info("Redis rate limiter initialized")
                except Exception as e:
                    logger.warning(f"Redis rate limiter failed to initialize: {e}")

    def _setup_middlewares(self):
        # CORS (allow origins from env)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS or ["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Trusted host middleware - include local dev hosts
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "essco.ai", "api.essco.ai", "*"]
        )

        # Only redirect HTTP->HTTPS in prod and when explicitly enabled
        if ENFORCE_HTTPS:
            self.app.add_middleware(HTTPSRedirectMiddleware)

        # gzip
        self.app.add_middleware(GZipMiddleware, minimum_size=500)

    def _setup_state(self):
        # runtime state values
        self.items: List[Item] = []
        self.project_root: Path = PROJECT_ROOT
        self.data_path: Path = DATA_DIR
        self.feedback_file: Path = self.data_path / "feedback.jsonl"
        self.training_file: Path = self.data_path / "training_ready.jsonl"
        self.qa_retriever = None
        self.feedback_store = None
        self.env = ENV
        self._rate_mode = "off"
        self._memory_buckets: Dict[str, dict] = {}
        self._memory_lock = asyncio.Lock()
        # initialize optional components
        self._init_optional_components()

    def _init_optional_components(self):
        # QARetriever init
        if QARetrieverSQL is not None:
            try:
                logger.info("Initializing QARetriever...")
                db_url = f"sqlite:///{self.data_path / 'essco_ai.db'}"
                self.qa_retriever = QARetrieverSQL(db_url=db_url)
                self.qa_retriever.load_and_build()
                logger.info("Q&A retrieval system ready")
            except Exception as e:
                logger.exception("Failed initializing QARetriever: %s", e)
                self.qa_retriever = None
        else:
            logger.warning("QARetriever implementation not present - QA disabled")

        # Feedback store
        if FeedbackStore is not None:
            try:
                self.feedback_store = FeedbackStore(self.feedback_file)
                logger.info("Feedback store initialized")
            except Exception as e:
                logger.exception("Failed to initialize FeedbackStore: %s", e)
                self.feedback_store = None

        # LLM service
        self.llm_service = None
        if get_llm_service is not None:
            try:
                self.llm_service = get_llm_service()
                if getattr(self.llm_service, "is_available", lambda: False)():
                    logger.info("LLM service available")
                else:
                    logger.info("LLM service initialized but not available")
            except Exception as e:
                logger.exception("Failed to initialize LLM service: %s", e)
                self.llm_service = None

    def _setup_static_files(self):
        # Mount static and validate
        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
            logger.info("Static files mounted from %s", STATIC_DIR)
        else:
            logger.error("Static directory not found: %s", STATIC_DIR)

    # --- simple rate limiting fallback (memory) ---
    def _make_memory_limiter(self, times: int, seconds: int, key_func):
        limit = float(times)
        refill_rate = times / float(seconds)

        async def dep(request: Request = Depends(lambda: True)):
            key = key_func(request) if callable(key_func) else request.client.host
            if not key:
                key = "anon"
            now = monotonic()
            async with self._memory_lock:
                b = self._memory_buckets.get(key)
                if b is None:
                    b = {"tokens": limit, "last": now, "limit": limit, "refill": refill_rate}
                    self._memory_buckets[key] = b
                elapsed = now - b["last"]
                b["tokens"] = min(b["limit"], b["tokens"] + elapsed * b["refill"])
                b["last"] = now
                if b["tokens"] >= 1.0:
                    b["tokens"] -= 1.0
                    return True
            from fastapi import HTTPException
            retry_after = max(1, int(1.0 / refill_rate))
            raise HTTPException(status_code=429, detail="Too Many Requests", headers={"Retry-After": str(retry_after)})
        return dep

    def _rate_deps(self, times: int, seconds: int):
        if RATE_LIMIT_DISABLE:
            async def _noop(): return True
            return [Depends(_noop)]
        if REDIS_AVAILABLE:
            return [Depends(RateLimiter(times=times, seconds=seconds, identifier=lambda req: (req.headers.get("X-Session-Id") or req.client.host)))]
        return [Depends(self._make_memory_limiter(times, seconds, lambda req: (req.headers.get("X-Session-Id") or req.client.host)))]

    # -----------------------
    # Routes
    # -----------------------
    def _setup_routes(self):
        app = self.app

        basic_auth = HTTPBasic(auto_error=False)
        auth_bearer = HTTPBearer(auto_error=False)

        def verify_admin_basic(creds: HTTPBasicCredentials = Depends(basic_auth)):
            if not ADMIN_USERS:
                raise HTTPException(status_code=401, detail="Admin is not configured")
            if creds is None:
                raise HTTPException(status_code=401, detail="Missing credentials")
            username = creds.username or ""
            password = creds.password or ""
            expected = ADMIN_USERS.get(username)
            if not expected or not secrets.compare_digest(password, expected):
                raise HTTPException(status_code=401, detail="Incorrect username or password")
            return True

        def verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(auth_bearer)) -> str:
            if credentials is None or not credentials.credentials:
                raise HTTPException(status_code=401, detail="Missing token")
            token = credentials.credentials
            if token == "invalid":
                raise HTTPException(status_code=401, detail="Invalid Token")
            return token

        @app.get("/", include_in_schema=False)
        def root():
            index_file = STATIC_DIR / "main.html"
            if index_file.exists():
                return FileResponse(index_file)
            # fall back to docs if no main.html
            return JSONResponse({"ok": True, "docs": "/docs", "status": "/api/status"})

        @app.get("/api/status", response_model=SystemStatus)
        def status():
            qa_loaded = self.qa_retriever is not None
            total_pairs = None
            if qa_loaded and getattr(self.qa_retriever, "df", None) is not None:
                try:
                    total_pairs = int(self.qa_retriever.df.shape[0])
                except Exception:
                    total_pairs = None
            return SystemStatus(
                status="ok",
                qa_system_loaded=qa_loaded,
                total_items=len(self.items),
                total_qa_pairs=total_pairs,
                data_path=str(self.data_path),
                feedback_path=str(self.feedback_file),
                env=self.env,
                admin_link_enabled=SHOW_ADMIN_LINK,
            )

        @app.get("/api/runtime")
        def runtime_status():
            return {"ok": True, "llm_active": bool(self.llm_service), "mode": "dynamic_retrieval"}

        @app.get("/api/items", response_model=List[Item])
        def list_items():
            return self.items

        @app.post("/api/items", response_model=Item)
        def add_item(item: Item):
            self.items.append(item)
            return item

        @app.post("/api/query", response_model=QueryResponse, dependencies=self._rate_deps(times=10, seconds=60))
        async def query(payload: Query) -> QueryResponse:
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            start = time.time()
            QUERY_COUNT.inc()
            digest = base64.urlsafe_b64encode(hashlib.sha256(f"{payload.query}|{payload.top_k}".encode("utf-8")).digest())[:16].decode()
            cache_key = f"qa:query:{digest}"
            # Try redis cache if available
            try:
                if hasattr(self, "_redis"):
                    cached = await self._redis.get(cache_key)
                    if cached:
                        return QueryResponse(**json.loads(cached))
            except Exception:
                pass
            try:
                with QUERY_DURATION.time():
                    results = self.qa_retriever.retrieve(payload.query, top_k=payload.top_k or 3, user_profile=payload.user_profile)
            except Exception as e:
                logger.exception("Retrieval failed")
                raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")
            elapsed_ms = (time.time() - start) * 1000.0
            resp_items = [{"question": r.question, "answer": r.answer, "similarity": r.similarity_score, "index": r.index, "metadata": r.metadata} for r in results]
            max_score = max((r.similarity_score for r in results), default=0.0)
            conf_label = get_confidence_label(max_score)
            resp = QueryResponse(query=payload.query, results=resp_items, total_results=len(resp_items), query_time_ms=elapsed_ms, has_high=max_score >= 0.7, confidence=conf_label)
            try:
                if hasattr(self, "_redis"):
                    await self._redis.setex(cache_key, 300, resp.model_dump_json())
            except Exception:
                pass
            return resp

        @app.post("/api/feedback")
        def submit_feedback(fb: Feedback):
            record = fb.model_dump()
            record.setdefault("ts", time.time())
            try:
                if self.feedback_store is not None:
                    self.feedback_store.append(record)
                else:
                    self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.feedback_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.exception("Failed to write feedback")
                raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")
            return {"ok": True}

        @app.post("/api/retrain", dependencies=self._rate_deps(times=3, seconds=60))
        def retrain(_: bool = Depends(verify_admin_basic)):
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            try:
                self.qa_retriever.retrain_with_feedback(self.feedback_file)
                total_pairs = int(self.qa_retriever.df.shape[0]) if getattr(self.qa_retriever, "df", None) is not None else None
                return {"ok": True, "message": "Retrained on collected feedback.", "total_qa_pairs": total_pairs}
            except Exception as e:
                logger.exception("Retrain failed")
                raise HTTPException(status_code=500, detail=f"Retrain error: {e}")

        @app.post("/api/chat", response_model=ChatResponse, dependencies=self._rate_deps(times=20, seconds=60))
        def chat(payload: ChatRequest) -> ChatResponse:
            start = time.time()
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            try:
                k = payload.top_k or 3
                response_data = self.qa_retriever.retrieve_with_dynamic_response(payload.message, top_k=k, user_profile={"session_id": payload.session_id} if payload.session_id else None)
                if isinstance(response_data, list):
                    response_data = response_data[0] if response_data else {}
                answer = (response_data or {}).get("answer", "I couldn't find an answer.")
                conf = (response_data or {}).get("confidence", 0.0)
                results = (response_data or {}).get("results", [])
            except Exception as e:
                logger.exception("Chat Retrieval failed")
                raise HTTPException(status_code=500, detail=f"Chat Retrieval error: {e}")
            sources = [{"index": r.index, "question": r.question, "answer": r.answer, "similarity": r.similarity_score} for r in results]
            elapsed_ms = (time.time() - start) * 1000.0
            logger.info(f"/chat processed in {elapsed_ms:.1f} ms (top_k={k})")
            response = ChatResponse(message=payload.message, answer=answer, conf=conf, sources=sources, meta={"llm_active": False, "mode": "dynamic_retrieval"})
            try:
                append_conversation({"session_id": payload.session_id, "user_message": payload.message, "bot_answer": response.answer, "conf": response.conf, "timestamp": time.time()})
            except Exception:
                logger.debug("append_conversation failed/skipped", exc_info=True)
            return response

        @app.post("/api/chat-llm", response_model=ChatResponse, dependencies=self._rate_deps(times=15, seconds=60))
        def chat_with_llm(payload: ChatRequest) -> ChatResponse:
            start = time.time()
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            try:
                k = payload.top_k or 3
                results = self.qa_retriever.retrieve(payload.message, top_k=k, user_profile={"session_id": payload.session_id} if payload.session_id else None)
                max_score = max((r.similarity_score for r in results), default=0.0)
                response_data = self.qa_retriever.retrieve_with_dynamic_response(payload.message, top_k=k, user_profile={"session_id": payload.session_id} if payload.session_id else None)
                fallback_answer = response_data.get("answer", "I couldn't find an answer.")
                llm_result = None
                if self.llm_service and getattr(self.llm_service, "is_available", lambda: False)():
                    llm_result = self.llm_service.process_query(user_query=payload.message, retrieval_results=results, confidence_score=max_score, fallback_answer=fallback_answer)
                else:
                    llm_result = {"answer": fallback_answer, "mode": "retrieval_only", "llm_used": False, "confidence": max_score, "reason": "LLM service not available"}
                answer = llm_result.get("answer", fallback_answer)
                mode = llm_result.get("mode", "retrieval_only")
                llm_used = llm_result.get("llm_used", False)
            except Exception as e:
                logger.exception("Chat with LLM failed")
                raise HTTPException(status_code=500, detail=f"Chat error: {e}")
            sources = [{"index": r.index, "question": r.question, "answer": r.answer, "similarity": r.similarity_score} for r in results]
            elapsed_ms = (time.time() - start) * 1000.0
            logger.info(f"/chat-llm processed in {elapsed_ms:.1f} ms (top_k={k}, mode={mode}, llm={llm_used})")
            response = ChatResponse(message=payload.message, answer=answer, conf=max_score, sources=sources, meta={"llm_active": llm_used, "mode": mode, "confidence_label": get_confidence_label(max_score), "reason": llm_result.get("reason")})
            try:
                append_conversation({"session_id": payload.session_id, "user_message": payload.message, "bot_answer": response.answer, "conf": response.conf, "mode": mode, "llm_used": llm_used, "timestamp": time.time()})
            except Exception:
                logger.debug("append_conversation failed/skipped", exc_info=True)
            return response

        @app.get("/api/llm-status")
        def llm_status():
            if self.llm_service is None:
                return {
                    "available": False,
                    "enabled": False,
                    "model": None,
                    "provider": None,
                    "reason": "LLM service not initialized",
                }

            cfg = getattr(self.llm_service, "config", None)

            return {
                "available": bool(self.llm_service.is_available()),
                "enabled": bool(getattr(cfg, "enabled", False)),
                "model": getattr(cfg, "model", None),
                "provider": getattr(cfg, "provider", None),
                "confidence_threshold": getattr(cfg, "confidence_threshold", None),
                "max_tokens": getattr(cfg, "max_tokens", None),
            }
        @app.get("/api/conversations", response_model=List[ConversationSummary])
        def get_conversations(limit: int = 50):
            if not CHAT_LOG_FILE.exists():
                return []
            conversations: Dict[str, List[dict]] = {}
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    entry = json.loads(line)
                    sid = entry.get("session_id") or "default"
                    conversations.setdefault(sid, []).append(entry)
            summaries = []
            for sid, msgs in conversations.items():
                msgs.sort(key=lambda x: x.get("timestamp", 0))
                last_msg = msgs[-1]
                first_user_msg = next(((m.get("user_message") or "") for m in msgs if (m.get("user_message") or "")), "New Conversation")
                title = (first_user_msg[:50] + ("..." if len(first_user_msg) > 50 else ""))
                preview = (last_msg.get("bot_answer") or "")[:100]
                summaries.append(ConversationSummary(session_id=sid, title=title, message_count=len(msgs), last_message_time=last_msg.get("timestamp") or 0, preview=preview))
            summaries.sort(key=lambda x: x.last_message_time, reverse=True)
            return summaries[:limit]

        @app.get("/api/conversations/{session_id}", response_model=ConversationDetails)
        def get_conversation(session_id: str, limit: int = 100, offset: int = 0):
            if not CHAT_LOG_FILE.exists():
                return ConversationDetails(session_id=session_id, messages=[], total_messages=0)
            messages = []
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        messages.append(entry)
            messages.sort(key=lambda x: x.get("timestamp", 0))
            total = len(messages)
            return ConversationDetails(session_id=session_id, messages=messages[offset:offset+limit], total_messages=total)

        @app.delete("/api/conversations/{session_id}")
        def delete_conversation(session_id: str):
            if not CHAT_LOG_FILE.exists():
                return {"ok": True, "message": "No conversations to delete"}
            kept = []
            deleted = 0
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    entry = json.loads(line)
                    if entry.get("session_id") != session_id:
                        kept.append(line)
                    else:
                        deleted += 1
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(kept)
            return {"ok": True, "message": f"Deleted {deleted} messages from session {session_id}"}

        if METRICS_AVAILABLE:
            @app.get("/api/metrics")
            def metrics():
                return Response(generate_latest(), media_type="text/plain")

        @app.get("/healthz")
        def healthz():
            return {"ok": True}

        @app.get("/readyz")
        def readyz():
            return {"ready": self.qa_retriever is not None}

        @app.get("/api/admin/dashboard",response_model=AdminDashboardSummary)
        def admin_dashboard(_: bool = Depends(verify_admin_basic)):
            """Aggregate high-level stats for dev/admin dashboard."""
            now = time.time()
            last_24h_cutoff = now - 24 * 3600

            # --- Conversations & traffic ---
            total_conversations = 0
            total_messages = 0
            last_24h_messages = 0
            first_ts = None
            last_ts = None

            if CHAT_LOG_FILE.exists():
                conv_sessions: Dict[str, int] = {}
                with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue
                        sid = entry.get("session_id") or "default"
                        ts = float(entry.get("timestamp") or 0.0)

                        conv_sessions[sid] = conv_sessions.get(sid, 0) + 1
                        total_messages += 1

                        if ts > 0:
                            if first_ts is None or ts < first_ts:
                                first_ts = ts
                            if last_ts is None or ts > last_ts:
                                last_ts = ts
                            if ts >= last_24h_cutoff:
                                last_24h_messages += 1

                total_conversations = len(conv_sessions)

            # --- QA system status ---
            qa_loaded = self.qa_retriever is not None
            total_qa_pairs = None
            if qa_loaded and getattr(self.qa_retriever, "df", None) is not None:
                try:
                    total_qa_pairs = int(self.qa_retriever.df.shape[0])
                except Exception:
                    total_qa_pairs = None

            # --- LLM status ---
            llm_available = bool(self.llm_service and getattr(self.llm_service, "is_available", lambda: False)())
            cfg = getattr(self.llm_service, "config", None) if self.llm_service else None

            llm_enabled = bool(getattr(cfg, "enabled", False)) if cfg else False
            llm_model = getattr(cfg, "model", None) if cfg else None
            llm_provider = getattr(cfg, "provider", None) if cfg else None

            # --- Feedback stats ---
            feedback_total = feedback_up = feedback_down = feedback_neutral = 0
            if self.feedback_file.exists():
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            fb = json.loads(line)
                        except Exception:
                            continue
                        feedback_total += 1
                        label = (fb.get("label") or "neutral").lower()
                        if label == "up":
                            feedback_up += 1
                        elif label == "down":
                            feedback_down += 1
                        else:
                            feedback_neutral += 1

            return AdminDashboardSummary(
                total_conversations=total_conversations,
                total_messages=total_messages,
                last_24h_messages=last_24h_messages,
                first_message_time=first_ts,
                last_message_time=last_ts,
                qa_system_loaded=qa_loaded,
                total_qa_pairs=total_qa_pairs,
                llm_available=llm_available,
                llm_enabled=llm_enabled,
                llm_model=llm_model,
                llm_provider=llm_provider,
                feedback_total=feedback_total,
                feedback_up=feedback_up,
                feedback_down=feedback_down,
                feedback_neutral=feedback_neutral,
                env=self.env,
            )

    def get_app(self) -> FastAPI:
        return self.app

# Export app
app = API().app

# If run as module, uvicorn will import app variable
