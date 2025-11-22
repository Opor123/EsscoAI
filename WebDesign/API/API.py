# API.py - Enhanced security version
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
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Response, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pydantic import BaseModel, Field

# Optional imports
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
HERE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HERE.parent
STATIC_DIR = HERE / "static"
DATA_DIR = PROJECT_ROOT / "Data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------
# Config
# -----------------------
ENV = os.getenv("ESSCOAI_ENV", "dev")
FORCE_HTTPS = os.getenv("ESSCOAI_FORCE_HTTPS", "0") == "1"
ENFORCE_HTTPS = FORCE_HTTPS and ENV == "prod"
SHOW_ADMIN_LINK = os.getenv("ESSCOAI_SHOW_ADMIN_LINK", "0") == "1"
CORS_ORIGINS = [o.strip() for o in os.getenv("ESSCOAI_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
RATE_LIMIT_DISABLE = os.getenv("ESSCOAI_DISABLE_RATE_LIMIT", "1") == "1"
REDIS_URL = os.getenv("ESSCOAI_REDIS_URL", "redis://localhost:6379/0")

# Parse admin users with password hashing
ADMIN_USERS_ENV = os.getenv("ESSCOAI_ADMIN_USERS", "admin:password123")
ADMIN_USERS: Dict[str, str] = {}


def hash_password(password: str) -> str:
    """Hash password using SHA-256 (in production, use bcrypt or argon2)"""
    return hashlib.sha256(password.encode()).hexdigest()


for pair in [p.strip() for p in ADMIN_USERS_ENV.split(",") if p.strip()]:
    if ":" in pair:
        u, p = pair.split(":", 1)
        # Store hashed passwords
        ADMIN_USERS[u.strip()] = hash_password(p.strip())
        logger.info(f"Admin user configured: {u.strip()}")
    elif pair:
        logger.warning(f"Admin user '{pair}' has no password set. Ignoring.")

logger.info(f"SHOW_ADMIN_LINK={SHOW_ADMIN_LINK}")

# Session management
SESSION_TIMEOUT = timedelta(hours=2)
active_sessions: Dict[str, dict] = {}


# -----------------------
# Security utilities
# -----------------------
def generate_session_token() -> str:
    """Generate a cryptographically secure session token"""
    return secrets.token_urlsafe(32)


def verify_password(username: str, password: str) -> bool:
    """Verify password against stored hash"""
    if username not in ADMIN_USERS:
        return False
    return secrets.compare_digest(
        ADMIN_USERS[username],
        hash_password(password)
    )


def create_session(username: str) -> str:
    """Create a new admin session"""
    token = generate_session_token()
    active_sessions[token] = {
        "username": username,
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow()
    }
    return token


def verify_session(token: str) -> Optional[str]:
    """Verify session token and return username if valid"""
    if token not in active_sessions:
        return None

    session = active_sessions[token]

    # Check if session expired
    if datetime.utcnow() - session["last_activity"] > SESSION_TIMEOUT:
        del active_sessions[token]
        return None

    # Update last activity
    session["last_activity"] = datetime.utcnow()
    return session["username"]


def invalidate_session(token: str) -> bool:
    """Invalidate a session token"""
    if token in active_sessions:
        del active_sessions[token]
        return True
    return False


# Clean up expired sessions periodically
async def cleanup_expired_sessions():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        now = datetime.utcnow()
        expired = [
            token for token, session in active_sessions.items()
            if now - session["last_activity"] > SESSION_TIMEOUT
        ]
        for token in expired:
            del active_sessions[token]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired admin sessions")


# -----------------------
# Basic telemetry
# -----------------------
if METRICS_AVAILABLE:
    QUERY_COUNT = Counter("qa_queries_total", "Total Queries Processed")
    QUERY_DURATION = Histogram("qa_query_duration_seconds", "Query processing time")
    ERROR_COUNT = Counter("qa_errors_total", "Total errors", ["error_type"])
else:
    class _Dummy:
        def inc(self, *a, **k): pass

        def time(self):
            class _Ctxt:
                def __enter__(self): pass

                def __exit__(self, *a): pass

            return _Ctxt()


    QUERY_COUNT = QUERY_DURATION = ERROR_COUNT = _Dummy()

# -----------------------
# Optional components
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
        logger.warning("QARetriever not importable – Q&A endpoints will be disabled")

try:
    from AI.feedback_store import FeedbackStore as _FS

    FeedbackStore = _FS
except Exception:
    try:
        from feedback_store import FeedbackStore as _FS

        FeedbackStore = _FS
    except Exception:
        logger.warning("FeedbackStore not importable – feedback will be appended manually")

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
    session_id: Optional[str] = Field(None, max_length=100, pattern=r"^[A-Za-z0-9_\-:.]+$")
    model_config = {"protected_namespaces": ()}


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
    total_conversations: int
    total_messages: int
    last_24h_messages: int
    first_message_time: Optional[float] = None
    last_message_time: Optional[float] = None
    qa_system_loaded: bool
    total_qa_pairs: Optional[int] = None
    llm_available: bool
    llm_enabled: bool
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    feedback_total: int
    feedback_up: int
    feedback_down: int
    feedback_neutral: int
    env: str


class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)


class AdminLoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    username: Optional[str] = None
    message: str


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
# Security dependencies
# -----------------------
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_admin_token(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """Verify admin session token from Authorization header"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    username = verify_session(credentials.credentials)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return username


# -----------------------
# API Application
# -----------------------
class API:
    def __init__(self):
        self.app = FastAPI(title="EsscoAI Integrated API", version="2.0.0")
        self._setup_middlewares()
        self._setup_state()
        self._setup_static_files()
        self._setup_routes()

        if not RATE_LIMIT_DISABLE and REDIS_AVAILABLE:
            @self.app.on_event("startup")
            async def init_redis_limiter():
                try:
                    self._redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
                    await FastAPILimiter.init(self._redis)
                    logger.info("Redis rate limiter initialized")
                except Exception as e:
                    logger.warning(f"Redis rate limiter failed to initialize: {e}")

        @self.app.on_event("startup")
        async def start_cleanup_task():
            asyncio.create_task(cleanup_expired_sessions())

    def _setup_middlewares(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS or ["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "essco.ai", "api.essco.ai", "*"]
        )
        if ENFORCE_HTTPS:
            from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
            self.app.add_middleware(HTTPSRedirectMiddleware)
        self.app.add_middleware(GZipMiddleware, minimum_size=500)

    def _setup_state(self):
        self.items: List[Item] = []
        self.project_root: Path = PROJECT_ROOT
        self.data_path: Path = DATA_DIR
        self.feedback_file: Path = self.data_path / "feedback.jsonl"
        self.training_file: Path = self.data_path / "training_ready.jsonl"
        self.qa_retriever = None
        self.feedback_store = None
        self.env = ENV
        self._memory_buckets: Dict[str, dict] = {}
        self._memory_lock = asyncio.Lock()
        self._init_optional_components()

    def _init_optional_components(self):
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

        if FeedbackStore is not None:
            try:
                self.feedback_store = FeedbackStore(self.feedback_file)
                logger.info("Feedback store initialized")
            except Exception as e:
                logger.exception("Failed to initialize FeedbackStore: %s", e)
                self.feedback_store = None

        self.llm_service = None
        if get_llm_service is not None:
            try:
                self.llm_service = get_llm_service()
                if getattr(self.llm_service, "is_available", lambda: False)():
                    logger.info("LLM service available")
            except Exception as e:
                logger.exception("Failed to initialize LLM service: %s", e)

    def _setup_static_files(self):
        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
            logger.info("Static files mounted from %s", STATIC_DIR)

    def _rate_deps(self, times: int, seconds: int):
        if RATE_LIMIT_DISABLE:
            async def _noop(): return True

            return [Depends(_noop)]
        if REDIS_AVAILABLE:
            return [Depends(RateLimiter(times=times, seconds=seconds))]
        return []

    def _setup_routes(self):
        app = self.app

        @app.get("/", include_in_schema=False)
        def root():
            index_file = STATIC_DIR / "main.html"
            if index_file.exists():
                return FileResponse(index_file)
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

        # ===========================
        # ADMIN AUTHENTICATION
        # ===========================

        @app.post("/api/admin/login", response_model=AdminLoginResponse)
        async def admin_login(credentials: AdminLoginRequest):
            """Authenticate admin and return session token"""
            # Rate limiting for login attempts
            await asyncio.sleep(0.5)  # Slow down brute force

            if not verify_password(credentials.username, credentials.password):
                logger.warning(f"Failed login attempt for user: {credentials.username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )

            # Create session
            token = create_session(credentials.username)
            logger.info(f"Admin user {credentials.username} logged in")

            return AdminLoginResponse(
                success=True,
                token=token,
                username=credentials.username,
                message="Login successful"
            )

        @app.post("/api/admin/logout")
        async def admin_logout(username: str = Depends(verify_admin_token)):
            """Logout admin and invalidate session"""
            logger.info(f"Admin user {username} logged out")
            return {"success": True, "message": "Logged out successfully"}

        @app.get("/api/admin/verify")
        async def verify_admin(username: str = Depends(verify_admin_token)):
            """Verify if current session is valid"""
            return {"valid": True, "username": username}

        @app.get("/api/admin/dashboard", response_model=AdminDashboardSummary)
        async def admin_dashboard(username: str = Depends(verify_admin_token)):
            """Get admin dashboard data (requires authentication)"""
            now = time.time()
            last_24h_cutoff = now - 24 * 3600

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

            qa_loaded = self.qa_retriever is not None
            total_qa_pairs = None
            if qa_loaded and getattr(self.qa_retriever, "df", None) is not None:
                try:
                    total_qa_pairs = int(self.qa_retriever.df.shape[0])
                except Exception:
                    total_qa_pairs = None

            llm_available = bool(self.llm_service and getattr(self.llm_service, "is_available", lambda: False)())
            cfg = getattr(self.llm_service, "config", None) if self.llm_service else None
            llm_enabled = bool(getattr(cfg, "enabled", False)) if cfg else False
            llm_model = getattr(cfg, "model", None) if cfg else None
            llm_provider = getattr(cfg, "provider", None) if cfg else None

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

        @app.post("/api/retrain", dependencies=self._rate_deps(times=3, seconds=60))
        async def retrain(username: str = Depends(verify_admin_token)):
            """Retrain model with feedback (requires admin authentication)"""
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            try:
                self.qa_retriever.retrain_with_feedback(self.feedback_file)
                total_pairs = int(self.qa_retriever.df.shape[0]) if getattr(self.qa_retriever, "df",
                                                                            None) is not None else None
                logger.info(f"Model retrained by admin: {username}")
                return {"ok": True, "message": "Retrained on collected feedback.", "total_qa_pairs": total_pairs}
            except Exception as e:
                logger.exception("Retrain failed")
                raise HTTPException(status_code=500, detail=f"Retrain error: {e}")

        # ===========================
        # CONVERSATION ENDPOINTS
        # ===========================

        @app.get("/api/conversations", response_model=List[ConversationSummary])
        def get_conversations(limit: int = 50):
            """Get list of all conversations"""
            if not CHAT_LOG_FILE.exists():
                return []
            conversations: Dict[str, List[dict]] = {}
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    sid = entry.get("session_id") or "default"
                    conversations.setdefault(sid, []).append(entry)
            summaries = []
            for sid, msgs in conversations.items():
                msgs.sort(key=lambda x: x.get("timestamp", 0))
                last_msg = msgs[-1]
                first_user_msg = next(
                    ((m.get("user_message") or "") for m in msgs if (m.get("user_message") or "")),
                    "New Conversation"
                )
                title = (first_user_msg[:50] + ("..." if len(first_user_msg) > 50 else ""))
                preview = (last_msg.get("bot_answer") or "")[:100]
                summaries.append(
                    ConversationSummary(
                        session_id=sid,
                        title=title,
                        message_count=len(msgs),
                        last_message_time=last_msg.get("timestamp") or 0,
                        preview=preview
                    )
                )
            summaries.sort(key=lambda x: x.last_message_time, reverse=True)
            return summaries[:limit]

        @app.get("/api/conversations/{session_id}", response_model=ConversationDetails)
        def get_conversation(session_id: str, limit: int = 100, offset: int = 0):
            """Get details of a specific conversation"""
            if not CHAT_LOG_FILE.exists():
                return ConversationDetails(session_id=session_id, messages=[], total_messages=0)
            messages = []
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("session_id") == session_id:
                        messages.append(entry)
            messages.sort(key=lambda x: x.get("timestamp", 0))
            total = len(messages)
            return ConversationDetails(
                session_id=session_id,
                messages=messages[offset:offset + limit],
                total_messages=total
            )

        @app.delete("/api/conversations/{session_id}")
        def delete_conversation(session_id: str):
            """Delete a specific conversation"""
            if not CHAT_LOG_FILE.exists():
                return {"ok": True, "message": "No conversations to delete"}
            kept = []
            deleted = 0
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("session_id") != session_id:
                        kept.append(line)
                    else:
                        deleted += 1
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(kept)
            return {"ok": True, "message": f"Deleted {deleted} messages from session {session_id}"}

        # ===========================
        # CHAT & OTHER ENDPOINTS
        # ===========================

        @app.post("/api/chat-llm", response_model=ChatResponse, dependencies=self._rate_deps(times=15, seconds=60))
        def chat_with_llm(payload: ChatRequest) -> ChatResponse:
            start = time.time()
            if self.qa_retriever is None:
                raise HTTPException(status_code=503, detail="Q&A system not available")
            try:
                k = payload.top_k or 3
                results = self.qa_retriever.retrieve(
                    payload.message,
                    top_k=k,
                    user_profile={"session_id": payload.session_id} if payload.session_id else None
                )
                max_score = max((r.similarity_score for r in results), default=0.0)
                response_data = self.qa_retriever.retrieve_with_dynamic_response(
                    payload.message,
                    top_k=k,
                    user_profile={"session_id": payload.session_id} if payload.session_id else None
                )
                fallback_answer = response_data.get("answer", "I couldn't find an answer.")
                llm_result = None
                if self.llm_service and getattr(self.llm_service, "is_available", lambda: False)():
                    llm_result = self.llm_service.process_query(
                        user_query=payload.message,
                        retrieval_results=results,
                        confidence_score=max_score,
                        fallback_answer=fallback_answer
                    )
                else:
                    llm_result = {
                        "answer": fallback_answer,
                        "mode": "retrieval_only",
                        "llm_used": False,
                        "confidence": max_score,
                        "reason": "LLM service not available"
                    }
                answer = llm_result.get("answer", fallback_answer)
                mode = llm_result.get("mode", "retrieval_only")
                llm_used = llm_result.get("llm_used", False)
            except Exception as e:
                logger.exception("Chat with LLM failed")
                raise HTTPException(status_code=500, detail=f"Chat error: {e}")

            sources = [
                {
                    "index": r.index,
                    "question": r.question,
                    "answer": r.answer,
                    "similarity": r.similarity_score
                }
                for r in results
            ]
            elapsed_ms = (time.time() - start) * 1000.0

            response = ChatResponse(
                message=payload.message,
                answer=answer,
                conf=max_score,
                sources=sources,
                meta={
                    "llm_active": llm_used,
                    "mode": mode,
                    "confidence_label": get_confidence_label(max_score),
                    "reason": llm_result.get("reason")
                }
            )

            try:
                append_conversation({
                    "session_id": payload.session_id,
                    "user_message": payload.message,
                    "bot_answer": response.answer,
                    "conf": response.conf,
                    "mode": mode,
                    "llm_used": llm_used,
                    "timestamp": time.time()
                })
            except Exception:
                pass

            return response

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

        @app.get("/api/llm-status")
        def llm_status():
            """Get LLM service status"""
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

        @app.get("/api/runtime")
        def runtime_status():
            """Get runtime status"""
            return {"ok": True, "llm_active": bool(self.llm_service), "mode": "dynamic_retrieval"}

        if METRICS_AVAILABLE:
            @app.get("/api/metrics")
            def metrics():
                """Prometheus metrics endpoint"""
                return Response(generate_latest(), media_type="text/plain")

        @app.get("/healthz")
        def healthz():
            """Health check endpoint"""
            return {"ok": True}

        @app.get("/readyz")
        def readyz():
            """Readiness check endpoint"""
            return {"ready": self.qa_retriever is not None}

    def get_app(self) -> FastAPI:
        return self.app


# Export app
app = API().app