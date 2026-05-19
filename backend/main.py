"""
backend/main.py

FastAPI application entry point.
Run with: uvicorn backend.main:app --reload --port 8000
Swagger UI: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth.routes import router as auth_router
from backend.chat.routes import router as chat_router
from backend.chat.rag_pipeline import setup_langsmith
from backend.chat.guardrails import _get_presidio
from backend.db.mongodb import get_db, close_db
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    startup:  connect to MongoDB, setup LangSmith
    shutdown: close MongoDB connection cleanly
    """
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Starting NexusAI backend...")
    get_db()            # verify MongoDB connection on startup
    setup_langsmith()   # enable LangSmith tracing if configured
    _get_presidio()     # warm up Presidio PII engine on startup
    logger.info("NexusAI backend ready ✓")

    yield  # app runs here

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down NexusAI backend...")
    close_db()
    logger.info("Shutdown complete.")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexusAI — Enterprise Agentic RAG Platform",
    description="""
## NexusAI Internal Knowledge Assistant

A production-grade agentic RAG system with:
- **Role-Based Access Control (RBAC)** — enforced at Qdrant filter level
- **Employee-based Auth** — login with Employee ID, role derived from department
- **MCP Tool Integration** — HR agent can query/update employee records
- **Guardrails** — Presidio PII detection + prompt injection blocking
- **Multi-turn memory** — follow-up questions work within a session
- **Streaming responses** — tokens stream via SSE

### Roles (derived from department)
| Role | Departments | Access |
|------|------------|--------|
| HR | hr | HR docs + General |
| FINANCE | finance | Finance docs + General |
| ENGINEERING | technology, data, product, design | Engineering docs + General |
| MARKETING | marketing, sales, business | Marketing docs + General |
| GENERAL | operations, compliance, risk, QA | General docs only |
| C_LEVEL | (is_clevel flag) | All documents |

### How to use
1. **Login** via `POST /auth/login` with employee_id + password
2. Click **Authorize** and enter: `Bearer <your_token>`
3. Use `POST /chat/query` or `POST /chat/stream` to ask questions
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
# Allows Streamlit frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    """Quick health check — use this to verify server is running."""
    return {
        "status": "ok",
        "app": "NexusAI",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
    }