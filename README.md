# NexusAI — Enterprise Agentic RAG Platform

NexusAI is an internal **company knowledge assistant** that combines **Retrieval-Augmented Generation (RAG)** over policy documents with an **agentic MCP tool layer** for live HR employee data. Employees log in with their Employee ID; answers are scoped by **role-based access control (RBAC)** at retrieval time and again at output.

Built for a fintech-style org with departments (HR, Finance, Engineering, Marketing, Operations) and ~100 employees in MongoDB.

---

## Highlights

| Capability | Description |
|------------|-------------|
| **Hybrid intelligence** | Policy/docs → Qdrant RAG; live employee records → MongoDB via MCP tools |
| **RBAC at vector DB** | Qdrant payload filter on `allowed_roles` — users never retrieve unauthorized chunks |
| **Agentic HR tools** | LLM selects `hr_get_employee`, `hr_update_employee`, or `hr_list_employees` (MCP-style JSON tool calls) |
| **Guardrails** | Prompt-injection blocking, out-of-scope detection, Presidio PII awareness, output scrubbing |
| **Multi-turn chat** | In-memory session history (last ~10 exchanges per user) |
| **Streaming** | SSE endpoint for token-by-token UI (`/chat/stream`) |
| **Observability** | LangSmith tracing on pipeline steps; MongoDB query logs for Ragas evaluation |

---

**Request flow (simplified)**

1. User asks a question → JWT validated → input guardrails.
2. **Intent detection**: `hr_read` / `hr_write` → MCP agent; else → RAG.
3. **RAG path**: query rewrite → retrieve (RBAC filter) → [rerank on `/stream`] → aggregate if summary → single LLM answer.
4. **MCP path**: LLM picks tool + params → RBAC check → MongoDB CRUD → LLM formats natural-language reply.
5. Answer scrubbed, logged to MongoDB, returned with optional source citations.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn, Pydantic Settings |
| Frontend | Streamlit |
| Orchestration | LlamaIndex |
| Vector DB | Qdrant Cloud |
| Operational DB | MongoDB Atlas (employees, chat logs) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, local) |
| LLM | Groq (`llama3-8b-8192`) or Gemini (configurable) |
| Reranker | `BAAI/bge-reranker-base` (streaming path) |
| Auth | JWT (python-jose), bcrypt passwords |
| Guardrails | Regex patterns + Microsoft Presidio |
| Evaluation | Ragas (faithfulness, answer relevancy, context recall) |
| Tracing | LangSmith |

---

## Project Structure

```
finrag/
├── backend/
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # Settings from .env
│   ├── auth/                   # Login, JWT, department → role
│   ├── chat/
│   │   ├── rag_pipeline.py     # RAG v3 + intent routing
│   │   ├── routes.py           # /chat/query, /chat/stream, feedback
│   │   ├── guardrails.py       # Input/output safety
│   │   ├── rbac.py             # Role → Qdrant filter
│   │   └── llm_provider.py     # Groq / Gemini switch
│   ├── mcp/
│   │   ├── hr_tools.py         # MCP tool definitions + permissions
│   │   ├── hr_db.py            # MongoDB CRUD for employees
│   │   └── tool_executor.py    # Agent: tool selection + execution
│   └── db/mongodb.py           # Connection singleton
├── frontend/
│   ├── app.py                  # Login page
│   └── pages/chat.py           # Chat UI (streaming + feedback)
├── ingestion/
│   ├── ingest.py               # Chunk + embed → Qdrant
│   ├── migrate_hr_to_mongo.py    # CSV → MongoDB employees
│   └── setup_employee_auth.py  # bcrypt passwords on employee records
├── data/                       # Source documents by department
│   ├── hr/                     # hr_data.csv + policies
│   ├── finance/
│   ├── engineering/
│   ├── marketing/
│   └── general/
├── evals/ragas_eval.py         # Offline quality metrics
├── requirements.txt
└── .env.example
```

---

## Roles & Access

Roles are **derived from department** at login (see `backend/auth/utils.py`), unless `is_clevel` is set on the employee record.

| Role | Typical departments | Document access |
|------|---------------------|-----------------|
| **HR** | hr, human resources | HR + general |
| **FINANCE** | finance, accounts | Finance + general |
| **ENGINEERING** | technology, data, product, design | Engineering + general |
| **MARKETING** | marketing, sales, business | Marketing + general |
| **GENERAL** | operations, compliance, risk, QA, legal, … | General only |
| **C_LEVEL** | `is_clevel=true` | All departments |

**MCP tools** (`hr_get_employee`, `hr_update_employee`, `hr_list_employees`) are restricted to **HR** and **C_LEVEL** only.

---

## MCP Tools (HR Agent)

| Tool | Purpose |
|------|---------|
| `hr_get_employee` | Lookup by name, employee_id, or department; optional single field |
| `hr_update_employee` | Update one field (e.g. `leave_balance`, `leaves_taken`, `status`) |
| `hr_list_employees` | Filtered list by department, role, location, status |

The LLM receives tool schemas in the prompt and returns JSON like:

```json
{"tool": "hr_update_employee", "parameters": {"employee_id": "FINEMP1009", "field": "leave_balance", "value": "7"}}
```

**Updatable fields** (HR): `leave_balance`, `leaves_taken`, `phone`, `email`, `location`, `manager_id`, `emergency_contact`, `date_of_joining`, `status`. Salary and org structure fields require C_LEVEL.

---

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB Atlas URI
- Qdrant Cloud URL + API key
- Groq API key (or Gemini)
- Optional: LangSmith API key

### 1. Clone and install

```bash
git clone <your-repo-url>
cd finrag
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys (JWT_SECRET_KEY must be 32+ chars)
```

Key variables: `MONGO_URI`, `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`, `JWT_SECRET_KEY`, `QDRANT_COLLECTION`, `MONGO_DB_NAME`.

### 3. Ingest documents into Qdrant

Place markdown/CSV under `data/<department>/`, then:

```bash
python -m ingestion.ingest
```

This chunks text (512 tokens, 50 overlap), embeds with BGE-small, and upserts to Qdrant with `allowed_roles` metadata.

### 4. Load employees into MongoDB

```bash
python -m ingestion.migrate_hr_to_mongo
python -m ingestion.setup_employee_auth
```

Default password pattern: `EMP` + last 4 digits of employee ID + `@Nexus`  
Example: `FINEMP1012` → `EMP1012@Nexus`

### 5. Run backend

```bash
uvicorn backend.main:app --reload --port 5050
```

Swagger UI: http://localhost:5050/docs

### 6. Run frontend

```bash
streamlit run frontend/app.py
```

Set `API_BASE_URL=http://localhost:5050` in `.env` if needed.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/auth/login` | Employee ID + password → JWT |
| `GET` | `/auth/me?token=` | Decode JWT payload |
| `POST` | `/chat/query` | Full JSON response (non-streaming) |
| `POST` | `/chat/stream` | SSE token stream |
| `PATCH` | `/chat/feedback/{log_id}?feedback=positive\|negative` | Thumbs up/down |
| `DELETE` | `/chat/session` | Clear server-side chat memory |

**Authenticated requests** require header:

```
Authorization: Bearer <access_token>
```

---

## RAG Pipeline (v3) — Design Notes

- **Intent types**: `hr_read`, `hr_write`, `summary`, `entity`, `default`
- **Query rewrite**: LLM condenses conversational questions to search keywords (skipped for `summary`)
- **RBAC filter**: Qdrant `allowed_roles` must match user role
- **Summary queries**: extra retrieval (`top_k=10`) + aggregation LLM step before final answer
- **Single final LLM call**: avoids LlamaIndex `ContextChatEngine` double-retrieval latency
- **Greetings / acknowledgements**: short-circuited without retrieval

> **Note:** Cross-encoder reranking is applied on the **`/chat/stream`** path. The non-streaming `/chat/query` path uses vector similarity scores directly.

---

## Guardrails

**Input** (`check_input`):

- Prompt injection patterns (ignore instructions, jailbreak, etc.)
- Out-of-scope (jokes, weather, math trivia)
- Unsupported external actions (email, Jira, calendar) — *HR updates via MCP are routed separately*
- Sensitive keyword blocking for ENGINEERING/MARKETING roles

**Output** (`scrub_output`):

- Presidio-based PII redaction where configured

---

## Evaluation

Chat interactions are logged to MongoDB `chat_logs`. Run Ragas offline:

```bash
python -m evals.ragas_eval
python -m evals.ragas_eval --limit 20
```

Metrics: **faithfulness**, **answer relevancy**, **context recall**.

---

## Development Tips

- **LangSmith**: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to trace `query_rewrite`, `chunk_aggregation`, and full pipeline runs.
- **Qdrant index**: if filtering fails, run `python -m ingestion.fix_qdrant_index`.
- **HR policy vs employee data**: “What is leave policy?” → RAG; “What is Isha’s leave balance?” → MCP.
- **Streaming + MCP**: `/chat/stream` routes `hr_read`/`hr_write` to the MCP agent (same as `/chat/query`).

---

## Security Considerations (Production)

- Restrict CORS origins (currently `*` in `main.py`)
- Rotate `JWT_SECRET_KEY`; use HTTPS in production
- Salary redacted at ingestion (Qdrant) and in MongoDB responses for non–C-Level
- Rate-limit auth and chat endpoints
- Move session store to Redis for multi-instance deployments

---

## License
This project is intended for educational, research, and portfolio demonstration purposes only.

---

## Author

Built as an enterprise-style RAG + agentic HR assistant demonstration project.
