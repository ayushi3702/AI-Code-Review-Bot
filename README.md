# AI Code Review Platform

A multi-agent **full-repository** code review platform. Point it at a GitHub URL
or a local folder and four specialist agents — **security, performance,
architecture, and code-quality** — scan the *entire* codebase in parallel and
produce a rated, categorised report (Markdown **and** HTML) with a 0–100 health
score and an A–F grade.

Unlike a PR-comment bot that only sees diffs, this analyses the whole repo, so it
can catch cross-file issues a diff can never see: duplicated logic across modules,
dead code, tangled imports, and inconsistent patterns. To stay cheap and fast on
large repos, it **embeds the code into a vector store (ChromaDB)** and each agent
retrieves only the chunks relevant to its concern instead of dumping every file
into the prompt.

> A legacy **PR-diff mode** (GitHub App + GitHub Actions) is still included — see
> [PR mode](#legacy-pr-diff-mode) at the bottom.

## How it works

```
repo URL / local path
        │
        ▼
  Crawler  ── clone or walk, skip node_modules/.git/binaries/lockfiles
        │
        ▼
  Chunker  ── split by function/class boundaries (AST for Python, regex elsewhere)
        │
        ▼
 Embeddings ── Azure OpenAI embeddings → ChromaDB collection (per scan)
        │
        ▼
   LangGraph orchestrator  (fan-out / fan-in)
        │
   ┌────┼────────┬───────────────┐
   ▼    ▼        ▼               ▼
Security Performance Architecture Quality   ← run in parallel, each retrieves
   │       │          │            │           only its relevant chunks
   └───────┴──────────┴────────────┘
        │
        ▼
   Report agent ── dedupe · score (0–100, A–F) · Markdown + HTML
```

## Tech stack

- **Orchestration**: LangGraph (fan-out/fan-in over the four agents)
- **LLM + embeddings**: **Azure OpenAI** (`AzureChatOpenAI` + `AzureOpenAIEmbeddings`)
- **Retrieval**: ChromaDB vector store (one collection per scan)
- **Languages**: Python, JavaScript/TypeScript, Go, Java, Ruby, PHP, C#, Rust, C/C++, Kotlin, Swift, Scala, SQL, Shell
- **Reports**: Jinja2 (HTML) + Markdown, with health score & grade
- **CLI**: `rich`-powered terminal output
- **API**: FastAPI (background scans + polling)
- **Frontend**: React + Vite
- **Database**: SQLite via SQLAlchemy (scan + finding audit trail)

## Quick start (deep-scan mode)

```bash
pip install -r requirements.txt

cp .env.example .env
# Fill in AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
#   AZURE_OPENAI_DEPLOYMENT (chat), AZURE_OPENAI_EMBEDDING_DEPLOYMENT (embeddings)
```

### 1. CLI

```bash
# Scan a remote repo
python cli.py scan https://github.com/pallets/flask.git --open

# Scan the current project and write reports to ./out
python cli.py scan . --out out
```

Reports are written to `reports/<repo>-<timestamp>.md` and `.html`. The CLI exits
with code `2` if any **high**-severity issue is found, so it can gate CI.

### 2. Web app (API + React UI)

```bash
# Terminal 1 — backend
uvicorn api.server:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev      # http://localhost:5173
```

Paste a repo URL into the UI, watch the live progress (crawl → embed → analyze →
report), then read the scored findings or open the full HTML report.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/scan` | Start a scan `{ "repo_source": "<url\|path>" }` → `{ scan_id }` |
| `GET` | `/api/scan/{id}` | Status, stage, score, grade, findings |
| `GET` | `/api/scan/{id}/report.html` | Standalone HTML report |
| `GET` | `/api/scans` | Recent scans |

## Why embeddings instead of stuffing files into the prompt?

A 100+ file repo blows past any context window. Instead, every file is chunked on
semantic boundaries and embedded into ChromaDB; each agent issues concern-specific
queries (the security agent searches for "authentication", "SQL query", "secret";
the performance agent for "loop", "N+1", "blocking I/O") and only ever sees a
relevant subset. Cost stays roughly flat as the repo grows.

## Project structure

```
.
├── cli.py                       # `python cli.py scan <url|path>`
├── scanner/
│   ├── crawler.py               # clone/walk + exclusion rules + language detection
│   ├── chunker.py               # function/class-boundary chunking (multi-language)
│   └── vector_store.py          # ChromaDB + Azure embeddings wrapper
├── agents/
│   ├── repo_orchestrator.py     # crawl→index→fan-out→report (LangGraph)
│   ├── repo_security_agent.py
│   ├── repo_performance_agent.py
│   ├── repo_architecture_agent.py   # cross-file: duplication, dead code, coupling
│   ├── repo_quality_agent.py
│   ├── repo_report_agent.py     # dedupe + score + Markdown/HTML
│   └── _scan_common.py          # shared retrieve→prompt→parse helper
├── core/
│   ├── config.py                # Azure chat/embeddings factories + tuning
│   ├── state.py                 # ScanState / ScanFinding models
│   └── database.py              # SQLAlchemy models (Scan, ScanFinding, AgentRun)
├── api/
│   └── server.py                # FastAPI deep-scan API
├── frontend/                    # React + Vite UI
└── tests/
```

## Tests

```bash
pytest tests/test_scanner.py -v   # offline: crawler + chunker (no Azure needed)
pytest tests/ -v                  # full suite
```

