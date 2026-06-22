# AI Code Review Platform

A multi-agent **full-repository** code review platform. Point it at a GitHub
repository URL and four specialist agents — **security, performance,
architecture, and code-quality** — scan the *entire* codebase in parallel and
produce a rated, categorised report (Markdown **and** HTML) with a 0–100 health
score and an A–F grade.

Unlike a PR-comment bot that only sees diffs, this analyses the whole repo, so it
can catch cross-file issues a diff can never see: duplicated logic across modules,
dead code, tangled imports, and inconsistent patterns. To stay cheap and fast on
large repos, it **embeds the code into a vector store (ChromaDB)** and each agent
retrieves only the chunks relevant to its concern instead of dumping every file
into the prompt.

It doesn't stop at reporting: from the web UI you can **generate a concrete fix**
for a finding, preview it as a diff, select any number of fixes, and **commit
them together** — with safety gates that block merge conflicts and code-breaking
changes. Sign in with **GitHub** to commit and push (or open a pull request) on
remote repositories you have access to.

> A legacy **PR-diff mode** (GitHub App + GitHub Actions) is still included — see
> [PR mode](#legacy-pr-diff-mode) at the bottom.

## How it works

```
GitHub repo URL
        │
        ▼
  Crawler  ── clone, skip node_modules/.git/binaries/lockfiles
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
- **Frontend**: React + Vite (navbar, repo folder-tree sidebar, light/dark theme)
- **Fixes & commits**: git worktree apply + conflict/break safety gates, single batched commit, push / pull-request
- **Auth**: GitHub OAuth web flow (stdlib `urllib`, server-side sessions via HttpOnly cookie)
- **Database**: SQLite via SQLAlchemy (scan + finding audit trail, cached fixes, sessions)

## Quick start (deep-scan mode)

```bash
pip install -r requirements.txt

cp .env.example .env
# Fill in AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
#   AZURE_OPENAI_DEPLOYMENT (chat), AZURE_OPENAI_EMBEDDING_DEPLOYMENT (embeddings)
#
# Optional, to commit/push fixes to remote GitHub repos:
#   GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET   (OAuth app)
# Optional, to gate commits on a build/test command:
#   CODE_REVIEW_VERIFY_CMD="pytest -q"
```

### 1. CLI

```bash
# Scan a remote repo
python cli.py scan https://github.com/pallets/flask.git --open

# Write reports to ./out
python cli.py scan https://github.com/owner/repo.git --out out
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

### 3. Apply fixes & commit

For each finding the UI can generate a concrete change:

- **Fix** — a verbatim, in-file code replacement, shown as a colour diff. Real
  fixes are selectable and **committable**.
- **Suggestion** — advisory/architectural remedies (e.g. "adopt Alembic", "add
  tests") that can't be expressed as a safe code edit. These are shown as advice
  only: they never remove lines and are **not committable**.

Select any number of fixes and commit them together. Before anything is written,
two safety gates must pass:

1. **No conflicts** — overlapping edits / stale or ambiguous snippets are rejected.
2. **Still builds** — patched files must parse (Python `compile`, `node --check`
   for JS) and an optional `CODE_REVIEW_VERIFY_CMD` must exit `0`; otherwise the
   working tree is restored and nothing is committed.

Commit modes:

- **direct** — commit and push to the repo's branch on `origin`.
- **pr** — push a new `ai-code-review/<id>` branch and open a pull request.

### Sign in with GitHub

Committing and pushing fixes requires a **GitHub sign-in** — click **Sign in
with GitHub** in the navbar. The OAuth flow stores the access token server-side
(keyed by an HttpOnly cookie — it never reaches the browser); pushes only
succeed on repos your account can actually write to. Configure
`GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` to enable it (see `.env.example`).

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/scan` | Start a scan `{ "repo_source": "<github-url>" }` → `{ scan_id }` |
| `GET` | `/api/scan/{id}` | Status, stage, score, grade, findings, file list |
| `GET` | `/api/scan/{id}/report.html` | Standalone HTML report |
| `GET` | `/api/scans` | Recent scans |
| `POST` | `/api/scan/{id}/fix` | Generate (or return cached) a fix for one finding |
| `POST` | `/api/scan/{id}/commit` | Apply + commit selected fixes `{ finding_ids, message, mode }` |
| `GET` | `/api/scan/{id}/access` | Commit/push capability for the scan (mode, login, push rights) |
| `GET` | `/api/auth/me` | Current GitHub session (if any) |
| `GET` | `/api/auth/github/login` | Start the GitHub OAuth flow |
| `GET` | `/api/auth/github/callback` | OAuth callback (sets session cookie) |
| `POST` | `/api/auth/logout` | Clear the session |

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
│   ├── crawler.py               # clone + exclusion rules + language detection
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
│   ├── config.py                # Azure chat/embeddings factories + tuning + OAuth settings
│   ├── state.py                 # ScanState / ScanFinding models
│   ├── fixer.py                 # generate fixes + conflict/break-safe apply & commit/push
│   ├── github_auth.py           # GitHub OAuth flow, push-access checks, PR creation, sessions
│   └── database.py              # SQLAlchemy models (Scan, ScanFinding, AgentRun, ScanFix, GitHubSession)
├── api/
│   └── server.py                # FastAPI deep-scan API + fix/commit + auth endpoints
├── frontend/                    # React + Vite UI (navbar, folder-tree sidebar, theme toggle)
└── tests/
```

## Tests

```bash
pytest tests/test_scanner.py -v   # offline: crawler + chunker (no Azure needed)
pytest tests/ -v                  # full suite
```

