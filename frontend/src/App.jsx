import { useEffect, useMemo, useRef, useState } from "react";
import {
  startScan,
  getScan,
  reportUrl,
  generateFix,
  commitFixes,
  getAccess,
  getAuth,
  githubLoginUrl,
  logout,
} from "./api";

const STAGES = ["queued", "crawl", "index", "analyze", "report", "done"];
const STAGE_LABEL = {
  queued: "Queued",
  crawl: "Crawling repository",
  index: "Embedding into vector store",
  analyze: "Running review agents",
  report: "Building report",
  done: "Done",
};
const SEV_CLASS = { high: "sev-high", medium: "sev-med", low: "sev-low" };
const SEV_ORDER = ["high", "medium", "low"];
const SEV_LABEL = { high: "High", medium: "Medium", low: "Low" };
const AGENT_LABEL = {
  security: "Security",
  performance: "Performance",
  architecture: "Architecture",
  quality: "Code Quality",
};

function DiffView({ diff }) {
  if (!diff) return null;
  return (
    <pre className="diff">
      {diff.split("\n").map((ln, i) => {
        let cls = "diff-ctx";
        if (ln.startsWith("+") && !ln.startsWith("+++")) cls = "diff-add";
        else if (ln.startsWith("-") && !ln.startsWith("---")) cls = "diff-del";
        else if (ln.startsWith("@@")) cls = "diff-hunk";
        return (
          <div key={i} className={cls}>
            {ln || " "}
          </div>
        );
      })}
    </pre>
  );
}

function buildTree(paths) {
  const root = {};
  for (const p of paths) {
    const parts = p.split("/");
    let node = root;
    parts.forEach((part, i) => {
      const isFile = i === parts.length - 1;
      node.children = node.children || {};
      node.children[part] = node.children[part] || { name: part, isFile, children: {} };
      node = node.children[part];
    });
  }
  return root.children || {};
}

function TreeNodes({ nodes }) {
  const entries = Object.values(nodes).sort((a, b) =>
    a.isFile === b.isFile ? a.name.localeCompare(b.name) : a.isFile ? 1 : -1
  );
  return entries.map((n) =>
    n.isFile ? (
      <div className="tree-file" key={n.name} title={n.name}>
        <span className="tree-ico">📄</span>
        {n.name}
      </div>
    ) : (
      <details className="tree-folder" key={n.name} open>
        <summary>
          <span className="tree-ico">📁</span>
          {n.name}
        </summary>
        <div className="tree-children">
          <TreeNodes nodes={n.children} />
        </div>
      </details>
    )
  );
}

function FileTree({ paths }) {
  const tree = useMemo(() => buildTree(paths), [paths]);
  return (
    <div className="filetree">
      <TreeNodes nodes={tree} />
    </div>
  );
}

export default function App() {
  const [source, setSource] = useState("");
  const [scan, setScan] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pollRef = useRef(null);

  // fix workflow state
  const [fixes, setFixes] = useState({});         // finding_id -> fix | {loading} | {error}
  const [selected, setSelected] = useState({});   // finding_id -> true
  const [committed, setCommitted] = useState({}); // finding_id -> short_sha
  const [conflicts, setConflicts] = useState({}); // finding_id -> reason
  const [commitMsg, setCommitMsg] = useState("");
  const [commitMode, setCommitMode] = useState("pr");
  const [committing, setCommitting] = useState(false);
  const [commitResult, setCommitResult] = useState(null);

  // auth + repo access
  const [auth, setAuth] = useState({ authenticated: false });
  const [access, setAccess] = useState(null);

  // theme
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);
  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  useEffect(() => () => clearInterval(pollRef.current), []);
  useEffect(() => {
    getAuth().then(setAuth).catch(() => {});
  }, []);

  function resetFixState() {
    setFixes({});
    setSelected({});
    setCommitted({});
    setConflicts({});
    setCommitMsg("");
    setCommitResult(null);
    setAccess(null);
  }

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setScan(null);
    resetFixState();
    setBusy(true);
    try {
      const { scan_id } = await startScan(source.trim());
      pollRef.current = setInterval(async () => {
        try {
          const data = await getScan(scan_id);
          setScan(data);
          if (data.status === "done" || data.status === "failed") {
            clearInterval(pollRef.current);
            setBusy(false);
            if (data.status === "done") {
              getAccess(scan_id)
                .then((a) => {
                  setAccess(a);
                  setCommitMode("pr");
                })
                .catch(() => {});
            }
          }
        } catch (err) {
          clearInterval(pollRef.current);
          setError(err.message);
          setBusy(false);
        }
      }, 1500);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  async function loadFix(findingId) {
    if (fixes[findingId] && !fixes[findingId].error && !fixes[findingId].loading) {
      return fixes[findingId];
    }
    setFixes((m) => ({ ...m, [findingId]: { loading: true } }));
    try {
      const fix = await generateFix(scan.scan_id, findingId);
      setFixes((m) => ({ ...m, [findingId]: fix }));
      return fix;
    } catch (err) {
      setFixes((m) => ({ ...m, [findingId]: { error: err.message } }));
      return null;
    }
  }

  async function toggleSelect(findingId) {
    if (committed[findingId]) return;
    if (selected[findingId]) {
      setSelected((s) => {
        const n = { ...s };
        delete n[findingId];
        return n;
      });
      return;
    }
    // ensure an applicable fix exists before selecting
    const fix = await loadFix(findingId);
    if (fix && fix.applicable) {
      setConflicts((c) => {
        const n = { ...c };
        delete n[findingId];
        return n;
      });
      setSelected((s) => ({ ...s, [findingId]: true }));
    }
  }

  async function onCommit() {
    const ids = Object.keys(selected);
    if (!ids.length) return;
    setCommitting(true);
    setCommitResult(null);
    setConflicts({});
    try {
      const res = await commitFixes(scan.scan_id, ids, commitMsg.trim(), commitMode);
      if (res.committed) {
        const mark = {};
        (res.applied || []).forEach((id) => (mark[id] = res.short_sha));
        setCommitted((c) => ({ ...c, ...mark }));
        setSelected({});
        setCommitMsg("");
        setCommitResult({ ok: true, ...res });
      } else {
        const conf = {};
        (res.conflicts || []).forEach((c) => (conf[c.finding_id] = c.reason));
        setConflicts(conf);
        setCommitResult({ ok: false, ...res });
      }
    } catch (err) {
      setCommitResult({ ok: false, message: err.message });
    } finally {
      setCommitting(false);
    }
  }

  async function onLogout() {
    await logout();
    setAuth({ authenticated: false });
    if (scan?.status === "done") {
      getAccess(scan.scan_id).then(setAccess).catch(() => {});
    }
  }

  const findingsBySeverity = { high: [], medium: [], low: [] };
  (scan?.findings || []).forEach((f) => {
    (findingsBySeverity[f.severity] = findingsBySeverity[f.severity] || []).push(f);
  });

  const committable = access ? access.committable : scan?.committable;
  const selectedCount = Object.keys(selected).length;
  const projectFiles = scan?.files || [];
  const showSidebar = scan?.status === "done" && projectFiles.length > 0;
  const modeOptions = [
    { v: "pr", label: "Push branch & open PR" },
    { v: "direct", label: "Push to default branch" },
  ];

  return (
    <div className="app">
      <nav className="navbar">
        <div className="brand">🔍 AI Code Review Platform</div>
        <div className="nav-right">
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            aria-label="Toggle color theme"
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          {auth.authenticated && (
            <span className="nav-user">
              {auth.avatar_url && <img src={auth.avatar_url} alt="" className="avatar" />}
              <span className="uname">{auth.login}</span>
            </span>
          )}
          {auth.authenticated ? (
            <button className="nav-btn ghost" onClick={onLogout}>
              Sign out
            </button>
          ) : auth.oauth_enabled ? (
            <a className="nav-btn gh-login" href={githubLoginUrl()}>
              <span className="gh-mark">⬢</span> Sign in with GitHub
            </a>
          ) : (
            <span className="muted small">Sign-in not configured</span>
          )}
        </div>
      </nav>

      <div className={`layout ${showSidebar ? "has-side" : ""}`}>
        {showSidebar && (
          <aside className="sidebar">
            <div className="side-title">
              {scan.repo_name || "Project"}
              <span className="muted small"> · {projectFiles.length} files</span>
            </div>
            <FileTree paths={projectFiles} />
          </aside>
        )}

        <div className="wrap">
          <header className="apphead">
            <p className="muted lead">
              Hand it a GitHub repository URL — four agents scan the whole repo for
              security, performance, architecture and quality issues, then let you
              commit the fixes you approve.
            </p>
          </header>

      <form onSubmit={onSubmit} className="bar">
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="https://github.com/owner/repo.git"
          disabled={busy}
        />
        <button disabled={busy || !source.trim()}>{busy ? "Scanning…" : "Scan"}</button>
      </form>

      {error && <div className="error">⚠ {error}</div>}

      {scan && (
        <>
          <div className="progress">
            {STAGES.map((s) => {
              const active = STAGES.indexOf(scan.stage) >= STAGES.indexOf(s);
              const current = scan.stage === s && scan.status !== "done";
              return (
                <div key={s} className={`step ${active ? "active" : ""} ${current ? "current" : ""}`}>
                  <span className="dot" />
                  {STAGE_LABEL[s]}
                </div>
              );
            })}
          </div>

          <div className="meta muted">
            {scan.repo_name || scan.repo_source} · {scan.file_count || 0} files ·{" "}
            {scan.chunk_count || 0} chunks
          </div>

          {scan.status === "failed" && <div className="error">Scan failed: {scan.error}</div>}

          {scan.status === "done" && (
            <>
              <div className="scorecard">
                <div className={`grade grade-${scan.grade}`}>{scan.grade}</div>
                <div>
                  <div className="score">{scan.score}/100 health</div>
                  <div className="muted">{scan.finding_count} findings</div>
                </div>
                <a className="report-link" href={reportUrl(scan.scan_id)} target="_blank" rel="noreferrer">
                  Open full HTML report ↗
                </a>
              </div>

              {(scan.findings || []).length === 0 ? (
                <div className="ok">✅ No significant issues found.</div>
              ) : (
                <>
                  {!committable && access?.mode === "github" && access?.login_required && (
                    <div className="hint">
                      🔐 This is a remote GitHub repo.{" "}
                      {access.oauth_enabled ? (
                        <>
                          <a className="gh-login inline" href={githubLoginUrl()}>
                            Sign in with GitHub
                          </a>{" "}
                          to apply, commit and push fixes you have access to.
                        </>
                      ) : (
                        <>GitHub sign-in isn't configured on the server (set GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET).</>
                      )}
                    </div>
                  )}
                  {!committable && access?.mode === "github" && !access?.login_required && (
                    <div className="hint">
                      ⛔ Signed in as <strong>{access.login}</strong>, but{" "}
                      {access.reason || "you don't have push access to this repository"}.
                    </div>
                  )}
                  {committable && access?.mode === "github" && (
                    <div className="hint ok-hint">
                      ✅ Signed in as <strong>{access.login}</strong> with push access to{" "}
                      <strong>{access.owner}/{access.repo}</strong> — select fixes to commit & push.
                    </div>
                  )}

                  {SEV_ORDER.filter((sev) => (findingsBySeverity[sev] || []).length).map((sev) => (
                    <details key={sev} className="agent" open>
                      <summary>
                        <h2>{SEV_LABEL[sev]} ({findingsBySeverity[sev].length})</h2>
                      </summary>
                      {findingsBySeverity[sev].map((f) => {
                        const fix = fixes[f.id];
                        const isSel = !!selected[f.id];
                        const sha = committed[f.id];
                        const conflictReason = conflicts[f.id];
                        return (
                          <div
                            key={f.id}
                            className={`finding ${SEV_CLASS[f.severity]} ${isSel ? "selected" : ""} ${
                              conflictReason ? "conflict" : ""
                            }`}
                          >
                            <div className="ftitle">
                              {committable && (
                                <input
                                  type="checkbox"
                                  className="pick"
                                  checked={isSel}
                                  disabled={!!sha}
                                  onChange={() => toggleSelect(f.id)}
                                  title={sha ? "Already committed" : "Select to commit"}
                                />
                              )}
                              <span className="badge">{f.severity}</span> {f.title}
                              <span className="fagent muted"> · {AGENT_LABEL[f.agent] || f.agent}</span>
                              {sha && <span className="committed-tag">committed {sha}</span>}
                            </div>
                            <div className="floc">
                              {f.file}{f.line ? `:${f.line}` : ""}
                            </div>
                            {f.detail && <p className="fdetail">{f.detail}</p>}
                            {f.recommendation && <div className="ffix">💡 {f.recommendation}</div>}

                            {committable && !sha && (
                              <div className="fixrow">
                                <button
                                  className="ghost"
                                  disabled={fix?.loading}
                                  onClick={() => loadFix(f.id)}
                                >
                                  {fix?.loading
                                    ? "Generating fix…"
                                    : fix && !fix.error
                                    ? "Regenerate fix"
                                    : "Preview fix"}
                                </button>
                                {fix && !fix.loading && !fix.error && !fix.applicable && (
                                  <span className="muted small">
                                    {fix.kind === "suggestion"
                                      ? "Advisory only — not auto-committable."
                                      : "No safe automatic fix available."}
                                  </span>
                                )}
                                {conflictReason && (
                                  <span className="conflict-msg">⚠ {conflictReason}</span>
                                )}
                              </div>
                            )}

                            {fix?.error && <div className="error small">⚠ {fix.error}</div>}
                            {fix && !fix.loading && !fix.error && !fix.applicable && fix.kind === "suggestion" && fix.explanation && (
                              <div className="suggestion-note">💡 {fix.explanation}</div>
                            )}
                            {fix && !fix.loading && !fix.error && fix.applicable && (
                              <>
                                {fix.explanation && <div className="fixexpl">{fix.explanation}</div>}
                                <DiffView diff={fix.diff} />
                              </>
                            )}
                          </div>
                        );
                      })}
                    </details>
                  ))}
                </>
              )}
            </>
          )}
        </>
      )}

      {committable && selectedCount > 0 && (
        <div className="commitbar">
          <div className="commitbar-inner">
            <span className="cb-count">
              {selectedCount} fix{selectedCount > 1 ? "es" : ""} selected
            </span>
            <input
              className="cb-msg"
              value={commitMsg}
              onChange={(e) => setCommitMsg(e.target.value)}
              placeholder="Commit message (optional)"
              disabled={committing}
            />
            <select
              className="cb-mode"
              value={commitMode}
              onChange={(e) => setCommitMode(e.target.value)}
              disabled={committing}
              title="What to do after committing"
            >
              {modeOptions.map((o) => (
                <option key={o.v} value={o.v}>
                  {o.label}
                </option>
              ))}
            </select>
            <button className="cb-btn" onClick={onCommit} disabled={committing}>
              {committing
                ? "Working…"
                : commitMode === "pr"
                ? `Commit ${selectedCount} & open PR`
                : commitMode === "direct"
                ? `Commit ${selectedCount} & push`
                : `Commit ${selectedCount} in 1 commit`}
            </button>
          </div>
          {commitResult && !commitResult.ok && (
            <div className="cb-result error">
              ⚠ {commitResult.message}
              {commitResult.breaks?.length > 0 && (
                <ul className="cb-breaks">
                  {commitResult.breaks.map((b, i) => (
                    <li key={i}>
                      <code>{b.file}</code> — {b.reason}
                    </li>
                  ))}
                </ul>
              )}
              {commitResult.verify_error && (
                <pre className="cb-verify">{commitResult.verify_error}</pre>
              )}
            </div>
          )}
        </div>
      )}

      {commitResult?.ok && (
        <div className="cb-success">
          ✅ Committed {commitResult.applied?.length} fix
          {commitResult.applied?.length > 1 ? "es" : ""} as <code>{commitResult.short_sha}</code>
          {commitResult.files?.length ? ` · ${commitResult.files.join(", ")}` : ""}
          {commitResult.pr_url && (
            <span>
              {" "}·{" "}
              <a href={commitResult.pr_url} target="_blank" rel="noreferrer" className="pr-link">
                Pull Request opened ↗
              </a>
            </span>
          )}
          {commitResult.pushed && !commitResult.pr_url && (
            <span className="pushed-tag"> · pushed to {commitResult.branch} ↗</span>
          )}
          {!commitResult.pushed && commitResult.push_error && (
            <div className="cb-pushwarn">⚠ Committed locally but not pushed: {commitResult.push_error}</div>
          )}
          {commitResult.pushed && commitResult.push_error && (
            <div className="cb-pushwarn">⚠ {commitResult.push_error}</div>
          )}
        </div>
      )}
        </div>
      </div>
    </div>
  );
}
