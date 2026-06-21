import { useEffect, useRef, useState } from "react";
import { startScan, getScan, reportUrl } from "./api";

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
const AGENT_LABEL = {
  security: "Security",
  performance: "Performance",
  architecture: "Architecture",
  quality: "Code Quality",
};

export default function App() {
  const [source, setSource] = useState("");
  const [scan, setScan] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setScan(null);
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

  const findingsByAgent = {};
  (scan?.findings || []).forEach((f) => {
    (findingsByAgent[f.agent] = findingsByAgent[f.agent] || []).push(f);
  });

  return (
    <div className="wrap">
      <header>
        <h1>🔍 AI Code Review Platform</h1>
        <p className="muted">
          Hand it a GitHub URL or local path — four agents scan the whole repo for
          security, performance, architecture and quality issues.
        </p>
      </header>

      <form onSubmit={onSubmit} className="bar">
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="https://github.com/owner/repo.git  or  /path/to/project"
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

              {Object.keys(findingsByAgent).length === 0 ? (
                <div className="ok">✅ No significant issues found.</div>
              ) : (
                Object.entries(findingsByAgent).map(([agent, items]) => (
                  <section key={agent} className="agent">
                    <h2>{AGENT_LABEL[agent] || agent} ({items.length})</h2>
                    {items.map((f, i) => (
                      <div key={i} className={`finding ${SEV_CLASS[f.severity]}`}>
                        <div className="ftitle">
                          <span className="badge">{f.severity}</span> {f.title}
                        </div>
                        <div className="floc">
                          {f.file}{f.line ? `:${f.line}` : ""}
                        </div>
                        {f.detail && <p className="fdetail">{f.detail}</p>}
                        {f.recommendation && <div className="ffix">💡 {f.recommendation}</div>}
                      </div>
                    ))}
                  </section>
                ))
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
