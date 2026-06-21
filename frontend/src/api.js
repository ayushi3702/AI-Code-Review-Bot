// Thin API client for the FastAPI backend. Paths are proxied via vite.config.js.

export async function startScan(repoSource) {
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_source: repoSource }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Failed to start scan");
  return res.json();
}

export async function getScan(scanId) {
  const res = await fetch(`/api/scan/${scanId}`);
  if (!res.ok) throw new Error("Failed to fetch scan");
  return res.json();
}

export function reportUrl(scanId) {
  return `/api/scan/${scanId}/report.html`;
}
