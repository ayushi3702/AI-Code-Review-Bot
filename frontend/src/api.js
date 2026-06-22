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

export async function generateFix(scanId, findingId) {
  const res = await fetch(`/api/scan/${scanId}/fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ finding_id: findingId }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Failed to generate fix");
  return res.json();
}

export async function commitFixes(scanId, findingIds, message, mode = "local") {
  const res = await fetch(`/api/scan/${scanId}/commit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ finding_ids: findingIds, message, mode }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Failed to commit fixes");
  return res.json();
}

export async function getAccess(scanId) {
  const res = await fetch(`/api/scan/${scanId}/access`);
  if (!res.ok) throw new Error("Failed to check repo access");
  return res.json();
}

export async function getAuth() {
  const res = await fetch("/api/auth/me");
  if (!res.ok) return { authenticated: false };
  return res.json();
}

export function githubLoginUrl() {
  return "/api/auth/github/login";
}

export async function logout() {
  await fetch("/api/auth/logout", { method: "POST" });
}
