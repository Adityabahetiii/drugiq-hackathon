// Centralized client for the Flask backend (see ../../drug-chatbot/app.py).
// Can connect to a separate backend URL (e.g. Vercel frontend -> Render backend)
// via VITE_API_URL, or defaults to same-origin / Vite proxy.

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

async function asJson(res) {
  let data
  try {
    data = await res.json()
  } catch {
    throw new Error(`Server returned ${res.status} with a non-JSON response`)
  }
  if (!res.ok && !data.error) {
    throw new Error(`Request failed (${res.status})`)
  }
  return data
}

export function newSessionId(prefix = 'sid') {
  return `${prefix}_${Math.random().toString(36).slice(2, 11)}`
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/api/status`)
  return asJson(res)
}

export async function getDrugs(role = 'doctor') {
  const res = await fetch(`${API_BASE}/api/drugs?role=${encodeURIComponent(role || 'doctor')}`)
  return asJson(res)
}

export async function chat({ question, sessionId, chatHistory, apiKey, role = 'patient' }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      chat_history: chatHistory || null,
      api_key: apiKey || null,
      role: role || 'patient',
    }),
  })
  return asJson(res)
}

export async function uploadPdf(file, role = 'doctor') {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('role', role)
  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: fd })
  return asJson(res)
}

export async function rebuildDatabase(role = 'doctor') {
  const res = await fetch(`${API_BASE}/api/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role }),
  })
  return asJson(res)
}

export async function removeDrug(filename, role = 'doctor') {
  const res = await fetch(`${API_BASE}/api/drugs/${encodeURIComponent(filename)}?role=${encodeURIComponent(role)}`, { method: 'DELETE' })
  return asJson(res)
}

export async function clearSession(sessionId) {
  try {
    await fetch(`${API_BASE}/api/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    })
  } catch {
    // best-effort cleanup only
  }
}

export async function setApiKey(apiKey) {
  const res = await fetch(`${API_BASE}/api/set_key`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey }),
  })
  return asJson(res)
}

export async function getKnowledgeGraph(forceRebuild = false, role = 'doctor') {
  const url = forceRebuild
    ? `${API_BASE}/api/knowledge-graph/rebuild?role=${encodeURIComponent(role)}`
    : `${API_BASE}/api/knowledge-graph?role=${encodeURIComponent(role)}`
  const res = await fetch(url, {
    method: forceRebuild ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: forceRebuild ? JSON.stringify({ role }) : undefined,
  })
  return asJson(res)
}

export function pdfUrl(filename) {
  return `${API_BASE}/api/download/${encodeURIComponent(filename)}`
}
