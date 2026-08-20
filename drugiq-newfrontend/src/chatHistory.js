// Client-side chat history. The Flask backend has no session-listing/
// persistence endpoint (chat_sessions is an in-memory dict keyed by
// session_id, gone on restart), so "history" here is real but local-only:
// browsing a past conversation reloads its messages from this device: the
// next message you send in it starts a fresh server-side session (no
// backend context to resume), same as how the previous UI only ever kept
// one session per page load.

const BASE_KEY = 'drugiq_chat_history_v2'
const MAX_CONVERSATIONS = 60

function getStorageKey(role) {
  const r = (role || 'patient').toLowerCase()
  return `${BASE_KEY}_${r}`
}

export function loadConversations(role = 'patient') {
  try {
    const raw = localStorage.getItem(getStorageKey(role))
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function persist(list, role = 'patient') {
  try {
    localStorage.setItem(getStorageKey(role), JSON.stringify(list.slice(0, MAX_CONVERSATIONS)))
  } catch {
    // storage full/unavailable — history just won't persist this session
  }
}

export function saveConversation(id, { title, messages }, role = 'patient') {
  const list = loadConversations(role)
  const idx = list.findIndex((c) => c.id === id)
  const entry = { id, title, messages, updatedAt: Date.now() }
  if (idx === -1) list.unshift(entry)
  else list[idx] = entry
  list.sort((a, b) => b.updatedAt - a.updatedAt)
  persist(list, role)
  return list
}

export function deleteConversation(id, role = 'patient') {
  const list = loadConversations(role).filter((c) => c.id !== id)
  persist(list, role)
  return list
}

export function titleFromFirstMessage(text) {
  const clean = (text || '').trim().replace(/\s+/g, ' ')
  return clean.length > 60 ? clean.slice(0, 57) + '…' : clean || 'New conversation'
}

// Groups into Today / Yesterday / Previous 7 days / Older, same shape the
// UI's history sidebar expects: [{ title, items: [{id, title}] }]
export function groupConversations(list) {
  const now = new Date()
  const startOf = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const today = startOf(now)
  const yesterday = today - 86400000
  const weekAgo = today - 7 * 86400000

  const buckets = { Today: [], Yesterday: [], 'Previous 7 days': [], Older: [] }
  for (const convo of list) {
    const day = startOf(new Date(convo.updatedAt))
    if (day === today) buckets.Today.push(convo)
    else if (day === yesterday) buckets.Yesterday.push(convo)
    else if (day >= weekAgo) buckets['Previous 7 days'].push(convo)
    else buckets.Older.push(convo)
  }

  return Object.entries(buckets)
    .filter(([, items]) => items.length)
    .map(([title, items]) => ({ title, items }))
}
