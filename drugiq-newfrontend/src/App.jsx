import { useEffect, useRef, useState } from 'react'
import {
  ArrowDown, ChevronRight, Database, FlaskConical, LogOut, Menu, MessageCircle,
  Network, Paperclip, PanelLeft, Plus, Send, ShieldCheck, Stethoscope, Trash2, User,
} from './components/icons'
import MessageBubble from './components/MessageBubble'
import OverviewCard from './components/OverviewCard'
import Toasts from './components/Toasts'
import LoadingOverlay from './components/LoadingOverlay'
import SplashScreen from './components/SplashScreen'
import DocumentsModal from './components/DocumentsModal'
import InteractionCheckerModal from './components/InteractionCheckerModal'
import KnowledgeGraphModal from './components/KnowledgeGraphModal'
import SourceViewer from './components/SourceViewer'
import RoleSelectScreen from './components/RoleSelectScreen'
import { getStatus, chat as chatApi, uploadPdf, clearSession, newSessionId } from './api'
import { drugDisplayName } from './format'
import { registerSourceViewerListener } from './openSource'
import {
  loadConversations, saveConversation, deleteConversation,
  titleFromFirstMessage, groupConversations,
} from './chatHistory'

function slugify(s) {
  return (s || 'drug').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-+|-+$)/g, '') || 'drug'
}

function App() {
  // Role is stored in React state ONLY (no localStorage or sessionStorage)
  const [role, setRole] = useState(null) // 'patient' | 'doctor' | null (null shows login screen)

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sidebar, setSidebar] = useState(true)
  const [modal, setModal] = useState(null)
  const [busy, setBusy] = useState(false)
  const [historyList, setHistoryList] = useState([])
  const [toasts, setToasts] = useState([])
  const [dbReady, setDbReady] = useState(false)
  const [docsRefreshKey, setDocsRefreshKey] = useState(0)
  const [loadingMessage, setLoadingMessage] = useState(null)
  const [sourceView, setSourceView] = useState(null)
  const [splashHidden, setSplashHidden] = useState(false)
  const [showScrollBottom, setShowScrollBottom] = useState(false)

  const sessionIdRef = useRef(newSessionId())
  const conversationIdRef = useRef(newSessionId('conv'))
  const overviewCardsRef = useRef(new Map()) // drug key -> cardId, for this conversation
  const cardNodeRefs = useRef({}) // cardId -> DOM node
  const fileRef = useRef(null)
  const chatScrollRef = useRef(null)
  const bottomAnchorRef = useRef(null)

  const apiKey = ''

  const scrollToBottom = (smooth = true) => {
    if (bottomAnchorRef.current) {
      bottomAnchorRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'end' })
    } else if (chatScrollRef.current) {
      chatScrollRef.current.scrollTo({
        top: chatScrollRef.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      })
    }
  }

  // Automatically scroll down whenever new messages are added or busy state changes
  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom(true)
    }, 40)
    return () => clearTimeout(timer)
  }, [messages, busy])

  const handleChatScroll = () => {
    if (!chatScrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = chatScrollRef.current
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 120
    setShowScrollBottom(!isNearBottom)
  }

  useEffect(() => {
    if (role) {
      setHistoryList(loadConversations(role))
    } else {
      setHistoryList([])
    }
    refreshStatus()
    registerSourceViewerListener(setSourceView)
    return () => registerSourceViewerListener(null)
  }, [role])

  useEffect(() => {
    const t = setTimeout(() => setSplashHidden(true), 2300)
    return () => clearTimeout(t)
  }, [])

  async function refreshStatus() {
    try {
      setDbReady((await getStatus()).db_ready)
    } catch {
      /* status pill just stays as-is */
    }
  }

  function showToast(msg, type = 'info') {
    const id = Math.random().toString(36).slice(2)
    setToasts((prev) => [...prev, { id, msg, type }])
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000)
  }

  function persistCurrentConversation(nextMessages) {
    if (nextMessages.length === 0 || !role) return
    const firstUser = nextMessages.find((m) => m.role === 'user')
    saveConversation(conversationIdRef.current, {
      title: titleFromFirstMessage(firstUser?.text),
      messages: nextMessages,
    }, role)
    setHistoryList(loadConversations(role))
  }

  function scrollToOverviewCard(cardId) {
    const el = cardNodeRefs.current[cardId]
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.classList.add('flash-highlight')
    setTimeout(() => el.classList.remove('flash-highlight'), 1200)
  }

  const ask = async (text) => {
    const clean = text.trim()
    if (!clean || busy) return
    setMessages((prev) => [...prev, { role: 'user', text: clean }])
    setInput('')
    setBusy(true)

    try {
      const chatHistory = messages
        .filter((m) => m.text)
        .map((m) => ({ role: m.role, content: m.text }))

      const data = await chatApi({
        question: clean,
        sessionId: sessionIdRef.current,
        chatHistory,
        apiKey,
        role: role || 'patient',
      })
      if (data.error) {
        showToast(data.error, 'error')
        setBusy(false)
        return
      }

      if (data.query_type === 'overview_menu' && data.drug_name) {
        const key = data.drug_name.toLowerCase()
        if (overviewCardsRef.current.has(key)) {
          const cardId = overviewCardsRef.current.get(key)
          setMessages((prev) => {
            const next = [...prev, { role: 'assistant', query_type: 'back_to_menu', drugName: data.drug_name, cardId }]
            persistCurrentConversation(next)
            return next
          })
          setTimeout(() => scrollToOverviewCard(cardId), 60)
        } else {
          const cardId = 'overview-' + slugify(data.drug_name) + '-' + Math.random().toString(36).slice(2, 6)
          overviewCardsRef.current.set(key, cardId)
          setMessages((prev) => {
            const next = [...prev, { role: 'assistant', query_type: 'overview_menu', drugName: data.drug_name, cardId }]
            persistCurrentConversation(next)
            return next
          })
        }
        setBusy(false)
        return
      }

      setMessages((prev) => {
        const next = [...prev, {
          role: 'assistant',
          text: data.answer,
          citations: data.citations,
          query_type: data.query_type,
          refused: data.refused,
          drug_name: data.drug_name,
        }]
        persistCurrentConversation(next)
        return next
      })
    } catch {
      showToast('Failed to get response. Is the server running?', 'error')
    } finally {
      setBusy(false)
    }
  }

  function newChat() {
    clearSession(sessionIdRef.current)
    sessionIdRef.current = newSessionId()
    conversationIdRef.current = newSessionId('conv')
    overviewCardsRef.current = new Map()
    cardNodeRefs.current = {}
    setMessages([])
  }

  function handleSwitchRole() {
    newChat()
    setRole(null)
  }

  function handleDeleteConversation(id, e) {
    e?.stopPropagation()
    const updated = deleteConversation(id, role)
    setHistoryList(updated)
    if (conversationIdRef.current === id) {
      newChat()
    }
  }

  function openConversation(convo) {
    setMessages(convo.messages)
    conversationIdRef.current = convo.id
    sessionIdRef.current = convo.id
    overviewCardsRef.current = new Map(
      convo.messages
        .filter((m) => m.query_type === 'overview_menu' && m.drugName)
        .map((m) => [m.drugName.toLowerCase(), m.cardId])
    )
  }

  async function onFile(e) {
    if (role === 'patient') {
      showToast('PDF upload is restricted to healthcare professionals', 'error')
      return
    }
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast(`${file.name} is not a PDF`, 'error')
      return
    }
    setLoadingMessage(`Indexing ${file.name}â€¦`)
    try {
      const data = await uploadPdf(file, role)
      if (data.success) {
        showToast(data.message, 'success')
        refreshStatus()
        setDocsRefreshKey((k) => k + 1)
      } else {
        showToast(data.error || 'Upload failed', 'error')
      }
    } catch {
      showToast(`Error uploading ${file.name}`, 'error')
    } finally {
      setLoadingMessage(null)
    }
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      ask(input)
    }
  }

  const groupedHistory = groupConversations(historyList)

  return (
    <>
      <SplashScreen hidden={splashHidden} onDismiss={() => setSplashHidden(true)} />

      {!role ? (
        <RoleSelectScreen onSelectRole={(selectedRole) => setRole(selectedRole)} />
      ) : (
        <div className="app-shell">
          <header className="topbar">
          <div className="brand-area">
            <button className="icon-btn mobile-only" onClick={() => setSidebar((v) => !v)} aria-label="Toggle sidebar">
              <Menu size={19} />
            </button>
            <div className="brand-mark">
              <FlaskConical size={20} strokeWidth={2.2} />
            </div>
            <div>
              <div className="brand-name">DrugIQ</div>
              <div className="brand-sub">
                {role === 'doctor' ? 'CLINICAL INTELLIGENCE' : 'HOME MEDICATION SAFE-MODE'}
              </div>
            </div>
          </div>

          <div className="top-actions">
            {/* Visible Role Badge */}
            {role === 'doctor' ? (
              <div className="role-badge-topbar doctor">
                <Stethoscope size={14} /> <span>Doctor</span>
              </div>
            ) : (
              <div className="role-badge-topbar patient">
                <User size={14} /> <span>Patient</span>
              </div>
            )}

            {/* Status indicator */}
            <div className="status-pill">
              <span className="status-dot" />
              {role === 'doctor' ? (dbReady ? 'RAG ACTIVE' : 'NO DOCUMENTS') : 'OPENFDA OTC ACTIVE'}
            </div>

            <button className="ghost-btn switch-role-btn" onClick={handleSwitchRole} title="Switch between Patient and Doctor roles">
              <LogOut size={15} /> Switch Role
            </button>

            <button className="ghost-btn" onClick={newChat}>
              <Trash2 size={16} /> Clear chat
            </button>
          </div>
        </header>

        <div className="workspace">
          <aside className={`sidebar ${sidebar ? '' : 'collapsed'}`}>
            <div className="sidebar-inner">
              {/* Conditionally hide Documents and Knowledge Graph for Patient role */}
              <div className="side-nav">
                {role === 'doctor' && (
                  <button className="side-nav-btn" onClick={() => setModal('documents')}>
                    <Database size={17} />
                    <span>Documents</span>
                    <ChevronRight size={15} />
                  </button>
                )}

                <button className="side-nav-btn" onClick={() => setModal('interactions')}>
                  <FlaskConical size={17} />
                  <span>Interaction checker</span>
                  <ChevronRight size={15} />
                </button>

                {role === 'doctor' && (
                  <button className="side-nav-btn" onClick={() => setModal('knowledge-graph')}>
                    <Network size={17} />
                    <span>Knowledge graph</span>
                    <ChevronRight size={15} />
                  </button>
                )}
              </div>

              <div className="history-head">
                <div>
                  <span className="section-heading">CHAT HISTORY</span>
                </div>
                <button className="new-chat-btn" onClick={newChat}>
                  <Plus size={14} />
                  <span>New</span>
                </button>
              </div>

              <div className="history-list">
                {groupedHistory.length === 0 && (
                  <div className="kg-info-empty" style={{ padding: '0 7px' }}>
                    No saved conversations yet.
                  </div>
                )}
                {groupedHistory.map((group) => (
                  <div className="history-group" key={group.title}>
                    <div className="history-group-title">{group.title}</div>
                    {group.items.map((item) => (
                      <div
                        className={`history-item-wrap ${conversationIdRef.current === item.id ? 'active' : ''}`}
                        key={item.id}
                      >
                        <button className="history-item" onClick={() => openConversation(item)} title={item.title}>
                          <MessageCircle size={14} />
                          <span>{item.title}</span>
                        </button>
                        <button
                          className="history-item-del"
                          onClick={(e) => handleDeleteConversation(item.id, e)}
                          title="Delete conversation"
                          aria-label="Delete conversation"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            <div className="safety-note">
              <ShieldCheck size={16} />
              <span>
                {role === 'doctor'
                  ? 'For professional clinical reference only. Verify with full FDA prescribing documentation.'
                  : 'For informational home use only. Always consult your doctor or pharmacist for prescriptions.'}
              </span>
            </div>
          </aside>

          <main className="main-panel">
            <div className="chat-toolbar">
              <div className="chat-title">
                <MessageCircle size={17} />
                <strong>Chat</strong>
                
              </div>
              <div className="chat-toolbar-actions">
                <button className="panel-toggle" onClick={() => setSidebar((v) => !v)}>
                  <PanelLeft size={16} />
                  {sidebar ? 'Hide history' : 'Show history'}
                </button>
              </div>
            </div>

            <section className="chat-scroll" ref={chatScrollRef} onScroll={handleChatScroll}>
              <div className="conversation">
                {messages.map((m, i) => {
                  if (m.query_type === 'overview_menu') {
                    return (
                      <OverviewCard
                        key={i}
                        drugName={m.drugName}
                        apiKey={apiKey}
                        cardRef={(el) => {
                          cardNodeRefs.current[m.cardId] = el
                        }}
                      />
                    )
                  }
                  if (m.query_type === 'back_to_menu') {
                    return (
                      <div className="back-to-menu-box" key={i}>
                        <span>{drugDisplayName(m.drugName)}'s menu is already open above.</span>
                        <button className="back-to-menu-btn" onClick={() => scrollToOverviewCard(m.cardId)}>
                          Back to menu
                        </button>
                      </div>
                    )
                  }
                  return <MessageBubble key={i} message={m} />
                })}
                {busy && (
                  <div className="message-row assistant">
                    <div className="avatar">
                      <FlaskConical size={16} />
                    </div>
                    <div className="message assistant typing">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                )}
                <div ref={bottomAnchorRef} style={{ height: 1, width: '100%' }} />
              </div>
            </section>

            {showScrollBottom && (
              <button
                className="scroll-to-bottom-btn"
                onClick={() => scrollToBottom(true)}
                title="Scroll to bottom"
                aria-label="Scroll to bottom"
              >
                <ArrowDown size={18} />
              </button>
            )}

            <div className="composer-wrap">
              <div className="composer">
                {/* Conditionally hide Upload PDF button for Patient role */}
                {role === 'doctor' && (
                  <button className="attach-btn" onClick={() => fileRef.current?.click()} aria-label="Attach PDF">
                    <Paperclip size={18} />
                  </button>
                )}
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKey}
                  placeholder={
                    role === 'doctor'
                      ? 'Ask about dosage, warnings, interactions, or anything elseâ€¦'
                      : 'Ask about over-the-counter medications, directions, uses, or safetyâ€¦'
                  }
                />
                <button className="send-btn" onClick={() => ask(input)} disabled={!input.trim() || busy}>
                  Send <Send size={15} />
                </button>
              </div>
            </div>
          </main>
        </div>

        {role === 'doctor' && <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={onFile} />}

        {modal === 'documents' && role === 'doctor' && (
          <DocumentsModal
            key={docsRefreshKey}
            role={role}
            onClose={() => setModal(null)}
            onUpload={() => fileRef.current?.click()}
            onChanged={refreshStatus}
            toast={showToast}
          />
        )}
        {modal === 'interactions' && (
          <InteractionCheckerModal apiKey={apiKey} role={role} onClose={() => setModal(null)} toast={showToast} />
        )}
        {modal === 'knowledge-graph' && role === 'doctor' && (
          <KnowledgeGraphModal onClose={() => setModal(null)} toast={showToast} />
        )}

        {sourceView && <SourceViewer {...sourceView} onClose={() => setSourceView(null)} />}
        {loadingMessage && <LoadingOverlay message={loadingMessage} />}
        <Toasts toasts={toasts} />
      </div>
      )}
    </>
  )
}

export default App
