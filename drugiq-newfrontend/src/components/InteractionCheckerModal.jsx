import { useEffect, useState } from 'react'
import { FlaskConical, ShieldCheck, X } from './icons'
import CitationsBlock from './CitationsBlock'
import { formatAnswerHtml, parseInteractionStatus, drugDisplayName } from '../format'
import { getDrugs, chat, clearSession, newSessionId } from '../api'

export default function InteractionCheckerModal({ apiKey, role, onClose, toast }) {
  const [drugs, setDrugs] = useState([])
  const [drugA, setDrugA] = useState('')
  const [drugB, setDrugB] = useState('')
  const [checking, setChecking] = useState(false)
  const [result, setResult] = useState(null)

  useEffect(() => {
    getDrugs(role).then((data) => setDrugs(data.drugs || [])).catch(() => toast?.('Failed to load drug list', 'error'))
  }, [role])

  async function runCheck() {
    if (!drugA || !drugB) { toast?.('Select two drugs to check.', 'error'); return }
    if (drugA === drugB) { toast?.('Select two different drugs.', 'error'); return }

    const nameA = drugDisplayName(drugA)
    const nameB = drugDisplayName(drugB)
    const question = role === 'patient'
      ? `What is the safety guidance and interaction between ${nameA} and ${nameB}? Are they safe to use together or contraindicated?`
      : `What is the drug interaction between ${nameA} and ${nameB}? Are they contraindicated or safe to use together?`

    setChecking(true)
    setResult(null)
    const tempSessionId = newSessionId('sid_tmp')
    try {
      const data = await chat({ question, sessionId: tempSessionId, apiKey, role })
      if (data.error) { toast?.(data.error, 'error'); return }
      setResult(data)
    } catch {
      toast?.('Failed to check interaction. Is the server running?', 'error')
    } finally {
      clearSession(tempSessionId)
      setChecking(false)
    }
  }

  const parsed = result ? parseInteractionStatus(result.answer) : null

  return (
    <div className="window-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <section className="app-window interaction-window">
        <header className="window-header">
          <div className="window-title">
            <span className="window-title-icon"><FlaskConical size={18} /></span>
            <div><strong>Drug interaction checker</strong><small>Check whether two drugs are safe to use together</small></div>
          </div>
          <button className="window-close" onClick={onClose}><X size={18} /></button>
        </header>
        <div className="window-body">
          <div className="window-section-head">
            <div>
              <span className="window-eyebrow">INTERACTION CHECK</span>
              <h2>Compare two drugs</h2>
              <p>Pulls from each drug's own prescribing document, cross-referenced with the extracted knowledge graph when available.</p>
            </div>
          </div>

          <div className="interaction-picker">
            <select className="interaction-select" value={drugA} onChange={(e) => setDrugA(e.target.value)}>
              <option value="">Drug 1…</option>
              {drugs.map((d) => <option key={d.drug_name} value={d.drug_name}>{drugDisplayName(d.drug_name)}</option>)}
            </select>
            <span className="interaction-picker-x">×</span>
            <select className="interaction-select" value={drugB} onChange={(e) => setDrugB(e.target.value)}>
              <option value="">Drug 2…</option>
              {drugs.map((d) => <option key={d.drug_name} value={d.drug_name}>{drugDisplayName(d.drug_name)}</option>)}
            </select>
            <button className="document-action-card interaction-check-btn" onClick={runCheck} disabled={checking}>
              <span>{checking ? 'Checking…' : 'Check interaction'}</span>
            </button>
          </div>

          {result && (
            <div className="interaction-result">
              {parsed?.status && (
                <div className={`interaction-status-badge ${parsed.status.cls}`}>
                  {parsed.status.label}
                </div>
              )}
              <div className="message-text" dangerouslySetInnerHTML={formatAnswerHtml(parsed?.text ?? result.answer)} />
              {result.refused ? (
                <div className="refused-note">Could not find reliable information in the loaded documents.</div>
              ) : (
                <CitationsBlock citations={result.citations} />
              )}
            </div>
          )}

          <div className="window-info-strip">
            <ShieldCheck size={16} />
            <div><strong>Always verify</strong><span>This check reflects only the indexed documents — confirm with a pharmacist or physician before acting on it.</span></div>
          </div>
        </div>
      </section>
    </div>
  )
}
