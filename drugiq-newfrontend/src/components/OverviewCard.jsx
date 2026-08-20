import { useState } from 'react'
import { FlaskConical } from './icons'
import CitationsBlock from './CitationsBlock'
import { formatAnswerHtml } from '../format'
import { chat, clearSession, newSessionId } from '../api'

// Matches the 4-section overview (Composition/Usage/Dosage/Side Effects —
// "Alternative Brands" was deliberately dropped from the original 5).
const OVERVIEW_SECTIONS = [
  { key: 'composition', label: 'Composition', question: (d) => `What is the composition of ${d}? Answer in bullet points only.` },
  { key: 'usage', label: 'Usage / Indications', question: (d) => `What is ${d} used for / indicated for? Answer in bullet points only.` },
  { key: 'dosage', label: 'Dosage', question: (d) => `What is the recommended dosage of ${d}? Answer in bullet points only.` },
  { key: 'side_effects', label: 'Side Effects', question: (d) => `What are the side effects of ${d}? Answer in bullet points only.` },
]

export default function OverviewCard({ drugName, apiKey, cardRef }) {
  const [activeSection, setActiveSection] = useState(null)
  const [loadingKey, setLoadingKey] = useState(null)
  const [cache, setCache] = useState({})

  async function openSection(section) {
    setActiveSection(section.key)
    if (cache[section.key]) return

    setLoadingKey(section.key)
    const tempSessionId = newSessionId('sid_tmp')
    try {
      const data = await chat({ question: section.question(drugName), sessionId: tempSessionId, apiKey })
      setCache((prev) => ({ ...prev, [section.key]: data.error ? { error: data.error } : data }))
    } catch {
      setCache((prev) => ({ ...prev, [section.key]: { error: 'Failed to load this section. Try again.' } }))
    } finally {
      clearSession(tempSessionId)
      setLoadingKey(null)
    }
  }

  const activeData = activeSection ? cache[activeSection] : null

  return (
    <div className="message-row assistant" ref={cardRef}>
      <div className="avatar"><FlaskConical size={16} /></div>
      <div className="message assistant overview-card">
        <div className="overview-card-header">
          <div>
            <div className="overview-card-title">{drugName}</div>
            <div className="overview-card-subtitle">Choose a section to explore</div>
          </div>
        </div>
        <div className="overview-card-grid">
          {OVERVIEW_SECTIONS.map((s) => (
            <button
              key={s.key}
              className={`overview-section-btn ${activeSection === s.key ? 'active' : ''}`}
              onClick={() => openSection(s)}
            >
              <span className="label">{s.label}</span>
            </button>
          ))}
        </div>
        {activeSection && (
          <div className="overview-card-panel">
            {loadingKey === activeSection && !activeData ? (
              <div className="overview-panel-loading"><span /><span /><span /></div>
            ) : activeData?.error ? (
              <div className="overview-panel-error">{activeData.error}</div>
            ) : activeData ? (
              <>
                <div className="message-text" dangerouslySetInnerHTML={formatAnswerHtml(activeData.answer)} />
                {activeData.refused ? (
                  <div className="refused-note">Could not find reliable information in the loaded documents.</div>
                ) : (
                  <CitationsBlock citations={activeData.citations} />
                )}
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}
