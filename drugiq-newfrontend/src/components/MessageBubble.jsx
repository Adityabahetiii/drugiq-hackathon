import { Check, FlaskConical } from './icons'
import CitationsBlock from './CitationsBlock'
import { formatAnswerHtml, parseInteractionStatus, parseWarningPrecaution } from '../format'

const QUERY_BADGES = {
  condition_to_drug: { label: 'Condition lookup', cls: 'condition' },
  clarification_needed: { label: 'Needs more info', cls: 'clarification' },
}

export default function MessageBubble({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="message user"><p>{message.text}</p></div>
      </div>
    )
  }

  const badge = QUERY_BADGES[message.query_type]
  let bodyText = message.text
  let interactionStatus = null
  if (message.query_type === 'drug_interaction') {
    const parsed = parseInteractionStatus(message.text)
    interactionStatus = parsed.status
    bodyText = parsed.text
  }

  // Parse any warning/precaution out so it's placed at the very top with yellow background
  const { warning, text: cleanText } = parseWarningPrecaution(bodyText)
  bodyText = cleanText

  const lowerText = (bodyText || '').toLowerCase()
  const isRefusal = message.refused ||
    lowerText.includes("cannot find reliable information") ||
    lowerText.includes("i'm sorry, but i can't answer") ||
    lowerText.includes("i'm sorry, but i cannot answer") ||
    lowerText.includes("i cannot answer that") ||
    lowerText.includes("i am not able to answer")

  const showRefusedNote = isRefusal && message.query_type !== 'unknown_drug' && message.query_type !== 'system_info' && !lowerText.includes("i'm sorry")
  const showCitations = !isRefusal && message.citations && message.citations.length > 0

  return (
    <div className="message-row assistant">
      <div className="avatar"><FlaskConical size={16} /></div>
      <div className={`message assistant ${message.query_type === 'clarification_needed' ? 'clarification' : ''}`}>
        {badge && <div className={`query-badge ${badge.cls}`}>{badge.label}</div>}
        {interactionStatus && (
          <div className={`interaction-status-badge ${interactionStatus.cls}`}>
            {interactionStatus.label}
          </div>
        )}
        {!badge && !interactionStatus && !isRefusal && showCitations && (
          <div className="answer-kicker"><Check size={13} /> SOURCE VERIFIED</div>
        )}

        {/* Top Warning/Precaution Banner with Yellow Background */}
        {warning && (
          <div className="patient-warning-banner">
            <div className="patient-warning-icon">⚠️</div>
            <div
              className="patient-warning-content"
              dangerouslySetInnerHTML={formatAnswerHtml(warning.replace(/^⚠️\s*/, ''))}
            />
          </div>
        )}

        {bodyText && <div className="message-text" dangerouslySetInnerHTML={formatAnswerHtml(bodyText)} />}
        {showCitations && <CitationsBlock citations={message.citations} />}
        {showRefusedNote && (
          <div className="refused-note">
            Could not find reliable information in the loaded documents. Please consult official prescribing information or a healthcare professional.
          </div>
        )}
      </div>
    </div>
  )
}
