const ICONS = { success: '✓', error: '!', info: 'i' }

export default function Toasts({ toasts }) {
  if (!toasts.length) return null
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div className={`toast ${t.type}`} key={t.id}>
          <span className="toast-icon">{ICONS[t.type] || ICONS.info}</span>
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  )
}
