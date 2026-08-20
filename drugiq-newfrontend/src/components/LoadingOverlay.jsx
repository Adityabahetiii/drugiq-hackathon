export default function LoadingOverlay({ message }) {
  return (
    <div className="loading-overlay">
      <div className="loading-spinner" />
      <div className="loading-text">{message}</div>
    </div>
  )
}
