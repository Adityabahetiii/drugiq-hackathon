// Full-screen intro overlay, matching the previous UI's splash: dark
// background, gradient wordmark, animated loading bar, dismissible early
// by click. Rendered as a sibling of the app shell (not a route) so the
// app underneath mounts and starts fetching immediately while this covers
// it visually for the first ~2.3s.
export default function SplashScreen({ hidden, onDismiss }) {
  return (
    <div className={`splash-screen ${hidden ? 'hidden' : ''}`} onClick={onDismiss}>
      <div className="splash-title">DrugIQ</div>
      <div className="splash-tagline">AI-Powered Drug Information Assistant</div>
      <div className="splash-bar-wrap"><div className="splash-bar" /></div>
    </div>
  )
}

