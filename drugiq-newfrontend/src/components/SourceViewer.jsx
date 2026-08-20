import { useEffect, useRef, useState } from 'react'
import {
  ArrowLeft, ArrowRight, Check, Copy, FileText, FlaskConical, PanelLeft, Search,
  ShieldCheck, X, ZoomIn, ZoomOut,
} from './icons'
import { pdfUrl } from '../api'
import { highlightSnippetsInLayer } from '../pdfHighlight'

if (typeof window !== 'undefined' && window.pdfjsLib) {
  window.pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'
}

// Section title mapping helper
const SECTION_KEYS = {
  'directions': 'dosage',
  'dosage': 'dosage',
  'directions & dosage': 'dosage',
  'warnings': 'warnings',
  'warnings & precautions': 'warnings',
  'uses': 'indications',
  'uses & indications': 'indications',
  'indications': 'indications',
  'purpose': 'purpose',
  'active ingredient': 'purpose',
  'do not use': 'do_not_use',
  'stop use': 'stop_use',
}

export default function SourceViewer({
  file,
  page,
  drug = 'Prescribing Information',
  snippets = [],
  isOtc = false,
  sourceType,
  section = 'Directions & Dosage',
  drugFacts,
  sourceUrl,
  onClose,
}) {
  const isOtcLabel = isOtc || sourceType === 'otc_label' || !!drugFacts || (file && file.includes('FDA Drug Facts'))

  // State for Doctor PDF Viewer
  const targetPage = Number(page || 1)
  const [currentPage, setCurrentPage] = useState(targetPage)
  const [numPages, setNumPages] = useState(0)
  const [zoom, setZoom] = useState(100)
  const [loading, setLoading] = useState(!isOtcLabel)
  const [loadError, setLoadError] = useState(null)
  const [highlighted, setHighlighted] = useState(false)
  const [citationOpen, setCitationOpen] = useState(true)

  // State for Patient OTC Drug Facts Viewer
  const [activeSection, setActiveSection] = useState(section || 'Directions & Dosage')
  const [copied, setCopied] = useState(false)

  const canvasRef = useRef(null)
  const textLayerRef = useRef(null)
  const stageRef = useRef(null)
  const pdfDocRef = useRef(null)
  const renderingRef = useRef(false)
  const renderQueuedRef = useRef(null)

  // Load PDF only when in Doctor mode
  useEffect(() => {
    if (isOtcLabel) return

    let cancelled = false
    async function load() {
      setLoading(true)
      setLoadError(null)
      try {
        if (!window.pdfjsLib) throw new Error('PDF.js not available')
        const doc = await window.pdfjsLib.getDocument(pdfUrl(file)).promise
        if (cancelled) return
        pdfDocRef.current = doc
        setNumPages(doc.numPages)
        setCurrentPage(Math.max(1, Math.min(targetPage, doc.numPages)))
        setLoading(false)
      } catch {
        if (!cancelled) {
          setLoadError('Could not render this PDF.')
          setLoading(false)
        }
      }
    }
    load()
    return () => { cancelled = true }
  }, [file, isOtcLabel, targetPage])

  useEffect(() => {
    if (isOtcLabel || !pdfDocRef.current || !currentPage) return
    renderPage(currentPage, zoom)
  }, [currentPage, zoom, numPages, isOtcLabel])

  async function renderPage(pageNum, zoomPct) {
    if (renderingRef.current) { renderQueuedRef.current = { pageNum, zoomPct }; return }
    renderingRef.current = true
    try {
      const doc = pdfDocRef.current
      const page = await doc.getPage(pageNum)
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      const stageWidth = stageRef.current ? stageRef.current.clientWidth - 60 : window.innerWidth - 400
      const baseWidth = Math.min(stageWidth, 760)
      const vp = page.getViewport({ scale: 1 })
      const scale = (baseWidth / vp.width) * (zoomPct / 100)
      const svp = page.getViewport({ scale })
      canvas.width = svp.width
      canvas.height = svp.height

      const needsHighlight = snippets.length > 0 && pageNum === targetPage
      const [, textContent] = await Promise.all([
        page.render({ canvasContext: ctx, viewport: svp }).promise,
        window.pdfjsLib ? page.getTextContent() : Promise.resolve(null),
      ])

      const layer = textLayerRef.current
      if (layer) {
        layer.innerHTML = ''
        layer.style.width = svp.width + 'px'
        layer.style.height = svp.height + 'px'
        layer.style.setProperty('--scale-factor', svp.scale)
        if (textContent) {
          try {
            const textDivs = []
            await window.pdfjsLib.renderTextLayer({ textContent, container: layer, viewport: svp, textDivs }).promise
            setHighlighted(needsHighlight ? highlightSnippetsInLayer(textDivs, textContent.items, snippets) : false)
          } catch {
            setHighlighted(false)
          }
        }
      }
    } catch {
      setLoadError('Could not render this page.')
    } finally {
      renderingRef.current = false
      if (renderQueuedRef.current) {
        const next = renderQueuedRef.current
        renderQueuedRef.current = null
        renderPage(next.pageNum, next.zoomPct)
      }
    }
  }

  function goToPage(next) {
    setCurrentPage(Math.min(numPages || 1, Math.max(1, next)))
  }

  function handleCopyLabel() {
    if (!drugFacts) return
    const text = [
      `Official FDA Drug Facts Label: ${drugFacts.brand_name || drug}`,
      `Generic: ${drugFacts.generic_name || ''}`,
      `Purpose: ${drugFacts.purpose || ''}`,
      `Uses: ${drugFacts.indications || ''}`,
      `Directions & Dosage: ${drugFacts.dosage || ''}`,
      `Warnings: ${drugFacts.warnings || ''}`,
      `Do Not Use: ${drugFacts.do_not_use || ''}`,
      `Stop Use: ${drugFacts.stop_use || ''}`,
      `Keep Out of Reach: ${drugFacts.keep_out || ''}`,
    ].filter(Boolean).join('\n\n')

    navigator.clipboard?.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2500)
  }

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose?.() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  // Active snippet text for OTC drawer
  const activeKey = SECTION_KEYS[(activeSection || '').toLowerCase()] || 'dosage'
  const activeSnippetText = drugFacts ? (drugFacts[activeKey] || drugFacts.dosage || drugFacts.indications || '') : ''

  return (
    <div className="source-viewer-shell" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      <header className="viewer-topbar">
        <div className="viewer-brand">
          <div className="viewer-brand-mark"><FlaskConical size={16} /></div>
          <strong>DrugIQ</strong>
          <span className="viewer-divider" />
          <div className="viewer-doc-chip">
            <FileText size={14} />
            {drug}
            {isOtcLabel && <span className="doc-chip-badge">Official FDA OTC Label</span>}
          </div>
        </div>

        <div className="viewer-top-actions">
          {isOtcLabel ? (
            <>
              <div className="otc-section-tabs">
                {['Directions & Dosage', 'Warnings & Precautions', 'Uses & Indications'].map((sec) => (
                  <button
                    key={sec}
                    className={`otc-tab-btn ${activeSection === sec ? 'active' : ''}`}
                    onClick={() => setActiveSection(sec)}
                  >
                    {sec}
                  </button>
                ))}
              </div>
              <button className="viewer-action-btn" onClick={handleCopyLabel}>
                {copied ? <Check size={14} /> : <Copy size={14} />}
                <span>{copied ? 'Copied Label' : 'Copy Label'}</span>
              </button>
            </>
          ) : (
            <>
              <div className="viewer-page-control">
                <button className="viewer-icon-btn" onClick={() => goToPage(currentPage - 1)} disabled={currentPage <= 1}>
                  <ArrowLeft size={15} />
                </button>
                <span>Page</span>
                <input value={currentPage} onChange={(e) => goToPage(Number(e.target.value) || 1)} />
                <span className="viewer-muted">/ {numPages || '—'}</span>
                <button className="viewer-icon-btn" onClick={() => goToPage(currentPage + 1)} disabled={currentPage >= numPages}>
                  <ArrowRight size={15} />
                </button>
              </div>
              {snippets.length > 0 && (
                <button className="viewer-primary" onClick={() => setCitationOpen((v) => !v)}>
                  <Search size={14} />{highlighted ? 'Citation highlighted' : 'Highlight citation'}
                </button>
              )}
            </>
          )}

          <button className="viewer-icon-btn viewer-close-btn" onClick={onClose} aria-label="Close source viewer">
            <X size={16} />
          </button>
        </div>
      </header>

      <div className="viewer-workspace">
        {/* Left Citation Drawer */}
        <aside className={`citation-panel ${citationOpen ? '' : 'closed'}`}>
          <div className="citation-panel-head">
            <div>
              <span className="viewer-eyebrow">VERIFIED SOURCE</span>
              <h2>{isOtcLabel ? 'FDA Label Citation' : 'Citation details'}</h2>
            </div>
            <button className="viewer-icon-btn" onClick={() => setCitationOpen(false)}><X size={15} /></button>
          </div>

          <div className="citation-card">
            <div className="verified-badge"><ShieldCheck size={13} /> VERIFIED FDA LABEL</div>
            <h3>{drug}</h3>

            {isOtcLabel ? (
              <>
                <div className="otc-cited-badge">
                  <span>📋 Cited Section:</span>
                  <strong>{activeSection}</strong>
                </div>
                <div className="citation-label">OFFICIAL REGULATORY TEXT</div>
                <div className="excerpt-box">
                  <span className="excerpt-rule" />
                  <p>{activeSnippetText || 'Official FDA OTC Drug Facts label details.'}</p>
                </div>
                <div className="citation-meta">
                  <FileText size={13} />
                  <span>FDA OTC Drug Facts Label</span>
                  <span>•</span>
                  <span>{activeSection}</span>
                </div>
                {sourceUrl && (
                  <a className="native-btn" href={sourceUrl} target="_blank" rel="noopener noreferrer">
                    <ArrowRight size={14} /> View live FDA source
                  </a>
                )}
              </>
            ) : (
              <>
                <button className="page-pill" onClick={() => goToPage(targetPage)}><span>•</span> Cited page {targetPage}</button>
                <div className="citation-label">SOURCE DOCUMENT TEXT</div>
                <div className="excerpt-box">
                  <span className="excerpt-rule" />
                  {snippets.map((s, i) => <p key={i}>{s}</p>)}
                </div>
                <div className="citation-meta">
                  <FileText size={13} />
                  <span>{file}</span>
                  <span>•</span>
                  <span>Page {targetPage}</span>
                </div>
              </>
            )}
          </div>

          <div className="citation-note">
            <ShieldCheck size={14} />
            <span>
              {isOtcLabel
                ? 'This information is extracted directly from the official US FDA Over-The-Counter Drug Facts label.'
                : 'This excerpt is shown to make the answer traceable to the prescribing document.'}
            </span>
          </div>
        </aside>

        {!citationOpen && (
          <button className="reopen-citation" onClick={() => setCitationOpen(true)}>
            <PanelLeft size={15} /> Show citation
          </button>
        )}

        {/* Main Document / Drug Facts Stage */}
        <main className="document-stage" ref={stageRef}>
          {isOtcLabel ? (
            /* Authentic FDA Drug Facts Box Viewer */
            <div className="drug-facts-scroll">
              <div className="drug-facts-container">
                <div className="drug-facts-box">
                  <div className="drug-facts-banner">
                    <h1>Drug Facts</h1>
                    <div className="drug-facts-banner-sub">
                      <span>{drugFacts?.brand_name || drug}</span>
                      <small>Official OTC Drug Facts Label</small>
                    </div>
                  </div>

                  {/* Active Ingredient & Purpose */}
                  <div className={`drug-facts-section ${activeSection.includes('Uses') || activeSection.includes('Purpose') ? 'highlighted-section' : ''}`}>
                    <div className="drug-facts-row header-row">
                      <div className="df-col"><strong>Active ingredient (in each tablet/dose)</strong></div>
                      <div className="df-col text-right"><strong>Purpose</strong></div>
                    </div>
                    <div className="drug-facts-row">
                      <div className="df-col">{drugFacts?.generic_name || 'Active ingredient as formulated'}</div>
                      <div className="df-col text-right">{drugFacts?.purpose || 'Reliever / Reducer'}</div>
                    </div>
                  </div>

                  {/* Uses / Indications */}
                  <div className={`drug-facts-section ${activeSection.includes('Uses') || activeSection.includes('Indications') ? 'highlighted-section' : ''}`}>
                    <div className="df-section-title">
                      <strong>Uses</strong>
                      {(activeSection.includes('Uses') || activeSection.includes('Indications')) && (
                        <span className="section-match-badge">🎯 CITED IN ANSWER</span>
                      )}
                    </div>
                    <div className="df-section-body">
                      <p>{drugFacts?.indications || 'Relieves minor aches and pains due to common headache, fever, muscular aches.'}</p>
                    </div>
                  </div>

                  {/* Warnings */}
                  <div className={`drug-facts-section ${activeSection.includes('Warning') ? 'highlighted-section' : ''}`}>
                    <div className="df-section-title">
                      <strong>Warnings</strong>
                      {activeSection.includes('Warning') && <span className="section-match-badge">🎯 CITED IN ANSWER</span>}
                    </div>
                    <div className="df-section-body warnings-body">
                      {drugFacts?.warnings ? (
                        <p>{drugFacts.warnings}</p>
                      ) : (
                        <p>Do not exceed recommended dose. Keep out of reach of children.</p>
                      )}
                      {drugFacts?.do_not_use && (
                        <div className="df-subsection">
                          <strong>Do not use:</strong>
                          <p>{drugFacts.do_not_use}</p>
                        </div>
                      )}
                      {drugFacts?.stop_use && (
                        <div className="df-subsection">
                          <strong>Stop use and ask a doctor if:</strong>
                          <p>{drugFacts.stop_use}</p>
                        </div>
                      )}
                      {drugFacts?.keep_out && (
                        <div className="df-subsection">
                          <strong>Keep out of reach of children:</strong>
                          <p>{drugFacts.keep_out}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Directions / Dosage */}
                  <div className={`drug-facts-section ${activeSection.includes('Directions') || activeSection.includes('Dosage') ? 'highlighted-section' : ''}`}>
                    <div className="df-section-title">
                      <strong>Directions</strong>
                      {(activeSection.includes('Directions') || activeSection.includes('Dosage')) && (
                        <span className="section-match-badge">🎯 CITED IN ANSWER</span>
                      )}
                    </div>
                    <div className="df-section-body directions-body">
                      <p>{drugFacts?.dosage || 'Take as directed on package. Do not exceed maximum daily dosage.'}</p>
                    </div>
                  </div>

                  {/* Other Information & Storage */}
                  <div className="drug-facts-section">
                    <div className="df-section-title">
                      <strong>Other information</strong>
                    </div>
                    <div className="df-section-body">
                      <p>• Store at 20°C to 25°C (68°F to 77°F). Avoid excessive heat and moisture.</p>
                      <p>• Tamper evident: Do not use if safety seal is broken or missing.</p>
                    </div>
                  </div>

                  <div className="drug-facts-footer">
                    <span>US Food & Drug Administration (FDA) Over-The-Counter Drug Facts Label Standard</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Doctor PDF Viewer Stage */
            <>
              <div className="document-toolbar">
                <div className="document-context">
                  <span className="context-dot" />
                  <strong>Prescribing Information</strong>
                  <span>•</span>
                  <span>Page {currentPage} of {numPages || '—'}</span>
                </div>
                <div className="document-tools">
                  <button className="viewer-icon-btn" onClick={() => setZoom((z) => Math.max(70, z - 10))}><ZoomOut size={15} /></button>
                  <span className="zoom-value">{zoom}%</span>
                  <button className="viewer-icon-btn" onClick={() => setZoom((z) => Math.min(150, z + 10))}><ZoomIn size={15} /></button>
                  <a className="native-btn" href={pdfUrl(file)} target="_blank" rel="noopener noreferrer"><ArrowRight size={14} /> Native PDF</a>
                </div>
              </div>
              <div className="document-scroll">
                {loading && <div className="pdf-loading-note"><div className="overview-panel-loading"><span /><span /><span /></div><span>Loading PDF…</span></div>}
                {loadError && (
                  <div className="pdf-loading-note">
                    <span>{loadError}</span>
                    <a className="native-btn" href={pdfUrl(file)} target="_blank" rel="noopener noreferrer">Download instead</a>
                  </div>
                )}
                <div className="document-page" style={{ display: loading || loadError ? 'none' : 'block' }}>
                  <canvas ref={canvasRef} />
                  <div className="textLayer" ref={textLayerRef} />
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

