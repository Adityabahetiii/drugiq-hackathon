import { useEffect, useState } from 'react'
import { BookOpen, ChevronRight, Database, FileText, Info, RefreshCw, ShieldCheck, Upload, X } from './icons'
import { getDrugs, rebuildDatabase, removeDrug } from '../api'
import { openSourceViewer } from '../openSource'

export default function DocumentsModal({ role = 'doctor', onClose, onUpload, onChanged, toast }) {
  const [drugs, setDrugs] = useState([])
  const [loading, setLoading] = useState(true)
  const [rebuilding, setRebuilding] = useState(false)

  async function refresh() {
    setLoading(true)
    try {
      const data = await getDrugs(role || 'doctor')
      setDrugs(data.drugs || [])
    } catch {
      toast?.('Failed to load documents', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [role])

  async function handleRebuild() {
    setRebuilding(true)
    try {
      const data = await rebuildDatabase()
      if (data.success) {
        toast?.(data.message, 'success')
        await refresh()
        onChanged?.()
      } else {
        toast?.(data.error || 'Rebuild failed', 'error')
      }
    } catch {
      toast?.('Error rebuilding database', 'error')
    } finally {
      setRebuilding(false)
    }
  }

  async function handleRemove(filename) {
    if (!window.confirm(`Remove ${filename} from the database?`)) return
    try {
      const data = await removeDrug(filename)
      if (data.success) {
        toast?.(data.message, 'success')
        await refresh()
        onChanged?.()
      } else {
        toast?.(data.error || 'Failed to remove', 'error')
      }
    } catch {
      toast?.('Error removing document', 'error')
    }
  }

  const totalChunks = drugs.reduce((sum, d) => sum + (d.chunk_count || 0), 0)

  return (
    <div className="window-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <section className="app-window documents-window">
        <header className="window-header">
          <div className="window-title">
            <span className="window-title-icon"><Database size={18} /></span>
            <div><strong>Documents & database</strong><small>Manage prescribing documents used by DrugIQ</small></div>
          </div>
          <button className="window-close" onClick={onClose}><X size={18} /></button>
        </header>
        <div className="window-body">
          <div className="window-section-head">
            <div>
              <span className="window-eyebrow">DOCUMENT LIBRARY</span>
              <h2>Clinical source documents</h2>
              <p>Upload official prescribing information, rebuild the index, and review what DrugIQ can cite.</p>
            </div>
            <div className="database-state"><span className="live-dot" /> {drugs.length ? 'Database active' : 'No documents yet'}</div>
          </div>

          <div className="document-actions-grid">
            <button className="document-action-card" onClick={onUpload}>
              <span className="action-icon"><Upload size={18} /></span>
              <span><strong>Upload document</strong><small>Add a prescribing information PDF</small></span>
              <ChevronRight size={17} />
            </button>
            <button className="document-action-card" onClick={handleRebuild} disabled={rebuilding}>
              <span className="action-icon"><RefreshCw size={18} /></span>
              <span><strong>{rebuilding ? 'Rebuilding…' : 'Rebuild database'}</strong><small>Re-index chunks and refresh retrieval</small></span>
              <ChevronRight size={17} />
            </button>
          </div>

          <div className="indexed-header">
            <div><span className="window-eyebrow">INDEXED DOCUMENTS</span><h3>{drugs.length} document{drugs.length === 1 ? '' : 's'} available</h3></div>
            <div className="indexed-summary"><span>{drugs.length} PDF{drugs.length === 1 ? '' : 's'}</span><span>{totalChunks} chunks</span></div>
          </div>

          {loading ? (
            <div className="overview-panel-loading"><span /><span /><span /></div>
          ) : drugs.length === 0 ? (
            <div className="window-info-strip">
              <Info size={16} />
              <div><strong>Nothing indexed yet</strong><span>Upload a prescribing information PDF to get started.</span></div>
            </div>
          ) : (
            drugs.map((d) => (
              <div className="indexed-document-card" key={d.source_file}>
                <div className="indexed-document-icon"><FileText size={21} /></div>
                <div className="indexed-document-info">
                  <strong>{d.drug_name}</strong>
                  <span>{d.source_file}</span>
                  <div className="indexed-meta">
                    <span>PDF</span><span>{d.page_count} pages</span><span>{d.chunk_count} chunks</span><span>Indexed</span>
                  </div>
                </div>
                <button
                  className="window-outline-btn"
                  onClick={() => openSourceViewer({ file: d.source_file, page: 1, drug: d.drug_name, snippets: [] })}
                >
                  <BookOpen size={15} /> View source
                </button>
                <button className="window-outline-btn danger" onClick={() => handleRemove(d.source_file)}>
                  <X size={15} /> Remove
                </button>
              </div>
            ))
          )}

          <div className="window-info-strip">
            <ShieldCheck size={16} />
            <div><strong>Source integrity</strong><span>DrugIQ answers are constrained to indexed prescribing documents and cite the page used for the response.</span></div>
          </div>
        </div>
      </section>
    </div>
  )
}
