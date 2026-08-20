import { useEffect, useRef, useState } from 'react'
import { Network, RefreshCw, Search, X } from './icons'
import { getKnowledgeGraph } from '../api'

const KG_NODE_STYLES = {
  drug: { color: { background: '#3b82f6', border: '#60a5fa' }, size: 30 },
  condition: { color: { background: '#10b981', border: '#34d399' }, size: 20 },
  class: { color: { background: '#8b5cf6', border: '#a78bfa' }, size: 18 },
  warning: { color: { background: '#ef4444', border: '#f87171' }, size: 16 },
}
const KG_EDGE_STYLES = {
  treats: { color: '#10b981', width: 1.5, dashes: false },
  interacts_with: { color: '#f59e0b', width: 1.5, dashes: false },
  contraindicated_with: { color: '#ef4444', width: 3, dashes: false },
  belongs_to: { color: '#8b5cf6', width: 1.5, dashes: false },
  shares_warning: { color: '#f87171', width: 1.5, dashes: [4, 4] },
}

export default function KnowledgeGraphModal({ onClose, toast }) {
  const containerRef = useRef(null)
  const networkRef = useRef(null)
  const nodesRef = useRef(null)
  const edgesRef = useRef(null)
  const graphDataRef = useRef(null)
  const searchDebounceRef = useRef(null)

  const [status, setStatus] = useState('loading') // loading | ready | empty | error
  const [errorMessage, setErrorMessage] = useState('')
  const [rebuilding, setRebuilding] = useState(false)
  const [searchValue, setSearchValue] = useState('')
  const [selection, setSelection] = useState(null) // { title, subtitle, sections: [{label, tags:[{text,cls}]}] }
  const [edgeTooltip, setEdgeTooltip] = useState(null) // { x, y, title, evidenceText, evidenceSource }

  useEffect(() => {
    load(false)
    return () => {
      if (networkRef.current) { networkRef.current.destroy(); networkRef.current = null }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function load(forceRebuild) {
    setStatus('loading')
    setSelection(null)
    setEdgeTooltip(null)
    try {
      const data = await getKnowledgeGraph(forceRebuild)
      if (data.error) {
        setErrorMessage(data.error)
        setStatus('error')
        toast?.(data.error, 'error')
        return
      }
      if (!data.nodes || data.nodes.length === 0) {
        setStatus('empty')
        return
      }
      renderGraph(data)
      setStatus('ready')
    } catch {
      setErrorMessage('Failed to load the knowledge graph.')
      setStatus('error')
      toast?.('Failed to load knowledge graph', 'error')
    }
  }

  async function handleRebuild() {
    setRebuilding(true)
    await load(true)
    setRebuilding(false)
  }

  function renderGraph(data) {
    graphDataRef.current = data
    const vis = window.vis
    if (!vis || !containerRef.current) return

    const visNodes = data.nodes.map((n) => {
      const style = KG_NODE_STYLES[n.type] || KG_NODE_STYLES.condition
      return {
        id: n.id,
        label: n.label,
        color: { background: style.color.background, border: style.color.border, highlight: style.color, hover: style.color },
        size: style.size,
        font: { color: '#1d2b30', size: 13 },
        borderWidth: 2,
        opacity: 1,
        _meta: n,
      }
    })

    const visEdges = data.edges.map((e, i) => {
      const style = KG_EDGE_STYLES[e.type] || { color: '#94a3b8', width: 1, dashes: false }
      return {
        id: 'e' + i,
        from: e.from,
        to: e.to,
        color: { color: style.color, opacity: 1 },
        width: style.width,
        dashes: style.dashes,
        arrows: { to: { enabled: true, scaleFactor: 0.55 }, from: { enabled: !!e.bidirectional, scaleFactor: 0.55 } },
        smooth: { type: 'curvedCW', roundness: 0.15 },
        title: e.label,
        _meta: e,
      }
    })

    nodesRef.current = new vis.DataSet(visNodes)
    edgesRef.current = new vis.DataSet(visEdges)

    const options = {
      physics: {
        enabled: true,
        stabilization: { iterations: 150 },
        barnesHut: { gravitationalConstant: -4000, springLength: 140, springConstant: 0.04 },
      },
      interaction: { hover: true, zoomView: true, dragView: true, tooltipDelay: 150 },
      nodes: { shape: 'dot' },
      edges: { smooth: true },
      layout: { improvedLayout: true },
    }

    if (networkRef.current) networkRef.current.destroy()
    networkRef.current = new vis.Network(containerRef.current, { nodes: nodesRef.current, edges: edgesRef.current }, options)
    networkRef.current.on('click', handleNetworkClick)
    networkRef.current.on('doubleClick', handleDoubleClick)
  }

  function handleNetworkClick(params) {
    setEdgeTooltip(null)
    if (params.nodes.length > 0) selectNode(params.nodes[0])
    else if (params.edges.length > 0) showEdgeTooltip(params.edges[0], params.pointer.DOM.x, params.pointer.DOM.y)
    else clearSelection()
  }

  function handleDoubleClick(params) {
    if (params.nodes.length === 0 && params.edges.length === 0) resetView()
  }

  function selectNode(nodeId) {
    highlightConnections(nodeId)
    showNodeInfo(nodesRef.current.get(nodeId)._meta)
    networkRef.current.focus(nodeId, { scale: 1.1, animation: { duration: 400 } })
  }

  function highlightConnections(nodeId) {
    const connectedNodes = new Set(networkRef.current.getConnectedNodes(nodeId))
    connectedNodes.add(nodeId)
    const connectedEdges = new Set(networkRef.current.getConnectedEdges(nodeId))
    nodesRef.current.forEach((n) => nodesRef.current.update({ id: n.id, opacity: connectedNodes.has(n.id) ? 1 : 0.12 }))
    edgesRef.current.forEach((e) => {
      const dim = !connectedEdges.has(e.id)
      edgesRef.current.update({ id: e.id, color: Object.assign({}, e.color, { opacity: dim ? 0.08 : 1 }) })
    })
  }

  function clearHighlightVisuals() {
    if (!nodesRef.current) return
    nodesRef.current.forEach((n) => nodesRef.current.update({ id: n.id, opacity: 1 }))
    edgesRef.current.forEach((e) => edgesRef.current.update({ id: e.id, color: Object.assign({}, e.color, { opacity: 1 }) }))
  }

  function clearSelection() {
    clearHighlightVisuals()
    setSelection(null)
    setEdgeTooltip(null)
  }

  function resetView() {
    clearSelection()
    if (networkRef.current) networkRef.current.fit({ animation: { duration: 400 } })
  }

  function showNodeInfo(node) {
    const graph = graphDataRef.current
    if (node.type === 'drug') {
      const details = (graph.drug_details || {})[node.full_name] || {}
      const interactions = details.interactions || []
      setSelection({
        title: node.label,
        subtitle: 'Drug',
        sections: [
          { label: 'Drug Class', tags: details.drug_class ? [{ text: details.drug_class }] : [] },
          { label: 'Conditions Treated', tags: (details.conditions_treated || []).map((c) => ({ text: c, cls: 'condition' })) },
          { label: 'Interactions', tags: interactions.map((i) => ({ text: i.drug, cls: i.type === 'contraindicated_with' ? 'contraindicated' : 'interacts' })) },
          { label: 'Warning', tags: (details.black_box_warnings || []).map((w) => ({ text: w, cls: 'warning' })) },
        ],
      })
    } else {
      const drugs = networkRef.current.getConnectedNodes(node.id).map((id) => nodesRef.current.get(id)._meta).filter((n) => n.type === 'drug')
      const typeLabel = node.type === 'condition' ? 'Condition' : node.type === 'class' ? 'Drug Class' : 'Warning'
      const relLabel = node.type === 'condition' ? 'Drugs That Treat This' : node.type === 'class' ? 'Drugs In This Class' : 'Drugs With This Warning'
      setSelection({
        title: node.label,
        subtitle: typeLabel,
        sections: [{ label: relLabel, tags: drugs.map((d) => ({ text: d.label })) }],
      })
    }
  }

  function showEdgeTooltip(edgeId, domX, domY) {
    const edge = edgesRef.current.get(edgeId)
    const meta = edge._meta
    const fromNode = nodesRef.current.get(meta.from)._meta
    const toNode = nodesRef.current.get(meta.to)._meta
    const evidence = meta.evidence
    setEdgeTooltip({
      x: Math.min(domX + 10, 400),
      y: Math.min(domY + 10, 300),
      title: `${fromNode.label} — ${meta.label} — ${toNode.label}`,
      evidenceText: evidence?.text || null,
      evidenceSource: evidence ? `${evidence.source_file || ''} — Page ${evidence.page}` : null,
    })
  }

  function handleSearch(value) {
    setSearchValue(value)
    clearTimeout(searchDebounceRef.current)
    searchDebounceRef.current = setTimeout(() => {
      if (!nodesRef.current || !value.trim()) return
      const q = value.trim().toLowerCase()
      const match = nodesRef.current.get().find((n) => n._meta.label.toLowerCase().includes(q))
      if (match) selectNode(match.id)
    }, 200)
  }

  return (
    <div className="window-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <section className="app-window kg-window">
        <header className="window-header">
          <div className="window-title">
            <span className="window-title-icon"><Network size={18} /></span>
            <div><strong>Drug knowledge graph</strong><small>Relationships extracted from prescribing documents</small></div>
          </div>
          <div className="kg-header-actions">
            <div className="kg-search"><Search size={14} /><input value={searchValue} onChange={(e) => handleSearch(e.target.value)} placeholder="Search drug or condition…" /></div>
            <button className="window-outline-btn" onClick={handleRebuild} disabled={rebuilding}><RefreshCw size={14} /> {rebuilding ? 'Rebuilding…' : 'Rebuild graph'}</button>
            <button className="window-close" onClick={onClose}><X size={18} /></button>
          </div>
        </header>

        <div className="kg-body">
          {status === 'loading' && (
            <div className="kg-status-state">
              <div className="overview-panel-loading"><span /><span /><span /></div>
              <span>Extracting relationships from prescribing documents…</span>
              <span className="kg-status-hint">First load may take 10–15 seconds — the LLM is reading each drug's prescribing document.</span>
            </div>
          )}
          {status === 'empty' && (
            <div className="kg-status-state">
              <Network size={28} />
              <span>No relationships yet. Upload drug PDFs, then click Rebuild Graph.</span>
            </div>
          )}
          {status === 'error' && (
            <div className="kg-status-state">
              <Network size={28} />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="kg-network" ref={containerRef} style={{ display: status === 'ready' ? 'block' : 'none' }} />

          {status === 'ready' && (
            <div className="kg-legend">
              <div className="kg-legend-title">Legend</div>
              <div className="kg-legend-section">
                <div className="kg-legend-item"><span className="kg-legend-dot" style={{ background: '#3b82f6', borderColor: '#60a5fa' }} />Drug</div>
                <div className="kg-legend-item"><span className="kg-legend-dot" style={{ background: '#10b981', borderColor: '#34d399' }} />Condition</div>
                <div className="kg-legend-item"><span className="kg-legend-dot" style={{ background: '#8b5cf6', borderColor: '#a78bfa' }} />Drug Class</div>
                <div className="kg-legend-item"><span className="kg-legend-dot" style={{ background: '#ef4444', borderColor: '#f87171' }} />Warning</div>
              </div>
              <div className="kg-legend-divider" />
              <div className="kg-legend-section">
                <div className="kg-legend-item"><span className="kg-legend-line" style={{ background: '#10b981' }} />treats</div>
                <div className="kg-legend-item"><span className="kg-legend-line" style={{ background: '#f59e0b' }} />interacts with</div>
                <div className="kg-legend-item"><span className="kg-legend-line kg-legend-line-thick" style={{ background: '#ef4444' }} />contraindicated with</div>
                <div className="kg-legend-item"><span className="kg-legend-line" style={{ background: '#8b5cf6' }} />belongs to</div>
                <div className="kg-legend-item"><span className="kg-legend-line kg-legend-line-dashed" style={{ borderColor: '#f87171' }} />shares warning</div>
              </div>
            </div>
          )}

          {selection && (
            <div className="kg-info-panel">
              <button className="kg-info-close" onClick={clearSelection}><X size={14} /></button>
              <div className="kg-info-title">{selection.title}</div>
              <div className="kg-info-subtitle">{selection.subtitle}</div>
              {selection.sections.map((sec) => (
                <div className="kg-info-section" key={sec.label}>
                  <div className="kg-info-label">{sec.label}</div>
                  <div className="kg-info-tag-row">
                    {sec.tags.length
                      ? sec.tags.map((t, i) => <span key={i} className={`kg-info-tag ${t.cls || ''}`}>{t.text}</span>)
                      : <span className="kg-info-empty">None found</span>}
                  </div>
                </div>
              ))}
            </div>
          )}

          {edgeTooltip && (
            <div className="kg-edge-tooltip" style={{ left: edgeTooltip.x, top: edgeTooltip.y }}>
              <button className="kg-edge-tooltip-close" onClick={() => setEdgeTooltip(null)}><X size={12} /></button>
              <div className="kg-edge-tooltip-title">{edgeTooltip.title}</div>
              {edgeTooltip.evidenceText ? (
                <>
                  <div>"{edgeTooltip.evidenceText}"</div>
                  <div className="kg-edge-tooltip-source">{edgeTooltip.evidenceSource}</div>
                </>
              ) : (
                <div className="kg-info-empty">No source text captured for this relationship.</div>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
