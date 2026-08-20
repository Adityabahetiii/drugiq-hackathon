// Opens the source viewer as an in-page modal (matching the previous UI's
// behavior — clicking a citation never left the page or opened a new tab).
// This is a tiny pub-sub: App.jsx registers a listener once (its
// setSourceView state setter), and every citation click site
// (CitationsBlock, DocumentsModal, ...) just calls openSourceViewer(data)
// without needing to know App owns the modal state — no prop drilling
// through every component that can trigger a citation.
let listener = null

export function registerSourceViewerListener(fn) {
  listener = fn
}

export function openSourceViewer({
  file, sourceFile, page, drug, snippets, isOtc, sourceType, section, drugFacts, sourceUrl,
}) {
  if (!listener) return
  listener({
    file: file || sourceFile || '',
    page: page || 1,
    drug: drug || '',
    snippets: snippets || [],
    isOtc: !!isOtc,
    sourceType,
    section,
    drugFacts,
    sourceUrl,
  })
}
