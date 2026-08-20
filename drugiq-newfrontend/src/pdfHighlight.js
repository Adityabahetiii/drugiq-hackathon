// Ports the exact text-layer highlight-matching algorithm from the
// previous UI (templates/index.html). Matches a citation's source
// snippet(s) against pdf.js's rendered text layer so the reader sees
// precisely which words the answer came from, not just which page.
//
// The core difficulty: the snippet text came from a server-side extractor
// (pdfplumber) while the text layer comes from pdf.js in the browser — the
// two disagree on bullets, hyphenation, brackets, and incidental
// whitespace. Matching on an alphanumeric-only "stripped" skeleton of both
// sides makes the comparison resilient to all of that.

export function normalizeForMatch(s) {
  return s.replace(/\s+/g, ' ').trim().toLowerCase()
}

// Keeps only letters/digits (lowercased) plus a map back to the position
// in the original string for each kept character.
export function buildStrippedIndex(s) {
  let stripped = ''
  const map = []
  for (let i = 0; i < s.length; i++) {
    if (/[a-z0-9]/i.test(s[i])) {
      stripped += s[i].toLowerCase()
      map.push(i)
    }
  }
  return { stripped, map }
}

// Looks for the snippet's alphanumeric skeleton in the page's stripped
// text, backing off to shorter prefixes if the full snippet doesn't line
// up exactly (e.g. the chunk was truncated mid-sentence server-side).
export function findBestMatch(stripped, needleRaw) {
  const needle = (needleRaw.match(/[a-z0-9]/gi) || []).join('').toLowerCase()
  if (needle.length < 20) return null
  const lens = [...new Set([needle.length, 150, 90, 50].filter((l) => l <= needle.length && l >= 20))]
  for (const len of lens) {
    const idx = stripped.indexOf(needle.slice(0, len))
    if (idx !== -1) return { idx, len }
  }
  return null
}

// A '.' flanked by digits on both sides is a decimal point in a section
// reference like "(5.4)", not a sentence end.
function isDecimalPoint(combined, i) {
  return (
    combined[i] === '.' &&
    i > 0 && /[0-9]/.test(combined[i - 1]) &&
    i + 1 < combined.length && /[0-9]/.test(combined[i + 1])
  )
}

// Grows a raw character match out to full sentence/bullet/line boundaries
// so the reader sees a complete, readable claim instead of a fragment cut
// off mid-word or mid-clause.
export function expandToSentence(combined, start, end) {
  const maxRadius = 180
  const backLimit = Math.max(0, start - maxRadius)
  let s = backLimit
  for (let i = start - 1; i >= backLimit; i--) {
    const c = combined[i]
    if (c === '\n' || c === '•') { s = i + 1; break }
    if ((c === '.' || c === '!' || c === '?' || c === ':') && !isDecimalPoint(combined, i)) { s = i + 1; break }
  }
  while (s < start && (combined[s] === ' ' || combined[s] === '\n')) s++

  const fwdLimit = Math.min(combined.length, end + maxRadius)
  let e = fwdLimit
  for (let i = end; i < fwdLimit; i++) {
    const c = combined[i]
    if (c === '\n' || c === '•') { e = i; break }
    if ((c === '.' || c === '!' || c === '?' || c === ':') && !isDecimalPoint(combined, i)) { e = i + 1; break }
  }
  return { start: s, end: e }
}

// Picks which spans within [matchStart, matchEnd) should actually get
// highlighted, using word-boundary-aware rules rather than naive overlap
// (see templates/index.html's selectSpansForRange for the full reasoning).
export function selectSpansForRange(ranges, matchStart, matchEnd) {
  const overlapping = ranges.filter(
    (r) => r.end > matchStart && r.start < matchEnd && (r.span.textContent || '').trim() !== ''
  )
  if (!overlapping.length) return []

  const lastIdx = overlapping.length - 1
  const strict = overlapping.filter((r, idx) => {
    if (idx === lastIdx) return true
    if (idx === 0 && matchStart > r.start) return false
    const overlapStart = Math.max(r.start, matchStart)
    const overlapEnd = Math.min(r.end, matchEnd)
    const spanLen = Math.max(1, r.end - r.start)
    return (overlapEnd - overlapStart) / spanLen >= 0.4
  })

  return strict.length >= 2 ? strict : overlapping
}

// Runs the whole pipeline against a rendered text layer's <span> divs.
// Returns true if at least one snippet was found and highlighted.
export function highlightSnippetsInLayer(divs, items, snippets) {
  if (!divs.length) return false

  let combined = ''
  const ranges = []
  divs.forEach((span, i) => {
    const norm = normalizeForMatch((span && span.textContent) || '')
    if (!norm) return
    if (combined.length) combined += (items[i] && items[i].hasEOL) ? '\n' : ' '
    const start = combined.length
    combined += norm
    ranges.push({ start, end: combined.length, span })
  })
  if (!combined) return false

  const { stripped, map } = buildStrippedIndex(combined)

  let didHighlight = false
  let firstSpan = null
  snippets.forEach((snippetText) => {
    const match = findBestMatch(stripped, snippetText)
    if (!match) return
    const combinedStart = map[match.idx]
    const combinedEnd = map[match.idx + match.len - 1] + 1
    const { start: expStart, end: expEnd } = expandToSentence(combined, combinedStart, combinedEnd)
    selectSpansForRange(ranges, expStart, expEnd).forEach((r) => {
      r.span.classList.add('pdf-highlight')
      didHighlight = true
      if (!firstSpan) firstSpan = r.span
    })
  })

  if (firstSpan) firstSpan.scrollIntoView({ block: 'center', behavior: 'smooth' })
  return didHighlight
}
