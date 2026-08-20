import { FileText, ShieldCheck } from './icons'
import { openSourceViewer } from '../openSource'

// Real, multi-drug citations from /api/chat.
// - For Doctors: Shows exact cited PDF pages (e.g. Page 1, Page 4).
// - For Patients: Shows exact FDA Drug Facts sections (e.g. Directions & Dosage, Warnings).
export default function CitationsBlock({ citations }) {
  if (!citations || !citations.length) return null

  return (
    <div className="citations-block">
      <div className="citations-block-label">
        <FileText size={12} /> Sources referenced
      </div>
      {citations.map((c, i) => {
        const isOtc = c.source_type === 'otc_label' || !!c.drug_facts || (!c.pages && !!c.sections)
        const sections = c.sections || (c.primary_section ? [c.primary_section] : ['Directions & Dosage', 'Warnings & Precautions'])

        return (
          <div className="citation-row" key={i}>
            <div className="citation-row-drug">
              <FileText size={13} /> {c.drug_name}
              {isOtc && <span className="otc-badge-sub">FDA OTC Label</span>}
              {isOtc && c.source_url && (
                <a
                  className="otc-live-source-link"
                  href={c.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  View official FDA source ↗
                </a>
              )}
            </div>
            <div className="citation-row-pills">
              {isOtc ? (
                sections.map((sec) => (
                  <button
                    key={sec}
                    className="page-pill otc-pill"
                    onClick={() =>
                      openSourceViewer({
                        isOtc: true,
                        sourceType: 'otc_label',
                        drug: c.drug_name,
                        section: sec,
                        drugFacts: c.drug_facts,
                        sourceFile: c.source_file,
                        sourceUrl: c.source_url,
                      })
                    }
                  >
                    <span>📋</span> {sec}
                  </button>
                ))
              ) : (
                (c.pages || []).map((p) => (
                  <button
                    key={p}
                    className="page-pill"
                    onClick={() =>
                      openSourceViewer({
                        isOtc: false,
                        file: c.source_file,
                        page: p,
                        drug: c.drug_name,
                        snippets: (c.page_snippets || {})[String(p)] || [],
                      })
                    }
                  >
                    <span>•</span> Page {p}
                  </button>
                ))
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

