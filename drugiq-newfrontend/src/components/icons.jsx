// Native character-based icon replacements, matching the style already
// established in App.jsx — keeps the project dependency-free beyond
// React/Vite (see README: no Lucide/icon library).
const glyph = (char) => ({ size = 16, className = '' }) => (
  <span
    className={'icon glyph-icon ' + className}
    style={{ fontSize: size, lineHeight: 1, display: 'inline-grid', placeItems: 'center' }}
    aria-hidden="true"
  >
    {char}
  </span>
)

export const AlertTriangle = glyph('!')
export const ArrowLeft = glyph('←')
export const ArrowRight = glyph('→')
export const ArrowDown = ({ size = 16, className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <line x1="12" y1="5" x2="12" y2="19" />
    <polyline points="19 12 12 19 5 12" />
  </svg>
)
export const BookOpen = glyph('▤')
export const Check = glyph('✓')
export const ChevronDown = glyph('⌄')
export const ChevronRight = glyph('›')
export const CircleHelp = glyph('?')
export const Clock3 = glyph('◷')
export const Copy = glyph('📋')
export const Database = glyph('▦')
export const FileText = glyph('▤')
export const FlaskConical = glyph('⚗')
export const FolderOpen = glyph('▱')
export const History = glyph('↶')
export const Info = glyph('i')
export const Menu = glyph('☰')
export const MessageCircle = glyph('◌')
export const Mic = glyph('●')
export const MicOff = glyph('⊘')
export const Paperclip = glyph('⌕')
export const PanelLeft = glyph('◫')
export const Plus = glyph('+')
export const RefreshCw = glyph('↻')
export const Search = glyph('⌕')
export const Send = glyph('➤')
export const Settings2 = glyph('⚙')
export const ShieldCheck = glyph('✓')
export const Trash2 = glyph('×')
export const Upload = glyph('↑')
export const UserRound = glyph('●')
export const X = glyph('×')
export const ZoomIn = glyph('+')
export const ZoomOut = glyph('−')
export const Network = glyph('◉')
export const Skull = glyph('✕')
export const Lock = glyph('🔒')
export const User = glyph('👤')
export const Stethoscope = glyph('🩺')
export const LogOut = glyph('↩')
export const ShieldAlert = glyph('🛡')
