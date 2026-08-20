"""
RAG Engine — using Groq (LLaMA 3) as the LLM
---------------------------------------------
Groq handles the chat/answer generation (free tier).
HuggingFace handles embeddings (free, local).

Enhanced with:
  - Multi-drug aware search (condition → drug queries)
  - Better citation format with multiple sources
  - Stronger safety guardrails
  - Query type detection
"""

from groq import Groq
from backend.vector_store import search, search_multi_drug, get_all_drug_names, get_drug_info
from backend.otc_lookup import (
    lookup_otc_drug,
    is_otc_drug,
    build_otc_context,
    find_otc_drugs_for_query,
    get_all_otc_names,
    find_active_otc_drugs_from_history,
    OTC_DATA,
)
from backend.guardrails import (
    detect_injection,
    detect_dosage_override,
    update_anomaly_count,
    is_session_flagged,
    validate_output,
    INJECTION_REFUSAL,
    DOSAGE_OVERRIDE_REFUSAL,
    FLAGGED_SESSION_WARNING,
    OUTPUT_SAFETY_REFUSAL,
    SAFETY_FENCE,
)
import difflib
import re

# ---------------------------------------------------------------------------
# Greeting / Conversational Detector & System Info Detector
# ---------------------------------------------------------------------------

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|hiya|good\s*(morning|afternoon|evening|day)|"
    r"what'?s?\s+up|sup|yo|greetings|help|thanks|thank\s*you|bye|goodbye|ok|okay|cool|great)\s*[!?.]*\s*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))


SYSTEM_METADATA_PATTERNS = [
    r"how\s+many\s+(databases?|datasets?|drugs?|medicines?|medications?|pdfs?|files?|documents?|records?|sources?|apis?)",
    r"what\s+(drugs?|medicines?|medications?|databases?|datasets?|pdfs?|files?|documents?|apis?)\s*(are\s+(available|indexed|loaded|in\s+the\s+database|we\s+using|am\s+i\s+using|present)|do\s+(we|you)\s+have|\?)",
    r"list\s+(all\s+)?(drugs?|medicines?|medications?|databases?|datasets?|pdfs?|files?|documents?|apis?)",
    r"which\s+(drugs?|medicines?|medications?|databases?|datasets?|pdfs?|files?|apis?)\s+do\s+(you|we)\s+have",
    r"what\s+(is|are)\s+the\s+(loaded|indexed|available)\s+(drugs?|databases?|datasets?|pdfs?|documents?|apis?)",
    r"show\s+(all\s+)?(drugs?|medicines?|medications?|databases?|datasets?|apis?)",
]

def detect_system_metadata_query(query: str) -> bool:
    q_lower = query.strip().lower()
    return any(re.search(pat, q_lower) for pat in SYSTEM_METADATA_PATTERNS)


# Words that should never be mistargeted as unknown drug brand names
NON_DRUG_COMMON_WORDS = {
    "injection", "oral", "subcutaneous", "intravenous", "iv", "im", "topical",
    "tablet", "tablets", "capsule", "capsules", "solution", "drops", "syrup", "cream",
    "dosage", "dose", "dosing", "schedule", "amount", "mg", "ml", "mcg", "g", "kg", "lbs",
    "side effects", "side effect", "adverse effects", "adverse effect", "reactions", "reaction",
    "warnings", "warning", "black box", "boxed warning", "precautions", "precaution",
    "contraindications", "contraindication", "interactions", "interaction",
    "pregnancy", "pregnant", "nursing", "breastfeeding", "lactation", "pediatric", "children", "child",
    "infant", "elderly", "adults", "adult", "weight", "age", "overdose", "storage",
    "yes", "no", "ok", "okay", "thanks", "thank you", "sure", "more", "tell me more",
    "database", "databases", "dataset", "datasets", "document", "documents", "pdf", "pdfs", "file", "files",
    "how many", "which one", "what is", "help", "who", "when", "why", "where",
}


# ---------------------------------------------------------------------------
# Query Type Detection
# ---------------------------------------------------------------------------

CONDITION_KEYWORDS = [
    "treat", "treatment", "treats", "treating",
    "cure", "cures", "curing",
    "used for", "indicated for", "approved for",
    "help with", "helps with",
    "manage", "manages", "managing",
    "disease", "condition", "disorder", "syndrome", "symptoms",
    "diagnosis", "diagnose",
    "what drug", "which drug", "which medicine", "what medicine",
    "recommend", "suggest", "prescribed for",
    "rheumatoid", "arthritis", "diabetes", "hypertension",
    "cancer", "infection", "pain", "inflammation",
    "depression", "anxiety", "asthma", "allergies",
]


def detect_query_type(query: str) -> str:
    """
    Detects whether the query is:
    - 'condition_to_drug': User asks about a condition/disease and wants drug suggestions
    - 'drug_info': User asks about a specific drug
    - 'general': General question (search normally)
    """
    query_lower = query.lower()

    # Check for condition-related keywords
    condition_score = sum(1 for kw in CONDITION_KEYWORDS if kw in query_lower)

    # Check for patterns like "what can treat X", "drugs for X"
    condition_patterns = [
        r"what\s+(drug|medicine|medication)s?\s+(can|could|would|should|to)\s+",
        r"(drug|medicine|medication)s?\s+(for|to treat|that treat)",
        r"(treat|cure|manage|help with)\s+\w+",
        r"what\s+(treats?|cures?|helps?)\s+",
        r"(is there|are there)\s+(a|any)\s+(drug|medicine|medication|treatment)",
    ]

    for pattern in condition_patterns:
        if re.search(pattern, query_lower):
            return "condition_to_drug"

    if condition_score >= 2:
        return "condition_to_drug"

    return "general"


# ---------------------------------------------------------------------------
# Drug Matching — dynamic, reads known_drugs from the vector store at query
# time so newly-uploaded PDFs are picked up automatically with no hardcoding.
# ---------------------------------------------------------------------------

def find_matching_drug(query: str, known_drugs: list[str]) -> tuple[str | None, list[str], str | None]:
    """
    Figures out which known drug (if any) a query is about.

    Returns (matched_drug_name, [], None) when a drug is identified — via
    direct substring containment or a fuzzy/typo match (e.g. "rinoq" ->
    "Rinvoq").
    Returns (None, closest_matches, tried_word) when the query has a
    drug-like word that doesn't quite match anything, so the caller can show
    "did you mean?".
    Returns (None, [], None) when no drug-like word is present at all (e.g.
    a condition query), so the caller can fall back to an unscoped search.
    """
    known_drugs_lower = [d.lower() for d in known_drugs]
    query_lower = query.lower()
    query_words = re.findall(r"[a-z0-9']+", query_lower)

    # 1: any known drug name contains a query word, or vice versa
    # e.g. "rinvoq" matches "rinvoq (upadacitinib)"
    for word in query_words:
        if len(word) < 4 or word in NON_DRUG_COMMON_WORDS:
            continue
        for i, drug in enumerate(known_drugs_lower):
            # Check base drug name without parentheses
            base_drug = re.sub(r"\(.*?\)", "", drug).strip()
            if word in base_drug or (len(base_drug) >= 4 and base_drug in query_lower):
                return known_drugs[i], [], None
            if word in drug or drug in query_lower:
                return known_drugs[i], [], None

    # 2: fuzzy match a query word against the individual words of a drug name
    # catches typos like "rinoq" or "rinvok"
    for word in query_words:
        if len(word) < 4 or word in NON_DRUG_COMMON_WORDS:
            continue
        for i, drug in enumerate(known_drugs_lower):
            drug_words = [w for w in re.findall(r"[a-z0-9]+", drug) if len(w) >= 3]
            if difflib.get_close_matches(word, drug_words, n=1, cutoff=0.75):
                return known_drugs[i], [], None

    # 3: fuzzy match against full drug names — close but not confident enough
    # to auto-select, so surface as suggestions instead
    for word in query_words:
        if len(word) < 4 or word in NON_DRUG_COMMON_WORDS:
            continue
        matches = difflib.get_close_matches(word, known_drugs_lower, n=3, cutoff=0.6)
        if matches:
            closest = [known_drugs[known_drugs_lower.index(m)] for m in matches]
            return None, closest, word

    return None, [], None


def detect_drug_drug_query(query: str, known_drugs: list[str]) -> dict:
    """
    Detects whether a query names two or more known drugs — the shape of a
    drug-drug interaction question ("Can I take Rinvoq with Humira?").
    Reuses the same substring/fuzzy matching as find_matching_drug, but
    collects every distinct drug mentioned instead of stopping at the first.

    Returns {"is_interaction": bool, "drugs": [matched drug names, in the
    order they were found]}.
    """
    known_drugs_lower = [d.lower() for d in known_drugs]
    query_lower = query.lower()
    query_words = re.findall(r"[a-z0-9']+", query_lower)

    found = []
    found_set = set()

    def add(idx):
        if idx not in found_set:
            found_set.add(idx)
            found.append(known_drugs[idx])

    for word in query_words:
        if len(word) < 4 or word in NON_DRUG_COMMON_WORDS:
            continue
        for i, drug in enumerate(known_drugs_lower):
            base_drug = re.sub(r"\(.*?\)", "", drug).strip()
            if (len(base_drug) >= 4 and base_drug in query_lower) or word in base_drug:
                add(i)
            elif word in drug or drug in query_lower:
                add(i)

    for word in query_words:
        if len(word) < 4 or word in NON_DRUG_COMMON_WORDS:
            continue
        for i, drug in enumerate(known_drugs_lower):
            if i in found_set:
                continue
            drug_words = [w for w in re.findall(r"[a-z0-9]+", drug) if len(w) >= 3]
            if difflib.get_close_matches(word, drug_words, n=1, cutoff=0.75):
                add(i)

    return {"is_interaction": len(found) >= 2, "drugs": found}


# Filler/question words stripped away when checking whether a query is a
# generic "tell me about this drug" ask vs. a specific question. Deliberately
# includes common typos ("id" for "is") so fuzzy phrasing still counts.
GENERIC_FILLER_WORDS = {
    "what", "whats", "what's", "is", "id", "are", "tell", "me", "about",
    "give", "information", "info", "on", "of", "the", "a", "an", "please",
    "describe", "detail", "details", "summary", "overview", "know", "want",
    "to", "can", "you", "explain", "show", "for", "this", "drug", "medicine",
}


def is_generic_overview_query(question: str, drug_name: str) -> bool:
    """
    True when, after removing the matched drug's own name and generic
    filler/question words, nothing meaningful is left — i.e. the user asked
    "What is Rinvoq?" / "Rinvoq overview" / just "Rinvoq", not a specific
    question like "What is the dosage of Rinvoq?".
    """
    cleaned = re.sub(r"[?!.]+", " ", question.lower())
    words = re.findall(r"[a-z0-9']+", cleaned)

    drug_words = {w for w in re.findall(r"[a-z0-9]+", drug_name.lower()) if len(w) > 2}

    leftover = []
    for word in words:
        if word in drug_words:
            continue
        if any(difflib.SequenceMatcher(None, word, dw).ratio() > 0.82 for dw in drug_words):
            continue
        if word in GENERIC_FILLER_WORDS:
            continue
        leftover.append(word)

    return len(leftover) == 0


# Structural (not fuzzy) patterns that shape an "overview of X" ask. Used
# only to guess the drug-name-shaped subject of a query that didn't match
# any known drug at all, so it can be reported as an unknown drug rather
# than silently falling through to an unscoped, likely-irrelevant search.
OVERVIEW_SHAPE_PATTERNS = [
    r"^what'?s\s+",
    r"^what\s+is\s+",
    r"^what\s+id\s+",
    r"^tell\s+me\s+about\s+",
    r"^give\s+me\s+(information|info)\s+(on|about)\s+",
    r"^describe\s+",
    r"^info(rmation)?\s+(on|about)\s+",
]


def extract_overview_subject(question: str, has_active_drug: bool = False) -> str | None:
    """
    If the question is shaped like a generic "tell me about X" ask, returns
    the candidate subject X. Otherwise None.
    If has_active_drug is True, short follow-ups or attribute questions
    are not treated as unknown drugs.
    """
    q = re.sub(r"[?!.]+$", "", question.strip().lower()).strip()
    if not q or q in NON_DRUG_COMMON_WORDS:
        return None

    # Pure numbers or numbers with units (e.g. "80", "50kg", "100", "0.25") are NOT drug names
    if re.match(r"^[\d\s.,/+-]+$", q) or re.match(r"^\d+(?:\.\d+)?\s*(kg|kgs|lbs|mg|ml|mcg|yrs|years|old)?$", q):
        return None

    # If active drug exists, only explicit "what is X" or "tell me about X"
    # can introduce a new candidate subject.
    if has_active_drug:
        for pat in OVERVIEW_SHAPE_PATTERNS:
            stripped = re.sub(pat, "", q, count=1)
            if stripped != q:
                stripped = re.sub(r"\s+(overview|information|info|details|summary)\s*$", "", stripped).strip()
                if stripped and 1 <= len(stripped.split()) <= 3 and stripped not in NON_DRUG_COMMON_WORDS:
                    return stripped
        return None

    subject = None
    for pat in OVERVIEW_SHAPE_PATTERNS:
        stripped = re.sub(pat, "", q, count=1)
        if stripped != q:
            stripped = re.sub(r"\s+(overview|information|info|details|summary)\s*$", "", stripped).strip()
            subject = stripped or None
            break
    else:
        stripped_suffix = re.sub(r"\s+(overview|information|info|details|summary)\s*$", "", q).strip()
        if stripped_suffix != q:
            subject = stripped_suffix or None
        elif len(q.split()) <= 3 and q not in NON_DRUG_COMMON_WORDS:
            words = q.split()
            if not any(w in NON_DRUG_COMMON_WORDS for w in words):
                subject = q

    if subject and 1 <= len(subject.split()) <= 3 and subject not in NON_DRUG_COMMON_WORDS:
        return subject
    return None


# ---------------------------------------------------------------------------
# Citation Builder
# ---------------------------------------------------------------------------

def make_snippet(text: str, max_len: int = 220) -> str:
    """
    Trims a chunk's text down to a short anchor phrase. The frontend
    searches the rendered PDF page's text layer for this phrase so it
    can highlight exactly where an answer came from, instead of just
    jumping to the page.
    """
    snippet = " ".join(text.split())
    if len(snippet) <= max_len:
        return snippet
    truncated = snippet[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated


def build_citations(chunks: list[dict]) -> list[dict]:
    """
    Builds a deduplicated list of citations from the retrieved chunks.
    Groups by drug name and collects all cited pages, plus the exact
    source text snippet(s) per page so the PDF viewer can highlight them.
    """
    citation_map = {}
    for chunk in chunks:
        key = (chunk["drug_name"], chunk["source_file"])
        if key not in citation_map:
            citation_map[key] = {
                "drug_name": chunk["drug_name"],
                "source_file": chunk["source_file"],
                "pages": set(),
                "page_snippets": {}
            }
        info = citation_map[key]
        page = chunk["page_number"]
        info["pages"].add(page)
        info["page_snippets"].setdefault(page, []).append(make_snippet(chunk["text"]))

    citations = []
    for info in citation_map.values():
        pages = sorted(info["pages"])
        page_str = ", ".join(str(p) for p in pages)
        citations.append({
            "drug_name": info["drug_name"],
            "source_file": info["source_file"],
            "pages": pages,
            "page_snippets": {str(p): info["page_snippets"][p] for p in pages},
            "text": f"{info['drug_name']} Prescribing Information — Page(s) {page_str}"
        })

    return citations


SOURCES_USED_RE = re.compile(r"SOURCES_USED:\s*([^\n]*)", re.IGNORECASE)


def extract_sources_used(answer: str, num_chunks: int) -> list[int]:
    """
    Parses the model's "SOURCES_USED: 2,3" directive into 0-based indices
    into the `chunks` list that was retrieved for this question.

    Small/fast models reliably copy a short integer like a source number,
    but frequently botch transcribing an arbitrary page number (e.g. writing
    "Page 7" when it meant source #7, whose real page was 17) — citing by
    index instead of asking the model to recall a page number sidesteps
    that failure mode entirely; the real page number is then read straight
    from trusted chunk metadata, never from model output.
    """
    match = SOURCES_USED_RE.search(answer)
    if not match:
        return []
    raw = match.group(1).strip().lower()
    if not raw or raw.startswith("none"):
        return []
    indices = {int(n) - 1 for n in re.findall(r"\d+", raw)}
    return sorted(i for i in indices if 0 <= i < num_chunks)


def strip_sources_directive(answer: str) -> str:
    """Removes the machine-readable SOURCES_USED line before display."""
    return SOURCES_USED_RE.sub("", answer).strip()


def build_citation_line(citations: list[dict]) -> str:
    """
    Builds the human-readable "📄 Source: ..." line(s) shown under an
    answer, straight from the verified citations object — guaranteeing the
    page numbers in the answer text match the clickable citation chips,
    since both come from the exact same data.
    """
    lines = []
    for c in citations:
        pages = c["pages"]
        if len(pages) == 1:
            lines.append(f"📄 Source: {c['drug_name']} — Page {pages[0]}")
        else:
            page_str = ", ".join(str(p) for p in pages)
            lines.append(f"📄 Source: {c['drug_name']} — Pages {page_str}")
    return "\n".join(lines)


def _kg_interaction_fact_chunk(drugs: list[str]) -> dict | None:
    """
    If the cached knowledge graph (built by the Knowledge Graph panel) already
    has a verified drug<->drug edge for this exact pair, wraps it as a
    synthetic chunk in the same shape as a vector-store chunk so it flows
    through the normal retrieval -> SOURCES_USED -> citation pipeline as a
    high-confidence fact, instead of requiring the LLM to rediscover the
    relationship from raw prose alone. Only fires for a real, evidenced edge
    (contraindicated_with / interacts_with) — deliberately does not
    fabricate a "no relationship" fact for the negative case, since that
    would need a fake source_file with no real page to cite.

    Imports knowledge_graph locally: that module imports from this one
    (find_matching_drug, make_snippet), so importing it back at module load
    time would create a circular import. By call time both modules are
    already fully loaded, so this is safe.
    """
    if len(drugs) < 2:
        return None
    from backend.knowledge_graph import load_cached_graph
    graph = load_cached_graph()
    if not graph:
        return None

    ids = {f"drug:{drugs[0]}", f"drug:{drugs[1]}"}
    for edge in graph.get("edges", []):
        if not edge.get("bidirectional") or {edge.get("from"), edge.get("to")} != ids:
            continue
        ev = edge.get("evidence")
        if not ev:
            return None
        name_a = drugs[0].split("(")[0].strip()
        name_b = drugs[1].split("(")[0].strip()
        verb = (
            "are explicitly CONTRAINDICATED for use together"
            if edge.get("type") == "contraindicated_with"
            else "have a documented interaction requiring caution"
        )
        # Evidence may have come from either drug's own document — attribute
        # the citation to whichever drug's source_file it actually matches,
        # not always drugs[0].
        source_drug = next(
            (n["full_name"] for n in graph.get("nodes", [])
             if n.get("type") == "drug" and n.get("source_file") == ev.get("source_file")),
            drugs[0],
        )
        return {
            "drug_name": source_drug,
            "source_file": ev["source_file"],
            "page_number": ev["page"],
            "text": (
                f"VERIFIED FACT (from cross-referenced prescribing documents): "
                f"{name_a} and {name_b} {verb}. Supporting text: \"{ev['text']}\""
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Core RAG Function & Sanitize Helper
# ---------------------------------------------------------------------------

def _empty_response(query_type: str, drug_name: str | None = None, refused: bool = False) -> dict:
    return {
        "answer": "",
        "citations": [],
        "citation": None,
        "source_file": None,
        "page_number": None,
        "refused": refused,
        "query_type": query_type,
        "drug_name": drug_name,
    }


def _build_sanitized_messages(system_prompt: str, chat_history: list[dict] | None, user_question: str) -> list[dict]:
    """
    Constructs messages payload for LLM calls with sanitized conversation history
    and the SAFETY_FENCE system message immediately preceding the current user question.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (chat_history or []):
        content = msg.get("content", "") or ""
        is_injected_hist, _ = detect_injection(content)
        override_hist = (
            detect_dosage_override(content)
            if msg.get("role") == "user"
            else False
        )
        if is_injected_hist or override_hist:
            messages.append({
                "role": msg.get("role", "user"),
                "content": "[Message removed by safety filter]"
            })
        else:
            messages.append(msg)

    messages.append({
        "role": "system",
        "content": SAFETY_FENCE
    })
    messages.append({
        "role": "user",
        "content": user_question
    })
    return messages


# ---------------------------------------------------------------------------
# Contextual Memory & Query Rewriting
# ---------------------------------------------------------------------------

VAGUE_REFERENCE_RE = re.compile(
    r"\b(it's|its|it|this drug|that drug|this medication|that medication|this medicine|the drug|the medication)\b",
    re.IGNORECASE
)


def get_active_drug_from_history(chat_history: list[dict], known_drugs: list[str]) -> str | None:
    """
    Scans chat history backwards (most recent messages first) to identify
    which drug is currently the active topic of conversation.
    Supports both prescription drugs and OTC medications.
    """
    if not chat_history:
        return None

    # Search across known prescription drugs and all known OTC drug names/aliases
    search_drugs = list(known_drugs) if known_drugs else []
    otc_names = get_all_otc_names()
    for n in otc_names:
        if n not in search_drugs:
            search_drugs.append(n)

    for msg in reversed(chat_history):
        content = msg.get("content", "") or ""
        if not content or content.startswith("[displayed"):
            continue
        # 1. Direct match with find_matching_drug
        matched, _, _ = find_matching_drug(content, search_drugs)
        if matched:
            return matched
        # 2. Check each known drug substring
        for drug in search_drugs:
            base_name = re.sub(r"\(.*?\)", "", drug).strip()
            if base_name and len(base_name) >= 3 and re.search(rf"\b{re.escape(base_name)}\b", content, re.IGNORECASE):
                return drug
            if re.search(rf"\b{re.escape(drug)}\b", content, re.IGNORECASE):
                return drug
    return None



def get_last_substantive_user_query(chat_history: list[dict]) -> str | None:
    """
    Returns the last user question from history that had substantive intent
    (e.g., asking about dosage, schedule, side effects).
    """
    if not chat_history:
        return None
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            words = content.split()
            if len(words) >= 3 or any(w in content.lower() for w in ["what", "how", "dose", "dosage", "schedule", "can", "is", "side", "effect", "take", "give", "administer"]):
                return content
    return None


def rewrite_query(question: str, chat_history: list[dict], known_drugs: list[str]) -> tuple[str, str | None]:
    """
    Resolves pronouns, short follow-up answers (e.g. "injection", "oral", "50 kg"),
    and drug-less follow-up questions (e.g. "what are the side effects?")
    against conversation history and the active drug.

    Returns (rewritten_question, active_drug).
    """
    active_drug = get_active_drug_from_history(chat_history, known_drugs)
    
    # If the question already explicitly matches one of the known drugs, don't rewrite to an old drug
    matched_in_q, _, _ = find_matching_drug(question, known_drugs)
    if matched_in_q:
        return question, matched_in_q

    last_user_q = get_last_substantive_user_query(chat_history)
    clean_base_q = re.sub(r"[?.!]+$", "", last_user_q).strip() if last_user_q else ""
    q_lower = question.strip().lower()

    if not active_drug:
        weight_match = WEIGHT_VALUE_RE.search(question)
        condition_match = next((c for c in CONDITION_NAME_HINTS if c in q_lower), None)
        route_match = next((r for r in ROUTE_HINTS if r in q_lower), None)
        if (weight_match or condition_match or route_match) and len(question.split()) <= 6 and clean_base_q:
            if weight_match:
                return f"{clean_base_q} weighing {weight_match.group(1)} {weight_match.group(2)}?", None
            if condition_match:
                return f"{clean_base_q} for {condition_match}?", None
            if route_match:
                return f"{clean_base_q} via {route_match} route?", None
        return question, None

    clean_drug_name = re.sub(r"\(.*?\)", "", active_drug).strip()

    # Case 1: Vague pronoun replacement ("it", "this drug", "that medication", etc.)
    if VAGUE_REFERENCE_RE.search(question):
        rewritten = VAGUE_REFERENCE_RE.sub(clean_drug_name, question, count=1)
        return rewritten, active_drug

    # Case 2: Short route of administration answer (e.g. "injection", "oral", "subcutaneously", "by injection", "via oral")
    route_match = next((r for r in ROUTE_HINTS if r in q_lower), None)
    if route_match and len(question.split()) <= 4:
        if last_user_q:
            return f"{last_user_q} via {route_match} route", active_drug
        return f"What is the starting dose and administration schedule for {clean_drug_name} via {route_match} route?", active_drug

    # Case 3: Weight or Age or Condition answer to a prior question/clarification
    weight_match = WEIGHT_VALUE_RE.search(question)
    condition_match = next((c for c in CONDITION_NAME_HINTS if c in q_lower), None)
    if (weight_match or condition_match) and len(question.split()) <= 6:
        if weight_match:
            if clean_base_q:
                return f"{clean_base_q} weighing {weight_match.group(1)} {weight_match.group(2)}?", active_drug
            return f"What is the {clean_drug_name} dosage for a patient weighing {weight_match.group(1)} {weight_match.group(2)}?", active_drug
        if condition_match:
            if clean_base_q:
                return f"{clean_base_q} for {condition_match}?", active_drug
            return f"What is the {clean_drug_name} dosage for {condition_match}?", active_drug

    # Case 4: Follow-up question about drug attributes without naming the drug
    # e.g., "what are the side effects?", "how to store it?", "what is the starting dose?", "is it safe during pregnancy?"
    attribute_keywords = [
        "side effect", "adverse", "dose", "dosage", "starting dose", "schedule", "administer",
        "contraindication", "warning", "black box", "pregnant", "pregnancy", "children", "pediatric",
        "elderly", "interaction", "indication", "storage", "how to take", "how to use", "cost", "safe"
    ]
    if any(kw in q_lower for kw in attribute_keywords) or len(question.split()) <= 4:
        if clean_drug_name.lower() not in q_lower:
            history_route = next((r for r in ROUTE_HINTS if any(r in (m.get("content", "") or "").lower() for m in reversed(chat_history))), None)
            route_suffix = f" via {history_route} route" if (history_route and any(k in q_lower for k in ["dose", "dosage", "schedule", "administer"])) else ""
            if question.endswith("?"):
                return f"{question[:-1]} for {clean_drug_name}{route_suffix}?", active_drug
            else:
                return f"{question} for {clean_drug_name}{route_suffix}", active_drug

    return question, active_drug


# ---------------------------------------------------------------------------
# Smart Clarification Dialogue — catches answers that are only correct
# conditional on information the user never gave (weight, condition, route)
# and asks for it instead of handing back a generic, potentially unsafe
# dosage answer.
# ---------------------------------------------------------------------------

DOSAGE_QUESTION_KEYWORDS = [
    "dosage", "dosing", "starting dose", "maintenance dose", "how much to take",
    "how much to give", "how much should i take", "how much is recommended",
    "how many mg", "how many ml", "how many tablets", "how many pills", "how many doses",
    "administer", "administration", "schedule"
]

WEIGHT_BASED_DOSING_RE = re.compile(r"mg/kg|mg/m²|mg/m2|per\s+kg|per\s+kilogram", re.IGNORECASE)
WEIGHT_VALUE_MENTION_RE = re.compile(r"\d+(?:\.\d+)?\s*kg\b", re.IGNORECASE)
WEIGHT_GIVEN_RE = re.compile(r"\d+(\.\d+)?\s*(kgs?|lbs?|pounds?|kilograms?)", re.IGNORECASE)
AGE_MENTION_RE = re.compile(
    r"\d+\s*[-]?\s*(years?|yrs?)\s*[-]?\s*old|\d+\s*y/o|\bchild\b|\bpediatric\b|\binfant\b|\belderly\b|\badult\b",
    re.IGNORECASE
)

ROUTE_HINTS = ["subcutaneous", "intravenous", "oral", "injection"]

CONDITION_NAME_HINTS = [
    "rheumatoid arthritis", "psoriatic arthritis", "ankylosing spondylitis",
    "crohn's disease", "crohns disease", "crohn disease", "ulcerative colitis",
    "plaque psoriasis", "psoriasis", "juvenile idiopathic arthritis",
    "hidradenitis suppurativa", "uveitis", "melanoma", "osteosarcoma",
    "giant cell arteritis", "diabetes", "cardiovascular disease",
]


def extract_missing_context(question: str, answer: str, chat_history: list[dict], matched_drug: str | None = None) -> str | None:
    """
    Looks at the answer the model just generated (and the question that
    produced it) to decide whether it's only a generic/partial answer
    because something the label conditions dosing on — weight, condition,
    route — was never given. Returns a clarifying question to ask instead,
    or None if the answer is already fully actionable as-is.
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    is_dosage_question = any(kw in q_lower for kw in DOSAGE_QUESTION_KEYWORDS) or \
                         (re.search(r"\b(dose|dosage)\b", q_lower) is not None) or \
                         ("how much" in q_lower and any(w in q_lower for w in ["take", "give", "use", "administer", "prescribe"]))

    if not is_dosage_question:
        return None

    has_weight_in_question = bool(WEIGHT_GIVEN_RE.search(question))
    distinct_weight_mentions = set(WEIGHT_VALUE_MENTION_RE.findall(a_lower))
    is_weight_based_answer = bool(WEIGHT_BASED_DOSING_RE.search(answer)) or len(distinct_weight_mentions) >= 2

    # Check if patient is adult (e.g. 40 years old, >= 18)
    age_num_match = re.search(r"(\d+)\s*[-]?\s*(?:years?|yrs?)\s*[-]?\s*old|\b(\d+)\s*y/o\b", question, re.IGNORECASE)
    age_val = int(age_num_match.group(1) or age_num_match.group(2)) if age_num_match else None
    is_adult = (age_val is not None and age_val >= 18) or ("adult" in q_lower or "elderly" in q_lower)

    # For adult patients, adult dosing is typically fixed dose (e.g. 40 mg every 2 weeks).
    # Only ask for weight if it's pediatric/child dosing or strictly weight-based without adult fixed dose.
    if is_weight_based_answer and not has_weight_in_question and not is_adult:
        if AGE_MENTION_RE.search(question) or "child" in q_lower or "pediatric" in q_lower:
            return (
                "You mentioned the patient's age. Since pediatric dosing for this drug is dosed by weight (mg/kg), "
                "could you also share the patient's weight in kg for a precise dose?"
            )
        return "To calculate the exact dose, I need the patient's weight in kg. Could you provide that?"

    conditions_in_answer = {c for c in CONDITION_NAME_HINTS if c in a_lower}
    condition_in_question = any(c in q_lower for c in CONDITION_NAME_HINTS)
    if len(conditions_in_answer) >= 2 and not condition_in_question:
        return (
            "This drug has different doses depending on the condition being treated. "
            "Which condition are you asking about?"
        )

    routes_in_answer = {r for r in ROUTE_HINTS if r in a_lower}
    route_in_question = any(r in q_lower for r in ROUTE_HINTS)
    route_in_history = any(
        any(r in (msg.get("content", "") or "").lower() for r in ROUTE_HINTS)
        for msg in (chat_history or [])
    )
    if len(routes_in_answer) >= 2 and not route_in_question and not route_in_history:
        return (
            "This drug can be given in different ways (e.g., injection vs. oral). "
            "Which route of administration are you asking about?"
        )

    return None


CLARIFICATION_MARKER_PHRASES = (
    "i need", "could you provide", "could you share", "once you provide this", "which route", "which condition"
)

WEIGHT_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|lbs?|pounds?)", re.IGNORECASE)


def resolve_clarification_followup(question: str, chat_history: list[dict], known_drugs: list[str], active_drug: str | None = None) -> str:
    """
    If any recent assistant turn was a clarifying question (weight / condition / route),
    and this question looks like the answer to it, folds the missing piece into a
    self-contained question that preserves the entire previous conversation context.
    """
    last_assistant = None
    for msg in reversed(chat_history or []):
        if msg.get("role") == "assistant":
            content = (msg.get("content", "") or "").lower()
            if any(phrase in content for phrase in CLARIFICATION_MARKER_PHRASES):
                last_assistant = content
                break

    if not last_assistant:
        return question

    q_lower = question.lower().strip()
    weight_match = WEIGHT_VALUE_RE.search(question)
    condition_match = next((c for c in CONDITION_NAME_HINTS if c in q_lower), None)
    route_match = next((r for r in ROUTE_HINTS if r in q_lower), None)
    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", question)

    drug = active_drug or get_active_drug_from_history(chat_history, known_drugs)
    clean_drug = re.sub(r"\(.*?\)", "", drug).strip() if drug else ""

    last_user_q = get_last_substantive_user_query(chat_history)
    clean_base_q = re.sub(r"[?.!]+$", "", last_user_q).strip() if last_user_q else ""

    # Case A: Weight answer (handles bare numbers like "80" as well as "80kg", "50kgs")
    if ("weight in kg" in last_assistant or "patient's weight" in last_assistant) and (weight_match or number_match):
        w_val = weight_match.group(1) if weight_match else number_match.group(1)
        w_unit = weight_match.group(2) if weight_match else "kg"
        if clean_base_q:
            return f"{clean_base_q} weighing {w_val} {w_unit}?"
        if clean_drug:
            return f"What is the {clean_drug} dosage for a patient weighing {w_val} {w_unit}?"
        return f"What is the dosage for a patient weighing {w_val} {w_unit}?"

    # Case B: Age answer
    if ("patient's age" in last_assistant or "age in years" in last_assistant) and number_match:
        age_val = number_match.group(1)
        if clean_base_q:
            return f"{clean_base_q} (age {age_val} years old)?"
        if clean_drug:
            return f"What is the {clean_drug} dosage for a {age_val} year old patient?"
        return f"What is the dosage for a {age_val} year old patient?"

    # Case C: Condition answer
    if condition_match and ("which condition" in last_assistant or "condition being treated" in last_assistant):
        if clean_base_q:
            return f"{clean_base_q} for {condition_match}?"
        if clean_drug:
            return f"What is the {clean_drug} dosage for {condition_match}?"
        return f"What is the dosage for {condition_match}?"

    # Case D: Route answer
    if route_match and ("which route" in last_assistant or "route of administration" in last_assistant):
        if clean_base_q:
            return f"{clean_base_q} via {route_match} route?"
        if clean_drug:
            return f"What is the starting dose and administration schedule for {clean_drug} via the {route_match} route?"
        return f"What is the starting dose and administration schedule via the {route_match} route?"

    if weight_match:
        if clean_base_q:
            return f"{clean_base_q} weighing {weight_match.group(1)} {weight_match.group(2)}?"
        if clean_drug:
            return f"What is the {clean_drug} dosage for a patient weighing {weight_match.group(1)} {weight_match.group(2)}?"

    return question


def build_otc_citation(d: dict, query_or_question: str = "") -> dict:
    """
    Constructs a structured citation for an OTC medication based on the official
    FDA Drug Facts label data and the user's specific query topic.
    """
    brand = d.get("brand_name") or d.get("generic_name") or d.get("drug_name", "OTC Drug")
    generic = d.get("generic_name", "")
    display_name = f"{brand} ({generic})" if (generic and generic.lower() not in brand.lower()) else brand

    q_low = (query_or_question or "").lower()
    sections = []
    if any(w in q_low for w in ["dose", "dosage", "dosing", "how much", "how many", "directions", "take", "give"]):
        sections.append("Directions & Dosage")
    if any(w in q_low for w in ["warning", "warnings", "side effect", "liver", "stomach", "bleeding", "safe", "safety", "allergy", "allergic", "pregnant", "child", "children"]):
        sections.append("Warnings & Precautions")
    if any(w in q_low for w in ["use", "uses", "treat", "indication", "purpose", "headache", "fever", "pain", "heartburn", "allergy", "cold"]):
        sections.append("Uses & Indications")

    if not sections:
        sections = ["Directions & Dosage", "Warnings & Precautions", "Uses & Indications"]

    return {
        "drug_name": display_name,
        "source_type": "otc_label",
        "source_file": f"FDA Drug Facts — {brand}",
        "source_url": d.get("source_url", ""),
        "sections": sections,
        "primary_section": sections[0],
        "drug_facts": {
            "brand_name": d.get("brand_name", brand),
            "generic_name": d.get("generic_name", generic),
            "purpose": d.get("purpose", ""),
            "indications": d.get("indications", ""),
            "dosage": d.get("dosage", ""),
            "warnings": d.get("warnings", ""),
            "do_not_use": d.get("do_not_use", d.get("when_not_to_use", "")),
            "stop_use": d.get("stop_use", ""),
            "keep_out": d.get("keep_out", ""),
        },
        "text": f"Official FDA OTC Drug Facts Label — {brand}",
    }


def is_patient_refusal(text: str) -> bool:
    """
    Checks if the model answer is a refusal, meta question deflection, or cannot find info.
    """
    if not text:
        return True
    t_lower = text.lower().strip()
    refusal_markers = [
        "cannot find reliable information",
        "cannot find any",
        "i'm sorry, but i can't answer",
        "i'm sorry, but i cannot answer",
        "i cannot answer that",
        "i am not able to answer",
        "i don't have information",
        "could not find",
        "i am unable to answer",
        "as an ai",
        "i do not have access to that",
        "i am an ai",
    ]
    return any(m in t_lower for m in refusal_markers)


def get_otc_citations_for_answer(raw_answer: str, query: str = "") -> list[dict]:
    """
    Scans the actual generated answer for mentions of OTC drugs (Ibuprofen,
    Acetaminophen, Cetirizine, Omeprazole, Loratadine and brand aliases) so only
    drugs actually discussed in the response receive citations. If no drug facts
    are discussed or the response is a refusal/meta conversation, returns [].
    """
    if is_patient_refusal(raw_answer):
        return []

    ans_lower = raw_answer.lower()
    cited_drugs = []

    # Check each known OTC drug against the answer text
    for key, data in OTC_DATA.items():
        brand = (data.get("brand_name") or "").lower()
        generic = (data.get("generic_name") or "").lower()

        # Check key (e.g. "ibuprofen"), brand, or generic in answer
        key_pattern = r"\b" + re.escape(key) + r"\b"
        generic_words = [w for w in re.findall(r"[a-zA-Z]{4,}", generic)]
        brand_words = [w for w in re.findall(r"[a-zA-Z]{4,}", brand) if w not in ["pain", "relief", "free", "strength", "reliever", "childrens"]]

        matched = False
        if re.search(key_pattern, ans_lower):
            matched = True
        elif any(re.search(r"\b" + re.escape(bw) + r"\b", ans_lower) for bw in brand_words):
            matched = True
        elif any(re.search(r"\b" + re.escape(gw) + r"\b", ans_lower) for gw in generic_words):
            matched = True
        else:
            # Check aliases (advil, tylenol, zyrtec, prilosec, claritin, etc.)
            aliases = {
                "ibuprofen": ["advil", "motrin", "nsaid"],
                "acetaminophen": ["tylenol", "paracetamol", "crocin", "dolo"],
                "cetirizine": ["zyrtec"],
                "omeprazole": ["prilosec"],
                "loratadine": ["claritin"],
            }
            for alias in aliases.get(key, []):
                if re.search(r"\b" + re.escape(alias) + r"\b", ans_lower):
                    matched = True
                    break

        if matched and data not in cited_drugs:
            cited_drugs.append(data)

    if not cited_drugs:
        return []

    return [build_otc_citation(d, query) for d in cited_drugs]


def handle_rinvoq_procedure_query(question: str) -> dict | None:
    """
    Detects queries asking how to use, prepare, or administer Rinvoq / Rinvoq LQ
    and returns the official step-by-step procedure with authentic FDA label figure diagrams.
    """
    q_low = question.lower()
    if "rinvo" not in q_low and "upadacitinib" not in q_low:
        return None

    is_proc = any(w in q_low for w in [
        "how to use", "how to prepare", "how to give", "preparation", "prepare",
        "instructions for use", "administer", "administration", "procedure", "steps",
        "bottle", "syringe", "figure"
    ])
    if not is_proc:
        return None

    answer = """• **Rinvoq Extended-Release Tablets**: Swallow tablets whole with or without food. Do not split, crush, or chew.
• **Rinvoq LQ (Oral Solution)**: Follow the official step-by-step preparation and administration guide:

**Step 1: Check supplies and gather equipment**
• Check that you have the bottle, oral dosing syringe, and press-in bottle adapter.
![Figure A: Supplies Overview](/api/figures/rinvoq_figure_a.png)

**Step 2: Check expiration date**
• Check the bottle and carton to ensure the expiration date has not passed.
• Do not use if expired.
![Figure B: Check expiration date](/api/figures/rinvoq_figure_b.png)

**Step 3: Check supplies & syringe**
• Check that supplies are clean, dry, undamaged, and plunger is fully inserted.
![Figure C: Check supplies & syringe](/api/figures/rinvoq_figure_c.png)

**Step 4: Open the bottle**
• Press down firmly and twist the child-resistant cap counterclockwise to open.
![Figure D: Open bottle](/api/figures/rinvoq_figure_d.png)

**Step 5: Insert bottle adapter (first use only)**
• Push the press-in bottle adapter firmly into the neck of the bottle until flush.
![Figure E: Push bottle adapter](/api/figures/rinvoq_figure_e.png)

**Step 6: Push plunger into syringe**
• Ensure the plunger is pushed all the way down into the oral dosing syringe.
![Figure F: Push plunger into syringe](/api/figures/rinvoq_figure_f.png)

**Step 7: Insert syringe into bottle**
• Insert the oral syringe tip firmly into the center opening of the bottle adapter.
![Figure G: Insert syringe into bottle](/api/figures/rinvoq_figure_g.png)

**Step 8: Turn bottle upside down**
• Turn the bottle upside down with the syringe still firmly attached.
![Figure H: Turn bottle upside down](/api/figures/rinvoq_figure_h.png)

**Step 9: Draw the prescribed dose**
• Slowly pull the plunger down until the top edge matches the prescribed dose mark.
![Figure I: Draw prescribed dose](/api/figures/rinvoq_figure_i.png)

**Step 10: Turn bottle upright & remove syringe**
• Turn the bottle upright and gently pull the oral syringe out of the adapter.
![Figure J: Turn upright & remove syringe](/api/figures/rinvoq_figure_j.png)

**Step 11: Check the dose**
• Hold syringe at eye level to confirm correct dose volume before administering.
![Figure K: Verify dose volume](/api/figures/rinvoq_figure_k.png)

**Step 12: Administer dose into child's mouth**
• Place syringe tip inside cheek and gently push plunger to deliver medication.
![Figure L: Administer dose](/api/figures/rinvoq_figure_l.png)

**Step 13: Cap the bottle**
• Screw the child-resistant cap tightly back onto the bottle over the adapter.
![Figure M: Cap bottle](/api/figures/rinvoq_figure_m.png)

**Step 14: Rinse oral syringe**
• Separate plunger and barrel, rinse both thoroughly with clean water, and air dry.
![Figure N: Rinse syringe](/api/figures/rinvoq_figure_n.png)

SOURCES_USED: 1"""

    return {
        "answer": answer,
        "citation": "Source: Rinvoq (upadacitinib) — Page 84-90",
        "citations": [{
            "drug_name": "Rinvoq (upadacitinib)",
            "source_file": "rinvoq.pdf",
            "pages": [84, 85, 86, 87, 88, 89, 90],
            "page_snippets": {
                "86": ["Check expiration date (Figure B)", "Check supplies (Figure C)", "Open the bottle (Figure D)"],
                "87": ["Push bottle adapter (Figure E)", "Insert syringe (Figure G)"],
                "88": ["Turn bottle upside down (Figure H)", "Pull plunger to dose (Figure I)"],
                "89": ["Check dose in syringe (Figure K)", "Give dose (Figure L)"],
                "90": ["Put cap back on (Figure M)", "Rinse syringe (Figure N)"]
            },
            "text": "Instructions for Use: Preparation and Administration of Rinvoq LQ"
        }],
        "source_file": "rinvoq.pdf",
        "page_number": 86,
        "refused": False,
        "query_type": "dosage",
        "drug_name": "Rinvoq (upadacitinib)"
    }


def ask(
    question: str,
    collection,
    chat_history: list[dict],
    groq_api_key: str,
    role: str = "patient",
    session: dict | None = None
) -> dict:
    """
    Answers a drug question using RAG + Groq (LLaMA 3).
    Supports role="doctor" (ChromaDB + Prescribing PDFs) and role="patient" (OpenFDA OTC labels).

    Returns:
    {
        "answer": str,
        "citations": [{"drug_name", "source_file", "pages", "page_snippets", "text"}],
        "citation": str (primary citation for backward compat),
        "source_file": str,
        "page_number": int,
        "refused": bool,
        "query_type": str,
        "drug_name": str | None
    }
    """
    # ── GUARD 1: Flagged Session Check ───────────────────────────────────────
    if session and is_session_flagged(session):
        resp = _empty_response("guardrail_block", refused=True)
        resp["answer"] = FLAGGED_SESSION_WARNING
        return resp

    # ── GUARD 2: Current Question Injection Detection ────────────────────────
    is_injected, matched = detect_injection(question)
    if is_injected:
        if session:
            update_anomaly_count(session, question)
        resp = _empty_response("guardrail_block", refused=True)
        resp["answer"] = INJECTION_REFUSAL
        return resp

    # ── GUARD 3: Dosage Override Detection ───────────────────────────────────
    if detect_dosage_override(question):
        if session:
            update_anomaly_count(session, question)
        resp = _empty_response("guardrail_block", refused=True)
        resp["answer"] = DOSAGE_OVERRIDE_REFUSAL
        return resp

    # Step 0a: Handle greetings / conversational openers directly — no LLM call
    if is_greeting(question):
        resp = _empty_response("greeting")
        if role == "doctor":
            resp["answer"] = (
                "Hello! I'm DrugIQ, your clinical drug information assistant. I can help you "
                "with questions about dosage, indications, side effects, interactions, contraindications, "
                "and more — based on official FDA prescribing documents.\n\nWhat would you like to explore?"
            )
        else:
            resp["answer"] = (
                "Hello! I'm DrugIQ, your safe home medication assistant. I can help answer questions "
                "about over-the-counter (OTC) medications, directions, and safe use based on official FDA drug labels.\n\n"
                "What medication or symptom would you like to ask about?"
            )
        return resp

    # Step 0b: Handle system/database info questions (e.g. "how many databases are we using?")
    if detect_system_metadata_query(question):
        if role == "patient":
            resp = _empty_response("system_info", drug_name="5 FDA OTC Medicines")
            lines = [
                "1. **Ibuprofen** (*Advil / Motrin / Dye Free*) — Pain Reliever & Fever Reducer (NSAID)",
                "2. **Acetaminophen** (*Tylenol / Extra Strength*) — Pain Reliever & Fever Reducer",
                "3. **Cetirizine** (*Children's Zyrtec / Zyrtec*) — 24-Hour Antihistamine & Allergy Relief",
                "4. **Omeprazole** (*Prilosec OTC*) — Acid Reducer & Frequent Heartburn Relief (PPI)",
                "5. **Loratadine** (*Claritin / Allergy Relief*) — 24-Hour Non-Drowsy Antihistamine",
            ]
            resp["answer"] = (
                "We currently have **5 official FDA Over-The-Counter (OTC) drug datasets** indexed in the system:\n\n" +
                "\n".join(lines) +
                "\n\nAll answers are grounded directly in official OpenFDA and DailyMed OTC Drug Facts labels. " +
                "You can ask about dosage, directions, uses, warnings, or symptom relief for any of these medications."
            )
            resp["citations"] = [build_otc_citation(d, question) for d in OTC_DATA.values()]
            return resp

        drug_info = get_drug_info(collection) if collection else []
        count = len(drug_info)
        lines = [f"• **{d['drug_name']}** (`{d['source_file']}` — {d.get('page_count', 0)} pages)" for d in drug_info]
        resp = _empty_response("system_info")
        resp["answer"] = (
            f"We currently have **{count} drug databases/documents** indexed in the system:\n\n" +
            "\n".join(lines) +
            "\n\nYou can ask any questions about dosage, indications, administration, side effects, or interactions for these drugs."
        )
        return resp

    known_drugs = get_all_drug_names(collection) if collection else []
    all_context_drugs = list(known_drugs)
    for n in get_all_otc_names():
        if n not in all_context_drugs:
            all_context_drugs.append(n)

    # Step -1: Resolve vague drug references and contextual follow-ups against conversation history
    resolved_question, active_drug = rewrite_query(question, chat_history, all_context_drugs)

    # Step -1b: If the previous turn asked for missing context, fold it in
    resolved_question = resolve_clarification_followup(resolved_question, chat_history, all_context_drugs, active_drug)

    client = Groq(api_key=groq_api_key)

    # ───────────────────────────────────────────────────────────────────────────
    # PATIENT ROLE: OTC OpenFDA Pipeline & Prescription Blocking
    # ───────────────────────────────────────────────────────────────────────────
    if role == "patient":
        # 1. Check if the question mentions any known prescription drug from our collection
        matched_drug, _, _ = find_matching_drug(resolved_question, known_drugs)
        if not matched_drug and active_drug and active_drug in known_drugs:
            matched_drug = active_drug

        # 2. Extract potential candidate keywords from the query
        stopwords = {
            "what", "when", "where", "which", "who", "why", "how", "dose", "dosage", "dosing",
            "uses", "usage", "warnings", "warning", "side", "effects", "effect", "safe", "safety",
            "take", "give", "much", "many", "tell", "about", "information", "info", "for", "with",
            "and", "the", "are", "can", "should", "could", "would", "is", "it", "this", "that",
            "schedule", "administration", "prescribe", "medicine", "medication", "drug", "drugs"
        }
        candidates = [
            w for w in re.findall(r"[a-zA-Z0-9]+", resolved_question)
            if len(w) >= 3 and w.lower() not in stopwords and w.lower() not in NON_DRUG_COMMON_WORDS
        ]

        # 3. Find matching OTC drugs by drug name, brand alias, OR symptom/condition
        otc_matches = find_otc_drugs_for_query(resolved_question)

        if not otc_matches:
            for c in candidates:
                d = lookup_otc_drug(c)
                if d and d not in otc_matches:
                    otc_matches.append(d)

        if not otc_matches and matched_drug and is_otc_drug(matched_drug):
            d = lookup_otc_drug(matched_drug)
            if d and d not in otc_matches:
                otc_matches.append(d)

        if not otc_matches and active_drug and is_otc_drug(active_drug):
            d = lookup_otc_drug(active_drug)
            if d and d not in otc_matches:
                otc_matches.append(d)

        # Context Memory Fallback: if user asked a follow-up ("dosage", "how much", "side effects")
        # and current query didn't name a new drug, retrieve the active OTC medications from history
        if not otc_matches:
            history_otc = find_active_otc_drugs_from_history(chat_history)
            if history_otc:
                otc_matches.extend(history_otc)

        # Broad summary / compare / general OTC query detection
        summary_keywords = ["summarize", "summary", "overview", "compare", "comparison", "all medicines", "all drugs", "what drugs", "what medicines", "list drugs", "list medicines", "available medicines"]
        if not otc_matches and any(k in resolved_question.lower() for k in summary_keywords):
            otc_matches = list(OTC_DATA.values())

        if otc_matches:
            # Verified OTC drug or condition! Return patient guidance using OpenFDA local data for all matched OTC drugs
            drug_names = [d.get("brand_name") or d.get("generic_name", "OTC Drug") for d in otc_matches]
            combined_name = ", ".join(drug_names)
            combined_context = "\n\n---\n\n".join(build_otc_context(d) for d in otc_matches)

            is_interaction_q = len(otc_matches) >= 2 or any(k in resolved_question.lower() for k in ["interaction", "combine", "together", "contraindicated", "safe to use together"])

            patient_system_prompt = f"""You are DrugIQ, a safe, friendly, and responsible home medication assistant for patients and families.
Answer ONLY using the provided OTC drug label information below.

AUDIENCE & STRUCTURE (FOR PATIENTS):
- Structure your response as a friendly, concise SUMMARY first, followed by a few (2-4 max) essential bullet points ('•').
- Keep the overall response neat, concise, easy-to-understand, and in simple, plain English without overwhelming technical medical jargon.

DRUG COMBINATION & INTERACTION STATUS:
- If the user is asking whether two medications can be taken together or if they interact, your response MUST begin on line 1 with one of these status lines:
  🚫 CONTRAINDICATED (if taking them together is unsafe or duplicates NSAIDs/antihistamines excessively)
  ⚠️ USE WITH CAUTION (if combining requires precautions, staggered timing, or monitoring)
  ✅ NO DIRECT INTERACTION (if both can be taken according to their standard package directions)

TOP WARNING RULE (MANDATORY):
- If any warning, precaution, allergy alert, liver warning, stomach bleeding warning, heart warning, or 'do not use / stop use' precaution exists in the provided context for the drug(s), you MUST place it at the VERY TOP of your response (immediately after the status badge if present), formatted as:
⚠️ **WARNING / PRECAUTION:** <Concise warning/precaution text here>
- Do NOT put warnings or precautions at the end or bottom of the message.

PATIENT SAFETY GUIDANCE:
- Always include appropriate patient advice: 'Please consult your physician or pharmacist before changing or starting any medication.'
- For questions about combining medications or persistent/severe symptoms, add: 'Please consult your doctor or pharmacist before combining medications.'
- Never speculate beyond what the official OTC label says.

Document context:
{combined_context}"""
            messages = _build_sanitized_messages(patient_system_prompt, chat_history, resolved_question)

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                reasoning_effort="low"
            )
            raw_answer = response.choices[0].message.content.strip()

            is_unsafe, unsafe_match = validate_output(raw_answer)
            if is_unsafe:
                if session:
                    update_anomaly_count(session, question)
                resp = _empty_response("guardrail_block", refused=True)
                resp["answer"] = OUTPUT_SAFETY_REFUSAL
                return resp

            refused = is_patient_refusal(raw_answer)
            citations = [] if refused else get_otc_citations_for_answer(raw_answer, resolved_question)

            resp_type = "drug_interaction" if is_interaction_q else "otc_info"
            resp = _empty_response(resp_type, drug_name=combined_name, refused=refused)
            resp["answer"] = raw_answer
            resp["citation"] = f"FDA OTC Label — {combined_name}" if citations else None
            resp["citations"] = citations
            return resp

        # 4. If not OTC, check if it's a known prescription drug in our collection or candidate
        rx_target = matched_drug
        if not rx_target:
            for c in candidates:
                if any(c.lower() in d.lower() for d in known_drugs):
                    rx_target = c
                    break

        if rx_target:
            # Block prescription drug for patient
            resp = _empty_response("prescription_blocked", drug_name=rx_target, refused=True)
            resp["answer"] = (
                "This medication requires a prescription and must be managed by a licensed "
                "healthcare provider. I'm not able to provide dosing or home-use guidance "
                "for this drug. Please consult your doctor or pharmacist."
            )
            return resp

        # 5. Direct query OTC lookup fallback
        direct_otc = lookup_otc_drug(resolved_question)
        if direct_otc:
            context = (
                f"Drug Name: {direct_otc['drug_name']}\n"
                f"Purpose: {direct_otc['purpose']}\n"
                f"Indications & Uses: {direct_otc['indications']}\n"
                f"Dosage & Directions: {direct_otc['dosage']}\n"
                f"Warnings & Precautions: {direct_otc['warnings']}\n"
                f"When Not to Use: {direct_otc.get('when_not_to_use', '')}"
            )
            patient_system_prompt = f"""You are DrugIQ, a safe, friendly, and responsible home medication assistant for patients and families.
Answer ONLY using the provided drug label information below.

AUDIENCE & STRUCTURE (FOR PATIENTS):
- Structure your response as a friendly, concise SUMMARY first, followed by a few (2-4 max) essential bullet points ('•').
- Use simple, plain English without medical jargon.
- TOP WARNING RULE (MANDATORY): If any warning, precaution, allergy alert, or 'do not use / stop use' note applies, place it at the VERY TOP of your response before anything else, formatted as:
⚠️ **WARNING / PRECAUTION:** <Concise warning/precaution text here>
- Do NOT place the warning or precaution at the bottom or end of the message.

PATIENT SAFETY GUIDANCE:
- Always add: 'Please consult your physician or pharmacist before changing or starting any medication.'
- For questions about interactions or serious symptoms, add: 'Please consult your doctor or pharmacist before combining medications.'
- Never speculate beyond what the label says.

Document context:
{context}"""
            messages = _build_sanitized_messages(patient_system_prompt, chat_history, resolved_question)

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                reasoning_effort="low"
            )
            raw_answer = response.choices[0].message.content.strip()

            is_unsafe, unsafe_match = validate_output(raw_answer)
            if is_unsafe:
                if session:
                    update_anomaly_count(session, question)
                resp = _empty_response("guardrail_block", refused=True)
                resp["answer"] = OUTPUT_SAFETY_REFUSAL
                return resp

            refused = is_patient_refusal(raw_answer)
            citations = [] if refused else get_otc_citations_for_answer(raw_answer, resolved_question)

            resp = _empty_response("otc_info", drug_name=direct_otc['drug_name'], refused=refused)
            resp["answer"] = raw_answer
            resp["citation"] = f"FDA OTC Label — {direct_otc['drug_name']}" if citations else None
            resp["citations"] = citations
            return resp

        resp = _empty_response("patient_general", refused=True)
        resp["answer"] = (
            "I couldn't find over-the-counter (OTC) drug facts for this question in the official FDA database. "
            "If this concerns a prescription medication, specific illness, or severe symptoms, please consult your doctor or pharmacist."
        )
        return resp

    # ───────────────────────────────────────────────────────────────────────────
    # DOCTOR ROLE: ChromaDB & Clinical RAG Pipeline
    # ───────────────────────────────────────────────────────────────────────────

    # Step 1: Detect query type (condition-to-drug takes priority)
    query_type = detect_query_type(resolved_question)

    # Step 1b: Drug-drug interaction detection
    interaction_drugs: list[str] = []
    if query_type != "condition_to_drug":
        interaction_check = detect_drug_drug_query(resolved_question, known_drugs)
        if interaction_check["is_interaction"]:
            query_type = "drug_interaction"
            interaction_drugs = interaction_check["drugs"][:2]

    matched_drug = None
    if query_type not in ("condition_to_drug", "drug_interaction"):
        matched_drug, suggestions, tried = find_matching_drug(resolved_question, known_drugs)

        # If no drug was matched directly in query, fallback to active drug from conversation history
        if matched_drug is None and active_drug:
            matched_drug = active_drug

        # Step 2: Unknown drug detection (only if no active drug is in context)
        if matched_drug is None:
            if not suggestions:
                subject = extract_overview_subject(question, has_active_drug=bool(active_drug))
                if subject:
                    tried = subject
            if tried:
                resp = _empty_response("unknown_drug", refused=True)
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                resp["answer"] = (
                    f"I couldn't find '{tried}' in our database.{suggestion_text}\n\n"
                    "Please check the spelling or upload the PDF for this drug."
                )
                return resp

    # Step 3b: Procedural preparation/administration query check (e.g. Rinvoq Instructions for Use)
    proc_resp = handle_rinvoq_procedure_query(resolved_question)
    if proc_resp:
        return proc_resp

    # Step 4: Retrieve relevant chunks
    search_query = resolved_question
    if query_type == "condition_to_drug":
        chunks = search_multi_drug(collection, search_query, top_k=8)
    elif query_type == "drug_interaction":
        interaction_query = (
            "drug interactions contraindications not recommended for use in "
            "combination with other drugs biologic DMARDs JAK inhibitors live vaccines"
        )
        chunks = []
        for d in interaction_drugs:
            chunks.extend(search(collection, interaction_query, top_k=4, where={"drug_name": d}))
        kg_fact = _kg_interaction_fact_chunk(interaction_drugs)
        if kg_fact:
            chunks.insert(0, kg_fact)
    elif matched_drug:
        chunks = search(collection, search_query, top_k=5, where={"drug_name": matched_drug})
    else:
        chunks = search(collection, search_query, top_k=5)

    if not chunks:
        resp = _empty_response(query_type, drug_name=matched_drug, refused=True)
        resp["answer"] = "I could not find any relevant information in our drug documents. Please rephrase your question."
        return resp

    # Step 5: Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['drug_name']}, Page {chunk['page_number']}, File: {chunk['source_file']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Step 6: System prompt (adapted based on query type) — doctor clinical format
    common_rules = """AUDIENCE & CLINICAL STRUCTURE (FOR PHYSICIANS & HEALTHCARE PROFESSIONALS):
- You are providing clinical pharmacotherapy intelligence to licensed physicians, clinicians, and medical specialists.
- Structure your response into two distinct, high-yield clinical sections:
  1. **Clinical Summary**: A synthesized 2-3 sentence clinical executive summary directly answering the physician's query with therapeutic rationale and clinical significance.
  2. **Technical Details**: Structured, precise technical bullet points starting with "•" covering exact dosing regimens, titration schedules, pharmacokinetics (CYP, clearance, half-life), monitoring parameters, or black-box contraindications found in the prescribing context.

CLINICAL COMMUNICATION RULES:
- Use precise medical and pharmacological terminology appropriate for a clinician.
- DO NOT tell the doctor to "consult a physician", "ask your doctor", or "consult a pharmacist" (they are the prescribing physician). If information requires clinical verification, state: "Refer to full FDA prescribing documentation and institutional protocols."
- Never mention "Source N", a page number, or a citation anywhere inside the text or bullets — keep text to pure clinical content. The only place source numbers appear is the single SOURCES_USED line at the very end.
- Do NOT write your own "Source" or "Page" citation line — the app adds an accurate one automatically from the source numbers you report below.

SOURCE TRACKING — follow this strictly:
- After your response, on its own line, write: SOURCES_USED: <comma-separated source numbers>
  Use the numbers from the "[Source N: ...]" labels below for every source you actually drew from, e.g. SOURCES_USED: 2,3
- If you could not answer from the documents, write: SOURCES_USED: none"""

    if query_type == "condition_to_drug":
        system_prompt = f"""You are a clinical pharmacotherapy specialist assisting licensed healthcare professionals and physicians.

The physician is asking about therapeutic options for a condition/disease. Search the provided document context for any drugs that mention treating, managing, or being indicated for this condition.
Provide:
1. **Clinical Summary**: High-yield synthesis of indicated therapies and clinical rationale.
2. **Technical Details**: Bulleted clinical specifics for each drug (approved indications, dosing & administration protocols, key warnings, contraindications).

If NO drugs in the context mention the asked condition, say exactly:
"I cannot find any drugs in the loaded prescribing documents that specifically mention treating this condition. Refer to full FDA prescribing indexes or clinical practice guidelines for unindexed therapies."
and write SOURCES_USED: none — nothing else.

{common_rules}

Document context:
""" + context
    elif query_type == "drug_interaction":
        name_a = interaction_drugs[0].split("(")[0].strip() if interaction_drugs else "the first drug"
        name_b = interaction_drugs[1].split("(")[0].strip() if len(interaction_drugs) > 1 else "the second drug"
        system_prompt = f"""You are a clinical pharmacotherapy specialist assisting licensed healthcare professionals and physicians.

The physician is evaluating the concomitant use of {name_a} and {name_b}. Search the provided prescribing documentation for Section 7 (Drug Interactions), Section 4 (Contraindications), Section 5 (Warnings and Precautions), pharmacokinetic profiles, or shared toxicities.

Your response MUST start with a status line, EXACTLY as shown below, on line 1:
🚫 CONTRAINDICATED
⚠️ USE WITH CAUTION
✅ NO INTERACTION FOUND

Choose which: "🚫 CONTRAINDICATED" only if the documents explicitly say these drugs (or drug classes) should not be combined. "⚠️ USE WITH CAUTION" if the documents mention a risk, monitoring requirement, or interaction without an outright prohibition. "✅ NO INTERACTION FOUND" if the documents give no indication of any relationship between the two drugs.

After the status line, provide:
1. **Clinical Summary**: High-yield overview of the interaction mechanism and clinical relevance.
2. **Technical Details**: Bulleted clinical guidance covering mechanism, potential adverse outcomes, monitoring parameters, and dosage modification recommendations.
Do NOT tell the physician to "consult a physician".

{common_rules}

Document context:
""" + context
    else:
        system_prompt = f"""You are a clinical pharmacotherapy specialist assisting licensed healthcare professionals and physicians.

Answer ONLY using the provided document context below. Never use outside knowledge. Provide a clinician-tailored response:
1. **Clinical Summary**: High-yield 2-3 sentence overview answering the clinical query directly.
2. **Technical Details**: Precise, high-density clinical bullet points covering specific parameters, dosage/titration numbers, monitoring criteria, or warnings.

If the context does not contain enough information to answer the specific question asked, say exactly:
"I cannot find reliable information on this in the available prescribing documents. Refer to the full prescribing information and institutional clinical protocols."
and write SOURCES_USED: none — nothing else.

{common_rules}

Document context:
""" + context

    # Step 7: Build messages with sanitized conversation history & safety fence
    messages = _build_sanitized_messages(system_prompt, chat_history, question)

    # Step 8: Call Groq. openai/gpt-oss-20b is a reasoning model whose hidden
    # "thinking" tokens count against max_tokens — left uncapped it can burn
    # the entire budget reasoning and return empty content (finish_reason
    # "length"). This task doesn't need deep reasoning, so keep it low.
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
        reasoning_effort="low"
    )

    raw_answer = response.choices[0].message.content.strip()

    # Guard 5: Output Validation
    is_unsafe, unsafe_match = validate_output(raw_answer)
    if is_unsafe:
        if session:
            update_anomaly_count(session, question)
        resp = _empty_response("guardrail_block", refused=True)
        resp["answer"] = OUTPUT_SAFETY_REFUSAL
        return resp

    refused = "cannot find reliable information" in raw_answer.lower() or \
              "cannot find any drugs" in raw_answer.lower()

    # Step 8b: If the answer is only correct conditional on information the
    # user never gave (weight, condition, route), ask for it instead of
    # handing back a generic answer.
    #
    # Normally this scans the model's own answer. But when the documents
    # dose by weight and the user asked by age (e.g. "my son is 6 years
    # old, how much Humira?"), the model frequently can't reconcile the two
    # and just refuses outright ("I cannot find reliable information...")
    # instead of recognizing it should ask for weight — the refusal
    # sentence itself carries no "mg/kg" signal even though the retrieved
    # document chunks do. So on a refusal specifically, fall back to
    # scanning the raw retrieved context (what the documents actually say)
    # instead of the canned refusal text — this is what lets the app know,
    # straight from the source document, whether a drug is dosed by weight
    # or by age, independent of whether the model managed to say so.
    scan_text = context if refused else raw_answer
    clarifying_question = extract_missing_context(question, scan_text, chat_history)
    if clarifying_question:
        return {
            "answer": (
                f"{clarifying_question}\n\n"
                "💡 Once you provide this, I can give you the precise dosage from the prescribing information."
            ),
            "citations": [],
            "citation": None,
            "source_file": None,
            "page_number": None,
            "refused": False,
            "query_type": "clarification_needed",
            "drug_name": matched_drug,
        }

    # Only cite the chunk(s) the model actually says it drew from (by source
    # index, not by transcribing a page number — see extract_sources_used),
    # not every chunk that was merely retrieved and offered as context.
    # Falls back to the single best-matching chunk if the model didn't
    # report usable indices.
    used_indices = extract_sources_used(raw_answer, len(chunks))
    cited_chunks = [chunks[i] for i in used_indices]
    if not cited_chunks and not refused:
        cited_chunks = [chunks[0]]

    answer = strip_sources_directive(raw_answer)
    citations = build_citations(cited_chunks) if cited_chunks and not refused else []
    if citations:
        citation_line = build_citation_line(citations)
        if citation_line:
            answer = f"{answer}\n\n{citation_line}" if answer else citation_line

    top_chunk = cited_chunks[0] if cited_chunks else chunks[0]
    primary_citation = f"{top_chunk['drug_name']} Prescribing Information — Page {top_chunk['page_number']}"

    return {
        "answer": answer,
        "citations": citations,
        "citation": primary_citation if not refused else None,
        "source_file": top_chunk["source_file"] if not refused else None,
        "page_number": top_chunk["page_number"] if not refused else None,
        "refused": refused,
        "query_type": query_type,
        "drug_name": matched_drug or top_chunk["drug_name"],
    }
