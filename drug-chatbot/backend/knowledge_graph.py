"""
Knowledge Graph Extraction
--------------------------
Builds a graph of relationships (drug -> condition, drug -> class,
drug -> warning, drug <-> drug) purely from what's explicitly stated in the
indexed prescribing documents. Nothing here is hardcoded per-drug — every
relationship comes from an LLM extraction pass over retrieved chunks, run
fresh against whatever drugs are currently indexed.

Pipeline per drug:
  1. Five targeted ChromaDB searches, scoped to that drug's chunks, one per
     relationship category (indications, interactions, class, boxed
     warnings, contraindications).
  2. All five categories' chunks go to a single Groq call that returns
     structured JSON.
  3. Conditions/classes/warnings are normalized so near-duplicate phrasings
     ("RA" vs "Rheumatoid Arthritis") collapse onto one graph node.
  4. Interaction/contraindication partners are fuzzy-matched against the
     other indexed drugs — mentions of drugs/classes we haven't indexed are
     dropped, since there's nothing to draw an edge to.
  5. Drug<->drug relationships are deduplicated across both drugs' own
     extractions into a single bidirectional edge.

The assembled graph is cached to data/knowledge_graph.json so reopening the
panel doesn't redo the LLM work every time.
"""

import os
import re
import json
import difflib
from datetime import datetime, timezone

from groq import Groq
from backend.vector_store import search, get_drug_info
from backend.rag_engine import find_matching_drug, make_snippet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "knowledge_graph.json")

# Query text is deliberately stuffed with common domain keywords (condition
# names, class names, ...) as retrieval bait — this doesn't hardcode which
# ones apply to a given drug (the LLM still only extracts what's actually in
# the retrieved text), it just gives the embedding search richer anchor
# vocabulary, which measurably improves recall over a single generic phrase.
CATEGORY_QUERIES = {
    "conditions_treated": (
        "indications and usage approved for treatment of adults with rheumatoid "
        "arthritis psoriatic arthritis ulcerative colitis Crohn's disease "
        "ankylosing spondylitis atopic dermatitis psoriasis diabetes cancer"
    ),
    "drug_interactions": "drug interactions contraindicated concomitant use methotrexate biologic other drugs live vaccines",
    "drug_class": (
        "pharmacological class mechanism of action tumor necrosis factor TNF "
        "blocker Janus kinase JAK inhibitor monoclonal antibody PD-1 checkpoint "
        "inhibitor GLP-1 receptor agonist antimetabolite"
    ),
    "black_box_warnings": "boxed warning black box serious infections mortality malignancy cardiovascular thrombosis",
    "contraindicated_with": "not recommended in combination with other biologic DMARDs JAK inhibitors live vaccines",
}

EXTRACTION_SYSTEM_PROMPT = """You are extracting structured relationships from drug prescribing documents.
From the following text about {drug_name}, extract ONLY facts explicitly stated in the document. Never use outside knowledge.

Return a JSON object with exactly these fields:
{{
  "conditions_treated": ["list of diseases/conditions this drug is approved for"],
  "drug_interactions": ["list of drug names this drug interacts with"],
  "drug_class": "the pharmacological class of this drug (e.g. JAK inhibitor), or null",
  "black_box_warnings": ["list of serious warnings, if any black box warnings exist"],
  "contraindicated_with": ["list of drug names or drug types explicitly contraindicated"]
}}

Return ONLY the JSON object, nothing else — no markdown fences, no explanation. If information is not in the text, use an empty list or null."""


# ---------------------------------------------------------------------------
# Normalization — collapses near-duplicate phrasings onto one graph node.
# These are dedup aids, not a hardcoded relationship list: which drugs treat
# which conditions/classes/warnings is still entirely LLM-extracted; this
# only decides when two extracted strings mean the same node.
# ---------------------------------------------------------------------------

CONDITION_ALIASES = {
    "ra": "Rheumatoid Arthritis",
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "psa": "Psoriatic Arthritis",
    "psoriatic arthritis": "Psoriatic Arthritis",
    "uc": "Ulcerative Colitis",
    "ulcerative colitis": "Ulcerative Colitis",
    "cd": "Crohn's Disease",
    "crohn's disease": "Crohn's Disease",
    "crohns disease": "Crohn's Disease",
    "crohn disease": "Crohn's Disease",
    "t2d": "Type 2 Diabetes",
    "t2dm": "Type 2 Diabetes",
    "type 2 diabetes": "Type 2 Diabetes",
    "type 2 diabetes mellitus": "Type 2 Diabetes",
    "nsclc": "Non-Small Cell Lung Cancer",
    "non-small cell lung cancer": "Non-Small Cell Lung Cancer",
    "non small cell lung cancer": "Non-Small Cell Lung Cancer",
    "axial spondyloarthritis": "Axial Spondyloarthritis",
    "ankylosing spondylitis": "Ankylosing Spondylitis",
    "atopic dermatitis": "Atopic Dermatitis",
    "pjia": "Polyarticular Juvenile Idiopathic Arthritis",
    "polyarticular juvenile idiopathic arthritis": "Polyarticular Juvenile Idiopathic Arthritis",
}

# Checked as substrings, most specific first, so a drug described as both a
# "biologic DMARD" and a "TNF inhibitor" lands on the more specific class.
CLASS_KEYWORD_MAP = [
    (("jak inhibitor", "janus kinase"), "JAK Inhibitor"),
    (("pd-1", "pd1", "programmed death", "checkpoint inhibitor"), "PD-1 Inhibitor"),
    (("glp-1", "glp1", "glucagon-like peptide", "glucagon like peptide"), "GLP-1 Receptor Agonist"),
    (("tnf inhibitor", "tumor necrosis factor", "tnf blocker", "tnf-alpha"), "TNF Inhibitor"),
    (("antimetabolite", "folate analog", "folic acid antagonist"), "Antimetabolite"),
    (("biologic dmard", "dmard"), "Biologic DMARD"),
]

WARNING_KEYWORD_MAP = [
    (("serious infection",), "Serious Infections"),
    (("mortality", "death"), "Mortality"),
    (("malignan", "cancer", "lymphoma"), "Malignancy"),
    (("cardiovascular",), "Major Adverse Cardiovascular Events"),
    (("thrombosis", "blood clot", "embolism"), "Thrombosis"),
    (("immune-mediated", "immune mediated"), "Immune-Mediated Reactions"),
    (("embryo", "fetal", "pregnan"), "Embryo-Fetal Toxicity"),
    (("hepatotox", "liver"), "Hepatotoxicity"),
    (("bone marrow", "myelosuppression"), "Bone Marrow Suppression"),
]


def _normalize_condition(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    cleaned = re.sub(r"\s+", " ", raw.strip().strip(".,;:"))
    lowered = cleaned.lower()
    if lowered in CONDITION_ALIASES:
        return CONDITION_ALIASES[lowered]
    # Extracted phrasing often carries qualifiers around the actual disease
    # name ("adults with moderately to severely active rheumatoid
    # arthritis"), so match on the core name as a substring rather than
    # requiring an exact match — otherwise two drugs treating the same
    # condition can land on different nodes just because one extraction
    # included the severity qualifier and the other didn't. Short
    # abbreviations (RA, UC, CD, ...) are matched as whole words only, to
    # avoid them firing on an unrelated word that happens to contain them.
    for alias_key, canonical in sorted(CONDITION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if len(alias_key) <= 4:
            if re.search(rf"\b{re.escape(alias_key)}\b", lowered):
                return canonical
        elif alias_key in lowered:
            return canonical
    if len(cleaned) < 3 or len(cleaned) > 60:
        return None
    return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()


def _normalize_class(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    lowered = raw.lower()
    for keywords, canonical in CLASS_KEYWORD_MAP:
        if any(kw in lowered for kw in keywords):
            return canonical
    cleaned = re.sub(r"\s+", " ", raw.strip().strip(".,;:"))
    if len(cleaned) < 3 or len(cleaned) > 50:
        return None
    return cleaned[0].upper() + cleaned[1:]


def _normalize_warning(raw: str, drug_name: str) -> str | None:
    if not raw or not raw.strip():
        return None
    lowered = raw.lower()
    for keywords, canonical in WARNING_KEYWORD_MAP:
        if any(kw in lowered for kw in keywords):
            return canonical
    # No recognized theme — keep it drug-specific rather than risk merging
    # two genuinely different warnings onto the same node.
    cleaned = re.sub(r"\s+", " ", raw.strip().strip(".,;:"))
    if not cleaned:
        return None
    short = cleaned[:40].rsplit(" ", 1)[0] if len(cleaned) > 40 else cleaned
    return f"{short} ({drug_name.split('(')[0].strip()})"


def _extract_json(raw: str) -> dict | None:
    """Pulls a JSON object out of an LLM response, tolerating markdown
    fences or stray text around it."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _match_indexed_drug(mention: str, known_names: list[str], exclude: str) -> str | None:
    """Fuzzy-matches a free-text drug mention (e.g. "Humira") against the
    currently indexed drugs, excluding the drug whose own extraction
    produced the mention."""
    candidates = [n for n in known_names if n != exclude]
    if not candidates:
        return None
    matched, _, _ = find_matching_drug(mention, candidates)
    return matched


# Real FDA labels almost always phrase contraindications/interactions by
# drug *class* ("other JAK inhibitors", "biologic DMARDs"), not by naming a
# specific competitor brand. Without resolving class mentions to the actual
# indexed drugs of that class, a label saying "not recommended with biologic
# DMARDs" would never connect to Humira even though Humira *is* one.
CLASS_MENTION_GROUPS = [
    (("jak inhibitor", "janus kinase"), {"JAK Inhibitor"}),
    (("biologic dmard", "biologic ", "dmard"), {"TNF Inhibitor", "Biologic DMARD"}),
    (("tnf inhibitor", "tumor necrosis factor", "tnf blocker", "tnf-alpha"), {"TNF Inhibitor"}),
    (("pd-1", "pd1", "checkpoint inhibitor", "programmed death"), {"PD-1 Inhibitor"}),
    (("glp-1", "glp1", "glucagon-like peptide"), {"GLP-1 Receptor Agonist"}),
]


def _classes_mentioned(text: str) -> set[str]:
    lowered = text.lower()
    result = set()
    for keywords, classes in CLASS_MENTION_GROUPS:
        if any(kw in lowered for kw in keywords):
            result |= classes
    return result


def _resolve_drug_mentions(mention: str, known_names: list[str], exclude: str, class_by_drug: dict) -> list[str]:
    """Returns the indexed drug(s) a mention refers to: a direct fuzzy name
    match if there is one, otherwise every other indexed drug whose own
    class falls under a class the mention describes."""
    direct = _match_indexed_drug(mention, known_names, exclude)
    if direct:
        return [direct]
    mentioned_classes = _classes_mentioned(mention)
    if not mentioned_classes:
        return []
    return [n for n in known_names if n != exclude and class_by_drug.get(n) in mentioned_classes]


def _display_name(drug_name: str) -> str:
    return drug_name.split("(")[0].strip() or drug_name


def extract_drug_relationships(collection, drug_name: str, groq_api_key: str) -> dict:
    """
    Runs the five targeted queries for one drug, combines their chunks into
    a single Groq call, and returns the parsed relationships plus the pool
    of chunks retrieved (for fact-level evidence lookup — see
    _best_evidence_for). Category-level evidence isn't reliable here: since
    all five categories' text is sent to the model in one combined context,
    a fact can legitimately be drawn from a *different* category's chunk
    than the field it ends up in (e.g. the warnings list turning up inside
    the contraindications section's text), so evidence is matched to each
    extracted fact individually rather than trusted per-category.
    """
    client = Groq(api_key=groq_api_key)
    context_parts = []
    chunks_pool = []
    seen_chunk_keys = set()

    for category, query in CATEGORY_QUERIES.items():
        # Conditions/class tend to need the most context to find reliably
        # (a drug can have many indications, and the class statement is
        # often a single sentence buried among many other pages).
        top_k = {"conditions_treated": 6, "drug_class": 4, "contraindicated_with": 6}.get(category, 3)
        chunks = search(collection, query, top_k=top_k, where={"drug_name": drug_name})
        if not chunks:
            continue
        context_parts.append(f"[{category}]\n" + "\n".join(c["text"] for c in chunks))
        for c in chunks:
            key = (c["page_number"], c["text"][:60])
            if key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(key)
            chunks_pool.append(c)

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(drug_name=drug_name)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context or "No relevant sections were found."},
        ],
        temperature=0.1,
        max_tokens=1200,
        reasoning_effort="low",
    )

    data = _extract_json(response.choices[0].message.content) or {}

    return {
        "conditions_treated": data.get("conditions_treated") or [],
        "drug_interactions": data.get("drug_interactions") or [],
        "drug_class": data.get("drug_class"),
        "black_box_warnings": data.get("black_box_warnings") or [],
        "contraindicated_with": data.get("contraindicated_with") or [],
        "chunks_pool": chunks_pool,
    }


def _best_evidence_for(fact_text: str, chunks_pool: list[dict]) -> dict | None:
    """Finds whichever retrieved chunk's text best overlaps with a specific
    extracted fact (by shared significant words), so the edge-click tooltip
    shows real PDF text that's actually about that fact — not just
    whatever chunk happened to be first in an unrelated category."""
    if not fact_text or not chunks_pool:
        return None
    fact_words = set(re.findall(r"[a-z]{4,}", fact_text.lower()))
    best, best_score = None, -1
    for chunk in chunks_pool:
        chunk_words = set(re.findall(r"[a-z]{4,}", chunk["text"].lower()))
        score = len(fact_words & chunk_words)
        if score > best_score:
            best_score = score
            best = chunk
    chosen = best if best_score > 0 else chunks_pool[0]
    return {
        "text": make_snippet(chosen["text"], max_len=280),
        "page": chosen["page_number"],
        "source_file": chosen["source_file"],
    }


def build_knowledge_graph(collection, groq_api_key: str) -> dict:
    """
    Runs extraction for every indexed drug and assembles the full
    nodes/edges graph, then writes it to the cache file.
    """
    drug_infos = get_drug_info(collection)
    known_names = [d["drug_name"] for d in drug_infos]

    nodes = {}
    edges = []
    drug_details = {}
    per_drug_relationships = {}

    def drug_id(name):
        return f"drug:{name}"

    def add_node(node_id, **kwargs):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, **kwargs}

    for info in drug_infos:
        add_node(
            drug_id(info["drug_name"]),
            label=_display_name(info["drug_name"]),
            type="drug",
            full_name=info["drug_name"],
            source_file=info["source_file"],
        )

    for info in drug_infos:
        name = info["drug_name"]
        try:
            rel = extract_drug_relationships(collection, name, groq_api_key)
        except Exception as e:
            print(f"Knowledge graph extraction failed for {name}: {e}")
            rel = {
                "conditions_treated": [], "drug_interactions": [], "drug_class": None,
                "black_box_warnings": [], "contraindicated_with": [], "chunks_pool": [],
            }
        per_drug_relationships[name] = rel

        did = drug_id(name)
        pool = rel["chunks_pool"]

        seen_conditions = set()
        for cond in rel["conditions_treated"]:
            norm = _normalize_condition(cond)
            if not norm or norm.lower() in seen_conditions:
                continue
            seen_conditions.add(norm.lower())
            cid = f"condition:{norm.lower()}"
            add_node(cid, label=norm, type="condition")
            edges.append({
                "from": did, "to": cid, "label": "treats", "type": "treats",
                "evidence": _best_evidence_for(cond, pool),
            })

        cls = _normalize_class(rel["drug_class"])
        if cls:
            clid = f"class:{cls.lower()}"
            add_node(clid, label=cls, type="class")
            edges.append({
                "from": did, "to": clid, "label": "belongs to", "type": "belongs_to",
                "evidence": _best_evidence_for(rel["drug_class"], pool),
            })

        seen_warnings = set()
        for warning in rel["black_box_warnings"]:
            norm = _normalize_warning(warning, name)
            if not norm or norm.lower() in seen_warnings:
                continue
            seen_warnings.add(norm.lower())
            wid = f"warning:{norm.lower()}"
            add_node(wid, label=norm, type="warning")
            edges.append({
                "from": did, "to": wid, "label": "shares warning", "type": "shares_warning",
                "evidence": _best_evidence_for(warning, pool),
            })

        drug_details[name] = {
            "label": _display_name(name),
            "conditions_treated": sorted({_normalize_condition(c) for c in rel["conditions_treated"] if _normalize_condition(c)}),
            "drug_class": cls,
            "black_box_warnings": rel["black_box_warnings"],
            "interactions": [],  # filled in below once drug-drug edges are resolved
        }

    # Drug <-> drug edges: dedupe both directions into one relationship per
    # pair, keeping "contraindicated with" if either side reported it.
    # Mentions are resolved by name first, falling back to expanding a
    # class mention ("biologic DMARDs") to every indexed drug of that class.
    class_by_drug = {name: details["drug_class"] for name, details in drug_details.items()}
    pair_info = {}
    for info in drug_infos:
        name = info["drug_name"]
        rel = per_drug_relationships[name]
        pool = rel["chunks_pool"]

        for mention in rel["contraindicated_with"]:
            for match in _resolve_drug_mentions(mention, known_names, name, class_by_drug):
                key = frozenset({name, match})
                pair_info[key] = {"type": "contraindicated_with", "label": "contraindicated with",
                                   "evidence": _best_evidence_for(mention, pool)}

        for mention in rel["drug_interactions"]:
            for match in _resolve_drug_mentions(mention, known_names, name, class_by_drug):
                key = frozenset({name, match})
                if key not in pair_info:
                    pair_info[key] = {"type": "interacts_with", "label": "interacts with",
                                       "evidence": _best_evidence_for(mention, pool)}

    for pair, info in pair_info.items():
        a, b = tuple(pair)
        edges.append({
            "from": drug_id(a), "to": drug_id(b),
            "label": info["label"], "type": info["type"],
            "bidirectional": True, "evidence": info["evidence"],
        })
        drug_details[a]["interactions"].append({"drug": _display_name(b), "type": info["type"]})
        drug_details[b]["interactions"].append({"drug": _display_name(a), "type": info["type"]})

    graph = {
        "nodes": list(nodes.values()),
        "edges": edges,
        "drug_details": drug_details,
        "drug_count": len(drug_infos),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache(graph)
    return graph


def _save_cache(graph: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GRAPH_CACHE_PATH, "w") as f:
        json.dump(graph, f, indent=2)


def load_cached_graph() -> dict | None:
    if not os.path.exists(GRAPH_CACHE_PATH):
        return None
    try:
        with open(GRAPH_CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
