import json
import os
import re

# Load the local JSON file once at module load time
DATA_PATH = os.path.join(os.path.dirname(__file__), "otc_drug_data.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    OTC_DATA = json.load(f)

ALLOWED_OTC_DRUGS = list(OTC_DATA.keys())

# Also map common aliases
DRUG_ALIASES = {
    "paracetamol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "tylenol": "acetaminophen",
    "crocin": "acetaminophen",
    "dolo": "acetaminophen",
    "zyrtec": "cetirizine",
    "prilosec": "omeprazole",
    "claritin": "loratadine",
}

# Symptom to OTC drug mapping for patient condition queries
SYMPTOM_MAP = {
    "headache": ["acetaminophen", "ibuprofen"],
    "headaches": ["acetaminophen", "ibuprofen"],
    "migraine": ["ibuprofen", "acetaminophen"],
    "fever": ["acetaminophen", "ibuprofen"],
    "fevers": ["acetaminophen", "ibuprofen"],
    "temperature": ["acetaminophen", "ibuprofen"],
    "pain": ["ibuprofen", "acetaminophen"],
    "body ache": ["ibuprofen", "acetaminophen"],
    "body aches": ["ibuprofen", "acetaminophen"],
    "body pain": ["ibuprofen", "acetaminophen"],
    "backache": ["ibuprofen", "acetaminophen"],
    "back pain": ["ibuprofen", "acetaminophen"],
    "toothache": ["ibuprofen", "acetaminophen"],
    "cramp": ["ibuprofen", "acetaminophen"],
    "cramps": ["ibuprofen", "acetaminophen"],
    "period pain": ["ibuprofen", "acetaminophen"],
    "menstrual": ["ibuprofen", "acetaminophen"],
    "muscle ache": ["ibuprofen", "acetaminophen"],
    "muscle aches": ["ibuprofen", "acetaminophen"],
    "cold": ["acetaminophen", "ibuprofen"],
    "colds": ["acetaminophen", "ibuprofen"],
    "allergy": ["cetirizine", "loratadine"],
    "allergies": ["cetirizine", "loratadine"],
    "allergic": ["cetirizine", "loratadine"],
    "hay fever": ["cetirizine", "loratadine"],
    "sneezing": ["cetirizine", "loratadine"],
    "sneeze": ["cetirizine", "loratadine"],
    "runny nose": ["cetirizine", "loratadine"],
    "watery eyes": ["cetirizine", "loratadine"],
    "itchy": ["cetirizine", "loratadine"],
    "itching": ["cetirizine", "loratadine"],
    "hives": ["cetirizine"],
    "heartburn": ["omeprazole"],
    "acid reflux": ["omeprazole"],
    "acid": ["omeprazole"],
    "acidity": ["omeprazole"],
    "indigestion": ["omeprazole"],
    "gerd": ["omeprazole"],
    "gastric": ["omeprazole"],
    "stomach acid": ["omeprazole"],
}

def lookup_otc_drug(drug_name: str) -> dict | None:
    """
    Returns the drug data dict if drug_name matches an OTC drug,
    else returns None (meaning it is prescription-only or unknown).
    Checks DRUG_ALIASES first, then direct match against OTC_DATA keys.
    Case-insensitive.
    """
    if not drug_name:
        return None
    name = drug_name.lower().strip()
    name = DRUG_ALIASES.get(name, name)
    return OTC_DATA.get(name, None)

def is_otc_drug(drug_name: str) -> bool:
    return lookup_otc_drug(drug_name) is not None

def find_otc_drugs_for_query(query: str) -> list[dict]:
    """
    Finds relevant OTC drugs for a user's question, matching direct drug names,
    brand aliases, and condition/symptom keywords (e.g. 'headache', 'heartburn', 'allergy').
    """
    if not query:
        return []
    
    q_clean = query.lower()
    matched_keys = []

    # 1. Check direct drug names and aliases
    for word in re.findall(r"[a-zA-Z]+", q_clean):
        mapped = DRUG_ALIASES.get(word, word)
        if mapped in OTC_DATA and mapped not in matched_keys:
            matched_keys.append(mapped)

    # 2. Check symptom keywords
    for symptom, drugs in SYMPTOM_MAP.items():
        if re.search(r"\b" + re.escape(symptom) + r"\b", q_clean):
            for d in drugs:
                if d in OTC_DATA and d not in matched_keys:
                    matched_keys.append(d)

    # 3. Text search in indications/purpose if still empty
    if not matched_keys:
        for key, data in OTC_DATA.items():
            text = f"{data.get('purpose', '')} {data.get('indications', '')}".lower()
            for w in re.findall(r"[a-zA-Z]{4,}", q_clean):
                if w in text and key not in matched_keys:
                    matched_keys.append(key)

    return [OTC_DATA[k] for k in matched_keys if k in OTC_DATA]

def get_all_otc_names() -> list[str]:
    """
    Returns a comprehensive list of all known OTC drug names, generic names,
    and brand aliases.
    """
    names = set(ALLOWED_OTC_DRUGS)
    names.update(DRUG_ALIASES.keys())
    for data in OTC_DATA.values():
        if data.get("brand_name"):
            names.add(data["brand_name"].lower().strip())
            for w in re.findall(r"[a-zA-Z]{3,}", data["brand_name"].lower()):
                names.add(w)
        if data.get("generic_name"):
            names.add(data["generic_name"].lower().strip())
            for w in re.findall(r"[a-zA-Z]{3,}", data["generic_name"].lower()):
                names.add(w)
    return list(names)

def find_active_otc_drugs_from_history(chat_history: list[dict]) -> list[dict]:
    """
    Scans chat_history backwards (most recent first) to identify active OTC
    drugs or symptoms discussed in previous turns so conversational context is preserved.
    """
    if not chat_history:
        return []

    for msg in reversed(chat_history):
        content = (msg.get("content", "") or "").lower().strip()
        if not content or content.startswith("[displayed"):
            continue

        matched_keys = []

        # 1. Check direct drug names / aliases in message
        for word in re.findall(r"[a-zA-Z]+", content):
            mapped = DRUG_ALIASES.get(word, word)
            if mapped in OTC_DATA and mapped not in matched_keys:
                matched_keys.append(mapped)

        # 2. Check symptom keywords in message
        for symptom, drugs in SYMPTOM_MAP.items():
            if re.search(r"\b" + re.escape(symptom) + r"\b", content):
                for d in drugs:
                    if d in OTC_DATA and d not in matched_keys:
                        matched_keys.append(d)

        # 3. Check for mentions of brand/generic names in full text
        for key, data in OTC_DATA.items():
            b_name = (data.get("brand_name") or "").lower()
            g_name = (data.get("generic_name") or "").lower()
            if key in content or (b_name and b_name in content) or (g_name and g_name in content):
                if key not in matched_keys:
                    matched_keys.append(key)

        if matched_keys:
            return [OTC_DATA[k] for k in matched_keys if k in OTC_DATA]

    return []

def build_otc_context(drug_data: dict) -> str:
    """
    Formats the OTC drug data into a plain text context string
    for the LLM to use as its document context.
    """
    parts = [
        f"Drug: {drug_data.get('brand_name', '')} ({drug_data.get('generic_name', '')})",
        f"Type: Over-the-Counter (OTC) — available without prescription",
        f"Purpose: {drug_data.get('purpose', '')}",
        f"What it is used for: {drug_data.get('indications', '')}",
        f"Dosage: {drug_data.get('dosage', '')}",
        f"Warnings: {drug_data.get('warnings', '')}",
        f"Do not use if: {drug_data.get('do_not_use', '')}",
        f"Stop use if: {drug_data.get('stop_use', '')}",
        f"Keep out of reach of children: {drug_data.get('keep_out', '')}",
    ]
    return "\n\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


