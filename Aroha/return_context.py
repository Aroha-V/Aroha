import os
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PC_API_Key"))
index = pc.Index("aroha-db")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Known entity lists for dynamic filter building
# ---------------------------------------------------------------------------
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
    "Andaman and Nicobar", "Dadra and Nagar Haveli", "Daman and Diu", "Lakshadweep",
]

DISEASES = [
    "Dengue", "Malaria", "Cholera", "Typhoid", "Chikungunya", "Leptospirosis",
    "Hepatitis", "Hepatitis A", "Hepatitis B", "Hepatitis E", "Influenza",
    "COVID", "COVID-19", "Rabies", "Anthrax", "Plague", "Meningitis",
    "Encephalitis", "Japanese Encephalitis", "Scrub Typhus", "Kala Azar",
    "Leishmaniasis", "Tuberculosis", "TB", "Diarrhea", "Diarrhoea",
    "Acute Diarrheal Disease", "Acute Diarrhoeal Disease", "Acute Respiratory Illness",
    "Measles", "Mumps", "Chickenpox", "Swine Flu", "H1N1", "Zika",
    "Nipah", "Hand Foot Mouth Disease", "Food Poisoning", "Gastroenteritis",
    "Rubella", "Fever", "Viral Fever",
]

_STATE_MAP   = {s.lower(): s for s in INDIAN_STATES}
_DISEASE_MAP = {d.lower(): d for d in DISEASES}

# Dataset stores disease names with roman-numeral list prefixes like "v. Dengue"
# Strip them before exact-match filtering
_PREFIX_RE = re.compile(r'^[ivxlcdmIVXLCDM]+\.\s*', re.IGNORECASE)


def _clean_disease(name: str) -> str:
    return _PREFIX_RE.sub("", name).strip()


def _extract_entities(query: str):
    """Return (state, disease) found in the query, or (None, None)."""
    q = query.lower()
    found_state, found_disease = None, None

    for key, canonical in _STATE_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', q):
            found_state = canonical
            break

    for key, canonical in _DISEASE_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', q):
            found_disease = canonical
            break

    return found_state, found_disease


def _build_filter(state, disease):
    if state and disease:
        return {"$and": [{"state": {"$eq": state}}, {"disease": {"$eq": disease}}]}
    if state:
        return {"state": {"$eq": state}}
    if disease:
        return {"disease": {"$eq": disease}}
    return None


def return_context(query: str) -> str:
    state, disease = _extract_entities(query)
    pinecone_filter = _build_filter(state, disease)
    query_embedding = model.encode(query).tolist()

    # --- First pass: with metadata filter ---
    kwargs = dict(vector=query_embedding, top_k=15, include_metadata=True)
    if pinecone_filter:
        kwargs["filter"] = pinecone_filter

    results = index.query(**kwargs)
    matches = [m for m in results.get("matches", []) if m.get("score", 0) >= 0.2]

    # --- Fallback: drop filter if too narrow ---
    if not matches and pinecone_filter:
        results = index.query(vector=query_embedding, top_k=15, include_metadata=True)
        matches = [m for m in results.get("matches", []) if m.get("score", 0) >= 0.2]

    if not matches:
        return "query_error"

    seen, documents = set(), []
    for match in matches:
        meta = match["metadata"]
        # Deduplicate
        key = (meta.get("state"), meta.get("district"),
               _clean_disease(meta.get("disease", "")),
               meta.get("start_date"), meta.get("cases"))
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            f"State: {meta.get('state', '')}\n"
            f"District: {meta.get('district', '')}\n"
            f"Disease: {_clean_disease(meta.get('disease', ''))}\n"
            f"Cases: {meta.get('cases', 0)}\n"
            f"Deaths: {meta.get('deaths', 0)}\n"
            f"Status: {meta.get('status', '')}\n"
            f"Start Date: {meta.get('start_date', '')}\n\n"
            f"{meta.get('text', '')}"
        )

    return "\n\n---\n\n".join(documents)
