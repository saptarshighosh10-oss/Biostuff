"""
Cross-reference candidates against protein databases via free REST APIs.

Sources:
  EBI PDBe API    — maps PDB ID → UniProt accession(s)
  UniProt REST    — aggregation/disease annotations for each accession
  EBI Proteins    — known natural/pathogenic variants at mutation positions
"""

import requests
import time

# ── API endpoints ────────────────────────────────────────────────────────────
PDBE_UNIPROT_URL   = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
UNIPROT_ENTRY_URL  = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
EBI_VARIATION_URL  = "https://www.ebi.ac.uk/proteins/api/variation/{accession}"

AGGREGATION_KEYWORDS = {
    "aggregat", "amyloid", "inclusion body", "insoluble",
    "precipitat", "fibrillat", "misfold", "fibril",
}


def _get(url: str, **kwargs) -> dict | None:
    try:
        r = requests.get(url, timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ── PDB → UniProt ────────────────────────────────────────────────────────────

def pdb_to_uniprot(pdb_id: str) -> list[str]:
    """
    Map a PDB ID to UniProt accession(s) using the EBI PDBe mappings API.
    Returns up to 3 accessions, most common chains first.
    """
    data = _get(PDBE_UNIPROT_URL.format(pdb_id=pdb_id.lower()))
    if not data:
        return _pdb_to_uniprot_fallback(pdb_id)

    entry = data.get(pdb_id.lower(), {})
    uniprot_map = entry.get("UniProt", {})
    return list(uniprot_map.keys())[:3]


def _pdb_to_uniprot_fallback(pdb_id: str) -> list[str]:
    """Fallback: search UniProt by PDB cross-reference."""
    data = _get(
        UNIPROT_SEARCH_URL,
        params={
            "query": f"(database:pdb) AND (xref_pdb:{pdb_id.upper()})",
            "format": "json",
            "size": 3,
            "fields": "accession",
        },
    )
    if not data:
        return []
    return [r["primaryAccession"] for r in data.get("results", [])]


# ── UniProt annotations ──────────────────────────────────────────────────────

def get_uniprot_aggregation_evidence(accession: str) -> dict:
    """
    Fetch UniProt entry and extract aggregation/disease annotations.
    """
    data = _get(UNIPROT_ENTRY_URL.format(accession=accession))
    if not data:
        return {
            "has_aggregation_annotation": False,
            "disease_associations": [],
            "relevant_comments": [],
            "protein_name": "",
        }

    protein_name = (
        data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
    )

    comments = data.get("comments", [])
    relevant_comments = []
    has_agg = False
    diseases = []

    for comment in comments:
        ctype = comment.get("commentType", "")
        if ctype == "DISEASE":
            name = comment.get("disease", {}).get("diseaseName", "")
            if name:
                diseases.append(name)
        for text_obj in comment.get("texts", []):
            val = text_obj.get("value", "").lower()
            if any(kw in val for kw in AGGREGATION_KEYWORDS):
                has_agg = True
                relevant_comments.append(text_obj.get("value", "")[:200])

    # also scan keywords section
    for kw_group in data.get("keywords", []):
        name = kw_group.get("name", "").lower()
        if any(k in name for k in AGGREGATION_KEYWORDS):
            has_agg = True

    return {
        "has_aggregation_annotation": has_agg,
        "disease_associations": diseases[:3],
        "relevant_comments": relevant_comments[:2],
        "protein_name": protein_name,
    }


# ── Known variants at mutation positions ─────────────────────────────────────

def get_known_variants_at_positions(accession: str, positions: list[int]) -> list[dict]:
    """
    Query EBI Proteins API for natural/pathogenic variants at our mutation positions.
    positions are 0-indexed; EBI uses 1-indexed so we convert.
    """
    data = _get(
        EBI_VARIATION_URL.format(accession=accession),
        headers={"Accept": "application/json"},
    )
    if not data:
        return []

    pos_set = {p + 1 for p in positions}  # convert to 1-indexed
    hits = []

    for feat in data.get("features", []):
        try:
            begin = int(feat.get("begin", 0))
        except (ValueError, TypeError):
            continue
        if begin not in pos_set:
            continue

        sigs = feat.get("clinicalSignificances", [])
        clinical = sigs[0].get("type", "") if sigs else ""
        hits.append({
            "position_0idx": begin - 1,
            "type": feat.get("type", ""),
            "description": feat.get("description", "")[:120],
            "clinical_significance": clinical,
            "is_pathogenic": "pathogenic" in clinical.lower(),
        })

    return hits[:5]


# ── Public API ───────────────────────────────────────────────────────────────

def lookup_candidate(candidate: dict) -> dict:
    """
    Full database lookup for one candidate.
    Returns a database_evidence dict ready to embed in the candidate.
    """
    pdb_id = candidate.get("anchor_pdb", "")
    mutations = candidate.get("mutations", [])
    mutation_positions = [m[0] for m in mutations]

    evidence = {
        "uniprot_ids": [],
        "protein_name": "",
        "has_aggregation_annotation": False,
        "disease_associations": [],
        "relevant_comments": [],
        "variants_at_mutation_sites": [],
        "database_confidence": "none",
    }

    if not pdb_id:
        return evidence

    uniprot_ids = pdb_to_uniprot(pdb_id)
    if not uniprot_ids:
        return evidence

    evidence["uniprot_ids"] = uniprot_ids
    accession = uniprot_ids[0]

    agg = get_uniprot_aggregation_evidence(accession)
    evidence.update(agg)

    if mutation_positions:
        variants = get_known_variants_at_positions(accession, mutation_positions)
        evidence["variants_at_mutation_sites"] = variants

    # confidence scoring
    signals = 0
    if evidence["has_aggregation_annotation"]:
        signals += 2
    if any(v["is_pathogenic"] for v in evidence["variants_at_mutation_sites"]):
        signals += 3
    elif evidence["variants_at_mutation_sites"]:
        signals += 1
    if evidence["disease_associations"]:
        signals += 1

    evidence["database_confidence"] = (
        "high"   if signals >= 4 else
        "medium" if signals >= 2 else
        "low"    if signals >= 1 else
        "none"
    )

    return evidence
