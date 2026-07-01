"""
Fetch confirmed-working antibody sequences from PDB.
Searches multiple keywords (ANTIBODY, IMMUNOGLOBULIN, FAB) to maximise coverage.
Uses the PDB REST API — no account needed, runs on any laptop.
"""

import requests
import json
from pathlib import Path

PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"

SEARCH_TERMS = ["ANTIBODY", "IMMUNOGLOBULIN", "FAB FRAGMENT"]


def _build_query(keyword: str, rows: int) -> dict:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct_keywords.pdbx_keywords",
                "operator": "contains_words",
                "value": keyword,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }


def search_antibody_entries(max_results: int = 300) -> list[str]:
    """
    Search PDB with multiple antibody keywords and return deduplicated PDB IDs.
    Queries ANTIBODY + IMMUNOGLOBULIN + FAB FRAGMENT to maximise diversity.
    """
    per_term = max(max_results // len(SEARCH_TERMS), 100)
    all_ids: set[str] = set()

    for term in SEARCH_TERMS:
        try:
            response = requests.post(
                PDB_SEARCH_URL,
                json=_build_query(term, per_term),
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()
            ids = [hit["identifier"] for hit in results.get("result_set", [])]
            all_ids.update(ids)
            print(f"      '{term}': {len(ids)} entries found")
        except requests.RequestException as e:
            print(f"      '{term}': search failed ({e})")

    return list(all_ids)[:max_results]


def fetch_entry_metadata(pdb_id: str) -> dict:
    """Fetch metadata for a single PDB entry."""
    url = f"{PDB_DATA_URL}/{pdb_id.lower()}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def save_pdb_ids(pdb_ids: list[str], output_path: str = "data/antibody_pdb_ids.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pdb_ids, f, indent=2)
    print(f"Saved {len(pdb_ids)} PDB IDs to {output_path}")


if __name__ == "__main__":
    print("Fetching antibody PDB entries...")
    pdb_ids = search_antibody_entries(max_results=300)
    print(f"Found {len(pdb_ids)} unique entries")
    save_pdb_ids(pdb_ids)
