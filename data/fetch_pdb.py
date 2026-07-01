"""
Fetch confirmed-working antibody sequences from PDB.
Uses the PDB REST API — no account needed, runs on any laptop.
"""

import requests
import json
from pathlib import Path

PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"

ANTIBODY_QUERY = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "struct_keywords.pdbx_keywords",
            "operator": "contains_words",
            "value": "ANTIBODY"
        }
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {"start": 0, "rows": 100},
        "results_content_type": ["experimental"]
    }
}


def search_antibody_entries(max_results: int = 100) -> list[str]:
    """Return a list of PDB IDs for antibody structures."""
    query = ANTIBODY_QUERY.copy()
    query["request_options"]["paginate"]["rows"] = max_results

    response = requests.post(PDB_SEARCH_URL, json=query, timeout=30)
    response.raise_for_status()

    results = response.json()
    return [hit["identifier"] for hit in results.get("result_set", [])]


def fetch_entry_metadata(pdb_id: str) -> dict:
    """Fetch metadata for a single PDB entry."""
    url = f"{PDB_DATA_URL}/{pdb_id.lower()}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def save_pdb_ids(pdb_ids: list[str], output_path: str = "data/antibody_pdb_ids.json"):
    """Save fetched PDB IDs to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pdb_ids, f, indent=2)
    print(f"Saved {len(pdb_ids)} PDB IDs to {output_path}")


if __name__ == "__main__":
    print("Fetching antibody PDB entries...")
    pdb_ids = search_antibody_entries(max_results=100)
    print(f"Found {len(pdb_ids)} entries")
    save_pdb_ids(pdb_ids)
