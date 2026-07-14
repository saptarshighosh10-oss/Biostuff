"""Identity-graph utilities for Plan C leakage control."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import re

from data.contract import CANONICAL_AMINO_ACIDS, NormalizedRow


MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_OPEN = -2
GAP_EXTEND = -1
IDENTITY_MIN = 0.9
LENGTH_RATIO_MIN = 0.85
LENGTH_RATIO_MAX = 1.15

_NEG_INF = -10_000_000
_AA_RE = re.compile(rf"^[{CANONICAL_AMINO_ACIDS}]+$")


def _normalize_aa_chain(chain: str) -> str:
    if not isinstance(chain, str):
        raise TypeError("chain sequences must be strings")
    value = chain.strip().upper()
    if not value:
        raise ValueError("chain sequences must be non-empty")
    if not _AA_RE.fullmatch(value):
        raise ValueError("chain sequences must contain only canonical amino-acid codes")
    return value


def _identity_key(row: NormalizedRow) -> str:
    if not isinstance(row.pair_id, str) or not row.pair_id:
        chain_key = row.chain_id or "chain_unknown"
        molecule = row.molecule_id or "molecule_unknown"
        return f"single:{chain_key}:{molecule}"
    return f"pair:{row.pair_id}"


def _row_sequences(row: NormalizedRow) -> tuple[str, ...]:
    sequences: list[str] = []
    if row.vh_sequence:
        sequences.append(_normalize_aa_chain(row.vh_sequence))
    if row.vl_sequence:
        sequences.append(_normalize_aa_chain(row.vl_sequence))
    if not sequences:
        sequences.append(_normalize_aa_chain(row.sequence))
    dedup: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        if sequence in seen:
            continue
        seen.add(sequence)
        dedup.append(sequence)
    return tuple(dedup)


def _state_key(state: tuple[int, int, int]) -> tuple[int, int, int]:
    score, matches, alignment_len = state
    return score, matches, -alignment_len


def _best_state(*states: tuple[int, int, int]) -> tuple[int, int, int]:
    return max(states, key=_state_key)


def _score_aligned_pairs(a: str, b: str) -> tuple[int, int]:
    """Return `(matches, alignment_length)` for a globally aligned pair."""
    if not a or not b:
        return 0, 0

    ratio = len(a) / len(b)
    if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
        return 0, 0

    if a == b:
        return len(a), len(a)

    n = len(a)
    m = len(b)
    mtx_m = [[(_NEG_INF, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    mtx_x = [[(_NEG_INF, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    mtx_y = [[(_NEG_INF, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    mtx_m[0][0] = (0, 0, 0)
    for i in range(1, n + 1):
        mtx_x[i][0] = (GAP_OPEN + GAP_EXTEND * (i - 1), 0, i)
    for j in range(1, m + 1):
        mtx_y[0][j] = (GAP_OPEN + GAP_EXTEND * (j - 1), 0, j)

    for i in range(1, n + 1):
        char_a = a[i - 1]
        for j in range(1, m + 1):
            char_b = b[j - 1]
            match = 1 if char_a == char_b else 0
            from_m = _best_state(
                mtx_m[i - 1][j - 1],
                mtx_x[i - 1][j - 1],
                mtx_y[i - 1][j - 1],
            )
            state_m = (
                from_m[0] + (MATCH_SCORE if match else MISMATCH_SCORE),
                from_m[1] + match,
                from_m[2] + 1,
            )

            state_x = _best_state(
                (mtx_m[i - 1][j][0] + GAP_OPEN, mtx_m[i - 1][j][1], mtx_m[i - 1][j][2] + 1),
                (mtx_x[i - 1][j][0] + GAP_EXTEND, mtx_x[i - 1][j][1], mtx_x[i - 1][j][2] + 1),
            )
            state_y = _best_state(
                (mtx_m[i][j - 1][0] + GAP_OPEN, mtx_m[i][j - 1][1], mtx_m[i][j - 1][2] + 1),
                (mtx_y[i][j - 1][0] + GAP_EXTEND, mtx_y[i][j - 1][1], mtx_y[i][j - 1][2] + 1),
            )

            mtx_m[i][j] = state_m
            mtx_x[i][j] = state_x
            mtx_y[i][j] = state_y

    final_state = _best_state(mtx_m[n][m], mtx_x[n][m], mtx_y[n][m])
    if final_state[0] == _NEG_INF:
        return 0, 0
    return final_state[1], final_state[2]


def is_similar(chain_a: str, chain_b: str) -> bool:
    """Conservative predicate for near-identity sequence links."""
    sequence_a = _normalize_aa_chain(chain_a)
    sequence_b = _normalize_aa_chain(chain_b)
    if sequence_a == sequence_b:
        return True

    if len(sequence_a) != len(sequence_b):
        ratio = len(sequence_a) / len(sequence_b)
        if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
            return False

    matches, alignment_length = _score_aligned_pairs(sequence_a, sequence_b)
    if alignment_length == 0:
        return False
    return (matches / alignment_length) >= IDENTITY_MIN


def build_identity_graph(rows: Iterable[NormalizedRow]) -> dict[str, set[str]]:
    """Return an adjacency map of deterministic identity components."""
    rows = list(rows)
    if not rows:
        return {}

    sequences_by_node: dict[str, set[str]] = {}
    for row in rows:
        key = _identity_key(row)
        sequences_by_node.setdefault(key, set()).update(_row_sequences(row))

    node_ids = sorted(sequences_by_node)
    adjacency = {node_id: set() for node_id in node_ids}

    for i, first_node in enumerate(node_ids):
        first_sequences = sorted(sequences_by_node[first_node])
        first_is_single = first_node.startswith("single:")
        first_chain_id = None
        if first_is_single:
            _, first_chain_id, _ = first_node.split(":", 2)
        for second_node in node_ids[i + 1 :]:
            second_sequences = sorted(sequences_by_node[second_node])
            second_is_single = second_node.startswith("single:")
            if first_is_single and second_is_single:
                _, second_chain_id, _ = second_node.split(":", 2)
                if first_chain_id != second_chain_id:
                    continue
            for sequence_a in first_sequences:
                for sequence_b in second_sequences:
                    if is_similar(sequence_a, sequence_b):
                        adjacency[first_node].add(second_node)
                        adjacency[second_node].add(first_node)
                        break
                if second_node in adjacency[first_node]:
                    break

    return adjacency


def build_identity_components(
    rows: Iterable[NormalizedRow],
) -> dict[str, str]:
    """
    Assign deterministic identity component IDs per row record.
    """
    rows = list(rows)
    if not rows:
        return {}

    adjacency = build_identity_graph(rows)
    if not adjacency:
        return {}

    row_to_node = [_identity_key(row) for row in rows]
    node_to_records: dict[str, list[str]] = {}
    for row, node in zip(rows, row_to_node):
        node_to_records.setdefault(node, []).append(row.record_id)

    record_to_component: dict[str, str] = {}
    visited: set[str] = set()
    for start_node in sorted(adjacency):
        if start_node in visited:
            continue

        component_nodes: list[str] = []
        queue = deque([start_node])
        visited.add(start_node)
        while queue:
            node = queue.popleft()
            component_nodes.append(node)
            for neighbor in sorted(adjacency.get(node, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component_nodes.sort()
        component_id = component_nodes[0]
        component_set = set(component_nodes)
        for node in component_set:
            for record_id in node_to_records.get(node, []):
                if record_id in record_to_component:
                    raise ValueError(
                        "duplicate record IDs cannot be assigned to multiple identity components"
                    )
                record_to_component[record_id] = component_id

    return record_to_component


def assign_identity_components(
    rows: Iterable[NormalizedRow],
) -> dict[str, str]:
    """Backward-compatible alias for `build_identity_components`."""
    return build_identity_components(rows)
