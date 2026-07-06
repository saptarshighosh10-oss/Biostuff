import os
import sys

# Pawar et al. 2005 beta-strand propensity scale
PAWAR_BETA_SCALE = {
    "V": 1.00, "I": 0.95, "Y": 0.85, "F": 0.80, "W": 0.75,
    "L": 0.70, "T": 0.65, "C": 0.60, "A": 0.45, "M": 0.40,
    "G": 0.35, "S": 0.30, "H": 0.25, "Q": 0.20, "N": 0.15,
    "E": 0.10, "D": 0.05, "K": 0.05, "R": 0.05, "P": 0.00
}

# AGGRESCAN a3v scale (Conchillo-Sole et al. 2007)
AGGRESCAN_A3V_SCALE = {
    "I": 1.822, "V": 1.392, "F": 1.254, "L": 1.019, "W": 0.596,
    "M": 0.513, "A": 0.285, "Y": 0.167, "C": -0.015, "T": -0.160,
    "G": -0.189, "S": -0.370, "H": -0.421, "Q": -0.569, "N": -0.662,
    "E": -0.835, "D": -0.982, "K": -1.042, "R": -1.142, "P": -1.411
}

# Hydrophobicity and charge matrices mapping to original features/sequence.py schema
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {"R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

def _sliding_window_average(sequence: str, scale: dict, window: int = 5) -> list:
    L = len(sequence)
    scores = []
    half_w = window // 2
    for i in range(L):
        start = max(0, i - half_w)
        end = min(L, i + half_w + 1)
        win_seq = sequence[start:end]
        win_score = sum(scale.get(aa, 0.0) for aa in win_seq) / len(win_seq)
        scores.append(win_score)
    return scores

def _get_hotspots(scores: list, threshold: float) -> list:
    segments = []
    in_hotspot = False
    start_idx = -1
    for i, s in enumerate(scores):
        if s > threshold:
            if not in_hotspot:
                in_hotspot = True
                start_idx = i
        else:
            if in_hotspot:
                in_hotspot = False
                mean_seg_score = sum(scores[start_idx:i]) / (i - start_idx)
                segments.append([start_idx, i - 1, round(mean_seg_score, 3)])
    if in_hotspot:
        mean_seg_score = sum(scores[start_idx:]) / (len(scores) - start_idx)
        segments.append([start_idx, len(scores) - 1, round(mean_seg_score, 3)])
    return segments

def run_all_predictors(sequence: str) -> dict:
    L = len(sequence)
    if L == 0:
        return {"tango": {}, "aggrescan": {}, "zyggregator": {}, "consensus_hotspots": []}

    # 1. TANGO approximation
    tango_scores = _sliding_window_average(sequence, PAWAR_BETA_SCALE, window=5)
    tango_threshold = 0.45
    tango_hotspots = _get_hotspots(tango_scores, tango_threshold)

    # 2. AGGRESCAN approximation
    agg_scores = _sliding_window_average(sequence, AGGRESCAN_A3V_SCALE, window=5)
    agg_threshold = 0.0
    agg_hotspots = _get_hotspots(agg_scores, agg_threshold)

    # 3. Zyggregator approximation (Combining hydrophobicity, beta propensity and charge limits)
    zygg_scale = {}
    for aa in PAWAR_BETA_SCALE:
        hyd_norm = (HYDROPHOBICITY.get(aa, 0.0) + 4.5) / 9.0
        beta = PAWAR_BETA_SCALE.get(aa, 0.0)
        ch = abs(CHARGE.get(aa, 0.0))
        zygg_scale[aa] = 0.4 * hyd_norm + 0.6 * beta - 0.2 * ch

    zygg_scores = _sliding_window_average(sequence, zygg_scale, window=5)
    zygg_threshold = 0.4
    zygg_hotspots = _get_hotspots(zygg_scores, zygg_threshold)

    # Calculate consensus regions
    tango_flags = [s > tango_threshold for s in tango_scores]
    agg_flags = [s > agg_threshold for s in agg_scores]
    zygg_flags = [s > zygg_threshold for s in zygg_scores]

    consensus_hotspots = []
    in_consensus = False
    start_idx = -1
    for i in range(L):
        agree_count = int(tango_flags[i]) + int(agg_flags[i]) + int(zygg_flags[i])
        if agree_count >= 2:
            if not in_consensus:
                in_consensus = True
                start_idx = i
        else:
            if in_consensus:
                in_consensus = False
                max_agree = max(int(tango_flags[j]) + int(agg_flags[j]) + int(zygg_flags[j]) for j in range(start_idx, i))
                consensus_hotspots.append([start_idx, i - 1, max_agree])
    if in_consensus:
        max_agree = max(int(tango_flags[j]) + int(agg_flags[j]) + int(zygg_flags[j]) for j in range(start_idx, L))
        consensus_hotspots.append([start_idx, L - 1, max_agree])

    return {
        "tango": {
            "mean_score": round(sum(tango_scores) / L, 3),
            "hotspot_segments": tango_hotspots,
            "per_residue_scores": [round(s, 3) for s in tango_scores]
        },
        "aggrescan": {
            "mean_score": round(sum(agg_scores) / L, 3),
            "hotspot_segments": agg_hotspots,
            "per_residue_scores": [round(s, 3) for s in agg_scores]
        },
        "zyggregator": {
            "mean_score": round(sum(zygg_scores) / L, 3),
            "hotspot_segments": zygg_hotspots,
            "per_residue_scores": [round(s, 3) for s in zygg_scores]
        },
        "consensus_hotspots": consensus_hotspots
    }

def mutation_hits_hotspot(mutations: list, consensus_hotspots: list) -> bool:
    for mut in mutations:
        pos = mut[0]
        for start, end, _ in consensus_hotspots:
            if start <= pos <= end:
                return True
    return False
