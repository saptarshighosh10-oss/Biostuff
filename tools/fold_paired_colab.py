"""COLAB v2 — proper paired antibody folding + REAL surface-aggregation features.

Improves on v1 (ESMFold, single chain, approximate surface) by:
  1. Folding heavy + light chains as a real PAIR with IgFold (correct 3D interface).
  2. Computing REAL solvent-accessible surface area with freesasa (not a neighbour proxy).
  3. A real SAP (Spatial Aggregation Propensity) score = surface-exposed hydrophobicity.

Input : paired_chains_to_fold.csv  (columns: seq_id, mode, H, L)
Output: structure_features_v2.csv   (one row per seq_id)

Run on Colab with a GPU. IgFold is CPU-fine too but GPU is faster. Resumable.
"""
import csv, os, io
import numpy as np

IN_CSV, OUT_CSV = "paired_chains_to_fold.csv", "structure_features_v2.csv"
KD = {"ALA":1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS":2.5,"GLN":-3.5,"GLU":-3.5,
      "GLY":-0.4,"HIS":-3.2,"ILE":4.5,"LEU":3.8,"LYS":-3.9,"MET":1.9,"PHE":2.8,
      "PRO":-1.6,"SER":-0.8,"THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL":4.2}


def parse_ca(pdb_path):
    """Return list of (chain, resnum, resname, x, y, z, bfactor) for CA atoms."""
    out = []
    for line in open(pdb_path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.append((line[21], int(line[22:26]), line[17:20].strip(),
                        float(line[30:38]), float(line[38:46]), float(line[46:54]),
                        float(line[60:66])))
    return out


def features_from_pdb(pdb_path, radius=10.0):
    import freesasa
    ca = parse_ca(pdb_path)
    n = len(ca)
    if n == 0:
        raise ValueError("no CA atoms")
    struct = freesasa.Structure(pdb_path)
    res_areas = freesasa.calc(struct).residueAreas()  # [chain][resnum].relativeTotal

    rel = np.zeros(n); kd = np.zeros(n)
    coords = np.array([[a[3], a[4], a[5]] for a in ca])
    bfac = np.array([a[6] for a in ca])
    for i, (ch, num, resname, *_ ) in enumerate(ca):
        try:
            rel[i] = min(res_areas[ch][str(num)].relativeTotal, 1.0)
        except Exception:
            rel[i] = 0.0
        kd[i] = KD.get(resname, 0.0)
    hydrophobic = np.clip(kd, 0, None)
    surf_h = rel * hydrophobic                       # surface-exposed hydrophobicity per residue
    sap = float(surf_h.sum())
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    patch = float(max((surf_h[(d[i] < radius)].sum()) for i in range(n)))
    return {
        "sap_score": sap,
        "sap_per_residue": sap / n,
        "max_hydrophobic_patch": patch,
        "hydrophobic_surface_fraction": float((surf_h > 0.1).mean()),
        "mean_rel_sasa": float(rel.mean()),
        "mean_prmsd": float(bfac.mean()),   # IgFold confidence (lower = better)
        "n_res": n,
    }


def main():
    from igfold import IgFoldRunner
    igfold = IgFoldRunner()
    rows = list(csv.DictReader(open(IN_CSV)))
    done = {r["seq_id"] for r in csv.DictReader(open(OUT_CSV))} if os.path.exists(OUT_CSV) else set()
    cols = ["seq_id", "sap_score", "sap_per_residue", "max_hydrophobic_patch",
            "hydrophobic_surface_fraction", "mean_rel_sasa", "mean_prmsd", "n_res"]
    new = not os.path.exists(OUT_CSV)
    ok = fail = 0
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        for i, r in enumerate(rows, 1):
            if r["seq_id"] in done:
                continue
            seqs = {"H": r["H"]} if r["mode"] == "single" or not r["L"] else {"H": r["H"], "L": r["L"]}
            pdb = f"/tmp/{r['seq_id']}.pdb"
            try:
                igfold.fold(pdb, sequences=seqs, do_refine=False, do_renum=False)
                feats = features_from_pdb(pdb)
                w.writerow({"seq_id": r["seq_id"], **feats}); f.flush(); ok += 1
                print(f"[{i}/{len(rows)}] {r['seq_id']} ok sap={feats['sap_score']:.1f}", flush=True)
            except Exception as e:
                fail += 1; print(f"[{i}/{len(rows)}] {r['seq_id']} FAILED: {e}", flush=True)
            finally:
                if os.path.exists(pdb): os.remove(pdb)
    print(f"DONE: {ok} folded, {fail} failed -> {OUT_CSV}")


if __name__ == "__main__":
    main()
