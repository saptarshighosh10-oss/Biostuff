import os
import sys

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import numpy as np
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

def run_short_md(pdb_string: str, steps: int = 5000) -> dict | None:
    if not OPENMM_AVAILABLE:
        print("  OpenMM is not installed. Skipping molecular dynamics calculation gracefully.")
        return None
    try:
        from io import StringIO
        pdb_file = StringIO(pdb_string)
        pdb = app.PDBFile(pdb_file)

        # Load implicit solvent forcefields (GBn2)
        forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

        integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)

        # Select fastest execution backend available (CUDA -> OpenCL -> Reference CPU)
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
        except Exception:
            try:
                platform = mm.Platform.getPlatformByName('OpenCL')
                simulation = app.Simulation(pdb.topology, system, integrator, platform)
            except Exception:
                platform = mm.Platform.getPlatformByName('Reference')
                simulation = app.Simulation(pdb.topology, system, integrator, platform)

        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=50)

        positions_history = []
        sample_interval = 100
        total_samples = steps // sample_interval

        for _ in range(total_samples):
            simulation.step(sample_interval)
            state = simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            positions_history.append(pos)

        if not positions_history:
            return None

        positions_history = np.array(positions_history)
        mean_positions = np.mean(positions_history, axis=0)
        sq_deviations = np.sum((positions_history - mean_positions) ** 2, axis=2)
        mean_sq_deviations = np.mean(sq_deviations, axis=0)
        atomic_rmsf = np.sqrt(mean_sq_deviations) * 10.0  # nm to Angstroms

        # Map atomic indices back to per-residue CA positions
        residue_rmsf = {}
        for atom in pdb.topology.atoms():
            if atom.name == 'CA':
                residue_rmsf[atom.residue.index] = float(atomic_rmsf[atom.index])

        # Resolve edge-case structural missing atoms
        residue_rmsf_all = {}
        for atom in pdb.topology.atoms():
            res_idx = atom.residue.index
            if res_idx not in residue_rmsf:
                if res_idx not in residue_rmsf_all:
                    residue_rmsf_all[res_idx] = []
                residue_rmsf_all[res_idx].append(float(atomic_rmsf[atom.index]))

        per_residue_rmsf = []
        for res in pdb.topology.residues():
            idx = res.index
            if idx in residue_rmsf:
                per_residue_rmsf.append(residue_rmsf[idx])
            elif idx in residue_rmsf_all:
                per_residue_rmsf.append(float(np.mean(residue_rmsf_all[idx])))
            else:
                per_residue_rmsf.append(0.0)

        if not per_residue_rmsf:
            return None

        mean_rmsf = float(np.mean(per_residue_rmsf))
        std_rmsf = float(np.std(per_residue_rmsf))
        threshold = mean_rmsf + std_rmsf

        high_flexibility_segments = []
        in_segment = False
        start_idx = -1
        for i, r in enumerate(per_residue_rmsf):
            if r > threshold:
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    in_segment = False
                    high_flexibility_segments.append([start_idx, i - 1])
        if in_segment:
            high_flexibility_segments.append([start_idx, len(per_residue_rmsf) - 1])

        return {
            "per_residue_rmsf": [round(x, 3) for x in per_residue_rmsf],
            "high_flexibility_segments": high_flexibility_segments,
            "mean_rmsf": round(mean_rmsf, 3)
        }
    except Exception as e:
        print(f"  Warning: Implicit solvent MD calculation failed gracefully: {e}")
        return None

def rmsf_at_mutations(rmsf: list[float], mutations: list) -> list[float]:
    results = []
    for mut in mutations:
        pos = mut[0]
        if 0 <= pos < len(rmsf):
            results.append(round(rmsf[pos], 3))
        else:
            results.append(0.0)
    return results
