#!/usr/bin/env python3
"""
optimal_path.py

Find the globally optimal "smoothest" trajectory through multiple candidates at each timestep,
minimizing a weighted sum of:
  - per-candidate fit factor (parsed from filename)
  - signal delta between consecutive timesteps (MSE on interpolated common q-grid)
  - structure delta between consecutive timesteps (Kabsch-aligned RMSD)

Assumes files like:
  01_000.17533577.dat
  01_000.17533577.xyz
  02_000.46654335.dat
  02_000.46654335.xyz
etc.

Outputs:
  - prints the chosen path
  - writes a multi-frame XYZ trajectory containing the chosen structures
"""

import os
import re
import glob
import argparse
import numpy as np
import heapq

# -----------------------------
# Filename parsing
# Example: 01_000.17533577.dat
#  - timestep: 01
#  - fit: 0.17533577
# -----------------------------
NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path):
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    t = int(m.group("t"))
    fit = float(m.group("fit"))
    return t, fit


# -----------------------------
# IO helpers
# -----------------------------
def read_dat(path):
    """
    Reads .dat as either:
      - two columns: q, I
      - one column: I (q assumed to be index)
    Returns (q, I).
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        I = arr.astype(np.float64)
        q = np.arange(I.size, dtype=np.float64)
    else:
        if arr.shape[1] >= 2:
            q = arr[:, 0].astype(np.float64)
            I = arr[:, 1].astype(np.float64)
        else:
            I = arr[:, 0].astype(np.float64)
            q = np.arange(I.size, dtype=np.float64)
    return q, I


def read_xyz_lines(path):
    """
    Reads an XYZ file and returns:
      n (int), comment (str), body_lines (list[str] length n)
    This preserves original atom symbols/order in the file.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    comment = lines[1].rstrip("\n") if len(lines) > 1 else ""
    body = lines[2 : 2 + n]
    return n, comment, body


def read_xyz_coords(path):
    """
    Minimal XYZ reader for coordinates only: returns coords (N,3) float64.
    Assumes standard XYZ format.
    """
    with open(path, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()  # comment
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            xyz[i, 0] = float(parts[1])
            xyz[i, 1] = float(parts[2])
            xyz[i, 2] = float(parts[3])
    return xyz


# -----------------------------
# Kabsch RMSD
# -----------------------------
def kabsch_rmsd(P, Q, indices=None):
    """
    Calculate Kabsch-aligned RMSD between structures P and Q.
    
    Args:
        P: (N, 3) array of coordinates for structure 1
        Q: (N, 3) array of coordinates for structure 2
        indices: Optional list of atom indices to include in RMSD calculation.
                 If None, all atoms are used.
    
    Returns:
        RMSD value (float)
    """
    if indices is not None:
        P = P[indices, :]
        Q = Q[indices, :]
    
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Pr = Pc @ R
    d = Pr - Qc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# -----------------------------
# Signal helpers
# -----------------------------
def interp_to(q_src, I_src, q_ref):
    return np.interp(q_ref, q_src, I_src).astype(np.float64)


def signal_mse(a, b):
    d = b - a
    return float(np.mean(d * d))


# -----------------------------
# Classical energy
# -----------------------------
def compute_classical_energy(xyz, bond_params, angle_params, torsion_params):
    """
    Total classical potential energy of a structure.

    Uses the same functional forms as modules/sa.py:
      bond:    0.5 * k * (r - r0)^2
      angle:   0.5 * k * (theta - theta0)^2
      torsion: k * (1 + cos(phi - delta0))

    Args:
        xyz: (N, 3) coordinates
        bond_params:    (n_bonds, 4)   columns: [i, j, r0, k]
        angle_params:   (n_angles, 5)  columns: [i, j, k, theta0, k_theta]
        torsion_params: (n_torsions, 6) columns: [i, j, k, l, delta0, k_delta]

    Returns:
        total energy (float)
    """
    energy = 0.0

    if bond_params.size > 0:
        for b in range(bond_params.shape[0]):
            i, j = int(bond_params[b, 0]), int(bond_params[b, 1])
            r0 = bond_params[b, 2]
            k = bond_params[b, 3]
            d = xyz[i] - xyz[j]
            r = float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
            energy += 0.5 * k * (r - r0) ** 2

    if angle_params.size > 0:
        for a in range(angle_params.shape[0]):
            i, j, k_idx = int(angle_params[a, 0]), int(angle_params[a, 1]), int(angle_params[a, 2])
            theta0 = angle_params[a, 3]
            k = angle_params[a, 4]
            ba = xyz[i] - xyz[j]
            bc = xyz[k_idx] - xyz[j]
            cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            theta = float(np.arccos(cos_theta))
            energy += 0.5 * k * (theta - theta0) ** 2

    if torsion_params.size > 0:
        for t in range(torsion_params.shape[0]):
            i = int(torsion_params[t, 0])
            j = int(torsion_params[t, 1])
            k = int(torsion_params[t, 2])
            l = int(torsion_params[t, 3])
            delta0 = torsion_params[t, 4]
            k_delta = torsion_params[t, 5]
            b0 = -(xyz[j] - xyz[i])
            b1 = xyz[k] - xyz[j]
            b2 = xyz[l] - xyz[k]
            b1n = b1 / np.linalg.norm(b1)
            v = b0 - np.dot(b0, b1n) * b1n
            w = b2 - np.dot(b2, b1n) * b1n
            x = np.dot(v, w)
            y = np.dot(np.cross(b1n, v), w)
            phi = np.arctan2(y, x)
            energy += k_delta * (1.0 + np.cos(phi - delta0))

    return float(energy)


# -----------------------------
# Pruned candidate loader
# -----------------------------
def load_candidates(
    directory=".",
    dat_ext=".dat",
    xyz_ext=".xyz",
    prune_topM=100,
    prune_delta=None,
    random_sample=None,
    seed=0,
    sample_after_prune: bool = False,
):
    dats = glob.glob(os.path.join(directory, f"*{dat_ext}"))
    xyzs = glob.glob(os.path.join(directory, f"*{xyz_ext}"))

    # Map by (timestep, fit) -> file
    dat_map = {}
    xyz_map = {}
    for p in dats:
        parsed = parse_name(p)
        if parsed:
            dat_map[parsed] = p
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            xyz_map[parsed] = p

    # Keep only keys that have both dat and xyz
    keys = sorted(set(dat_map.keys()) & set(xyz_map.keys()), key=lambda x: (x[0], x[1]))
    if not keys:
        raise RuntimeError(
            "No matched (timestep,fit) pairs found with both .dat and .xyz. "
            "Expected names like 01_0.12345678.dat and 01_0.12345678.xyz"
        )

    # Group by timestep first
    by_t = {}
    for (t, fit) in keys:
        by_t.setdefault(t, []).append(
            {
                "t": t,
                "fit": fit,
                "dat": dat_map[(t, fit)],
                "xyz": xyz_map[(t, fit)],
            }
        )

    timesteps = sorted(by_t.keys())

    # Choose sampling order.
    # - Default (historical): sample BEFORE pruning to reduce work.
    # - Requested mode: prune BEFORE sampling so random subsets are drawn only from the top pool.
    source_by_t = by_t
    if random_sample is not None and not sample_after_prune:
        n_sample = int(random_sample)
        if n_sample < 1:
            raise ValueError(f"random_sample must be >= 1, got {n_sample}")
        rng = np.random.default_rng(seed)
        total_before = sum(len(layer) for layer in by_t.values())
        sampled_by_t = {}
        for t in timesteps:
            layer = by_t[t]
            if len(layer) > n_sample:
                indices = rng.choice(len(layer), size=n_sample, replace=False)
                sampled_by_t[t] = [layer[i] for i in indices]
            else:
                sampled_by_t[t] = list(layer)
        total_after = sum(len(layer) for layer in sampled_by_t.values())
        print(
            f"Randomly sampled up to {n_sample} files per timestep (seed={seed}) BEFORE pruning: "
            f"{total_before} -> {total_after} total files across {len(timesteps)} timesteps"
        )
        source_by_t = sampled_by_t

    # Fit-only pruning per timestep (optionally followed by sampling)
    pruned_layers = []
    for t in timesteps:
        layer = list(source_by_t[t])
        if len(layer) < 1:
            raise RuntimeError(f"Timestep {t} has no candidates.")

        # best fit without sorting whole layer
        best_fit = min(c["fit"] for c in layer)

        if prune_delta is not None:
            delta = float(prune_delta)
            layer = [c for c in layer if c["fit"] <= best_fit + delta]

        # topM without sorting whole layer
        if prune_topM is not None:
            M = int(prune_topM)
            if M < 1:
                raise ValueError(f"topM must be >= 1, got {M}")
            if len(layer) > M:
                layer = heapq.nsmallest(M, layer, key=lambda c: c["fit"])
            else:
                layer = sorted(layer, key=lambda c: c["fit"])

        # Sampling after prune (requested mode)
        if random_sample is not None and sample_after_prune:
            n_sample = int(random_sample)
            if n_sample < 1:
                raise ValueError(f"random_sample must be >= 1, got {n_sample}")
            if len(layer) > n_sample:
                # stable per-timestep, but different per run because compare_random_subsets varies --seed
                rng = np.random.default_rng(seed + int(t))
                indices = rng.choice(len(layer), size=n_sample, replace=False)
                layer = [layer[i] for i in indices]

        if len(layer) < 1:
            raise RuntimeError(f"Timestep {t} has no candidates after pruning.")

        pruned_layers.append(layer)

    return timesteps, pruned_layers


# -----------------------------
# Optional normalization of terms
# -----------------------------
def robust_scale(values, eps=1e-12):
    med = float(np.median(values))
    return 1.0 / max(med, eps)


# -----------------------------
# XYZ trajectory writer
# -----------------------------
def write_xyz_trajectory(path, out_xyz):
    """
    Writes a multi-frame XYZ trajectory from the optimal path.
    Each frame corresponds to one timestep.
    Frame comment line records timestep + fit.
    """
    with open(out_xyz, "w") as f:
        for step in path:
            xyz_path = step["xyz"]
            fit = step["fit"]
            timestep = step["timestep"]

            n, _comment, body = read_xyz_lines(xyz_path)

            f.write(f"{n}\n")
            f.write(f"timestep={timestep:02d} fit={fit:.8f} source={os.path.basename(xyz_path)}\n")
            for line in body:
                # body already includes newline
                f.write(line)


# -----------------------------
# DP solver (Viterbi / shortest path on layered graph)
# -----------------------------
def solve_optimal_path(
    directory=".",
    w_fit=1.0,
    w_sig=1.0,
    w_rmsd=1.0,
    w_energy=0.0,
    reference_xyz=None,
    auto_scale=True,
    prune_topM=100,
    prune_delta=None,
    edge_sample_cap=3000,
    seed=0,
    rmsd_indices=None,
    random_sample=None,
    sample_after_prune: bool = False,
):
    timesteps, layers = load_candidates(
        directory=directory,
        prune_topM=prune_topM,
        prune_delta=prune_delta,
        random_sample=random_sample,
        seed=seed,
        sample_after_prune=sample_after_prune,
    )

    T = len(layers)
    if T < 2:
        raise RuntimeError("Need at least 2 timesteps.")

    use_rmsd = (w_rmsd != 0.0)
    use_energy = (w_energy != 0.0)
    need_coords = use_rmsd or use_energy

    # Extract force field parameters from reference geometry if energy is enabled
    bond_params = angle_params = torsion_params = None
    if use_energy:
        if reference_xyz is None:
            raise ValueError("--reference-xyz is required when energy-weight != 0")
        from modules.openff_retreive_mm_params import Openff_retreive_mm_params
        mm = Openff_retreive_mm_params()
        bond_params, angle_params, torsion_params = mm.extract_params_from_geometry(reference_xyz)
        print(f"Classical energy enabled: {bond_params.shape[0]} bonds, "
              f"{angle_params.shape[0]} angles, {torsion_params.shape[0]} torsions "
              f"(from {reference_xyz})")

    # 1) Load all signals and compute a common q grid (intersection range, min length)
    all_q = []
    for layer in layers:
        for c in layer:
            q, I = read_dat(c["dat"])
            c["q"] = q
            c["I_raw"] = I
            all_q.append(q)

    qmin = max(q[0] for q in all_q)
    qmax = min(q[-1] for q in all_q)
    L = min(len(q) for q in all_q)
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        raise RuntimeError("Failed to build a common q-grid (check your .dat q ranges).")

    q_ref = np.linspace(qmin, qmax, L, dtype=np.float64)

    for layer in layers:
        for c in layer:
            c["I"] = interp_to(c["q"], c["I_raw"], q_ref)
            if need_coords:
                c["X"] = read_xyz_coords(c["xyz"])
            if use_energy:
                c["energy"] = compute_classical_energy(
                    c["X"], bond_params, angle_params, torsion_params
                )

    # 2) Auto-scale terms so weights are comparable
    if auto_scale:
        fit_vals = np.array([c["fit"] for layer in layers for c in layer], dtype=np.float64)

        sig_samp = []
        rmsd_samp = [] if use_rmsd else None
        energy_samp = [] if use_energy else None
        rng = np.random.default_rng(seed)

        for t in range(T - 1):
            A = layers[t]
            B = layers[t + 1]
            Ka, Kb = len(A), len(B)
            n_edges = min(edge_sample_cap, Ka * Kb)

            if n_edges == Ka * Kb:
                pairs = ((i, j) for i in range(Ka) for j in range(Kb))
            else:
                ii = rng.integers(0, Ka, size=n_edges)
                jj = rng.integers(0, Kb, size=n_edges)
                pairs = zip(ii, jj)

            for i, j in pairs:
                sig_samp.append(signal_mse(A[i]["I"], B[j]["I"]))
                if use_rmsd:
                    rmsd_samp.append(kabsch_rmsd(A[i]["X"], B[j]["X"], indices=rmsd_indices))
                if use_energy:
                    energy_samp.append(abs(A[i]["energy"] - B[j]["energy"]))

        s_fit = robust_scale(fit_vals)
        s_sig = robust_scale(np.array(sig_samp, dtype=np.float64))
        w_fit *= s_fit
        w_sig *= s_sig

        if use_rmsd:
            s_rmsd = robust_scale(np.array(rmsd_samp, dtype=np.float64))
            w_rmsd *= s_rmsd

        if use_energy:
            s_energy = robust_scale(np.array(energy_samp, dtype=np.float64))
            w_energy *= s_energy

        parts = [f"w_fit={w_fit:.6g}", f"w_sig={w_sig:.6g}"]
        if use_rmsd:
            parts.append(f"w_rmsd={w_rmsd:.6g}")
        else:
            parts.append("RMSD disabled")
        if use_energy:
            parts.append(f"w_energy={w_energy:.6g}")
        else:
            parts.append("energy disabled")
        print("Auto-scale enabled (weights scaled by 1/median term):")
        print(f"  {', '.join(parts)}")

    # 3) DP
    dp = []
    prev = []

    # init at t=0 with node costs
    K0 = len(layers[0])
    dp0 = np.array([w_fit * layers[0][k]["fit"] for k in range(K0)], dtype=np.float64)
    prev0 = np.full(K0, -1, dtype=np.int32)
    dp.append(dp0)
    prev.append(prev0)

    # transitions
    for t in range(T - 1):
        A = layers[t]
        B = layers[t + 1]
        Ka, Kb = len(A), len(B)

        dp_next = np.full(Kb, np.inf, dtype=np.float64)
        prev_next = np.full(Kb, -1, dtype=np.int32)

        for j in range(Kb):
            node_cost = w_fit * B[j]["fit"]
            BjI = B[j]["I"]
            BjX = B[j]["X"] if use_rmsd else None
            BjE = B[j]["energy"] if use_energy else 0.0

            best = np.inf
            best_i = -1

            for i in range(Ka):
                c_sig = w_sig * signal_mse(A[i]["I"], BjI)
                c_rmsd = w_rmsd * kabsch_rmsd(A[i]["X"], BjX, indices=rmsd_indices) if use_rmsd else 0.0
                c_energy = w_energy * abs(A[i]["energy"] - BjE) if use_energy else 0.0
                cand = dp[t][i] + node_cost + c_sig + c_rmsd + c_energy
                if cand < best:
                    best = cand
                    best_i = i

            dp_next[j] = best
            prev_next[j] = best_i

        dp.append(dp_next)
        prev.append(prev_next)

        print(
            f"Transition {timesteps[t]:02d}->{timesteps[t+1]:02d}: "
            f"Ka={Ka}, Kb={Kb}, best_cost_so_far={dp_next.min():.6g}"
        )

    # 4) Backtrack
    last_k = int(np.argmin(dp[-1]))
    best_cost = float(dp[-1][last_k])

    path_indices = [last_k]
    for t in range(T - 1, 0, -1):
        last_k = int(prev[t][last_k])
        path_indices.append(last_k)
    path_indices.reverse()

    # 5) Build path and report
    path = []
    for t, k in enumerate(path_indices):
        c = layers[t][k]
        path.append(
            {
                "timestep": timesteps[t],
                "fit": c["fit"],
                "dat": c["dat"],
                "xyz": c["xyz"],
            }
        )

    # compute unweighted totals along chosen path (interpretability)
    total_fit = sum(p["fit"] for p in path)
    total_sig = 0.0
    total_rmsd = 0.0
    total_energy_delta = 0.0
    for t in range(T - 1):
        cA = layers[t][path_indices[t]]
        cB = layers[t + 1][path_indices[t + 1]]
        total_sig += signal_mse(cA["I"], cB["I"])
        if use_rmsd:
            total_rmsd += kabsch_rmsd(cA["X"], cB["X"], indices=rmsd_indices)
        if use_energy:
            total_energy_delta += abs(cA["energy"] - cB["energy"])

    print("\n=== Optimal Path ===")
    for item in path:
        print(
            f"t={item['timestep']:02d}  fit={item['fit']:.8f}  "
            f"dat={os.path.basename(item['dat'])}  xyz={os.path.basename(item['xyz'])}"
        )

    print("\n=== Unweighted totals along chosen path ===")
    print(f"sum fit factors   = {total_fit:.8g}")
    print(f"sum signal MSE    = {total_sig:.8g}")
    if use_rmsd:
        print(f"sum RMSD          = {total_rmsd:.8g}")
    else:
        print(f"sum RMSD          = (disabled)")
    if use_energy:
        print(f"sum delta energy  = {total_energy_delta:.8g}")
    else:
        print(f"sum delta energy  = (disabled)")

    print("\n=== Best weighted cost ===")
    print(best_cost)

    return {
        "timesteps": timesteps,
        "path": path,
        "path_indices": path_indices,
        "best_cost": best_cost,
        "weights_used": (w_fit, w_sig, w_rmsd, w_energy),
        "q_ref": q_ref,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find smoothest trajectory through candidate structures (DP shortest path)."
    )
    parser.add_argument(
        "directory",
        help="Directory containing *.dat and *.xyz files",
    )
    parser.add_argument(
        "--topM",
        type=int,
        default=100,
        help="Number of lowest-fit candidates to keep per timestep (fit-only pruning).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Optional fit window: keep candidates with fit <= best_fit + delta (applied before topM cap).",
    )
    parser.add_argument(
        "--fit-weight",
        type=float,
        default=1.0,
        help="Weight for fit factor term (will be auto-scaled unless --no-autoscale).",
    )
    parser.add_argument(
        "--signal-weight",
        type=float,
        default=1.0,
        help="Weight for signal MSE term (auto-scaled unless --no-autoscale).",
    )
    parser.add_argument(
        "--rmsd-weight",
        type=float,
        default=1.0,
        help="Weight for RMSD term (auto-scaled unless --no-autoscale).",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.0,
        help="Weight for classical delta-energy edge cost. 0 disables it (default: 0).",
    )
    parser.add_argument(
        "--reference-xyz",
        type=str,
        default=None,
        help="Reference XYZ file for deriving force field parameters (connectivity, equilibrium "
             "values, generic force constants). Required when --energy-weight != 0.",
    )
    parser.add_argument(
        "--no-autoscale",
        action="store_true",
        help="Disable automatic scaling of cost terms.",
    )
    parser.add_argument(
        "--xyz-out",
        default="optimal_trajectory.xyz",
        help="Output XYZ trajectory filename.",
    )
    parser.add_argument(
        "--edge-sample-cap",
        type=int,
        default=3000,
        help="Number of random edges sampled per timestep transition for auto-scaling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used for auto-scale edge sampling.",
    )
    parser.add_argument(
        "--rmsd-indices",
        type=str,
        default=None,
        help="Comma-separated list of atom indices (0-based) to include in RMSD calculation. "
             "Example: '0,1,2,5' or '3,5,6,10,12'. If not specified, all atoms are used.",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Randomly sample up to N files per timestep before applying topM/delta pruning. "
             "Useful for testing with smaller datasets or exploring different subsets. "
             "Sampling happens after grouping by timestep, so each timestep gets up to N files. "
             "Uses the same seed as --seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-after-prune",
        action="store_true",
        help=(
            "Apply topM/delta pruning first, then randomly sample up to N files per timestep "
            "from the pruned pool. This is useful when you want subsets drawn only from the "
            "top candidates (ignore everything outside topM immediately)."
        ),
    )

    args = parser.parse_args()

    if args.energy_weight != 0.0 and args.reference_xyz is None:
        parser.error("--reference-xyz is required when --energy-weight != 0")
    
    # Parse RMSD indices if provided
    rmsd_indices = None
    if args.rmsd_indices is not None:
        try:
            rmsd_indices = [int(x.strip()) for x in args.rmsd_indices.split(",")]
            rmsd_indices = np.array(rmsd_indices, dtype=np.int32)
            print(f"Using RMSD indices: {rmsd_indices.tolist()}")
        except ValueError as e:
            raise ValueError(
                f"Invalid --rmsd-indices format: '{args.rmsd_indices}'. "
                f"Expected comma-separated integers (e.g., '0,1,2,5'). Error: {e}"
            )

    result = solve_optimal_path(
        directory=args.directory,
        w_fit=args.fit_weight,
        w_sig=args.signal_weight,
        w_rmsd=args.rmsd_weight,
        w_energy=args.energy_weight,
        reference_xyz=args.reference_xyz,
        auto_scale=not args.no_autoscale,
        prune_topM=args.topM,
        prune_delta=args.delta,
        edge_sample_cap=args.edge_sample_cap,
        seed=args.seed,
        rmsd_indices=rmsd_indices,
        random_sample=args.random_sample,
        sample_after_prune=bool(args.sample_after_prune),
    )

    write_xyz_trajectory(result["path"], args.xyz_out)
    print(f"\nWrote XYZ trajectory: {args.xyz_out}")


if __name__ == "__main__":
    main()

