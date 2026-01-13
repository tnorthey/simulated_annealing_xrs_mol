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
# Pruned candidate loader
# -----------------------------
def load_candidates(
    directory=".",
    dat_ext=".dat",
    xyz_ext=".xyz",
    prune_topM=100,
    prune_delta=None,
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

    # Group by timestep
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

    # Fit-only pruning per timestep
    pruned_layers = []
    for t in timesteps:
        layer = sorted(by_t[t], key=lambda c: c["fit"])
        best = layer[0]["fit"]

        if prune_delta is not None:
            delta = float(prune_delta)
            layer = [c for c in layer if c["fit"] <= best + delta]

        if prune_topM is not None:
            layer = layer[: int(prune_topM)]

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
    auto_scale=True,
    prune_topM=100,
    prune_delta=None,
    edge_sample_cap=3000,
    seed=0,
    rmsd_indices=None,
):
    timesteps, layers = load_candidates(
        directory=directory,
        prune_topM=prune_topM,
        prune_delta=prune_delta,
    )

    T = len(layers)
    if T < 2:
        raise RuntimeError("Need at least 2 timesteps.")

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

    # interpolate signals + load xyz coords
    for layer in layers:
        for c in layer:
            c["I"] = interp_to(c["q"], c["I_raw"], q_ref)
            c["X"] = read_xyz_coords(c["xyz"])

    # 2) Auto-scale terms so weights are comparable
    if auto_scale:
        fit_vals = np.array([c["fit"] for layer in layers for c in layer], dtype=np.float64)

        sig_samp = []
        rmsd_samp = []
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
                rmsd_samp.append(kabsch_rmsd(A[i]["X"], B[j]["X"], indices=rmsd_indices))

        s_fit = robust_scale(fit_vals)
        s_sig = robust_scale(np.array(sig_samp, dtype=np.float64))
        s_rmsd = robust_scale(np.array(rmsd_samp, dtype=np.float64))

        w_fit *= s_fit
        w_sig *= s_sig
        w_rmsd *= s_rmsd

        print("Auto-scale enabled (weights scaled by 1/median term):")
        print(f"  w_fit={w_fit:.6g}, w_sig={w_sig:.6g}, w_rmsd={w_rmsd:.6g}")

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
            BjX = B[j]["X"]

            best = np.inf
            best_i = -1

            # brute-force incoming edges
            for i in range(Ka):
                c_sig = w_sig * signal_mse(A[i]["I"], BjI)
                c_rmsd = w_rmsd * kabsch_rmsd(A[i]["X"], BjX, indices=rmsd_indices)
                cand = dp[t][i] + node_cost + c_sig + c_rmsd
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
    for t in range(T - 1):
        cA = layers[t][path_indices[t]]
        cB = layers[t + 1][path_indices[t + 1]]
        total_sig += signal_mse(cA["I"], cB["I"])
        total_rmsd += kabsch_rmsd(cA["X"], cB["X"], indices=rmsd_indices)

    print("\n=== Optimal Path ===")
    for item in path:
        print(
            f"t={item['timestep']:02d}  fit={item['fit']:.8f}  "
            f"dat={os.path.basename(item['dat'])}  xyz={os.path.basename(item['xyz'])}"
        )

    print("\n=== Unweighted totals along chosen path ===")
    print(f"sum fit factors = {total_fit:.8g}")
    print(f"sum signal MSE  = {total_sig:.8g}")
    print(f"sum RMSD        = {total_rmsd:.8g}")

    print("\n=== Best weighted cost ===")
    print(best_cost)

    return {
        "timesteps": timesteps,
        "path": path,
        "path_indices": path_indices,
        "best_cost": best_cost,
        "weights_used": (w_fit, w_sig, w_rmsd),
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

    args = parser.parse_args()
    
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
        auto_scale=not args.no_autoscale,
        prune_topM=args.topM,
        prune_delta=args.delta,
        edge_sample_cap=args.edge_sample_cap,
        seed=args.seed,
        rmsd_indices=rmsd_indices,
    )

    write_xyz_trajectory(result["path"], args.xyz_out)
    print(f"\nWrote XYZ trajectory: {args.xyz_out}")


if __name__ == "__main__":
    main()

