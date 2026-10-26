"""One NPZ archive of the top-M structures at each trajectory timestep.

The file lives in the results directory: ``results_foo/trajectory.npz``.
Arrays:

- ``q``       (qlen,) scattering-vector grid, stored once
- ``t``       (N,) timestep index
- ``f_xray``  (N,) fit value used to rank structures (lowest is best)
- ``xyz``     (N, natoms, 3) coordinates
- ``signal``  (N, qlen) predicted I(q)
- ``atoms``   (natoms,) element symbols
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

import numpy as np

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


_CHAIN_FILE_RE = re.compile(
    r"^(?P<t>\d+)_(?P<fit>\d+\.\d+)(?P<dup>_dup\d+)?\.(?P<ext>xyz|dat)$"
)
_POOL_XYZ_RE = re.compile(
    r"^(?P<t>\d+)_(?P<fit>\d+\.\d+)(?P<dup>_dup\d+)?\.xyz$"
)
_ARCHIVE_KEYS = ("q", "t", "f_xray", "xyz", "signal", "atoms")
ARCHIVE_NAME = "trajectory.npz"


def archive_path(results_dir: str) -> str:
    """``results_foo`` -> ``results_foo/trajectory.npz``."""
    path = os.path.normpath(results_dir)
    if path.lower().endswith(".npz"):
        return path
    return os.path.join(path, ARCHIVE_NAME)


def parse_timestep_id(run_id) -> int:
    """Numeric trajectory run id (``00``, ``01``, ...) as an int."""
    text = str(run_id).strip()
    if not text.isdigit():
        raise ValueError(
            f"npz mode requires a numeric run_id (timestep), got {run_id!r}"
        )
    return int(text)


def format_fit_stub(fit: float) -> str:
    """Fit token used in chain filenames, matching ``wrap.py``."""
    return ("%10.8f" % float(fit)).zfill(12)


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _ARCHIVE_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"Archive {path} is missing array(s): {', '.join(missing)}"
            )
        return {key: np.array(data[key]) for key in _ARCHIVE_KEYS}


def _same_q(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return aa.shape == bb.shape and bool(np.allclose(aa, bb, rtol=1e-5, atol=1e-8))


def _atom_symbols(atoms) -> np.ndarray:
    symbols = [str(a).strip() for a in np.asarray(atoms).reshape(-1)]
    if not symbols or any(not s for s in symbols):
        raise ValueError("atom symbols must be a non-empty list of element names")
    return np.asarray(symbols, dtype="U16")


def _normalize_batch(timestep, f_xray, xyz, signal, q, atoms):
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    atom_arr = _atom_symbols(atoms)
    fit = np.asarray(f_xray, dtype=np.float64).reshape(-1)
    coords = np.asarray(xyz, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    n = int(fit.shape[0])
    if n == 0:
        raise ValueError("no structures to archive")
    if coords.ndim != 3 or coords.shape[0] != n or coords.shape[2] != 3:
        raise ValueError(
            f"xyz must have shape (n, natoms, 3); got {coords.shape} for n={n}"
        )
    if coords.shape[1] != atom_arr.shape[0]:
        raise ValueError(
            f"xyz has {coords.shape[1]} atoms but atoms has length {atom_arr.shape[0]}"
        )
    if sig.shape != (n, q_arr.shape[0]):
        raise ValueError(
            f"signal must have shape (n, qlen)=({n}, {q_arr.shape[0]}); got {sig.shape}"
        )
    if (
        not np.isfinite(fit).all()
        or not np.isfinite(coords).all()
        or not np.isfinite(sig).all()
        or not np.isfinite(q_arr).all()
    ):
        raise ValueError("refusing to archive non-finite q, f_xray, xyz, or signal")
    t = np.full(n, int(timestep), dtype=np.int32)
    return t, fit, coords, sig, q_arr, atom_arr


def _prune_timestep(t, fit, xyz, signal, timestep: int, top_m: int):
    """Drop rows of ``timestep`` outside the ``top_m`` lowest ``fit`` values."""
    idx = np.flatnonzero(np.asarray(t) == int(timestep))
    if idx.size <= int(top_m):
        return t, fit, xyz, signal
    order = idx[np.argsort(fit[idx], kind="mergesort")]
    drop = order[int(top_m) :]
    keep = np.ones(t.shape[0], dtype=bool)
    keep[drop] = False
    return t[keep], fit[keep], xyz[keep], signal[keep]


def _atomic_savez(path: str, **arrays) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=directory)
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def merge_and_prune(
    results_dir: str,
    *,
    timestep: int,
    f_xray,
    xyz,
    signal,
    q,
    atoms,
    top_m: int,
) -> str:
    """Append this timestep's chains and keep only its ``top_m`` lowest fits.

    Other timesteps already in the archive are left unchanged. A different ``q``
    grid or atom list is an error. The archive is replaced atomically.
    """
    top_m = int(top_m)
    if top_m < 1:
        raise ValueError(f"top_m must be >= 1, got {top_m}")
    t_new, fit_new, xyz_new, sig_new, q_new, atoms_new = _normalize_batch(
        timestep, f_xray, xyz, signal, q, atoms
    )
    path = archive_path(results_dir)
    if os.path.isfile(path):
        old = _load(path)
        if not _same_q(old["q"], q_new):
            raise ValueError(
                f"q grid in {path} does not match this run; refusing to merge"
            )
        if _atom_symbols(old["atoms"]).tolist() != atoms_new.tolist():
            raise ValueError(
                f"atom list in {path} does not match this run; refusing to merge"
            )
        t = np.concatenate([np.asarray(old["t"], dtype=np.int32), t_new])
        fit = np.concatenate([np.asarray(old["f_xray"], dtype=np.float64), fit_new])
        xyz_all = np.concatenate(
            [np.asarray(old["xyz"], dtype=np.float64), xyz_new], axis=0
        )
        sig = np.concatenate(
            [np.asarray(old["signal"], dtype=np.float64), sig_new], axis=0
        )
    else:
        t, fit, xyz_all, sig = t_new, fit_new, xyz_new, sig_new
    t, fit, xyz_all, sig = _prune_timestep(t, fit, xyz_all, sig, int(timestep), top_m)
    _atomic_savez(
        path,
        q=q_new,
        t=t,
        f_xray=fit,
        xyz=xyz_all,
        signal=sig,
        atoms=atoms_new,
    )
    return path


def load_timestep_layers(path: str, top_m: int | None = None):
    """Group archive rows by timestep, lowest ``f_xray`` first.

    Each candidate dict has ``t``, ``fit``, ``coords``, ``atoms``, ``signal``,
    ``comment``, and ``xyz=None`` (there is no per-chain file).
    """
    data = _load(path)
    t = np.asarray(data["t"], dtype=np.int64).reshape(-1)
    fit = np.asarray(data["f_xray"], dtype=np.float64).reshape(-1)
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    signal = np.asarray(data["signal"], dtype=np.float64)
    atoms = [str(a) for a in np.asarray(data["atoms"]).reshape(-1)]
    if t.size == 0:
        raise RuntimeError(f"No structures in archive {path}")
    by_t: dict[int, list[int]] = {}
    for i, ti in enumerate(t.tolist()):
        by_t.setdefault(int(ti), []).append(i)
    timesteps = sorted(by_t)
    layers = []
    for ti in timesteps:
        idxs = sorted(by_t[ti], key=lambda i: float(fit[i]))
        if top_m is not None and len(idxs) > int(top_m):
            idxs = idxs[: int(top_m)]
        layer = []
        for i in idxs:
            layer.append(
                {
                    "t": ti,
                    "fit": float(fit[i]),
                    "xyz": None,
                    "coords": np.array(xyz[i], dtype=np.float64, copy=True),
                    "atoms": list(atoms),
                    "signal": np.array(signal[i], dtype=np.float64, copy=True),
                    "comment": f"{float(fit[i]):.8f}",
                }
            )
        layers.append(layer)
    return timesteps, layers


def _step_label(step) -> tuple[int, str]:
    text = str(step).strip()
    if not text.isdigit():
        raise ValueError(f"timestep label must be numeric, got {step!r}")
    width = max(len(text), 2)
    number = int(text)
    return number, f"{number:0{width}d}"


def _write_xyz(path: str, atoms, coords, comment: str) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(f"{comment}\n")
        for i, atom in enumerate(atoms):
            x, y, z = coords[i]
            handle.write(f"{atom} {x:.16e} {y:.16e} {z:.16e}\n")


def materialize_timestep(
    results_dir: str,
    step,
    dest_dir: str,
    top_n: int | None = None,
    *,
    warn: bool = True,
) -> list[str]:
    """Write one timestep's kept frames as ``{step}_{fit}.xyz`` under ``dest_dir``.

    Paths are returned lowest ``f_xray`` first. Missing archive or timestep yields
    an empty list.
    """
    timestep, prefix = _step_label(step)
    path = archive_path(results_dir)
    if not os.path.isfile(path):
        if warn:
            print(f"  WARNING: no archive {path}", file=sys.stderr)
        return []
    _timesteps, layers = load_timestep_layers(path, top_m=top_n)
    layer = None
    for candidate_t, candidate_layer in zip(_timesteps, layers):
        if candidate_t == timestep:
            layer = candidate_layer
            break
    if not layer:
        if warn:
            print(
                f"  WARNING: no pool for step {prefix} in {path}",
                file=sys.stderr,
            )
        return []
    os.makedirs(dest_dir, exist_ok=True)
    prefix_re = re.compile(
        rf"^{re.escape(prefix)}_\d+\.\d+(_dup\d+)?\.xyz$"
    )
    for name in os.listdir(dest_dir):
        if prefix_re.match(name):
            os.remove(os.path.join(dest_dir, name))
    filename_counts: dict[str, int] = {}
    written: list[str] = []
    for cand in layer:
        fit_stub = format_fit_stub(cand["fit"])
        base_stub = f"{prefix}_{fit_stub}"
        count = filename_counts.get(base_stub, 0)
        filename_counts[base_stub] = count + 1
        file_stub = base_stub if count == 0 else f"{base_stub}_dup{count}"
        out = os.path.join(dest_dir, file_stub + ".xyz")
        _write_xyz(out, cand["atoms"], cand["coords"], cand["comment"])
        written.append(out)
    if warn:
        print(
            f"  pooled {len(written)} structures from archive step {prefix} "
            f"({path}, lowest f_xray first)",
            file=sys.stderr,
        )
    return written


def materialize_tradius(
    results_dir: str,
    timestep,
    radius: int,
    top_n: int,
    dest_dir: str,
) -> list[str]:
    """Union of the top ``top_n`` frames at ``t+/-1 .. t+/-radius`` (no global sort)."""
    t_now, _prefix = _step_label(timestep)
    radius = int(radius)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    width = max(len(str(timestep).strip()), 2)
    union: list[str] = []
    seen: set[str] = set()
    used_steps: list[str] = []
    skipped_steps: list[str] = []
    for k in range(1, radius + 1):
        for step_num in (t_now - k, t_now + k):
            if step_num < 0:
                continue
            label = f"{step_num:0{width}d}"
            paths = materialize_timestep(
                results_dir, label, dest_dir, top_n, warn=False
            )
            if not paths:
                skipped_steps.append(label)
                print(
                    f"  WARNING: no pool for step {label} "
                    f"(t{'+' if step_num > t_now else '-'}{k}), skipping",
                    file=sys.stderr,
                )
                continue
            used_steps.append(label)
            print(
                f"  pooled {len(paths)} from step {label} (archive, lowest f_xray first)",
                file=sys.stderr,
            )
            for item in paths:
                if item in seen:
                    continue
                seen.add(item)
                union.append(item)
    if union:
        skipped_msg = (
            f" (skipped: {', '.join(skipped_steps)})" if skipped_steps else ""
        )
        print(
            f"  tradius N={radius}: {len(union)} unique structures in union from "
            f"steps {', '.join(used_steps)}{skipped_msg}",
            file=sys.stderr,
        )
    return union


def iter_chain_files(results_dir: str, timestep: int | None = None):
    """Top-level chain ``{t}_{fit}[_dupN].xyz/.dat`` files, optionally one timestep."""
    if not os.path.isdir(results_dir):
        return
    for name in sorted(os.listdir(results_dir)):
        match = _CHAIN_FILE_RE.match(name)
        if not match:
            continue
        if timestep is not None and int(match.group("t")) != int(timestep):
            continue
        yield os.path.join(results_dir, name)


def delete_chain_files(results_dir: str, timestep: int) -> int:
    """Remove chain xyz/dat for ``timestep``. Leave mean, target, and start files."""
    removed = 0
    for path in list(iter_chain_files(results_dir, timestep)):
        os.remove(path)
        removed += 1
    return removed


def count_chain_xyz(results_dir: str) -> int:
    """Number of top-level per-chain xyz files (not mean/target/start)."""
    if not os.path.isdir(results_dir):
        return 0
    n = 0
    for name in os.listdir(results_dir):
        if _POOL_XYZ_RE.match(name):
            n += 1
    return n


def _read_config(config_path: str) -> dict:
    if tomllib is None:
        raise RuntimeError("TOML support is missing (tomllib or tomli)")
    with open(config_path, "rb") as handle:
        return tomllib.load(handle)


def npz_mode_enabled(config_path: str | None, results_dir: str) -> bool:
    """True when the config flag is set or the sibling archive already exists."""
    enabled = False
    if config_path and os.path.isfile(config_path):
        try:
            data = _read_config(config_path)
            enabled = bool(data.get("options", {}).get("npz_mode_bool", False))
        except Exception:
            enabled = False
    return enabled or os.path.isfile(archive_path(results_dir))


def configured_archive_top_m(config_path: str) -> int | None:
    """``archive_top_m`` when NPZ mode is on in the config, else None."""
    if not config_path or not os.path.isfile(config_path):
        return None
    data = _read_config(config_path)
    if not bool(data.get("options", {}).get("npz_mode_bool", False)):
        return None
    return int(data.get("options", {}).get("archive_top_m", 200))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trajectory_archive")
    sub = parser.add_subparsers(dest="cmd", required=True)

    mode = sub.add_parser("npz-mode", help="Print 1 if NPZ mode is active")
    mode.add_argument("--config", required=True)
    mode.add_argument("--results-dir", required=True)

    cap = sub.add_parser(
        "archive-top-m",
        help="Print archive_top_m when NPZ mode is on, else nothing",
    )
    cap.add_argument("--config", required=True)

    mat = sub.add_parser("materialize", help="Write one timestep's pool xyz files")
    mat.add_argument("--results-dir", required=True)
    mat.add_argument("--step", required=True)
    mat.add_argument("--top-n", type=int, required=True)
    mat.add_argument("--dest", required=True)

    rad = sub.add_parser(
        "materialize-tradius",
        help="Write the t+/-1..t+/-N pool xyz files",
    )
    rad.add_argument("--results-dir", required=True)
    rad.add_argument("--timestep", required=True)
    rad.add_argument("--radius", type=int, required=True)
    rad.add_argument("--top-n", type=int, required=True)
    rad.add_argument("--dest", required=True)

    count = sub.add_parser("count-chains", help="Print the number of chain xyz files")
    count.add_argument("--results-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "npz-mode":
        print("1" if npz_mode_enabled(args.config, args.results_dir) else "0")
        return 0
    if args.cmd == "archive-top-m":
        top_m = configured_archive_top_m(args.config)
        if top_m is not None:
            print(top_m)
        return 0
    if args.cmd == "materialize":
        paths = materialize_timestep(
            args.results_dir, args.step, args.dest, args.top_n
        )
        for path in paths:
            print(path)
        return 0
    if args.cmd == "materialize-tradius":
        paths = materialize_tradius(
            args.results_dir,
            args.timestep,
            args.radius,
            args.top_n,
            args.dest,
        )
        for path in paths:
            print(path)
        return 0
    if args.cmd == "count-chains":
        print(count_chain_xyz(args.results_dir))
        return 0
    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
