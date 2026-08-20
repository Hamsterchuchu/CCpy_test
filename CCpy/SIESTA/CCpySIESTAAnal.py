#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCpySIESTAAnal.py

Mixes the CLI style of CCpyVASPAnal.py (option-number based, sub-options like -sub, usage output) with
the analysis logic of siesta_analyze.py (energy/force convergence, ARC generation, TIMES scan, etc.).

- Option 1 : extract only the final structure (last MD_CAR frame -> <dirname>_final.car, collected in the parent folder)
- Option 3 : full analysis of a single current directory (same as the main() logic of siesta_analyze.py)
- Options 0, 2 : select several SIESTA job directories and batch process (style of options 0, 2 in CCpyVASPAnal.py)
- -time : scan TIMES files (the --time logic of siesta_analyze.py)
- -d : clean up outputs (input files *.fdf/*.psf/*.vps/*.ion are kept)

Required packages: numpy, pandas, matplotlib
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os, sys
import re
import csv
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import OrderedDict

# On a server without X, the Agg backend is safe (CCpyVASPAnal.py style)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

version = sys.version
if version[0] == '3':
    raw_input = input


# ============================================================
# 0. Find/select SIESTA job directories (SIESTA helper replacing
#    selectVASPOutputs / selectInputs of CCpyVASPAnal.py)
# ============================================================

def find_siesta_dirs(root=".", sub=False):
    """Find directories that contain a *.fdf file and treat them as SIESTA job directories."""
    root = Path(root)
    dirs: List[Path] = []

    if sub:
        for p in root.rglob("*.fdf"):
            d = p.parent
            if d not in dirs:
                dirs.append(d)
    else:
        if list(root.glob("*.fdf")):
            dirs.append(root)
        for p in sorted(root.iterdir()):
            if p.is_dir() and list(p.glob("*.fdf")):
                dirs.append(p)

    uniq = sorted(set(dirs), key=lambda x: str(x))
    return uniq


def selectSiestaOutputs(root=".", ask=True, sub=False, dir_list=None):
    """
    Mimics the usage pattern of selectVASPOutputs(...) in CCpyVASPAnal.py.
    If dir_list is given it is used as is, otherwise SIESTA job directories
    (*.fdf present) are searched under root and confirmed/selected by the user when ask=True.
    """
    if dir_list is not None:
        return [str(d) for d in dir_list]

    found = [str(d) for d in find_siesta_dirs(root, sub=sub)]

    if not found:
        print("No SIESTA job directory (*.fdf) found.")
        return []

    if not ask:
        return found

    print("\nFound SIESTA job directories:")
    for i, d in enumerate(found):
        print(f"  [{i}] {d}")
    yn = raw_input("\nUse all of these directories? (y/n) ")
    if yn.lower() in ("y", "yes"):
        return found

    sel = raw_input("Enter indices to use (comma separated, e.g. 0,2,3): ")
    idxs = [int(x.strip()) for x in sel.split(",") if x.strip().isdigit()]
    return [found[i] for i in idxs if 0 <= i < len(found)]


def parse_index_selection(sel: str, n: int) -> List[int]:
    """'1-3,5' -> [1,2,3,5] (1-based). '0' -> all indices 1..n. (same syntax as CCpySIESTABandSubmit.py)"""
    sel = sel.strip()
    if sel == "0":
        return list(range(1, n + 1))
    idxs: List[int] = []
    for tok in sel.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            idxs.extend(range(int(a), int(b) + 1))
        else:
            idxs.append(int(tok))
    return idxs


def select_siesta_dirs(dirs: List[Path], ask: bool = True, preselect: Optional[str] = None) -> List[Path]:
    """
    Same UX as select_systems(...) in CCpySIESTABandSubmit.py:
        1 : Ti2CTx_O_NT_2
        2 : Ti2CTx_O_NT_Bandtest
        ...
        0 : All directories
        choose file :
    Supports comma/range (1-3,5) input, and asks again on invalid input.
    If preselect is given (-systems=... sub-option), selects right away without a prompt.
    If ask=False, selects all of them (no prompt).
    """
    if not dirs:
        print("No SIESTA job directory (*.fdf) found.")
        quit()

    if preselect is not None:
        idxs = parse_index_selection(preselect, len(dirs))
        return [dirs[i - 1] for i in idxs]

    if not ask:
        return dirs

    for i, d in enumerate(dirs):
        print(f"{i + 1} : {d}")
    print("0 : All directories")

    while True:
        sel = raw_input("choose file : ").strip()
        if not sel:
            continue
        try:
            idxs = parse_index_selection(sel, len(dirs))
            return [dirs[i - 1] for i in idxs]
        except (ValueError, IndexError) as exc:
            print(f"  invalid selection ({exc}), try again")


def _get_kv_arg(name, default=None, cast=str):
    """Read a sub-option value of the form --name=value or -name=value from sys.argv."""
    prefixes = (f"--{name}=", f"-{name}=")
    for arg in sys.argv:
        for prefix in prefixes:
            if arg.startswith(prefix):
                val = arg[len(prefix):]
                try:
                    return cast(val)
                except Exception:
                    return default
    return default


# ============================================================
# 1. Analysis engine of siesta_analyze.py (ported almost as is)
# ============================================================

_TOT_LINE_RE = re.compile(r"^\s*Tot\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$")
_MAX_LINE_RE = re.compile(r"^\s*Max\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*(?:\S.*)?$")
_RES_LINE_RE = re.compile(r"^\s*Res\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*(?:\S.*)?$")

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")


def _extract_floats(line: str) -> List[float]:
    out: List[float] = []
    for m in _FLOAT_RE.findall(line):
        try:
            out.append(float(m))
        except Exception:
            pass
    return out


def _extract_last_float(line: str) -> Optional[float]:
    vals = _extract_floats(line)
    return None if not vals else vals[-1]


def infer_basename_from_fdf() -> str:
    """Infer the basename from the current directory (or the parent/restart chain for Restart/RestartN)."""
    search_dirs: List[Path] = [Path(".")]

    cwd = Path.cwd()
    if cwd.name == "Restart" or re.fullmatch(r"Restart(\d+)", cwd.name):
        search_dirs.append(cwd.parent)
        for d in collect_restart_dirs():
            if d not in search_dirs:
                search_dirs.append(d)

    seen = set()
    for d in search_dirs:
        rp = d.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        fdfs = sorted(d.glob("*.fdf"))
        if fdfs:
            return fdfs[0].stem

    raise SystemExit("No *.fdf found in current directory or restart chain. Provide --base=BASENAME explicitly.")


def _sec_to_hms(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec - 3600 * h) // 60)
    s = sec - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def collect_restart_dirs() -> List[Path]:
    """List of directories to merge for the Normal / Restart / RestartN layout."""
    cwd = Path.cwd()
    parent = cwd.parent

    candidates: List[Path]
    if cwd.name == "Restart":
        candidates = [parent, cwd]
    else:
        m = re.fullmatch(r"Restart(\d+)", cwd.name)
        if not m:
            return [cwd]

        current_idx = int(m.group(1))
        candidates = [parent]

        unnamed_restart = parent / "Restart"
        if unnamed_restart.exists() and unnamed_restart.is_dir():
            candidates.append(unnamed_restart)

        for i in range(1, current_idx + 1):
            d = parent / f"Restart{i}"
            if d.exists() and d.is_dir():
                candidates.append(d)

    dirs: List[Path] = []
    seen = set()
    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        rp = d.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        dirs.append(d)
    return dirs


def collect_existing_files(dirs: List[Path], filename: str, restart_filename: Optional[str] = None) -> List[Path]:
    """Collect the files that exist across the Normal/Restart directories, in order."""
    files: List[Path] = []
    for d in dirs:
        is_restart_dir = (d.name == 'Restart') or re.fullmatch(r'Restart(\d+)', d.name) is not None
        candidates: List[Path] = []
        if is_restart_dir and restart_filename:
            candidates.append(d / restart_filename)
        candidates.append(d / filename)

        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is not None:
            files.append(chosen)
    return files


# ----------------------------
# FDF parsing (atom order + lattice)
# ----------------------------
@dataclass
class FDFInfo:
    natom: int
    symbols: List[str]
    pbc_line: Optional[str]
    pbc_on: bool


def _read_fdf_blocks(lines: List[str]) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.lower().startswith("%block"):
            parts = ln.split()
            if len(parts) >= 2:
                name = parts[1]
                i += 1
                content: List[str] = []
                while i < len(lines):
                    if lines[i].strip().lower().startswith("%endblock"):
                        break
                    content.append(lines[i].rstrip("\n"))
                    i += 1
                blocks[name] = content
        i += 1
    return blocks


def parse_fdf(base: str) -> FDFInfo:
    fdf_path = Path(f"{base}.fdf")
    if not fdf_path.exists():
        for d in collect_restart_dirs():
            cand = d / f"{base}.fdf"
            if cand.exists():
                fdf_path = cand
                break
        else:
            raise RuntimeError(f"Missing {base}.fdf (needed for ARC atom symbols and lattice)")

    lines = fdf_path.read_text(errors="ignore").splitlines()
    blocks = _read_fdf_blocks(lines)

    species_map: Dict[int, str] = {}

    for key, content in blocks.items():
        norm = re.sub(r"[^a-z]", "", key.strip().lower())
        if norm == "chemicalspecieslabel":
            for ln in content:
                toks = ln.split()
                if len(toks) >= 3:
                    try:
                        idx_sp = int(float(toks[0]))
                        sym = toks[2]
                        species_map[idx_sp] = sym
                    except Exception:
                        pass

    if not species_map:
        for ln in lines:
            if ln.strip().lower().startswith("chemicalspecieslabel"):
                toks = ln.split()
                if len(toks) >= 4:
                    try:
                        idx_sp = int(float(toks[1]))
                        sym = toks[3]
                        species_map[idx_sp] = sym
                    except Exception:
                        pass

    symbols: List[str] = []
    for key, content in blocks.items():
        if key.lower() == "atomiccoordinatesandatomicspecies":
            for ln in content:
                toks = ln.split()
                if len(toks) >= 4:
                    try:
                        sp = int(float(toks[3]))
                        symbols.append(species_map.get(sp, "X"))
                    except Exception:
                        pass

    natom = len(symbols)
    if natom == 0:
        raise RuntimeError("Could not parse AtomicCoordinatesAndAtomicSpecies from FDF (needed for ARC).")

    pbc_line = None
    pbc_on = False

    lat_params = None
    for ln in lines:
        if ln.strip().lower().startswith("latticeparameters"):
            vals = _extract_floats(ln)
            if len(vals) >= 6:
                lat_params = vals[:6]
                break
    if lat_params is not None:
        a, b, c, alpha, beta, gamma = lat_params
        pbc_line = f"PBC   {a:10.4f}   {b:10.4f}   {c:10.4f}   {alpha:8.4f}   {beta:8.4f}   {gamma:8.4f}"
        pbc_on = True
    else:
        vecs = None
        for key, content in blocks.items():
            if key.lower() == "latticevectors":
                raw: List[List[float]] = []
                for ln in content:
                    vals = _extract_floats(ln)
                    if len(vals) >= 3:
                        raw.append(vals[:3])
                if len(raw) == 3:
                    vecs = np.array(raw, float)
                    break

        lat_const = None
        for ln in lines:
            if ln.strip().lower().startswith("latticeconstant"):
                vals = _extract_floats(ln)
                if vals:
                    lat_const = vals[0]
                    break

        if vecs is not None:
            if lat_const is None:
                lat_const = 1.0
            v = vecs * float(lat_const)
            a = float(np.linalg.norm(v[0]))
            b = float(np.linalg.norm(v[1]))
            c = float(np.linalg.norm(v[2]))

            def ang(u, w):
                cuw = float(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)))
                cuw = max(-1.0, min(1.0, cuw))
                return float(np.degrees(np.arccos(cuw)))

            alpha = ang(v[1], v[2])
            beta = ang(v[0], v[2])
            gamma = ang(v[0], v[1])
            pbc_line = f"PBC   {a:10.4f}   {b:10.4f}   {c:10.4f}   {alpha:8.4f}   {beta:8.4f}   {gamma:8.4f}"
            pbc_on = True

    return FDFInfo(natom=natom, symbols=symbols, pbc_line=pbc_line, pbc_on=pbc_on)


def get_natoms_from_fdf_file(fdf_path) -> Optional[int]:
    """Light atom count estimate used in option 2 (batch energy list) (works from the file path only, without changing cwd)."""
    fdf_path = Path(fdf_path)
    if not fdf_path.exists():
        return None
    try:
        lines = fdf_path.read_text(errors="ignore").splitlines()
        blocks = _read_fdf_blocks(lines)
        n = 0
        for key, content in blocks.items():
            if key.lower() == "atomiccoordinatesandatomicspecies":
                for ln in content:
                    if ln.split():
                        n += 1
        return n if n > 0 else None
    except Exception:
        return None


# ----------------------------
# MDE
# ----------------------------
@dataclass
class MDEData:
    energy_col4: np.ndarray
    energy_opt: np.ndarray


def read_mde_with_auto_skip(path: Path) -> MDEData:
    last_err = None
    for skip in (0, 1, 2, 3, 4, 5):
        try:
            arr = np.loadtxt(path, skiprows=skip)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 4:
                raise ValueError("MDE has <4 columns")
            e4 = arr[:, 3]
            eopt = e4[2:] if len(e4) > 2 else e4.copy()
            return MDEData(energy_col4=e4, energy_opt=eopt)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read MDE: {path} ({last_err})")


def read_combined_mde(mde_files: List[Path]) -> MDEData:
    if not mde_files:
        raise RuntimeError("No MDE files found to merge.")

    energy_col4_parts: List[np.ndarray] = []
    for i, path in enumerate(mde_files):
        part = read_mde_with_auto_skip(path)
        e4 = np.asarray(part.energy_col4, dtype=float)
        if i > 0 and e4.size > 2:
            e4 = e4[2:]
        energy_col4_parts.append(e4)

    energy_col4 = np.concatenate(energy_col4_parts) if energy_col4_parts else np.array([], dtype=float)
    energy_opt = energy_col4[2:] if len(energy_col4) > 2 else energy_col4.copy()
    return MDEData(energy_col4=energy_col4, energy_opt=energy_opt)


# ----------------------------
# OUT forces (vector Tot + Max)
# ----------------------------
@dataclass
class ForceSeries:
    steps: List[int]
    fx: List[Optional[float]]
    fy: List[Optional[float]]
    fz: List[Optional[float]]
    f_total: List[Optional[float]]
    max_force: List[Optional[float]]
    res_force: List[Optional[float]]


def _is_max_line(ln: str) -> bool:
    s = ln.rstrip("\n")
    if "constrained" in s.lower():
        return False
    return _MAX_LINE_RE.match(s) is not None


def _is_res_line(ln: str) -> bool:
    return _RES_LINE_RE.match(ln.rstrip("\n")) is not None


def _is_tot_line(ln: str) -> bool:
    return _TOT_LINE_RE.match(ln.rstrip("\n")) is not None


def parse_out_forces(out_path: Path) -> Tuple[List[Tuple[Optional[float], Optional[float], Optional[float]]],
                                              List[Optional[float]],
                                              List[Optional[float]]]:
    """Parse Tot (total force vector)/Max/Res (RMS force) values from a SIESTA .out file."""
    lines = out_path.read_text(errors="ignore").splitlines()

    tot_vecs: List[Tuple[Optional[float], Optional[float], Optional[float]]] = []
    max_vals: List[Optional[float]] = []
    res_vals: List[Optional[float]] = []

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if _is_tot_line(ln):
            m_tot = _TOT_LINE_RE.match(ln.rstrip("\n"))
            if m_tot:
                fx, fy, fz = float(m_tot.group(1)), float(m_tot.group(2)), float(m_tot.group(3))
                tot_vecs.append((fx, fy, fz))
            else:
                tot_vecs.append((None, None, None))

            blk_max: Optional[float] = None
            blk_res: Optional[float] = None
            j = i + 1
            while j < n and not _is_tot_line(lines[j]):
                s = lines[j].rstrip("\n")
                if blk_max is None:
                    m_max = _MAX_LINE_RE.match(s)
                    if m_max and ("constrained" not in s.lower()):
                        try:
                            blk_max = float(m_max.group(1))
                        except ValueError:
                            blk_max = None
                if blk_res is None:
                    m_res = _RES_LINE_RE.match(s)
                    if m_res:
                        try:
                            blk_res = float(m_res.group(1))
                        except ValueError:
                            blk_res = None
                if blk_max is not None and blk_res is not None:
                    break
                j += 1

            max_vals.append(blk_max)
            res_vals.append(blk_res)

            i = j
            continue

        i += 1

    return tot_vecs, max_vals, res_vals


def build_force_series_from_arrays(
    tot_vecs: List[Tuple[Optional[float], Optional[float], Optional[float]]],
    max_vals: List[Optional[float]],
    res_vals: List[Optional[float]],
    nopt: int = 0,
) -> ForceSeries:
    if nopt and nopt > 0:
        if len(tot_vecs) > nopt:
            tot_vecs = tot_vecs[-nopt:]
        if len(max_vals) > nopt:
            max_vals = max_vals[-nopt:]
        if len(res_vals) > nopt:
            res_vals = res_vals[-nopt:]

    n = len(tot_vecs)
    if len(max_vals) < n:
        max_vals = list(max_vals) + [None] * (n - len(max_vals))
    if len(res_vals) < n:
        res_vals = list(res_vals) + [None] * (n - len(res_vals))

    fx = [v[0] for v in tot_vecs]
    fy = [v[1] for v in tot_vecs]
    fz = [v[2] for v in tot_vecs]

    def _mag(a, b, c):
        if a is None or b is None or c is None:
            return None
        return float(np.sqrt(float(a) ** 2 + float(b) ** 2 + float(c) ** 2))

    ft = [_mag(a, b, c) for a, b, c in zip(fx, fy, fz)]
    steps = list(range(1, n + 1))
    return ForceSeries(
        steps=steps,
        fx=fx,
        fy=fy,
        fz=fz,
        f_total=ft,
        max_force=max_vals[:n],
        res_force=res_vals[:n],
    )


def parse_combined_out_forces(out_files: List[Path]) -> Tuple[List[Tuple[Optional[float], Optional[float], Optional[float]]],
                                                               List[Optional[float]],
                                                               List[Optional[float]]]:
    tot_vecs: List[Tuple[Optional[float], Optional[float], Optional[float]]] = []
    max_vals: List[Optional[float]] = []
    res_vals: List[Optional[float]] = []

    for path in out_files:
        t, m, r = parse_out_forces(path)
        tot_vecs.extend(t)
        max_vals.extend(m)
        res_vals.extend(r)

    return tot_vecs, max_vals, res_vals


def build_force_series(out_path: Path, nopt: int = 0) -> ForceSeries:
    tot_vecs, max_vals, res_vals = parse_out_forces(out_path)
    return build_force_series_from_arrays(tot_vecs, max_vals, res_vals, nopt=nopt)


# ----------------------------
# Plotting
# ----------------------------
def plot_energy(energy_opt: np.ndarray, out_png: Path, xstart: int = 1):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if xstart < 1:
        xstart = 1
    if xstart > len(energy_opt):
        xstart = len(energy_opt) if len(energy_opt) > 0 else 1

    energy_plot = energy_opt[xstart - 1:]
    x = np.arange(xstart, len(energy_opt) + 1)

    plt.figure()
    plt.plot(x, energy_plot, marker="o", markersize=2.5, linewidth=1)
    plt.xlabel("Optimization step")
    plt.ylabel("Energy (eV)")
    plt.title("Energy convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _robust_ylim(vals: np.ndarray) -> Optional[float]:
    v = vals[np.isfinite(vals)]
    v = v[v >= 0]
    if v.size < 5:
        return None
    vmax = float(np.max(v))
    p98 = float(np.percentile(v, 98))
    if p98 > 0 and vmax > 5.0 * p98:
        return p98 * 1.2
    return None


def plot_forces(series: ForceSeries, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    s = np.array(series.steps, dtype=int)

    fx = np.array([np.nan if v is None else float(v) for v in series.fx], dtype=float)
    fy = np.array([np.nan if v is None else float(v) for v in series.fy], dtype=float)
    fz = np.array([np.nan if v is None else float(v) for v in series.fz], dtype=float)
    ft = np.array([np.nan if v is None else float(v) for v in series.f_total], dtype=float)

    def _robust_ylim_abs(vals: np.ndarray) -> Optional[float]:
        v = np.abs(vals[np.isfinite(vals)])
        if v.size < 5:
            return None
        vmax = float(np.max(v))
        p98 = float(np.percentile(v, 98))
        if p98 > 0 and vmax > 5.0 * p98:
            return p98 * 1.2
        return None

    def _plot(title: str, outpath: Path, ylim: Optional[Tuple[float, float]]):
        plt.figure()
        plotted = False
        if np.isfinite(fx).any():
            plt.plot(s, fx, marker="o", label="Fx (eV/Å)", markersize=2.5, linewidth=1)
            plotted = True
        if np.isfinite(fy).any():
            plt.plot(s, fy, marker="o", label="Fy (eV/Å)", markersize=2.5, linewidth=1)
            plotted = True
        if np.isfinite(fz).any():
            plt.plot(s, fz, marker="o", label="Fz (eV/Å)", markersize=2.5, linewidth=1)
            plotted = True
        if np.isfinite(ft).any():
            plt.plot(s, ft, marker="o", label="|F_tot| (eV/Å)", markersize=2.5, linewidth=1)
            plotted = True

        plt.xlabel("Optimization step")
        plt.ylabel("Force (eV/Å)")
        plt.title(title)
        plt.grid(True)
        if plotted:
            plt.legend()
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

    y_candidates = [
        _robust_ylim_abs(fx),
        _robust_ylim_abs(fy),
        _robust_ylim_abs(fz),
        _robust_ylim_abs(ft),
    ]
    y_candidates = [v for v in y_candidates if v is not None and v > 0]
    if y_candidates:
        y = max(y_candidates)
        _plot("Forces (clipped for readability)", out_png, ylim=(-y, y))
    else:
        _plot("Forces", out_png, ylim=None)


def plot_force_convergence(fs: ForceSeries, out_png: Path):
    """Res (RMS force) vs step plot."""
    x = np.array(fs.steps, dtype=float)
    y = np.array([np.nan if v is None else float(v) for v in fs.res_force], dtype=float)

    if len(x) == 0:
        return

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", markersize=2.5, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Res (eV/Å)")
    plt.title("Force convergence (Res vs step)")
    plt.grid(True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# ----------------------------
# ARC (03_GenArc.sh + pos2car.py logic)
# ----------------------------
def _find_mdcar_markers(lines: List[str], marker: str) -> List[int]:
    return [i for i, ln in enumerate(lines) if marker in ln]


def _poscar_type_and_counts(base: str, fdf: FDFInfo) -> Tuple[str, str]:
    lf = Path("list")
    if lf.exists():
        rows = [ln.strip() for ln in lf.read_text(errors="ignore").splitlines() if ln.strip()]
        if len(rows) >= 2:
            return rows[0], rows[1]

    order: List[str] = []
    counts: Dict[str, int] = {}
    for sym in fdf.symbols:
        if sym not in counts:
            order.append(sym)
            counts[sym] = 0
        counts[sym] += 1
    type_line = " ".join(order)
    count_line = " ".join(str(counts[s]) for s in order)
    return type_line, count_line


def _run_pos2car(pos_text: str) -> List[str]:
    import subprocess

    candidates = [
        Path("pos2car.py"),
        Path(__file__).resolve().with_name("pos2car.py"),
        Path("/home/work/bin/pos2car.py"),
    ]
    pos2car = next((p for p in candidates if p.exists()), None)
    if pos2car is None:
        raise RuntimeError("pos2car.py not found (need it to generate ARC). Place it next to this script or in CWD.")

    cmd = [sys.executable, str(pos2car)]
    p = subprocess.run(cmd, input=pos_text, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"pos2car.py failed: {p.stderr.strip()}")
    return p.stdout.splitlines()


def _format_arc_coord_line(sym: str, per_symbol_idx: int, atom_idx: int,
                           x: float, y: float, z: float,
                           charge: float = 0.0, residue: str = "XXXX") -> str:
    atom_name = f"{sym}{per_symbol_idx}"
    return (
        f"{atom_name:<5}"
        f"{x:15.9f}{y:15.9f}{z:15.9f} "
        f"{residue:<5} "
        f"{atom_idx:<6d}"
        f"{sym:<8}"
        f"{sym:<4}"
        f"{charge:5.3f}"
    )


def _parse_car_coord_line(line: str) -> Optional[Tuple[float, float, float, str, float]]:
    toks = line.split()
    if len(toks) < 4:
        return None

    head = toks[0].lower()
    if head.startswith(("!biosym", "!date", "pbc", "end", "materials")):
        return None

    try:
        x = float(toks[1])
        y = float(toks[2])
        z = float(toks[3])
    except Exception:
        return None

    residue = toks[4] if len(toks) >= 5 else "XXXX"

    charge = 0.0
    if len(toks) >= 9:
        try:
            charge = float(toks[8])
        except Exception:
            charge = 0.0
    elif len(toks) >= 1:
        try:
            charge = float(toks[-1])
        except Exception:
            charge = 0.0

    return x, y, z, residue, charge


def _apply_fdf_symbols_to_car(car_lines: List[str], symbols: List[str]) -> List[str]:
    if not car_lines:
        return car_lines

    out: List[str] = []
    coord_idx = 0
    sym_counts: Dict[str, int] = {}

    for line in car_lines:
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue
        if lower.startswith("materials studio generated car file"):
            continue
        if lower == "end":
            out.append("end")
            continue
        if lower.startswith("!biosym") or lower.startswith("pbc=") or lower.startswith("!date") or lower.startswith("pbc "):
            out.append(line)
            continue

        parsed = _parse_car_coord_line(line)
        if parsed is None:
            continue

        if coord_idx >= len(symbols):
            raise RuntimeError(
                f"CAR coordinate count exceeds FDF atom count: got >{len(symbols)} coordinates."
            )

        x, y, z, residue, charge = parsed
        sym = symbols[coord_idx]
        coord_idx += 1
        sym_counts[sym] = sym_counts.get(sym, 0) + 1

        out.append(
            _format_arc_coord_line(
                sym=sym,
                per_symbol_idx=sym_counts[sym],
                atom_idx=coord_idx,
                x=x,
                y=y,
                z=z,
                charge=charge,
                residue=residue,
            )
        )

    if coord_idx != len(symbols):
        raise RuntimeError(
            f"Failed to map all FDF symbols onto CAR coordinates: expected {len(symbols)}, got {coord_idx}."
        )

    return out


def write_arc_from_lines(base: str, mdcar_lines: List[str], mde: MDEData, fdf: FDFInfo, out_arc: Path):
    lines = list(mdcar_lines)
    idxs = _find_mdcar_markers(lines, base)
    if len(idxs) < 2:
        raise RuntimeError("MD_CAR frame markers not found (need >=2 occurrences of basename in MD_CAR)")

    atom = idxs[1] - idxs[0]
    if atom <= 7:
        raise RuntimeError(f"Invalid MD_CAR block length inferred: {atom}")

    end_line_no = idxs[-1] + 1
    step = int((end_line_no - 1) / atom)
    if step <= 0:
        raise RuntimeError("Could not infer number of steps from MD_CAR.")

    energies = mde.energy_col4[1:] if len(mde.energy_col4) > 1 else mde.energy_col4

    type_line, count_line = _poscar_type_and_counts(base, fdf)

    out_arc.parent.mkdir(parents=True, exist_ok=True)
    with out_arc.open("w") as fw:
        for i in range(1, step + 1):
            stot = i * atom
            blk = lines[stot - atom: stot]
            if len(blk) != atom:
                break

            pos_lines: List[str] = []
            pos_lines.append(type_line)
            pos_lines.extend(blk[1:5])
            pos_lines.append(type_line)
            pos_lines.append(count_line)
            pos_lines.extend(blk[6:])
            pos_text = "\n".join(pos_lines) + "\n"

            car_lines = _run_pos2car(pos_text)
            if len(car_lines) < 5:
                raise RuntimeError("pos2car output too short; cannot compose ARC.")

            car_lines = _apply_fdf_symbols_to_car(car_lines, fdf.symbols)

            e = float(energies[i - 1]) if (i - 1) < len(energies) else float(energies[-1])

            if i == 1:
                for ln in car_lines:
                    low = ln.strip().lower()
                    if low.startswith("!biosym") or low.startswith("pbc="):
                        fw.write(ln + "\n")

            fw.write(f"{e:.8f}\n")

            wrote_date = False
            wrote_pbc = False
            for ln in car_lines:
                low = ln.strip().lower()
                if low.startswith("!date") and not wrote_date:
                    fw.write(ln + "\n")
                    wrote_date = True
                elif low.startswith("pbc ") and not wrote_pbc:
                    fw.write(ln + "\n")
                    wrote_pbc = True
                elif low.startswith("!biosym") or low.startswith("pbc="):
                    continue
                elif low == "end":
                    continue
                elif ln.strip():
                    fw.write(ln + "\n")
            fw.write("end\n")
            fw.write("end\n")


def write_arc(base: str, mdcar_path: Path, mde: MDEData, fdf: FDFInfo, out_arc: Path):
    lines = mdcar_path.read_text(errors="ignore").splitlines()
    write_arc_from_lines(base, lines, mde, fdf, out_arc)


def write_arc_from_files(base: str, mdcar_files: List[Path], mde: MDEData, fdf: FDFInfo, out_arc: Path):
    if not mdcar_files:
        raise RuntimeError("No MD_CAR files found to merge.")
    lines: List[str] = []
    for path in mdcar_files:
        lines.extend(path.read_text(errors="ignore").splitlines())
    write_arc_from_lines(base, lines, mde, fdf, out_arc)


def extract_final_structure_car(base: str, mdcar_files: List[Path], fdf: FDFInfo, out_car: Path) -> None:
    """
    Instead of an ARC (full trajectory), take only the last MD_CAR frame and save it as a single-structure .car file.
    Reuses the same block slicing/pos2car.py/FDF symbol reordering logic as write_arc_from_lines(), but
    writes the single-structure CAR content made by pos2car.py as is, without the ARC-specific
    trajectory format (energy line + end/end repeated twice).
    """
    if not mdcar_files:
        raise RuntimeError("No MD_CAR files found to merge.")

    lines: List[str] = []
    for path in mdcar_files:
        lines.extend(path.read_text(errors="ignore").splitlines())

    idxs = _find_mdcar_markers(lines, base)
    if len(idxs) < 2:
        raise RuntimeError("MD_CAR frame markers not found (need >=2 occurrences of basename in MD_CAR)")

    atom = idxs[1] - idxs[0]
    if atom <= 7:
        raise RuntimeError(f"Invalid MD_CAR block length inferred: {atom}")

    end_line_no = idxs[-1] + 1
    step = int((end_line_no - 1) / atom)
    if step <= 0:
        raise RuntimeError("Could not infer number of steps from MD_CAR.")

    type_line, count_line = _poscar_type_and_counts(base, fdf)

    stot = step * atom
    blk = lines[stot - atom: stot]
    if len(blk) != atom:
        raise RuntimeError("Last MD_CAR frame block is incomplete.")

    pos_lines: List[str] = [type_line]
    pos_lines.extend(blk[1:5])
    pos_lines.append(type_line)
    pos_lines.append(count_line)
    pos_lines.extend(blk[6:])
    pos_text = "\n".join(pos_lines) + "\n"

    car_lines = _run_pos2car(pos_text)
    if len(car_lines) < 5:
        raise RuntimeError("pos2car output too short; cannot compose final structure file.")

    car_lines = _apply_fdf_symbols_to_car(car_lines, fdf.symbols)

    out_car.parent.mkdir(parents=True, exist_ok=True)
    with out_car.open("w") as fw:
        for ln in car_lines:
            fw.write(ln + "\n")


def run_final_structure(base: Optional[str] = None, out_name: Optional[str] = None) -> Optional[Path]:
    """Used in option 1: extract only the final structure of the current directory SIESTA job to {base}_final.car."""
    base = base or infer_basename_from_fdf()

    restart_dirs = collect_restart_dirs()
    restart_base = f"{base}_R"
    mdcar_files = collect_existing_files(restart_dirs, f"{base}.MD_CAR", f"{restart_base}.MD_CAR")

    if not mdcar_files:
        print(f"[{base}] No {base}.MD_CAR found (in current directory or restart chain) - cannot extract final structure.")
        return None

    try:
        fdf_info = parse_fdf(base)
    except Exception as e:
        print(f"[{base}] Failed to parse {base}.fdf: {e}")
        return None

    out_car = Path(out_name or f"{base}_final.car")
    try:
        extract_final_structure_car(base, mdcar_files, fdf_info, out_car)
    except Exception as e:
        print(f"[{base}] Failed to extract final structure: {e}")
        return None

    print(f"* Saved final structure to {out_car.name}")
    return out_car


# ----------------------------
# TIMES scan
# ----------------------------
def _time_from_times_file(tf: Path) -> Optional[float]:
    sec = None
    for ln in tf.read_text(errors="ignore").splitlines():
        if "timer: Total elapsed wall-clock time (sec)" in ln:
            v = _extract_last_float(ln)
            if v is not None:
                sec = float(v)
    return sec


def _time_from_mtime(d: Path) -> Optional[float]:
    start = d / "input.fdf"
    candidates = [d / "denchar.out", d / "out.fdf"]

    if not start.exists():
        return None

    end = next((p for p in candidates if p.exists()), None)
    if end is None:
        return None

    sec = end.stat().st_mtime - start.stat().st_mtime
    if sec < 0:
        return None
    return float(sec)


def _time_from_tmp_force(d: Path) -> Optional[float]:
    tmp_files = sorted(p for p in d.glob("INPUT_TMP.*") if p.is_file())
    fs_files = sorted(
        p for p in list(d.glob("FORCE_STRESS")) + list(d.glob("FORCE_STRESS.*"))
        if p.is_file()
    )

    if not tmp_files or not fs_files:
        return None

    start = min(p.stat().st_mtime for p in tmp_files)
    end = max(p.stat().st_mtime for p in fs_files)

    sec = end - start
    if sec < 0:
        return None
    return float(sec)


def scan_times(root: Path = Path("."), recursive: bool = True) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []

    if recursive:
        candidate_dirs = set()
        candidate_dirs |= {p.parent for p in root.rglob("TIMES") if p.is_file()}
        candidate_dirs |= {p.parent for p in root.rglob("input.fdf") if p.is_file()}
        candidate_dirs |= {p.parent for p in root.rglob("INPUT_TMP.*") if p.is_file()}
        candidate_dirs |= {p.parent for p in root.rglob("FORCE_STRESS") if p.is_file()}
        candidate_dirs |= {p.parent for p in root.rglob("FORCE_STRESS.*") if p.is_file()}
        dirs = sorted(candidate_dirs)
    else:
        dirs = sorted(d for d in root.iterdir() if d.is_dir())

    for d in dirs:
        sec = None
        tf = d / "TIMES"
        if tf.exists() and tf.is_file():
            sec = _time_from_times_file(tf)

        if sec is None:
            sec = _time_from_mtime(d)

        if sec is None:
            sec = _time_from_tmp_force(d)

        if sec is None:
            continue

        rel_dir = d.relative_to(root)
        name = "." if str(rel_dir) == "." else rel_dir.as_posix()
        rows.append((name, sec))

    rows.sort(key=lambda x: x[0])
    return rows


def group_times(rows: List[Tuple[str, float]]) -> "OrderedDict[str, List[Tuple[str, float]]]":
    groups: "OrderedDict[str, List[Tuple[str, float]]]" = OrderedDict()
    for name, sec in rows:
        if name in (".", ""):
            group = "."
        else:
            group = name.split("/", 1)[0]
        groups.setdefault(group, []).append((name, sec))
    return groups


ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def _supports_ansi_color() -> bool:
    term = os.environ.get("TERM", "")
    no_color = os.environ.get("NO_COLOR")
    return sys.stdout.isatty() and bool(term) and term.lower() != "dumb" and not no_color


def _green(text: str) -> str:
    if _supports_ansi_color():
        return f"{ANSI_GREEN}{text}{ANSI_RESET}"
    return text


def print_times(rows: List[Tuple[str, float]]):
    if not rows:
        print("No TIMES found under sub-directories.")
        return

    groups = group_times(rows)
    name_w = max(9, max(len(n) for n, _ in rows), max(len(f"{g} [GROUP TOTAL]") for g in groups))
    total = sum(sec for _, sec in rows)

    header = f"{'Directory':<{name_w}}  {'Seconds':>12}  {'H:M:S':>12}"
    print(header)
    print("-" * len(header))

    first_group = True
    for group, items in groups.items():
        if not first_group:
            print()
        first_group = False

        group_total = sum(sec for _, sec in items)
        for name, sec in items:
            print(f"{name:<{name_w}}  {sec:12.2f}  {_sec_to_hms(sec):>12}")
        group_label = f"{group} [GROUP TOTAL]"
        line = f"{group_label:<{name_w}}  {group_total:12.2f}  {_sec_to_hms(group_total):>12}"
        print(_green(line))

    print("-" * len(header))
    print(f"{'TOTAL':<{name_w}}  {total:12.2f}  {_sec_to_hms(total):>12}")


# ----------------------------
# Full analysis of a single current directory (siesta_analyze.py main() logic)
# ----------------------------
def run_full_analysis(base: Optional[str] = None, analysis_dir: str = "analysis", xstart: int = 1) -> Dict[str, Any]:
    adir = Path(analysis_dir)
    adir.mkdir(parents=True, exist_ok=True)

    base = base or infer_basename_from_fdf()

    summary: Dict[str, Any] = {"base": base}

    restart_dirs = collect_restart_dirs()
    restart_base = f"{base}_R"
    mde_files = collect_existing_files(restart_dirs, f"{base}.MDE", f"{restart_base}.MDE")
    out_files = collect_existing_files(restart_dirs, f"{base}.out", f"{restart_base}.out")
    mdcar_files = collect_existing_files(restart_dirs, f"{base}.MD_CAR", f"{restart_base}.MD_CAR")

    if not mde_files:
        raise SystemExit(f"Missing required file(s): {base}.MDE in current directory or restart chain")
    mde = read_combined_mde(mde_files)
    summary["merged_directories"] = [str(d) for d in restart_dirs]
    summary["mde_files"] = [str(p) for p in mde_files]
    summary["out_files"] = [str(p) for p in out_files]
    summary["mdcar_files"] = [str(p) for p in mdcar_files]

    with open("Energy.dat", "w") as f:
        f.write(f"{float(mde.energy_col4[-1]):15.8f}  {base}\n")

    energy_opt = mde.energy_opt
    np.savetxt(adir / "energy.csv",
               np.column_stack([np.arange(1, len(energy_opt) + 1), energy_opt]),
               delimiter=",", header="step,energy_eV", comments="")
    plot_energy(energy_opt, adir / "convergence_energy.png", xstart=xstart)

    summary["n_opt_steps"] = int(len(energy_opt))
    summary["final_energy_eV"] = float(mde.energy_col4[-1])

    if out_files:
        tot_vecs, max_vals, res_vals = parse_combined_out_forces(out_files)
        fs = build_force_series_from_arrays(tot_vecs, max_vals, res_vals, nopt=0)
        with open(adir / "forces.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "Fx", "Fy", "Fz", "Ftot", "Max", "Res"])
            for i, st in enumerate(fs.steps):
                w.writerow([st, fs.fx[i], fs.fy[i], fs.fz[i], fs.f_total[i], fs.max_force[i], fs.res_force[i]])

        plot_forces(fs, adir / "forces.png")
        plot_force_convergence(fs, adir / "convergence_force.png")
        summary["n_force_steps"] = int(len(fs.steps))
        summary["n_max_force_nonnull"] = int(sum(v is not None for v in fs.max_force))
        summary["n_tot_force_nonnull"] = int(sum(v is not None for v in fs.f_total))
    else:
        summary["n_force_steps"] = 0

    try:
        fdf_info = parse_fdf(base)
        summary["fdf_file"] = str(Path(f"{base}.fdf"))
    except Exception as e:
        fdf_info = None
        summary["fdf_error"] = str(e)

    if mdcar_files and fdf_info is not None:
        try:
            out_arc = adir / f"{base}.arc"
            write_arc_from_files(base, mdcar_files, mde, fdf_info, out_arc)
            summary["arc_file"] = str(out_arc)
        except Exception as e:
            summary["arc_error"] = str(e)
    else:
        summary["arc_file"] = None

    (adir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Done. Outputs in: {adir} and Energy.dat in current directory.")
    return summary


# ----------------------------
# Batch helpers for options 0 / 2
# ----------------------------
# SIESTA가 치명적으로 죽을 때 남기는 대표적인 종료 문구들.
# "ERROR STOP from Node" / "Error in Cholesky factorisation in cdiag" 는
# 실제로 원자가 너무 가까워서(rij ~ 0) overlap matrix 대각화(cdiag)가 실패했을 때
# 나오는 조합으로 확인됨 (사용자 제공 로그 기준). "stopping program" 은 SIESTA의
# die() 서브루틴이 다른 여러 치명적 오류에서도 공통으로 찍는 문구라 넓게 같이 잡아준다.
_ERROR_PATTERNS = [
    r"error stop",                        # "ERROR STOP from Node..." etc - the wording may not
                                           # match exactly, so relaxed to match even without "from node"
    r"error in cholesky factori[sz]ation", # cdiag diagonalization failure (from overlapping atoms, etc.)
    r"stopping program",
    r"siesta error",
    r"segmentation fault",
    r"floating[- ]point exception",
    r"forrtl",                            # Intel Fortran runtime error prefix
    r"not enough memory",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)

# 종료 여부와 무관하게 눈여겨봐야 할 경고성 문구들. "Atoms .. too close" 는 실제로
# 이 경고가 잔뜩 찍히고 나서 cdiag 실패로 죽는 걸 사용자가 확인해준 케이스라,
# Error 로 안 잡히는 경우(아직 죽지는 않았지만 기하구조가 이미 깨진 경우)에도
# 눈에 띄도록 경고 태그로 남긴다. "Bad DM normalization" 은 fatal 인지 단순 경고인지
# 확실치 않아 마찬가지로 Error 판정에는 안 쓰고 태그로만 노출.
_WARNING_DEFS = [
    ("Atoms too close", re.compile(r"too close", re.IGNORECASE)),
    ("Bad DM normalization", re.compile(r"bad dm normali[sz]ation", re.IGNORECASE)),
]

# If the .out file has not been updated for this time (seconds) while there is no End of run
# and no error message, it is more likely dead than "still running", so it is marked separately.
_STALE_SECONDS = 600


def get_siesta_status(d) -> Dict[str, Optional[str]]:
    """
    Roughly determine the SIESTA job status of a single directory.
    - If there is no .out file: "Not started"
    - If any of _ERROR_PATTERNS is present: "Error" (takes priority regardless of ">> End of run:")
    - If ">> End of run:" is present and no error pattern: "Converged" (normal exit)
    - Otherwise: if the file was updated recently, "Running/Incomplete",
                 if it has not been updated for a while (default 10 min), "Incomplete (stalled/crashed?)"
    - Warning phrases such as "Atoms .. too close" / "Bad DM normalization" are
      appended after the status as [Warning: ...] together with their counts
      (can show up even if it ended as Converged - geometry is odd but SCF itself ran)
    (Heuristic - if you hit error wording different from this, report it so it can be tuned more precisely)
    """
    d = Path(d)
    fdfs = sorted(d.glob("*.fdf"))
    base = fdfs[0].stem if fdfs else None

    out_path = None
    if base and (d / f"{base}.out").exists():
        out_path = d / f"{base}.out"
    else:
        out_candidates = sorted(d.glob("*.out"))
        if out_candidates:
            out_path = out_candidates[0]

    if out_path is None:
        return {"base": base, "status": "Not started", "detail": "no .out file"}

    text = out_path.read_text(errors="ignore")
    has_end = ">> End of run:" in text
    err_match = _ERROR_RE.search(text)

    if err_match:
        status = "Error"
        detail = f"{out_path.name} ({err_match.group(0)})"
    elif has_end:
        status = "Converged"
        detail = out_path.name
    else:
        age = time.time() - out_path.stat().st_mtime
        if age > _STALE_SECONDS:
            status = "Incomplete (stalled/crashed?)"
        else:
            status = "Running/Incomplete"
        detail = out_path.name

    warn_notes = []
    for label, pat in _WARNING_DEFS:
        n = len(pat.findall(text))
        if n:
            warn_notes.append(f"{label} x{n}" if n > 1 else label)
    if warn_notes:
        status = f"{status} [Warning: {', '.join(warn_notes)}]"

    return {"base": base, "status": status, "detail": detail}


def get_final_energy(d) -> Tuple[Optional[float], Optional[str]]:
    """Used in option 2. Prefers Energy.dat (output of run_full_analysis) if present, otherwise reads directly from MDE."""
    d = Path(d)
    edat = d / "Energy.dat"
    if edat.exists():
        try:
            line = edat.read_text().strip().splitlines()[0]
            parts = line.split()
            return float(parts[0]), (parts[1] if len(parts) > 1 else None)
        except Exception:
            pass

    fdfs = sorted(d.glob("*.fdf"))
    if not fdfs:
        return None, None
    base = fdfs[0].stem
    mde_path = d / f"{base}.MDE"
    if not mde_path.exists():
        return None, base
    try:
        mde = read_mde_with_auto_skip(mde_path)
        return float(mde.energy_col4[-1]), base
    except Exception:
        return None, base


# ============================================================
# 2. CLI (CCpyVASPAnal.py style: sys.argv[1] option branching + usage output)
# ============================================================

USAGE = """
How to use : CCpySIESTAAnal.py [option] [sub_option1] [sub_option2..]
--------------------------------------
[suboptions]
-sub : deep in subdirectories (used with options 0, 1, 2, 3, -d)

[options]
-d : Clear SIESTA output files (except of *.fdf, *.psf, *.vps, *.ion)
    ex) CCpySIESTAAnal.py -d

 0 : Check SIESTA job status (Converged / Running-Incomplete / Error / Not started)
     -> saves 00_job_status.txt / 00_job_status.csv
    ex) CCpySIESTAAnal.py 0
    ex) CCpySIESTAAnal.py 0 -sub

 1 : Extract ONLY the final structure (last MD_CAR frame, via pos2car.py +
     FDF atom order) as a standalone <dirname>_final.car file - no full ARC
     trajectory. Shows an interactive picker (CCpySIESTABandSubmit.py style):
         1 : Ti2CTx_O_NT_2
         2 : Ti2CTx_O_NT_Bandtest
         ...
         0 : All directories
         choose file :
     (accepts a single number, a comma/range list like 1-3,5, or 0 for all)
     By default the .car is written ONE LEVEL UP from each job directory
     (so all final structures land together, not buried among each job's
     own output files) - use -here to write it inside the job directory
     instead (as <base>_final.car).
    ex) CCpySIESTAAnal.py 1
    ex) CCpySIESTAAnal.py 1 -sub                  -> search sub-directories recursively
    ex) CCpySIESTAAnal.py 1 -a                    -> select all directories, no prompt
    ex) CCpySIESTAAnal.py 1 -systems=1-3,5        -> pick specific numbers, no prompt
    ex) CCpySIESTAAnal.py 1 -here                 -> save inside each job dir instead of the parent
    ex) CCpySIESTAAnal.py 1 --base=SystemName     -> override basename (applies to every selected dir)
    ex) CCpySIESTAAnal.py 1 --out=final_structure.car   -> explicit path/name (single dir only)

 2 : Get final total energy list across SIESTA job directories
     -> saves 03_<folder>_FinalEnergies.txt / .csv / .png
        (same file names as CCpyVASPAnal.py 2)
    ex) CCpySIESTAAnal.py 2
    ex) CCpySIESTAAnal.py 2 n     : sub option 'n' -> do not save the plot
    ex) CCpySIESTAAnal.py 2 -st   : sub option '-st' -> sort by total energy
    ex) CCpySIESTAAnal.py 2 -sa   : sub option '-sa' -> sort by energy/atom

 3 : Run full analysis (energy convergence, force convergence, ARC file
     generation) on one or more SIESTA job directories found under here.
     Same interactive picker as option 1 (-a / -systems= / -sub).
    ex) CCpySIESTAAnal.py 3
    ex) CCpySIESTAAnal.py 3 -sub                  -> search sub-directories recursively
    ex) CCpySIESTAAnal.py 3 -a                    -> select all directories, no prompt
    ex) CCpySIESTAAnal.py 3 -systems=1-3,5        -> pick specific numbers, no prompt
    ex) CCpySIESTAAnal.py 3 --base=SystemName     -> override basename (applies to every selected dir)
    ex) CCpySIESTAAnal.py 3 --xstart=10
    ex) CCpySIESTAAnal.py 3 --analysis-dir=analysis

-time : Scan TIMES files (or fallback timestamps) and print wall-clock time table
    ex) CCpySIESTAAnal.py -time       -> recursive scan (default)
    ex) CCpySIESTAAnal.py -time -no   -> only immediate sub-directories

"""


def print_usage():
    print(USAGE)


try:
    chk = sys.argv[1]
except Exception:
    print_usage()
    quit()

sub = False
if "-sub" in sys.argv:
    sub = True


if sys.argv[1] == "-d":
    keep_ext = (".fdf", ".psf", ".vps", ".ion")
    dirs = selectSiestaOutputs("./", ask=True, sub=sub)
    if not dirs:
        quit()
    for d in dirs:
        print(d)
    yn = raw_input("Are you sure to remove output files in these dirs (keep *.fdf/*.psf/*.vps/*.ion)? (y/n) ")
    if yn.lower() not in ("y", "yes"):
        quit()
    pwd = os.getcwd()
    for d in dirs:
        os.chdir(d)
        for f in os.listdir("."):
            if os.path.isdir(f):
                continue
            if f.endswith(keep_ext):
                continue
            try:
                os.remove(f)
            except Exception:
                pass
        os.chdir(pwd)

elif sys.argv[1] == "0":
    dirs = selectSiestaOutputs("./", ask=False, sub=sub)
    rows = []
    for d in dirs:
        st = get_siesta_status(d)
        rows.append({"Directory": d, "Base": st["base"], "Status": st["status"], "Detail": st["detail"]})

    if not rows:
        print("No SIESTA job directory found.")
        quit()

    df = pd.DataFrame(rows)
    txt = df.to_string(index=False)
    print(txt)
    with open("00_job_status.txt", "w") as f:
        f.write(txt + "\n")
    df.to_csv("00_job_status.csv", index=False)
    print("\n* Saved job status to 00_job_status.txt / 00_job_status.csv")

elif sys.argv[1] == "1":
    base = _get_kv_arg("base", None, str)
    out_name = _get_kv_arg("out", None, str)
    preselect = _get_kv_arg("systems", None, str)
    ask = "-a" not in sys.argv
    # 기본값: 각 job 디렉토리의 "상위" 폴더에 <디렉토리이름>_final.car 로 모아서 저장
    # (계산 폴더 안에 넣으면 파일이 너무 많아서 안 보이므로). -here 주면 예전처럼
    # job 디렉토리 안에 <base>_final.car 로 저장.
    write_here = "-here" in sys.argv

    dirs = find_siesta_dirs("./", sub=sub)
    dirs = select_siesta_dirs(dirs, ask=ask, preselect=preselect)

    if not dirs:
        print("Nothing selected.")
        quit()

    pwd = os.getcwd()
    for d in dirs:
        print(f"\n# ----------- {d} ----------- #")
        dirname = Path(d).resolve().name
        os.chdir(d)
        try:
            if out_name:
                this_out = out_name
            elif write_here:
                this_out = None  # use the run_final_structure default (<base>_final.car)
            else:
                this_out = str(Path("..") / f"{dirname}_final.car")
            run_final_structure(base=base, out_name=this_out)
        except Exception as e:
            print(f"Error extracting final structure for {d}: {e}")
        finally:
            os.chdir(pwd)

elif sys.argv[1] == "2":
    sort_mode = None  # "tot" or "atom"

    # Sub-option n : do not create the figure (CCpyVASPAnal.py 2 style)
    show_plot = "n" not in sys.argv[2:]

    if "-st" in sys.argv:
        sort_mode = "tot"
    elif "-sa" in sys.argv:
        sort_mode = "atom"

    dirs = selectSiestaOutputs("./", ask=False, sub=sub)

    data = []
    for d in dirs:
        E, base = get_final_energy(d)
        if E is None:
            continue
        nat = None
        if base:
            nat = get_natoms_from_fdf_file(Path(d) / f"{base}.fdf")
        E_per_atom = (E / nat) if nat else None
        data.append({"dir": d, "base": base, "E": E, "natoms": nat, "E_per_atom": E_per_atom})

    if len(data) == 0:
        print("No SIESTA energy data (Energy.dat / *.MDE) found.")
        quit()

    df = pd.DataFrame(data)

    if sort_mode == "tot":
        df = df.sort_values(by="E").reset_index(drop=True)
    elif sort_mode == "atom":
        df = df.dropna(subset=["E_per_atom"])
        df = df.sort_values(by="E_per_atom").reset_index(drop=True)

    if len(df) == 0:
        print("No valid energy data after sorting.")
        quit()

    txt = df.to_string(index=False)
    print(txt)

    # The output files share the names of CCpyVASPAnal.py 2 :
    # 03_<folder>_FinalEnergies.txt / .csv / .png, so txt, csv and the figure of
    # one run stay together and the folder name keeps runs from overwriting each other.
    base_filename = "03_" + Path.cwd().name + "_FinalEnergies"
    txt_filename = base_filename + ".txt"
    csv_filename = base_filename + ".csv"
    png_filename = base_filename + ".png"
    with open(txt_filename, "w") as f:
        f.write(txt + "\n")
    df.to_csv(csv_filename, index=False)
    print("\n* Saved energy list to " + txt_filename + " / " + csv_filename)

    if show_plot:
        ycol = "E_per_atom" if sort_mode == "atom" else "E"
        plot_df = df.dropna(subset=[ycol])
        if len(plot_df) == 0:
            print("* No energy data to plot.")
        else:
            x = range(len(plot_df))
            fig = plt.figure(figsize=(8, 7))
            plt.plot(x, plot_df[ycol].values, marker="o", color="#0054FF")
            plt.xticks(x, plot_df["dir"].tolist(), rotation=45, ha="right")
            plt.ylabel("Energy/atom (eV)" if sort_mode == "atom" else "Total energy (eV)")
            plt.grid()
            plt.tight_layout()
            plt.savefig(png_filename, dpi=300)
            plt.close(fig)
            print("* Saved energy plot to " + png_filename)

elif sys.argv[1] == "3":
    base = _get_kv_arg("base", None, str)
    analysis_dir = _get_kv_arg("analysis-dir", "analysis", str)
    xstart = _get_kv_arg("xstart", 1, int)
    preselect = _get_kv_arg("systems", None, str)
    ask = "-a" not in sys.argv

    dirs = find_siesta_dirs("./", sub=sub)
    dirs = select_siesta_dirs(dirs, ask=ask, preselect=preselect)

    if not dirs:
        print("Nothing selected.")
        quit()

    pwd = os.getcwd()
    for d in dirs:
        print(f"\n# ----------- {d} ----------- #")
        os.chdir(d)
        try:
            run_full_analysis(base=base, analysis_dir=analysis_dir, xstart=xstart)
        except SystemExit as e:
            print(e)
        except Exception as e:
            print(f"Error analyzing {d}: {e}")
        finally:
            os.chdir(pwd)

elif sys.argv[1] == "-time":
    recursive = "-no" not in sys.argv
    rows = scan_times(root=Path("."), recursive=recursive)
    print_times(rows)

    with open("times.txt", "w") as f:
        f.write(f"{'Directory':<65}{'Seconds':>12}   {'H:M:S'}\n")
        f.write("-" * 91 + "\n")

        for group, items in group_times(rows).items():
            for name, sec in items:
                f.write(f"{name:<65}{sec:>12.2f}   {_sec_to_hms(sec)}\n")

            group_total = sum(sec for _, sec in items)
            f.write(
                f"{group + ' [GROUP TOTAL]':<65}"
                f"{group_total:>12.2f}   {_sec_to_hms(group_total)}\n"
            )
            f.write("\n")

        total_sec = sum(sec for _, sec in rows)
        f.write(f"{'ALL TOTAL':<65}{total_sec:>12.2f}   {_sec_to_hms(total_sec)}\n")

    print("\n* Saved time table to times.txt")

else:
    print_usage()
