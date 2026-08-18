#!/home/shared/anaconda3/envs/siesta/bin/python
# -*- coding: utf-8 -*-
"""
siesta_band.py
========================

Single consolidated driver for the SIESTA Band / FatBand / DOS workflow.

This file merges five previously-separate scripts into one, as sub-commands:

    genfdf        <- siesta_Band-DOS_lineband_fat.py
                     (Band-DOS/BANDS, Band-DOS/DOS dir + fdf generation,
                      BandLines from .car, fatbands Wfs.band.min/max auto-set)
    fatband       <- siesta_FatBand.py
                     (element / moiety decomposed fatbands .mpr, runs `fat`
                      and `eigfat2plot`)
    plot-band     <- siesta_bandplot.py
                     (gnubands + band structure plot, bandgap annotation)
    plot-fatband  <- siesta_fatbandplot.py
                     (fatband overlay / fat-only plot)
    plot-dos      <- siesta_dosplot.py
                     (DOS plot, Fermi alignment, van Hove peak detection)
    pipeline      <- new: runs the whole thing end-to-end (equivalent to
                     WorkFlow_FatBandDOS.sh), calling the sub-commands above
                     in sequence and running SIESTA itself in between.
                     Use --steps to run only a subset (세분화).

Each sub-command keeps the exact same flag names as the original standalone
script, so any existing usage/notes transfer directly - just prefix the
command with the sub-command name, e.g.:

    OLD: python siesta_bandplot.py -e -2 -E 2 --lw-band 1
    NEW: python -m CCpy.SIESTA.siesta_band plot-band -e -2 -E 2 --lw-band 1

Running with no sub-command prints the menu below and exits (same
convention as CCpyJobSubmit.py in this lab's CCpy framework).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# =============================================================================
# Self-bootstrap re-exec (must run BEFORE numpy/matplotlib are imported)
# =============================================================================
#
# This script is invoked by CCpySIESTABandSubmit.py using whatever python the
# lab's queue_config.yaml points at (self.python_path) - normally the general
# CCpy framework env. All fdf/geometry parsing here is done with plain
# regex/stdlib (no sisl), so the only real dependencies are numpy and
# matplotlib. As a safety net in case some node's python lacks those too,
# check for them and os.execv-relaunch under a known-good env python if
# needed, rather than just crashing with an ImportError. Same pattern as
# bootstrap_reexec_if_needed() in test_CCpySIESTAInputGen.py.
#
# Set SIESTA_BAND_WORKFLOW_NO_AUTOENV=1 to disable this and run under
# whatever interpreter was actually invoked (e.g. for debugging).

_REEXEC_GUARD_ENV = "_SIESTA_BAND_WORKFLOW_REEXECED"

# Known-good fallback python interpreters that have numpy/matplotlib
# installed, tried in order, only used if the current interpreter is
# missing them. <<< EDIT/ADD paths here if needed for this lab's cluster.
KNOWN_ENV_PYTHONS: Tuple[str, ...] = (
    "/home/shared/anaconda3/envs/CCpy/bin/python",
    "/home/shared/anaconda3/envs/siesta/bin/python",
)


def _current_interpreter_has_deps() -> bool:
    try:
        import numpy  # noqa: F401
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def bootstrap_reexec_if_needed() -> None:
    if os.environ.get(_REEXEC_GUARD_ENV) == "1":
        return
    if os.environ.get("SIESTA_BAND_WORKFLOW_NO_AUTOENV") == "1":
        return
    if _current_interpreter_has_deps():
        return

    this_file = os.path.abspath(__file__)
    for candidate in KNOWN_ENV_PYTHONS:
        if not os.path.isfile(candidate):
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "import numpy, matplotlib"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            continue
        if probe.returncode != 0:
            continue
        print(f"[siesta_band] current python is missing numpy/matplotlib; "
              f"relaunching under {candidate}", file=sys.stderr)
        sys.stderr.flush()
        os.environ[_REEXEC_GUARD_ENV] = "1"
        os.execv(candidate, [candidate, this_file] + sys.argv[1:])
    # No known candidate had the deps either - fall through and let the
    # normal import below fail with its own clear ImportError.


bootstrap_reexec_if_needed()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def add_bool_flag(ap: argparse.ArgumentParser, flag: str, dest: Optional[str] = None,
                   default: bool = True, help: Optional[str] = None) -> None:
    """
    Python-3.8-compatible stand-in for argparse.BooleanOptionalAction (which
    was only added in 3.9 - this lab's CCpy conda env is still 3.8). Adds
    both --flag and --no-flag manually, sharing one dest, same CLI surface
    ('--element' / '--no-element' etc. keep working exactly as before).
    """
    name = flag.lstrip("-")
    dest = dest or name.replace("-", "_")
    group = ap.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)

# =============================================================================
# Shared defaults (single place to change binary locations)
# =============================================================================

DEFAULT_SIESTA_BIN_DIR = Path("/opt/siesta/siesta-5.4.2/siesta-mkl-mpi/bin")
DEFAULT_SIESTA = DEFAULT_SIESTA_BIN_DIR / "siesta"
FAT_EXE_NAME = "fat"
EIGFAT2PLOT_EXE_NAME = "eigfat2plot"
GNUBANDS_EXE_NAME = "gnubands"


# =============================================================================
# ============================  genfdf  ======================================
# (from siesta_Band-DOS_lineband_fat.py)
# =============================================================================

def parse_systemlabel_from_fdf(fdf_path: Path) -> Optional[str]:
    try:
        for line in fdf_path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"(?i)^\s*SystemLabel\s+(\S+)", s)
            if m:
                return m.group(1)
    except FileNotFoundError:
        return None
    return None


def auto_detect_label(workdir: Path) -> str:
    for f in sorted(workdir.glob("*.fdf")):
        lbl = parse_systemlabel_from_fdf(f)
        if lbl:
            return lbl
    raise SystemExit("ERROR: SystemLabel auto-detect failed. Use --label")


def safe_symlink(src: Path, dst_dir: Path):
    if src.exists():
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def copy_base_fdf(base_fdf: Path, dst_dir: Path, label: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_fdf, dst_dir / f"{label}.fdf")


def stage_common_files(
    top: Path,
    dst_dir: Path,
    label: str,
    *,
    mode: str = "copy",
    exts=("DM", "XV", "EIG", "WFSX", "HSX", "ORB_INDX"),
):
    """Stage common label.* files into a run directory.

    - label.{DM,XV,EIG,WFSX,HSX,ORB_INDX} are staged (copy or symlink)
    - label*.WFSX wildcard is ALSO staged to cover:
        label.bands.WFSX / label.fullBZ.WFSX (Siesta 5.4.x)
    - Pseudopotentials (*.psf/*.psml) are always symlinked.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    mode_l = mode.lower().strip()
    if mode_l not in {"copy", "symlink"}:
        raise SystemExit(f'ERROR: --stage-common must be "copy" or "symlink" (got: {mode})')

    def put(src: Path):
        if not src.exists():
            return
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode_l == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

    for ext in exts:
        put(top / f"{label}.{ext}")

    for w in sorted(top.glob(f"{label}*.WFSX")):
        put(w)

    for p in list(top.glob("*.psf")) + list(top.glob("*.psml")):
        safe_symlink(p, dst_dir)


def upsert_key(lines: List[str], key: str, newline: str):
    """Replace first occurrence of key, else append."""
    for i, ln in enumerate(lines):
        if re.match(rf"(?i)^\s*{re.escape(key)}\b", ln):
            lines[i] = newline
            return
    lines.append(newline)


def patch_single_point(lines: List[str]):
    """Make it effectively single-point (bands-friendly) and enable SaveDM/SaveXV usage."""
    def set_bool(key):
        for i, ln in enumerate(lines):
            if re.match(rf"(?i)^\s*{re.escape(key)}\b", ln):
                lines[i] = f"{key} .true."
                return
        lines.append(f"{key} .true.")

    found_type = False
    found_steps = False

    for i, ln in enumerate(lines):
        if re.match(r"(?i)MD\.TypeOfRun", ln):
            lines[i] = "MD.TypeOfRun CG"
            found_type = True
        elif re.match(r"(?i)MD\.NumCGsteps", ln):
            lines[i] = "MD.NumCGsteps 0"
            found_steps = True
        elif re.match(r"(?i)WriteBands\b", ln):
            lines[i] = "WriteBands .true."

    if not found_type:
        lines.append("MD.TypeOfRun CG")
    if not found_steps:
        lines.append("MD.NumCGsteps 0")

    set_bool("WriteBands")
    set_bool("DM.UseSaveDM")
    set_bool("MD.UseSaveXV")


def scale_k_int_for_dos(ki: int) -> int:
    """
    0 or 1 -> unchanged, else *4 capped at 8.

    kgrid_Monkhorst_Pack for a 1D tube (2 vacuum directions + 1 periodic
    direction) has k=1 along the vacuum directions and k>1 only along the
    real periodic axis. Scaling k=1 up too (the original *4-always rule)
    silently turns a "1 vacuum-direction k-point" into 4, bloating the total
    k-point count by up to 16x (4 x scaled-periodic x 4) for something with
    zero physical dispersion - for a large system this alone is the
    difference between a DOS run finishing and one that appears to "never
    end". Only the direction(s) that already have real periodicity (k>1)
    get refined for DOS; k=1 (and k=0) are left alone.
    """
    if ki <= 1:
        return ki
    return min(8, ki * 4)


def patch_kgrid_for_dos(lines: List[str]):
    """Scale ints in 3x3 kgrid_Monkhorst_Pack block (keep shifts)."""
    start = end = None
    for i, ln in enumerate(lines):
        if re.match(r"(?i)%block\s+kgrid_Monkhorst_Pack", ln):
            start = i
        elif re.match(r"(?i)%endblock\s+kgrid_Monkhorst_Pack", ln):
            end = i
            break
    if start is None or end is None:
        return

    for row in range(3):
        idx = start + 1 + row
        if idx >= end:
            return
        parts = lines[idx].split()
        if len(parts) < 3:
            return
        try:
            k0 = scale_k_int_for_dos(int(parts[0]))
            k1 = scale_k_int_for_dos(int(parts[1]))
            k2 = scale_k_int_for_dos(int(parts[2]))
        except ValueError:
            return
        parts[0], parts[1], parts[2] = str(k0), str(k1), str(k2)
        lines[idx] = " ".join(parts)


def detect_periodic_axis_from_kgrid(fdf_path: Path) -> Optional[str]:
    """
    Auto-detect the periodic (tube-stacking) axis from %block kgrid_Monkhorst_Pack
    in the base fdf: for a 1D system with vacuum along two directions, SIESTA's
    k-grid only needs >1 k-point along the truly periodic lattice vector, so the
    corresponding diagonal entry (row 0 -> a, row 1 -> b, row 2 -> c) is the only
    one > 1. Returns None if the block is missing or the result is ambiguous
    (zero or more than one row with a diagonal > 1), so the caller can fall back
    to a default instead of guessing.
    """
    if not fdf_path.exists():
        return None
    lines = _fdf_block(_resolve_fdf_includes(fdf_path), "kgrid_Monkhorst_Pack")
    if lines is None or len(lines) < 3:
        return None

    diag = []
    for row in range(3):
        parts = lines[row].split()
        if len(parts) < 3:
            return None
        try:
            diag.append(abs(int(float(parts[row]))))
        except ValueError:
            return None

    axis_names = ["a", "b", "c"]
    periodic = [axis_names[i] for i, k in enumerate(diag) if k > 1]
    if len(periodic) == 1:
        return periodic[0]
    return None


def ensure_dos_options(lines: List[str], emin: float, emax: float, broad: float, npts: int):
    upsert_key(lines, "WriteDOS", "WriteDOS .true.")
    upsert_key(lines, "DOS.EnergyMin", f"DOS.EnergyMin {emin} eV")
    upsert_key(lines, "DOS.EnergyMax", f"DOS.EnergyMax {emax} eV")
    upsert_key(lines, "DOS.Broadening", f"DOS.Broadening {broad} eV")
    upsert_key(lines, "DOS.NumberPoints", f"DOS.NumberPoints {npts}")


def ensure_pdos_block(lines: List[str], enable: bool, emin: float, emax: float, broad: float, npts: int):
    start = end = None
    for i, ln in enumerate(lines):
        if re.match(r"(?i)%block ProjectedDensityOfStates", ln):
            start = i
        if re.match(r"(?i)%endblock ProjectedDensityOfStates", ln):
            end = i

    if enable:
        block = [
            "%block ProjectedDensityOfStates",
            f" {emin} {emax} {broad} {npts} eV",
            "%endblock ProjectedDensityOfStates",
        ]
        if start is not None and end is not None:
            lines[start:end + 1] = block
        else:
            lines += [""] + block
    else:
        if start is not None and end is not None:
            del lines[start:end + 1]


def find_existing_outfile(basename: str, rootdir: Path) -> Path:
    candidates = [
        rootdir / f"{basename}.out",
        rootdir / basename / f"{basename}.out",
        Path.cwd() / f"{basename}.out",
        Path.cwd() / basename / f"{basename}.out",
    ]
    outfile = next((p for p in candidates if p.exists()), None)
    if outfile is None:
        tried = "\n".join(str(x) for x in candidates)
        raise FileNotFoundError(
            f"Cannot find an existing SIESTA .out for '{basename}' (needed to read Nelec for "
            f"fatbands/NumberOfBands auto-setup).\nTried:\n{tried}\n"
            f"Run a plain SCF calculation for this system first (producing {basename}.out "
            f"directly in the system's top-level directory), then re-run.")
    return outfile


def read_nelec_from_out(out_path: Path) -> float:
    if not out_path.exists():
        raise FileNotFoundError(f"Cannot find SIESTA out: {out_path}")
    text = out_path.read_text(errors="ignore")

    vals = re.findall(r"Total number of electrons\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if not vals:
        raise RuntimeError("Cannot find 'Total number of electrons' in .out")
    return float(vals[-1])


def get_number_of_bands_from_out(basename: str, rootdir: Path, extra: int) -> int:
    outfile = find_existing_outfile(basename, rootdir)
    nelec = read_nelec_from_out(outfile)
    return int(math.ceil(nelec / 2.0) + extra)


def estimate_fermi_band_index(nelec: float) -> int:
    """Heuristic for non-spin-polarized runs: occupied bands ~ ceil(Nelec/2)."""
    return int(math.ceil(nelec / 2.0))


def ensure_fatbands_options(
    lines: List[str],
    *,
    enable: bool,
    nwindow: int,
    nelec: float,
    number_of_bands: int,
):
    """Add fatbands-related FDF options and set Wfs.band.min/max around EF."""
    if not enable:
        return

    n_ef = estimate_fermi_band_index(nelec)
    bmin = max(1, n_ef - nwindow)
    bmax = min(number_of_bands, n_ef + nwindow)

    upsert_key(lines, "Save-HS", "Save-HS T")
    upsert_key(lines, "WFS.Write.For.Bands", "WFS.Write.For.Bands T")
    upsert_key(lines, "Wfs.band.min", f"Wfs.band.min {bmin}")
    upsert_key(lines, "Wfs.band.max", f"Wfs.band.max {bmax}")


@dataclass
class CarStructure:
    lattice: np.ndarray
    species: List[str]
    cart_coords: np.ndarray


def lattice_from_abc_angles(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)

    ax, ay, az = a, 0.0, 0.0
    bx = b * math.cos(ga)
    by = b * math.sin(ga)
    bz = 0.0

    cx = c * math.cos(be)
    sin_ga = math.sin(ga)
    if abs(sin_ga) < 1e-10:
        raise ValueError("Invalid gamma angle (sin(gamma) ~ 0).")
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / sin_ga
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))

    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=float)


def read_car(path: str) -> CarStructure:
    lines = Path(path).read_text(errors="ignore").splitlines()
    pbc_line = None
    for ln in lines:
        if ln.strip().startswith("PBC") and re.search(r"\bPBC\b", ln):
            toks = ln.split()
            if len(toks) >= 7 and toks[0] == "PBC":
                pbc_line = ln
                break
    if pbc_line is None:
        raise ValueError("Could not find a valid PBC cell line in .car")

    toks = pbc_line.split()
    a, b, c = map(float, toks[1:4])
    alpha, beta, gamma = map(float, toks[4:7])
    lattice = lattice_from_abc_angles(a, b, c, alpha, beta, gamma)

    start_idx = lines.index(pbc_line) + 1
    species: List[str] = []
    coords: List[List[float]] = []

    for ln in lines[start_idx:]:
        s = ln.strip()
        if not s:
            continue
        if s.lower() == "end":
            break
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = map(float, parts[1:4])
        except Exception:
            continue

        elem = None
        for tok in reversed(parts):
            if re.fullmatch(r"[A-Z][a-z]?", tok):
                elem = tok
                break
        if elem is None:
            m = re.match(r"([A-Za-z]+)", parts[0])
            elem = (m.group(1).capitalize() if m else "X")

        species.append(elem)
        coords.append([x, y, z])

    if not coords:
        raise ValueError("No atoms parsed from .car")

    return CarStructure(lattice=lattice, species=species, cart_coords=np.array(coords, float))


def frac_coords(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    inv_lat = np.linalg.inv(lattice)
    return cart @ inv_lat


def guess_lattice_system(a: float, b: float, c: float,
                          alpha: float, beta: float, gamma: float,
                          tol_len: float = 1e-3, tol_ang: float = 1e-2) -> str:
    def eq(x, y, tol):
        return abs(x - y) <= tol

    a90 = eq(alpha, 90.0, tol_ang)
    b90 = eq(beta, 90.0, tol_ang)
    g90 = eq(gamma, 90.0, tol_ang)

    ab = eq(a, b, tol_len)
    bc = eq(b, c, tol_len)
    ac = eq(a, c, tol_len)

    if ab and a90 and b90 and eq(gamma, 120.0, tol_ang):
        return "hexagonal"
    if ab and bc and a90 and b90 and g90:
        return "cubic"
    if ab and (not ac) and a90 and b90 and g90:
        return "tetragonal"
    if a90 and b90 and g90:
        return "orthorhombic"
    if ab and bc and eq(alpha, beta, tol_ang) and eq(beta, gamma, tol_ang) and (not a90):
        return "rhombohedral"
    if (a90 and b90 and (not g90)) or (a90 and g90 and (not b90)) or (b90 and g90 and (not a90)):
        return "monoclinic"
    return "triclinic"


def hex_path() -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    P = {
        "Gamma": (0.0, 0.0, 0.0),
        "M": (0.0, 0.5, 0.0),
        "K": (-1.0 / 3.0, 2.0 / 3.0, 0.0),
        "K'": (1.0 / 3.0, 1.0 / 3.0, 0.0),
        "A": (0.0, 0.0, 0.5),
    }
    path = [["Gamma", "M", "K", "Gamma", "K'", "M", "Gamma", "A"]]
    return P, path


def cubic_path() -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    P = {
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.0, 0.5, 0.0),
        "M": (0.5, 0.5, 0.0),
        "R": (0.5, 0.5, 0.5),
    }
    path = [["Gamma", "X", "M", "Gamma", "R", "X"], ["R", "M"]]
    return P, path


def tetragonal_path() -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    P = {
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.0, 0.5, 0.0),
        "M": (0.5, 0.5, 0.0),
        "Z": (0.0, 0.0, 0.5),
        "R": (0.0, 0.5, 0.5),
        "A": (0.5, 0.5, 0.5),
    }
    path = [["Gamma", "X", "M", "Gamma", "Z", "R", "A", "Z"], ["X", "R"], ["M", "A"]]
    return P, path


def orthorhombic_path() -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    P = {
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.5, 0.0, 0.0),
        "Y": (0.0, 0.5, 0.0),
        "Z": (0.0, 0.0, 0.5),
        "S": (0.5, 0.5, 0.0),
        "T": (0.0, 0.5, 0.5),
        "U": (0.5, 0.0, 0.5),
        "R": (0.5, 0.5, 0.5),
    }
    path = [["Gamma", "X", "S", "Y", "Gamma", "Z", "U", "R", "T", "Z"], ["X", "U"], ["Y", "T"], ["S", "R"]]
    return P, path


def bravais_default_path(lattice: np.ndarray) -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])

    def angle(u, v):
        return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)))

    alpha = angle(lattice[1], lattice[2])
    beta = angle(lattice[0], lattice[2])
    gamma = angle(lattice[0], lattice[1])

    sys_ = guess_lattice_system(a, b, c, alpha, beta, gamma)

    if sys_ == "hexagonal":
        return hex_path()
    if sys_ == "cubic":
        return cubic_path()
    if sys_ == "tetragonal":
        return tetragonal_path()
    if sys_ == "orthorhombic":
        return orthorhombic_path()

    P = {"Gamma": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.0), "Y": (0.0, 0.5, 0.0), "Z": (0.0, 0.0, 0.5)}
    path = [["Gamma", "X", "Gamma", "Y", "Gamma", "Z"]]
    return P, path


def one_d_path(axis: str = "c") -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    axis = axis.lower()
    if axis not in ("a", "b", "c", "x", "y", "z"):
        raise ValueError("axis must be one of a,b,c (or x,y,z).")
    if axis in ("a", "x"):
        Z = (0.5, 0.0, 0.0)
    elif axis in ("b", "y"):
        Z = (0.0, 0.5, 0.0)
    else:
        Z = (0.0, 0.0, 0.5)
    P = {"Gamma": (0.0, 0.0, 0.0), "Z": Z}
    path = [["Gamma", "Z"]]
    return P, path


def parse_manual_kpath_file(path: str) -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    """
    Very simple manual format:
      %block BandPoints
      Gamma 0.0 0.0 0.0
      X     0.5 0.0 0.0
      ...
      %endblock BandPoints

      %block BandPath
      Gamma X M Gamma
      X R
      %endblock BandPath
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    points: Dict[str, Tuple[float, float, float]] = {}
    paths: List[List[str]] = []
    mode = None
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if low == "%block bandpoints":
            mode = "points"
            continue
        if low == "%endblock bandpoints":
            mode = None
            continue
        if low == "%block bandpath":
            mode = "path"
            continue
        if low == "%endblock bandpath":
            mode = None
            continue
        if mode == "points":
            toks = s.split()
            if len(toks) != 4:
                raise ValueError(f"Invalid BandPoints line: {s}")
            lab = toks[0]
            points[lab] = (float(toks[1]), float(toks[2]), float(toks[3]))
        elif mode == "path":
            toks = s.split()
            if len(toks) < 2:
                raise ValueError(f"Invalid BandPath line: {s}")
            paths.append(toks)

    if not points or not paths:
        raise ValueError("Manual k-path file must contain both %block BandPoints and %block BandPath")
    return points, paths


def get_band_path(
    bandpath: str,
    lattice: np.ndarray,
    species: List[str],
    cart_coords: np.ndarray,
    *,
    axis: str = "c",
    use_seekpath: bool = False,
    manual_kpath_file: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[float, float, float]], List[List[str]]]:
    if bandpath == "1d":
        return one_d_path(axis)

    elif bandpath == "seekpath":
        if use_seekpath:
            try:
                return try_seekpath_kpath(lattice, species, cart_coords)
            except Exception as e:
                print("[WARN] seekpath failed; using Bravais fallback path.")
                print(f"[WARN] reason: {e}")
        else:
            print("[WARN] --bandpath seekpath requested without --use-seekpath; using Bravais fallback path.")
        return bravais_default_path(lattice)

    elif bandpath == "hex":
        return hex_path()

    elif bandpath == "manual":
        if not manual_kpath_file:
            raise ValueError("--bandpath manual requires --manual-kpath-file")
        return parse_manual_kpath_file(manual_kpath_file)

    else:
        raise ValueError(f"Unknown bandpath option: {bandpath}")


def try_seekpath_kpath(lattice: np.ndarray, species: List[str], cart_coords: np.ndarray):
    try:
        import seekpath  # type: ignore
    except Exception as e:
        raise RuntimeError("seekpath is not available") from e

    f = frac_coords(cart_coords, lattice) % 1.0

    uniq: Dict[str, int] = {}
    nums: List[int] = []
    n = 1
    for s in species:
        if s not in uniq:
            uniq[s] = n
            n += 1
        nums.append(uniq[s])
    nums_a = np.array(nums, int)

    cell = (lattice, f, nums_a)
    sp = seekpath.get_path(cell)
    point_coords = sp["point_coords"]
    path_labels = sp["path"]

    grouped: List[List[str]] = []
    current: List[str] = []
    for a, b in path_labels:
        if not current:
            current = [a, b]
        else:
            if current[-1] == a:
                current.append(b)
            else:
                grouped.append(current)
                current = [a, b]
    if current:
        grouped.append(current)

    P: Dict[str, Tuple[float, float, float]] = {}
    for k, v in point_coords.items():
        lab = "Gamma" if k.upper() == "GAMMA" else k
        P[lab] = tuple(float(x) for x in v)

    new_grouped: List[List[str]] = []
    for seq in grouped:
        new_seq = ["Gamma" if s.upper() == "GAMMA" else s for s in seq]
        new_grouped.append(new_seq)

    return P, new_grouped


def format_siesta_bandlines(points: Dict[str, Tuple[float, float, float]],
                             paths: List[List[str]],
                             nseg: int = 40) -> str:
    out: List[str] = []
    out.append("BandLinesScale  ReciprocalLatticeVectors\n")
    out.append("%block BandLines\n")
    for seq in paths:
        for lab in seq:
            k = points[lab]
            out.append(f"{nseg:4d}  {k[0]: .8f} {k[1]: .8f} {k[2]: .8f}   {lab}\n")
    out.append("%endblock BandLines\n")
    return "".join(out)


def insert_after_kgrid_end(fdf_lines: List[str], insert_text: str) -> List[str]:
    newlines: List[str] = []
    inserted = False
    for ln in fdf_lines:
        newlines.append(ln)
        if (not inserted) and ("%endblock kgrid_Monkhorst_Pack" in ln):
            newlines.append("\n" + insert_text + "\n")
            inserted = True
    if not inserted:
        newlines.append("\n" + insert_text + "\n")
    return newlines


def make_mode(top: Path, label: str, base_fdf: Path, mode: str, args, parent: Optional[Path] = None):
    dst = (parent / mode.upper()) if parent else (top / mode.upper())

    copy_base_fdf(base_fdf, dst, label)
    stage_common_files(top, dst, label, mode=args.stage_common)

    src = dst / f"{label}.fdf"
    out = dst / f"{label}.fdf"

    lines = src.read_text().splitlines()
    patch_single_point(lines)

    nelec = None
    nbands = None
    # NumberOfBands must be raised whenever we're building the BANDS fdf
    # (a real band-path block is always written now, with or without
    # --carfile - see below) or fatbands are requested, regardless of
    # whether a .car file happens to be given.
    need_nbands = (mode == "bands") or args.set_nbands_from_out or args.fatbands
    if need_nbands:
        try:
            outfile = find_existing_outfile(label, top)
            nelec = read_nelec_from_out(outfile)
            nbands = get_number_of_bands_from_out(label, top, args.extra_bands)
            if mode != "bands":
                upsert_key(lines, "NumberOfBands", f"NumberOfBands = {nbands}")
                print(f"NumberOfBands = {nbands}")
            # for mode == "bands", NumberOfBands is printed/inserted together
            # with the band-path block below (always built now, car-file or not)
        except Exception as e:
            if args.strict:
                raise
            print(f"[WARN] NumberOfBands auto-set skipped: {e}")

    if mode == "bands":
        if args.fatbands:
            if nelec is None:
                try:
                    nelec = read_nelec_from_out(top / f"{label}.out")
                except Exception as e:
                    if args.strict:
                        raise
                    print(f"[WARN] fatbands enabled but Nelec not found: {e}")
                    nelec = None

            nbands_val = None
            for ln in lines:
                m = re.match(r"(?i)^\s*NumberOfBands\s*(?:=\s*)?(\d+)", ln)
                if m:
                    nbands_val = int(m.group(1))
                    break
            if nbands_val is None:
                nbands_val = 999999

            if nelec is not None:
                ensure_fatbands_options(
                    lines,
                    enable=True,
                    nwindow=args.ef_window,
                    nelec=nelec,
                    number_of_bands=nbands_val,
                )

                n_ef = estimate_fermi_band_index(nelec)
                bmin = max(1, n_ef - args.ef_window)
                bmax = min(nbands_val, n_ef + args.ef_window)
                print(f"--ef-window = {args.ef_window}  (≈ {2*args.ef_window+1} bands around EF)")
                print(f"EF band index (heuristic) = {n_ef}")
                print(f"Wfs.band.min = {bmin}")
                print(f"Wfs.band.max = {bmax}")

        # Band path (BandLines block): a .car file is genuinely required only
        # for --bandpath seekpath --use-seekpath (needs full atomic positions
        # for symmetry detection). Every other bandpath choice (the default
        # '1d', 'hex', 'manual', or 'seekpath' without --use-seekpath, which
        # falls back to a Bravais-lattice path) only needs the lattice, which
        # is read directly from this fdf's own LatticeVectors/LatticeParameters
        # - no separate .car file needed. This also means NumberOfBands (below)
        # no longer silently depends on --carfile being supplied.
        if args.carfile:
            st = read_car(args.carfile)
            lattice, species, cart_coords = st.lattice, st.species, st.cart_coords
        else:
            if args.bandpath == "seekpath" and args.use_seekpath:
                raise SystemExit(
                    "ERROR: --bandpath seekpath --use-seekpath needs full atomic positions "
                    "(symmetry detection) - provide --carfile, or drop --use-seekpath to fall "
                    "back to a lattice-only Bravais path."
                )
            lattice = read_lattice_from_fdf_text(_resolve_fdf_includes(base_fdf))
            species, cart_coords = [], np.zeros((0, 3))

        points, paths = get_band_path(
            args.bandpath,
            lattice,
            species,
            cart_coords,
            axis=args.axis,
            use_seekpath=args.use_seekpath,
            manual_kpath_file=args.manual_kpath_file,
        )

        bandblock = format_siesta_bandlines(points, paths, nseg=args.nseg)

        if nbands is not None:
            print(f"NumberOfBands = {nbands}\n")
        print(bandblock.rstrip())

        insert_text = ""
        if nbands is not None:
            insert_text += f"NumberOfBands = {nbands}\n\n"
        insert_text += bandblock
        lines = insert_after_kgrid_end(lines, insert_text)

    if mode == "dos":
        patch_kgrid_for_dos(lines)
        ensure_dos_options(lines, args.dos_emin, args.dos_emax, args.dos_broad, args.dos_npts)
        ensure_pdos_block(lines, args.pdos, args.dos_emin, args.dos_emax, args.dos_broad, args.dos_npts)

    out.write_text("\n".join(lines) + "\n")


def make_all(top: Path, label: str, base_fdf: Path, args):
    root = top / "Band-DOS"
    root.mkdir(exist_ok=True)
    make_mode(top, label, base_fdf, "bands", args, root)
    make_mode(top, label, base_fdf, "dos", args, root)


def add_genfdf_subparser(subparsers):
    ap = subparsers.add_parser(
        "genfdf",
        help="Generate Band-DOS/{BANDS,DOS} dirs + patched fdf (BandLines, fatbands options)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " -------------------------------------------------------------------------------------------------- \n"
        " python -m CCpy.SIESTA.siesta_band genfdf --mode all \\\n"
        "        --fatbands --ef-window 15 --bandpath 1d --axis c --carfile CAR_POSCAR/SWNT7-6W.car \n"
        " -------------------------------------------------------------------------------------------------- \n"
        " Note: this sub-command only writes files (fdf + directories); it does not submit anything to\n"
        " SLURM. Use CCpySIESTABandSubmit.py (or the 'pipeline' sub-command from inside an sbatch job)\n"
        " to actually run SIESTA. \n"
        " -------------------------------------------------------------------------------------------------- \n"
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--mode", choices=["bands", "dos", "all"], default="all")

    ap.add_argument("--dos-emin", type=float, default=-25.0)
    ap.add_argument("--dos-emax", type=float, default=25.0)
    ap.add_argument("--dos-broad", type=float, default=0.05)
    ap.add_argument("--dos-npts", type=int, default=2000)
    add_bool_flag(ap, "--pdos", default=True)

    ap.add_argument("--stage-common", choices=["copy", "symlink"], default="copy")

    ap.add_argument("--carfile", default=None, help="Materials Studio .car file for BandLines")
    ap.add_argument("--nseg", type=int, default=40, help="points per segment in BandLines (default 40)")
    ap.add_argument("--bandpath", choices=["seekpath", "1d", "hex", "manual"], default="1d")
    ap.add_argument("--axis", default=None,
                     help="For 1d bandpath: a/b/c. Default: auto-detect from the base fdf's "
                          "kgrid_Monkhorst_Pack (whichever axis has >1 k-point), falls back to "
                          "'c' if that block is missing/ambiguous.")
    ap.add_argument("--use-seekpath", action="store_true")
    ap.add_argument("--manual-kpath-file", default=None)

    ap.add_argument("--set-nbands-from-out", action="store_true")
    ap.add_argument("--extra-bands", type=int, default=50)
    ap.add_argument("--fatbands", action="store_true")
    ap.add_argument("--ef-window", type=int, default=15)
    ap.add_argument("--strict", action="store_true")
    ap.set_defaults(func=cmd_genfdf)
    return ap


def cmd_genfdf(args):
    top = Path.cwd()
    label = args.label or auto_detect_label(top)
    base_fdf = top / f"{label}.fdf"
    if not base_fdf.exists():
        raise SystemExit(f"ERROR: base fdf not found: {base_fdf}")

    if args.bandpath == "1d" and args.axis is None:
        detected = detect_periodic_axis_from_kgrid(base_fdf)
        if detected is not None:
            print(f"[info] --axis not given: auto-detected '{detected}' from "
                  f"{base_fdf.name}'s kgrid_Monkhorst_Pack (periodic axis)")
            args.axis = detected
        else:
            print(f"[warn] --axis not given and could not auto-detect a unique periodic "
                  f"axis from {base_fdf.name}'s kgrid_Monkhorst_Pack; falling back to 'c'. "
                  f"Pass --axis a/b/c explicitly if this is wrong.")
            args.axis = "c"

    if args.mode == "all":
        make_all(top, label, base_fdf, args)
    else:
        # Always nest under Band-DOS/ (not just for --mode all) - callers
        # like cmd_pipeline's bands_dir/dos_dir always expect
        # top/Band-DOS/{BANDS,DOS}, regardless of which single mode was
        # requested here.
        root = top / "Band-DOS"
        root.mkdir(exist_ok=True)
        make_mode(top, label, base_fdf, args.mode, args, parent=root)


# =============================================================================
# ============================  fatband  =====================================
# (from siesta_FatBand.py)
# =============================================================================

@dataclass(frozen=True)
class Moiety:
    name: str
    sel_raw: str
    indices_1based: List[int]


def run_cmd(cmd: Sequence[str], dry_run: bool = False, check: bool = True) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=check)


def ensure_symlink_wfsx(system_label: str, dry_run: bool = False) -> None:
    target = Path(f"{system_label}.bands.WFSX")
    link = Path(f"{system_label}.WFSX")

    if not target.exists():
        print(f"[warn] {target} not found. Skipping WFSX symlink step.")
        return

    if link.exists() or link.is_symlink():
        if dry_run:
            print(f"[dry-run] would remove existing {link}")
        else:
            link.unlink()

    if dry_run:
        print(f"[dry-run] would symlink {link} -> {target}")
        return

    os.symlink(target, link)
    print(f"[ok] symlinked {link} -> {target}")


def find_default_fdf_in_cwd() -> Path:
    fdfs = sorted(Path.cwd().glob("*.fdf"))
    if not fdfs:
        raise FileNotFoundError("No .fdf found in current directory. Provide it explicitly.")
    if len(fdfs) == 1:
        return fdfs[0]
    newest = max(fdfs, key=lambda p: p.stat().st_mtime)
    print(f"[warn] multiple .fdf found; using newest: {newest.name}")
    return newest


def get_system_label_from_fdf(fdf_path: Path) -> str:
    return fdf_path.stem


# Atomic number -> element symbol, for interpreting SIESTA's
# %block ChemicalSpeciesLabel (idx, atomic_number, label). Using the
# atomic_number column is more robust than parsing the label string, since
# labels are often custom basis-set tags (e.g. "Ti_pbe", "O2") rather than
# plain element symbols. SIESTA also allows negative Z for "ghost" atoms
# (same basis, no pseudopotential) - abs() recovers the real element.
ATOMIC_NUMBER_TO_SYMBOL: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh",
    46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe",
    55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu",
    64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
    87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu",
}

_BOHR_TO_ANG = 0.52917721067


def _fdf_norm(s: str) -> str:
    """
    Normalize an fdf keyword/label the way SIESTA's own fdf reader (libFDF)
    does: case-insensitive, and '_', '-', '.' are not significant, so
    'ChemicalSpeciesLabel' / 'Chemical_Species_Label' / 'chemical-species.label'
    are all the same key. Matching on exact spelling (as a plain regex would)
    silently misses real fdfs that use one of the underscored/dashed spelling
    conventions, so all key lookups below go through this normalization.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fdf_scalar(text: str, key: str) -> Optional[str]:
    """Remainder of a simple 'Key value ...' fdf line, matched fdf-style (see _fdf_norm)."""
    target = _fdf_norm(key)
    for ln in text.splitlines():
        s = ln.split("#", 1)[0].strip()
        if not s:
            continue
        parts = s.split(None, 1)
        if not parts:
            continue
        if _fdf_norm(parts[0]) == target:
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def _fdf_block(text: str, name: str) -> Optional[List[str]]:
    """Non-comment, non-blank lines inside %block NAME ... %endblock NAME (fdf-style name match)."""
    target = _fdf_norm(name)
    lines = text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        s = ln.split("#", 1)[0].strip()
        m = re.match(r"(?i)^%block\s+(\S+)", s)
        if m and _fdf_norm(m.group(1)) == target:
            start = i
            continue
        if start is not None:
            m2 = re.match(r"(?i)^%endblock\b(?:\s+(\S+))?", s)
            if m2 and (m2.group(1) is None or _fdf_norm(m2.group(1)) == target):
                end = i
                break
    if start is None or end is None:
        return None
    out = []
    for ln in lines[start + 1:end]:
        s = ln.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _fdf_length_unit_to_ang(unit: str) -> float:
    u = unit.strip().lower()
    if u.startswith("ang"):
        return 1.0
    if u.startswith("bohr") or u == "b":
        return _BOHR_TO_ANG
    raise ValueError(f"Unknown length unit in fdf: {unit!r}")


def read_lattice_from_fdf_text(text: str) -> np.ndarray:
    """%block LatticeVectors (most common) or %block LatticeParameters, scaled by LatticeConstant."""
    lc_line = _fdf_scalar(text, "LatticeConstant")
    if lc_line is None:
        raise ValueError("fdf has no 'LatticeConstant' line")
    lc_parts = lc_line.split()
    lc_ang = float(lc_parts[0]) * _fdf_length_unit_to_ang(lc_parts[1] if len(lc_parts) > 1 else "Bohr")

    vec_lines = _fdf_block(text, "LatticeVectors")
    if vec_lines is not None:
        rows = [list(map(float, ln.split()[:3])) for ln in vec_lines[:3]]
        return np.array(rows, dtype=float) * lc_ang

    param_lines = _fdf_block(text, "LatticeParameters")
    if param_lines is not None:
        a, b, c, alpha, beta, gamma = map(float, param_lines[0].split()[:6])
        return lattice_from_abc_angles(a * lc_ang, b * lc_ang, c * lc_ang, alpha, beta, gamma)

    raise ValueError("fdf has neither %block LatticeVectors nor %block LatticeParameters")


def read_species_table_from_fdf_text(text: str) -> Dict[int, str]:
    """%block ChemicalSpeciesLabel -> {species index (as used in the coordinates block): symbol}."""
    lines = _fdf_block(text, "ChemicalSpeciesLabel")
    if lines is None:
        raise ValueError("fdf has no %block ChemicalSpeciesLabel")
    table: Dict[int, str] = {}
    for ln in lines:
        parts = ln.split()
        idx = int(parts[0])
        z = abs(int(float(parts[1])))
        table[idx] = ATOMIC_NUMBER_TO_SYMBOL.get(z, parts[2] if len(parts) > 2 else f"Z{z}")
    return table


def _resolve_fdf_includes(fdf_path: Path, _seen: Optional[set] = None) -> str:
    """
    SIESTA fdf files can split settings across files with '%include other.fdf'
    (relative to the including file's directory) - common for keeping the
    geometry/species blocks in a separate file from BandLines/DOS/etc.
    settings. Recursively inline all %include'd content (depth-first, in
    place) so block/scalar lookups below see the fully-assembled input,
    the same way SIESTA itself does when it reads the file.
    """
    _seen = _seen or set()
    fdf_path = Path(fdf_path).resolve()
    if fdf_path in _seen:
        return ""  # guard against include cycles
    _seen.add(fdf_path)

    text = fdf_path.read_text(errors="ignore")

    def _sub(m: "re.Match") -> str:
        inc_name = m.group(1).strip().strip("'\"")
        inc_path = (fdf_path.parent / inc_name)
        if not inc_path.exists():
            print(f"[warn] %include target not found, skipping: {inc_path}")
            return ""
        return _resolve_fdf_includes(inc_path, _seen)

    return re.sub(r"(?im)^\s*%include\s+(\S+)\s*$", _sub, text)


def read_atoms_from_fdf(fdf_path: Path) -> List[str]:
    return read_geometry_from_fdf(fdf_path)[0]


def read_geometry_from_fdf(fdf_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Parse atomic species, Cartesian coordinates (N,3, Angstrom) and the
    lattice (3,3, Angstrom) directly from the fdf's standard SIESTA blocks
    (LatticeVectors/LatticeParameters, ChemicalSpeciesLabel,
    AtomicCoordinatesAndAtomicSpecies + AtomicCoordinatesFormat). Follows
    '%include' directives, so it also works when geometry is split into a
    separate included file.

    Implemented with plain regex/stdlib (no sisl) so this - and everything
    that depends on it (fatband's element/moiety validation) - runs under
    any Python that already has numpy, without needing a separate conda
    env just for fdf geometry IO.
    """
    text = _resolve_fdf_includes(Path(fdf_path))
    lattice = read_lattice_from_fdf_text(text)
    species_table = read_species_table_from_fdf_text(text)

    coord_lines = _fdf_block(text, "AtomicCoordinatesAndAtomicSpecies")
    if coord_lines is None:
        raise ValueError("fdf has no %block AtomicCoordinatesAndAtomicSpecies")

    fmt = (_fdf_scalar(text, "AtomicCoordinatesFormat") or "NotScaledCartesianBohr").strip().lower()

    raw = []
    species: List[str] = []
    for ln in coord_lines:
        parts = ln.split()
        x, y, z = map(float, parts[:3])
        sp_idx = int(parts[3])
        raw.append([x, y, z])
        species.append(species_table.get(sp_idx, f"Sp{sp_idx}"))
    raw_arr = np.array(raw, dtype=float)

    if fmt in ("notscaledcartesianbohr", "bohr"):
        coords = raw_arr * _BOHR_TO_ANG
    elif fmt in ("notscaledcartesianang", "notscaledcartesianangstrom", "ang", "angstrom"):
        coords = raw_arr
    elif fmt == "scaledcartesian":
        lc_line = _fdf_scalar(text, "LatticeConstant")
        lc_parts = lc_line.split()
        lc_ang = float(lc_parts[0]) * _fdf_length_unit_to_ang(lc_parts[1] if len(lc_parts) > 1 else "Bohr")
        coords = raw_arr * lc_ang
    elif fmt in ("fractional", "scaledbylatticevectors"):
        coords = raw_arr @ lattice
    else:
        raise ValueError(f"Unsupported AtomicCoordinatesFormat: {fmt!r}")

    return species, coords, lattice


def format_atoms_multicol(atoms: List[str], ncols: Optional[int] = None) -> str:
    entries = [f"{i+1:5d} {atoms[i]:<2s}" for i in range(len(atoms))]

    if ncols is None:
        term_w = shutil.get_terminal_size((120, 20)).columns
        col_w = max(len(e) for e in entries) + 2
        ncols = max(1, min(8, term_w // col_w))

    rows = (len(entries) + ncols - 1) // ncols
    lines = []
    for r in range(rows):
        row_items = []
        for c in range(ncols):
            idx = r + c * rows
            if idx < len(entries):
                row_items.append(entries[idx])
        lines.append("  ".join(row_items))
    return "\n".join(lines)


def normalize_elements_arg(elements_arg: Optional[str], present_atoms: List[str]) -> List[str]:
    present_unique = sorted(set(present_atoms))
    if elements_arg is None or elements_arg.strip().lower() in {"all", "*"}:
        return present_unique

    wanted = [e.strip() for e in elements_arg.split(",") if e.strip()]
    present_set = set(present_unique)
    filtered = [e for e in wanted if e in present_set]
    missing = [e for e in wanted if e not in present_set]

    if missing:
        print(f"[warn] elements not found in structure and will be ignored: {missing}")
    if not filtered:
        raise ValueError("No valid elements selected (none present in the structure).")
    return filtered


def parse_index_selection(sel: str) -> List[int]:
    sel = sel.strip()
    if not sel:
        raise ValueError("Empty atom index selection.")

    out = set()
    parts = [p.strip() for p in sel.split(",") if p.strip()]
    for p in parts:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if a <= 0 or b <= 0:
                raise ValueError(f"Atom indices must be positive (got {p}).")
            if b < a:
                a, b = b, a
            for i in range(a, b + 1):
                out.add(i)
        else:
            if not re.fullmatch(r"\d+", p):
                raise ValueError(f"Bad atom selection token: {p}")
            i = int(p)
            if i <= 0:
                raise ValueError(f"Atom indices must be positive (got {p}).")
            out.add(i)

    return sorted(out)


def parse_moiety_args(items: Sequence[str]) -> List[Moiety]:
    moieties: List[Moiety] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--moiety expects NAME:IDXSEL, got: {item}")
        name, sel = item.split(":", 1)
        name = name.strip()
        sel = sel.strip()
        if not name:
            raise ValueError(f"Empty moiety name in: {item}")
        if not sel:
            raise ValueError(f"Empty moiety selection in: {item}")

        indices = parse_index_selection(sel)
        moieties.append(Moiety(name=name, sel_raw=sel, indices_1based=indices))
    return moieties


def validate_moieties(moieties: List[Moiety], natoms: int) -> None:
    for m in moieties:
        bad = [i for i in m.indices_1based if i < 1 or i > natoms]
        if bad:
            raise ValueError(
                f"Moiety '{m.name}' has out-of-range indices (1..{natoms}): "
                f"{bad[:20]}{'...' if len(bad) > 20 else ''}"
            )


def chunk_indices_for_mpr(indices: List[int]) -> List[str]:
    """
    This fat build doesn't accept range tokens like '1-508' for atom subsets,
    so we expand to explicit indices - but every projection block in the
    DOS-style .mpr file is exactly TWO lines (name, then one values line):
    `fat` reads one physical line as the whole atom-index subset and then
    expects the NEXT physical line to be the next projection's name. If a
    moiety's index list is wrapped across multiple lines, `fat` misreads the
    2nd+ continuation lines as bogus projection names/values (visible as
    spurious 'fatbands.448 449 450 ... .EIGFAT' files), silently corrupting
    or dropping that moiety's real fatband weights and shifting every
    projection defined after it. So: always return exactly ONE line with
    every index, no matter how long - never wrap.
    """
    return [" ".join(str(i) for i in indices)] if indices else []


def write_fatbands_mpr_dos_style(
    out_path: Path,
    system_label: str,
    element_projs: List[str],
    moiety_projs: List[Moiety],
) -> List[str]:
    """
    DOS-style jobfile for fat:
      line1: SystemLabel
      line2: DOS
      repeating blocks:
        proj_name
        subset-of-AO (element symbol OR atom index list)
      terminator: ----
    """
    proj_names: List[str] = []
    lines: List[str] = []
    lines.append(system_label)
    lines.append("DOS")

    for el in element_projs:
        proj = f"{el}-orbitals"
        proj_names.append(proj)
        lines.append(proj)
        lines.append(el)

    for m in moiety_projs:
        proj = f"{m.name}-orbitals"
        proj_names.append(proj)
        lines.append(proj)
        idx_lines = chunk_indices_for_mpr(m.indices_1based)
        lines.extend(idx_lines)
        if idx_lines and len(idx_lines[0]) > 4000:
            print(f"[warn] moiety '{m.name}' index line is {len(idx_lines[0])} chars "
                  f"({len(m.indices_1based)} atoms) - if its EIGFAT/plot ends up "
                  f"missing or wrong, this line may be getting truncated by `fat`; "
                  f"check {out_path.name} and the resulting *.EIGFAT atom count.")

    lines.append("----")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path} (DOS-style) with {len(proj_names)} projections")
    return proj_names


def cleanup_bad_eigfat(mpr_stem: str, keep_projs: List[str], dry_run: bool = False) -> None:
    """
    Safety net only: delete spurious files like
      'fatbands.448 449 450 ... .EIGFAT'
    which appeared when a moiety's atom-index list was wrapped across
    multiple physical lines in the .mpr file (fat misread the continuation
    line(s) as a new projection). write_fatbands_mpr_dos_style() now always
    writes each projection's atom-index subset on a single line, so this
    should no longer trigger in normal use - kept as a defensive cleanup in
    case of any other malformed .mpr input.
    """
    keep = {f"{mpr_stem}.{p}.EIGFAT" for p in keep_projs}

    candidates: List[Path] = []
    for p in Path(".").glob(f"{mpr_stem}.*.EIGFAT"):
        name = p.name
        if name in keep:
            continue
        if (" " in name) or ("-orbitals" not in name):
            candidates.append(p)

    if not candidates:
        print("[info] no spurious EIGFAT files to delete")
        return

    print("[info] deleting spurious EIGFAT files:")
    for p in sorted(candidates, key=lambda x: x.name):
        print("  rm", repr(p.name))
        if not dry_run:
            p.unlink()


def add_fatband_subparser(subparsers):
    ap = subparsers.add_parser(
        "fatband",
        help="Run SIESTA fatbands pipeline (DOS-style .mpr): element and/or moiety decomposition",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " -------------------------------------------------------------------------------------------------------------------- \n"
        " python -m CCpy.SIESTA.siesta_band fatband --element  \n"
        " python -m CCpy.SIESTA.siesta_band fatband --list-atoms --list-cols 12  \n"
        " python -m CCpy.SIESTA.siesta_band fatband --moiety-decomp --moiety Tube:1-508 --moiety Ads:509-511  \n"
        " -------------------------------------------------------------------------------------------------------------------- \n"
        " Atom indices for --moiety are always entered manually (use --list-atoms to look them up) - \n"
        " automatic bond-based detection was tried and dropped: chemisorbed adsorbates sit close enough \n"
        " to the surface that they often end up in the same bonded fragment as the framework, making \n"
        " auto-detection unreliable. \n"
        " -------------------------------------------------------------------------------------------------------------------- \n"
    )
    ap.add_argument("fdf", nargs="?", default=None, help="Input .fdf (optional; default: auto-find *.fdf in cwd)")
    ap.add_argument("--bin-dir", default=str(DEFAULT_SIESTA_BIN_DIR), help="SIESTA bin directory containing fat, eigfat2plot")
    ap.add_argument("--prefix", default="fatbands", help="Stem for .mpr and outputs (default: fatbands)")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--list-atoms", action="store_true",
                     help="print a numbered atom/element table and exit - use this to look up the atom "
                          "indices for --moiety NAME:IDXSEL")
    ap.add_argument("--list-cols", type=int, default=None)

    ap.add_argument("--element", dest="do_element", action="store_true")
    ap.add_argument("--moiety-decomp", dest="do_moiety", action="store_true")

    ap.add_argument("--elements", default=None,
                     help="Comma-separated elements to project (e.g. 'O,Mg'). Default: all elements. Use 'all'.")

    ap.add_argument("--moiety", action="append", default=[],
                     help="Moiety as NAME:IDXSEL (1-based), e.g. 'Tube:1-508'. Repeatable. Look up indices "
                          "with --list-atoms first.")
    ap.set_defaults(func=cmd_fatband)
    return ap


def cmd_fatband(args) -> None:
    if not args.do_element and not args.do_moiety:
        args.do_element = True
        args.do_moiety = True

    if args.fdf is None:
        fdf_path = find_default_fdf_in_cwd().resolve()
        print(f"[info] using default fdf in cwd: {fdf_path.name}")
    else:
        fdf_path = Path(args.fdf).resolve()

    if not fdf_path.exists():
        raise FileNotFoundError(f"FDF not found: {fdf_path}")

    system_label = get_system_label_from_fdf(fdf_path)
    print(f"[info] SystemLabel = {system_label}")

    atoms, coords, lattice = read_geometry_from_fdf(fdf_path)
    natoms = len(atoms)
    print(f"[info] natoms = {natoms}")

    if args.list_atoms:
        print(format_atoms_multicol(atoms, ncols=args.list_cols))
        return

    element_projs: List[str] = []
    if args.do_element:
        element_projs = normalize_elements_arg(args.elements, atoms)
        print(f"[info] element projections: {element_projs}")
    else:
        print("[info] element decomposition: OFF")

    moiety_projs: List[Moiety] = []
    if args.do_moiety:
        moiety_projs = parse_moiety_args(args.moiety)
        validate_moieties(moiety_projs, natoms)
        if moiety_projs:
            for m in moiety_projs:
                print(f"[info] moiety '{m.name}': {len(m.indices_1based)} atoms (raw='{m.sel_raw}')")
        else:
            print("[warn] moiety decomposition ON but no --moiety produced any group.")
    else:
        print("[info] moiety decomposition: OFF")

    if not element_projs and not moiety_projs:
        raise ValueError("No projections selected. Enable --element and/or --moiety-decomp with valid selections.")

    ensure_symlink_wfsx(system_label, dry_run=args.dry_run)

    mpr_stem = args.prefix
    mpr_path = Path(f"{mpr_stem}.mpr")

    proj_names = write_fatbands_mpr_dos_style(
        out_path=mpr_path,
        system_label=system_label,
        element_projs=element_projs,
        moiety_projs=moiety_projs,
    )

    bin_dir = Path(args.bin_dir)
    fat_exe = (bin_dir / FAT_EXE_NAME).resolve()
    eigfat2plot_exe = (bin_dir / EIGFAT2PLOT_EXE_NAME).resolve()

    if not fat_exe.exists():
        raise FileNotFoundError(f"fat executable not found: {fat_exe}")
    if not eigfat2plot_exe.exists():
        raise FileNotFoundError(f"eigfat2plot executable not found: {eigfat2plot_exe}")

    run_cmd([str(fat_exe), mpr_stem], dry_run=args.dry_run)

    cleanup_bad_eigfat(mpr_stem, proj_names, dry_run=args.dry_run)

    for proj in proj_names:
        eigfat = Path(f"{mpr_stem}.{proj}.EIGFAT")
        outdat = Path(f"{proj}.dat")

        if not eigfat.exists():
            print(f"[warn] missing {eigfat}. (fat did not produce it; check selection)")
            continue

        print(f"[info] eigfat2plot: {eigfat} -> {outdat}")
        if args.dry_run:
            print(f"[dry-run] would write {outdat}")
            continue

        with outdat.open("w") as f:
            subprocess.run([str(eigfat2plot_exe), str(eigfat)], check=True, stdout=f)

    print("[ok] done")


# =============================================================================
# ==================  shared band-structure helpers  =========================
# (identical logic previously duplicated in siesta_bandplot.py /
#  siesta_fatbandplot.py - kept here once, used by both plot-band & plot-fatband)
# =============================================================================

def label_pretty(lab: str) -> str:
    if not lab:
        return ""
    t = lab.strip()
    if t.lower() in ("g", "ga", "gam", "gamma", "\\gamma", "gamm"):
        return "Γ"
    return t


def parse_bandlines_from_fdf(fdf_path: PathLike):
    """
    Parse %block BandLines ... %endblock BandLines
    Each line:  n  kx ky kz  Label
    """
    with Path(fdf_path).open("r", errors="ignore") as f:
        text = f.read()

    m = re.search(r"%block\s+BandLines\s*(.*?)%endblock\s+BandLines", text, flags=re.S | re.I)
    if not m:
        raise RuntimeError(f"Cannot find %block BandLines in {fdf_path}")

    pts = []
    for ln in m.group(1).splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            n = int(float(parts[0]))
            kx, ky, kz = map(float, parts[1:4])
        except ValueError:
            continue
        label = parts[4] if len(parts) >= 5 else ""
        pts.append({"n": n, "k": (kx, ky, kz), "label": label_pretty(label)})

    if len(pts) < 2:
        raise RuntimeError("BandLines has <2 points; cannot build path.")
    return pts


def detect_bands_dat_format(path: PathLike, max_lines: int = 500) -> str:
    """
    Returns:
      'block'     : blank line separates bands (each numeric line has >=2 cols)
      'multi_col' : each line has k + many energies (>=3 cols)
    """
    saw_blank_after_data = False
    numeric_lines = 0
    blank_lines = 0
    col_counts = []
    started_data = False

    with Path(path).open("r", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            s = line.strip()

            if not s:
                if started_data:
                    blank_lines += 1
                    saw_blank_after_data = True
                continue
            if s.startswith("#") or s.startswith("!"):
                continue

            parts = s.split()
            try:
                _ = [float(x) for x in parts]
            except ValueError:
                continue

            started_data = True
            numeric_lines += 1
            col_counts.append(len(parts))

    if numeric_lines == 0:
        raise RuntimeError(f"No numeric data found in {path}")

    if saw_blank_after_data and blank_lines > 0:
        return "block"

    typical = int(round(np.median(col_counts)))
    if typical >= 3:
        return "multi_col"
    return "block"


def read_gnubands_block(path: PathLike):
    """
    Block format: each band = block of lines, separated by a blank line.
    Each numeric line: k E [maybe spin/band index ...] (only first two columns used).
    """
    bands = []
    cur = []

    with Path(path).open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                if cur:
                    bands.append(np.array(cur, float))
                    cur = []
                continue
            if s.startswith("#") or s.startswith("!"):
                continue

            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                k = float(parts[0])
                e = float(parts[1])
            except ValueError:
                continue
            cur.append([k, e])

    if cur:
        bands.append(np.array(cur, float))

    if not bands:
        raise RuntimeError(f"No band blocks found in {path}")

    maxlen = max(b.shape[0] for b in bands)
    bands = [b for b in bands if b.shape[0] >= max(2, maxlen // 2)]

    ref = max(bands, key=lambda a: a.shape[0])
    kgrid = ref[:, 0]
    nk = len(kgrid)

    energies = []
    for b in bands:
        kk = b[:, 0]
        ee = b[:, 1]
        if len(ee) != nk or np.max(np.abs(kk - kgrid[:len(kk)])) > 1e-8:
            ee = np.interp(kgrid, kk, ee)
        energies.append(ee)

    energies = np.array(energies)
    return kgrid, energies


def read_gnubands_multi_col(path: PathLike):
    """
    Multi-column format: col0 = k-distance, col1.. = energies of bands.
    If a trailing integer-ish constant column exists (e.g. spin index), it is dropped.
    """
    rows = []
    with Path(path).open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            parts = s.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)

    if not rows:
        raise RuntimeError(f"No numeric rows found in {path}")

    arr = np.array(rows, float)

    if arr.shape[1] >= 3:
        last = arr[:, -1]
        if np.nanmax(np.abs(last - np.round(last))) < 1e-8 and np.nanmax(last) == np.nanmin(last):
            arr = arr[:, :-1]

    kgrid = arr[:, 0]
    energies = arr[:, 1:].T
    return kgrid, energies


def read_bands_dat_auto(path: PathLike):
    fmt = detect_bands_dat_format(path)
    if fmt == "block":
        kgrid, energies = read_gnubands_block(path)
    else:
        kgrid, energies = read_gnubands_multi_col(path)
    return fmt, kgrid, energies


def ticks_from_bandlines(bandline_pts, kgrid):
    labels = [p["label"] for p in bandline_pts]

    if len(bandline_pts) == 2:
        idxs = [0, len(kgrid) - 1]
        xticks = [float(kgrid[0]), float(kgrid[-1])]
        return xticks, labels, idxs

    expected = [0]
    cur = 0
    for i in range(1, len(bandline_pts)):
        cur += int(bandline_pts[i]["n"])
        expected.append(cur)

    last_exp = expected[-1] if expected[-1] != 0 else 1
    N = len(kgrid)
    idxs = [int(round(e / last_exp * (N - 1))) for e in expected]
    idxs[0] = 0
    idxs[-1] = N - 1
    idxs = [min(max(i, 0), N - 1) for i in idxs]

    xticks = [float(kgrid[i]) for i in idxs]
    return xticks, labels, idxs


def compute_bandgap(energies, kgrid=None, tol: float = 1e-6):
    """
    Compute fundamental bandgap from energies referenced to E_F=0.
    Returns dict with: vbm, cbm, gap, vbm_k, cbm_k, is_metal, is_direct
    """
    E = np.array(energies, float)
    below_mask = E <= tol
    above_mask = E >= -tol

    if not np.any(below_mask) or not np.any(above_mask):
        return {"vbm": np.nan, "cbm": np.nan, "gap": np.nan,
                "vbm_k": None, "cbm_k": None, "is_metal": False, "is_direct": False}

    vbm = float(np.nanmax(np.where(below_mask, E, -np.inf)))
    cbm = float(np.nanmin(np.where(above_mask, E, np.inf)))
    gap = cbm - vbm

    v_idx = np.unravel_index(np.nanargmax(np.where(below_mask, E, -np.inf)), E.shape)
    c_idx = np.unravel_index(np.nanargmin(np.where(above_mask, E, np.inf)), E.shape)
    v_k_idx = int(v_idx[1])
    c_k_idx = int(c_idx[1])

    v_k = float(kgrid[v_k_idx]) if kgrid is not None else v_k_idx
    c_k = float(kgrid[c_k_idx]) if kgrid is not None else c_k_idx

    is_metal = gap <= 1e-3
    is_direct = (v_k_idx == c_k_idx)

    return {"vbm": vbm, "cbm": cbm, "gap": gap,
            "vbm_k": v_k, "cbm_k": c_k, "is_metal": is_metal, "is_direct": is_direct}


# =============================================================================
# ============================  plot-band  ===================================
# (from siesta_bandplot.py)
# =============================================================================

def plot_bands(
    dat_path: str,
    fdf_path: str,
    out_png: str,
    pad_y: float = 0.05,
    draw_vlines: bool = True,
    lw_band: float = 1.5,
    lw_fermi: float = 1.2,
    show_gap: bool = True,
    gap_tol: float = 1e-6,
):
    fmt, kgrid, energies = read_bands_dat_auto(dat_path)
    bandline_pts = parse_bandlines_from_fdf(fdf_path)

    gap_info = compute_bandgap(energies, kgrid=kgrid, tol=gap_tol)

    e_min = float(np.nanmin(energies))
    e_max = float(np.nanmax(energies))
    yr = e_max - e_min if e_max > e_min else 1.0
    ylo = e_min - pad_y * yr
    yhi = e_max + pad_y * yr

    xticks, xlabels, idxs = ticks_from_bandlines(bandline_pts, kgrid)

    fig = plt.figure(figsize=(5, 9))
    ax = fig.add_subplot(111)

    for ib in range(energies.shape[0]):
        ax.plot(kgrid, energies[ib, :], color="blue", linewidth=lw_band)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=lw_fermi)

    if draw_vlines:
        for x in xticks:
            ax.axvline(x, color="black", linewidth=0.6, alpha=0.35)

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(float(kgrid[0]), float(kgrid[-1]))
    ax.set_ylim(ylo, yhi)

    if show_gap:
        if gap_info.get('is_metal', False):
            txt = f"Metallic (Eg ≲ 0)"
        else:
            g = gap_info['gap']
            kind = "direct" if gap_info.get('is_direct', False) else "indirect"
            txt = f"Eg = {g:.6f} eV ({kind})"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Detected bands.dat format: {fmt}")
    if gap_info.get("is_metal", False):
        print("Bandgap: metallic (Eg ≲ 0)")
    else:
        kind = "direct" if gap_info.get("is_direct", False) else "indirect"
        print(f"Bandgap: Eg = {gap_info['gap']:.6f} eV ({kind})")
        print(f"  VBM = {gap_info['vbm']:.6f} eV at k = {gap_info['vbm_k']:.8f}")
        print(f"  CBM = {gap_info['cbm']:.6f} eV at k = {gap_info['cbm_k']:.8f}")
    print(f"Saved: {out_png}")
    print(f"Y-range from bands.dat: [{e_min:.6f}, {e_max:.6f}] eV")
    print("XTicks (label, k-distance, index):")
    for lab, xt, ii in zip(xlabels, xticks, idxs):
        print(f"  {lab:>4s}  {xt: .8f}   idx={ii}")


def auto_detect_basename():
    bands_files = sorted(Path(".").glob("*.bands"))
    if not bands_files:
        raise RuntimeError("No *.bands file found in current directory. Please specify --basename.")
    if len(bands_files) > 1:
        names = ", ".join(f.name for f in bands_files)
        raise RuntimeError(f"Multiple *.bands files found: {names}. Please specify --basename.")
    return bands_files[0].stem


def run_gnubands(basename: str, emin: float, emax: float, out_dat: str, bin_dir: PathLike = DEFAULT_SIESTA_BIN_DIR):
    gnubands = str(Path(bin_dir) / GNUBANDS_EXE_NAME)
    bands_file = f"{basename}.bands"

    cmd = [gnubands, "-F", "-e", str(emin), "-E", str(emax), bands_file]

    print("Running:", " ".join(cmd))
    try:
        with open(out_dat, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(f"gnubands not found: {gnubands}")
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"gnubands failed.\nCommand: {' '.join(cmd)}\n{err}")


def add_plot_band_subparser(subparsers):
    ap = subparsers.add_parser(
        "plot-band",
        help="Run gnubands + plot band structure (with bandgap annotation)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " ----------------------------------------------------------------------------------------------- \n"
        " python -m CCpy.SIESTA.siesta_band plot-band -e -3 -E 3 --lw-band 2  \n"
        " ----------------------------------------------------------------------------------------------- \n"
    )
    ap.add_argument("--basename", default=None, help="basename for <basename>.bands (default: auto-detect)")
    ap.add_argument("-e", "--emin", type=float, required=True, help="min energy for gnubands (-e)")
    ap.add_argument("-E", "--emax", type=float, required=True, help="max energy for gnubands (-E)")

    ap.add_argument("--dat", default="bands.dat", help="output bands.dat filename (default: bands.dat)")
    ap.add_argument("--fdf", default=None, help="bands fdf file (default: <basename>.fdf)")
    ap.add_argument("--out", default=None, help="output png (default: <basename>.png)")
    ap.add_argument("--bin-dir", default=str(DEFAULT_SIESTA_BIN_DIR), help="dir containing gnubands")

    ap.add_argument("--pad-y", type=float, default=0.05)
    ap.add_argument("--no-vlines", action="store_true")
    ap.add_argument("--lw-band", type=float, default=1.0)
    ap.add_argument("--lw-fermi", type=float, default=1.2)
    ap.add_argument("--no-gap", action="store_true")
    ap.add_argument("--gap-tol", type=float, default=1e-6)
    ap.set_defaults(func=cmd_plot_band)
    return ap


def cmd_plot_band(args):
    basename = args.basename if args.basename else auto_detect_basename()
    print(f"Using basename: {basename}")
    emin = args.emin
    emax = args.emax

    fdf_file = args.fdf if args.fdf is not None else f"{basename}.fdf"
    out_png = args.out if args.out is not None else f"{basename}.png"

    run_gnubands(basename=basename, emin=emin, emax=emax, out_dat=args.dat, bin_dir=args.bin_dir)

    plot_bands(
        dat_path=args.dat,
        fdf_path=fdf_file,
        out_png=out_png,
        pad_y=args.pad_y,
        draw_vlines=(not args.no_vlines),
        lw_band=args.lw_band,
        lw_fermi=args.lw_fermi,
        show_gap=(not args.no_gap),
        gap_tol=args.gap_tol,
    )


# =============================================================================
# ============================  plot-fatband  =================================
# (from siesta_fatbandplot.py)
# =============================================================================

_FERMI_PAT_LIST = [
    re.compile(r"Fermi\s+energy\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I),
    re.compile(r"Fermi\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I),
]


def read_fermi_from_textfile(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        for line in path.read_text(errors="ignore").splitlines():
            for pat in _FERMI_PAT_LIST:
                m = pat.search(line)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        pass
    except Exception:
        return None
    return None


def autodetect_fermi_from_cwd() -> Tuple[Optional[float], Optional[Path]]:
    """
    Scan *.EIG first, then *.out in CWD and return the first EF found.
    If multiple files contain EF, choose the newest (mtime) among the matching ones.
    """
    candidates: List[Path] = []
    for pat in ("*.EIG", "*.out"):
        candidates.extend(Path(".").glob(pat))

    hits: List[Tuple[float, Path]] = []
    for p in candidates:
        ef = read_fermi_from_textfile(p)
        if ef is not None:
            hits.append((ef, p))

    if not hits:
        return None, None

    hits.sort(key=lambda t: t[1].stat().st_mtime, reverse=True)
    return hits[0][0], hits[0][1]


def get_fermi_level(basename: Optional[str], fermi_arg: Optional[float]) -> Tuple[float, Optional[str]]:
    """Return (E_F, detected_basename)."""
    if fermi_arg is not None:
        ef = float(fermi_arg)
        print(f"[info] Fermi level (from --fermi) = {ef:.6f} eV")
        return ef, basename

    if basename:
        eig = Path(f"{basename}.EIG")
        out = Path(f"{basename}.out")

        ef = read_fermi_from_textfile(eig)
        if ef is not None:
            print(f"[info] Fermi level (from {eig.name}) = {ef:.6f} eV")
            return ef, basename

        ef = read_fermi_from_textfile(out)
        if ef is not None:
            print(f"[info] Fermi level (from {out.name}) = {ef:.6f} eV")
            return ef, basename

    ef, src = autodetect_fermi_from_cwd()
    if ef is not None and src is not None:
        print(f"[info] Fermi level (auto-detected from {src.name}) = {ef:.6f} eV")
        return ef, src.stem

    print("[warn] Could not parse Fermi level from *.EIG/*.out; assuming EF = 0.0 eV")
    return 0.0, basename


def clean_fat_stem(stem: str) -> str:
    """
    'C-orbitals' -> 'C', 'Tube-orbitals' -> 'Tube'
    """
    return re.sub(r"[-_]?orbitals$", "", stem, flags=re.IGNORECASE)


def read_fat_dat_kdist(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, ndmin=2, comments="#")
    if arr.size == 0:
        return np.array([]), np.array([]), np.array([])
    if arr.shape[1] < 3:
        raise RuntimeError(f"{path} doesn't look like eigfat2plot output (need >=3 cols).")
    x = arr[:, 0].astype(float)
    E = arr[:, 1].astype(float)
    w = arr[:, 2].astype(float)
    m = np.isfinite(x) & np.isfinite(E) & np.isfinite(w)
    return x[m], E[m], w[m]


def apply_energy_window(x: np.ndarray, E: np.ndarray, w: np.ndarray,
                         emin: Optional[float], emax: Optional[float]):
    if emin is None or emax is None:
        return x, E, w
    m = (E >= emin) & (E <= emax)
    return x[m], E[m], w[m]


def plot_one(
    out_png: Path,
    fat_files: List[Path],
    *,
    ef: float,
    bands_dat: Optional[Path],
    fdf_path: Optional[Path],
    emin: Optional[float],
    emax: Optional[float],
    scale: float,
    alpha: float,
    pad_y: float,
    lw_band: float,
    lw_fermi: float,
    vlines: bool,
    show_gap: bool,
    gap_tol: float,
    fat_color: str,
    fat_cmap: Optional[str],
):
    fig = plt.figure(figsize=(6.2, 9))
    ax = fig.add_subplot(111)

    xgrid_for_limits = None
    fmt = None
    gap_info = None

    if bands_dat is not None:
        fmt, kgrid, energies = read_bands_dat_auto(bands_dat)
        xgrid_for_limits = kgrid

        for ib in range(energies.shape[0]):
            ax.plot(kgrid, energies[ib, :], linewidth=lw_band, color="blue")

        ax.axhline(0.0, linestyle="--", linewidth=lw_fermi)

        if show_gap:
            gap_info = compute_bandgap(energies, kgrid=kgrid, tol=gap_tol)
            if gap_info.get("is_metal", False):
                txt = "Metallic (Eg ≲ 0)"
            else:
                kind = "direct" if gap_info.get("is_direct", False) else "indirect"
                txt = f"Eg = {gap_info['gap']:.6f} eV ({kind})"
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )

        if fdf_path is not None and fdf_path.exists():
            try:
                bandline_pts = parse_bandlines_from_fdf(fdf_path)
                xticks, xlabels, _ = ticks_from_bandlines(bandline_pts, kgrid)
                ax.set_xticks(xticks)
                ax.set_xticklabels([label_pretty(x) for x in xlabels])
                if vlines:
                    for x in xticks:
                        ax.axvline(x, linewidth=0.7, alpha=0.35)
            except Exception as e:
                print(f"[warn] Could not parse BandLines from {fdf_path.name}: {e}")

    used = 0
    x_for_limits = []
    y_for_limits = []

    sc = None
    for fp in fat_files:
        label = fp.stem
        x, Eabs, w = read_fat_dat_kdist(fp)
        if x.size == 0:
            print(f"[warn] {fp.name}: empty. Skipping.")
            continue

        E = Eabs - ef
        x, E, w = apply_energy_window(x, E, w, emin, emax)
        if x.size == 0:
            print(f"[warn] {fp.name}: no points in energy window [{emin},{emax}] (E-EF). Skipping.")
            continue

        if np.any(w > 0):
            wclip = np.clip(w, 0.0, np.percentile(w[w > 0], 99.5))
        else:
            wclip = np.zeros_like(w)
        s = scale * wclip

        if fat_cmap:
            sc = ax.scatter(x, E, s=s, c=wclip, cmap=fat_cmap, alpha=alpha, linewidths=0.2, label=label)
        else:
            sc = ax.scatter(x, E, s=s, color=fat_color, alpha=alpha, linewidths=0.2, label=label)
        used += 1
        x_for_limits.append(x)
        y_for_limits.append(E)

    if fat_cmap and used > 0:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Fat weight")

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy − $E_F$ (eV)")

    if emin is not None and emax is not None:
        ax.set_ylim(emin, emax)
    else:
        ys = []
        if bands_dat is not None:
            _, _, energies = read_bands_dat_auto(bands_dat)
            ys.append(energies.ravel())
        if y_for_limits:
            ys.append(np.concatenate(y_for_limits))
        if ys:
            y_all = np.concatenate(ys)
            e_min = float(np.nanmin(y_all))
            e_max = float(np.nanmax(y_all))
            yr = e_max - e_min if e_max > e_min else 1.0
            ax.set_ylim(e_min - pad_y * yr, e_max + pad_y * yr)

    if xgrid_for_limits is not None and len(xgrid_for_limits) > 0:
        ax.set_xlim(float(np.nanmin(xgrid_for_limits)), float(np.nanmax(xgrid_for_limits)))
    elif x_for_limits:
        xx = np.concatenate(x_for_limits)
        ax.set_xlim(float(np.nanmin(xx)), float(np.nanmax(xx)))

    if used > 0 and len(fat_files) > 1:
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    if fmt:
        print(f"[info] bands.dat format: {fmt}")
    if gap_info and not gap_info.get("is_metal", False):
        print(f"[info] Bandgap Eg = {gap_info['gap']:.6f} eV "
              f"({'direct' if gap_info.get('is_direct', False) else 'indirect'})")
    print(f"[ok] saved: {out_png}")


def add_plot_fatband_subparser(subparsers):
    ap = subparsers.add_parser(
        "plot-fatband",
        help="Plot fatbands (eigfat2plot *.dat), optionally overlaid on bands.dat. Does NOT run gnubands.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " -------------------------------------------------------------------------------------------------------------------------------- \n"
        " python -m CCpy.SIESTA.siesta_band plot-fatband --fat-auto -e -2 -E 2 --lw-band 1 --each --fat-cmap inferno [--fat-only] \n"
        "   --basename option: shown as k-symbol labels on the x-axis \n"
        " -------------------------------------------------------------------------------------------------------------------------------- \n"
    )
    ap.add_argument("--basename", default=None,
                    help="basename for <basename>.fdf/.EIG/.out (recommended for EF + ticks).")
    ap.add_argument("-e", "--emin", type=float, default=None,
                    help="min energy window relative to EF, i.e. (E-EF)_min [eV]")
    ap.add_argument("-E", "--emax", type=float, default=None,
                    help="max energy window relative to EF, i.e. (E-EF)_max [eV]")
    ap.add_argument("--fermi", type=float, default=None,
                    help="override EF (eV). If omitted, parse from <basename>.EIG/.out or auto-detect from cwd")

    ap.add_argument("--dat", default="bands.dat",
                    help="existing bands.dat filename (default: bands.dat). Not used in --fat-only.")
    ap.add_argument("--fdf", default=None,
                    help="FDF file for BandLines ticks (default: <basename>.fdf). Not used in --fat-only.")
    ap.add_argument("--out", default=None,
                    help="overlay output png (default: <basename>_fat.png or fat_fat.png)")
    ap.add_argument("--outdir", default=".", help="output directory for --each mode (default: .)")

    ap.add_argument("--fat", nargs="*", default=None,
                    help="fat .dat files (eigfat2plot output), e.g. Tube-orbitals.dat Ads-orbitals.dat")
    ap.add_argument("--fat-auto", action="store_true", help="auto-detect fat files '*-orbitals.dat' in cwd")

    ap.add_argument("--each", action="store_true", help="generate one figure per fat file (instead of overlay)")
    ap.add_argument("--fat-only", action="store_true", help="plot fatbands only (do NOT draw band lines)")

    ap.add_argument("--scale", type=float, default=220.0, help="marker size scale (default 220)")
    ap.add_argument("--alpha", type=float, default=0.65, help="marker alpha (default 0.65)")
    ap.add_argument("--fat-color", default="orange", help="fat marker color (default: orange)")
    ap.add_argument("--fat-cmap", default=None, help="colormap for fat weights (e.g. viridis, inferno)")
    ap.add_argument("--pad-y", type=float, default=0.05, help="extra y padding when auto-scaling (default 0.05)")
    ap.add_argument("--lw-band", type=float, default=1.2, help="band line width")
    ap.add_argument("--lw-fermi", type=float, default=1.2, help="fermi line width")
    ap.add_argument("--no-vlines", action="store_true", help="do not draw vertical high-symmetry lines")

    ap.add_argument("--no-gap", action="store_true", help="do not compute/annotate bandgap (bands overlay only)")
    ap.add_argument("--gap-tol", type=float, default=1e-6,
                    help="tolerance (eV) when finding VBM/CBM around E_F=0 in bands.dat")
    ap.set_defaults(func=cmd_plot_fatband)
    return ap


def cmd_plot_fatband(args):
    if args.fat_auto:
        fat_files = [p for p in sorted(Path(".").glob("*-orbitals.dat"))]
    elif args.fat:
        fat_files = [Path(x) for x in args.fat]
    else:
        fat_files = []
    if not fat_files:
        raise RuntimeError("No fat files selected. Use --fat-auto or --fat ...")

    ef, detected_base = get_fermi_level(args.basename, args.fermi)

    bands_dat = None
    fdf_path = None
    if not args.fat_only:
        bands_dat = Path(args.dat)
        if not bands_dat.exists():
            raise RuntimeError(
                f"bands.dat not found: {bands_dat}. "
                "This sub-command does not run gnubands. Create bands.dat first (e.g. `plot-band`) "
                "or pass the correct --dat path."
            )
        if args.basename is None and args.fdf is None:
            print("[warn] No --basename/--fdf given: high-symmetry ticks may be missing.")
        fdf_path = Path(args.fdf) if args.fdf else (Path(f"{args.basename}.fdf") if args.basename else None)

    if args.basename is not None:
        name_for_out = args.basename
    elif detected_base:
        name_for_out = detected_base
    else:
        name_for_out = clean_fat_stem(fat_files[0].stem)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = "_fat" if args.fat_only else ""

    if args.each:
        for fp in fat_files:
            fat_tag = clean_fat_stem(fp.stem)
            out_png = outdir / f"{name_for_out}_{fat_tag}{suffix}.png"
            plot_one(
                out_png=out_png, fat_files=[fp], ef=ef, bands_dat=bands_dat, fdf_path=fdf_path,
                emin=args.emin, emax=args.emax, scale=args.scale, alpha=args.alpha, pad_y=args.pad_y,
                lw_band=args.lw_band, lw_fermi=args.lw_fermi, vlines=(not args.no_vlines),
                show_gap=(not args.no_gap), gap_tol=args.gap_tol, fat_color=args.fat_color, fat_cmap=args.fat_cmap,
            )
    else:
        if args.out:
            out_png = outdir / args.out
        else:
            if len(fat_files) == 1:
                fat_tag = clean_fat_stem(fat_files[0].stem)
                out_png = outdir / f"{name_for_out}_{fat_tag}{suffix}.png"
            else:
                out_png = outdir / f"{name_for_out}{suffix}.png"
        plot_one(
            out_png=out_png, fat_files=fat_files, ef=ef, bands_dat=bands_dat, fdf_path=fdf_path,
            emin=args.emin, emax=args.emax, scale=args.scale, alpha=args.alpha, pad_y=args.pad_y,
            lw_band=args.lw_band, lw_fermi=args.lw_fermi, vlines=(not args.no_vlines),
            show_gap=(not args.no_gap), gap_tol=args.gap_tol, fat_color=args.fat_color, fat_cmap=args.fat_cmap,
        )


# =============================================================================
# ============================  plot-dos  =====================================
# (from siesta_dosplot.py)
# =============================================================================

def load_siesta_dos(path: str):
    """SIESTA DOS-like file with 2 columns: E(eV) DOS."""
    E = []
    D = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                e = float(parts[0])
                d = float(parts[1])
            except ValueError:
                continue
            E.append(e)
            D.append(d)

    if not E:
        raise RuntimeError(f"No numeric DOS data found in: {path}")

    return np.array(E, float), np.array(D, float)


def read_fermi_from_bands(bands_path: str) -> float:
    """SIESTA .bands format: first line is typically the Fermi energy (eV)."""
    with open(bands_path, "r", errors="ignore") as f:
        first = f.readline().strip()
    if not first:
        raise RuntimeError(f"Failed to read first line (E_Fermi) from: {bands_path}")
    try:
        return float(first.split()[0])
    except Exception as e:
        raise RuntimeError(f"Failed to parse E_Fermi from first line of {bands_path}: {first}") from e


def gaussian_smooth(y, sigma_pts: float):
    """Gaussian smoothing via convolution. sigma_pts in points."""
    if sigma_pts is None or sigma_pts <= 0:
        return y
    sigma_pts = float(sigma_pts)
    half = int(max(3, round(3 * sigma_pts)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= np.sum(k)
    return np.convolve(y, k, mode="same")


def detect_van_hove_peaks(E, D, window_eV: float = 0.5, prom_rel: float = 0.05, max_peaks: int = 20):
    """
    Lightweight local-maximum peak detector (no SciPy).
    Returns list of dicts: {idx, E, D, prominence}
    """
    E = np.asarray(E, float)
    D = np.asarray(D, float)
    if len(E) < 3:
        return []

    dmax = float(np.nanmax(D)) if np.isfinite(np.nanmax(D)) else 0.0
    if dmax <= 0:
        return []

    dE = float(np.median(np.diff(E)))
    if dE <= 0:
        dE = float(np.mean(np.diff(E[E.size // 4: 3 * E.size // 4])))

    win = int(max(2, round(window_eV / abs(dE))))
    prom_th = prom_rel * dmax

    peaks = []
    for i in range(1, len(E) - 1):
        if not (D[i] > D[i - 1] and D[i] > D[i + 1]):
            continue

        l0 = max(0, i - win)
        r0 = min(len(E) - 1, i + win)

        left_min = float(np.nanmin(D[l0:i + 1]))
        right_min = float(np.nanmin(D[i:r0 + 1]))
        base = max(left_min, right_min)
        prom = float(D[i] - base)

        if prom >= prom_th:
            peaks.append({"idx": i, "E": float(E[i]), "D": float(D[i]), "prominence": prom})

    peaks.sort(key=lambda p: p["prominence"], reverse=True)
    return peaks[:max_peaks]


def plot_dos(E, D, out_png: str, fermi_line: float = 0.0,
             xlim=None, ylim=None, filled: bool = False,
             peaks=None, peak_lines: bool = True, peak_labels: bool = False):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(E, D, linewidth=1.5)
    if filled:
        ax.fill_between(E, D, 0.0, alpha=0.3)

    ax.axvline(fermi_line, linestyle="--", linewidth=1.0)

    if peaks and peak_lines:
        for p in peaks:
            ax.axvline(p['E'], linestyle=':', linewidth=0.9, alpha=0.8)
            if peak_labels:
                ax.text(p['E'], p['D'], f"{p['E']:.2f}", rotation=90, va='bottom', ha='center', fontsize=8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (arb. units)")

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_png}")


def add_plot_dos_subparser(subparsers):
    ap = subparsers.add_parser(
        "plot-dos",
        help="Plot SIESTA DOS file, optionally Fermi-aligned to match .bands (E_F=0), with van Hove peak detection",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " --------------------------------------------------------------------------- \n"
        "  python -m CCpy.SIESTA.siesta_band plot-dos   \n"
        " --------------------------------------------------------------------------- \n"
    )
    ap.add_argument("dos", nargs="?", default=None, help="SIESTA DOS file (default: auto-detect *.DOS)")
    ap.add_argument("--out", default=None, help="output png (default: <dos>.png)")

    ap.add_argument("--bands", default=None, help="SIESTA .bands file. Default: ../BANDS/{dos_basename}.bands")
    ap.add_argument("--no-shift", action="store_true",
                    help="Do not shift energies even if --bands is given (still draw Fermi line at 0).")

    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--filled", action="store_true", help="fill DOS area")

    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian smoothing sigma in eV (0 disables).")
    ap.add_argument("--vh", action="store_true", help="detect van Hove-like DOS peaks and report them")
    ap.add_argument("--vh-window", type=float, default=0.5, help="peak prominence window (+-eV) for local baseline")
    ap.add_argument("--vh-prom", type=float, default=0.05, help="peak prominence threshold as fraction of max(DOS)")
    ap.add_argument("--vh-max", type=int, default=20, help="maximum number of peaks to report")
    ap.add_argument("--vh-labels", action="store_true", help="label detected peaks on plot")
    ap.set_defaults(func=cmd_plot_dos)
    return ap


def cmd_plot_dos(args):
    if args.dos is None:
        dos_files = sorted(Path(".").glob("*.DOS"))
        if not dos_files:
            raise RuntimeError("No *.DOS file found in current directory. Please specify DOS file.")
        if len(dos_files) > 1:
            names = ", ".join(f.name for f in dos_files)
            raise RuntimeError(f"Multiple *.DOS files found: {names}. Please specify one.")
        args.dos = str(dos_files[0])
        print(f"Using DOS file: {args.dos}")

    if args.bands is None:
        dos_base = Path(args.dos).stem
        default_bands = Path("..") / "BANDS" / f"{dos_base}.bands"
        if default_bands.exists():
            args.bands = str(default_bands)
            print(f"Using default bands file: {args.bands}")

    out_png = args.out if args.out else (args.dos + ".png")

    E, D = load_siesta_dos(args.dos)

    if args.bands:
        ef = read_fermi_from_bands(args.bands)
        if not args.no_shift:
            E = E - ef
            print(f"Shifted DOS energies by -E_Fermi (E_Fermi={ef:.6f} eV) to match bands reference (E_F=0).")
        else:
            print(f"--no-shift: using DOS energies as-is (E_Fermi from bands={ef:.6f} eV only for info).")

    peaks = None
    if args.smooth_sigma and args.smooth_sigma > 0:
        dE = float(np.median(np.diff(E)))
        if dE == 0:
            dE = float(np.mean(np.diff(E)))
        sigma_pts = abs(args.smooth_sigma / dE)
        D = gaussian_smooth(D, sigma_pts)
        print(f"Applied Gaussian smoothing: sigma={args.smooth_sigma:.6f} eV (~{sigma_pts:.2f} pts)")

    if args.vh:
        peaks = detect_van_hove_peaks(E, D, window_eV=args.vh_window, prom_rel=args.vh_prom, max_peaks=args.vh_max)
        if peaks:
            print("Detected van Hove-like DOS peaks (sorted by prominence):")
            for p in peaks:
                print(f"  E = {p['E']:+.6f} eV   DOS = {p['D']:.6e}   prom = {p['prominence']:.6e}")
        else:
            print("No van Hove-like peaks detected (try smaller --vh-prom or adjust --vh-window / smoothing).")

    xlim = None
    if args.xmin is not None or args.xmax is not None:
        xmin = args.xmin if args.xmin is not None else float(np.min(E))
        xmax = args.xmax if args.xmax is not None else float(np.max(E))
        xlim = (xmin, xmax)

    ylim = None
    if args.ymin is not None or args.ymax is not None:
        ymin = args.ymin if args.ymin is not None else 0.0
        ymax = args.ymax if args.ymax is not None else float(np.max(D))
        ylim = (ymin, ymax)

    plot_dos(E, D, out_png=out_png, fermi_line=0.0, xlim=xlim, ylim=ylim, filled=args.filled,
              peaks=peaks, peak_lines=bool(args.vh and peaks), peak_labels=args.vh_labels)


# =============================================================================
# ============================  pipeline  =====================================
# End-to-end driver equivalent to WorkFlow_FatBandDOS.sh, calling the
# sub-commands above in sequence.  --steps lets you run a subset only.
# =============================================================================

PIPELINE_STEPS = ["genfdf", "band-calc", "fatband", "plot-band", "plot-fatband", "dos-calc", "plot-dos"]
# NOTE: there is deliberately no "fatband-calc" step. siesta_FatBand.py (the
# `fatband` step) only post-processes the WFSX/HSX/ORB_INDX already produced
# by "band-calc" - the fdf does not change in between, so re-running SIESTA
# a second time before plotting would just reproduce the same result at the
# cost of wall-clock time. The original WorkFlow_FatBandDOS.sh did run SIESTA
# twice here; that duplicate run was dropped in this workflow.


def add_pipeline_subparser(subparsers):
    ap = subparsers.add_parser(
        "pipeline",
        help="Run the full Band/FatBand/DOS workflow end-to-end (replaces WorkFlow_FatBandDOS.sh)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n"
        " python -m CCpy.SIESTA.siesta_band pipeline --label SWNT7-6W --carfile CAR_POSCAR/SWNT7-6W.car \\\n"
        "        --moiety Tube:1-508 --moiety Ads:700-786 --mpi-run srun \n"
        " -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n"
        f" Steps (in order, default=all): {', '.join(PIPELINE_STEPS)} \n"
        " --steps genfdf,fatband     (only generate inputs + run fatband projection, skip siesta/plots) \n"
        " --steps plot-band,plot-fatband,plot-dos   (re-plot only, skip all calculations) \n"
        " This sub-command is normally invoked from inside an sbatch job (see CCpySIESTABandSubmit.py),\n"
        " which is what supplies the actual node/core allocation for {mpi-run} {siesta-bin}. \n"
        " -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n"
    )
    ap.add_argument("--label", default=None, help="SystemLabel / base .fdf name (default: auto-detect)")
    ap.add_argument("--steps", default="all",
                    help=f"comma list of steps to run, subset/order of: {','.join(PIPELINE_STEPS)} (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print planned actions without executing")

    # genfdf passthrough
    ap.add_argument("--carfile", default=None)
    ap.add_argument("--bandpath", choices=["seekpath", "1d", "hex", "manual"], default="1d")
    ap.add_argument("--axis", default=None,
                     help="a/b/c for 1d bandpath; default: let the genfdf step auto-detect "
                          "from kgrid_Monkhorst_Pack")
    ap.add_argument("--use-seekpath", action="store_true")
    ap.add_argument("--manual-kpath-file", default=None)
    ap.add_argument("--nseg", type=int, default=40)
    ap.add_argument("--ef-window", type=int, default=15)
    ap.add_argument("--extra-bands", type=int, default=50)

    # DOS options
    ap.add_argument("--dos-emin", type=float, default=-25.0)
    ap.add_argument("--dos-emax", type=float, default=25.0)
    ap.add_argument("--dos-broad", type=float, default=0.05)
    ap.add_argument("--dos-npts", type=int, default=2000)
    add_bool_flag(ap, "--pdos", default=True)

    # fatband options
    add_bool_flag(ap, "--element", dest="do_element", default=True,
                  help="run element-decomposed fatbands (default: on)")
    ap.add_argument("--moiety", action="append", default=[],
                     help="Moiety as NAME:IDXSEL (1-based), e.g. 'Tube:1-508'. Repeatable. Look up indices "
                          "with `fatband --list-atoms` first.")

    # plot options
    ap.add_argument("--plot-emin", type=float, default=-2.0)
    ap.add_argument("--plot-emax", type=float, default=2.0)
    ap.add_argument("--lw-band", type=float, default=1.0)
    ap.add_argument("--fat-cmap", default="inferno")

    # binaries
    ap.add_argument("--siesta-bin", default=None, help="full path to siesta executable (default: derived from --bin-dir)")
    ap.add_argument("--bin-dir", default=str(DEFAULT_SIESTA_BIN_DIR), help="dir containing siesta/fat/eigfat2plot/gnubands")
    ap.add_argument("--mpi-run", default="srun", help="MPI launcher prefix (default: srun)")
    ap.set_defaults(func=cmd_pipeline)
    return ap


def _run_siesta(fdf_path: Path, out_path: Path, mpi_run: str, siesta_bin: str, dry_run: bool = False):
    cmd_str = f"{mpi_run} {siesta_bin} < {fdf_path.name} > {out_path.name}"
    print(f"[cmd] (cwd={fdf_path.parent})  {cmd_str}")
    if dry_run:
        return
    with open(fdf_path, "rb") as fin, open(out_path, "wb") as fout:
        subprocess.run([mpi_run, siesta_bin], stdin=fin, stdout=fout, check=True, cwd=str(fdf_path.parent))


def _expand_steps(steps_arg: str) -> List[str]:
    if steps_arg.strip().lower() == "all":
        return list(PIPELINE_STEPS)
    wanted = [s.strip() for s in steps_arg.split(",") if s.strip()]
    bad = [s for s in wanted if s not in PIPELINE_STEPS]
    if bad:
        raise SystemExit(f"ERROR: unknown step(s) {bad}. Valid steps: {PIPELINE_STEPS}")
    # keep canonical order, but only the ones requested
    return [s for s in PIPELINE_STEPS if s in wanted]


def cmd_pipeline(args):
    parser = _PARSER
    assert parser is not None

    top = Path.cwd()
    label = args.label or auto_detect_label(top)
    steps = _expand_steps(args.steps)
    print(f"[info] label = {label}")
    print(f"[info] steps = {steps}")

    bin_dir = Path(args.bin_dir)
    siesta_bin = args.siesta_bin or str(bin_dir / "siesta")

    bands_dir = top / "Band-DOS" / "BANDS"
    dos_dir = top / "Band-DOS" / "DOS"

    def call(subcmd: str, argv: List[str]):
        ns = parser.parse_args([subcmd] + argv)
        ns.func(ns)

    # Each step runs independently: a failure in one (e.g. fatband) must not
    # silently prevent later, unrelated steps (dos-calc, plot-dos, ...) from
    # running - same "don't let one failure kill the rest" philosophy as the
    # batch mpi.sh (no `set -e`). Failures are collected and reported at the
    # end; the pipeline exits non-zero if anything failed, but every
    # requested step still gets attempted.
    failed_steps: List[str] = []

    def run_step(step_name: str, fn):
        try:
            fn()
        except SystemExit as exc:
            print(f"[ERROR] step '{step_name}' aborted: {exc}")
            failed_steps.append(step_name)
        except Exception as exc:
            print(f"[ERROR] step '{step_name}' failed: {type(exc).__name__}: {exc}")
            failed_steps.append(step_name)

    if "genfdf" in steps:
        def _genfdf():
            # --strict: fatbands (Wfs.band.min/max, Save-HS, WFS.Write.For.Bands)
            # need Nelec from an EXISTING plain-SCF <label>.out in the system's
            # top-level directory (one level above Band-DOS/) - this is the
            # same precondition the original siesta_Band-DOS_lineband_fat.py /
            # WorkFlow_FatBandDOS.sh always had. Without --strict, a missing
            # .out is only a soft [WARN] and genfdf silently writes an fdf with
            # NO fatbands options at all - band-calc then runs to completion
            # "successfully" but produces no WFSX, and the fatband step fails
            # hours later having wasted the whole SIESTA run. --strict turns
            # that into an immediate, clear failure before any SIESTA run starts.
            want_fatbands = "fatband" in steps
            # Only generate the Band-DOS/{BANDS,DOS} sub-directory this mode
            # actually needs, instead of always writing both (e.g. mode 2
            # "band only" has no dos-calc/plot-dos in its steps, so there's
            # no reason for a Band-DOS/DOS/ folder to appear at all).
            needs_bands_fdf = any(s in steps for s in ("band-calc", "fatband", "plot-band", "plot-fatband"))
            needs_dos_fdf = any(s in steps for s in ("dos-calc", "plot-dos"))
            if needs_bands_fdf and needs_dos_fdf:
                genfdf_mode = "all"
            elif needs_bands_fdf:
                genfdf_mode = "bands"
            elif needs_dos_fdf:
                genfdf_mode = "dos"
            else:
                genfdf_mode = "all"  # genfdf requested on its own with nothing downstream - keep old behavior
            call("genfdf", [
                "--label", label,
                "--mode", genfdf_mode,
                "--ef-window", str(args.ef_window),
                "--bandpath", args.bandpath,
                "--extra-bands", str(args.extra_bands),
                "--dos-emin", str(args.dos_emin), "--dos-emax", str(args.dos_emax),
                "--dos-broad", str(args.dos_broad), "--dos-npts", str(args.dos_npts),
            ] + (["--fatbands", "--strict"] if want_fatbands else [])
              + (["--axis", args.axis] if args.axis is not None else [])
              + (["--carfile", args.carfile, "--nseg", str(args.nseg)] if args.carfile else [])
              + (["--use-seekpath"] if args.use_seekpath else [])
              + (["--manual-kpath-file", args.manual_kpath_file] if args.manual_kpath_file else []))
        run_step("genfdf", _genfdf)
        if "genfdf" in failed_steps:
            # Everything else reads the fdf/directories genfdf is supposed to
            # have written correctly - don't burn a SIESTA run on a broken input.
            print("[FAILED] 'genfdf' failed; skipping all remaining steps "
                  "(band-calc/fatband/dos-calc/... all depend on it).")
            raise SystemExit(1)

    if "band-calc" in steps:
        def _band_calc():
            if not bands_dir.exists():
                raise SystemExit(f"{bands_dir} not found. Run the 'genfdf' step first.")
            fdf = bands_dir / f"{label}.fdf"
            out = bands_dir / f"{label}.out"
            _run_siesta(fdf, out, args.mpi_run, siesta_bin, dry_run=args.dry_run)
        run_step("band-calc", _band_calc)

    if "fatband" in steps:
        def _fatband():
            if not bands_dir.exists():
                raise SystemExit(f"{bands_dir} not found. Run 'genfdf' + 'band-calc' first "
                                  f"(fatband only post-processes an existing band calculation).")
            os.chdir(bands_dir)
            try:
                if args.do_element:
                    call("fatband", ["--element"])
                if args.moiety:
                    moiety_argv = []
                    for m in args.moiety:
                        moiety_argv += ["--moiety", m]
                    call("fatband", ["--moiety-decomp"] + moiety_argv)
            finally:
                os.chdir(top)
        run_step("fatband", _fatband)

    if "plot-band" in steps:
        def _plot_band():
            os.chdir(bands_dir)
            try:
                call("plot-band", ["-e", str(args.plot_emin), "-E", str(args.plot_emax),
                                    "--lw-band", str(args.lw_band), "--bin-dir", str(bin_dir), "--basename", label])
            finally:
                os.chdir(top)
        run_step("plot-band", _plot_band)

    if "plot-fatband" in steps:
        def _plot_fatband():
            os.chdir(bands_dir)
            try:
                # plot-fatband (without --fat-only) overlays onto bands.dat but
                # does NOT run gnubands itself - normally bands.dat comes from
                # a prior 'plot-band' step, but mode 4 deliberately does not
                # include plot-band (no separate plain-band png wanted), so
                # generate bands.dat here directly to keep this step
                # self-contained regardless of whether 'plot-band' also ran.
                bands_dat = Path("bands.dat")
                if not bands_dat.exists():
                    try:
                        run_gnubands(basename=label, emin=args.plot_emin, emax=args.plot_emax,
                                     out_dat=str(bands_dat), bin_dir=bin_dir)
                    except Exception as exc:
                        print(f"[warn] could not generate bands.dat via gnubands ({exc}); "
                              f"the overlay plot below will be skipped, --fat-only plot will still run")
                # Two independent plots - overlay needs bands.dat (best-effort
                # above), --fat-only never does. Run both regardless of
                # whether the other one succeeds.
                if bands_dat.exists():
                    try:
                        call("plot-fatband", ["--fat-auto", "-e", str(args.plot_emin), "-E", str(args.plot_emax),
                                               "--lw-band", str(args.lw_band), "--each", "--fat-cmap", args.fat_cmap,
                                               "--basename", label])
                    except Exception as exc:
                        print(f"[warn] plot-fatband (overlay) failed: {exc}")
                call("plot-fatband", ["--fat-auto", "-e", str(args.plot_emin), "-E", str(args.plot_emax),
                                       "--lw-band", str(args.lw_band), "--each", "--fat-cmap", args.fat_cmap,
                                       "--fat-only", "--basename", label])
            finally:
                os.chdir(top)
        run_step("plot-fatband", _plot_fatband)

    if "dos-calc" in steps:
        def _dos_calc():
            if not dos_dir.exists():
                raise SystemExit(f"{dos_dir} not found. Run the 'genfdf' step first.")
            fdf = dos_dir / f"{label}.fdf"
            out = dos_dir / f"{label}.out"
            _run_siesta(fdf, out, args.mpi_run, siesta_bin, dry_run=args.dry_run)
        run_step("dos-calc", _dos_calc)

    if "plot-dos" in steps:
        def _plot_dos():
            os.chdir(dos_dir)
            try:
                call("plot-dos", [])
            finally:
                os.chdir(top)
        run_step("plot-dos", _plot_dos)

    if failed_steps:
        print(f"[FAILED] pipeline finished with {len(failed_steps)} failed step(s): {failed_steps}")
        raise SystemExit(1)
    print("[ok] pipeline finished.")


# =============================================================================
# ============================  CLI entry  ====================================
# =============================================================================

_PARSER: Optional[argparse.ArgumentParser] = None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m CCpy.SIESTA.siesta_band",
        description="Unified SIESTA Band / FatBand / DOS workflow (see module docstring for details).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = ap.add_subparsers(dest="command")
    add_genfdf_subparser(subparsers)
    add_fatband_subparser(subparsers)
    add_plot_band_subparser(subparsers)
    add_plot_fatband_subparser(subparsers)
    add_plot_dos_subparser(subparsers)
    add_pipeline_subparser(subparsers)
    return ap


def print_menu():
    print("""
--------------------------------------------------------------------------------------
    How to use: python -m CCpy.SIESTA.siesta_band <command> [options]
--------------------------------------------------------------------------------------
< Commands >
    genfdf         : generate Band-DOS/{BANDS,DOS} dirs + patched fdf (BandLines, fatbands)
    fatband        : element / moiety decomposed fatbands (.mpr -> fat -> eigfat2plot)
    plot-band      : gnubands + band structure plot (bandgap annotation)
    plot-fatband   : fatband overlay / fat-only plot
    plot-dos       : DOS plot (Fermi-aligned, van Hove peak detection)
    pipeline       : run the full workflow end-to-end (equivalent to WorkFlow_FatBandDOS.sh)
                     use --steps to run only a subset, e.g. --steps genfdf,fatband

Run `python -m CCpy.SIESTA.siesta_band <command> --help` for the options of each command.
--------------------------------------------------------------------------------------
""")


def main():
    global _PARSER
    parser = build_parser()
    _PARSER = parser

    if len(sys.argv) < 2:
        print_menu()
        sys.exit(0)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        print_menu()
        sys.exit(0)

    try:
        args.func(args)
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
