#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCpySIESTAInputGen.py  (formerly car2siesta.py, v4)
----------------------------------------------------
Materials Studio / BIOSYM .car (+ .cif / POSCAR / CONTCAR via pymatgen)
-> SIESTA .fdf generator (+ optional Slurm mpi.sh + gen[id].list)

NOTE ON THE NAME: the lab already has an installed CCpy-package script
also called CCpySIESTAInputGen.py (at
/home/shared/anaconda3/envs/CCpy/bin/CCpySIESTAInputGen.py, pinned there
via easy-install). This file is a *different*, standalone script that
happens to share that name by request - it does not replace or modify the
installed CCpy package version. Keep this file's directory earlier in your
PATH (or call it as ./CCpySIESTAInputGen.py / python3 CCpySIESTAInputGen.py)
if you want to make sure this one runs instead of the installed one.

Requests applied (2026-02):
1) Removed the DOS, BAND, PDOS related options from car2siesta.py
2) Added the restart options of siesta_Band-DOS.py (works without a .car file, in the OPT result directory) to car2siesta.py
   - Auto-extracts SystemLabel from *.fdf in the current directory, then creates a Restart/ directory.
   - Into Restart/ the original fdf is copied, and with DM.UseSaveDM / MD.UseSaveXV forced to .true., a
     <label>_restart.fdf and a Slurm mpi.sh are generated.
3) (Adding/modifying DOS/PDOS input options is implemented in siesta_Band-DOS.py)

Requests applied (2026-07, v2):
4) Support .cif as well as .car as the input structure file (replaced by a pymatgen-based reader in v3)
5) YAML config file support so long CLI options need not be retyped every time (--config, or
   auto-detection of siesta_default.yaml in cwd). Config file values are used as CLI option defaults,
   and an explicitly given CLI argument takes precedence.

Requests applied (2026-07, v3) - merged in features of CCpySIESTAInputGen.py (the lab CCpy tool):
6) Unified the structure file parser on pymatgen -> supports VASP POSCAR/CONTCAR as well as .cif
   (uses IStructure.from_file just like CCpySIESTAInputGen.py, only .car keeps the built-in parser)
7) Ported the prompt CCpySIESTAInputGen.py used to ask interactively at run time, "Add or modify option
   (ex: XC.functional=LDA, MaxSCFIterations=200)", to the non-interactive --extra option.
   No more stalling on input in --all/Slurm batch jobs. If it collides with an existing fdf option line,
   that line is overwritten (the fdf parser honors only the first definition); otherwise it is appended.
8) The defaults (DM.Tolerance, MaxSCFIterations, etc.) and the kgrid auto-calculation rules keep the
   existing car2siesta.py behavior (the CCpy kgrid calculation had a bug copying the b axis value to c)
9) Run interactively in a terminal without --extra and, just like CCpySIESTAInputGen.py, the
   "Add or modify option ... / else Enter" prompt appears once. When stdin is not a terminal,
   as with Slurm/cron, it is skipped automatically so batch jobs do not stall.
10) By request (installing pymatgen/PyYAML by hand every time is tedious), just as CCpySIESTAInputGen.py
    is pinned to a conda env (envs/CCpy) by shebang, if the running python lacks pymatgen/yaml it
    re-execs automatically with a conda env python registered in KNOWN_ENV_PYTHONS. To turn this off,
    run with the environment variable CAR2SIESTA_NO_AUTOENV=1.
11) So --pseudo-dir need not be given every time, DEFAULT_PSEUDO_DIRS (default: Dojo only) is built in
    as the default search path. Given --pseudo-dir it always wins; otherwise these defaults are searched
    per element in order and copied/symlinked. If the storage location moves, edit DEFAULT_PSEUDO_DIRS only.
    PSF was dropped from the defaults since its element coverage is a subset of Dojo (searching both
    automatically copied both .psf/.psml for the same element, which kept SIESTA from proceeding).
    If PSF/PSF_old is needed, specify it explicitly with --pseudo-dir.
12) Like selectInputs() in CCpySIESTAInputGen.py/CCpyVASPInputGen.py, running with no file given
    (and without --all) scans the current folder for .car/.cif/POSCAR/CONTCAR, lists them with numbers
    and lets you pick via "Choose file :" (0 = all). Passing a file directly or using --all goes
    straight through as before, without this screen. It never appears non-interactively (Slurm/cron/pipe).
13) Like the INCAR option preview in CCpyVASPInputGen.py, before the --extra prompt appears the current
    mode/basis/kgrid rule/MeshCutoff/DM.Tolerance/MaxSCFIterations/per-mode MD options/
    pseudo_dir are shown on screen first. Both an "n" entry and a plain Enter are treated as skip
    (same as CCpyVASPInputGen.py).

Requests applied (2026-07, v4) - car2siesta.py renamed to CCpySIESTAInputGen.py, CLI switched to CCpy style:
14) Dropped the --mode flag; like the original CCpySIESTAInputGen.py/CCpyVASPInputGen.py, the
    calculation mode number (1~7) is taken as the first positional argument: 1=opt 2=nvt 3=nve 4=npe 5=npt 6=anneal 7=scf
    (same order the old interactive menu used, opt is always 1). If the number is missing or wrong
    (and it is not --restart-from-outdir either), it prints a VASPInputGen.py-style "How to use" guide
    and exits cleanly. NOTE: argparse cannot split two positional arguments (mode number, structure file)
    properly when option flags sit between them, so extract_mode_number() is implemented to strip
    only the leading number manually before argparse runs.
15) The basis set can be given in lowercase, e.g. -basis=sz (-basis/--basis behave identically,
    case-insensitive, written uppercase into the actual fdf). If --basis is omitted entirely, the default
    is now fixed straight to SZ instead of an interactive prompt.
16) A predefined pseudopotential set can be chosen with -pot=dojo / -pot=psf / -pot=psfold
    (POT_ALIASES dictionary, default dojo). An explicit --pseudo-dir always has top priority and
    -pot= is ignored. The DEFAULT_PSEUDO_DIRS scheme (multi-path fallback) is replaced by this
    POT_ALIASES scheme - exactly one pot directory is always chosen, so the mixed .psf/.psml copy
    problem cannot arise by construction.

Usage examples (v4, CCpySIESTAInputGen.py):
  # (A) generate fdf with a mode number + CCpy-style short options
  python CCpySIESTAInputGen.py 1 -basis=sz -pot=dojo --kgrid-from-car --max-force 0.02 ZnOHX3L.car
  python CCpySIESTAInputGen.py 7 -basis=dzp structure.cif        # 7 = scf
  python CCpySIESTAInputGen.py 1 POSCAR

  # (A-2) run with no file given: scans the current folder and lets you pick by number
  python CCpySIESTAInputGen.py 1 -basis=sz

  # (A-3) fewer options via a config file (auto-loaded if siesta_default.yaml is in the same directory)
  python CCpySIESTAInputGen.py 1 ZnOHX3L.car
  python CCpySIESTAInputGen.py 1 --config ../lab_defaults.yaml ZnOHX3L.car

  # (A-4) CCpy-style option override (non-interactive)
  python CCpySIESTAInputGen.py 1 --extra "XC.functional=LDA,MaxSCFIterations=50" structure.cif

  # (B) build the Restart/ workflow in an OPT result directory (= without a structure file; no mode number)
  python3 CCpySIESTAInputGen.py --restart-from-outdir

NOTE:
- This script no longer provides Bands/DOS/PDOS input blocks or the related CLI options.
- The old "patch coordinates/cell from a siesta .out into the fdf" feature was renamed to --patch-from-out.
- As of v4 --mode has been removed. If you have scripts/aliases that used --mode opt and the like,
  please change them to put a number (1~7) at the front.

Requests applied (2026-07, v4 additions):
17) Reordered so mode_number validation happens before any file search/read. If mode_number is
    missing or wrong (and it is not --restart-from-outdir either), it prints How-to-use and exits
    immediately without touching the file picker or the --all scan at all.
18) -maxforce=0.02 (same as --max-force, default 0.02 - corrected after feedback that the lab
    actually uses 0.02 by default. Comparison set: 0.05(rough)/0.02(default)/0.01(tight)). The How-to-use
    screen shows the choices and meaning of -basis/-maxforce/-pot in a table (it appears the same
    way when simply run with no mode_number).
19) In CCpyVASPInputGen.py style, the original structure file (.car/.cif/POSCAR/
    CONTCAR) whose fdf generation is done is moved to ./structures/ relative to cwd (moved, not copied).
    Regardless of --outdir, they always pile up in structures/ of the run directory.
- .cif/POSCAR/CONTCAR input requires pymatgen (`pip install pymatgen`) to be installed.
  (already installed in the lab CCpy environment)
- The config file feature requires PyYAML (`pip install pyyaml`) to be installed. Without either one,
  the existing .car workflow still works (the needed module is imported only when that feature is used).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# -----------------------------
# Auto environment bootstrap
# -----------------------------
# .car-only, no-config runs need nothing but the standard library. But
# .cif/POSCAR/CONTCAR input needs pymatgen, and --config/siesta_default.yaml
# needs PyYAML - and asking every lab member to `pip install` those by hand
# on every machine/account is exactly the friction we're trying to remove.
#
# CCpySIESTAInputGen.py sidesteps this by being installed via easy-install
# with a shebang pinned to a specific conda env
# (/home/shared/anaconda3/envs/CCpy/bin/python) that already has pymatgen.
# We mirror that here at runtime instead of via shebang, so it also works
# when someone runs `python3 car2siesta.py ...` explicitly (which bypasses
# the shebang line entirely): if the *currently running* interpreter is
# missing pymatgen/yaml, and a known lab conda env already has them, this
# script quietly relaunches itself under that interpreter.
#
# Add more paths here if your lab sets up additional/other shared envs.
KNOWN_ENV_PYTHONS: Tuple[str, ...] = (
    "/home/shared/anaconda3/envs/CCpy/bin/python",  # existing shared CCpy env (has pymatgen)
)

# Named pseudopotential sets, selectable with -pot=dojo / -pot=psf /
# -pot=psfold (case-insensitive). Default is "dojo". --pseudo-dir, if given,
# always overrides this entirely (arbitrary custom path, highest priority).
#
# PSF/PSF_old are intentionally not auto-combined with Dojo: PSF's element
# coverage is a strict subset of Dojo's (same elements, different/older
# pseudopotential format - .psf vs .psml), so searching both automatically
# only risked copying BOTH a .psf and a .psml for the same element into a
# job dir, which SIESTA can't resolve and the run won't proceed. Exactly one
# named set (or one --pseudo-dir) is used per run. Edit this dict if the
# lab's pseudopotential storage moves or a new set gets added.
POT_ALIASES: Dict[str, str] = {
    "dojo": "/opt/siesta/POT/Dojo",      # .psml
    "psf": "/opt/siesta/POT/PSF",        # .psf
    "psfold": "/opt/siesta/POT/PSF_old",
}
DEFAULT_POT = "dojo"


def resolve_pseudo_dir(args) -> str:
    """--pseudo-dir (arbitrary path) always wins if given; otherwise resolve
    -pot=dojo/psf/psfold via POT_ALIASES (default: dojo)."""
    if args.pseudo_dir:
        return args.pseudo_dir
    return POT_ALIASES[args.pot]


# Calculation modes, numbered 1-7 for the leading positional argument
# (e.g. `CCpySIESTAInputGen.py 1`), in the same order the old interactive
# menu already used - opt is always 1, matching the lab's most common case.
MODE_NUMBERS: Tuple[str, ...] = ("opt", "nvt", "nve", "npe", "npt", "anneal", "scf")


def mode_number_to_name(raw: Optional[str]) -> Optional[str]:
    """Map a leading positional like '1' to its mode name ('opt'), or None
    if raw is missing/not a valid 1-7 integer string."""
    if raw is None:
        return None
    if not raw.isdigit():
        return None
    n = int(raw)
    if not (1 <= n <= len(MODE_NUMBERS)):
        return None
    return MODE_NUMBERS[n - 1]


def extract_mode_number(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    """
    Pulls a leading mode-number token (e.g. '1') out of argv, CCpy-style,
    *before* argparse ever runs - see the NOTE above the (removed)
    mode_number positional registration for why: argparse can't cleanly
    handle two separate positional actions (mode_number, car) once optional
    flags are interspersed between them.

    Only strips argv[0] if it's a bare digit string; anything else (a flag,
    a filename, --restart-from-outdir, etc.) is left completely untouched
    and passed through to argparse as normal.
    Returns (mode_number_token_or_None, remaining_argv).
    """
    rest = list(argv)
    if rest and rest[0].isdigit():
        return rest[0], rest[1:]
    return None, rest


MODE_DESCRIPTIONS: Dict[str, str] = {
    "opt": "Geometry relaxation (CG)",
    "nvt": "MD, constant volume/temperature (Nose thermostat)",
    "nve": "MD, constant volume/energy (Verlet)",
    "npe": "MD, constant pressure/energy (Parrinello-Rahman)",
    "npt": "MD, constant pressure/temperature (Nose-Parrinello-Rahman)",
    "anneal": "MD, simulated annealing",
    "scf": "Single-point SCF only (no relaxation/MD)",
}

BASIS_DESCRIPTIONS: List[Tuple[str, str]] = [
    ("SZ", "single-zeta - fastest, least accurate (default)"),
    ("SZP", "single-zeta + polarization - a bit more accurate than SZ, still cheap"),
    ("DZ", "double-zeta - noticeably more accurate than SZ, more expensive"),
    ("DZP", "double-zeta + polarization - common accuracy/cost balance"),
    ("TZ", "triple-zeta - high accuracy, expensive"),
    ("TZP", "triple-zeta + polarization - most accurate here, most expensive"),
]

MAXFORCE_EXAMPLES: List[Tuple[str, str]] = [
    ("0.02", "default - normal relaxation"),
    ("0.05", "looser/rougher - faster, less strict"),
    ("0.01", "very tight - final/high-precision check (slow)"),
]


def print_mode_number_usage(prog: str) -> None:
    """CCpyVASPInputGen.py-style short usage reminder, shown when the
    leading mode-number positional is missing or invalid (and this isn't
    the no-file --restart-from-outdir workflow, which doesn't need one).
    Also doubles as the "what values exist" reference for -basis/-maxforce,
    since those are otherwise easy to forget the choices for."""
    print(f"\nHow to use : {prog} [mode_number] [options...]")
    print("--------------------------------------")
    print("[calculation modes]")
    for i, name in enumerate(MODE_NUMBERS, start=1):
        print(f"  {i} : {name:<8s} {MODE_DESCRIPTIONS.get(name, '')}")
    print("")
    print("[-basis / --basis : PAO.BasisSize]")
    for name, desc in BASIS_DESCRIPTIONS:
        print(f"  {name:<5s} : {desc}")
    print("")
    print("[-maxforce / --max-force : MD.MaxForceTol (eV/Ang), for opt mode]")
    for val, desc in MAXFORCE_EXAMPLES:
        print(f"  {val:<6s} : {desc}")
    print("")
    print("[-pot / --pot : pseudopotential set]")
    for name, path_ in POT_ALIASES.items():
        marker = " (default)" if name == DEFAULT_POT else ""
        print(f"  {name:<7s}: {path_}{marker}")
    print("")
    print(f"ex) {prog} 1 -basis=sz -pot=dojo --kgrid-from-car -maxforce=0.02")
    print("")
    print("Structure files (.car/.cif/POSCAR/CONTCAR): give them as extra")
    print("arguments, use --all, or just omit them to pick from the current")
    print("directory interactively.")
    print("")
    print(f"Full option list: {prog} --help")
    print(f"Restart workflow (no structure file needed): {prog} --restart-from-outdir")


_REEXEC_GUARD_ENV = "_CAR2SIESTA_REEXECED"


def _current_interpreter_has_deps() -> bool:
    try:
        import pymatgen  # noqa: F401
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def bootstrap_reexec_if_needed() -> None:
    """
    If the current python is missing pymatgen/PyYAML, try relaunching this
    same script (same argv) under one of KNOWN_ENV_PYTHONS that does have
    them. No-op if the current interpreter already has everything, if no
    known env qualifies, or if CAR2SIESTA_NO_AUTOENV=1 is set.

    Safe by construction: guarded by an env var so it can only re-exec once
    per process tree (no infinite loop), and if nothing suitable is found it
    simply falls through and runs under the current interpreter as before -
    plain .car workflows without --config are unaffected either way.
    """
    if os.environ.get(_REEXEC_GUARD_ENV) == "1":
        return
    if os.environ.get("CAR2SIESTA_NO_AUTOENV") == "1":
        return
    if _current_interpreter_has_deps():
        return

    this_file = os.path.abspath(__file__)
    for candidate in KNOWN_ENV_PYTHONS:
        # NOTE: deliberately not comparing candidate vs sys.executable by
        # realpath here - venv-created interpreters are often *symlinks*
        # back to the base system python binary while still having their
        # own (different) site-packages, so that comparison would wrongly
        # treat a genuinely different/better environment as "the same one"
        # and skip it. We already know the current interpreter lacks the
        # deps (checked above), so we just trust the probe below instead.
        try:
            if not os.path.isfile(candidate):
                continue
            probe = subprocess.run(
                [candidate, "-c", "import pymatgen, yaml"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if probe.returncode != 0:
                continue
        except OSError:
            continue

        print(f"[car2siesta] pymatgen/PyYAML not found in current interpreter; "
              f"relaunching under {candidate}", file=sys.stderr)
        sys.stderr.flush()
        os.environ[_REEXEC_GUARD_ENV] = "1"
        os.execv(candidate, [candidate, this_file] + sys.argv[1:])
    # Fell through: no known env has the deps either. Continue under the
    # current interpreter - missing-package errors (if any) surface later
    # with a clear "pip install ..." message when actually needed.


# -----------------------------
# Periodic table (1..118)
# -----------------------------
_Z: Dict[str, int] = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,
    "Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,
    "Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,
    "Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,
    "Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,
    "Lu":71,"Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,
    "Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,
    "Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,"Cf":98,"Es":99,"Fm":100,
    "Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,"Sg":106,"Bh":107,"Hs":108,"Mt":109,"Ds":110,
    "Rg":111,"Cn":112,"Nh":113,"Fl":114,"Mc":115,"Lv":116,"Ts":117,"Og":118,
}

def _norm_el(sym: str) -> str:
    sym = sym.strip()
    if not sym:
        return sym
    return sym[0].upper() + (sym[1:].lower() if len(sym) > 1 else "")

@dataclass
class Atom:
    name: str
    x: float
    y: float
    z: float
    el: str

@dataclass
class CarData:
    atoms: List[Atom]
    header: List[str]
    cell_params: Optional[Tuple[float,float,float,float,float,float]] = None  # a b c alpha beta gamma
    cell_vectors: Optional[List[Tuple[float,float,float]]] = None


# -----------------------------
# CAR parsing
# -----------------------------
def read_car(path: Path) -> CarData:
    lines = path.read_text(errors="ignore").splitlines()
    header = lines[:4] if len(lines) >= 4 else lines[:]
    body = lines[4:]

    atoms: List[Atom] = []
    cell_params: Optional[Tuple[float, float, float, float, float, float]] = None
    cell_vectors = None

    # PBC cell line
    for ln in body:
        s = ln.strip()
        if not s:
            continue
        if s.upper().startswith("PBC="):
            continue
        if s.split()[0].upper() == "PBC":
            parts = s.split()
            if len(parts) >= 7:
                try:
                    a = float(parts[1]); b = float(parts[2]); c = float(parts[3])
                    alpha = float(parts[4]); beta = float(parts[5]); gamma = float(parts[6])
                    cell_params = (a, b, c, alpha, beta, gamma)
                except Exception:
                    cell_params = None
            break

    # atoms until 'end'
    for ln in body:
        s = ln.strip()
        if not s:
            continue
        if s.lower() == "end":
            break
        if s.upper().startswith("PBC="):
            continue
        if s.split()[0].upper() == "PBC":
            continue

        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            continue

        name = parts[0]
        el = parts[7] if len(parts) >= 8 else re.sub(r"\d+$", "", name)
        el = _norm_el(el)
        if not el:
            raise ValueError(f"Failed to parse element for line: {ln}")
        atoms.append(Atom(name=name, x=x, y=y, z=z, el=el))

    return CarData(atoms=atoms, header=header, cell_params=cell_params, cell_vectors=cell_vectors)


def read_with_pymatgen(path: Path) -> CarData:
    """
    Read a .cif / POSCAR / CONTCAR file via pymatgen and convert it into the
    same CarData shape used for .car files, so the rest of the pipeline
    (sorting, cell handling, fdf generation) doesn't need to know which
    format the structure came from. This mirrors how the lab's
    CCpySIESTAInputGen.py reads structures (IStructure.from_file), so
    results should match what that tool would parse.

    Requires: pip install pymatgen
    """
    try:
        from pymatgen.core.structure import IStructure  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: reading .cif/POSCAR/CONTCAR files requires pymatgen. "
            "Please run: pip install pymatgen"
        ) from exc

    st = IStructure.from_file(str(path))

    lat = st.lattice
    cell_params = (float(lat.a), float(lat.b), float(lat.c),
                   float(lat.alpha), float(lat.beta), float(lat.gamma))

    counts: Dict[str, int] = {}
    atoms: List[Atom] = []
    for site in st.sites:
        el = _norm_el(str(site.specie))
        if not el:
            raise ValueError(f"Failed to parse element from site: {site!r}")
        counts[el] = counts.get(el, 0) + 1
        name = f"{el}{counts[el]}"
        x, y, z = (float(c) for c in site.coords)  # Cartesian, Angstrom
        atoms.append(Atom(name=name, x=x, y=y, z=z, el=el))

    if not atoms:
        raise ValueError(f"No atoms parsed from structure file: {path}")

    return CarData(atoms=atoms, header=[f"# imported via pymatgen: {path.name}"],
                    cell_params=cell_params, cell_vectors=None)


# Structure file formats this script knows how to read. POSCAR/CONTCAR are
# matched by filename (VASP convention has no extension), everything else by
# suffix - see is_supported_structure_file() / read_structure() below.
STRUCTURE_SUFFIXES: Tuple[str, ...] = (".car", ".cif")
POSCAR_LIKE_NAMES: Tuple[str, ...] = ("POSCAR", "CONTCAR")


def is_supported_structure_file(path: Path) -> bool:
    if path.suffix.lower() in STRUCTURE_SUFFIXES:
        return True
    stem_or_name = path.name.upper()
    return any(stem_or_name == n or stem_or_name.startswith(n + ".") for n in POSCAR_LIKE_NAMES)


def read_structure(path: Path) -> CarData:
    """Dispatch to the right reader based on file name (.car / .cif / POSCAR / CONTCAR)."""
    suf = path.suffix.lower()
    if suf == ".car":
        return read_car(path)
    if suf == ".cif" or path.name.upper().split(".")[0] in POSCAR_LIKE_NAMES:
        return read_with_pymatgen(path)
    raise ValueError(f"Unsupported structure file '{path}' "
                      f"(supported: {', '.join(STRUCTURE_SUFFIXES)}, {', '.join(POSCAR_LIKE_NAMES)})")


def select_structure_files_interactively(cwd: Path) -> List[Path]:
    """
    Mirrors CCpy's selectInputs() UX (seen in CCpySIESTAInputGen.py /
    CCpyVASPInputGen.py): when the user runs car2siesta.py with no file
    arguments at all, scan cwd for supported structure files, list them
    numbered, offer '0' for all of them, and return what was chosen.

    Only called when stdin is a real terminal (see main()) - never blocks
    batch/Slurm/cron runs, which must pass a file or use --all instead.
    """
    candidates = sorted(
        p for p in cwd.iterdir()
        if p.is_file() and is_supported_structure_file(p)
    )
    if not candidates:
        return []

    print("")
    for i, p in enumerate(candidates, start=1):
        print(f"{i} : {p.name}")
    print("0 : All files")
    try:
        choice = input("Choose file : ").strip()
    except EOFError:
        choice = ""

    if choice == "0":
        return candidates

    indices = _parse_index_selection(choice, len(candidates))
    if indices:
        return [candidates[i - 1] for i in indices]

    print(f"WARNING: invalid choice {choice!r} - no file selected.")
    return []


def _parse_index_selection(choice: str, n: int) -> List[int]:
    """
    Parse a picker selection like "2", "1,3", or "1-2,4" into a list of
    1-based indices (in the order given, de-duplicated), each checked to be
    within [1, n]. Returns [] for anything malformed or out of range -
    callers treat that the same as an invalid single choice.
    """
    tokens = [t.strip() for t in choice.split(",") if t.strip()]
    if not tokens:
        return []

    seen: set = set()
    result: List[int] = []
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok)
            if not (1 <= idx <= n):
                return []
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
            continue

        if "-" in tok:
            lo_s, sep, hi_s = tok.partition("-")
            if lo_s.strip().isdigit() and hi_s.strip().isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    lo, hi = hi, lo
                if not (1 <= lo <= n and 1 <= hi <= n):
                    return []
                for idx in range(lo, hi + 1):
                    if idx not in seen:
                        seen.add(idx)
                        result.append(idx)
                continue

        return []  # malformed token

    return result


# -----------------------------
# Sorting / cell helpers
# -----------------------------

# -----------------------------
# kgrid helper for metallic 1D systems (e.g., armchair CNT)
# -----------------------------
def _oddify(n: int) -> int:
    return n if (n % 2) == 1 else (n + 1)

def kz_metal_1d_from_c(c_ang: float, dk: float = 0.02, kz_min: int = 7) -> int:
    """
    Metallic 1D k-sampling rule along periodic axis (here: c / z):
      kz = ceil((2*pi/c) / dk)
    then enforce:
      kz >= kz_min
      kz is odd (so Γ is included for Γ-centered mesh when shift=0.0)

    Parameters
    ----------
    c_ang : float
        Cell length along periodic direction in Angstrom.
    dk : float
        Target k-point spacing in 1/Angstrom. Typical: 0.02 (band), 0.01 (DOS/metal).
    kz_min : int
        Minimum kz to avoid kz=1 for long supercells.

    Returns
    -------
    int : kz (odd)
    """
    if c_ang <= 0:
        raise ValueError(f"Invalid c cell length: {c_ang}")
    if dk <= 0:
        raise ValueError(f"Invalid dk: {dk}")
    kz = int(math.ceil((2.0 * math.pi / c_ang) / dk))
    kz = max(int(kz_min), kz)
    return _oddify(kz)

def kgrid_metal_1d(cell_params: Tuple[float,float,float,float,float,float],
                   dk: float = 0.02,
                   kz_min: int = 7,
                   kx: int = 1,
                   ky: int = 1) -> Tuple[int,int,int]:
    """
    Build (kx,ky,kz) with kz from metallic-1D rule using cell_params c-length.
    """
    c_ang = float(cell_params[2])
    return (max(1,int(kx)), max(1,int(ky)), kz_metal_1d_from_c(c_ang, dk=dk, kz_min=kz_min))

def sort_atoms(atoms: List[Atom], element_order: Optional[List[str]]) -> List[Atom]:
    if not atoms:
        return atoms

    # --no-sort: if element_order == [], keep the original atom order
    if element_order is not None and len(element_order) == 0:
        return atoms
    # sort only when element_order is explicitly given
    if element_order is not None:
        order = [_norm_el(x) for x in element_order]
        rank = {el:i for i,el in enumerate(order)}
        def key(a: Atom):
            return (rank.get(a.el, 10_000), a.el, a.name)
        return sorted(atoms, key=key)

    # default (as before): sort by element
    return sorted(atoms, key=lambda a: (a.el, a.name))

def bounding_box(atoms: Sequence[Atom]) -> Tuple[float,float,float,float,float,float]:
    xs=[a.x for a in atoms]; ys=[a.y for a in atoms]; zs=[a.z for a in atoms]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

def make_orthorhombic_cell_from_atoms(atoms: Sequence[Atom], vacuum: float) -> Tuple[Tuple[float,float,float,float,float,float], Tuple[float,float,float]]:
    xmin,xmax,ymin,ymax,zmin,zmax = bounding_box(atoms)
    ax = (xmax - xmin) + vacuum
    by = (ymax - ymin) + vacuum
    cz = (zmax - zmin) + vacuum
    shift = (-(xmin) + vacuum/2.0, -(ymin) + vacuum/2.0, -(zmin) + vacuum/2.0)
    return (ax, by, cz, 90.0, 90.0, 90.0), shift

def apply_shift(atoms: Sequence[Atom], shift: Tuple[float,float,float]) -> List[Atom]:
    sx,sy,sz = shift
    return [Atom(a.name, a.x+sx, a.y+sy, a.z+sz, a.el) for a in atoms]


# -----------------------------
# SIESTA blocks (presets)
# -----------------------------
def fdf_header(system_name: str, system_label: str, natoms: int, species: List[str], basis: str, net_charge: float) -> str:
    lines = []
    lines.append("# General System descriptors")
    lines.append(f"SystemName {system_name}")
    lines.append(f"SystemLabel {system_label}")
    lines.append("")
    lines.append(f"NumberOfAtoms      {natoms}")
    lines.append(f"NumberOfSpecies    {len(species)}")
    lines.append("%block Chemical_Species_Label")
    for i, sym in enumerate(species, start=1):
        z = _Z.get(sym)
        if z is None:
            raise ValueError(f"Unknown element symbol '{sym}'. Add it to periodic table mapping.")
        lines.append(f" {i:2d}     {z:<3d}    {sym:<2s}")
    lines.append("%endblock Chemical_Species_Label")
    lines.append("")
    lines.append(f"PAO.BasisSize      {basis}")
    lines.append("")
    lines.append(f"NetCharge          {net_charge:.3f}")
    lines.append("")
    lines.append("AtomicCoordinatesFormat Ang")
    lines.append("AtomicCoorFormatOut     Ang")
    return "\n".join(lines) + "\n"

def fdf_coords(atoms: Sequence[Atom], species: List[str]) -> str:
    idx = {sym:i+1 for i,sym in enumerate(species)}
    out = []
    out.append("")
    out.append("%block AtomicCoordinatesAndAtomicSpecies")
    for a in atoms:
        out.append(f" {a.x:10.5f} {a.y:10.5f} {a.z:10.5f}  {idx[a.el]:d}  {a.el}")
    out.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    return "\n".join(out) + "\n"

def fdf_cell(cell_params: Tuple[float,float,float,float,float,float], kgrid: Tuple[int,int,int]) -> str:
    a,b,c,al,be,ga = cell_params
    out=[]
    out.append("")
    out.append("LatticeConstant   1.00  Ang")
    out.append("")
    out.append("%block LatticeParameters")
    out.append(f" {a:10.7f} {b:10.7f} {c:10.7f}  {al:8.5f} {be:8.5f} {ga:8.5f}")
    out.append("%endblock LatticeParameters")
    out.append("")
    out.append("%block kgrid_Monkhorst_Pack")
    out.append(f" {kgrid[0]:d}   0   0  0.0")
    out.append(f" 0   {kgrid[1]:d}   0  0.0")
    out.append(f" 0   0   {kgrid[2]:d}  0.0")
    out.append("%endblock kgrid_Monkhorst_Pack")
    return "\n".join(out) + "\n"

def fdf_dft_defaults(meshcutoff_ry: float, etemp_k: float, dm_tol: float, max_scf: int, solution: str="diagon") -> str:
    out=[]
    out.append("")
    out.append("########################################")
    out.append(f"SolutionMethod          {solution}")
    out.append(f"ElectronicTemperature   {etemp_k:.0f} K")
    out.append("XC.functional           GGA")
    out.append("XC.authors              PBE")
    out.append("SpinPolarized           .false.")
    out.append(f"MeshCutoff              {meshcutoff_ry:.1f} Ry")
    out.append("MinSCFIterations        10")
    out.append(f"MaxSCFIterations        {max_scf}")
    out.append(f"DM.Tolerance            {dm_tol:.2e}")
    out.append("DM.NumberPulay          8")
    out.append("DM.MixingWeight         0.03")
    return "\n".join(out) + "\n"

def fdf_output_defaults(mulliken: int, write_wf: bool, continuation: bool) -> str:
    out=[]
    out.append("")
    out.append("# Output options")
    out.append("WriteCoorInitial        .true.")
    out.append("WriteCoorStep           .true.")
    out.append("WriteCoorXmol           .true.")
    out.append("WriteForces             .true.")
    out.append("WriteKpoints            .false.")
    out.append("WriteEigenvalues        .true.")
    out.append("WriteKbands             .false.")
    out.append("WriteBands              .false.")
    out.append(f"WriteMullikenPop        {mulliken}")
    out.append("WriteMDhistory          .true.")
    out.append("WriteXML                .false.")
    out.append(f"WriteWaveFunctions      {'.true.' if write_wf else '.false.'}")
    out.append("")
    out.append("# Options for saving/reading information")
    out.append(f"DM.UseSaveDM            {'.true.' if continuation else '.false.'}")
    out.append(f"MD.UseSaveXV            {'.true.' if continuation else '.false.'}")
    out.append(f"MD.UseSaveCG            {'.true.' if continuation else '.false.'}")
    return "\n".join(out) + "\n"

def fdf_relax_opt(max_force: float, nsteps: int, variable_cell: bool) -> str:
    out=[]
    out.append("")
    out.append("# Molecular dynamics and relaxations")
    out.append("MD.TypeOfRun           CG")
    out.append(f"MD.MaxForceTol         {max_force:.3f} eV/Ang")
    out.append(f"MD.NumCGsteps          {nsteps}")
    out.append(f"MD.VariableCell        {'.true.' if variable_cell else '.false.'}")
    out.append("MD.MaxDispl            0.1058 Ang")
    return "\n".join(out) + "\n"
def fdf_scf_defaults() -> str:
    out = []
    out.append("")
    out.append("# Molecular dynamics and relaxations")
    out.append("MD.TypeOfRun           CG")
    out.append("MD.MaxForceTol         0.020 eV/Ang")
    out.append("MD.NumCGsteps          0")
    out.append("MD.VariableCell        .false.")
    out.append("MD.MaxDispl            0.1058 Ang")
    out.append("")
    out.append("# Output options")
    out.append("WriteCoorInitial        .true.")
    out.append("WriteCoorStep           .true.")
    out.append("WriteCoorXmol           .true.")
    out.append("WriteForces             .true.")
    out.append("WriteKpoints            .false.")
    out.append("WriteEigenvalues        .true.")
    out.append("WriteKbands             .false.")
    out.append("WriteBands              .false.")
    out.append("WriteMullikenPop        1")
    out.append("WriteMDhistory          .true.")
    out.append("WriteXML                .false.")
    out.append("WriteWaveFunctions      .true.")
    out.append("")
    out.append("# Options for saving/reading information")
    out.append("DM.UseSaveDM            .true.")
    out.append("MD.UseSaveXV            .true.")
    out.append("MD.UseSaveCG            .false.")
    out.append("SaveRho                 .true.")
    out.append("SaveDeltaRho            .true.")
    out.append("COOP.Write              .true.")
    out.append("WriteDenchar            .true.")
    return "\n".join(out) + "\n"

def fdf_md_common(init_step: int, final_step: int, dt_fs: float) -> str:
    out=[]
    out.append("")
    out.append(f"MD.Initial.Time.Step      {init_step}")
    out.append(f"MD.Final.Time.Step        {final_step}")
    out.append(f"MD.Length.Time.Step       {dt_fs:.3f} fs")
    return "\n".join(out) + "\n"

def fdf_md_nose(init_temp: float, target_temp: float, tau_fs: float, init_step:int, final_step:int, dt_fs:float) -> str:
    out=[]
    out.append("")
    out.append("MD.TypeOfRun              Nose")
    out.append(f"MD.InitialTemperature      {init_temp:.1f} K")
    out.append(f"MD.TargetTemperature       {target_temp:.1f} K")
    out.append(f"MD.TauRelax                {tau_fs:.1f} fs")
    out.append(fdf_md_common(init_step, final_step, dt_fs).strip("\n"))
    return "\n".join(out) + "\n"

def fdf_md_verlet(init_temp: float, init_step:int, final_step:int, dt_fs:float) -> str:
    out=[]
    out.append("")
    out.append("MD.TypeOfRun              Verlet")
    out.append(f"MD.InitialTemperature      {init_temp:.1f} K")
    out.append(fdf_md_common(init_step, final_step, dt_fs).strip("\n"))
    return "\n".join(out) + "\n"

def fdf_md_parrinello(target_pressure_gpa: float, pr_mass: float, init_step:int, final_step:int, dt_fs:float) -> str:
    out=[]
    out.append("")
    out.append("MD.TypeOfRun              ParrinelloRahman")
    out.append(f"MD.TargetPressure         {target_pressure_gpa:.3f} GPa")
    out.append(f"MD.ParrinelloRahmanMass   {pr_mass:.1f} Ry*fs**2")
    out.append(fdf_md_common(init_step, final_step, dt_fs).strip("\n"))
    return "\n".join(out) + "\n"

def fdf_md_npt(target_temp: float, target_pressure_gpa: float, tau_fs: float, nose_mass: float, pr_mass: float,
              init_step:int, final_step:int, dt_fs:float) -> str:
    out=[]
    out.append("")
    out.append("MD.TypeOfRun              NoseParrinelloRahman")
    out.append(f"MD.TargetTemperature       {target_temp:.1f} K")
    out.append(f"MD.TargetPressure          {target_pressure_gpa:.3f} GPa")
    out.append(f"MD.TauRelax                {tau_fs:.1f} fs")
    out.append(f"MD.NoseMass                {nose_mass:.1f} Ry*fs**2")
    out.append(f"MD.ParrinelloRahmanMass    {pr_mass:.1f} Ry*fs**2")
    out.append(fdf_md_common(init_step, final_step, dt_fs).strip("\n"))
    return "\n".join(out) + "\n"

def fdf_md_anneal(option: str, target_temp: float, target_pressure_gpa: float, tau_fs: float, bulk_modulus: float,
                 init_step:int, final_step:int, dt_fs:float) -> str:
    option = option.strip()
    if option not in ("Temperature","Pressure","TemperatureAndPressure","TemperatureandPressure"):
        raise ValueError("MD.AnnealOption must be one of Temperature, Pressure, TemperatureAndPressure")
    if option == "TemperatureandPressure":
        option = "TemperatureAndPressure"
    out=[]
    out.append("")
    out.append("MD.TypeOfRun              Anneal")
    out.append(f"MD.AnnealOption           {option}")
    out.append(f"MD.TargetTemperature      {target_temp:.1f} K")
    out.append(f"MD.TargetPressure         {target_pressure_gpa:.3f} GPa")
    out.append(f"MD.TauRelax               {tau_fs:.1f} fs")
    out.append(f"MD.BulkModulus            {bulk_modulus:.1f} Ry/Bohr**3")
    out.append(fdf_md_common(init_step, final_step, dt_fs).strip("\n"))
    return "\n".join(out) + "\n"


# -----------------------------
# Restart-from-outfile patching (legacy feature, option name changed)
# -----------------------------
def parse_siesta_out_relaxed(out_path: Path) -> Tuple[List[str], Optional[Tuple[float,float,float,float,float,float]]]:
    text = out_path.read_text(errors="ignore").splitlines()

    natoms = None
    for ln in text:
        if ln.startswith("NumberOfAtoms"):
            parts=ln.split()
            if len(parts)>=2:
                try:
                    natoms=int(parts[1])
                except Exception:
                    pass
    if natoms is None:
        raise ValueError(f"Could not find NumberOfAtoms in {out_path}")

    coord_block: Optional[List[str]] = None
    for i,ln in enumerate(text):
        w=ln.split()
        if len(w)>=2 and w[0]=="outcoor:" and w[1] in ("Atomic","Relaxed"):
            if i+1+natoms <= len(text):
                coord_block = text[i+1:i+1+natoms]

    cell = None
    params_line = None
    angles_line = None
    for ln in text:
        w=ln.split()
        if len(w)>=4 and w[0]=="outcell:" and w[3]=="modules":
            params_line = w
        if len(w)>=3 and w[0]=="outcell:" and w[2]=="angles":
            angles_line = w

    if params_line and angles_line:
        try:
            nums = [float(x) for x in params_line if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)]
            angs = [float(x) for x in angles_line if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)]
            if len(nums) >= 3 and len(angs) >= 3:
                a,b,c = nums[-3], nums[-2], nums[-1]
                alpha,beta,gamma = angs[-3], angs[-2], angs[-1]
                cell = (a,b,c,alpha,beta,gamma)
        except Exception:
            cell = None

    if coord_block is None:
        raise ValueError(f"Could not find outcoor: Atomic/Relaxed block in {out_path}")

    return coord_block, cell

def patch_fdf_from_out(old_fdf: str, coord_block: List[str], new_cell: Optional[Tuple[float,float,float,float,float,float]]) -> str:
    lines = old_fdf.splitlines()
    out=[]
    i=0
    while i < len(lines):
        ln = lines[i]
        out.append(ln)
        w = ln.split()
        if len(w)>=2 and w[0]=="%block" and w[1]=="AtomicCoordinatesAndAtomicSpecies":
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("%endblock"):
                i += 1
            for c in coord_block:
                ww=c.split()
                if len(ww) < 5:
                    continue
                out.append(f" {float(ww[0]):10.5f} {float(ww[1]):10.5f} {float(ww[2]):10.5f}  {int(ww[3])}  {ww[4]}")
            continue

        if new_cell and len(w)>=2 and w[0]=="%block" and w[1]=="LatticeParameters":
            if i+1 < len(lines):
                i += 1
                a,b,c,al,be,ga = new_cell
                out.append(f" {a:10.7f} {b:10.7f} {c:10.7f}  {al:8.5f} {be:8.5f} {ga:8.5f}")
            continue

        i += 1
    return "\n".join(out) + "\n"


# -----------------------------
# --extra KEY=VALUE overrides
# -----------------------------
# Non-interactive equivalent of CCpySIESTAInputGen.py's runtime prompt
# ("Add or modify option (ex: XC.functional=LDA, MaxSCFIterations=200)"),
# so batch/--all/Slurm workflows never block on input().

# A few fdf options take a number *and* a unit (e.g. "500 K"), and testers
# found it tedious to type the unit every time for these particular ones.
# If the value given is a bare number (no letters at all), silently append
# the usual/expected unit. Any value that already has a unit attached (even
# a different one, e.g. "300 eV" instead of "300 Ry", or "0.2 Bohr" instead
# of "0.2 Ang") is left completely untouched - so typing "KEY=value unit"
# with a space is exactly how to override the default unit.
EXTRA_OPTION_DEFAULT_UNITS: Dict[str, str] = {
    "electronictemperature": "K",
    "meshcutoff": "Ry",
    "md.maxforcetol": "eV/Ang",
    "md.maxdispl": "Ang",
}
_BARE_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _apply_default_unit(key: str, val: str) -> str:
    unit = EXTRA_OPTION_DEFAULT_UNITS.get(key.strip().lower())
    if unit and _BARE_NUMBER_RE.match(val.strip()):
        return f"{val.strip()} {unit}"
    return val


def build_canonical_key_map(tail_text: str) -> Dict[str, str]:
    """
    Scan a real generated fdf tail (see build_fdf_tail()) and build a
    {lowercase_key: canonical_cased_key} map from whatever option labels
    actually appear in it (e.g. "meshcutoff" -> "MeshCutoff"). This is
    derived from the live template rather than a hardcoded list, so it
    can't silently go stale if the fdf format changes later.
    """
    mapping: Dict[str, str] = {}
    in_block = False
    for ln in tail_text.split("\n"):
        stripped = ln.strip()
        low = stripped.lower()
        if low.startswith("%block"):
            in_block = True
            continue
        if low.startswith("%endblock"):
            in_block = False
            continue
        if in_block or not stripped or stripped.startswith("#"):
            continue
        first_tok = stripped.split()[0]
        mapping.setdefault(first_tok.lower(), first_tok)
    return mapping


def canonicalize_extra_keys(extra: "OrderedDict[str, str]", key_map: Dict[str, str]) -> "OrderedDict[str, str]":
    """
    For any --extra/interactive-prompt key that matches (case-insensitively)
    a known option already present in the fdf template, rewrite it to that
    option's canonical casing - so typing "meshcutoff=300" or "MESHCUTOFF=300"
    both land in the fdf as "MeshCutoff  300 Ry", same as if it had been
    typed with the usual capitalization. Keys with no template match (e.g.
    a genuinely new option like SCF.H.Tolerance) are passed through as typed.
    """
    result: "OrderedDict[str, str]" = OrderedDict()
    for k, v in extra.items():
        result[key_map.get(k.lower(), k)] = v
    return result


# ANSI yellow, used only for terminal display of what's been modified/added
# in the "Anything want to modify or add?" preview loop - never written to
# the actual .fdf file on disk.
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_COLOR_OK = sys.stdout.isatty()


def _colorize(text: str) -> str:
    if not _COLOR_OK:
        return text
    return f"{_ANSI_YELLOW}{text}{_ANSI_RESET}"


def highlight_fdf_lines(tail_text: str, keys_lower: Set[str]) -> str:
    """
    Return a copy of tail_text (the fdf preview shown in the modify loop)
    with any scalar option line whose key is in keys_lower wrapped in
    yellow ANSI codes, so it's obvious at a glance which lines have been
    touched by --extra / the interactive prompt so far. Only used for
    what's printed to the terminal - the real fdf_text written to disk
    is built separately in main() and never passed through this.
    """
    if not keys_lower:
        return tail_text
    out_lines: List[str] = []
    in_block = False
    for ln in tail_text.split("\n"):
        stripped = ln.strip()
        low = stripped.lower()
        if low.startswith("%block"):
            in_block = True
            out_lines.append(ln)
            continue
        if low.startswith("%endblock"):
            in_block = False
            out_lines.append(ln)
            continue
        if not in_block and stripped and not stripped.startswith("#"):
            first_tok = stripped.split()[0]
            if first_tok.lower() in keys_lower:
                out_lines.append(_colorize(ln))
                continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def parse_extra_options(s: Optional[str]) -> "OrderedDict[str, str]":
    result: "OrderedDict[str, str]" = OrderedDict()
    if not s:
        return result
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--extra entries must be KEY=VALUE (comma-separated), got: {chunk!r}")
        key, val = chunk.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"--extra has an empty key in: {chunk!r}")
        val = _apply_default_unit(key, val)
        result[key] = val
    return result


def apply_extra_overrides(fdf_text: str, extra: "OrderedDict[str, str]") -> str:
    """
    Apply --extra KEY=VALUE overrides to a generated fdf.

    SIESTA's fdf reader keeps the *first* occurrence of a duplicated label,
    so a key that already appears as a scalar option line (outside any
    %block) is replaced in place rather than appended again at the end
    (which would otherwise be silently ignored by SIESTA). Keys not already
    present are appended in a new '# User overrides' section.
    """
    if not extra:
        return fdf_text

    remaining: Dict[str, str] = dict(extra)
    lines = fdf_text.split("\n")
    out_lines: List[str] = []
    in_block = False
    for ln in lines:
        stripped = ln.strip()
        low = stripped.lower()
        if low.startswith("%block"):
            in_block = True
        elif low.startswith("%endblock"):
            in_block = False

        if not in_block and stripped and not stripped.startswith("#") and remaining:
            first_tok = stripped.split()[0]
            match_key = next((k for k in remaining if k.lower() == first_tok.lower()), None)
            if match_key is not None:
                out_lines.append(f"{match_key:<24s}{remaining.pop(match_key)}")
                continue

        out_lines.append(ln)

    result = "\n".join(out_lines)
    if remaining:
        extra_lines = ["", "# User overrides (--extra)"]
        for k, v in remaining.items():
            extra_lines.append(f"{k:<24s}{v}")
        result += "\n".join(extra_lines) + "\n"
    return result


def describe_kgrid_rule(args) -> str:
    """One-line human description of which kgrid rule is active, for the
    settings preview - the actual numeric kgrid is computed per-file when
    an auto rule is used, so we describe the *rule* here, not a value."""
    if getattr(args, "kgrid_metal_1d", False):
        return f"metal-1D rule (dk={args.dk}, kz_min={args.kz_min})"
    if getattr(args, "kgrid_from_car", False):
        return f"auto from cell (target={args.kgrid_target} Ang)"
    return f"fixed {tuple(args.kgrid)}"


def describe_pseudo_dir(args) -> str:
    if args.pseudo_dir:
        return f"{args.pseudo_dir}  (--pseudo-dir override)"
    return f"{POT_ALIASES[args.pot]}  (-pot={args.pot})"


def extract_fdf_value(fdf_text: str, key: str) -> Optional[str]:
    """Find a scalar 'KEY value...' line (outside any %block) in fdf_text
    and return everything after the key, or None if that key isn't there."""
    in_block = False
    for ln in fdf_text.split("\n"):
        s = ln.strip()
        low = s.lower()
        if low.startswith("%block"):
            in_block = True
            continue
        if low.startswith("%endblock"):
            in_block = False
            continue
        if in_block or not s or s.startswith("#"):
            continue
        parts = s.split(None, 1)
        if parts and parts[0].lower() == key.lower():
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def format_settings_preview(mode: str, basis: str, tail_text: str,
                             kgrid_desc: str, pseudo_desc: str,
                             highlight_keys: Optional[Set[str]] = None) -> str:
    """
    Mirrors CCpyVASPInputGen.py's INCAR dump ("Here are the current INCAR
    options") but for car2siesta's own settings, shown before the per-file
    loop (mode/basis/kgrid-rule/etc. don't vary per file - only SystemLabel
    and the actual computed kgrid numbers do).

    Values for keys that also appear in tail_text (the real generated fdf
    text below the atomic coordinates - see build_fdf_tail()) are read
    directly from it rather than from the raw preset dict, so this summary
    box can never disagree with what --extra overrides actually produced.

    If highlight_keys is given (lowercase fdf key names), rows whose key
    is in that set are printed in yellow - used to flag what's been
    modified/added so far in the interactive modify loop.
    """
    highlight_keys = highlight_keys or set()

    def v(key: str) -> str:
        got = extract_fdf_value(tail_text, key)
        return got if got is not None else "(not set)"

    def row(label: str, key: str) -> str:
        text = f"{label:<22s}: {v(key)}"
        return _colorize(text) if key.lower() in highlight_keys else text

    lines: List[str] = []
    lines.append("")
    lines.append("# " + "-" * 50 + " #")
    lines.append("#             Current SIESTA options              #")
    lines.append("# " + "-" * 50 + " #")
    lines.append(f"{'Mode':<22s}: {mode}")
    lines.append(f"{'PAO.BasisSize':<22s}: {basis}")
    lines.append(f"{'kgrid':<22s}: {kgrid_desc}")
    lines.append(row("MeshCutoff", "MeshCutoff"))
    lines.append(row("ElectronicTemperature", "ElectronicTemperature"))
    lines.append(row("DM.Tolerance", "DM.Tolerance"))
    lines.append(row("MaxSCFIterations", "MaxSCFIterations"))

    m = mode.lower()
    if m == "opt":
        lines.append(row("MD.MaxForceTol", "MD.MaxForceTol"))
        lines.append(row("MD.NumCGsteps", "MD.NumCGsteps"))
        lines.append(row("MD.MaxDispl", "MD.MaxDispl"))
    elif m == "nve":
        lines.append(row("MD.InitialTemperature", "MD.InitialTemperature"))
        lines.append(row("MD.Length.Time.Step", "MD.Length.Time.Step"))
    elif m == "nvt":
        lines.append(row("MD.TargetTemperature", "MD.TargetTemperature"))
        lines.append(row("MD.TauRelax", "MD.TauRelax"))
    elif m == "npe":
        lines.append(row("MD.TargetPressure", "MD.TargetPressure"))
    elif m == "npt":
        lines.append(row("MD.TargetTemperature", "MD.TargetTemperature"))
        lines.append(row("MD.TargetPressure", "MD.TargetPressure"))
    elif m == "anneal":
        lines.append(row("MD.AnnealOption", "MD.AnnealOption"))
        lines.append(row("MD.TargetTemperature", "MD.TargetTemperature"))
        lines.append(row("MD.TargetPressure", "MD.TargetPressure"))

    lines.append(f"{'pseudo_dir':<22s}: {pseudo_desc}")
    lines.append("# " + "-" * 50 + " #")
    return "\n".join(lines)


def maybe_prompt_for_extra(cli_extra: "OrderedDict[str, str]", mode: str, basis: str,
                            preset: Dict[str, Any], args,
                            continuation: bool, variable_cell: bool) -> "OrderedDict[str, str]":
    """
    Restores CCpySIESTAInputGen.py's interactive prompt, but now also shows
    a full settings preview - both the summary box AND the actual fdf text
    that will end up below the atomic coordinates block (see
    build_fdf_tail()) - matching CCpyVASPInputGen.py's INCAR dump. Unlike
    the original single-shot prompt, it keeps re-showing the box + fdf
    preview and asking again after every change, until you explicitly type
    "n" to finish - so you can add/adjust several settings one at a time
    and see exactly how each one lands in the real fdf before moving on.

    Only asks when it's actually safe to:
      - the user did NOT already give --extra (or an 'extra:' config value) -
        if they did, that answer is used and we don't ask/print again.
      - stdin is a real terminal (sys.stdin.isatty()). Under Slurm/cron/pipes
        stdin isn't a tty, so this is skipped silently instead of hanging
        the job forever waiting for input that will never come.
    NOTE: only typing "n" (case-insensitive) ends the loop now - pressing
    Enter with nothing typed just asks again, it does not finish early.
    """
    # Built once regardless of path below, since it doubles as the source
    # for canonical-casing known fdf keys typed in lowercase (see
    # build_canonical_key_map / canonicalize_extra_keys).
    base_tail = build_fdf_tail(mode, preset, continuation, variable_cell)
    key_map = build_canonical_key_map(base_tail)

    if cli_extra:
        return canonicalize_extra_keys(cli_extra, key_map)
    if not sys.stdin.isatty():
        return OrderedDict()

    kgrid_desc = describe_kgrid_rule(args)
    pseudo_desc = describe_pseudo_dir(args)

    accumulated: "OrderedDict[str, str]" = OrderedDict()
    while True:
        current_tail = apply_extra_overrides(base_tail, accumulated) if accumulated else base_tail
        highlight_keys = {k.lower() for k in accumulated}  # what's been touched so far - shown in yellow

        print(format_settings_preview(mode, basis, current_tail, kgrid_desc, pseudo_desc, highlight_keys))
        print("\n# ---- fdf preview (everything below the atomic coordinates block) ---- #")
        print(highlight_fdf_lines(current_tail, highlight_keys))
        print("# ------------------------------------------------------------------------ #\n")

        try:
            raw = input(
                'Anything want to modify or add? '
                '(ex: MaxSCFIterations=50,XC.functional=LDA,MeshCutoff=300,MD.MaxDispl=0.2 Bohr) '
                'else, enter "n" to finish\n: '
            ).strip()
        except EOFError:
            break

        if raw.lower() == "n":
            break
        if not raw:
            continue  # nothing typed - ask again, only "n" actually finishes

        try:
            new_entries = parse_extra_options(raw)
        except argparse.ArgumentTypeError as exc:
            print(f"  (ignored: {exc})")
            continue

        new_entries = canonicalize_extra_keys(new_entries, key_map)
        accumulated.update(new_entries)  # later entries for the same KEY win
        # loop back around: box + fdf preview reprint at the top reflecting
        # everything accumulated so far, then ask again.

    return accumulated


# -----------------------------
# Slurm mpi.sh generator
# -----------------------------
def write_slurm_mpi_sh(
    job_dir: Path,
    job_name: str,
    partition: str,
    ntasks: int,
    nodes: int,
    time_limit: str,
    mem: Optional[str],
    siesta_cmd: str,
    fdf_name: str,
    extra_lines: Optional[List[str]] = None,
) -> None:
    extra_lines = extra_lines or []
    mem_line = f"#SBATCH --mem={mem}" if mem else ""
    extra_block = "\n".join(extra_lines) if extra_lines else ""
    sh = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time={time_limit}
{mem_line}

set -euo pipefail

{extra_block}

echo "===== START: $(date) ====="
echo "WORKDIR: $(pwd)"

cd "$SLURM_SUBMIT_DIR"
echo "Running: {siesta_cmd} < {fdf_name}"
srun {siesta_cmd} < "{fdf_name}" > "{job_name}.out"

rc=$?
echo "SIESTA exit code: $rc"
echo "===== END: $(date) ====="
exit $rc
"""
    (job_dir / "mpi.sh").write_text(sh)
    (job_dir / "mpi.sh").chmod(0o755)


# -----------------------------
# Restart workflow (no .car) - imported idea from siesta_Band-DOS.py
# -----------------------------
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
    raise SystemExit("ERROR: failed to auto-extract SystemLabel. Please specify it with --label.")

def safe_symlink_as(src: Path, dst_path: Path) -> None:
    """Create/replace a symlink at dst_path pointing to src (if src exists)."""
    if not src.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.symlink_to(src)


def safe_symlink(src: Path, dst_dir: Path) -> None:
    """Backward-compatible helper: link into dst_dir with the same basename."""
    safe_symlink_as(src, dst_dir / src.name)


def link_common_files(top: Path, dst_dir: Path, src_label: str, dst_label: Optional[str] = None) -> None:
    """
    Link common restart files.

    - Source files are searched as:   <top>/<src_label>.(DM|XV|WFSX|EIG)
    - Destination link names become: <dst_dir>/<dst_label>.(DM|XV|WFSX|EIG)
      If dst_label is None, it defaults to src_label.

    Pseudopotentials (*.psf/*.psml) are linked by their original filenames.
    """
    if dst_label is None:
        dst_label = src_label

    for ext in ("DM", "XV", "WFSX", "EIG"):
        src = top / f"{src_label}.{ext}"
        dst = dst_dir / f"{dst_label}.{ext}"
        safe_symlink_as(src, dst)

    for p in list(top.glob("*.psf")) + list(top.glob("*.psml")):
        safe_symlink(p, dst_dir)


def link_pseudos_from_dir(pseudo_dir: str, dst_dir: Path):
    """Symlink every .psf/.psml found in pseudo_dir into dst_dir. Caller
    resolves which single directory to use (resolve_pseudo_dir(args):
    --pseudo-dir override, else -pot=dojo/psf/psfold via POT_ALIASES)."""
    src_dir = Path(pseudo_dir)
    if not src_dir.is_dir():
        return
    for p in list(src_dir.glob('*.psf')) + list(src_dir.glob('*.psml')):
        safe_symlink(p, dst_dir)

def make_restart_dir(top: Path, label: str, base_fdf: Path, args) -> None:
    dst = top / "Restart"
    dst.mkdir(exist_ok=True)

    shutil.copy2(base_fdf, dst / f"{label}.fdf")
    link_common_files(top, dst, label, dst_label=f"{label}_R")
    link_pseudos_from_dir(resolve_pseudo_dir(args), dst)

    src = dst / f"{label}.fdf"
    out = dst / f"{label}_R.fdf"
    lines = src.read_text(errors="ignore").splitlines()
    new = []
    have_dm = have_xv = False
    have_sysname = have_syslabel = False
    for ln in lines:
        if re.match(r"(?i)^\s*SystemName\b", ln):
            new.append(f"SystemName              {label}_R")
            have_sysname = True
        elif re.match(r"(?i)^\s*SystemLabel\b", ln):
            new.append(f"SystemLabel             {label}_R")
            have_syslabel = True
        elif re.match(r"(?i)^\s*DM\.UseSaveDM\b", ln):
            new.append("DM.UseSaveDM            .true.")
            have_dm = True
        elif re.match(r"(?i)^\s*MD\.UseSaveXV\b", ln):
            new.append("MD.UseSaveXV            .true.")
            have_xv = True
        else:
            new.append(ln)

    if not have_sysname:
        new.append(f"SystemName              {label}_R")
    if not have_syslabel:
        new.append(f"SystemLabel             {label}_R")
    if not have_dm:
        new.append("DM.UseSaveDM            .true.")
    if not have_xv:
        new.append("MD.UseSaveXV            .true.")
    out.write_text("\n".join(new) + "\n")

    write_slurm_mpi_sh(
        job_dir=dst,
        job_name=f"{label}_R", 
        partition=args.partition,
        ntasks=args.ntasks,
        nodes=args.nodes,
        time_limit=args.time_limit,
        mem=args.mem,
        siesta_cmd=args.siesta_cmd,
        fdf_name=out.name,
        extra_lines=[],
    )


# -----------------------------
# Main generation logic
# -----------------------------
def parse_kgrid(s: str) -> Tuple[int,int,int]:
    parts = re.split(r"[,\s]+", s.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("kgrid must be like '1 1 1' or '2,2,2'")
    try:
        k = tuple(int(x) for x in parts)
    except Exception:
        raise argparse.ArgumentTypeError("kgrid must be integers")
    if any(x <= 0 for x in k):
        raise argparse.ArgumentTypeError("kgrid values must be positive")
    return k  # type: ignore

def parse_cell(s: str) -> Tuple[float,float,float,float,float,float]:
    parts = re.split(r"[,\s]+", s.strip())
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("cell must be 'a b c alpha beta gamma'")
    try:
        vals = tuple(float(x) for x in parts)
    except Exception:
        raise argparse.ArgumentTypeError("cell values must be numbers")
    return vals  # type: ignore

def build_fdf_tail(mode: str, preset: Dict[str, float], continuation: bool, variable_cell: bool) -> str:
    """
    Builds everything that goes *after* the atomic-coordinates block: DFT
    defaults, the mode-specific relax/MD block, and output/save-reading
    options. This never depends on the structure itself (atoms/cell), so it
    can be computed and shown in the --extra settings-preview loop before
    any structure file is even read, and re-derived cheaply every time the
    user changes something there.
    """
    fdf = ""
    fdf += fdf_dft_defaults(
        meshcutoff_ry=preset.get("meshcutoff", 300.0),
        etemp_k=preset.get("etemp", 300.0),
        dm_tol=preset.get("dm_tol", 1e-5),
        max_scf=int(preset.get("max_scf", 200)),
        solution=preset.get("solution", "diagon"),
    )

    mode = mode.lower()
    if mode == "opt":
        fdf += fdf_relax_opt(
            max_force=preset.get("max_force", 0.02),
            nsteps=int(preset.get("cg_steps", 500)),
            variable_cell=variable_cell,
        )
    elif mode == "nvt":
        fdf += fdf_md_nose(
            init_temp=preset.get("init_temp", 300.0),
            target_temp=preset.get("target_temp", 300.0),
            tau_fs=preset.get("tau", 200.0),
            init_step=int(preset.get("init_step", 1)),
            final_step=int(preset.get("final_step", 5000)),
            dt_fs=preset.get("dt", 1.0),
        )
    elif mode == "nve":
        fdf += fdf_md_verlet(
            init_temp=preset.get("init_temp", 300.0),
            init_step=int(preset.get("init_step", 1)),
            final_step=int(preset.get("final_step", 5000)),
            dt_fs=preset.get("dt", 1.0),
        )
    elif mode == "npe":
        fdf += fdf_md_parrinello(
            target_pressure_gpa=preset.get("target_pressure", 0.0),
            pr_mass=preset.get("pr_mass", 100.0),
            init_step=int(preset.get("init_step", 1)),
            final_step=int(preset.get("final_step", 5000)),
            dt_fs=preset.get("dt", 1.0),
        )
    elif mode == "npt":
        fdf += fdf_md_npt(
            target_temp=preset.get("target_temp", 300.0),
            target_pressure_gpa=preset.get("target_pressure", 0.0),
            tau_fs=preset.get("tau", 200.0),
            nose_mass=preset.get("nose_mass", 100.0),
            pr_mass=preset.get("pr_mass", 100.0),
            init_step=int(preset.get("init_step", 1)),
            final_step=int(preset.get("final_step", 5000)),
            dt_fs=preset.get("dt", 1.0),
        )
    elif mode == "anneal":
        fdf += fdf_md_anneal(
            option=str(preset.get("anneal_option", "Temperature")),
            target_temp=preset.get("target_temp", 800.0),
            target_pressure_gpa=preset.get("target_pressure", 0.0),
            tau_fs=preset.get("tau", 100.0),
            bulk_modulus=preset.get("bulk_modulus", 100.0),
            init_step=int(preset.get("init_step", 1)),
            final_step=int(preset.get("final_step", 50000)),
            dt_fs=preset.get("dt", 2.0),
        )
    elif mode == "scf":
        fdf += fdf_scf_defaults()
        return fdf
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fdf += fdf_output_defaults(
        mulliken=int(preset.get("mulliken", 1)),
        write_wf=bool(preset.get("write_wf", True)),
        continuation=continuation,
    )
    return fdf


def build_fdf(
    car: CarData,
    system_label: str,
    system_name: str,
    basis: str,
    net_charge: float,
    kgrid: Tuple[int,int,int],
    cell_params: Optional[Tuple[float,float,float,float,float,float]],
    vacuum: float,
    sort_order: Optional[List[str]],
    mode: str,
    preset: Dict[str, float],
    continuation: bool,
    variable_cell: bool,
) -> str:
    atoms = sort_atoms(car.atoms, sort_order)

    shift = (0.0, 0.0, 0.0)
    if cell_params is None:
        if car.cell_params is not None:
            cell_params = car.cell_params
        else:
            cell_params, shift = make_orthorhombic_cell_from_atoms(atoms, vacuum=vacuum)
            atoms = apply_shift(atoms, shift)

    species = []
    for a in atoms:
        if a.el not in species:
            species.append(a.el)

    fdf = ""
    fdf += fdf_header(system_name, system_label, natoms=len(atoms), species=species, basis=basis, net_charge=net_charge)
    fdf += fdf_coords(atoms, species)
    fdf += fdf_cell(cell_params, kgrid)
    fdf += build_fdf_tail(mode, preset, continuation, variable_cell)
    return fdf

# -----------------------------
# Config file (YAML) support
# -----------------------------
DEFAULT_CONFIG_FILENAME = "siesta_default.yaml"


def fallback_config_search_dirs() -> List[Path]:
    """
    Directories to check for DEFAULT_CONFIG_FILENAME when the current
    working directory doesn't have one. This lets a lab keep a single
    shared siesta_default.yaml next to the installed script (e.g. in
    .../CCpy/SIESTA/) instead of copying it into every calculation folder.

    Two sources are tried, since where __file__ points depends on *how*
    the script was launched:
      1) The directory this .py file itself lives in - correct when the
         script is run directly (`python3 .../CCpy/SIESTA/CCpySIESTAInputGen.py`)
         or via a symlink to that path.
      2) The installed CCpy package's SIESTA/ subdirectory - needed when
         launched through the easy_install-style stub in envs/CCpy/bin/,
         which runs the script via pkg_resources.run_script() out of
         EGG-INFO/scripts/. In that case __file__ points at the EGG-INFO
         copy, not CCpy/SIESTA/, so this is resolved via `import CCpy`
         instead (works as long as the CCpy package itself is installed
         in the running interpreter, which it is inside the CCpy env).
    """
    dirs: List[Path] = []

    try:
        dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    try:
        import CCpy  # type: ignore
        ccpy_dir = Path(CCpy.__file__).resolve().parent
        dirs.append(ccpy_dir / "SIESTA")
    except Exception:
        pass

    # de-duplicate while preserving order
    seen = set()
    unique_dirs: List[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def resolve_default_config_path() -> Optional[Path]:
    """
    Look for DEFAULT_CONFIG_FILENAME, in priority order:
      1) current working directory (per-calculation override)
      2) shared fallback locations (see fallback_config_search_dirs())
    Returns None if not found anywhere.
    """
    cwd_candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if cwd_candidate.exists():
        return cwd_candidate

    for d in fallback_config_search_dirs():
        candidate = d / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return candidate

    return None


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into a flat dict of {dest_name: value}."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: using a config file requires PyYAML. Please run: pip install pyyaml"
        ) from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"ERROR: an error occurred while reading the config file: {path} ({exc})")
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: the config file must be a mapping of the form 'key: value': {path}")
    # Accept both 'kgrid-from-car' and 'kgrid_from_car' style keys.
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> List[str]:
    """
    Use values from a config dict as argparse defaults, honoring each
    action's declared `type=` so options like --kgrid/--cell keep working
    whether the YAML value is a string ("1 1 1") or a list ([1, 1, 1]).
    Actual CLI arguments (if given on the command line) still win over
    these defaults, since set_defaults() only changes what's used when
    the user does NOT pass the flag.
    """
    dests = {action.dest: action for action in parser._actions}
    applied: List[str] = []
    unknown: List[str] = []

    for key, raw_value in config.items():
        action = dests.get(key)
        if action is None or key in ("help",):
            unknown.append(key)
            continue

        value = raw_value
        if key == "kgrid" and isinstance(raw_value, (list, tuple)):
            value = tuple(int(x) for x in raw_value)
        elif key == "cell" and isinstance(raw_value, (list, tuple)):
            value = tuple(float(x) for x in raw_value)
        elif isinstance(raw_value, str) and action.type is not None:
            value = action.type(raw_value)
        # store_true flags / plain numbers/bools from YAML are used as-is.

        parser.set_defaults(**{key: value})
        applied.append(key)

    if unknown:
        print(f"WARNING: unknown options in the config file are ignored: {', '.join(unknown)}")

    return applied


def main() -> int:
    mode_number_raw, remaining_argv = extract_mode_number(sys.argv[1:])

    ap = argparse.ArgumentParser(
        description="Generate SIESTA .fdf (and optional Slurm mpi.sh + gen[id].list) from BIOSYM/Materials Studio .car, or .cif/POSCAR/CONTCAR (via pymatgen) structure files",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        " --------------------------------------------------------------------------------------------------------------------------- \n"
        "    Default : --make-dirs --write-mpi --partition 72core --ntasks 24 --nodes 1  \n"
        " --------------------------------------------------------------------------------------------------------------------------- \n"
        "0) mode_number: 1=opt 2=nvt 3=nve 4=npe 5=npt 6=anneal 7=scf (opt is always 1) \n"
        "1)* no-sort element & kgrid: 20/cellparam \n"
        "   python CCpySIESTAInputGen.py 1 -pot=psf --kgrid-from-car --max-force 0.02 --no-sort -basis=sz  SWNT7-6.car \n"
        "2)* Restart \n"
        "   python CCpySIESTAInputGen.py --restart-from-outdir  \n"
        "           DM.MixingWeight      0.02  (default = 0.03) \n"
        "3) sort & kgrid: 20/cellparam \n"
        "   python CCpySIESTAInputGen.py 1 -pot=dojo --kgrid-from-car --max-force 0.02 -basis=sz SWNT7-6.car \n"
        "4) kgrid: Delta_k = 0.02 (default) / 0.01, 0.02, 0.03 \n" 
        "   python CCpySIESTAInputGen.py 1 --kgrid-metal-1d --dk 0.02 -pot=psf --max-force 0.02 -basis=sz SWNT7-6.car \n"
        "5) cell-param opt & kgrid  \n"
        "   --kgrid 3 3 3  --variable-cell --element-order \n"
        "6) Calculation Time Limit \n"
        "   --time   :  time_limit default 7 days (7-00:00:00) !!! \n"
        "7) CIF / POSCAR / CONTCAR input (pymatgen) \n"
        "   python CCpySIESTAInputGen.py 1 -basis=sz structure.cif \n"
        "   python CCpySIESTAInputGen.py 1 -basis=sz POSCAR \n"
        "8) Fewer options via a config file (see siesta_default.yaml.example, auto-loaded if siesta_default.yaml is in cwd) \n"
        "   python CCpySIESTAInputGen.py 1 SWNT7-6.car \n"
        "   python CCpySIESTAInputGen.py 1 --config lab_defaults.yaml SWNT7-6.car \n"
        "9) Option override (non-interactive, an interactive run without --extra asks automatically) \n"
        "   python CCpySIESTAInputGen.py 1 --extra \"XC.functional=LDA,MaxSCFIterations=50\" structure.cif \n"
        "10) With no file given, scans the current folder and lets you pick by number \n"
        "   python CCpySIESTAInputGen.py 1 -basis=sz \n"
        " --------------------------------------------------------------------------------------------------------------------------- \n"
    )

    # (NEW) config file support - lets a lab keep one shared set of defaults
    # instead of retyping the same long command every time.
    ap.add_argument("--config", type=str, default=None,
                    help=f"YAML file with default option values (see {DEFAULT_CONFIG_FILENAME}.example). "
                         f"If omitted, auto-loads ./{DEFAULT_CONFIG_FILENAME} when present. "
                         f"CLI arguments always override the config file.")

    # (NEW) no-car restart workflow
    ap.add_argument("--restart-from-outdir", action="store_true",
                    help="(no .car) In current directory, auto-detect label from *.fdf and create Restart/ with <label>_restart.fdf + mpi.sh")

    # NOTE: the leading mode-number positional (1-7, e.g. `prog 1 ...`) is
    # deliberately NOT registered here as an argparse positional. argparse
    # can't cleanly handle two separate positional actions (mode_number,
    # car) when optional flags are interspersed between them, e.g.
    # `prog 1 -basis=sz file.car` would make argparse treat "file.car" as an
    # unrecognized argument. Instead, extract_mode_number() below peels the
    # leading digit off sys.argv *before* argparse ever sees it, so argparse
    # only has to deal with a single positional action (car) - which it
    # handles fine with interspersed optionals.
    ap.add_argument("car", nargs="*", help="Input .car / .cif / POSCAR / CONTCAR structure files. If omitted, use --all to process them (or use --restart-from-outdir)")
    ap.add_argument("--all", action="store_true", help="Process all *.car, *.cif, POSCAR, CONTCAR files in the current directory")

    ap.add_argument("-basis", "--basis", dest="basis", type=lambda s: s.upper(),
                    choices=["SZ","SZP","DZ","DZP","TZ","TZP"], default="SZ",
                    help="PAO.BasisSize, case-insensitive (e.g. -basis=sz). Default: SZ.")
    ap.add_argument("--label", help="Override SystemLabel (default: basename or auto-detected label for --restart-from-outdir)")
    ap.add_argument("--name", help="Override SystemName (default: SystemLabel)")

    ap.add_argument("--kgrid", type=parse_kgrid, default=(1,1,1), help="Monkhorst-Pack kgrid: 'kx ky kz'")
    ap.add_argument("--kgrid-from-car", action="store_true", help="Auto kgrid from CAR PBC and force odd")
    ap.add_argument("--kgrid-target", type=float, default=20.0, help="Target length for auto kgrid")
    ap.add_argument("--kgrid-metal-1d", action="store_true",
                    help="Override kz using metallic-1D rule: kz=ceil((2*pi/c)/dk), then kz>=kz-min and odd. Uses c from CAR PBC or --cell or inferred cell.")
    ap.add_argument("--dk", type=float, default=0.02,
                    help="Target k-point spacing (1/Angstrom) for --kgrid-metal-1d. Typical: 0.02 (band), 0.01 (DOS).")
    ap.add_argument("--kz-min", type=int, default=7,
                    help="Minimum kz when using --kgrid-metal-1d (prevents kz=1 for long supercells).")
    ap.add_argument("--cell", type=parse_cell, help="Cell parameters: 'a b c alpha beta gamma' (Ang, degrees). If omitted, infer from coords + vacuum.")
    ap.add_argument("--vacuum", type=float, default=20.0, help="Vacuum padding (Ang) used when inferring cell from coords.")
    ap.add_argument("--variable-cell", action="store_true", help="For OPT (CG), set MD.VariableCell .true.")

    ap.add_argument("--element-order", help="Comma-separated element order for sorting, e.g. 'Zn,O,H'. If omitted, alphabetical.")
    ap.add_argument("--no-sort", action="store_true", help="Do not re-order atoms (keep CAR order)")

    # DFT defaults
    ap.add_argument("--meshcutoff", type=float, default=300.0, help="MeshCutoff (Ry)")
    ap.add_argument("--etemp", type=float, default=300.0, help="ElectronicTemperature (K)")
    ap.add_argument("--dm-tol", type=float, default=1e-5, help="DM.Tolerance")
    ap.add_argument("--max-scf", type=int, default=200, help="MaxSCFIterations")

    # OPT defaults
    ap.add_argument("-maxforce", "--max-force", dest="max_force", type=float, default=0.02,
                    help="MD.MaxForceTol (eV/Ang) for OPT, e.g. -maxforce=0.02. Typical values: "
                         "0.02 (default, normal relax), 0.05 (looser/rougher), 0.01 (very tight/final check). "
                         "Lower = stricter force convergence = more SCF/CG steps needed.")
    ap.add_argument("--cg-steps", type=int, default=500, help="MD.NumCGsteps for OPT")

    # MD defaults
    ap.add_argument("--init-step", type=int, default=1, help="MD.Initial.Time.Step")
    ap.add_argument("--final-step", type=int, default=5000, help="MD.Final.Time.Step")
    ap.add_argument("--dt", type=float, default=1.0, help="MD.Length.Time.Step (fs)")
    ap.add_argument("--init-temp", type=float, default=300.0, help="MD.InitialTemperature (K) (NVE/NVT)")
    ap.add_argument("--target-temp", type=float, default=300.0, help="MD.TargetTemperature (K) (NVT/NPT/Anneal)")
    ap.add_argument("--tau", type=float, default=200.0, help="MD.TauRelax (fs) (NVT/NPT/Anneal)")
    ap.add_argument("--target-pressure", type=float, default=0.0, help="MD.TargetPressure (GPa) (NPE/NPT/Anneal/variable-cell OPT)")
    ap.add_argument("--nose-mass", type=float, default=100.0, help="MD.NoseMass (Ry*fs**2) (NPT)")
    ap.add_argument("--pr-mass", type=float, default=100.0, help="MD.ParrinelloRahmanMass (Ry*fs**2) (NPE/NPT)")
    ap.add_argument("--anneal-option", choices=["Temperature","Pressure","TemperatureAndPressure"], default="Temperature",
                    help="MD.AnnealOption (Anneal)")

    # Restart/continuation (CAR path mode)
    ap.add_argument("--continuation", action="store_true",
                    help="Turn on DM.UseSaveDM / MD.UseSaveXV / MD.UseSaveCG in output (for restarts/continuations).")
    ap.add_argument("--patch-from-out", type=str,
                    help="Given a SIESTA .out file, patch coordinates (and cell if available) in the generated fdf using last outcoor/outcell info.")
    ap.add_argument("--extra", type=parse_extra_options, default=None,
                    help="Comma-separated KEY=VALUE pairs to override/append in the generated fdf, "
                         "e.g. --extra 'XC.functional=LDA,MaxSCFIterations=50,MeshCutoff=300,"
                         "MD.MaxDispl=0.2 Bohr'. ElectronicTemperature/MeshCutoff/MD.MaxForceTol/"
                         "MD.MaxDispl get their usual unit auto-appended if you give a bare number; "
                         "add your own unit (space-separated) to override it. Known option names "
                         "are matched case-insensitively. If omitted AND running in an interactive "
                         "terminal, you'll be prompted once instead (same as CCpySIESTAInputGen.py); "
                         "under Slurm/cron/pipes this is skipped automatically so batch runs never "
                         "hang waiting for input.")

    # Output layout
    ap.add_argument("--outdir", type=str, default=".", help="Where to write outputs (or job dirs)")
    ap.add_argument("-pot", "--pot", dest="pot", type=lambda s: s.lower(),
                    choices=sorted(POT_ALIASES.keys()), default=DEFAULT_POT,
                    help="Named pseudopotential set: " +
                         ", ".join(f"{k}={v}" for k, v in POT_ALIASES.items()) +
                         f" (default: {DEFAULT_POT}). Overridden entirely by --pseudo-dir if given.")
    ap.add_argument("--pseudo-dir", default=None,
                    help="Explicit directory containing pseudopotentials (.psf/.psml) to copy/symlink "
                         "into job dirs (and Restart/). If given, this always wins over -pot=. "
                         "If neither is given, -pot defaults to 'dojo'.")
    ap.add_argument("--make-dirs", action="store_true", default=True, help="Create one directory per job (SystemLabel/) and put inputs inside.")
    ap.add_argument("--id", type=int, help="If set with --make-dirs, write gen<ID>.list listing the created job dirs.")

    # Slurm script
    ap.add_argument("--write-mpi", action="store_true", default=True, help="With --make-dirs (or --restart-from-outdir), write Slurm mpi.sh.")
    ap.add_argument("--partition", default="72core", help="Slurm partition name")
    ap.add_argument("--ntasks", type=int, default=24, help="Slurm -n (MPI ranks)")
    ap.add_argument("--nodes", type=int, default=1, help="Slurm -N")
    ap.add_argument("--time", dest="time_limit", default="7-00:00:00", help="Slurm --time")
    ap.add_argument("--mem", default=None, help="Slurm --mem (e.g. 64G). If omitted, do not set.")
    ap.add_argument("--siesta-cmd", default="/opt/siesta/siesta-5.4.2/siesta-mkl-mpi/bin/siesta", help="Command used in srun")
    ap.add_argument("--submit", action="store_true", help="After writing mpi.sh, run 'sbatch mpi.sh' in each job dir")

    # ---- Load YAML config (if any) and use it to set CLI defaults ----
    # Lightweight pre-parse just to discover --config, without triggering
    # the full parser's validation (choices/required) before defaults are set.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(remaining_argv)

    config_path: Optional[Path] = None
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            ap.error(f"--config file not found: {config_path}")
    else:
        config_path = resolve_default_config_path()

    if config_path is not None:
        cfg = load_yaml_config(config_path)
        apply_config_defaults(ap, cfg)
        print(f"[car2siesta] Using config defaults from {config_path}")

    args = ap.parse_args(remaining_argv)

    # (1) no-car restart workflow
    if args.restart_from_outdir:
        top = Path.cwd()
        label = args.label or auto_detect_label(top)
        base_fdf = top / f"{label}.fdf"
        if not base_fdf.exists():
            raise SystemExit(f"ERROR: {base_fdf} not found (if SystemLabel differs from the filename, match it with --label)")
        make_restart_dir(top, label, base_fdf, args)
        if args.submit:
            os.system("cd Restart && sbatch mpi.sh")
        return 0

    # Validate the mode number BEFORE touching any files at all - if it's
    # missing/invalid, show the usage reminder and exit immediately. This
    # must come before file discovery below: otherwise a bare invocation
    # with no mode number would first trigger the interactive file picker
    # (or --all's directory scan) for nothing, only to fail afterwards.
    mode = mode_number_to_name(mode_number_raw)
    if mode is None:
        print_mode_number_usage(ap.prog)
        return 0
    basis = args.basis

    # (2) structure -> FDF generation workflow
    cars: List[Path] = []
    if args.all:
        auto_globs = ["*.car", "*.cif"] + list(POSCAR_LIKE_NAMES) + [f"{n}.*" for n in POSCAR_LIKE_NAMES]
        found: List[str] = []
        for pat in auto_globs:
            found.extend(glob.glob(pat))
        cars = [Path(p) for p in sorted(set(found))]
    elif not args.car and sys.stdin.isatty():
        # No file given on the command line and --all not requested: offer
        # the numbered picker (mirrors CCpySIESTAInputGen.py/CCpyVASPInputGen.py)
        # instead of just erroring out. Non-interactive runs (Slurm/cron/pipe)
        # skip this and fall through to the usual "No .car/.cif/..." error,
        # since there's nobody there to answer a prompt.
        cars = select_structure_files_interactively(Path.cwd())
    cars += [Path(p) for p in args.car]
    cars = [p for p in cars if is_supported_structure_file(p)]
    if not cars:
        ap.error("No .car/.cif/POSCAR/CONTCAR specified. Use positional args or --all, or use --restart-from-outdir.")

    sort_order = None
    if args.no_sort:
        sort_order = []  # preserve appearance order (special meaning in sort_atoms)
    else:
        if args.element_order:
            sort_order = [x.strip() for x in args.element_order.split(",") if x.strip()]
        else:
            sort_order = None

    out_base = Path(args.outdir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    created_dirs: List[str] = []

    # preset never actually varies per-file (it's built entirely from global
    # args), so it's built once here rather than rebuilt every loop iteration -
    # this also lets the settings preview below show it before any files are
    # touched.
    preset = {
        "meshcutoff": args.meshcutoff,
        "etemp": args.etemp,
        "dm_tol": args.dm_tol,
        "max_scf": args.max_scf,
        "max_force": args.max_force,
        "cg_steps": args.cg_steps,
        "init_step": args.init_step,
        "final_step": args.final_step,
        "dt": args.dt,
        "init_temp": args.init_temp,
        "target_temp": args.target_temp,
        "tau": args.tau,
        "target_pressure": args.target_pressure,
        "nose_mass": args.nose_mass,
        "pr_mass": args.pr_mass,
        "anneal_option": args.anneal_option,
        "mulliken": 1,
        "write_wf": True,
        "solution": "diagon",
    }

    # Ask (once, before the loop) for extra fdf overrides - only if not
    # already given via --extra/config, and only if actually interactive.
    # Prints a CCpyVASPInputGen.py-style settings preview first.
    extra_options = maybe_prompt_for_extra(args.extra or OrderedDict(), mode, basis, preset, args, bool(args.continuation), bool(args.variable_cell))

    for car_path in cars:
        car_data = read_structure(car_path)
        if not car_data.atoms:
            raise SystemExit(f"ERROR: no atoms parsed from {car_path}")

        base = car_path.stem
        label = args.label or base
        name = args.name or label

        # if --no-sort, build appearance-ordered element list from original atoms
        sort_for_this = sort_order
        if args.no_sort:
            sort_for_this = []   #  an empty list is used only as a "do not sort" signal
#            sort_for_this = []
#            for a in car_data.atoms:
#                if a.el not in sort_for_this:
#                    sort_for_this.append(a.el)

        # Determine kgrid for this structure with priority:
        # 1) --kgrid-metal-1d (uses --dk)
        # 2) --kgrid-from-car (uses --kgrid-target)
        # 3) fallback to --kgrid
        kgrid_use = args.kgrid

        if getattr(args, "kgrid_metal_1d", False):
            if args.cell is not None:
                cell_for_k = args.cell
            elif car_data.cell_params is not None:
                cell_for_k = car_data.cell_params
            else:
                cell_for_k, _shift_tmp = make_orthorhombic_cell_from_atoms(car_data.atoms, vacuum=args.vacuum)

            kx, ky, _ = kgrid_use
            kgrid_use = kgrid_metal_1d(cell_for_k, dk=float(args.dk), kz_min=int(args.kz_min), kx=kx, ky=ky)

        elif getattr(args, "kgrid_from_car", False):
            if args.cell is not None:
                cell_for_k = args.cell
            elif car_data.cell_params is not None:
                cell_for_k = car_data.cell_params
            else:
                cell_for_k, _shift_tmp = make_orthorhombic_cell_from_atoms(car_data.atoms, vacuum=args.vacuum)

            a, b, c = cell_for_k[:3]
            import math
            def oddify(n): return n if n % 2 == 1 else n + 1
            kx = oddify(max(1, math.ceil(float(args.kgrid_target) / a)))
            ky = oddify(max(1, math.ceil(float(args.kgrid_target) / b)))
            kz = oddify(max(1, math.ceil(float(args.kgrid_target) / c)))
            kgrid_use = (kx, ky, kz)
        fdf_text = build_fdf(
            car=car_data,
            system_label=label,
            system_name=name,
            basis=basis,
            net_charge=0.0,
            kgrid=kgrid_use,
            cell_params=args.cell,
            vacuum=args.vacuum,
            sort_order=sort_for_this,
            mode=mode,
            preset=preset,
            continuation=bool(args.continuation),
            variable_cell=bool(args.variable_cell),
        )

        if args.patch_from_out:
            out_path = Path(args.patch_from_out)
            coord_block, cell2 = parse_siesta_out_relaxed(out_path)
            fdf_text = patch_fdf_from_out(fdf_text, coord_block, cell2)

        if extra_options:
            fdf_text = apply_extra_overrides(fdf_text, extra_options)

        if args.make_dirs:
            job_dir = out_base / label
            job_dir.mkdir(parents=True, exist_ok=True)
            fdf_name = f"{label}.fdf"
            (job_dir / fdf_name).write_text(fdf_text)

            # Copy pseudopotentials from the single resolved pot directory
            # (--pseudo-dir override, else -pot=dojo/psf/psfold), then bare
            # filename in the current directory as a last resort. Within
            # whichever directory matches, .psf is preferred over .psml and
            # we stop immediately at the first match per element - we never
            # look in a second directory for the other extension, since a
            # PSF-format .psf plus a Dojo-format .psml for the same element
            # both landing in the same job dir is something SIESTA can't
            # resolve, and the run won't proceed.
            species = sorted({a.el for a in car_data.atoms})
            search_dirs: List[str] = [resolve_pseudo_dir(args), "."]
            for el in species:
                for d in search_dirs:
                    matched = False
                    for ext in (".psf", ".psml"):
                        src = Path(d) / f"{el}{ext}"
                        if src.exists():
                            try:
                                shutil.copy2(src, job_dir / src.name)
                            except Exception:
                                pass
                            matched = True
                            break
                    if matched:
                        break

            created_dirs.append(label)

            if args.write_mpi:
                write_slurm_mpi_sh(
                    job_dir=job_dir,
                    job_name=label,
                    partition=args.partition,
                    ntasks=args.ntasks,
                    nodes=args.nodes,
                    time_limit=args.time_limit,
                    mem=args.mem,
                    siesta_cmd=args.siesta_cmd,
                    fdf_name=fdf_name,
                    extra_lines=[],
                )
                if args.submit:
                    os.system(f"cd '{job_dir.as_posix()}' && sbatch mpi.sh")
        else:
            out_fdf = out_base / f"{label}.fdf"
            out_fdf.write_text(fdf_text)

        # Archive the original input structure file, CCpyVASPInputGen.py-style:
        # move it (not copy) into ./structures/ (relative to cwd, not --outdir)
        # so raw structure files don't keep cluttering the working directory
        # once they've been consumed, but stay available for provenance/rerun.
        if car_path.exists():
            structures_dir = Path.cwd() / "structures"
            structures_dir.mkdir(exist_ok=True)
            dest = structures_dir / car_path.name
            try:
                shutil.move(str(car_path), str(dest))
            except Exception as exc:
                print(f"WARNING: could not move {car_path} into structures/: {exc}")

    if args.make_dirs and args.id is not None:
        gen = out_base / f"gen{args.id}.list"
        gen.write_text("\n".join(created_dirs) + "\n")

    return 0

if __name__ == "__main__":
    bootstrap_reexec_if_needed()
    raise SystemExit(main())


# -------------------------------------------------------------------------------------------------------
## --- Metal SCF stabilization (SIESTA 5.4.2) ---
#SCF.Mix                 charge
#DM.MixingWeight         0.03          # (the previous 0.10 may be too aggressive)
#SCF.RhoG.DIIS.Depth     6
#SCF.Kerker.q0sq         0.5 Ry
#SCF.RhoGMixingCutoff    9 Ry
#ElectronicTemperature   1000 K         # especially effective for metals/semimetals


####### Lattice Constrain ###########
##         2D setting
#####################################
# MD.ConstantVolume     false

# %block Geometry.Constraints
#  cell-vector 3
#  cell-angle 1
#  cell-angle 2
#  cell-angle 3
# %endblock Geometry.Constraints
