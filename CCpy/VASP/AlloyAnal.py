#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlloyAnal.py  (CCpy AlloyGen result analysis engine)

Reads back the folder sets produced by CCpyAlloyGen.py and answers the three
questions that come up once the VASP jobs are done:

1) Adsorption site of every adsorbate atom -- is it on top of one substrate
   atom, bridging two, or in a three/four-fold hollow (and, for a three-fold
   hollow of an fcc(111)-like surface, fcc or hcp).
2) Energy difference against the adsorbate-free twin:
       dE_surface = E(set/Sxxxxxx) - E(set_surface/Sxxxxxx)
3) Energy difference against every redox twin:
       dE_r1 = E(set/Sxxxxxx) - E(set_r1/Sxxxxxx),  dE_r2 = ...

The twins are matched by structure ID, which is safe because AlloyGen keeps
the atom order of the input file and writes the same ID in every twin folder
(see the "twin folders" section of the AlloyGen docs).

dE_surface is NOT a complete adsorption energy: the reference energy of the
adsorbate itself (E_ads) is not subtracted, so the number is only the
difference between the two folders. That is on purpose -- the reference state
is a choice the user has to make, and mixing an arbitrary one in silently
would produce numbers nobody can reproduce.

Energies are read exactly the way CCpyVASPAnal.py option 2 does, by calling
CCpy.VASP.VASPio.energy_from_outcar(): the 'free  energy   TOTEN' of OUTCAR and
nothing else. OSZICAR's E0 is not accepted as a substitute -- it is the sigma->0
extrapolated value and differs from TOTEN by the -TS term -- so a folder with no
readable OUTCAR is left blank rather than filled from a different reference.

Site assignment is done by two independent methods and both are reported:

  "dist"  -- nearest-neighbour counting. Distances from the adsorbate atom to
             every top-layer substrate atom are sorted; every atom within
             (1 + tol) of the shortest one is counted as coordinated.
             1 -> top, 2 -> bridge, 3 -> 3-fold hollow, 4 -> 4-fold hollow.
  "proj"  -- geometric projection. The adsorbate is projected onto the surface
             plane and located inside the Delaunay triangulation of the
             top-layer atoms; the barycentric coordinates of that point decide
             vertex (top) / edge (bridge) / interior (hollow).

They agree for clean sites and disagree exactly where the answer is genuinely
ambiguous (a distorted or half-way position), so the disagreement itself is
the useful output. The "agree" column flags it.

The single "site" column follows -main (projection by default, because its
barycentric test is scale-free and so does not drift when the surface elements
have different radii); the distance method stays next to it as the check that
catches what projection structurally cannot see -- it ignores height, so an
adsorbate that drifted away from the surface still gets a confident label,
while d_min and height show it at once.

Every atom also gets "sub_site": the same classification run against the
SECOND substrate layer. The adsorbate does not bond to that layer, so this is
a descriptor of what the site sits on rather than a site, and it stays a
secondary column. For an ideal fcc(111) slab it only repeats fcc/hcp
(hcp <=> sub_site 'top'). It earns its place where fcc/hcp says nothing: the
subsurface ELEMENTS under the site on a substituted / HEA surface, which is
what shifts the adsorption energy between otherwise identical sites.
"""

import ast
import csv
import os
import re
import sys

from collections import Counter

import numpy as np
import pandas as pd
from ase.io import read as ase_read

# Energy reading is shared with CCpyVASPAnal.py option 2 -- same function,
# same OUTCAR TOTEN source, so the two commands can never disagree. The
# import is deferred into the two functions that need it because VASPio pulls
# in pymatgen and matplotlib (~2 s), which a sites-only run never uses.

# The slab normal / adsorbate heuristics live in AlloyGen and are used through
# the module so there is a single definition of "which axis is the vacuum".
from CCpy.VASP import AlloyGen


# -----------------------------------------------------------------------------
# Folder conventions of CCpyAlloyGen
# -----------------------------------------------------------------------------

STRUCT_DIR_PATTERN = re.compile(r"^S\d+$")
REDOX_SUFFIX_PATTERN = re.compile(r"^_r(\d*)$")
SURFACE_SUFFIX = "_surface"

# Site assignment defaults
DEFAULT_DIST_TOL = 0.15     # nearest-neighbour distance window (fraction)
# 1.2 A, not 1.0: a relaxed surface buckles, and atoms of one physical layer
# then land in different groups, which is what actually breaks the projection
# method (its triangulation would cover only part of the top layer). 1.2 A is
# still well under the interlayer spacing of a close-packed metal (~2 A), so
# layer 1 and layer 2 do not merge. Raise it with -layer_tol for a rough slab.
DEFAULT_LAYER_TOL = 1.2     # A, thickness of one substrate layer
DEFAULT_HCP_TOL = 0.8       # A, in-plane match to a second-layer atom
# Which method fills the single 'site' column and the per-structure summary.
# Projection is the primary one: it is scale-free (barycentric weights), so a
# substituted surface whose elements have different radii does not shift the
# answer the way a distance window can.
DEFAULT_MAIN_METHOD = "proj"
BARY_TOP_MIN = 0.60         # barycentric weight above which it is a vertex
BARY_EDGE_MAX = 0.15        # barycentric weight below which the vertex is out

SITE_NAMES = {1: "top", 2: "bridge", 3: "hollow3", 4: "hollow4"}

# What the site file shows. One answer per atom (-main, projection by default)
# plus the flag that says whether the other method agreed -- the other method's
# own answer lives in the separate diff file, so this table stays readable.
# 'side' is added only when something actually sits on the bottom face.
SITE_COLUMNS = ["Structure", "folder", "atom", "local_no", "site", "ensemble",
                "neighbors", "sub_site", "sub_ensemble", "sub_neighbors",
                "d_min (A)", "height (A)", "agree"]
# The cross-check file: only the atoms the two methods disagree on, with both
# answers side by side and the numbers needed to judge which one to believe.
DIFF_COLUMNS = ["Structure", "folder", "atom", "local_no",
                "site_proj", "ensemble_proj", "neighbors_proj",
                "site_dist", "ensemble_dist", "neighbors_dist",
                "weights", "d_min (A)", "height (A)", "agree"]
# The roll-up: how often each (site geometry, element composition) occurs.
# A header must not start with '+', '=', '-' or '@': a spreadsheet reads such a
# cell as a formula and shows #NAME? instead of the column. That is why the
# error bar is "stderr (eV)" and not "+- (eV)".
ENSEMBLE_COLUMNS = ["folder", "element", "site", "ensemble", "atoms",
                    "structures", "dE vs group avg (eV)", "stderr (eV)",
                    "mean d_min (A)", "mean height (A)"]


def _is_dir(path):
    return os.path.isdir(path)


def has_contcar(directory):
    """True when this folder holds a non-empty CONTCAR, i.e. VASP has run."""
    path = os.path.join(directory, "CONTCAR")
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_metadata(directory):
    """
    Parse the 'key = value' lines of metadata.txt into a dict.

    Values are written by AlloyGen with plain f-strings, so lists and dicts
    come back as their repr; ast.literal_eval() restores them and anything it
    cannot parse is kept as the raw string.
    """
    path = os.path.join(directory, "metadata.txt")
    data = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                try:
                    data[key] = ast.literal_eval(value)
                except Exception:
                    data[key] = value
    except Exception:
        return data
    return data


def read_redox_map(set_dir):
    """
    Read redox_sets.csv (folder -> removed atoms) written by AlloyGen.
    Returns {folder_name: "atom numbers (elements)"} and {} when absent.
    """
    path = os.path.join(set_dir, "redox_sets.csv")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                folder = (row.get("folder") or "").strip()
                numbers = (row.get("removed_atom_numbers") or "").strip()
                elements = (row.get("removed_elements") or "").strip()
                if folder:
                    mapping[folder] = "%s (%s)" % (numbers, elements)
    except Exception:
        return mapping
    return mapping


def structure_dirs(set_dir):
    """Sorted S000001-style structure folders directly under set_dir."""
    try:
        names = os.listdir(set_dir)
    except OSError:
        return []
    return sorted(n for n in names
                  if STRUCT_DIR_PATTERN.match(n) and _is_dir(os.path.join(set_dir, n)))


def has_vasp_content(directory):
    """True when the directory itself looks like a VASP job / input folder."""
    try:
        files = os.listdir(directory)
    except OSError:
        return False
    for name in ("CONTCAR", "POSCAR", "OUTCAR", "OUTCAR.gz", "OSZICAR", "vasprun.xml"):
        if name in files:
            return True
    return False


def twin_suffix_of(name, base):
    """Return the twin suffix ('_surface', '_r', '_r1', ...) or None."""
    if not name.startswith(base) or name == base:
        return None
    suffix = name[len(base):]
    if suffix == SURFACE_SUFFIX or REDOX_SUFFIX_PATTERN.match(suffix):
        return suffix
    return None


def _redox_sort_key(label):
    digits = label[1:]
    return (0 if digits == "" else 1, int(digits) if digits.isdigit() else 0, label)


def find_twins(set_dir):
    """
    Find the sibling twin folders of a set: '<set>_surface' and '<set>_r*'.

    Returns (surface_dir_or_None, [(label, dir), ...]) with the redox twins
    ordered r, r1, r2, ... r10 (numerically, not as strings).
    """
    absolute = os.path.abspath(set_dir)
    parent = os.path.dirname(absolute)
    base = os.path.basename(absolute)
    surface, redox = None, []
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return None, []
    for name in names:
        path = os.path.join(parent, name)
        if not _is_dir(path):
            continue
        suffix = twin_suffix_of(name, base)
        if suffix is None:
            continue
        if suffix == SURFACE_SUFFIX:
            surface = path
        else:
            redox.append((suffix.lstrip("_"), path))
    redox.sort(key=lambda item: _redox_sort_key(item[0]))
    return surface, redox


def is_twin_name(name, sibling_names):
    """
    True when `name` is the twin of some other folder in the same directory,
    so that twins are not offered as analysis targets of their own.
    """
    for other in sibling_names:
        if other == name:
            continue
        if twin_suffix_of(name, other):
            return True
    return False


def find_alloy_sets(directory="./"):
    """
    Candidate AlloyGen output sets under `directory`, twins excluded.

    A set is a folder holding S000001-style structure folders. A folder that is
    itself a single VASP job is also accepted (the -fmt=vasp / single-structure
    case), and `directory` itself is offered when it holds structure folders.
    """
    try:
        names = sorted(n for n in os.listdir(directory) if _is_dir(os.path.join(directory, n)))
    except OSError:
        return []
    sets = []
    if structure_dirs(directory):
        sets.append(os.path.normpath(directory))
    for name in names:
        path = os.path.join(directory, name)
        if is_twin_name(name, names):
            continue
        if structure_dirs(path) or has_vasp_content(path):
            sets.append(path)
    return sets


def describe_set(set_dir):
    """One-line summary of a set for the selection list."""
    ids = structure_dirs(set_dir)
    surface, redox = find_twins(set_dir)
    parts = []
    parts.append("%d structure(s)" % len(ids) if ids else "single folder")
    twins = []
    if surface:
        twins.append("_surface")
    twins.extend("_" + label for label, _ in redox)
    parts.append("twins: " + ", ".join(twins) if twins else "no twin")
    return "  (" + " / ".join(parts) + ")"


def select_alloy_sets(directory="./", ask=True, chosen=None):
    """
    Ask which sets to analyze, laid out like CCpy's own pickers
    ('1 : folder', '0 : All folders', 'Choose folder : ') and accepting the
    same '1-3,5,7' syntax as selectVASPOutputs().
    """
    if chosen:
        return [c for c in chosen if _is_dir(c)]
    all_sets = find_alloy_sets(directory)
    if not all_sets:
        print("No AlloyGen output folder detected here.")
        print("Run this in the directory that holds the output folder(s), "
              "or give one with -i=[DIR].")
        return []
    if ask:
        for i, path in enumerate(all_sets):
            print(str(i + 1) + " : " + path + describe_set(path))
        print("0 : All folders")
        get_num = input("Choose folder : ")
    else:
        get_num = "0"

    try:
        if get_num.strip() == "0":
            return all_sets
        selected = []
        for token in get_num.split(","):
            token = token.strip()
            if "-" in token:
                start, end = token.split("-")
                for j in range(int(start), int(end) + 1):
                    selected.append(all_sets[j - 1])
            else:
                selected.append(all_sets[int(token) - 1])
    except Exception:
        print("Unvalid input type.")
        print("ex : 1-3,5-10,11,12,13")
        return []
    return selected


# -----------------------------------------------------------------------------
# Reading one job folder
# -----------------------------------------------------------------------------

def _vaspio():
    """The VASPio helpers, imported on first use (see the note at the top)."""
    from CCpy.VASP.VASPio import energy_from_outcar, natoms_from_poscar
    return energy_from_outcar, natoms_from_poscar


def count_atoms(directory):
    """
    Number of atoms of one folder. POSCAR is read with VASPio's own counter
    (the same one CCpyVASPAnal uses); CONTCAR is only looked at when POSCAR is
    missing, so a folder holding just a relaxed result still gets a count.
    """
    _outcar, natoms_from_poscar = _vaspio()
    natoms = natoms_from_poscar(directory)
    if natoms:
        return natoms
    path = os.path.join(directory, "CONTCAR")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as handle:
            lines = handle.readlines()
    except Exception:
        return None
    for line in lines[5:]:
        tokens = line.split()
        if tokens and all(token.isdigit() for token in tokens):
            try:
                return sum(int(token) for token in tokens)
            except Exception:
                continue
    return None


# -----------------------------------------------------------------------------
# Did the SCF loop converge?
# -----------------------------------------------------------------------------
# The check itself is VASPio.scf_converged_from_oszicar(), so this command and
# CCpyVASPAnal's 'Converged' column can never disagree about it. Why it is
# needed at all -- what the custodian pass of vasp_status() cannot see, and why
# a single-point folder falls through that pass entirely -- is written out
# there, next to the check.


def check_electronic_converged(directory, _cache={}):
    """
    Did the last SCF loop of this folder converge, or did it hit NELM?

    Returns "True" / "False" / "Unknown" ("Unknown" when there is no OSZICAR,
    when it holds no closed ionic step, or when nothing could be parsed).
    Cached per folder: a twin is asked about once per structure it takes part
    in. VASPio is imported on first use, like the energy helpers above.
    """
    directory = os.path.abspath(directory)
    if directory in _cache:
        return _cache[directory]
    from CCpy.VASP.VASPio import scf_converged_from_oszicar
    verdict = scf_converged_from_oszicar(directory)
    _cache[directory] = verdict
    return verdict


def check_finished(directory, _cache={}):
    """
    Has the run in this folder ended, and is what is lying in it the CURRENT
    run's output? "True" / "False" / "Unknown".

    The marker is vasp.done. CCpy's queue script touches it right after each
    folder's VASP call and deletes it again when that folder is resubmitted
    (CCpyJobControl), so its absence means one of two things, and both make the
    folder's numbers the wrong answer to the question being asked: the job has
    not run yet, or what is there is the leftovers of an earlier attempt.

    Nothing else in this file notices either case. CONTCAR decides whether a
    structure gets a row at all and survives a resubmission untouched; OUTCAR
    and OSZICAR survive with it, so a folder whose earlier attempt ended at
    -48461 eV reads exactly like a finished one. And vasp_status() only fills
    in its convergence verdict inside the branch that requires vasp.done, so
    without it the verdict is blank -- "Unknown", never "False".

    "Unknown" when the folder holds no VASP output at all, so a structure that
    has simply not been set up yet is not called unfinished. Cached per folder.
    """
    directory = os.path.abspath(directory)
    if directory in _cache:
        return _cache[directory]
    try:
        names = os.listdir(directory)
    except Exception:
        names = []
    if "vasp.done" in names:
        verdict = "True"
    elif any(name in names for name in ("OUTCAR", "OUTCAR.gz", "OSZICAR",
                                        "OSZICAR.gz", "vasprun.xml",
                                        "vasprun.xml.gz")):
        verdict = "False"
    else:
        verdict = "Unknown"
    _cache[directory] = verdict
    return verdict


def check_converged(directory, _cache={}):
    """
    Did VASP finish this folder cleanly? Two independent checks, and either one
    alone can fail the folder:

    1) VASPOutput.vasp_status() -- custodian's VaspErrorHandler over vasp.out
       plus the max-ionic check built from the INCAR's NSW. This is the verdict
       CCpyVASPAnal option 0 and the 'Converged' column of option 2 show.
    2) check_electronic_converged() -- did the last SCF loop hit NELM. Check 1
       cannot see this at all, and it is blindest exactly where it matters
       here: a single-point folder, whose max-ionic pattern can never fire.

    One difference, on purpose: vasp_status() reports "False" when there is no
    vasp.out at all, which for this command would mark every folder of a run
    that simply does not keep vasp.out as failed. That is not a verdict, so it
    is reported as "Unknown" here and left out of the counts.

    A second, narrower case falls the same way: vasp_status() only fills in
    its convergence verdict inside the branch that also requires a "vasp.done"
    marker file -- without one it leaves the field as its unset default (a
    blank string), not "False". A job killed mid-run (walltime, a node dying)
    can easily have OUTCAR and vasp.out without ever writing vasp.done, and
    that blank does not equal "False" or "Unknown", so it would silently slip
    past both checks below instead of being counted as one or the other.
    Anything that is not literally "True" or "False" is normalized to
    "Unknown" here for that reason.

    Returns "True" / "False" / "Unknown" ("Unknown" when custodian is missing
    and OSZICAR does not settle it either). Results are cached per folder: a
    twin is asked about once per structure it takes part in.
    """
    directory = os.path.abspath(directory)
    if directory in _cache:
        return _cache[directory]
    verdict = "Unknown"
    try:
        names = os.listdir(directory)
    except Exception:
        names = []
    has_log = any(name in names for name in ("vasp.out", "vasp.out.gz"))
    has_outcar = any(name in names for name in ("OUTCAR", "OUTCAR.gz"))
    if has_log and has_outcar:
        from CCpy.VASP.VASPio import VASPOutput
        pwd = os.getcwd()
        try:
            os.chdir(directory)
            raw = str(VASPOutput().vasp_status()[2])
        except Exception:
            raw = "Unknown"
        finally:
            os.chdir(pwd)
        verdict = raw if raw in ("True", "False") else "Unknown"
    # An SCF that ran out of steps fails the folder whatever check 1 said --
    # including when check 1 could not be made at all. The reverse does not
    # hold: a converged SCF says nothing about how the run ended, so it never
    # turns an "Unknown" (or a "False") into a "True".
    if check_electronic_converged(directory) == "False":
        verdict = "False"
    _cache[directory] = verdict
    return verdict


def read_energy(directory):
    """
    Final energy of one VASP folder, the CCpyVASPAnal.py way: the OUTCAR TOTEN
    and nothing else. OSZICAR's E0 is a different reference (sigma->0, differing
    by the -TS term) and is not used as a fallback, so a folder with no readable
    OUTCAR stays blank. Returns (energy, source) with source "" when nothing
    could be read.
    """
    energy_from_outcar, _natoms = _vaspio()
    energy = energy_from_outcar(directory)
    if energy is not None:
        return energy, "OUTCAR"
    return None, ""


def read_structure(directory, prefer_poscar=False):
    """
    Read the structure of one folder as an ase Atoms.
    CONTCAR (the relaxed result) is used by default, POSCAR when asked for or
    when CONTCAR is missing / empty. Returns (atoms, filename_used).
    """
    order = ["POSCAR", "CONTCAR"] if prefer_poscar else ["CONTCAR", "POSCAR"]
    for name in order:
        path = os.path.join(directory, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return ase_read(path, format="vasp"), name
            except Exception:
                continue
    return None, None


# -----------------------------------------------------------------------------
# Which atoms are the adsorbate
# -----------------------------------------------------------------------------

def resolve_adsorbate_elements(atoms, surface_atoms=None, metadata=None,
                               ads_override=None, pool_override=None):
    """
    Decide which elements are the adsorbate, and say where that came from.

    Priority, most trustworthy first:
      1. -ads= given by the user.
      2. The _surface twin: AlloyGen strips the adsorbate elements entirely, so
         whatever the main structure has and the surface twin has not IS the
         adsorbate. This needs no heuristic at all.
      3. metadata.txt of the surface twin ('adsorbate_elements_removed').
      4. AlloyGen's geometric detection, using the replacement pool from
         metadata.txt (or the most abundant element as a last resort).

    Returns dict(elements, pool, substrate, source, note). `pool` is what the
    geometric detection used as the substrate definition; `substrate` is which
    elements count as the surface when a site is assigned. They differ on
    purpose: metadata's replace_elements is the pool of the INPUT structure
    (e.g. ['Pt']), while the generated structure's surface also contains
    everything that was substituted into it (Fe, Co, Ni, Cu ...). So the
    substrate defaults to "everything that is not the adsorbate", and only an
    explicit -pool= narrows it.
    """
    present = sorted(set(atoms.get_chemical_symbols()))
    if ads_override:
        elements = [e for e in ads_override if e in present]
        missing = [e for e in ads_override if e not in present]
        note = "" if not missing else "not present in this structure: " + ",".join(missing)
        substrate = pool_override or [e for e in present if e not in elements]
        return {"elements": elements, "pool": substrate, "substrate": substrate,
                "source": "-ads=", "note": note}

    if surface_atoms is not None:
        bare = set(surface_atoms.get_chemical_symbols())
        elements = sorted(set(present) - bare)
        if elements:
            substrate = pool_override or [e for e in present if e not in elements]
            return {"elements": elements, "pool": substrate, "substrate": substrate,
                    "source": "_surface twin composition", "note": ""}

    if metadata:
        recorded = metadata.get("adsorbate_elements_removed")
        if isinstance(recorded, (list, tuple)) and recorded:
            elements = [str(e) for e in recorded if str(e) in present]
            if elements:
                substrate = pool_override or [e for e in present if e not in elements]
                return {"elements": elements, "pool": substrate, "substrate": substrate,
                        "source": "metadata.txt", "note": ""}

    pool = pool_override
    note = ""
    if not pool and metadata:
        recorded = metadata.get("replace_elements")
        if isinstance(recorded, (list, tuple)) and recorded:
            pool = [str(e) for e in recorded if str(e) in present]
    if not pool:
        counts = {}
        for symbol in atoms.get_chemical_symbols():
            counts[symbol] = counts.get(symbol, 0) + 1
        pool = [max(counts, key=lambda k: counts[k])]
        note = ("substrate assumed to be the most abundant element (%s); "
                "give -pool= or -ads= to fix it" % pool[0])
    detected = AlloyGen.detect_adsorbate_candidates(atoms, pool)
    if not detected["is_slab"]:
        return {"elements": [], "pool": pool, "substrate": pool_override or present,
                "source": "geometry",
                "note": "no vacuum layer found, adsorbate cannot be judged "
                        "geometrically; give -ads="}
    elements = list(detected["suggested"])
    substrate = pool_override or [e for e in present if e not in elements]
    return {"elements": elements, "pool": pool, "substrate": substrate,
            "source": "geometry (AlloyGen detection)", "note": note}


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def surface_frame(atoms, axis):
    """
    Return (normal_unit_vector, in_plane_basis_u1, in_plane_basis_u2).

    The normal is the true surface normal (cross product of the two in-plane
    cell vectors), not the `axis` cell vector, so a tilted cell still works.
    """
    cell = np.array(atoms.cell)
    in_plane = [i for i in range(3) if i != axis]
    a1, a2 = cell[in_plane[0]], cell[in_plane[1]]
    normal = np.cross(a1, a2)
    normal = normal / np.linalg.norm(normal)
    u1 = a1 / np.linalg.norm(a1)
    u2 = a2 - np.dot(a2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    return normal, u1, u2, (a1, a2)


def normal_coordinate(atoms, axis, normal):
    """Position of every atom along the surface normal, in Angstrom."""
    return np.dot(atoms.get_positions(), normal)


def layer_groups(coord, indices, tol):
    """
    Split substrate atoms into layers along the (signed) normal coordinate,
    highest first. Compared against the top of the current group, not the
    previous atom, so a gradual slab is not merged into one thick layer.
    """
    order = sorted(indices, key=lambda i: -coord[i])
    groups = [[order[0]]]
    for index in order[1:]:
        if coord[groups[-1][0]] - coord[index] <= tol:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def in_plane(vectors, normal):
    """Remove the normal component of one vector or an array of vectors."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    return vectors - np.outer(np.dot(vectors, normal), normal)


def ensemble_label(atoms, indices):
    """
    The site's element composition, e.g. 'Fe1Ni2' -- what the adsorbate is
    actually sitting on.

    On a substituted / HEA surface this, not 'hollow3', is the quantity that
    changes the adsorption energy: two hollows of the same geometry made of
    different elements are different sites. Elements are sorted alphabetically
    so the label is canonical and rows can be grouped and counted by it.
    """
    if not indices:
        return ""
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols[i] for i in indices)
    return "".join("%s%d" % (element, counts[element]) for element in sorted(counts))


def label_atom(atoms, index):
    """'Pt#12' -- element plus 1-based number in this folder's own POSCAR."""
    return "%s#%d" % (atoms.get_chemical_symbols()[index], index + 1)


def hollow_flavour(atoms, adsorbate_index, center_offset, second_layer, normal, tol):
    """
    fcc or hcp for a three-fold hollow: hcp when a second-layer atom sits
    directly under the hollow centre, fcc when nothing is there.
    """
    if not second_layer:
        return "", None
    vectors = atoms.get_distances(adsorbate_index, second_layer, mic=True, vector=True)
    offsets = in_plane(vectors, normal) - center_offset
    distances = np.linalg.norm(offsets, axis=1)
    k = int(np.argmin(distances))
    if distances[k] <= tol:
        return "hcp", second_layer[k]
    return "fcc", None


# -----------------------------------------------------------------------------
# Method 1: nearest-neighbour distance counting
# -----------------------------------------------------------------------------

def site_by_distance(atoms, adsorbate_index, top_layer, second_layer, normal,
                     dist_tol=DEFAULT_DIST_TOL, hcp_tol=DEFAULT_HCP_TOL):
    """
    Count the substrate atoms that share the shortest distance to this
    adsorbate atom (within a (1 + dist_tol) window) and name the site by how
    many there are.
    """
    distances = atoms.get_distances(adsorbate_index, top_layer, mic=True)
    vectors = atoms.get_distances(adsorbate_index, top_layer, mic=True, vector=True)
    shortest = float(np.min(distances))
    window = shortest * (1.0 + dist_tol)
    members = [k for k in range(len(top_layer)) if distances[k] <= window]
    count = len(members)
    site = SITE_NAMES.get(count, "hollow%d" % count)
    flavour, under = "", None
    if count == 3:
        center = in_plane(vectors[members], normal).mean(axis=0)
        flavour, under = hollow_flavour(atoms, adsorbate_index, center,
                                        second_layer, normal, hcp_tol)
    neighbours = [top_layer[k] for k in members]
    return {
        "site": site,
        "flavour": flavour,
        "neighbours": neighbours,
        "neighbour_labels": " ".join(label_atom(atoms, i) for i in neighbours),
        "d_min": shortest,
        "under_atom": under,
    }


# -----------------------------------------------------------------------------
# Method 2: projection onto the surface plane
# -----------------------------------------------------------------------------

def projection_frame(atoms, layer, normal, basis):
    """
    The surface plane's triangulation, built once per (structure, layer).

    Every atom of the layer is projected onto the plane in ABSOLUTE in-plane
    coordinates and replicated over the neighbouring periodic images, then
    triangulated. The triangulation does not depend on which adsorbate is being
    asked about -- only the query point does -- so building it per atom (as
    this did at first) repeats the most expensive step of the whole analysis
    once per adsorbate atom for no gain. Translating a point set does not
    change its Delaunay triangulation, which is why the switch from
    adsorbate-relative to absolute coordinates is exact and not an
    approximation.

    Returns None when there is nothing to triangulate.
    """
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return {"error": "scipy is required for the projection method"}
    if len(layer) < 3:
        return {"error": "too few surface atoms"}
    u1, u2, (a1, a2) = basis
    flat = in_plane(atoms.get_positions()[list(layer)], normal)
    base = np.column_stack([flat.dot(u1), flat.dot(u2)])
    shifts = []
    for k1 in (-1, 0, 1):
        for k2 in (-1, 0, 1):
            translation = in_plane(k1 * np.asarray(a1) + k2 * np.asarray(a2), normal)[0]
            shifts.append([float(translation.dot(u1)), float(translation.dot(u2))])
    points = np.concatenate([base + shift for shift in shifts])
    sources = list(layer) * len(shifts)
    # Spacing of the layer itself, used below to tell a real bridge from a
    # triangulation diagonal. Computed on the unreplicated layer: the images
    # add no shorter distance and squaring the point count is what made this
    # expensive.
    if len(base) > 1:
        spread = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=-1)
        np.fill_diagonal(spread, np.inf)
        nn_spacing = float(spread.min())
    else:
        nn_spacing = None
    try:
        triangulation = Delaunay(points)
    except Exception as error:
        return {"error": "triangulation failed (%s)" % error}
    return {"points": points, "sources": sources, "tri": triangulation,
            "nn_spacing": nn_spacing, "error": None}


def in_plane_point(atoms, index, normal, basis):
    """One atom's absolute position on the surface plane, as (u1, u2)."""
    u1, u2, _cell = basis
    flat = in_plane(atoms.get_positions()[index], normal)[0]
    return np.array([float(flat.dot(u1)), float(flat.dot(u2))])


def site_by_projection(atoms, adsorbate_index, frame, second_layer, normal,
                       basis, hcp_tol=DEFAULT_HCP_TOL):
    """
    Locate the adsorbate's projected position inside the triangulation of the
    surface plane. Barycentric coordinates then decide vertex (top) /
    edge (bridge) / interior (3-fold hollow).

    Returns a dict with site="unresolved" when the projected point falls
    outside every triangle, which happens on a badly broken surface -- that is
    reported rather than guessed.
    """
    if frame is None or frame.get("error"):
        note = frame.get("error") if frame else "no surface plane"
        site = "unavailable" if "scipy" in note else "unresolved"
        return {"site": site, "flavour": "", "neighbours": [],
                "neighbour_labels": "", "note": note}
    points, sources = frame["points"], frame["sources"]
    triangulation, nn_spacing = frame["tri"], frame["nn_spacing"]
    query = in_plane_point(atoms, adsorbate_index, normal, basis)
    simplex = int(triangulation.find_simplex(query.reshape(1, 2))[0])
    if simplex < 0:
        return {"site": "unresolved", "flavour": "", "neighbours": [],
                "neighbour_labels": "",
                "note": "projected position is outside the surface triangulation"}

    vertices = triangulation.simplices[simplex]
    transform = triangulation.transform[simplex]
    bary2 = transform[:2].dot(query - transform[2])
    weights = np.array([bary2[0], bary2[1], 1.0 - bary2.sum()])
    order = np.argsort(-weights)
    ranked = [vertices[i] for i in order]
    indices = [sources[v] for v in ranked]
    sorted_weights = weights[order]

    if sorted_weights[0] >= BARY_TOP_MIN:
        site, neighbours = "top", indices[:1]
    elif sorted_weights[2] <= BARY_EDGE_MAX:
        site, neighbours = "bridge", indices[:2]
        # A four-fold hollow of a (100)-like surface is not a bridge: the
        # triangulation has to cut every square in half, and the hollow centre
        # lands exactly on that diagonal. A diagonal is much longer than the
        # spacing of the layer, which is what separates the two cases.
        edge = float(np.linalg.norm(points[ranked[0]] - points[ranked[1]]))
        if nn_spacing and edge > 1.25 * nn_spacing:
            radius = edge / 2.0 * 1.2
            distances2d = np.linalg.norm(points - query, axis=1)
            corners = []
            for k in np.argsort(distances2d):
                if distances2d[k] > radius:
                    break
                if sources[k] not in corners:
                    corners.append(sources[k])
            if len(corners) >= 4:
                site, neighbours = "hollow%d" % len(corners), corners
    else:
        site, neighbours = "hollow3", indices[:3]

    flavour = ""
    if site == "hollow3":
        centre2d = np.array([points[v] for v in vertices]).mean(axis=0)
        offset2d = centre2d - query
        centre = offset2d[0] * basis[0] + offset2d[1] * basis[1]
        flavour, _ = hollow_flavour(atoms, adsorbate_index, centre,
                                   second_layer, normal, hcp_tol)
    return {
        "site": site,
        "flavour": flavour,
        "neighbours": neighbours,
        "neighbour_labels": " ".join(label_atom(atoms, i) for i in neighbours),
        "weights": [round(float(w), 3) for w in sorted_weights],
        "note": "",
    }


def classify_atom(atoms, index, layer, next_layer, normal, basis,
                  dist_tol=DEFAULT_DIST_TOL, hcp_tol=DEFAULT_HCP_TOL,
                  frame=None):
    """
    Run both methods against ONE substrate layer and return (dist, proj).

    `next_layer` is used only to tell fcc from hcp, i.e. to ask whether
    anything sits directly under a three-fold hollow. `frame` is that layer's
    triangulation; pass it in to reuse one across the atoms of a structure.
    """
    by_dist = site_by_distance(atoms, index, layer, next_layer, normal,
                               dist_tol=dist_tol, hcp_tol=hcp_tol)
    if frame is None:
        frame = projection_frame(atoms, layer, normal, basis)
    by_proj = site_by_projection(atoms, index, frame, next_layer, normal,
                                 basis, hcp_tol=hcp_tol)
    return by_dist, by_proj


def site_label(result):
    """'hollow3-fcc' from one method's result dict."""
    site = result.get("site", "")
    flavour = result.get("flavour", "")
    return site + ("-" + flavour if flavour else "")


UNRESOLVED = ("unresolved", "unavailable")


def _read_poscar_only(directory):
    """POSCAR of this folder, or None. Never falls back to CONTCAR."""
    path = os.path.join(directory, "POSCAR")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return None
    try:
        return ase_read(path, format="vasp")
    except Exception:
        return None


def map_to_main_labels(twin_dir, main_dir, twin_atoms, main_poscar=None):
    """
    Label every atom of a redox twin with the number it carries in the MAIN
    folder, so a site can be followed across the redox step.

    The two folders' POSCARs come from the same generated structure -- the twin
    is that structure minus the removed atoms, written by the same writer, with
    nothing relaxed yet -- so the unrelaxed positions match exactly and the map
    can be read off instead of guessed.

    POSCAR is REQUIRED here, and CONTCAR is never accepted in its place. During
    relaxation the substrate atoms themselves shift, so matching relaxed
    coordinates would be a nearest-neighbour guess dressed up as an identity,
    and it would succeed for the atoms that happened to move least -- a partial
    map that looks authoritative. Without both POSCARs the answer is "unknown",
    which is a blank column and a warning, not a number.

    Returns (labels, note): a list as long as twin_atoms ("" where that atom
    could not be matched) and a reason string when no map could be built.
    """
    twin_poscar = _read_poscar_only(twin_dir)
    main_poscar = main_poscar if main_poscar is not None else _read_poscar_only(main_dir)
    if twin_poscar is None or main_poscar is None:
        missing = twin_dir if twin_poscar is None else main_dir
        return [], ("no POSCAR in %s, so the atoms cannot be cross-referenced "
                    "with the main folder (CONTCAR is not used for this: its "
                    "coordinates have relaxed)" % missing)
    if len(twin_poscar) != len(twin_atoms):
        return [], ("POSCAR and the analysed structure of %s hold a different "
                    "number of atoms, so the cross-reference was not attempted"
                    % twin_dir)
    main_symbols = main_poscar.get_chemical_symbols()
    main_frac = main_poscar.get_scaled_positions(wrap=True)
    twin_symbols = twin_poscar.get_chemical_symbols()
    twin_frac = twin_poscar.get_scaled_positions(wrap=True)
    labels = []
    for i in range(len(twin_poscar)):
        delta = main_frac - twin_frac[i]
        delta -= np.round(delta)                      # nearest periodic image
        distance = np.linalg.norm(delta, axis=1)
        distance[[j for j in range(len(main_poscar))
                  if main_symbols[j] != twin_symbols[i]]] = np.inf
        j = int(np.argmin(distance))
        labels.append("%s#%d" % (main_symbols[j], j + 1) if distance[j] < 1e-3 else "")
    # Two twin atoms landing on one main atom would mean the map is not a map.
    # It cannot happen with unrelaxed coordinates, which is exactly why it is
    # worth checking: if it ever does, the assumption behind the whole
    # cross-reference is wrong and every number here is suspect.
    filled = [label for label in labels if label]
    if len(set(filled)) != len(filled):
        return [], ("two atoms of %s map onto the same atom of the main folder; "
                    "the cross-reference was dropped" % twin_dir)
    return labels, ""


# -----------------------------------------------------------------------------
# Site analysis of one structure
# -----------------------------------------------------------------------------

def analyze_sites(atoms, adsorbate_elements, substrate_elements=None,
                  dist_tol=DEFAULT_DIST_TOL, layer_tol=DEFAULT_LAYER_TOL,
                  hcp_tol=DEFAULT_HCP_TOL, main=DEFAULT_MAIN_METHOD):
    """
    Adsorption site of every adsorbate atom of one structure, by both methods.

    Besides the site on the top layer, the same classification is run against
    the SECOND substrate layer ("sub_site"). The adsorbate does not bond to
    that layer, so it is a descriptor of what the site sits on, not a site of
    its own -- which is why it stays a secondary column. On an ideal fcc(111)
    slab it repeats the fcc/hcp answer (hcp <=> sub_site 'top'); what it adds
    is the case fcc/hcp cannot express: which SUBSURFACE elements are under
    the site on a substituted / HEA surface, and a second layer that relaxed
    out of ideal stacking.

    Returns (rows, info). Every row is one adsorbate atom; info carries the
    slab normal, the vacuum gap and any warning worth printing once.
    """
    symbols = atoms.get_chemical_symbols()
    adsorbate_set = set(adsorbate_elements)
    ads_idx = [i for i, s in enumerate(symbols) if s in adsorbate_set]
    if substrate_elements:
        substrate_set = set(substrate_elements) - adsorbate_set
        sub_idx = [i for i, s in enumerate(symbols) if s in substrate_set]
    else:
        sub_idx = [i for i, s in enumerate(symbols) if s not in adsorbate_set]
    info = {"axis": None, "vacuum_gap": None, "warning": ""}
    if not ads_idx:
        info["warning"] = "no adsorbate atom in this structure"
        return [], info
    if len(sub_idx) < 3:
        info["warning"] = "fewer than 3 substrate atoms, site cannot be assigned"
        return [], info

    axis, gap, _origin = AlloyGen._largest_vacuum_axis(atoms)
    normal, u1, u2, cell_vectors = surface_frame(atoms, axis)
    coord = normal_coordinate(atoms, axis, normal)
    basis = (u1, u2, cell_vectors)
    info["axis"] = "xyz"[axis]
    info["vacuum_gap"] = float(gap)
    warnings_seen = []
    if gap < AlloyGen.VACUUM_MIN_GAP:
        warnings_seen.append("largest vacuum gap is only %.2f A, so 'surface' is "
                             "not well defined here" % gap)

    substrate_mid = float(np.mean(coord[sub_idx]))
    rows = []
    thin_layer_reported = False
    # One triangulation per layer, shared by every adsorbate atom that asks
    # about that layer -- the atoms of a structure nearly always share one.
    frames = {}

    def frame_for(layer):
        key = tuple(layer)
        if key not in frames:
            frames[key] = projection_frame(atoms, layer, normal, basis)
        return frames[key]
    for index in ads_idx:
        # Which face of the slab this atom sits on decides which layer is "top".
        side = 1.0 if coord[index] >= substrate_mid else -1.0
        signed = coord * side
        groups = layer_groups(signed, sub_idx, layer_tol)
        top_layer = groups[0]
        second_layer = groups[1] if len(groups) > 1 else []
        third_layer = groups[2] if len(groups) > 2 else []

        # A top layer much thinner than the layers below it means the split is
        # cutting through one buckled layer -- the one failure mode of the
        # projection method, and -layer_tol is the knob for it.
        widest = max(len(group) for group in groups)
        if not thin_layer_reported and len(groups) > 1 and len(top_layer) < 0.6 * widest:
            warnings_seen.append("top layer holds %d atom(s) against %d in the "
                                 "widest layer -- the surface may be buckled by "
                                 "more than -layer_tol (%.2f A)"
                                 % (len(top_layer), widest, layer_tol))
            thin_layer_reported = True

        by_dist, by_proj = classify_atom(atoms, index, top_layer, second_layer,
                                         normal, basis, dist_tol=dist_tol,
                                         hcp_tol=hcp_tol,
                                         frame=frame_for(top_layer))
        agree = "same" if by_dist["site"] == by_proj.get("site") else "DIFF"
        if by_proj.get("site") in UNRESOLVED:
            agree = by_proj["site"]

        # The single 'site' column follows -main, but never hides an answer:
        # when the primary method cannot resolve the position the other one
        # fills in, and the agree column is what says so.
        primary, secondary = ((by_proj, by_dist) if main == "proj"
                              else (by_dist, by_proj))
        if primary.get("site") in UNRESOLVED:
            primary = secondary
        main_neighbours = primary.get("neighbour_labels", "")

        sub_site, sub_neighbours, sub_ensemble, d_min_sub = "", "", "", None
        if second_layer:
            sub_dist, sub_proj = classify_atom(atoms, index, second_layer,
                                               third_layer, normal, basis,
                                               dist_tol=dist_tol, hcp_tol=hcp_tol,
                                               frame=frame_for(second_layer))
            sub_primary = sub_proj if main == "proj" else sub_dist
            if sub_primary.get("site") in UNRESOLVED:
                sub_primary = sub_dist
            sub_site = site_label(sub_primary)
            sub_neighbours = sub_primary.get("neighbour_labels", "")
            sub_ensemble = ensemble_label(atoms, sub_primary.get("neighbours", []))
            d_min_sub = round(float(sub_dist["d_min"]), 3)

        height = float(signed[index] - np.mean(signed[top_layer]))
        rows.append({
            "atom": label_atom(atoms, index),
            "element": symbols[index],
            "atom_no": index + 1,
            "side": "top" if side > 0 else "bottom",
            "site": site_label(primary),
            "ensemble": ensemble_label(atoms, primary.get("neighbours", [])),
            "neighbors": main_neighbours,
            "agree": agree,
            "sub_site": sub_site,
            "sub_ensemble": sub_ensemble,
            "sub_neighbors": sub_neighbours,
            "site_dist": site_label(by_dist),
            "ensemble_dist": ensemble_label(atoms, by_dist.get("neighbours", [])),
            "neighbors_dist": by_dist["neighbour_labels"],
            "site_proj": site_label(by_proj),
            "ensemble_proj": ensemble_label(atoms, by_proj.get("neighbours", [])),
            "neighbors_proj": by_proj.get("neighbour_labels", ""),
            "d_min (A)": round(by_dist["d_min"], 3),
            "d_min_sub (A)": d_min_sub,
            "height (A)": round(height, 3),
            "weights": by_proj.get("weights", ""),
        })
    info["warning"] = "; ".join(warnings_seen)
    return rows, info


# -----------------------------------------------------------------------------
# One set: energies, twins, sites
# -----------------------------------------------------------------------------

def analyze_set(set_dir, do_sites=True, do_energy=True, prefer_poscar=False,
                ads_override=None, pool_override=None, dist_tol=DEFAULT_DIST_TOL,
                layer_tol=DEFAULT_LAYER_TOL, hcp_tol=DEFAULT_HCP_TOL,
                main=DEFAULT_MAIN_METHOD, do_redox_surface=False,
                do_twin_sites=True, check_errors=True, quiet=False):
    """
    Analyze one AlloyGen output set and return (table, site_table, info).

    `table` has one row per structure ID (energy, dE against every twin, site
    summary); `site_table` has one row per adsorbate atom.

    With do_redox_surface, every twin is measured against the SURFACE twin
    instead of against the main folder:

        main - _surface,  _r1 - _surface,  _r2 - _surface, ...

    All of them share one reference (the clean surface), so the numbers can be
    read as one ladder -- what is left adsorbed after each redox step -- which
    differencing against the main folder cannot give.

    With do_twin_sites (the default) the sites are assigned in every redox twin
    as well, not only in the main folder: the atoms that survive a redox step
    often move to another site, and the 'folder' / 'main_atom' columns of the
    site table are what let that move be followed.
    """
    set_dir = os.path.normpath(set_dir)
    set_name = os.path.basename(os.path.abspath(set_dir))
    surface_dir, redox_dirs = find_twins(set_dir)
    ids = structure_dirs(set_dir)
    single = not ids
    metadata = read_metadata(surface_dir) if surface_dir else {}
    if not metadata:
        metadata = read_metadata(set_dir)
    redox_map = read_redox_map(set_dir)

    info = {
        "set": set_dir,
        "set_name": set_name,
        "surface_dir": surface_dir,
        "redox_dirs": redox_dirs,
        "n_structures": 1 if single else len(ids),
        "adsorbate": None,
        "warnings": [],
        "missing_twin_ids": {},
    }
    if not quiet:
        print("\n# ---------- %s ----------" % set_dir)
        if single:
            print("  structures : 1 (folder itself)")
        else:
            ready = sum(1 for i in ids
                        if prefer_poscar or has_contcar(os.path.join(set_dir, i)))
            print("  structures : %d%s"
                  % (ready, "" if ready == len(ids)
                     else "   (of %d; the rest have no CONTCAR yet)" % len(ids)))
        print("  surface twin : %s" % (surface_dir if surface_dir else "not found"))
        if redox_dirs:
            for label, path in redox_dirs:
                extra = redox_map.get(os.path.basename(path), "")
                print("  redox twin %-4s : %s%s" % (label, path,
                                                    ("   removed atoms: " + extra) if extra else ""))
        else:
            print("  redox twin : not found")

    rows, site_rows = [], []
    adsorbate_reported = False
    resolved = None
    skipped_ids = []
    for structure_id in (["."] if single else ids):
        main_dir = set_dir if single else os.path.join(set_dir, structure_id)
        # A folder with no CONTCAR is one VASP has not run yet. It has no
        # energy and no relaxed geometry, so every column it could fill would
        # be blank -- listing it only lengthens the table. -poscar is the
        # exception: there the unrelaxed input IS what was asked for.
        if not prefer_poscar and not single and not has_contcar(main_dir):
            skipped_ids.append(structure_id)
            continue
        row = {"Structure": set_name if single else structure_id}

        # Option 1 asks for sites only; reading energies there would load
        # VASPio (pymatgen, matplotlib) for numbers nobody asked for.
        energy, source, natoms = None, "", None
        if do_energy or do_redox_surface:
            energy, source = read_energy(main_dir)
            natoms = count_atoms(main_dir)
        # Which folders of this row did not finish cleanly. A difference is
        # only as good as the worse of its two folders, so the twin's verdict
        # belongs on the row that uses it -- an unconverged _surface silently
        # poisons dE_surface and every _rN - _surface with it.
        failed, unfinished = [], []
        if check_errors and (do_energy or do_redox_surface):
            row["Converged"] = check_converged(main_dir)
            row["Finished"] = check_finished(main_dir)
            if row["Converged"] == "False":
                failed.append("main")
            if row["Finished"] == "False":
                unfinished.append("main")
            twin_dirs = []
            if surface_dir:
                twin_dirs.append(("_surface", surface_dir if single
                                  else os.path.join(surface_dir, structure_id)))
            for label, path in redox_dirs:
                twin_dirs.append(("_" + label, path if single
                                  else os.path.join(path, structure_id)))
            for label, path in twin_dirs:
                if not _is_dir(path):
                    continue
                if check_converged(path) == "False":
                    failed.append(label)
                if check_finished(path) == "False":
                    unfinished.append(label)
            row["unconverged"] = ",".join(failed)
            row["unfinished"] = ",".join(unfinished)
        if do_energy:
            row["N atoms"] = natoms
            row["Energy (eV)"] = energy
            row["E source"] = source

        if do_energy and surface_dir:
            twin = surface_dir if single else os.path.join(surface_dir, structure_id)
            if _is_dir(twin):
                twin_energy, twin_source = read_energy(twin)
                row["dE_surface (eV)"] = (None if (energy is None or twin_energy is None)
                                          else energy - twin_energy)
                row["surface source"] = twin_source
            else:
                row["dE_surface (eV)"] = None
                info["missing_twin_ids"].setdefault("_surface", []).append(structure_id)

        if do_energy:
            for label, path in redox_dirs:
                twin = path if single else os.path.join(path, structure_id)
                column = "dE_%s (eV)" % label
                if _is_dir(twin):
                    twin_energy, _twin_source = read_energy(twin)
                    row[column] = (None if (energy is None or twin_energy is None)
                                   else energy - twin_energy)
                else:
                    row[column] = None
                    info["missing_twin_ids"].setdefault("_" + label, []).append(structure_id)

        if do_redox_surface:
            # One shared reference for the whole ladder: the clean surface.
            surface_energy = None
            if surface_dir:
                twin = surface_dir if single else os.path.join(surface_dir, structure_id)
                if _is_dir(twin):
                    surface_energy, _source = read_energy(twin)
                else:
                    info["missing_twin_ids"].setdefault("_surface", []).append(structure_id)
            row["E_surface (eV)"] = surface_energy
            row["main - _surface (eV)"] = (None if (energy is None or surface_energy is None)
                                           else energy - surface_energy)
            for label, path in redox_dirs:
                twin = path if single else os.path.join(path, structure_id)
                column = "_%s - _surface (eV)" % label
                if not _is_dir(twin):
                    row[column] = None
                    info["missing_twin_ids"].setdefault("_" + label, []).append(structure_id)
                    continue
                twin_energy, _source = read_energy(twin)
                row[column] = (None if (twin_energy is None or surface_energy is None)
                               else twin_energy - surface_energy)

        if do_sites:
            atoms, filename = read_structure(main_dir, prefer_poscar=prefer_poscar)
            if atoms is None:
                info["warnings"].append("%s: no readable POSCAR/CONTCAR" % main_dir)
            else:
                if resolved is None:
                    # WHICH ELEMENTS ARE THE ADSORBATE is a property of the set,
                    # so it is resolved once instead of reading the surface twin
                    # again for every structure.
                    surface_atoms = None
                    if surface_dir:
                        twin = surface_dir if single else os.path.join(surface_dir, structure_id)
                        if _is_dir(twin):
                            surface_atoms, _ = read_structure(twin, prefer_poscar=prefer_poscar)
                    resolved = resolve_adsorbate_elements(
                        atoms, surface_atoms=surface_atoms, metadata=metadata,
                        ads_override=ads_override, pool_override=pool_override)
                # ...but WHICH ELEMENTS FORM THE SURFACE is not: a set can hold
                # structures with different substitution patterns, and one that
                # has an element the first structure lacked would otherwise have
                # those atoms dropped from its surface -- leaving a hole in the
                # triangulation and a site assigned against the atoms that
                # remain. Only an explicit -pool= fixes the list for the set.
                substrate = pool_override or [
                    element for element in sorted(set(atoms.get_chemical_symbols()))
                    if element not in set(resolved["elements"])]
                if info["adsorbate"] is None:
                    info["adsorbate"] = resolved
                if not quiet and not adsorbate_reported:
                    print("  structure file : %s" % filename)
                    print("  adsorbate : %s   (from %s)"
                          % (",".join(resolved["elements"]) or "none", resolved["source"]))
                    if resolved["note"]:
                        print("  note : %s" % resolved["note"])
                    adsorbate_reported = True

                # The main folder, then every redox twin: after a redox step
                # the atoms that are left can sit somewhere else, and that move
                # is the point of the comparison. _surface is skipped -- there
                # is no adsorbate left in it by construction.
                folders = [("main", main_dir, atoms)]
                main_poscar = None
                if do_twin_sites and redox_dirs:
                    main_poscar = _read_poscar_only(main_dir)
                if do_twin_sites:
                    for label, path in redox_dirs:
                        twin = path if single else os.path.join(path, structure_id)
                        if not _is_dir(twin):
                            continue
                        if not prefer_poscar and not has_contcar(twin):
                            continue
                        twin_atoms, _twin_file = read_structure(
                            twin, prefer_poscar=prefer_poscar)
                        if twin_atoms is None:
                            info["warnings"].append(
                                "%s: no readable POSCAR/CONTCAR" % twin)
                            continue
                        folders.append((label, twin, twin_atoms))

                for folder_label, folder_dir, folder_atoms in folders:
                    site_result, site_info = analyze_sites(
                        folder_atoms, resolved["elements"],
                        substrate_elements=substrate,
                        dist_tol=dist_tol, layer_tol=layer_tol, hcp_tol=hcp_tol,
                        main=main)
                    if site_info["warning"]:
                        info["warnings"].append("%s: %s" % (folder_dir, site_info["warning"]))
                    # Atom numbers restart in every folder, so carry the main
                    # folder's number along; without it the same atom looks
                    # like a different one after a redox step.
                    labels, map_note = [], ""
                    if folder_label != "main":
                        labels, map_note = map_to_main_labels(
                            folder_dir, main_dir, folder_atoms,
                            main_poscar=main_poscar)
                    if map_note and map_note not in info["warnings"]:
                        info["warnings"].append(map_note)
                    for site_row in site_result:
                        entry = {"Structure": row["Structure"], "folder": folder_label}
                        entry.update(site_row)
                        # One identity per atom: the number it carries in the
                        # main folder, so a row means the same atom in every
                        # folder. 'local_no' is that folder's own number, for
                        # looking the atom up in its CONTCAR.
                        entry["local_no"] = site_row["atom_no"]
                        if folder_label != "main":
                            # Blank, not a wrong number: 'atom' otherwise still
                            # holds the value analyze_sites gave it, which is
                            # this folder's OWN local label (label_atom() on
                            # folder_atoms) -- a plausible-looking main-folder
                            # identity that is not actually one.
                            mapped = (labels[site_row["atom_no"] - 1] if labels else "")
                            entry["atom"] = mapped
                        site_rows.append(entry)
        rows.append(row)

    for column, odd in energy_outliers(rows).items():
        info["warnings"].append(
            "%s is far from the rest of the set in %d structure(s): %s -- a "
            "difference this size usually means the folder it came from ended "
            "in a different state, not that the chemistry changed"
            % (column, len(odd), ", ".join(odd[:10])
               + (" ..." if len(odd) > 10 else "")))
    # Two columns of "Unknown" say nothing. That happens when these folders
    # have neither OSZICAR nor vasp.out, or when custodian is not installed and
    # OSZICAR did not settle it -- either way the check could not be made, and
    # saying so once beats repeating it per row. 'Finished' is left alone here:
    # a set still being worked through has every convergence verdict Unknown,
    # and that is exactly when 'Finished' is the column worth reading.
    verdicts = {r.get("Converged") for r in rows if "Converged" in r}
    if verdicts and verdicts == {"Unknown"}:
        for r in rows:
            r.pop("Converged", None)
            r.pop("unconverged", None)
        info["warnings"].append(
            "convergence was not checked: these folders have no OSZICAR and "
            "no vasp.out, or custodian is not installed (use -nocheck to stop "
            "trying)")
    unconverged_rows = [r["Structure"] for r in rows if r.get("unconverged")]
    info["unconverged_rows"] = unconverged_rows
    if unconverged_rows:
        info["warnings"].append(
            "%d structure(s) have a folder that did not converge (its SCF hit "
            "NELM, or custodian reports an error), so their energies and "
            "differences are not trustworthy: %s"
            % (len(unconverged_rows), ", ".join(unconverged_rows[:10])
               + (" ..." if len(unconverged_rows) > 10 else "")))
    unfinished_rows = [r["Structure"] for r in rows if r.get("unfinished")]
    info["unfinished_rows"] = unfinished_rows
    if unfinished_rows:
        info["warnings"].append(
            "%d structure(s) have a folder with no vasp.done -- not run yet, or "
            "holding the output of an earlier attempt that was resubmitted -- "
            "so their energies are kept in the table but left out of the fit: "
            "%s (if these jobs simply do not write vasp.done, use -nocheck)"
            % (len(unfinished_rows), ", ".join(unfinished_rows[:10])
               + (" ..." if len(unfinished_rows) > 10 else "")))
    if skipped_ids:
        info["warnings"].append(
            "%d structure(s) have no CONTCAR yet and were left out: %s"
            % (len(skipped_ids), ", ".join(skipped_ids[:10])
               + (" ..." if len(skipped_ids) > 10 else "")))
    info["skipped_ids"] = skipped_ids
    table = pd.DataFrame(rows)
    site_table, diff_table = split_site_tables(site_rows)
    info["ensemble_table"] = ensemble_counts(site_rows, rows)
    info["n_atoms_checked"] = len(site_rows)
    info["n_disagree"] = len(diff_table) if diff_table is not None else 0
    return table, site_table, diff_table, info


def energy_outliers(rows, floor=3.0, mad_factor=6.0):
    """
    Structures whose energy column sits far outside the rest of the set.

    custodian answers "did VASP report an error"; it cannot answer "did this
    job converge to a different state than its siblings". The structures of one
    AlloyGen set differ only in their substitution pattern, so their energies
    and differences live in a narrow band -- a value tens of eV away is a
    calculation to look at, whatever the error handler said.

    Spread is measured with the median absolute deviation, not the standard
    deviation: the outliers themselves would inflate a standard deviation until
    they no longer look unusual. `floor` keeps a set whose values are nearly
    identical (MAD ~ 0) from flagging ordinary scatter.

    Returns {column: [structure ids]}.
    """
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    found = {}
    for column in frame.columns:
        if "(eV)" not in column or column == "Energy (eV)":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 5:
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        limit = max(floor, mad_factor * mad)
        odd = frame.loc[(values - median).abs() > limit, "Structure"]
        if len(odd):
            found[column] = [str(x) for x in odd]
    return found


def fit_target(frame, folder):
    """
    The energy a folder's sites are responsible for, and its name.

    For the main folder that is dE_surface. For a redox twin it is
    `_rN - _surface` -- the same quantity one redox step further along, and the
    only one whose value is a sum over THAT folder's sites. When option 4 has
    not been run the column is not there, but it is still recoverable from the
    two option-2 columns:

        (main - surface) - (main - rN) = rN - surface

    so the twins get their contributions either way.
    """
    if folder == "main":
        for candidate in ("dE_surface (eV)", "main - _surface (eV)"):
            if candidate in frame:
                return pd.to_numeric(frame[candidate], errors="coerce"), candidate
        return None, None
    direct = "_%s - _surface (eV)" % folder
    if direct in frame:
        return pd.to_numeric(frame[direct], errors="coerce"), direct
    surface, twin = "dE_surface (eV)", "dE_%s (eV)" % folder
    if surface in frame and twin in frame:
        derived = (pd.to_numeric(frame[surface], errors="coerce")
                   - pd.to_numeric(frame[twin], errors="coerce"))
        return derived, "_%s - _surface (eV)" % folder
    return None, None


def ensemble_energy_fit(site_rows, rows, folder="main", min_atoms=5):
    """
    How much each element combination contributes to the adsorption energy.

    A structure carries ONE energy but several adsorbate atoms, so no site owns
    a share of it outright. What can be done is to fit the shares: with the
    structures of a set differing only in their substitution pattern,

        dE(structure) = sum over its adsorbate atoms of contribution(site type)

    is a least-squares problem whose unknowns are the per-(element, site,
    ensemble) contributions. That is an ADDITIVITY assumption, and R^2 and the
    residual RMS are reported so it can be judged rather than believed.

    ⭐ Only differences WITHIN one (element, site) group are meaningful, and
    that is what the returned numbers are -- each contribution is expressed
    relative to the average ensemble of its own group. The reason is that every
    structure of a set holds the same number of each site type (e.g. always two
    fcc hollows and one bridge), so adding a constant to every bridge
    contribution and taking it off every hollow one changes no prediction at
    all: the absolute level of a group is not identifiable from this data, only
    the spread inside it. Returning raw coefficients would put a precise-looking
    number on something the data cannot say. The centring is a linear contrast,
    so its standard error is exact, not approximated.

    Structures whose energy could not be trusted -- an unconverged folder, one
    that has not finished (no vasp.done, so its output is from an earlier
    attempt or from nothing yet), or a value far outside the set -- are left
    out of the fit.

    Returns (contributions, stats): {key: (value, stderr)} keyed by
    "element site ensemble", and a dict with r2 / rms / n_structures / column.
    """
    if not rows or not site_rows:
        return {}, {}
    frame = pd.DataFrame(rows)
    values, column = fit_target(frame, folder)
    if column is None:
        return {}, {}

    energies = {}
    suspect = set()
    for position, (_i, row) in enumerate(frame.iterrows()):
        if row.get("unconverged") or row.get("unfinished"):
            suspect.add(row["Structure"])
        value = values.iloc[position]
        if pd.notna(value):
            energies[row["Structure"]] = float(value)
    for name, odd in energy_outliers(rows).items():
        # A structure whose main energy is off poisons every column built on
        # it, so one bad row is dropped from every folder's fit, not just its own.
        suspect.update(odd)
    for structure in suspect:
        energies.pop(structure, None)
    if len(energies) < 10:
        return {}, {}

    # counts[structure][key] -- one folder at a time: each folder answers to its
    # own energy column, and mixing them would fit two quantities at once.
    counts, totals = {}, Counter()
    for row in site_rows:
        if row.get("folder") != folder or row["Structure"] not in energies:
            continue
        key = "%s %s %s" % (str(row["atom"]).split("#")[0], row["site"], row["ensemble"])
        counts.setdefault(row["Structure"], Counter())[key] += 1
        totals[key] += 1
    if not counts:
        return {}, {}

    keys = sorted(k for k, n in totals.items() if n >= min_atoms)
    if not keys:
        return {}, {}
    index = {k: i for i, k in enumerate(keys)}
    structures = sorted(counts)
    matrix = np.zeros((len(structures), len(keys) + 1))
    target = np.zeros(len(structures))
    for r, structure in enumerate(structures):
        target[r] = energies[structure]
        for key, n in counts[structure].items():
            # Everything too rare to fit on its own still has to be in the sum,
            # or its energy would be pushed onto the sites that are.
            matrix[r, index.get(key, len(keys))] += n

    coefficients, _res, _rank, _sv = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix.dot(coefficients)
    rank = int(np.linalg.matrix_rank(matrix))
    dof = max(1, len(structures) - rank)
    variance = float(residual.dot(residual)) / dof
    covariance = variance * np.linalg.pinv(matrix.T.dot(matrix))

    # Centre within each (element, site) group -- see the note above.
    groups = {}
    for key in keys:
        element, site, _ensemble = key.split(" ", 2)
        groups.setdefault((element, site), []).append(key)
    contributions = {}
    for members in groups.values():
        weights = np.array([totals[k] for k in members], dtype=float)
        weights /= weights.sum()
        for key in members:
            contrast = np.zeros(len(keys) + 1)
            contrast[index[key]] = 1.0
            for other, weight in zip(members, weights):
                contrast[index[other]] -= weight
            value = float(contrast.dot(coefficients))
            error = float(np.sqrt(max(0.0, contrast.dot(covariance).dot(contrast))))
            contributions[key] = (round(value, 3), round(error, 3))

    spread = target - target.mean()
    stats = {
        "folder": folder,
        "column": column,
        "n_structures": len(structures),
        "n_terms": len(keys),
        "r2": round(1.0 - float(residual.dot(residual)) / float(spread.dot(spread)), 3)
              if float(spread.dot(spread)) > 0 else None,
        "rms": round(float(np.sqrt(residual.dot(residual) / len(structures))), 3),
        "spread": round(float(target.std()), 3),
        "dropped": len(suspect),
    }
    return contributions, stats


def ensemble_counts(site_rows, rows=None):
    """
    Count how often each (site geometry, element composition) is occupied.

    'hollow3' repeated 500 times says little; 'hollow3 on Fe1Ni2, 46 times'
    is the sentence a substituted surface is actually asked about. Structures
    are counted as well as atoms, because one structure can hold the same
    ensemble more than once and a raw atom count would read as more
    independent evidence than there is.

    Energies are deliberately not averaged in here: a structure carries one
    energy but can contain several different sites at once, so a per-site
    energy would be an unattributed share of it.
    """
    if not site_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(site_rows)
    if "ensemble" not in frame:
        return pd.DataFrame()
    frame = frame.copy()
    # The element comes from the row itself, not from parsing the 'atom' label.
    # 'atom' is blank whenever the map back to the main folder failed, and
    # splitting that string would file those atoms under an empty element --
    # the same species would then be counted in two separate groups.
    if "element" not in frame:
        frame["element"] = [str(atom).split("#")[0] for atom in frame["atom"]]
    grouped = frame.groupby(["folder", "element", "site", "ensemble"], dropna=False)
    out = grouped.agg(**{
        "atoms": ("atom", "size"),
        "structures": ("Structure", "nunique"),
        "mean d_min (A)": ("d_min (A)", "mean"),
        "mean height (A)": ("height (A)", "mean"),
    }).reset_index()
    out["mean d_min (A)"] = out["mean d_min (A)"].round(3)
    out["mean height (A)"] = out["mean height (A)"].round(3)
    contributions, fits = {}, []
    if rows:
        for folder in sorted(out.folder.unique(), key=lambda f: (f != "main", f)):
            found, stats = ensemble_energy_fit(site_rows, rows, folder=folder)
            if not found:
                continue
            contributions.update({(folder, k): v for k, v in found.items()})
            fits.append(stats)
    if contributions:
        keys = out.element + " " + out.site + " " + out.ensemble
        pairs = [contributions.get((folder, k), (None, None))
                 for k, folder in zip(keys, out.folder)]
        out["dE vs group avg (eV)"] = [p[0] for p in pairs]
        out["stderr (eV)"] = [p[1] for p in pairs]
        out.attrs["fit"] = fits
    out = out.sort_values(["folder", "element", "atoms"], ascending=[True, True, False])
    columns = [c for c in ENSEMBLE_COLUMNS if c in out]
    result = out[columns].reset_index(drop=True)
    result.attrs["fit"] = out.attrs.get("fit", [])
    return result


def split_site_tables(site_rows):
    """
    Turn the rich per-atom rows into the two tables that get written:

    - the site table: one answer per atom plus the agree flag
    - the diff table: only the atoms where the two methods did not agree,
      with both answers and the numbers needed to settle it by hand

    Keeping the losing method out of the first table is the point. With both
    methods' columns side by side nothing stands out, and the few rows that
    actually need a human get lost among the many that do not.
    """
    if not site_rows:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(site_rows)
    columns = list(SITE_COLUMNS)
    # A slab with nothing on its underside does not need a column saying so.
    if "side" in frame and (frame["side"] == "top").all():
        columns = [c for c in columns if c != "side"]
    elif "side" in frame:
        columns.insert(columns.index("atom") + 1, "side")
    site_table = frame[[c for c in columns if c in frame]]
    disagreed = frame[frame["agree"] != "same"] if "agree" in frame else frame.iloc[0:0]
    diff_table = disagreed[[c for c in DIFF_COLUMNS if c in disagreed]]
    return site_table.reset_index(drop=True), diff_table.reset_index(drop=True)


def write_tables(table, site_table, set_name, out_dir="./", do_sites=True,
                 do_energy=True, kind="AlloyAnal", diff_table=None,
                 ensemble_table=None):
    """
    Save the results next to the other CCpy analysis files, following the same
    '<number>_<folder>_<what>' naming as 03_..._FinalEnergies of CCpyVASPAnal.
    `kind` keeps a different question in a different file instead of
    overwriting the previous answer with a table of other columns.
    Returns the list of file names written.
    """
    written = []
    # With option 1 the per-structure table holds nothing but the ID column;
    # writing a file of row labels only would be noise.
    if table is not None and len(table) and len(table.columns) > 1:
        base = os.path.join(out_dir, "04_" + set_name + "_" + kind)
        table.to_csv(base + ".csv")
        with open(base + ".txt", "w") as handle:
            handle.write(table.to_string())
        written.extend([base + ".csv", base + ".txt"])
    if do_sites and site_table is not None and len(site_table):
        base = os.path.join(out_dir, "04_" + set_name + "_AdsorptionSites")
        site_table.to_csv(base + ".csv")
        with open(base + ".txt", "w") as handle:
            handle.write(site_table.to_string())
        written.extend([base + ".csv", base + ".txt"])
    if do_sites and ensemble_table is not None and len(ensemble_table):
        base = os.path.join(out_dir, "04_" + set_name + "_SiteEnsembles")
        ensemble_table.to_csv(base + ".csv")
        with open(base + ".txt", "w") as handle:
            handle.write(ensemble_table.to_string())
        written.extend([base + ".csv", base + ".txt"])
    if do_sites and diff_table is not None and len(diff_table):
        base = os.path.join(out_dir, "04_" + set_name + "_SiteDiff")
        diff_table.to_csv(base + ".csv")
        with open(base + ".txt", "w") as handle:
            handle.write(diff_table.to_string())
        written.extend([base + ".csv", base + ".txt"])
    return written
