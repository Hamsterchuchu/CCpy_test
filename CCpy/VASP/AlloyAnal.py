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
CCpy.VASP.VASPio.energy_from_outcar() first and falling back to
energy_from_oszicar() only when OUTCAR is missing (TOTEN and E0 differ by the
-TS term, so the two are never mixed within one column without saying which
file each value came from).

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

import numpy as np
import pandas as pd
from ase.io import read as ase_read

# Energy reading is shared with CCpyVASPAnal.py option 2 -- same functions,
# same OUTCAR-then-OSZICAR order, so the two commands can never disagree.
from CCpy.VASP.VASPio import energy_from_outcar, energy_from_oszicar, natoms_from_poscar

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


def _is_dir(path):
    return os.path.isdir(path)


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

def count_atoms(directory):
    """
    Number of atoms of one folder. POSCAR is read with VASPio's own counter
    (the same one CCpyVASPAnal uses); CONTCAR is only looked at when POSCAR is
    missing, so a folder holding just a relaxed result still gets a count.
    """
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


def read_energy(directory):
    """
    Final energy of one VASP folder, the CCpyVASPAnal.py way: OUTCAR TOTEN
    first, the last E0 of OSZICAR only as a fallback. Returns (energy, source)
    with source "" when nothing could be read.
    """
    energy = energy_from_outcar(directory)
    if energy is not None:
        return energy, "OUTCAR"
    energy = energy_from_oszicar(directory)
    if energy is not None:
        return energy, "OSZICAR"
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

def site_by_projection(atoms, adsorbate_index, top_layer, second_layer, normal,
                       basis, hcp_tol=DEFAULT_HCP_TOL):
    """
    Project the adsorbate onto the surface plane and locate it inside the
    Delaunay triangulation of the top-layer atoms, replicated over the
    neighbouring periodic images. Barycentric coordinates then decide
    vertex (top) / edge (bridge) / interior (3-fold hollow).

    Returns a dict with site="unresolved" when the projected point falls
    outside every triangle, which happens on a badly broken surface -- that is
    reported rather than guessed.
    """
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return {"site": "unavailable", "flavour": "", "neighbours": [],
                "neighbour_labels": "", "note": "scipy is required for the "
                                                "projection method"}
    u1, u2, (a1, a2) = basis
    vectors = atoms.get_distances(adsorbate_index, top_layer, mic=True, vector=True)
    points, sources = [], []
    for shift1 in (-1, 0, 1):
        for shift2 in (-1, 0, 1):
            translation = shift1 * np.asarray(a1) + shift2 * np.asarray(a2)
            shifted = vectors + translation
            flat = in_plane(shifted, normal)
            for k in range(len(top_layer)):
                points.append([float(np.dot(flat[k], u1)), float(np.dot(flat[k], u2))])
                sources.append(top_layer[k])
    points = np.array(points)
    # Spacing of the top layer itself, used below to tell a real bridge from a
    # triangulation diagonal.
    if len(points) > 1:
        spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(spread, np.inf)
        nn_spacing = float(spread.min())
    else:
        nn_spacing = None
    if len(points) < 3:
        return {"site": "unresolved", "flavour": "", "neighbours": [],
                "neighbour_labels": "", "note": "too few surface atoms"}
    try:
        triangulation = Delaunay(points)
    except Exception as error:
        return {"site": "unresolved", "flavour": "", "neighbours": [],
                "neighbour_labels": "", "note": "triangulation failed (%s)" % error}
    simplex = int(triangulation.find_simplex(np.array([[0.0, 0.0]]))[0])
    if simplex < 0:
        return {"site": "unresolved", "flavour": "", "neighbours": [],
                "neighbour_labels": "",
                "note": "projected position is outside the surface triangulation"}

    vertices = triangulation.simplices[simplex]
    transform = triangulation.transform[simplex]
    bary2 = transform[:2].dot(np.array([0.0, 0.0]) - transform[2])
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
            distances2d = np.linalg.norm(points, axis=1)
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
        centre = centre2d[0] * u1 + centre2d[1] * u2
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
                  dist_tol=DEFAULT_DIST_TOL, hcp_tol=DEFAULT_HCP_TOL):
    """
    Run both methods against ONE substrate layer and return (dist, proj).

    `next_layer` is used only to tell fcc from hcp, i.e. to ask whether
    anything sits directly under a three-fold hollow.
    """
    by_dist = site_by_distance(atoms, index, layer, next_layer, normal,
                               dist_tol=dist_tol, hcp_tol=hcp_tol)
    by_proj = site_by_projection(atoms, index, layer, next_layer, normal,
                                 basis, hcp_tol=hcp_tol)
    return by_dist, by_proj


def site_label(result):
    """'hollow3-fcc' from one method's result dict."""
    site = result.get("site", "")
    flavour = result.get("flavour", "")
    return site + ("-" + flavour if flavour else "")


UNRESOLVED = ("unresolved", "unavailable")


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
                                         hcp_tol=hcp_tol)
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

        sub_site, sub_neighbours, d_min_sub = "", "", None
        if second_layer:
            sub_dist, sub_proj = classify_atom(atoms, index, second_layer,
                                               third_layer, normal, basis,
                                               dist_tol=dist_tol, hcp_tol=hcp_tol)
            sub_primary = sub_proj if main == "proj" else sub_dist
            if sub_primary.get("site") in UNRESOLVED:
                sub_primary = sub_dist
            sub_site = site_label(sub_primary)
            sub_neighbours = sub_primary.get("neighbour_labels", "")
            d_min_sub = round(float(sub_dist["d_min"]), 3)

        height = float(signed[index] - np.mean(signed[top_layer]))
        rows.append({
            "atom": label_atom(atoms, index),
            "element": symbols[index],
            "atom_no": index + 1,
            "side": "top" if side > 0 else "bottom",
            "site": site_label(primary),
            "neighbors": main_neighbours,
            "agree": agree,
            "sub_site": sub_site,
            "sub_neighbors": sub_neighbours,
            "site_dist": site_label(by_dist),
            "neighbors_dist": by_dist["neighbour_labels"],
            "site_proj": site_label(by_proj),
            "neighbors_proj": by_proj.get("neighbour_labels", ""),
            "d_min (A)": round(by_dist["d_min"], 3),
            "d_min_sub (A)": d_min_sub,
            "height (A)": round(height, 3),
        })
    info["warning"] = "; ".join(warnings_seen)
    return rows, info


def summarize_sites(rows):
    """
    One short string per structure: 'Li#33 top>hollow3 / S#35 hollow3-fcc'.

    The sub-layer part is appended only where it says something new. For a
    three-fold hollow the fcc/hcp flavour already IS the second-layer answer,
    so repeating it would only make the line longer.
    """
    if not rows:
        return ""
    pieces = []
    for row in rows:
        label = row["site"]
        sub = row.get("sub_site") or ""
        if sub and not label.startswith("hollow3"):
            label = "%s>%s" % (label, sub.split("-")[0])
        pieces.append("%s %s" % (row["atom"], label))
    return " / ".join(pieces)


# -----------------------------------------------------------------------------
# One set: energies, twins, sites
# -----------------------------------------------------------------------------

def analyze_set(set_dir, do_sites=True, do_energy=True, prefer_poscar=False,
                ads_override=None, pool_override=None, dist_tol=DEFAULT_DIST_TOL,
                layer_tol=DEFAULT_LAYER_TOL, hcp_tol=DEFAULT_HCP_TOL,
                main=DEFAULT_MAIN_METHOD, quiet=False):
    """
    Analyze one AlloyGen output set and return (table, site_table, info).

    `table` has one row per structure ID (energy, dE against every twin, site
    summary); `site_table` has one row per adsorbate atom.
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
        print("  structures : %s" % ("1 (folder itself)" if single else len(ids)))
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
    for structure_id in (["."] if single else ids):
        main_dir = set_dir if single else os.path.join(set_dir, structure_id)
        row = {"Structure": set_name if single else structure_id}

        energy, source = read_energy(main_dir)
        natoms = count_atoms(main_dir)
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

        if do_sites:
            atoms, filename = read_structure(main_dir, prefer_poscar=prefer_poscar)
            if atoms is None:
                row["Sites"] = ""
                info["warnings"].append("%s: no readable POSCAR/CONTCAR" % main_dir)
            else:
                surface_atoms = None
                if surface_dir:
                    twin = surface_dir if single else os.path.join(surface_dir, structure_id)
                    if _is_dir(twin):
                        surface_atoms, _ = read_structure(twin, prefer_poscar=prefer_poscar)
                resolved = resolve_adsorbate_elements(
                    atoms, surface_atoms=surface_atoms, metadata=metadata,
                    ads_override=ads_override, pool_override=pool_override)
                if info["adsorbate"] is None:
                    info["adsorbate"] = resolved
                if not quiet and not adsorbate_reported:
                    print("  structure file : %s" % filename)
                    print("  adsorbate : %s   (from %s)"
                          % (",".join(resolved["elements"]) or "none", resolved["source"]))
                    if resolved["note"]:
                        print("  note : %s" % resolved["note"])
                    adsorbate_reported = True
                site_result, site_info = analyze_sites(
                    atoms, resolved["elements"],
                    substrate_elements=resolved.get("substrate"),
                    dist_tol=dist_tol, layer_tol=layer_tol, hcp_tol=hcp_tol,
                    main=main)
                if site_info["warning"]:
                    info["warnings"].append("%s: %s" % (main_dir, site_info["warning"]))
                for site_row in site_result:
                    entry = {"Structure": row["Structure"]}
                    entry.update(site_row)
                    site_rows.append(entry)
                row["Sites"] = summarize_sites(site_result)
        rows.append(row)

    table = pd.DataFrame(rows)
    site_table = pd.DataFrame(site_rows)
    return table, site_table, info


def write_tables(table, site_table, set_name, out_dir="./", do_sites=True,
                 do_energy=True):
    """
    Save the results next to the other CCpy analysis files, following the same
    '<number>_<folder>_<what>' naming as 03_..._FinalEnergies of CCpyVASPAnal.
    Returns the list of file names written.
    """
    written = []
    if table is not None and len(table):
        base = os.path.join(out_dir, "04_" + set_name + "_AlloyAnal")
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
    return written
