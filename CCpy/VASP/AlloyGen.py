#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlloyGen.py  (CCpy alloy / HEA structure generation engine)

Pt-based HEA / intermetallic substitution structure generator, integrated
into CCpy. Based on Gen-HEA-pomepaw_v8.py; the command-line front-end is
CCpy/bin/CCpyAlloyGen.py (CCpy-style options). On top of the original
generator, `generate_ccpy_vasp_inputs()` connects the generated structures
to CCpy's VASPInput (CCpy.VASP.VASPio), so every structure folder can get a
full INCAR / KPOINTS / POTCAR set from the yaml presets in CCpy's personal
config folder - the same machinery used by CCpyVASPInputGen.py. Where that
folder lives is decided in one place, CCpy/Tools/CCpyConfig.py, and this
module asks it at runtime (vasp_preset_dir_label()) instead of repeating the
path in help text.

Main improvements over the original Code500.py:
1) Correct symmetry deduplication using canonical decoration keys.
2) spglib symmetry search is performed only once on the parent structure.
3) Flexible composition, replacement element, and generation mode.
4) Supports both random sampling and exhaustive enumeration for small cases.
5) Four physical structure axes: random, spread, layered, and domain.
6) Layered/domain ordered-parent generation controlled by a symmetry-invariant,
   composition-corrected overlap Q plus first-shell Warren-Cowley SRO.
7) Short structure IDs (S000001) with full metadata in structures.csv.
8) Can write CIF files or VASP POSCAR-style folders.

Recommended entry point (installed CCpy command)
-------------------------------------------------
   CCpyAlloyGen.py [option] [sub_options]     (see CCpy/bin/CCpyAlloyGen.py)

Interactive usage (settings sheet)
---------------------------------
   CCpyAlloyGen.py w          (or any mode number 1-5; the number presets 'mode')
Running the CCpyAlloyGen.py command with no option prints its usage reference,
the same habit as CCpyVASPInputGen.py. Running *this module* directly with no
argument opens the sheet:
   python AlloyGen.py

Direct module command-line examples (legacy argparse interface)
----------------------------------------------------------------
1. Random Pt32 -> Pt16Fe4Co4Ni4Cu4:
   python AlloyGen.py --input Pt32.cif --output Pt16_HEA \
       --replace-element Pt --composition Fe:4,Co:4,Ni:4,Cu:4 \
       --mode random --target 500 --format cif

2. Spread-biased Pt32 -> Pt16Fe4Co4Ni4Cu4:
   python AlloyGen.py --input Pt32.cif --output Pt16_HEA_spread \
       --replace-element Pt --composition Fe:4,Co:4,Ni:4,Cu:4 \
       --mode spread --target 500 --format cif

3. Exhaustive Co8 -> Cu2Co2Ni2Fe2 on Pt24Co8:
   python AlloyGen.py --input Pt3Co.cif --output Pt24_Cu2Co2Ni2Fe2 \
       --replace-element Co --composition Cu:2,Co:2,Ni:2,Fe:2 \
       --mode exhaustive --format cif

4. Layered ordered parent -> controlled ordered-to-random structures:
   python AlloyGen.py --input Pt32.cif --output Pt_layered_HEA \
       --replace-element Pt --composition Fe:8,Co:8,Ni:8,Cu:8 \
       --mode layered --layer-axis z --order-levels 1,0.75,0.5,0.25,0 --target 100
"""

import argparse
import csv
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from math import exp, factorial

import numpy as np
import spglib
from ase.io import read, write

# Line-buffer stdout so progress messages (structure counts, template-copy
# progress, etc.) show up immediately instead of being held in a block
# buffer when stdout is not an interactive terminal (e.g. piped output,
# some IDE/GUI consoles). This has no effect on correctness, only on when
# already-printed lines become visible.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass


# -----------------------------------------------------------------------------
# Where the yaml presets live
# -----------------------------------------------------------------------------
# CCpy copies the packaged yaml files into a personal config folder on first
# run and reads that copy afterwards. The folder name is defined in exactly one
# place -- CCpy/Tools/CCpyConfig.py (currently `~/.CCpy_test`, overridable with
# $CCpy_HOME). Help text and sheet hints ask for it at runtime through the
# function below, so renaming the folder can never leave stale paths behind
# (which is what happened when `~/.CCpy` was hardcoded in five places).
try:
    from CCpy.Tools.CCpyConfig import vasp_config_dir as _ccpy_vasp_config_dir
except Exception:                       # running the module standalone
    _ccpy_vasp_config_dir = None


def vasp_preset_dir_label():
    """Display path of the preset yaml folder, e.g. '~/.CCpy_test/vasp/'."""
    if _ccpy_vasp_config_dir is None:
        return "the CCpy config folder (see CCpy/Tools/CCpyConfig.py)"
    path = str(_ccpy_vasp_config_dir()).replace(os.sep, "/")
    home = os.path.expanduser("~").replace(os.sep, "/")
    if home and path.startswith(home):
        path = "~" + path[len(home):]
    return path.rstrip("/") + "/"


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def parse_composition(comp_str):
    """
    Parse a replacement composition.

    Accepted examples
    -----------------
    Fe:4,Co:4,Ni:4,Cu:4
    Fe4 Co4 Ni4 Cu4
    Fe4, Co4, Ni4, Cu4
    Fe=4 Co=4 Ni=4 Cu=4
    """
    if comp_str is None or not str(comp_str).strip():
        raise ValueError("Composition is empty.")

    text = str(comp_str).strip()
    token_pattern = re.compile(r"([A-Z][a-z]?)\s*(?::|=)?\s*(\d+)")
    comp = {}
    cursor = 0

    for match in token_pattern.finditer(text):
        separator = text[cursor:match.start()]
        if separator.strip(" ,;\t\r\n"):
            raise ValueError(
                f"Invalid composition near {separator!r}. "
                "Use forms such as 'Co8 Fe8' or 'Co:8,Fe:8'."
            )

        el = match.group(1)
        n = int(match.group(2))
        if n <= 0:
            raise ValueError(f"Composition count must be positive: {el}{n}")
        comp[el] = comp.get(el, 0) + n
        cursor = match.end()

    trailing = text[cursor:]
    if trailing.strip(" ,;\t\r\n"):
        raise ValueError(
            f"Invalid composition near {trailing!r}. "
            "Use forms such as 'Co8 Fe8' or 'Co:8,Fe:8'."
        )
    if not comp:
        raise ValueError(
            "No valid composition was found. "
            "Use forms such as 'Co8 Fe8' or 'Co:8,Fe:8'."
        )
    return comp


def parse_element_list(text):
    """
    Parse a list of element symbols that defines a replacement site pool.

    Unlike parse_composition(), no counts are attached here: this only names
    which elements' current sites should be pooled together as replaceable.

    Accepted examples
    -----------------
    Pt
    Co,Fe,Ni,Cu
    Co Fe Ni Cu
    Co, Fe; Ni Cu
    """
    if text is None or not str(text).strip():
        raise ValueError("Element list is empty.")

    tokens = [t for t in re.split(r"[,\s;]+", str(text).strip()) if t]
    elements = []
    seen = set()
    for tok in tokens:
        if not re.fullmatch(r"[A-Z][a-z]?", tok):
            raise ValueError(
                f"Invalid element symbol: {tok!r}. "
                "Use forms such as 'Pt' or 'Co,Fe,Ni,Cu'."
            )
        if tok in seen:
            continue
        seen.add(tok)
        elements.append(tok)

    if not elements:
        raise ValueError("No valid element symbols were found.")
    return elements


def composition_to_list(composition):
    """Convert {'Fe':4,'Co':4} -> ['Fe','Fe','Fe','Fe','Co',...]."""
    elements = []
    for el, n in composition.items():
        elements.extend([el] * int(n))
    return elements


def multiset_count(composition):
    """Number of unique multiset permutations."""
    total = sum(composition.values())
    den = 1
    for n in composition.values():
        den *= factorial(n)
    return factorial(total) // den


def _is_same_or_inside(path, directory):
    """Return True when path is directory itself or is located below it."""
    path = os.path.abspath(os.path.realpath(path))
    directory = os.path.abspath(os.path.realpath(directory))
    try:
        return os.path.commonpath([path, directory]) == directory
    except ValueError:
        # Different Windows drives have no common path.
        return False


def ensure_clean_dir(path, overwrite=False, protected_paths=()):
    """Create an empty output directory without silently mixing or deleting data."""
    if path is None or not str(path).strip():
        raise ValueError("Output directory must not be empty.")

    output = os.path.abspath(os.path.realpath(path))
    drive_root = os.path.abspath(os.path.join(output, os.pardir)) == output
    if drive_root or output == os.path.abspath(os.getcwd()):
        raise ValueError(f"Refusing to use a filesystem root or current directory as output: {output}")

    for protected in protected_paths:
        if protected and _is_same_or_inside(protected, output):
            raise ValueError(
                f"Refusing to clean output directory because it contains a protected input/template: "
                f"{protected} (output={output})"
            )

    if os.path.exists(output):
        if not os.path.isdir(output):
            raise NotADirectoryError(f"Output path exists but is not a directory: {output}")
        if overwrite:
            shutil.rmtree(output)
        elif os.listdir(output):
            raise FileExistsError(
                f"Output directory is not empty: {output}. "
                "Choose a new directory or explicitly enable --overwrite."
            )
    os.makedirs(output, exist_ok=True)
    return output


def resolve_seed(seed=None):
    """
    Return an actual integer seed.

    If seed is None, generate a time-based seed and return it so that the
    run can be reproduced later from metadata.txt.
    """
    if seed is None:
        return int(time.time())
    return int(seed)


def write_metadata(output_dir, metadata):
    """Write run settings to metadata.txt for reproducibility."""
    path = os.path.join(output_dir, "metadata.txt")
    with open(path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key:<22} = {value}\n")
    return path


# -----------------------------------------------------------------------------
# Symmetry: precompute parent permutations once, then use canonical key
# -----------------------------------------------------------------------------

def _wrap_frac(x):
    return x - np.floor(x)


def precompute_symmetry_permutations(parent, symprec=1e-3):
    """
    Compute symmetry operations for the parent structure only once and convert
    each operation into an atom-index permutation.

    Returns
    -------
    list[np.ndarray]
        Each permutation p has p[i] = index of the atom to which atom i is mapped.
    """
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError(f"symprec must be a finite positive number, got {symprec!r}")

    lattice = np.asarray(parent.get_cell(), dtype=float)
    frac = _wrap_frac(parent.get_scaled_positions())
    atomic_numbers = np.asarray(parent.get_atomic_numbers())
    cell = (lattice, frac, atomic_numbers)
    symm = spglib.get_symmetry(cell, symprec=symprec)
    if symm is None:
        raise RuntimeError("spglib could not determine symmetry. Check the cell and symprec.")

    rots = symm["rotations"]
    trans = symm["translations"]
    n_atoms = len(parent)
    mapping_tol = max(1e-7, 5.0 * float(symprec))

    perms = []
    failed = 0
    for R, t in zip(rots, trans):
        new_frac = _wrap_frac(frac @ R.T + t)
        perm = np.empty(n_atoms, dtype=np.int32)
        used = set()
        ok = True
        for i in range(n_atoms):
            delta = frac - new_frac[i]
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ lattice, axis=1)
            compatible = np.where(atomic_numbers == atomic_numbers[i])[0]
            candidates = sorted(compatible, key=lambda j: distances[j])
            j = next((int(j) for j in candidates if int(j) not in used), None)
            if j is None or distances[j] > mapping_tol:
                ok = False
                failed += 1
                break
            perm[i] = j
            used.add(j)
        if ok:
            perms.append(perm)

    # Remove duplicate permutations
    unique = []
    seen = set()
    for p in perms:
        tp = tuple(p.tolist())
        if tp not in seen:
            seen.add(tp)
            unique.append(p)

    if not unique:
        raise RuntimeError(
            "No valid symmetry permutations were generated. Try increasing symprec, e.g., 1e-2."
        )
    if failed:
        raise RuntimeError(
            f"Failed to map {failed} spglib symmetry operations to atom indices. "
            "Uniqueness would not be guaranteed; check the structure or adjust symprec."
        )

    return unique, failed


def canonical_decoration_key(atoms, perms, key_indices=None):
    """
    Symmetry-invariant canonical key for a decorated structure.

    Unlike equivalent_atoms-only hashing, this keeps the full element decoration
    and removes only truly symmetry-equivalent configurations.
    """
    nums = np.asarray(atoms.get_atomic_numbers(), dtype=np.int16)
    best = None

    if key_indices is None:
        for p in perms:
            cand = tuple(nums[p].tolist())
            if best is None or cand < best:
                best = cand
    else:
        key_indices = np.asarray(key_indices, dtype=np.int32)
        for p in perms:
            cand = tuple(nums[p[key_indices]].tolist())
            if best is None or cand < best:
                best = cand
    return best


# -----------------------------------------------------------------------------
# Structure generation modes
# -----------------------------------------------------------------------------

def random_configuration(parent, replace_sites, elements):
    """Randomly choose sites and randomly assign elements."""
    atoms = parent.copy()
    chosen = random.sample(replace_sites, len(elements))
    els = elements[:]
    random.shuffle(els)
    for idx, el in zip(chosen, els):
        atoms[idx].symbol = el
    return atoms, chosen


def farthest_point_sites(atoms, candidate_sites, k):
    """Greedy selection of k sites that are mutually far apart under MIC."""
    candidate_sites = list(candidate_sites)
    if k > len(candidate_sites):
        raise ValueError("k is larger than the number of candidate sites.")

    first = random.choice(candidate_sites)
    chosen = [first]
    remaining = [x for x in candidate_sites if x != first]

    while len(chosen) < k:
        best_site = None
        best_score = -1.0
        for s in remaining:
            dmin = np.min(atoms.get_distances(s, chosen, mic=True))
            if dmin > best_score:
                best_score = dmin
                best_site = s
        chosen.append(best_site)
        remaining.remove(best_site)
    return chosen


def assign_elements_spread(atoms, chosen_sites, elements):
    """Assign elements so that identical elements are as far apart as possible."""
    out = atoms.copy()
    remaining_sites = list(chosen_sites)
    random.shuffle(remaining_sites)

    # Put rare/high-count elements in deterministic order, with random tie breaker.
    element_sequence = elements[:]
    random.shuffle(element_sequence)
    element_sequence.sort(key=lambda x: Counter(elements)[x], reverse=True)

    placed = {el: [] for el in set(elements)}

    for el in element_sequence:
        best_site = None
        best_score = -1.0
        for s in remaining_sites:
            if placed[el]:
                score = np.min(out.get_distances(s, placed[el], mic=True))
            else:
                score = np.mean(out.get_distances(s, chosen_sites, mic=True))
            if score > best_score:
                best_score = score
                best_site = s
        out[best_site].symbol = el
        placed[el].append(best_site)
        remaining_sites.remove(best_site)

    return out


def spread_configuration(parent, replace_sites, elements):
    """Choose replacement sites far apart and assign identical elements far apart."""
    chosen = farthest_point_sites(parent, replace_sites, len(elements))
    atoms = assign_elements_spread(parent, chosen, elements)
    return atoms, chosen


def parse_domain_pattern(pattern, composition):
    """
    Parse one human-readable 2x2 top-view domain pattern.

    Example
    -------
    pattern = "Co,Fe/Ni,Cu"

    means, for view_axis='z':
        top-left     = Co
        top-right    = Fe
        bottom-left  = Ni
        bottom-right = Cu

    The returned list is [(top_left, count), (top_right, count),
    (bottom_left, count), (bottom_right, count)].
    """
    if pattern is None or str(pattern).strip() == "":
        raise ValueError("No domain_pattern was provided.")

    raw = str(pattern).replace(";", "/").strip()
    rows = raw.split("/")
    if len(rows) != 2:
        raise ValueError(
            "domain_pattern must have two rows, e.g. 'Co,Fe/Ni,Cu'."
        )

    elems = []
    for row in rows:
        parts = [x.strip() for x in row.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError(
                "Each domain row must contain two elements, e.g. 'Co,Fe/Ni,Cu'."
            )
        elems.extend(parts)

    return validate_domain_elements(elems, composition)


def validate_domain_elements(elems, composition):
    """Validate four domain elements and attach composition counts."""
    if len(elems) != 4:
        raise ValueError("domain mode requires exactly four domain elements.")
    if len(set(elems)) != 4:
        raise ValueError("domain mode requires four different elements.")
    for el in elems:
        if el not in composition:
            raise ValueError(f"Element {el} in domain pattern is not in composition.")

    counts = [composition[el] for el in elems]
    if len(set(counts)) != 1:
        raise ValueError(
            "domain mode currently requires equal counts for the four elements, "
            "e.g. Fe:8,Co:8,Ni:8,Cu:8."
        )
    return [(el, composition[el]) for el in elems]


def unique_domain_orders(composition, domain_pattern=None):
    """
    Return domain orders for 2x2 top-view domain templates.

    If domain_pattern is provided, only that specific template is returned.
    If domain_pattern is omitted, all 4! possible element arrangements over
    TL/TR/BL/BR are generated. Symmetry-equivalent structures are later removed
    by canonical_decoration_key(), so the user does not need to manually specify
    patterns such as 'Co,Fe/Ni,Cu'.
    """
    if domain_pattern is not None and str(domain_pattern).strip() != "":
        return [parse_domain_pattern(domain_pattern, composition)]

    elems = list(composition.keys())
    if len(elems) != 4:
        raise ValueError(
            "Automatic domain mode requires exactly four elements in composition, "
            "e.g. Fe:8,Co:8,Ni:8,Cu:8."
        )
    if len(set(composition.values())) != 1:
        raise ValueError(
            "Automatic domain mode requires equal counts for all four elements, "
            "e.g. Fe:8,Co:8,Ni:8,Cu:8."
        )

    orders = []
    seen = set()
    for perm in itertools.permutations(elems, 4):
        if perm in seen:
            continue
        seen.add(perm)
        orders.append(validate_domain_elements(list(perm), composition))
    return orders


def validate_quincunx_elements(elems, composition):
    """
    Validate a 5-element quincunx domain order: center + TL/TR/BL/BR corners.

    The four outer (corner) elements must share an equal count so the outer
    ring can be split into four equal-count quadrants; the center element's
    count may differ from the corners.
    """
    if len(elems) != 5:
        raise ValueError("quincunx domain mode requires exactly five domain elements.")
    if len(set(elems)) != 5:
        raise ValueError("quincunx domain mode requires five different elements.")
    for el in elems:
        if el not in composition:
            raise ValueError(f"Element {el} in quincunx pattern is not in composition.")

    center_el = elems[0]
    outer_els = elems[1:]
    outer_counts = [composition[el] for el in outer_els]
    if len(set(outer_counts)) != 1:
        raise ValueError(
            "quincunx domain mode requires the four outer (corner) elements to have "
            "equal counts; the center element's count may differ from them. "
            f"got outer counts {dict(zip(outer_els, outer_counts))} for center={center_el}."
        )
    return [(el, composition[el]) for el in elems]


def parse_quincunx_pattern(pattern, composition):
    """
    Parse one human-readable quincunx (center + 2x2 top-view corners) pattern.

    Example
    -------
    pattern = "Cu:Co,Fe/Ni,Ti"

    means, for view_axis='z':
        center       = Cu
        top-left     = Co
        top-right    = Fe
        bottom-left  = Ni
        bottom-right = Ti
    """
    if pattern is None or str(pattern).strip() == "":
        raise ValueError("No quincunx pattern was provided.")

    raw = str(pattern).strip()
    if ":" not in raw:
        raise ValueError(
            "quincunx pattern must have the form 'CenterElement:TL,TR/BL,BR', "
            "e.g. 'Cu:Co,Fe/Ni,Ti'."
        )
    center_part, corners_part = raw.split(":", 1)
    center_el = center_part.strip()
    if not center_el:
        raise ValueError("quincunx pattern is missing the center element before ':'.")

    corner_order = parse_domain_pattern(corners_part, composition)
    corner_elems = [el for el, _ in corner_order]
    return validate_quincunx_elements([center_el] + corner_elems, composition)


def unique_quincunx_orders(composition, domain_pattern=None):
    """
    Return domain orders for the 5-element quincunx (center + 2x2 corners) template.

    If domain_pattern is provided (format 'CenterElement:TL,TR/BL,BR'), only that
    specific template is returned. If omitted, every element is tried as the
    center; only center choices whose remaining four elements share an equal
    count are valid, combined with all 4! corner arrangements. Symmetry-
    equivalent results are later removed by canonical_decoration_key().
    """
    if domain_pattern is not None and str(domain_pattern).strip() != "":
        return [parse_quincunx_pattern(domain_pattern, composition)]

    elems = list(composition.keys())
    if len(elems) != 5:
        raise ValueError(
            "Automatic quincunx domain mode requires exactly five elements in "
            "composition, e.g. Cu:4,Co:4,Fe:4,Ni:4,Ti:4."
        )

    orders = []
    seen = set()
    for center_el in elems:
        outer_els = [el for el in elems if el != center_el]
        outer_counts = [composition[el] for el in outer_els]
        if len(set(outer_counts)) != 1:
            # This center choice is incompatible: the remaining four elements
            # do not share an equal count, so it is simply not offered.
            continue
        for perm in itertools.permutations(outer_els, 4):
            key = (center_el, perm)
            if key in seen:
                continue
            seen.add(key)
            orders.append(validate_quincunx_elements([center_el] + list(perm), composition))

    if not orders:
        raise ValueError(
            "No valid quincunx center/corner assignment was found. The four "
            "outer (corner) elements must share an equal count for at least one "
            "choice of center element, e.g. counts Cu:2, Co:4, Fe:4, Ni:4, Ti:4 "
            "(Cu as center)."
        )
    return orders


def _view_axis_to_plane_indices(view_axis):
    """For a top view along view_axis, return (horizontal_axis, vertical_axis)."""
    axis = str(view_axis).lower().strip()
    if axis in ("z", "c", "2"):
        return 0, 1   # x-y plane
    if axis in ("y", "b", "1"):
        return 0, 2   # x-z plane
    if axis in ("x", "a", "0"):
        return 1, 2   # y-z plane
    raise ValueError("view_axis must be x, y, or z.")


def _split_quadrants(sites, frac, h_axis, v_axis, tl_n, tr_n, bl_n, br_n):
    """
    Rank-based 2x2 (top-view) split of `sites` into exact-count TL/TR/BL/BR groups.

    Shared by the 4-element rectangular domain template and the 5-element
    quincunx template (applied there to the outer ring after the center
    cluster has been carved out).

    1) sort sites by vertical coordinate to create top and bottom groups;
    2) inside each group, sort by horizontal coordinate to create left/right groups.

    This guarantees the requested atom counts even when fractional coordinates are
    not exactly separated by 0.5, and raises if a cut would split atoms that share
    the same coordinate plane (an ambiguous, non-crystallographic boundary).
    """
    sites = list(sites)
    top_n = tl_n + tr_n
    bottom_n = bl_n + br_n
    if top_n + bottom_n != len(sites):
        raise ValueError(
            "Quadrant split counts do not match the number of sites: "
            f"tl+tr+bl+br={top_n + bottom_n}, sites={len(sites)}."
        )

    # Top has larger vertical coordinate. Bottom has smaller vertical coordinate.
    sites_by_v_desc = sorted(sites, key=lambda i: frac[i, v_axis], reverse=True)
    _validate_rank_cuts(
        sites_by_v_desc, [top_n], frac[:, v_axis],
        "top/bottom domain boundary"
    )
    top_sites = sites_by_v_desc[:top_n]
    bottom_sites = sites_by_v_desc[top_n:top_n + bottom_n]

    # Left has smaller horizontal coordinate. Right has larger horizontal coordinate.
    top_by_h = sorted(top_sites, key=lambda i: frac[i, h_axis])
    bottom_by_h = sorted(bottom_sites, key=lambda i: frac[i, h_axis])
    _validate_rank_cuts(top_by_h, [tl_n], frac[:, h_axis], "top left/right domain boundary")
    _validate_rank_cuts(bottom_by_h, [bl_n], frac[:, h_axis], "bottom left/right domain boundary")

    tl_sites = top_by_h[:tl_n]
    tr_sites = top_by_h[tl_n:tl_n + tr_n]
    bl_sites = bottom_by_h[:bl_n]
    br_sites = bottom_by_h[bl_n:bl_n + br_n]
    return tl_sites, tr_sites, bl_sites, br_sites


def make_domain_template_configuration(parent, replace_sites, domain_order, view_axis="z"):
    """
    Create an intuitive 2x2 domain structure in top view.

    For view_axis='z', the structure is divided in the x-y plane as:

        top-left      top-right
        bottom-left   bottom-right

    Example domain_order from pattern 'Co,Fe/Ni,Cu':
        top-left     = Co
        top-right    = Fe
        bottom-left  = Ni
        bottom-right = Cu

    See _split_quadrants() for the rank-based splitting method used here.
    """
    if len(domain_order) != 4:
        raise ValueError("domain_order must contain four domains: TL, TR, BL, BR.")

    total_needed = sum(n for _, n in domain_order)
    sites = list(replace_sites)
    if total_needed != len(sites):
        raise ValueError(
            "domain mode requires full replacement of the selected sublattice: "
            f"composition_sum={total_needed}, replacement_sites={len(sites)}."
        )

    h_axis, v_axis = _view_axis_to_plane_indices(view_axis)
    frac = parent.get_scaled_positions()

    tl_el, tl_n = domain_order[0]
    tr_el, tr_n = domain_order[1]
    bl_el, bl_n = domain_order[2]
    br_el, br_n = domain_order[3]

    tl_sites, tr_sites, bl_sites, br_sites = _split_quadrants(
        sites, frac, h_axis, v_axis, tl_n, tr_n, bl_n, br_n
    )

    atoms = parent.copy()
    for idx in tl_sites:
        atoms[idx].symbol = tl_el
    for idx in tr_sites:
        atoms[idx].symbol = tr_el
    for idx in bl_sites:
        atoms[idx].symbol = bl_el
    for idx in br_sites:
        atoms[idx].symbol = br_el

    chosen = tl_sites + tr_sites + bl_sites + br_sites
    return atoms, chosen


def make_quincunx_configuration(parent, replace_sites, domain_order, view_axis="z"):
    r"""
    Create a 5-element quincunx (center + 2x2 corners) domain structure in top view.

    For view_axis='z', the structure is divided in the x-y plane as:

        top-left      top-right
                  \  /
                 center
                  /  \
        bottom-left  bottom-right

    domain_order is [(center_el, n), (tl_el, n), (tr_el, n), (bl_el, n), (br_el, n)],
    e.g. from parse_quincunx_pattern('Cu:Co,Fe/Ni,Ti', composition).

    The center cluster is the `center_n` sites ranked closest to the centroid
    of the replacement sublattice (rank-based, not a fixed geometric radius,
    so exact counts are guaranteed regardless of spacing). The remaining outer
    ring is then split into TL/TR/BL/BR with the same rank-based method used
    by the rectangular 2x2 domain template (see _split_quadrants()).
    """
    if len(domain_order) != 5:
        raise ValueError("domain_order must contain five domains: center, TL, TR, BL, BR.")

    total_needed = sum(n for _, n in domain_order)
    sites = list(replace_sites)
    if total_needed != len(sites):
        raise ValueError(
            "quincunx domain mode requires full replacement of the selected sublattice: "
            f"composition_sum={total_needed}, replacement_sites={len(sites)}."
        )

    center_el, center_n = domain_order[0]
    tl_el, tl_n = domain_order[1]
    tr_el, tr_n = domain_order[2]
    bl_el, bl_n = domain_order[3]
    br_el, br_n = domain_order[4]

    h_axis, v_axis = _view_axis_to_plane_indices(view_axis)
    frac = parent.get_scaled_positions()

    h_mean = float(np.mean(frac[sites, h_axis]))
    v_mean = float(np.mean(frac[sites, v_axis]))
    radial2_full = np.full(len(frac), np.nan)
    for i in sites:
        radial2_full[i] = (frac[i, h_axis] - h_mean) ** 2 + (frac[i, v_axis] - v_mean) ** 2

    # Center has the smallest distance from the centroid. Outer ring is everything else.
    sites_by_radius = sorted(sites, key=lambda i: radial2_full[i])
    _validate_rank_cuts(
        sites_by_radius, [center_n], radial2_full,
        "center/outer quincunx boundary"
    )
    center_sites = sites_by_radius[:center_n]
    outer_sites = sites_by_radius[center_n:]

    tl_sites, tr_sites, bl_sites, br_sites = _split_quadrants(
        outer_sites, frac, h_axis, v_axis, tl_n, tr_n, bl_n, br_n
    )

    atoms = parent.copy()
    for idx in center_sites:
        atoms[idx].symbol = center_el
    for idx in tl_sites:
        atoms[idx].symbol = tl_el
    for idx in tr_sites:
        atoms[idx].symbol = tr_el
    for idx in bl_sites:
        atoms[idx].symbol = bl_el
    for idx in br_sites:
        atoms[idx].symbol = br_el

    chosen = center_sites + tl_sites + tr_sites + bl_sites + br_sites
    return atoms, chosen


def _axis_to_index(layer_axis):
    """Convert layer axis label to fractional-coordinate index."""
    axis = str(layer_axis).lower().strip()
    if axis in ("x", "a", "0"):
        return 0
    if axis in ("y", "b", "1"):
        return 1
    if axis in ("z", "c", "2"):
        return 2
    raise ValueError("layer_axis must be x, y, or z.")


def _validate_rank_cuts(ordered_sites, cut_positions, coordinates, label, tol=1e-7):
    """Reject a rank cut that splits atoms sharing the same coordinate plane."""
    for cut in cut_positions:
        if cut <= 0 or cut >= len(ordered_sites):
            continue
        left = coordinates[ordered_sites[cut - 1]]
        right = coordinates[ordered_sites[cut]]
        if abs(float(right) - float(left)) <= tol:
            raise ValueError(
                f"{label} splits one coordinate plane at fractional coordinate "
                f"~{0.5 * (left + right):.8f}. Choose compatible composition counts "
                "or a different axis."
            )


def unique_layer_orders(composition):
    """
    Return all unique layer orders as tuples of (element, count).

    Example: {'Fe':8,'Co':8,'Ni':8,'Cu':8} -> 24 possible layer orders.
    Each order is interpreted as consecutive layers along layer_axis.
    """
    items = tuple(composition.items())
    seen = set()
    orders = []
    for order in itertools.permutations(items, len(items)):
        if order in seen:
            continue
        seen.add(order)
        orders.append(order)
    return orders


def make_layered_configuration(parent, replace_sites, layer_order, layer_axis="z"):
    """
    Create an ordered layered parent structure.

    The replace sites are sorted by fractional coordinate along layer_axis and
    then divided into consecutive chunks. Each chunk is filled with one element.

    This is designed for cases such as Pt32 -> Fe8Co8Ni8Cu8, where 32 Pt sites
    can be split into four 8-atom layers:
        layer 1 = Fe, layer 2 = Co, layer 3 = Ni, layer 4 = Cu

    Notes
    -----
    - composition sum must be equal to the number of replacement sites.
    - For slab/bulk cells where a different layer direction is desired, set
      layer_axis to x, y, or z.
    """
    axis_i = _axis_to_index(layer_axis)
    frac = parent.get_scaled_positions()
    ordered_sites = sorted(list(replace_sites), key=lambda i: frac[i, axis_i])

    required = sum(n for _, n in layer_order)
    if required != len(ordered_sites):
        raise ValueError(
            "Layered mode currently requires composition sum to equal the number "
            f"of replacement sites. composition_sum={required}, sites={len(ordered_sites)}"
        )

    cuts = np.cumsum([n for _, n in layer_order])[:-1]
    _validate_rank_cuts(ordered_sites, cuts, frac[:, axis_i], "layer boundary")

    atoms = parent.copy()
    cursor = 0
    for el, count in layer_order:
        layer_sites = ordered_sites[cursor:cursor + count]
        for idx in layer_sites:
            atoms[idx].symbol = el
        cursor += count
    return atoms, ordered_sites

_SYMMETRY_SITE_MAP_CACHE = {}
_SYMMETRY_RANDOM_BASELINE_CACHE = {}
Q_RANDOM_BASELINE_SAMPLES = 512
Q_DEFINITION_VERSION = "symmetry_random_calibrated_v2"
LOW_Q_MIN_SEARCH_STEPS = 12_000
LOW_Q_FALLBACK_SEEDS = (105, 54321, 271828, 314159, 161803)


def _symmetry_site_maps(perms, sites):
    """Return cached symmetry mappings restricted to a preserved variable sublattice."""
    sites_tuple = tuple(int(i) for i in sites)
    cache_key = (id(perms), sites_tuple)
    cached = _SYMMETRY_SITE_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    site_set = set(sites_tuple)
    mapped = []
    for perm in perms:
        row = np.asarray(perm, dtype=np.int32)[list(sites_tuple)]
        if set(row.tolist()) == site_set:
            mapped.append(row)
    if not mapped:
        raise ValueError("No symmetry permutation preserves variable_sites.")
    result = np.vstack(mapped)
    _SYMMETRY_SITE_MAP_CACHE[cache_key] = result
    return result


def symmetry_random_match_baseline(
    reference,
    variable_sites,
    perms,
    samples=Q_RANDOM_BASELINE_SAMPLES,
):
    """
    Estimate the random expectation of the symmetry-maximized site match.

    Taking the maximum match over many symmetry operations raises the random
    expectation above sum(x_i**2). A deterministic Monte Carlo calibration is
    therefore required for a symmetry-invariant Q whose random mean is near 0.
    """
    sites = sorted(int(site) for site in variable_sites)
    if not sites:
        raise ValueError("variable_sites must not be empty")
    if perms is None:
        fractions = np.asarray(
            list(Counter(reference[i].symbol for i in sites).values()),
            dtype=float,
        ) / len(sites)
        return float(np.sum(fractions ** 2))

    sample_count = max(1, int(samples))
    reference_signature = tuple(reference[i].symbol for i in sites)
    cache_key = (id(perms), tuple(sites), reference_signature, sample_count)
    cached = _SYMMETRY_RANDOM_BASELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mapped_sites = _symmetry_site_maps(perms, sites)
    site_to_local = {site: position for position, site in enumerate(sites)}
    mapped_local = np.asarray(
        [[site_to_local[int(site)] for site in row] for row in mapped_sites],
        dtype=np.int32,
    )
    reference_selected = np.asarray(
        [reference[i].symbol for i in sites],
        dtype="U3",
    )

    # A local fixed seed makes Q reproducible and does not alter structure-
    # generation randomness controlled by the user seed.
    rng = np.random.default_rng(20260724)
    maxima = np.empty(sample_count, dtype=float)
    for sample in range(sample_count):
        shuffled = rng.permutation(reference_selected)
        maxima[sample] = np.max(
            np.sum(shuffled[mapped_local] == reference_selected, axis=1)
        ) / len(sites)

    baseline = float(np.mean(maxima))
    _SYMMETRY_RANDOM_BASELINE_CACHE[cache_key] = baseline
    return baseline


def parent_overlap_order_parameter(reference, candidate, variable_sites, perms=None):
    """
    Composition-corrected overlap with an ordered reference decoration.

    Q = (p_match - p_random,sym) / (1 - p_random,sym)

    p_match is the best site-match fraction over parent symmetry operations.
    p_random,sym is the deterministic Monte Carlo expectation of that same
    symmetry-maximized statistic for random permutations of the composition.
    Thus Q=1 for the reference and random structures have mean Q~0. Q can be
    negative for configurations less correlated than the random expectation.
    This is an order parameter, not thermodynamic entropy.
    """
    sites = sorted(int(site) for site in variable_sites)
    if not sites:
        raise ValueError("variable_sites must not be empty")

    ref_symbols = [reference[i].symbol for i in sites]
    cand_symbols = [candidate[i].symbol for i in sites]
    if Counter(ref_symbols) != Counter(cand_symbols):
        raise ValueError("Reference and candidate compositions differ on variable_sites.")

    n_sites = len(sites)
    if perms is None:
        fractions = np.asarray(list(Counter(ref_symbols).values()), dtype=float) / n_sites
        random_match = float(np.sum(fractions ** 2))
        observed_match = sum(a == b for a, b in zip(ref_symbols, cand_symbols)) / n_sites
    else:
        random_match = symmetry_random_match_baseline(
            reference,
            sites,
            perms,
        )
        mapped_sites = _symmetry_site_maps(perms, sites)
        if hasattr(candidate, "get_chemical_symbols"):
            candidate_all = np.asarray(candidate.get_chemical_symbols(), dtype="U3")
        else:
            max_index = int(np.max(mapped_sites))
            candidate_all = np.asarray(
                [candidate[i].symbol for i in range(max_index + 1)], dtype="U3"
            )
        reference_selected = np.asarray(ref_symbols, dtype="U3")
        best_matches = int(np.max(np.sum(candidate_all[mapped_sites] == reference_selected, axis=1)))
        observed_match = best_matches / n_sites
    if np.isclose(random_match, 1.0):
        return 1.0
    return float((observed_match - random_match) / (1.0 - random_match))


def build_neighbor_map(atoms, variable_sites, cutoff_factor=1.20):
    """Build a first-neighbor map on the variable sublattice using MIC distances."""
    sites = list(variable_sites)
    if len(sites) < 2:
        raise ValueError("At least two variable sites are required for SRO.")
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    positive = [
        distances[i, j]
        for a, i in enumerate(sites)
        for j in sites[a + 1:]
        if distances[i, j] > 1e-10
    ]
    if not positive:
        raise ValueError("Could not determine a nonzero neighbor distance.")
    nearest = min(positive)
    cutoff = nearest * float(cutoff_factor)

    def _map_at(radius):
        return {
            i: [j for j in sites if j != i and distances[i, j] <= radius]
            for i in sites
        }

    neighbor_map = _map_at(cutoff)
    # A site with no neighbor inside the first shell is normal whenever the
    # substitution pool is not a single connected sublattice -- e.g. an
    # adsorbate species included in the pool sits far above the slab. That used
    # to abort the whole run with a traceback. Grow the shell a few times
    # instead, and if a site is still isolated just leave its neighbor list
    # empty (the Warren-Cowley average simply skips it) and say so once.
    if any(not neighbors for neighbors in neighbor_map.values()):
        grown = cutoff
        for _ in range(6):
            grown *= 1.25
            candidate_map = _map_at(grown)
            if all(candidate_map.values()):
                print("[notice] SRO first shell widened to %.6f Angstrom so every "
                      "variable site has a neighbor (cutoff_factor %.2f -> %.2f)."
                      % (grown, float(cutoff_factor), grown / nearest))
                return candidate_map, nearest, grown
        isolated = [i + 1 for i, neighbors in neighbor_map.items() if not neighbors]
        if len(isolated) == len(sites):
            raise ValueError(
                "No variable site has a first-shell neighbor within %.6f Angstrom. "
                "The substitution pool does not form a connected sublattice - check "
                "-re= (or raise -sro_cutoff=)." % cutoff
            )
        preview = ", ".join("#%d" % n for n in isolated[:10])
        if len(isolated) > 10:
            preview += ", ... (%d atoms)" % len(isolated)
        print("[notice] %d variable site(s) have no first-shell neighbor within "
              "%.6f Angstrom and are skipped in the SRO average: %s"
              % (len(isolated), cutoff, preview))
        print("         They are usually pool atoms that sit off the main "
              "sublattice (an adsorbate left in -re=). Raise -sro_cutoff= to "
              "include them.")
    return neighbor_map, nearest, cutoff


def warren_cowley_sro_pairs(atoms, variable_sites, neighbor_map):
    """Return the first-shell Warren-Cowley alpha value for every ordered pair."""
    sites = list(variable_sites)
    symbols = [atoms[i].symbol for i in sites]
    counts = Counter(symbols)
    total = len(sites)
    fractions = {element: count / total for element, count in counts.items()}

    pair_values = {}
    for center_element, center_fraction in fractions.items():
        centers = [i for i in sites if atoms[i].symbol == center_element]
        neighbor_symbols = [
            atoms[j].symbol
            for i in centers
            for j in neighbor_map[i]
        ]
        if not neighbor_symbols:
            continue
        neighbor_counts = Counter(neighbor_symbols)
        denominator = len(neighbor_symbols)
        for neighbor_element, neighbor_fraction in fractions.items():
            probability = neighbor_counts.get(neighbor_element, 0) / denominator
            alpha = 1.0 - probability / neighbor_fraction
            pair_values[f"{center_element}-{neighbor_element}"] = float(alpha)
    return pair_values


def warren_cowley_sro_rms(atoms, variable_sites, neighbor_map):
    """Return composition-weighted RMS Warren-Cowley SRO over the first shell."""
    sites = list(variable_sites)
    symbols = [atoms[i].symbol for i in sites]
    counts = Counter(symbols)
    total = len(sites)
    fractions = {element: count / total for element, count in counts.items()}
    pair_values = warren_cowley_sro_pairs(atoms, sites, neighbor_map)

    squared = 0.0
    total_weight = 0.0
    for center_element, center_fraction in fractions.items():
        for neighbor_element, neighbor_fraction in fractions.items():
            alpha = pair_values[f"{center_element}-{neighbor_element}"]
            weight = center_fraction * neighbor_fraction
            squared += weight * alpha * alpha
            total_weight += weight
    return float(np.sqrt(squared / total_weight)) if total_weight else 0.0


def _serialized_sro_pairs(atoms, variable_sites, neighbor_map):
    """Return stable compact JSON for storage in structures.csv."""
    if neighbor_map is None:
        return ""
    return json.dumps(
        warren_cowley_sro_pairs(atoms, variable_sites, neighbor_map),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def generate_to_order_target(
    reference,
    variable_sites,
    target_q,
    tolerance=0.05,
    search_steps=5000,
    restarts=8,
    perms=None,
    neighbor_map=None,
    sro_tolerance=0.12,
    sro_weight=0.5,
    initial_candidate=None,
    rng=None,
):
    """Search for a same-composition decoration near target Q with low first-shell SRO."""
    target_q = float(target_q)
    rng = random if rng is None else rng
    tolerance = float(tolerance)
    if not 0.0 <= target_q <= 1.0:
        raise ValueError(f"target_q must be between 0 and 1, got {target_q}")
    if tolerance <= 0:
        raise ValueError("order tolerance must be positive")

    sites = list(variable_sites)
    symbols = [reference[i].symbol for i in sites]
    if len(sites) < 2 or len(set(symbols)) < 2:
        if np.isclose(target_q, 1.0):
            reference_sro = (
                warren_cowley_sro_rms(reference, sites, neighbor_map)
                if neighbor_map is not None else None
            )
            return reference.copy(), 1.0, reference_sro
        raise ValueError("At least two element types are required for Q < 1.")

    if np.isclose(target_q, 1.0):
        reference_sro = (
            warren_cowley_sro_rms(reference, sites, neighbor_map)
            if neighbor_map is not None else None
        )
        return reference.copy(), 1.0, reference_sro

    reference_sro = (
        warren_cowley_sro_rms(reference, sites, neighbor_map)
        if neighbor_map is not None else None
    )
    target_sro = target_q * reference_sro if reference_sro is not None else None

    def evaluate(atoms):
        q_value = parent_overlap_order_parameter(reference, atoms, sites, perms=perms)
        sro_value = (
            warren_cowley_sro_rms(atoms, sites, neighbor_map)
            if neighbor_map is not None else None
        )
        q_error = abs(q_value - target_q) / tolerance
        sro_excess = 0.0
        if sro_value is not None:
            sro_excess = max(0.0, sro_value - target_sro) / sro_tolerance
        return q_value, sro_value, q_error + float(sro_weight) * sro_excess

    best_atoms = initial_candidate.copy() if initial_candidate is not None else reference.copy()
    best_q, best_sro, best_score = evaluate(best_atoms)

    # Very low Q is a max-correlation minimization problem. One long annealing
    # trajectory is markedly more reliable than many short restarts.
    low_q_search = target_q <= max(float(tolerance), 0.10)
    if low_q_search:
        n_restarts = 1
        effective_search_steps = max(int(search_steps), LOW_Q_MIN_SEARCH_STEPS)
    else:
        n_restarts = max(1, int(restarts))
        effective_search_steps = int(search_steps)
    steps_per_restart = max(1, effective_search_steps // n_restarts)

    for restart in range(n_restarts):
        if restart == 0 and initial_candidate is not None:
            current = initial_candidate.copy()
        else:
            current = reference.copy()
        if not (restart == 0 and initial_candidate is not None) and (target_q < 0.5 or restart > 0):
            shuffled = symbols[:]
            rng.shuffle(shuffled)
            for site, symbol in zip(sites, shuffled):
                current[site].symbol = symbol

        current_q, current_sro, current_score = evaluate(current)
        if current_score < best_score:
            best_atoms, best_q, best_sro, best_score = (
                current.copy(), current_q, current_sro, current_score
            )
        q_ok = abs(best_q - target_q) <= tolerance
        sro_ok = best_sro is None or best_sro <= target_sro + sro_tolerance
        if q_ok and sro_ok:
            return best_atoms, best_q, best_sro

        for step in range(steps_per_restart):
            i, j = rng.sample(sites, 2)
            if current[i].symbol == current[j].symbol:
                continue

            current[i].symbol, current[j].symbol = current[j].symbol, current[i].symbol
            proposed_q, proposed_sro, proposed_score = evaluate(current)

            # Simulated annealing: accept uphill moves early, then gradually
            # become greedy. This is substantially more robust for domain Q=0
            # than a fixed small exploration probability.
            progress = step / max(1, steps_per_restart - 1)
            temperature = max(0.03, 1.5 * (1.0 - progress) ** 2)
            score_delta = proposed_score - current_score
            accept_uphill = (
                score_delta > 0
                and rng.random() < exp(-score_delta / temperature)
            )
            if score_delta <= 0 or accept_uphill:
                current_q, current_sro, current_score = proposed_q, proposed_sro, proposed_score
            else:
                current[i].symbol, current[j].symbol = current[j].symbol, current[i].symbol

            if current_score < best_score:
                best_atoms, best_q, best_sro, best_score = (
                    current.copy(), current_q, current_sro, current_score
                )
                q_ok = abs(best_q - target_q) <= tolerance
                sro_ok = best_sro is None or best_sro <= target_sro + sro_tolerance
                if q_ok and sro_ok:
                    return best_atoms, best_q, best_sro

    # Deterministic best-improvement polishing of the best annealed candidate.
    # This removes residual seed sensitivity near the discrete Q/SRO boundary.
    polished = best_atoms.copy()
    polished_q, polished_sro, polished_score = evaluate(polished)
    polish_rounds = max(2, min(8, effective_search_steps // 1000))
    for _ in range(polish_rounds):
        best_move = None
        move_state = None
        move_score = polished_score
        for position, i in enumerate(sites):
            for j in sites[position + 1:]:
                if polished[i].symbol == polished[j].symbol:
                    continue
                polished[i].symbol, polished[j].symbol = polished[j].symbol, polished[i].symbol
                candidate_q, candidate_sro, candidate_score = evaluate(polished)
                polished[i].symbol, polished[j].symbol = polished[j].symbol, polished[i].symbol
                if candidate_score < move_score - 1e-12:
                    best_move = (i, j)
                    move_state = (candidate_q, candidate_sro)
                    move_score = candidate_score
        if best_move is None:
            break
        i, j = best_move
        polished[i].symbol, polished[j].symbol = polished[j].symbol, polished[i].symbol
        polished_q, polished_sro = move_state
        polished_score = move_score
        if polished_score < best_score:
            best_atoms, best_q, best_sro, best_score = (
                polished.copy(), polished_q, polished_sro, polished_score
            )
        q_ok = abs(best_q - target_q) <= tolerance
        sro_ok = best_sro is None or best_sro <= target_sro + sro_tolerance
        if q_ok and sro_ok:
            return best_atoms, best_q, best_sro

    return best_atoms, best_q, best_sro


def generate_multiset_permutations(elements):
    """Yield each multiset permutation directly, without traversing duplicate N! permutations."""
    counts = Counter(elements)
    keys = list(counts)
    length = len(elements)
    current = [None] * length

    def visit(position):
        if position == length:
            yield tuple(current)
            return
        for element in keys:
            if counts[element] <= 0:
                continue
            counts[element] -= 1
            current[position] = element
            yield from visit(position + 1)
            counts[element] += 1

    yield from visit(0)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

STRUCTURE_MANIFEST_FIELDS = [
    "structure_id", "path", "parent_id", "mode", "axis", "pattern",
    "target_q", "actual_q", "q_random_match", "q_definition",
    "sro_rms", "sro_pairs", "composition", "seed", "search_seed",
]


def _append_structure_manifest(out_dir, record):
    path = os.path.join(out_dir, "structures.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in STRUCTURE_MANIFEST_FIELDS})
    return path


def write_structure(
    atoms,
    out_dir,
    index,
    output_format="cif",
    vasp_folder=False,
    metadata=None,
    potcar_source=None,
):
    """
    Write a short-ID structure and append its scientific metadata to structures.csv.

    potcar_source, if given (only meaningful when vasp_folder=True), is the
    path to a pre-built combined POTCAR file that gets hard-linked (or copied,
    as a fallback) alongside the POSCAR in this structure's folder. It is
    identical for every structure of a run sharing the same composition, so
    it is built once by the caller rather than regenerated per structure.
    """
    structure_id = f"S{index:06d}"
    if vasp_folder:
        folder = os.path.join(out_dir, structure_id)
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, "POSCAR")
        write(output_path, atoms, format="vasp", direct=True, sort=True, vasp5=True)
        relative_path = os.path.join(structure_id, "POSCAR")
        if potcar_source:
            _hardlink_or_copy(potcar_source, os.path.join(folder, "POTCAR"))
    elif output_format.lower() in ("vasp", "poscar"):
        output_path = os.path.join(out_dir, f"{structure_id}.vasp")
        write(output_path, atoms, format="vasp", direct=True, sort=True, vasp5=True)
        relative_path = f"{structure_id}.vasp"
    else:
        output_path = os.path.join(out_dir, f"{structure_id}.cif")
        write(output_path, atoms)
        relative_path = f"{structure_id}.cif"

    record = dict(metadata or {})
    record.update({"structure_id": structure_id, "path": relative_path})
    _append_structure_manifest(out_dir, record)
    return output_path


def _strip_adsorbate_atoms(atoms, adsorbate_elements):
    """Return a copy of atoms with every atom whose symbol is in adsorbate_elements removed."""
    keep = [
        i for i, s in enumerate(atoms.get_chemical_symbols())
        if s not in adsorbate_elements
    ]
    return atoms[keep]


def parse_redox_spec(parent, spec, replace_sites=()):
    """
    Parse a redox-removal spec into the concrete sets of atoms to delete.

    Atom numbers are 1-based, matching the input structure file's coordinate
    order (which the generator preserves in every output structure).

    Accepted forms
    --------------
    Removal composition -- "how much of each species to take away". This is the
    natural way to write a redox step, because it fixes the stoichiometry of
    what is removed. **The count is mandatory**:

        'S1'              remove any 1 S atom              Li2S4 -> Li2S3
        'S2'              remove any 2 S atoms             Li2S4 -> Li2S2
        'Li2,S1'          remove any 2 Li and any 1 S
        'Li:2,S:1'        same thing
        'Li1,S1'          remove any 1 Li and any 1 S

      Every distinct choice is a different structure, so each gets its own
      output folder (_r1, _r2, ...). The number of folders is the product of
      C(n_el, k_el) over the listed elements.

      A bare element symbol ('S') is rejected here with a message pointing at
      both ways out. That is deliberate: --adsorbate-elements (-surface) takes
      a *list of species to strip entirely*, so if a bare element silently
      meant "one atom" here, the two options would accept identical-looking
      values and do opposite things. To strip a species entirely, use
      -surface=S; to remove some of it, write the count ('S1', 'S2').

    Explicit atom numbers -- named atoms, always removed, a single set ('_r'):

        '35'              remove exactly atom 35
        '35,36'           remove exactly atoms 35 and 36
        'S1,35'           atom 35 always, plus any 1 S (mixing is allowed)

    Candidate pool with '/k' -- "any k of *these* atoms". Use it to restrict
    the candidate sites (e.g. only the top-facing S):

        '35,36,37,38/2'   any 2 of those 4 atoms           -> C(4,2)=6 sets
        'S/2'             any 2 of every S atom            -> C(n,2) sets
        'Li,S/2'          pool = every Li and S atom, any 2 (mixes species)

    Element symbols expand to every atom of that element that is *not* part of
    the replacement pool (removing a substituted site would break the 1:1
    arrangement mapping between the main and redox structures).

    Returns (index_sets, pool, choose) with 0-based indices:
      index_sets : list of tuples, one per output folder
      pool       : sorted candidate indices the sets were drawn from
      choose     : total number of atoms removed per set, or None when the
                   whole pool is removed as a single set
    """
    symbols = parent.get_chemical_symbols()
    n_atoms = len(parent)
    replace_set = set(replace_sites)

    def _check_number(value):
        if value < 1 or value > n_atoms:
            raise ValueError(
                f"redox 원자 번호가 범위(1..{n_atoms})를 벗어났습니다: {value}"
            )
        return value - 1

    def _atoms_of(element):
        matched = [
            i for i, sym in enumerate(symbols)
            if sym == element and i not in replace_set
        ]
        if not matched:
            raise ValueError(
                f"'{element}' 원소에 해당하는 (치환 풀이 아닌) 원자가 없습니다."
            )
        return matched

    choose = None
    if isinstance(spec, str):
        text = spec.strip()

        # ---- removal composition: atom numbers and/or element counts ----
        # Atom numbers name specific atoms and are always removed. Element
        # symbols carry a count, defaulting to 1 -- so 'S' means "one S", the
        # way a chemical formula reads it (Li2S has one S). To strip an
        # element entirely, use -surface=S (that is exactly what it is for) or
        # spell the count out ('S4').
        if "/" not in text:
            tokens = [t for t in re.split(r"[,\s]+", text) if t]
            if not tokens:
                raise ValueError(f"redox 지정이 비어 있습니다: {spec!r}")
            fixed = []                              # explicit atoms
            counts = {}                             # element -> how many to remove
            for token in tokens:
                if token.isdigit():
                    fixed.append(_check_number(int(token)))
                    continue
                match = re.fullmatch(r"([A-Z][a-z]?)\s*[:=]?\s*(\d*)", token)
                if not match:
                    raise ValueError(
                        f"redox 지정을 해석할 수 없습니다: {token!r}\n"
                        "  제거 조성 : 'S1', 'S2', 'Li2,S1'  (개수를 반드시 적습니다)\n"
                        "  특정 원자 : '35', '35,36'\n"
                        "  후보 중 k : '19,20,21,22/2'\n"
                        "  (해당 원소 전부를 없애려면 surface 쪽에 지정하세요)"
                    )
                element = match.group(1)
                if not match.group(2):
                    # A bare element symbol is refused on purpose. In -surface=
                    # it means "strip this species entirely"; if it silently
                    # meant "one atom" here, the two options would take
                    # identical-looking values and do opposite things. Making
                    # the count explicit removes that trap.
                    raise ValueError(
                        f"redox 에는 제거할 개수를 함께 적어 주세요: '{element}1'(1개), "
                        f"'{element}2'(2개) 처럼요.\n"
                        f"  '{element}' 원자를 전부 없애려면 surface 쪽에 지정하세요 "
                        f"(CLI: -surface={element}).\n"
                        "  특정 원자만 지정할 때는 번호를 쓰면 됩니다 (예: '35', '35,36')."
                    )
                counts[element] = counts.get(element, 0) + int(match.group(2))

            fixed = sorted(set(fixed))
            fixed_overlap = [i + 1 for i in fixed if i in replace_set]
            if fixed_overlap:
                raise ValueError(
                    "redox 대상이 치환 풀 자리입니다: 원자 "
                    f"{fixed_overlap}. 치환된 자리를 제거하면 메인 구조와 redox 구조의 "
                    "1:1 배열 대응이 깨지므로 허용하지 않습니다."
                )

            if not counts:
                # Only explicit atoms -> a single removal set (one '_r' folder)
                return [tuple(fixed)], list(fixed), None

            per_element_choices = []
            for element, count in counts.items():
                candidates = [i for i in _atoms_of(element) if i not in set(fixed)]
                if count > len(candidates):
                    raise ValueError(
                        f"{element} 를 {count}개 제거하라고 했지만 후보는 "
                        f"{len(candidates)}개뿐입니다 (원자 "
                        f"{[i + 1 for i in candidates]})."
                    )
                per_element_choices.append(
                    list(itertools.combinations(candidates, count))
                )
            # One folder per (choice of Li atoms) x (choice of S atoms) x ...
            index_sets = [
                tuple(sorted(list(fixed) + [i for group in combo for i in group]))
                for combo in itertools.product(*per_element_choices)
            ]
            pool = sorted({i for s_ in index_sets for i in s_})
            return index_sets, pool, len(fixed) + sum(counts.values())

        if "/" in text:
            pool_text, _, count_text = text.rpartition("/")
            count_text = count_text.strip()
            if not count_text.isdigit():
                raise ValueError(
                    f"redox 지정의 '/' 뒤에는 제거할 개수(숫자)가 와야 합니다: {spec!r} "
                    "(예: 'S/2', '19,20,21,22/2')"
                )
            choose = int(count_text)
            text = pool_text
        tokens = [t for t in text.replace(" ", "").split(",") if t]
        if not tokens:
            raise ValueError(f"redox 지정이 비어 있습니다: {spec!r}")
        pool = []
        for token in tokens:
            if token.isdigit():
                pool.append(_check_number(int(token)))
            elif re.fullmatch(r"[A-Z][a-z]?", token):
                pool.extend(_atoms_of(token))
            else:
                raise ValueError(
                    f"redox 지정을 해석할 수 없습니다: {token!r}\n"
                    "  원소별 개수 : 'S2', 'Li2,S1'      (제거 조성을 고정)\n"
                    "  원자/원소   : '35', '35,36', 'S'   (그것들만 제거)\n"
                    "  후보 중 k개 : '35,36,37,38/2'      (지정 원자 중 아무 k개)"
                )
    else:
        pool = [_check_number(int(v)) for v in spec]

    pool = sorted(set(pool))
    overlap = [i + 1 for i in pool if i in replace_set]
    if overlap:
        raise ValueError(
            "redox 대상이 치환 풀 자리입니다: 원자 "
            f"{overlap}. 치환된 자리를 제거하면 메인 구조와 redox 구조의 "
            "1:1 배열 대응이 깨지므로 허용하지 않습니다."
        )

    if choose is None:
        index_sets = [tuple(pool)]
    else:
        if choose < 1 or choose > len(pool):
            raise ValueError(
                f"제거 개수는 1..{len(pool)} 범위여야 합니다 (후보 {len(pool)}개, 요청 {choose}개)."
            )
        index_sets = [tuple(c) for c in itertools.combinations(pool, choose)]
    return index_sets, pool, choose


def write_structure_with_twins(
    atoms,
    output_dir,
    index,
    output_format="cif",
    vasp_folder=False,
    metadata=None,
    potcar_source=None,
    bare_output_dir=None,
    bare_potcar_source=None,
    adsorbate_elements=None,
    redox_output_dirs=None,
    redox_index_sets=None,
    redox_potcar_sources=None,
):
    """
    Write a structure, and optionally its twin variants under the same
    structure ID in mirrored output trees:

    - surface twin (bare_output_dir): the identical decoration with every
      adsorbate_elements atom removed. For adsorption-energy workflows,
      where E_ads = E(slab+ads) - E(surface) - E(ads reference) requires the
      adsorbate-free surface to have exactly the same site arrangement as
      its adsorbed counterpart (not an independently re-randomized one).

    - redox twins (redox_output_dirs): one twin per entry of redox_index_sets,
      each the identical decoration with only that set of atoms removed
      (0-based indices into `atoms`, which keeps the input-file atom order).
      With a single set this is the usual `_r` folder; with several it is
      `_r1`, `_r2`, ... -- e.g. every way of removing 2 of the 4 S atoms when
      comparing Li2S4 -> Li2S2. redox_potcar_sources, when given, is aligned
      with redox_output_dirs (each set can have a different composition).
    """
    output_path = write_structure(
        atoms, output_dir, index, output_format, vasp_folder,
        metadata=metadata, potcar_source=potcar_source,
    )
    if bare_output_dir is not None:
        bare_atoms = _strip_adsorbate_atoms(atoms, adsorbate_elements)
        write_structure(
            bare_atoms, bare_output_dir, index, output_format, vasp_folder,
            metadata=metadata, potcar_source=bare_potcar_source,
        )
    if redox_output_dirs and redox_index_sets:
        sources = redox_potcar_sources or [None] * len(redox_output_dirs)
        for redox_dir, removed_indices, potcar_src in zip(
            redox_output_dirs, redox_index_sets, sources
        ):
            removed = set(removed_indices)
            redox_atoms = atoms[[i for i in range(len(atoms)) if i not in removed]]
            write_structure(
                redox_atoms, redox_dir, index, output_format, vasp_folder,
                metadata=metadata, potcar_source=potcar_src,
            )
    return output_path


# -----------------------------------------------------------------------------
# POTCAR auto-generation (potpaw_PBE-style library)
# -----------------------------------------------------------------------------

# Default element -> POTCAR variant folder name, matching the "recommended"
# PAW-PBE potentials the lab has historically used (carried over from an
# older server's mapping). Only used when --generate-potcar is requested;
# any element can be overridden via --potcar-variants.
DEFAULT_POTCAR_VARIANTS = {
    "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As", "Au": "Au",
    "B": "B", "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi", "Br": "Br", "C": "C",
    "Ca": "Ca_sv", "Cd": "Cd", "Ce": "Ce", "Cl": "Cl", "Co": "Co", "Cr": "Cr_pv",
    "Cs": "Cs_sv", "Cu": "Cu_pv", "Dy": "Dy_3", "Er": "Er_3", "Eu": "Eu",
    "F": "F", "Fe": "Fe_pv", "Ga": "Ga_d", "Gd": "Gd", "Ge": "Ge_d", "H": "H",
    "He": "He", "Hf": "Hf_pv", "Hg": "Hg", "Ho": "Ho_3", "I": "I",
    "In": "In_d", "Ir": "Ir", "K": "K_sv", "Kr": "Kr", "La": "La",
    "Li": "Li_sv", "Lu": "Lu_3", "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv",
    "N": "N", "Na": "Na_pv", "Nb": "Nb_pv", "Nd": "Nd_3", "Ne": "Ne",
    "Ni": "Ni_pv", "Np": "Np", "O": "O", "Os": "Os_pv", "P": "P", "Pa": "Pa",
    "Pb": "Pb_d", "Pd": "Pd", "Pm": "Pm_3", "Pr": "Pr_3", "Pt": "Pt",
    "Pu": "Pu", "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv", "Ru": "Ru_pv",
    "S": "S", "Sb": "Sb", "Sc": "Sc_sv", "Se": "Se", "Si": "Si", "Sm": "Sm_3",
    "Sn": "Sn_d", "Sr": "Sr_sv", "Ta": "Ta_pv", "Tb": "Tb_3", "Tc": "Tc_pv",
    "Te": "Te", "Th": "Th", "Ti": "Ti_pv", "Tl": "Tl_d", "Tm": "Tm_3",
    "U": "U", "V": "V_pv", "W": "W_pv", "Xe": "Xe", "Y": "Y_sv",
    "Yb": "Yb_2", "Zn": "Zn", "Zr": "Zr_sv",
}

# Shared library location(s), potpaw_PBE.54 (PBE PAW set). Different lab
# servers have used different casing for the "potential"/"Potential" folder
# name over time; only one of these exists on any given machine, so the
# first one found is used automatically unless --potcar-library overrides it.
POTCAR_LIBRARY_CANDIDATES = (
    "/opt/vasp/potential/potpaw_PBE.54",
    "/opt/vasp/Potential/potpaw_PBE.54",
)


def resolve_potcar_library(explicit=None):
    """
    Return the POTCAR library path to use.

    If `explicit` is given, it is used as-is (and validated later by
    resolve_potcar_variant_map). Otherwise the first existing directory in
    POTCAR_LIBRARY_CANDIDATES is returned; if none exist, a clear error lists
    every candidate that was checked so the actual path can be diagnosed
    instead of failing on a generic "not found" for a guessed default.
    """
    if explicit:
        return explicit
    for candidate in POTCAR_LIBRARY_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not auto-detect a POTCAR library; none of these exist: "
        + ", ".join(POTCAR_LIBRARY_CANDIDATES)
        + ". Pass --potcar-library explicitly."
    )


def parse_potcar_variant_overrides(text):
    """
    Parse 'Fe:Fe_sv,Co:Co_pv' style overrides into a dict.

    Used to replace or add entries on top of DEFAULT_POTCAR_VARIANTS for
    specific elements, e.g. when the default recommended variant is not
    what a particular study needs.
    """
    if text is None or not str(text).strip():
        return {}
    overrides = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid --potcar-variants entry {token!r}; use 'Element:Variant', "
                "e.g. 'Fe:Fe_sv,Co:Co_pv'."
            )
        el, variant = token.split(":", 1)
        el, variant = el.strip(), variant.strip()
        if not el or not variant:
            raise ValueError(f"Invalid --potcar-variants entry {token!r}.")
        overrides[el] = variant
    return overrides


def resolve_potcar_variant_map(elements, potcar_library, overrides=None):
    """
    Resolve element -> POTCAR variant folder name for every element in `elements`.

    DEFAULT_POTCAR_VARIANTS is the base; `overrides` replaces or adds entries.
    Raises a clear error listing any elements with no known default and no
    override (nothing is silently guessed), and validates that the resolved
    variant folder actually contains a POTCAR file in potcar_library,
    suggesting the variants that do exist for that element on mismatch.
    """
    if not potcar_library or not os.path.isdir(potcar_library):
        raise FileNotFoundError(f"POTCAR library not found: {potcar_library}")

    overrides = dict(overrides or {})
    variant_map = {}
    missing = []
    for el in sorted(set(elements)):
        if el in overrides:
            variant_map[el] = overrides[el]
        elif el in DEFAULT_POTCAR_VARIANTS:
            variant_map[el] = DEFAULT_POTCAR_VARIANTS[el]
        else:
            missing.append(el)

    if missing:
        suggestion = ",".join(f"{el}:{el}" for el in missing)
        raise ValueError(
            "No default POTCAR variant is known for: " + ", ".join(missing) +
            ". Specify it explicitly, e.g. --potcar-variants '" + suggestion + "' "
            "(replace with the correct variant name for each)."
        )

    bad = []
    for el, variant in variant_map.items():
        potcar_path = os.path.join(potcar_library, variant, "POTCAR")
        if not os.path.isfile(potcar_path):
            available = sorted(
                d for d in os.listdir(potcar_library)
                if (d == el or d.startswith(el + "_"))
                and os.path.isfile(os.path.join(potcar_library, d, "POTCAR"))
            )
            bad.append((el, variant, available))
    if bad:
        lines_out = []
        for el, variant, available in bad:
            hint = (
                f" (available in library: {', '.join(available)})" if available
                else " (no variants found for this element in the library at all)"
            )
            lines_out.append(f"  {el} -> {variant}{hint}")
        raise FileNotFoundError(
            "Could not find a POTCAR for the following resolved variants in "
            f"{potcar_library}:\n" + "\n".join(lines_out)
        )
    return variant_map


def build_combined_potcar(species_order, potcar_library, variant_map):
    """Concatenate each element's POTCAR file content in the given species order."""
    chunks = []
    for el in species_order:
        variant = variant_map[el]
        potcar_path = os.path.join(potcar_library, variant, "POTCAR")
        with open(potcar_path, "r", encoding="utf-8", errors="ignore") as fh:
            chunks.append(fh.read())
    return "".join(chunks)


def _read_poscar_species_order(poscar_path):
    """
    Read the VASP5 species-symbol line (line 6) from a written POSCAR file.

    Used as a safety check: the predicted species order (sorted(set(symbols)),
    matching ASE's `sort=True` alphabetical np.argsort behavior) is verified
    against what was actually written, so a future ASE behavior change would
    raise loudly instead of silently pairing a POTCAR in the wrong order.
    """
    with open(poscar_path, "r", encoding="utf-8") as fh:
        head_lines = [fh.readline() for _ in range(7)]
    species_line = head_lines[5].split()
    if not species_line or not all(re.fullmatch(r"[A-Z][a-z]?", s) for s in species_line):
        raise ValueError(
            f"Could not parse a VASP5 species line from {poscar_path}; "
            "POTCAR auto-generation requires vasp5=True POSCAR output."
        )
    return species_line


def _hardlink_or_copy(src, dst, use_hardlink=True):
    """
    Hard-link src -> dst when possible (all destinations share identical
    content), falling back to a real copy where hard links are not supported
    (e.g. across filesystems/drives). Returns True if the hard link succeeded.
    """
    if os.path.exists(dst):
        os.remove(dst)
    if use_hardlink:
        try:
            os.link(src, dst)
            return True
        except OSError:
            pass
    shutil.copy2(src, dst)
    return False



VASP_TEMPLATE_FILENAMES = ("INCAR", "KPOINTS", "POTCAR")


def copy_vasp_templates(out_dir, template_dir, filenames=VASP_TEMPLATE_FILENAMES):
    """
    Copy the given exact filenames (default INCAR/KPOINTS/POTCAR) from
    template_dir to each candidate folder.

    Only these exact filenames are copied. template_dir is often just pointed
    at a real (possibly finished) calculation folder found via the INCAR
    search in the wizard, and such folders can also contain OUTCAR, CONTCAR,
    WAVECAR, CHGCAR, vasprun.xml, job scripts, and other large or run-specific
    files that a fresh calculation on the newly generated structures does not
    need; those are intentionally left out. When --generate-potcar builds a
    composition-correct POTCAR itself, `filenames` is passed as just
    ("INCAR", "KPOINTS") so a possibly-mismatched POTCAR sitting in
    template_dir is not blindly copied over it.

    Every candidate folder receives byte-identical copies of the same source
    files, so a hard link is tried first (near-instant, no extra disk usage)
    and a real copy is used as a fallback wherever hard links are not
    supported (e.g. across filesystems/drives). POTCAR files in particular
    can be several MB, so copying them into hundreds of candidate folders can
    take a while; progress is printed periodically so this step never looks
    like it has silently frozen.
    """
    if not template_dir:
        return
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    template_files = [
        f for f in filenames
        if os.path.isfile(os.path.join(template_dir, f))
    ]
    missing = [f for f in filenames if f not in template_files]
    if missing:
        print(f"[notice] template folder is missing: {', '.join(missing)} (skipped, not fatal)", flush=True)
    if not template_files:
        print(
            f"[notice] none of {', '.join(filenames)} were found in {template_dir}; nothing copied.",
            flush=True,
        )
        return

    candidate_dirs = [
        cand for cand in sorted(os.listdir(out_dir))
        if os.path.isdir(os.path.join(out_dir, cand))
    ]
    total = len(candidate_dirs)
    if total == 0:
        return

    print(
        f"Copying {len(template_files)} template file(s) into {total} structure folder(s)...",
        flush=True,
    )
    progress_interval = max(1, total // 20)  # roughly 20 progress updates total
    use_hardlink = True

    for count, cand in enumerate(candidate_dirs, start=1):
        cand_dir = os.path.join(out_dir, cand)
        for f in template_files:
            src = os.path.join(template_dir, f)
            dst = os.path.join(cand_dir, f)
            linked = _hardlink_or_copy(src, dst, use_hardlink=use_hardlink)
            if use_hardlink and not linked:
                # Hard linking failed (e.g. cross-device); fall back for the
                # rest of the run rather than retrying every remaining file.
                use_hardlink = False

        if count % progress_interval == 0 or count == total:
            print(f"  Templates copied: {count}/{total} folders", flush=True)



# -----------------------------------------------------------------------------
# Balanced parent/child utilities
# -----------------------------------------------------------------------------

def _parse_order_levels(order_levels):
    if order_levels is None:
        return [1.0, 0.75, 0.5, 0.25, 0.0]
    if isinstance(order_levels, str):
        levels = [float(x) for x in order_levels.split(',') if x.strip()]
    else:
        levels = [float(x) for x in order_levels]
    if not levels:
        raise ValueError("At least one order level is required.")
    if any(not np.isfinite(level) or not 0.0 <= level <= 1.0 for level in levels):
        raise ValueError(f"order levels must be finite values between 0 and 1, got {levels}")
    return sorted(set(levels), reverse=True)


def _infer_children_per_parent(target, n_parents, order_levels, children_per_parent=None):
    """
    Decide how many randomized descendants to generate per parent for each
    non-parent order level.

    If children_per_parent is given, use it directly.
    If not, infer it from target. The generated balanced set can be slightly
    larger than target because all parents are sampled equally.
    """
    order_levels = _parse_order_levels(order_levels)
    child_levels = [q for q in order_levels if not np.isclose(q, 1.0)]
    has_parent = any(np.isclose(q, 1.0) for q in order_levels)

    if children_per_parent is not None:
        return max(0, int(children_per_parent))

    if not child_levels:
        return 0

    n_ordered = n_parents if has_parent else 0
    remaining = max(0, int(target) - n_ordered)
    denom = max(1, n_parents * len(child_levels))
    return max(1, int(np.ceil(remaining / denom)))


def _parent_map_write(output_dir, parent_records):
    """
    Write parent_map.txt so file labels are traceable.

    Each record contains:
        parent_id, mode, label, order, ordered_parent_saved
    """
    path = os.path.join(output_dir, "parent_map.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# parent_id\tmode\tlabel\torder\tordered_parent_saved\n")
        for rec in parent_records:
            f.write(
                f"{rec.get('parent_id')}\t{rec.get('mode')}\t{rec.get('label')}\t"
                f"{rec.get('order')}\t{rec.get('ordered_parent_saved')}\n"
            )
    return path


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def generate_structures(
    input_file,
    output_dir,
    replace_element,
    composition,
    mode="random",
    target=500,
    max_attempts=2_000_000,
    symprec=1e-3,
    seed=None,
    output_format="cif",
    vasp_folder=False,
    overwrite=False,
    order_levels=None,
    order_tolerance=0.05,
    order_search_steps=5000,
    sro_cutoff_factor=1.20,
    sro_tolerance=0.12,
    sro_weight=0.5,
    max_trials_per_bucket=30,
    template_dir=None,
    exhaustive_limit=2_000_000,
    layer_axis="z",
    view_axis="z",
    domain_pattern=None,
    children_per_parent=None,
    keep_composition=False,
    generate_potcar=False,
    potcar_library=None,
    potcar_variants=None,
    adsorbate_elements=None,
    redox_remove=None,
    redox_max_sets=50,
):
    if mode not in {"random", "spread", "layered", "domain", "exhaustive"}:
        raise ValueError(f"Unknown mode: {mode}")
    if mode != "exhaustive" and (target <= 0 or max_attempts <= 0):
        raise ValueError("target and max_attempts must be positive in sampling modes.")
    if mode in {"layered", "domain"}:
        if not np.isfinite(order_tolerance) or order_tolerance <= 0:
            raise ValueError("order_tolerance must be a finite positive number.")
        if int(order_search_steps) <= 0:
            raise ValueError("order_search_steps must be positive.")
        if not np.isfinite(sro_cutoff_factor) or sro_cutoff_factor <= 1.0:
            raise ValueError("sro_cutoff_factor must be greater than 1.")
        if not np.isfinite(sro_tolerance) or sro_tolerance <= 0:
            raise ValueError("sro_tolerance must be a finite positive number.")
        if not np.isfinite(sro_weight) or sro_weight < 0:
            raise ValueError("sro_weight must be a finite non-negative number.")
        if int(max_trials_per_bucket) <= 0:
            raise ValueError("max_trials_per_bucket must be positive.")

    actual_seed = resolve_seed(seed)
    random.seed(actual_seed)
    np.random.seed(actual_seed)

    parent = _read_structure_file(input_file)
    symbols = parent.get_chemical_symbols()

    if isinstance(replace_element, str):
        replace_elements = parse_element_list(replace_element)
    else:
        replace_elements = list(replace_element)
    replace_element_set = set(replace_elements)

    if adsorbate_elements is None:
        adsorbate_elements_set = None
    else:
        parsed_adsorbate = (
            list(adsorbate_elements) if not isinstance(adsorbate_elements, str)
            else parse_element_list(adsorbate_elements)
        )
        adsorbate_elements_set = set(parsed_adsorbate)
        overlap = adsorbate_elements_set & replace_element_set
        if overlap:
            raise ValueError(
                f"adsorbate_elements overlaps with the replacement pool: {sorted(overlap)}. "
                "An element cannot be both shuffled and stripped as an adsorbate."
            )
        present = adsorbate_elements_set & set(symbols)
        if not present:
            print(
                f"[notice] none of adsorbate_elements={sorted(adsorbate_elements_set)} were "
                "found in the input structure; the surface twin will be identical to the "
                "main structure.",
                flush=True,
            )

    replace_sites = [i for i, s in enumerate(symbols) if s in replace_element_set]

    if composition is None:
        if not keep_composition:
            raise ValueError(
                "composition is required unless keep_composition=True is used to "
                "reuse the existing composition on the selected element pool."
            )
        if not replace_sites:
            raise ValueError(
                "No sites were found for the requested replacement element pool: "
                f"{replace_elements}."
            )
        composition = dict(Counter(symbols[i] for i in replace_sites))

    elements = composition_to_list(composition)
    n_replace = len(elements)

    if len(replace_sites) != n_replace:
        raise ValueError(
            "Full substrate replacement requires the composition sum to equal "
            "the number of replacement sites. "
            f"pool={replace_elements} sites={len(replace_sites)}, "
            f"composition_sum={n_replace}."
        )

    print("Parent composition:", dict(Counter(symbols)))
    print(f"Replacement element pool: {replace_elements}")
    if keep_composition:
        print("Composition mode: reuse existing composition on selected sites (reshuffle only)")
    print(f"Candidate replacement sites: {len(replace_sites)}")
    print(f"Target composition on selected sites: {composition}")
    print(f"Mode: {mode}")
    if mode == "layered":
        print(f"Layer axis: {layer_axis}")
    if mode == "domain":
        _domain_n = len(composition)
        if _domain_n == 4:
            print(f"Domain mode: intuitive 2x2 top-view domain along {view_axis}-axis")
            print(f"Domain pattern: {domain_pattern if domain_pattern else 'composition order (TL,TR/BL,BR)'}")
        elif _domain_n == 5:
            print(f"Domain mode: quincunx top-view domain (center + 4 corners) along {view_axis}-axis")
            print(f"Domain pattern: {domain_pattern if domain_pattern else 'auto-generated (Center:TL,TR/BL,BR)'}")
        else:
            print(f"Domain mode: along {view_axis}-axis ({_domain_n}-element composition; only 4 or 5 are supported)")
    print(f"Random seed: {actual_seed}")

    perms, failed = precompute_symmetry_permutations(parent, symprec=symprec)
    print(f"Symmetry permutations: {len(perms)} (failed mappings: {failed})")

    neighbor_map = None
    nearest_distance = None
    # -- Never chase more structures than the arrangement actually admits.
    #    n!/prod(k_i!) (times the site choice when the pool is only partly
    #    replaced) bounds the distinct decorations from above; symmetry only
    #    reduces it further. Without this, a composition with a single
    #    arrangement (e.g. keeping the current one: Pt16 on 16 Pt sites) would
    #    spin through max_attempts looking for structures that cannot exist.
    if mode in {"random", "spread"} and target > 0:
        if len(replace_sites) == n_replace:
            site_choices = 1
        else:
            site_choices = factorial(len(replace_sites)) // (
                factorial(n_replace) * factorial(len(replace_sites) - n_replace)
            )
        arrangement_bound = site_choices * multiset_count(composition)
        if target > arrangement_bound:
            print(
                f"[notice] target {target} exceeds the {arrangement_bound} distinct "
                f"arrangement(s) this composition allows; generating {arrangement_bound}."
            )
            if arrangement_bound == 1:
                print("         (the composition matches the sites one-to-one, so there "
                      "is only one arrangement -- edit the composition to vary it)")
            target = arrangement_bound

    sro_cutoff = None
    if len(replace_sites) == n_replace:
        neighbor_map, nearest_distance, sro_cutoff = build_neighbor_map(
            parent, replace_sites, cutoff_factor=sro_cutoff_factor
        )
        print(
            f"SRO first shell: nearest={nearest_distance:.6f} Angstrom, "
            f"cutoff={sro_cutoff:.6f} Angstrom"
        )
    elif mode in {"layered", "domain"}:
        if len(replace_sites) != n_replace:
            raise ValueError(
                f"{mode} mode requires full replacement before SRO targeting: "
                f"sites={len(replace_sites)}, composition_sum={n_replace}."
            )

    # Destructive output preparation is intentionally delayed until structural,
    # symmetry, and neighbor-shell validation have all succeeded.
    output_dir = ensure_clean_dir(
        output_dir,
        overwrite=overwrite,
        protected_paths=(input_file, template_dir),
    )

    bare_output_dir = None
    if adsorbate_elements_set:
        bare_output_dir = ensure_clean_dir(
            str(output_dir).rstrip("/\\") + "_surface",
            overwrite=overwrite,
            protected_paths=(input_file, template_dir, output_dir),
        )
        print(f"Surface twin (adsorbate {sorted(adsorbate_elements_set)} removed): {bare_output_dir}")

    # ------------------- redox twins (_r, or _r1.._rN) -------------------
    # redox_remove: which atoms of the *input structure file* (1-based, the
    # order shown in the POSCAR/CIF) to delete from every generated structure.
    # See parse_redox_spec() for the accepted forms. One folder per removal
    # set: a single set -> "_r", several -> "_r1", "_r2", ... Used to compare
    # energies across a redox step, e.g. slab+Li2S4 (main) vs slab+Li2S2
    # (every way of taking away 2 of the 4 S atoms).
    parent_symbols_all = parent.get_chemical_symbols()
    parent_frac_all = parent.get_scaled_positions()
    redox_output_dirs = []
    redox_index_sets = []
    redox_pool = []
    redox_choose = None
    if redox_remove:
        redox_index_sets, redox_pool, redox_choose = parse_redox_spec(
            parent, redox_remove, replace_sites=replace_sites
        )
        n_sets = len(redox_index_sets)
        if n_sets > redox_max_sets:
            raise ValueError(
                f"redox 조합이 {n_sets}개입니다 (후보 {len(redox_pool)}개 중 {redox_choose}개 제거) "
                f"-- 현재 상한은 {redox_max_sets}개입니다.\n"
                "  후보를 줄이거나(원자 번호를 직접 지정), redox_max_sets 를 올려 주세요 "
                "(CLI: -redox_max=#). 구조 수 x 조합 수 만큼 폴더가 생기므로 주의하세요."
            )

        def _describe(indices):
            return " ".join(f"#{i + 1}{parent_symbols_all[i]}" for i in indices)

        if n_sets == 1:
            suffixes = ["_r"]
        else:
            suffixes = [f"_r{i + 1}" for i in range(n_sets)]
        base = str(output_dir).rstrip("/\\")
        for suffix, indices in zip(suffixes, redox_index_sets):
            redox_output_dirs.append(ensure_clean_dir(
                base + suffix,
                overwrite=overwrite,
                protected_paths=(input_file, template_dir, output_dir, bare_output_dir),
            ))

        spec_desc = redox_remove if isinstance(redox_remove, str) else str(redox_remove)
        if n_sets == 1:
            print(f"Redox twin (spec '{spec_desc}'): removing "
                  f"{_describe(redox_index_sets[0])} from every generated structure "
                  f"-> {redox_output_dirs[0]}")
        else:
            print(f"Redox twins (spec '{spec_desc}'): removing {redox_choose} atom(s) "
                  f"per structure -> {n_sets} combination(s), one folder each "
                  f"(candidates: {_describe(redox_pool)})")
            for suffix, indices in zip(suffixes, redox_index_sets):
                print(f"  {base + suffix}  <- remove {_describe(indices)}")

        # Folder -> removed atoms map, so the folders stay interpretable later.
        redox_map_path = os.path.join(output_dir, "redox_sets.csv")
        with open(redox_map_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["folder", "removed_atom_numbers", "removed_elements"])
            for suffix, indices in zip(suffixes, redox_index_sets):
                writer.writerow([
                    os.path.basename(base) + suffix,
                    " ".join(str(i + 1) for i in indices),
                    " ".join(parent_symbols_all[i] for i in indices),
                ])
        print(f"Redox set map written: {redox_map_path}")

    composition_label = " ".join(f"{el}{count}" for el, count in composition.items())

    potcar_source_path = None
    potcar_variant_map_used = None
    potcar_species_order = None
    bare_potcar_source_path = None
    redox_potcar_source_paths = []
    if generate_potcar:
        if not vasp_folder:
            print(
                "[notice] --generate-potcar only applies when writing per-structure "
                "POSCAR folders (vasp_folder=True); skipping POTCAR generation.",
                flush=True,
            )
        else:
            potcar_library = resolve_potcar_library(potcar_library)
            final_species = sorted(
                set(composition.keys())
                | {s for s in symbols if s not in replace_element_set}
            )
            overrides = (
                potcar_variants if isinstance(potcar_variants, dict)
                else parse_potcar_variant_overrides(potcar_variants)
            )
            potcar_variant_map_used = resolve_potcar_variant_map(
                final_species, potcar_library, overrides
            )
            potcar_species_order = final_species

            # Safety check: confirm ASE's sort=True POSCAR species order matches
            # the predicted alphabetical order used to build the combined
            # POTCAR. The decoration used here is throwaway (only which
            # species are present matters, not the arrangement).
            probe_atoms = parent.copy()
            for idx, el in zip(replace_sites, composition_to_list(composition)):
                probe_atoms[idx].symbol = el
            probe_path = os.path.join(output_dir, ".potcar_probe.POSCAR")
            write(probe_path, probe_atoms, format="vasp", direct=True, sort=True, vasp5=True)
            actual_order = _read_poscar_species_order(probe_path)
            os.remove(probe_path)
            if actual_order != potcar_species_order:
                raise RuntimeError(
                    "POTCAR species order mismatch: predicted "
                    f"{potcar_species_order} but the written POSCAR actually orders "
                    f"species as {actual_order}. Refusing to generate a possibly "
                    "mismatched POTCAR (ASE's POSCAR-sorting behavior may have "
                    "changed); please report this."
                )

            combined_text = build_combined_potcar(
                potcar_species_order, potcar_library, potcar_variant_map_used
            )
            potcar_source_path = os.path.join(output_dir, ".generated_POTCAR")
            with open(potcar_source_path, "w", encoding="utf-8") as fh:
                fh.write(combined_text)

            print(f"POTCAR library: {potcar_library}", flush=True)
            print(f"POTCAR species order: {potcar_species_order}", flush=True)
            print(
                "POTCAR variant map: " + ", ".join(
                    f"{el}->{potcar_variant_map_used[el]}" for el in potcar_species_order
                ),
                flush=True,
            )

            if bare_output_dir is not None:
                bare_species_order = [
                    el for el in potcar_species_order if el not in adsorbate_elements_set
                ]
                bare_variant_map = {el: potcar_variant_map_used[el] for el in bare_species_order}

                bare_probe_atoms = _strip_adsorbate_atoms(probe_atoms, adsorbate_elements_set)
                bare_probe_path = os.path.join(bare_output_dir, ".potcar_probe.POSCAR")
                write(bare_probe_path, bare_probe_atoms, format="vasp", direct=True, sort=True, vasp5=True)
                bare_actual_order = _read_poscar_species_order(bare_probe_path)
                os.remove(bare_probe_path)
                if bare_actual_order != bare_species_order:
                    raise RuntimeError(
                        "Bare-slab POTCAR species order mismatch: predicted "
                        f"{bare_species_order} but the written POSCAR actually orders "
                        f"species as {bare_actual_order}. Refusing to generate a "
                        "possibly mismatched POTCAR; please report this."
                    )

                bare_combined_text = build_combined_potcar(
                    bare_species_order, potcar_library, bare_variant_map
                )
                bare_potcar_source_path = os.path.join(bare_output_dir, ".generated_POTCAR")
                with open(bare_potcar_source_path, "w", encoding="utf-8") as fh:
                    fh.write(bare_combined_text)
                print(f"Surface POTCAR species order: {bare_species_order}", flush=True)

            # Each redox set can strip a different mix of elements, so every
            # _r* folder gets its own composition-correct POTCAR.
            for redox_dir, removed_indices in zip(redox_output_dirs, redox_index_sets):
                redox_removed_set = set(removed_indices)
                redox_probe_atoms = probe_atoms[
                    [i for i in range(len(probe_atoms)) if i not in redox_removed_set]
                ]
                redox_present = set(redox_probe_atoms.get_chemical_symbols())
                redox_species_order = [el for el in potcar_species_order if el in redox_present]
                redox_variant_map = {el: potcar_variant_map_used[el] for el in redox_species_order}

                redox_probe_path = os.path.join(redox_dir, ".potcar_probe.POSCAR")
                write(redox_probe_path, redox_probe_atoms, format="vasp", direct=True, sort=True, vasp5=True)
                redox_actual_order = _read_poscar_species_order(redox_probe_path)
                os.remove(redox_probe_path)
                if redox_actual_order != redox_species_order:
                    raise RuntimeError(
                        f"Redox POTCAR species order mismatch in {redox_dir}: predicted "
                        f"{redox_species_order} but the written POSCAR actually orders "
                        f"species as {redox_actual_order}. Refusing to generate a "
                        "possibly mismatched POTCAR; please report this."
                    )

                redox_combined_text = build_combined_potcar(
                    redox_species_order, potcar_library, redox_variant_map
                )
                redox_potcar_path = os.path.join(redox_dir, ".generated_POTCAR")
                with open(redox_potcar_path, "w", encoding="utf-8") as fh:
                    fh.write(redox_combined_text)
                redox_potcar_source_paths.append(redox_potcar_path)
                print(f"Redox POTCAR species order ({os.path.basename(redox_dir)}): "
                      f"{redox_species_order}", flush=True)

    def manifest_metadata(**values):
        record = {"composition": composition_label, "seed": actual_seed}
        record.update(values)
        return record

    def sro_metadata(atoms):
        if neighbor_map is None:
            return {}
        return {
            "sro_rms": f"{warren_cowley_sro_rms(atoms, replace_sites, neighbor_map):.8f}",
            "sro_pairs": _serialized_sro_pairs(atoms, replace_sites, neighbor_map),
        }

    # Save settings immediately so the run is reproducible even if it is interrupted.
    initial_metadata = {
        "input_file": input_file,
        "output_dir": output_dir,
        "replace_elements": replace_elements,
        "keep_composition": keep_composition,
        "parent_composition": dict(Counter(symbols)),
        "composition": composition,
        "mode": mode,
        "target": target,
        "max_attempts": max_attempts,
        "order_levels": _parse_order_levels(order_levels) if mode in {"layered", "domain"} else None,
        "order_parameter": (
            "Q=(symmetry_best_match-random_symmetry_best_match_mean)/(1-random_symmetry_best_match_mean); Q=1 ordered, random ensemble mean~0"
            if mode in {"layered", "domain"} else None
        ),
        "order_parameter_version": Q_DEFINITION_VERSION if mode in {"layered", "domain"} else None,
        "q_random_baseline_samples": Q_RANDOM_BASELINE_SAMPLES if mode in {"layered", "domain"} else None,
        "order_tolerance": order_tolerance if mode in {"layered", "domain"} else None,
        "order_search_steps": order_search_steps if mode in {"layered", "domain"} else None,
        "low_q_min_search_steps": LOW_Q_MIN_SEARCH_STEPS if mode in {"layered", "domain"} else None,
        "low_q_fallback_seeds": list(LOW_Q_FALLBACK_SEEDS) if mode in {"layered", "domain"} else None,
        "sro_definition": "composition-weighted RMS Warren-Cowley alpha over selected first shell",
        "sro_cutoff_factor": sro_cutoff_factor if neighbor_map is not None else None,
        "sro_nearest_distance": nearest_distance,
        "sro_cutoff": sro_cutoff,
        "sro_tolerance": sro_tolerance if mode in {"layered", "domain"} else None,
        "sro_weight": sro_weight if mode in {"layered", "domain"} else None,
        "max_trials_per_bucket": max_trials_per_bucket,
        "layer_axis": layer_axis if mode == "layered" else None,
        "view_axis": view_axis if mode == "domain" else None,
        "domain_pattern": domain_pattern if mode == "domain" else None,
        "children_per_parent": children_per_parent,
        "structure_axes": "random / spread(same-element dispersed) / layered / domain(2x2 phase-separated template)",
        "output_format": output_format,
        "vasp_folder": vasp_folder,
        "generate_potcar": generate_potcar,
        "potcar_library": potcar_library if potcar_source_path else None,
        "potcar_variant_map": potcar_variant_map_used,
        "potcar_species_order": potcar_species_order,
        "adsorbate_elements": sorted(adsorbate_elements_set) if adsorbate_elements_set else None,
        "surface_output_dir": bare_output_dir,
        "redox_output_dirs": redox_output_dirs or None,
        "redox_candidate_atoms": [i + 1 for i in redox_pool] if redox_pool else None,
        "redox_remove_count": redox_choose,
        "redox_sets": [[i + 1 for i in s_] for s_ in redox_index_sets] or None,
        "symprec": symprec,
        "seed_requested": seed,
        "seed_used": actual_seed,
        "overwrite": overwrite,
        "symmetry_permutations": len(perms),
        "failed_symmetry_mappings": failed,
        "status": "started",
    }
    metadata_path = write_metadata(output_dir, initial_metadata)
    print(f"Metadata written: {metadata_path}")

    seen_keys = set()
    kept = 0
    attempts = 0

    if mode == "exhaustive":
        if len(replace_sites) != n_replace:
            print(
                "[Notice] Exhaustive mode will enumerate assignments over the first/selected replacement sites. "
                "For full replacement of a sublattice, composition sum should equal the number of target sites."
            )
        if len(replace_sites) != n_replace:
            # If there are more candidate sites than composition elements, exhaustive site selection + assignment is huge.
            # We allow it only if the total upper bound is acceptable.
            site_comb_count = factorial(len(replace_sites)) // (
                factorial(n_replace) * factorial(len(replace_sites) - n_replace)
            )
        else:
            site_comb_count = 1
        perm_count = multiset_count(composition)
        total_bound = site_comb_count * perm_count
        print(f"Estimated exhaustive upper bound: {total_bound}")
        if total_bound > exhaustive_limit:
            raise ValueError(
                f"Exhaustive generation is too large ({total_bound}). "
                f"Increase --exhaustive-limit or use --mode random/spread."
            )

        site_sets = [replace_sites] if len(replace_sites) == n_replace else itertools.combinations(replace_sites, n_replace)
        for chosen_sites in site_sets:
            for assign in generate_multiset_permutations(elements):
                attempts += 1
                atoms = parent.copy()
                for idx, el in zip(chosen_sites, assign):
                    atoms[idx].symbol = el
                key = canonical_decoration_key(atoms, perms)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                kept += 1
                write_structure_with_twins(
                    atoms, output_dir, kept, output_format, vasp_folder,
                    potcar_source=potcar_source_path,
                    bare_output_dir=bare_output_dir,
                    bare_potcar_source=bare_potcar_source_path,
                    redox_output_dirs=redox_output_dirs,
                    redox_index_sets=redox_index_sets,
                    redox_potcar_sources=redox_potcar_source_paths,
                    adsorbate_elements=adsorbate_elements_set,
                    metadata=manifest_metadata(
                        mode="exhaustive",
                        **sro_metadata(atoms),
                    ),
                )
        print(f"All assignments tried: {attempts}")
        print(f"Symmetry-unique structures saved: {kept}")

    else:
        order_levels = _parse_order_levels(order_levels)

        # -----------------------------------------------------------------
        # Layered mode: exhaustive ordered layer parents first
        # -----------------------------------------------------------------
        if mode == "layered":
            if len(replace_sites) != n_replace:
                raise ValueError(
                    "Layered mode requires full replacement of the selected sublattice: "
                    f"found {len(replace_sites)} sites in pool {replace_elements}, but composition sum is {n_replace}."
                )

            orders = unique_layer_orders(composition)
            print(f"Layer permutations before symmetry filtering: {len(orders)}")
            print("Layered mode first filters symmetry-unique ordered parents.")
            print("Then each unique parent generates balanced children at target order-parameter levels.")

            # Step 1. Build symmetry-unique Q=1 ordered parent structures.
            # Not every layer order necessarily lines up with the real atomic
            # planes of this particular structure (this gets more likely as the
            # number of components grows and layer counts become unequal), so
            # an incompatible order is skipped rather than aborting the whole run.
            parent_entries = []
            parent_seen = set()
            skipped_orders = 0
            max_skip_messages = 5
            for order in orders:
                try:
                    base_atoms, chosen = make_layered_configuration(
                        parent, replace_sites, order, layer_axis=layer_axis
                    )
                except ValueError as exc:
                    skipped_orders += 1
                    if skipped_orders <= max_skip_messages:
                        print(f"  [skip] layer order {'-'.join(el for el, _ in order)} incompatible with atomic planes: {exc}")
                    elif skipped_orders == max_skip_messages + 1:
                        print("  [skip] (further incompatible-order messages suppressed)")
                    continue
                key = canonical_decoration_key(base_atoms, perms)
                if key in parent_seen:
                    continue
                parent_seen.add(key)
                parent_id = len(parent_entries) + 1
                order_label = "-".join(el for el, _ in order)
                parent_entries.append({
                    "parent_id": parent_id,
                    "order": order,
                    "order_label": order_label,
                    "base_atoms": base_atoms,
                    "chosen": chosen,
                    "parent_key": key,
                    "parent_sro": warren_cowley_sro_rms(base_atoms, chosen, neighbor_map),
                    "q_random_match": symmetry_random_match_baseline(
                        base_atoms, chosen, perms
                    ),
                })

            if skipped_orders:
                print(f"Layer orders skipped (incompatible with atomic planes): {skipped_orders}")
            if not parent_entries:
                raise ValueError(
                    "No layer order was compatible with the atomic planes of this structure "
                    f"along axis {layer_axis!r}. Try a different --layer-axis or composition "
                    "with layer counts that match the real number of atoms per plane."
                )
            print(f"Symmetry-unique layered parents: {len(parent_entries)}")

            child_order_levels = [q for q in order_levels if not np.isclose(q, 1.0)]
            cpp = _infer_children_per_parent(target, len(parent_entries), order_levels, children_per_parent)
            print(f"Target Q levels: {order_levels} (tolerance={order_tolerance})")
            print(f"Children per parent per non-parent Q level: {cpp}")

            parent_records = []
            global_seen = set()
            attempts = 0
            kept = 0

            # Step 2. Save Q=1 ordered parent structures once if requested.
            if any(np.isclose(q, 1.0) for q in order_levels):
                for entry in parent_entries:
                    if attempts >= max_attempts or kept >= target:
                        break
                    attempts += 1
                    atoms = entry["base_atoms"].copy()
                    key = canonical_decoration_key(atoms, perms)
                    if key in global_seen:
                        ordered_parent_saved = False
                    else:
                        global_seen.add(key)
                        kept += 1
                        write_structure_with_twins(
                            atoms, output_dir, kept, output_format, vasp_folder,
                            potcar_source=potcar_source_path,
                            bare_output_dir=bare_output_dir,
                            bare_potcar_source=bare_potcar_source_path,
                            redox_output_dirs=redox_output_dirs,
                            redox_index_sets=redox_index_sets,
                            redox_potcar_sources=redox_potcar_source_paths,
                            adsorbate_elements=adsorbate_elements_set,
                            metadata=manifest_metadata(
                                parent_id=f"P{entry['parent_id']:03d}", mode="layered",
                                axis=layer_axis, pattern=entry["order_label"],
                                target_q=1.0, actual_q=1.0,
                                q_random_match=f"{entry['q_random_match']:.8f}",
                                q_definition=Q_DEFINITION_VERSION,
                                sro_rms=f"{entry['parent_sro']:.8f}",
                                sro_pairs=_serialized_sro_pairs(
                                    atoms, entry["chosen"], neighbor_map
                                ),
                            ),
                        )
                        ordered_parent_saved = True
                    parent_records.append({
                        "parent_id": entry["parent_id"],
                        "mode": f"layer_{layer_axis}",
                        "label": f"parent{entry['parent_id']:03d}",
                        "order": entry["order_label"],
                        "ordered_parent_saved": ordered_parent_saved,
                    })
            else:
                for entry in parent_entries:
                    parent_records.append({
                        "parent_id": entry["parent_id"],
                        "mode": f"layer_{layer_axis}",
                        "label": f"parent{entry['parent_id']:03d}",
                        "order": entry["order_label"],
                        "ordered_parent_saved": False,
                    })

            # Step 3. Round-robin over parent x Q buckets so an exact target
            # does not systematically starve later parents or lower Q levels.
            child_jobs = [
                (entry, target_q)
                for _round in range(cpp)
                for entry in parent_entries
                for target_q in child_order_levels
            ]
            for entry, target_q in child_jobs:
                if kept >= target or attempts >= max_attempts:
                    break
                child_trials = 0
                max_child_trials = int(max_trials_per_bucket)
                resume_candidate = None
                while child_trials < max_child_trials and attempts < max_attempts:
                        child_trials += 1
                        attempts += 1
                        low_q_search = target_q <= max(float(order_tolerance), 0.10)
                        search_seed = "primary"
                        search_rng = random
                        initial_candidate = resume_candidate
                        if low_q_search and child_trials >= 2:
                            fallback_index = (child_trials - 2) % len(LOW_Q_FALLBACK_SEEDS)
                            search_seed = LOW_Q_FALLBACK_SEEDS[fallback_index]
                            search_rng = random.Random(search_seed)
                            initial_candidate = None if child_trials == 2 else resume_candidate
                        atoms, actual_q, actual_sro = generate_to_order_target(
                            entry["base_atoms"],
                            entry["chosen"],
                            target_q,
                            tolerance=order_tolerance,
                            search_steps=order_search_steps,
                            perms=perms,
                            neighbor_map=neighbor_map,
                            sro_tolerance=sro_tolerance,
                            sro_weight=sro_weight,
                            initial_candidate=initial_candidate,
                            rng=search_rng,
                        )
                        if abs(actual_q - target_q) > order_tolerance:
                            resume_candidate = atoms
                            continue
                        target_sro = target_q * entry["parent_sro"]
                        if actual_sro is not None and actual_sro > target_sro + sro_tolerance:
                            resume_candidate = atoms
                            continue
                        key = canonical_decoration_key(atoms, perms)
                        if key == entry["parent_key"] or key in global_seen:
                            resume_candidate = atoms
                            continue
                        global_seen.add(key)
                        kept += 1
                        write_structure_with_twins(
                            atoms, output_dir, kept, output_format, vasp_folder,
                            potcar_source=potcar_source_path,
                            bare_output_dir=bare_output_dir,
                            bare_potcar_source=bare_potcar_source_path,
                            redox_output_dirs=redox_output_dirs,
                            redox_index_sets=redox_index_sets,
                            redox_potcar_sources=redox_potcar_source_paths,
                            adsorbate_elements=adsorbate_elements_set,
                            metadata=manifest_metadata(
                                parent_id=f"P{entry['parent_id']:03d}", mode="layered",
                                axis=layer_axis, pattern=entry["order_label"],
                                target_q=f"{target_q:.8f}", actual_q=f"{actual_q:.8f}",
                                q_random_match=f"{entry['q_random_match']:.8f}",
                                q_definition=Q_DEFINITION_VERSION,
                                sro_rms=f"{actual_sro:.8f}",
                                sro_pairs=_serialized_sro_pairs(
                                    atoms, entry["chosen"], neighbor_map
                                ),
                                search_seed=search_seed,
                            ),
                        )
                        if kept % 50 == 0:
                            print(f"Saved {kept} unique structures (attempts={attempts})")
                        break

            parent_map_path = _parent_map_write(output_dir, parent_records)
            print(f"Parent map written: {parent_map_path}")
            print(f"Unique layered structures saved: {kept}")
            print(f"Layer parent/child candidates tried: {attempts}")
            if kept < target:
                print("[Notice] Target was not reached. Balanced parent sampling and symmetry duplicates can reduce the final count.")
            elif kept == target:
                print("Requested target count was reached exactly.")

        # -----------------------------------------------------------------
        # Domain mode: explicit human-intuitive 2x2 top-view domain parent
        # -----------------------------------------------------------------
        elif mode == "domain":
            if len(replace_sites) != n_replace:
                raise ValueError(
                    "Domain mode requires full replacement of the selected sublattice: "
                    f"found {len(replace_sites)} sites in pool {replace_elements}, but composition sum is {n_replace}."
                )

            domain_component_count = len(composition)
            if domain_component_count == 4:
                domain_template_orders = unique_domain_orders(composition, domain_pattern)
                domain_configurator = make_domain_template_configuration
                domain_scheme_desc = "intuitive 2x2 top-view domain (TL/TR/BL/BR)"
                default_pattern_desc = "composition order (TL,TR/BL,BR)"

                def _domain_label(order):
                    return (
                        "-".join(el for el, _ in order[:2])
                        + "_over_" + "-".join(el for el, _ in order[2:])
                    )
            elif domain_component_count == 5:
                domain_template_orders = unique_quincunx_orders(composition, domain_pattern)
                domain_configurator = make_quincunx_configuration
                domain_scheme_desc = "quincunx top-view domain (center + TL/TR/BL/BR corners)"
                default_pattern_desc = "auto-generated center/corner assignment (Center:TL,TR/BL,BR)"

                def _domain_label(order):
                    center_el = order[0][0]
                    return (
                        f"{center_el}center_"
                        + "-".join(el for el, _ in order[1:3])
                        + "_over_" + "-".join(el for el, _ in order[3:5])
                    )
            else:
                raise ValueError(
                    "domain mode currently supports exactly 4 elements (rectangular 2x2 "
                    "top-view template) or 5 elements (quincunx: center + 4 corners); "
                    f"got {domain_component_count} elements in composition: {list(composition.keys())}."
                )

            if domain_pattern:
                print(f"Domain template specified by user: {domain_pattern}")
            else:
                print(f"Domain templates before symmetry filtering: {len(domain_template_orders)}")
                print(f"Domain mode first filters symmetry-unique {domain_scheme_desc} parents.")
                print("Then each unique parent generates balanced children at target order-parameter levels.")
            print(f"View axis: {view_axis}")

            # Step 1. Build symmetry-unique Q=1 domain parent structures.
            # As with layered mode, a candidate order can fail the geometric
            # rank-cut validation (e.g. an ambiguous coordinate-plane split);
            # such orders are skipped instead of aborting the whole run.
            parent_entries = []
            parent_seen = set()
            skipped_orders = 0
            max_skip_messages = 5
            for domain_template_order in domain_template_orders:
                try:
                    base_atoms, chosen = domain_configurator(
                        parent, replace_sites, domain_template_order, view_axis=view_axis
                    )
                except ValueError as exc:
                    skipped_orders += 1
                    if skipped_orders <= max_skip_messages:
                        print(f"  [skip] domain order {_domain_label(domain_template_order)} incompatible: {exc}")
                    elif skipped_orders == max_skip_messages + 1:
                        print("  [skip] (further incompatible-order messages suppressed)")
                    continue
                key = canonical_decoration_key(base_atoms, perms)
                if key in parent_seen:
                    continue
                parent_seen.add(key)
                parent_id = len(parent_entries) + 1
                domain_label = _domain_label(domain_template_order)
                parent_entries.append({
                    "parent_id": parent_id,
                    "order": domain_template_order,
                    "order_label": domain_label,
                    "base_atoms": base_atoms,
                    "chosen": chosen,
                    "parent_key": key,
                    "parent_sro": warren_cowley_sro_rms(base_atoms, chosen, neighbor_map),
                    "q_random_match": symmetry_random_match_baseline(
                        base_atoms, chosen, perms
                    ),
                })

            if skipped_orders:
                print(f"Domain orders skipped (incompatible geometry): {skipped_orders}")
            if not parent_entries:
                raise ValueError(
                    "No domain order was geometrically compatible with this structure. "
                    "Try a different --view-axis, an explicit --domain-pattern, or a "
                    "composition whose counts match the real site geometry."
                )
            print(f"Symmetry-unique domain parents: {len(parent_entries)}")

            child_order_levels = [q for q in order_levels if not np.isclose(q, 1.0)]
            cpp = _infer_children_per_parent(target, len(parent_entries), order_levels, children_per_parent)
            print(f"Target Q levels: {order_levels} (tolerance={order_tolerance})")
            print(f"Children per parent per non-parent Q level: {cpp}")

            parent_records = []
            global_seen = set()
            attempts = 0
            kept = 0

            # Step 2. Save Q=1 ordered parent structures once if requested.
            if any(np.isclose(q, 1.0) for q in order_levels):
                for entry in parent_entries:
                    if attempts >= max_attempts or kept >= target:
                        break
                    attempts += 1
                    atoms = entry["base_atoms"].copy()
                    key = canonical_decoration_key(atoms, perms)
                    if key in global_seen:
                        ordered_parent_saved = False
                    else:
                        global_seen.add(key)
                        kept += 1
                        write_structure_with_twins(
                            atoms, output_dir, kept, output_format, vasp_folder,
                            potcar_source=potcar_source_path,
                            bare_output_dir=bare_output_dir,
                            bare_potcar_source=bare_potcar_source_path,
                            redox_output_dirs=redox_output_dirs,
                            redox_index_sets=redox_index_sets,
                            redox_potcar_sources=redox_potcar_source_paths,
                            adsorbate_elements=adsorbate_elements_set,
                            metadata=manifest_metadata(
                                parent_id=f"P{entry['parent_id']:03d}", mode="domain",
                                axis=view_axis, pattern=entry["order_label"],
                                target_q=1.0, actual_q=1.0,
                                q_random_match=f"{entry['q_random_match']:.8f}",
                                q_definition=Q_DEFINITION_VERSION,
                                sro_rms=f"{entry['parent_sro']:.8f}",
                                sro_pairs=_serialized_sro_pairs(
                                    atoms, entry["chosen"], neighbor_map
                                ),
                            ),
                        )
                        ordered_parent_saved = True
                    parent_records.append({
                        "parent_id": entry["parent_id"],
                        "mode": f"domain_{view_axis}",
                        "label": f"parent{entry['parent_id']:03d}",
                        "order": entry["order_label"],
                        "ordered_parent_saved": ordered_parent_saved,
                    })
            else:
                for entry in parent_entries:
                    parent_records.append({
                        "parent_id": entry["parent_id"],
                        "mode": f"domain_{view_axis}",
                        "label": f"parent{entry['parent_id']:03d}",
                        "order": entry["order_label"],
                        "ordered_parent_saved": False,
                    })

            # Step 3. Round-robin over parent x Q buckets.
            child_jobs = [
                (entry, target_q)
                for _round in range(cpp)
                for entry in parent_entries
                for target_q in child_order_levels
            ]
            for entry, target_q in child_jobs:
                if kept >= target or attempts >= max_attempts:
                    break
                child_trials = 0
                max_child_trials = int(max_trials_per_bucket)
                resume_candidate = None
                while child_trials < max_child_trials and attempts < max_attempts:
                        child_trials += 1
                        attempts += 1
                        low_q_search = target_q <= max(float(order_tolerance), 0.10)
                        search_seed = "primary"
                        search_rng = random
                        initial_candidate = resume_candidate
                        if low_q_search and child_trials >= 2:
                            # Retry a failed Q=0 basin reproducibly. The first
                            # fallback starts fresh; later fallbacks continue from
                            # the best candidate found so far.
                            fallback_index = (child_trials - 2) % len(LOW_Q_FALLBACK_SEEDS)
                            search_seed = LOW_Q_FALLBACK_SEEDS[fallback_index]
                            search_rng = random.Random(search_seed)
                            initial_candidate = None if child_trials == 2 else resume_candidate
                        atoms, actual_q, actual_sro = generate_to_order_target(
                            entry["base_atoms"],
                            entry["chosen"],
                            target_q,
                            tolerance=order_tolerance,
                            search_steps=order_search_steps,
                            perms=perms,
                            neighbor_map=neighbor_map,
                            sro_tolerance=sro_tolerance,
                            sro_weight=sro_weight,
                            initial_candidate=initial_candidate,
                            rng=search_rng,
                        )
                        if abs(actual_q - target_q) > order_tolerance:
                            if low_q_search:
                                print(
                                    f"  Q={target_q:g} retry {child_trials}: "
                                    f"actual Q={actual_q:.6f}, SRO={actual_sro:.6f}, "
                                    f"search seed={search_seed}"
                                )
                            resume_candidate = atoms
                            continue
                        target_sro = target_q * entry["parent_sro"]
                        if actual_sro is not None and actual_sro > target_sro + sro_tolerance:
                            if low_q_search:
                                print(
                                    f"  Q={target_q:g} retry {child_trials}: "
                                    f"Q passed ({actual_q:.6f}), SRO={actual_sro:.6f} "
                                    f"exceeded {target_sro + sro_tolerance:.6f}, "
                                    f"search seed={search_seed}"
                                )
                            resume_candidate = atoms
                            continue
                        key = canonical_decoration_key(atoms, perms)
                        if key == entry["parent_key"] or key in global_seen:
                            if low_q_search:
                                print(
                                    f"  Q={target_q:g} retry {child_trials}: symmetry duplicate, "
                                    f"search seed={search_seed}"
                                )
                            resume_candidate = atoms
                            continue
                        global_seen.add(key)
                        kept += 1
                        write_structure_with_twins(
                            atoms, output_dir, kept, output_format, vasp_folder,
                            potcar_source=potcar_source_path,
                            bare_output_dir=bare_output_dir,
                            bare_potcar_source=bare_potcar_source_path,
                            redox_output_dirs=redox_output_dirs,
                            redox_index_sets=redox_index_sets,
                            redox_potcar_sources=redox_potcar_source_paths,
                            adsorbate_elements=adsorbate_elements_set,
                            metadata=manifest_metadata(
                                parent_id=f"P{entry['parent_id']:03d}", mode="domain",
                                axis=view_axis, pattern=entry["order_label"],
                                target_q=f"{target_q:.8f}", actual_q=f"{actual_q:.8f}",
                                q_random_match=f"{entry['q_random_match']:.8f}",
                                q_definition=Q_DEFINITION_VERSION,
                                sro_rms=f"{actual_sro:.8f}",
                                sro_pairs=_serialized_sro_pairs(
                                    atoms, entry["chosen"], neighbor_map
                                ),
                                search_seed=search_seed,
                            ),
                        )
                        if kept % 50 == 0:
                            print(f"Saved {kept} unique structures (attempts={attempts})")
                        break

            parent_map_path = _parent_map_write(output_dir, parent_records)
            print(f"Parent map written: {parent_map_path}")
            print(f"Unique domain structures saved: {kept}")
            print(f"Domain parent/child candidates tried: {attempts}")
            if kept < target:
                print("[Notice] Target was not reached. Balanced parent sampling and symmetry duplicates can reduce the final count.")
            elif kept == target:
                print("Requested target count was reached exactly.")

        # -----------------------------------------------------------------
        # Other sampling modes
        # -----------------------------------------------------------------
        else:
            while kept < target and attempts < max_attempts:
                attempts += 1

                if mode == "random":
                    atoms, chosen = random_configuration(parent, replace_sites, elements)
                    generated_mode = "random"
                elif mode == "spread":
                    atoms, chosen = spread_configuration(parent, replace_sites, elements)
                    generated_mode = "spread"
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                key = canonical_decoration_key(atoms, perms)
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                kept += 1
                write_structure_with_twins(
                    atoms, output_dir, kept, output_format, vasp_folder,
                    potcar_source=potcar_source_path,
                    bare_output_dir=bare_output_dir,
                    bare_potcar_source=bare_potcar_source_path,
                    redox_output_dirs=redox_output_dirs,
                    redox_index_sets=redox_index_sets,
                    redox_potcar_sources=redox_potcar_source_paths,
                    adsorbate_elements=adsorbate_elements_set,
                    metadata=manifest_metadata(
                        mode=generated_mode,
                        **sro_metadata(atoms),
                    ),
                )

                if kept % 50 == 0:
                    print(f"Saved {kept}/{target} unique structures (attempts={attempts})")

            print(f"Unique structures saved: {kept}")
            print(f"Attempts: {attempts}")
            if kept < target:
                print("[Warning] Target was not reached. Increase max_attempts or lower target.")

    if vasp_folder and template_dir:
        template_filenames = (
            ("INCAR", "KPOINTS") if potcar_source_path else VASP_TEMPLATE_FILENAMES
        )
        copy_vasp_templates(output_dir, template_dir, filenames=template_filenames)
        print(f"Copied VASP template files from: {template_dir}")
        if bare_output_dir is not None:
            bare_template_filenames = (
                ("INCAR", "KPOINTS") if bare_potcar_source_path else VASP_TEMPLATE_FILENAMES
            )
            copy_vasp_templates(bare_output_dir, template_dir, filenames=bare_template_filenames)
        for _i, redox_dir in enumerate(redox_output_dirs):
            _has_potcar = _i < len(redox_potcar_source_paths)
            redox_template_filenames = (
                ("INCAR", "KPOINTS") if _has_potcar else VASP_TEMPLATE_FILENAMES
            )
            copy_vasp_templates(redox_dir, template_dir, filenames=redox_template_filenames)

    if potcar_source_path and os.path.exists(potcar_source_path):
        os.remove(potcar_source_path)
    if bare_potcar_source_path and os.path.exists(bare_potcar_source_path):
        os.remove(bare_potcar_source_path)
    for _p in redox_potcar_source_paths:
        if _p and os.path.exists(_p):
            os.remove(_p)

    final_metadata = initial_metadata.copy()
    final_metadata.update({
        "status": "finished",
        "attempts": attempts,
        "unique_structures_saved": kept,
        "template_dir": template_dir,
    })
    metadata_path = write_metadata(output_dir, final_metadata)
    print(f"Metadata updated: {metadata_path}")
    print(f"Output directory: {output_dir}")

    if bare_output_dir is not None:
        bare_metadata = dict(final_metadata)
        bare_metadata.update({
            "is_surface_twin_of": output_dir,
            "adsorbate_elements_removed": sorted(adsorbate_elements_set),
        })
        write_metadata(bare_output_dir, bare_metadata)
        print(f"Surface twin output directory: {bare_output_dir}")

    for redox_dir, removed_indices in zip(redox_output_dirs, redox_index_sets):
        redox_metadata = dict(final_metadata)
        redox_metadata.update({
            "is_redox_twin_of": output_dir,
            "redox_removed_atoms": [
                f"#{i + 1} {parent_symbols_all[i]}" for i in removed_indices
            ],
        })
        write_metadata(redox_dir, redox_metadata)
        print(f"Redox twin output directory: {redox_dir}")

    # Report the directories actually created, so callers (e.g. the -vasp
    # stage) can walk every twin without re-deriving folder names.
    return {
        "output_dir": output_dir,
        "surface_dir": bare_output_dir,
        "redox_dirs": list(redox_output_dirs),
        "unique_structures_saved": kept,
    }


# -----------------------------------------------------------------------------
# Adsorbate detection / interactive twin selection
# -----------------------------------------------------------------------------

# Below this largest periodic gap the cell is treated as bulk (no free surface),
# so "is this atom on the surface?" cannot be decided geometrically.
VACUUM_MIN_GAP = 4.0     # Angstrom
# Slack when deciding whether an atom sits outside the substrate slab.
ADSORBATE_Z_TOL = 0.5    # Angstrom


def _largest_vacuum_axis(atoms):
    """
    Return (axis, gap, origin_frac) for the cell axis with the largest periodic
    gap between atoms -- the likely slab normal. `origin_frac` is the fractional
    coordinate just after that gap, so shifting coordinates by it "unwraps" the
    slab into one contiguous block instead of splitting it across the boundary.
    """
    frac = atoms.get_scaled_positions(wrap=True)
    lengths = atoms.cell.lengths()
    best = None
    for axis in range(3):
        column = np.sort(frac[:, axis])
        if len(column) == 0:
            continue
        gaps = np.diff(np.append(column, column[0] + 1.0))
        k = int(np.argmax(gaps))
        gap = float(gaps[k] * lengths[axis])
        origin = float(column[(k + 1) % len(column)])
        if best is None or gap > best[1]:
            best = (axis, gap, origin)
    return best


def select_structure_file_interactively(prompt="Choose file : "):
    """
    Ask which structure to work on, laid out the way CCpy's own input pickers
    do it (`1 : CONTCAR`, then `Choose file : `). Only real files are listed --
    CCpy's selectInputs() matches by substring and so also offers output
    directories such as CONTCAR_random_..._output, which cannot be read.

    Each line also carries the formula and the guessed substrate/adsorbate
    split, so the choice can be made without opening the files. Returns the
    chosen filename, or None if the user quits.
    """
    files = _find_structure_files()
    if not files:
        print("현재 폴더에서 구조 파일을 찾지 못했습니다 (.cif, .vasp, POSCAR, CONTCAR).")
        return None

    print()
    for n, filename in enumerate(files, start=1):
        try:
            parent = _read_structure_file(filename)
            formula = parent.get_chemical_formula()
            substrate, adsorbate, _is_slab = guess_substrate_elements(parent)
            tag = "기판=%s" % ",".join(substrate) if substrate else ""
            if adsorbate:
                tag += " / 흡착물=%s" % ",".join(adsorbate)
        except Exception:
            formula, tag = "(읽기 실패)", ""
        print("%d : %-26s %-20s %s" % (n, filename, formula, tag))

    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            print("\n입력이 없어 취소합니다.")
            return None
        if answer.lower() in ("q", "quit"):
            print("취소했습니다.")
            return None
        if answer.isdigit():
            n = int(answer)
            if 1 <= n <= len(files):
                return files[n - 1]
            print("1 부터 %d 사이의 번호를 입력해 주세요. (q=취소)" % len(files))
            continue
        if os.path.isfile(answer):
            return answer
        print("번호나 파일명을 입력해 주세요. (q=취소)")


def guess_substrate_elements(parent, vacuum_min_gap=VACUUM_MIN_GAP,
                             z_tol=ADSORBATE_Z_TOL):
    """
    Guess which elements form the substrate, *without* being told the
    replacement pool first. Used to seed the settings sheet, so a Pt-free
    CONTCAR never shows "replace = Pt".

    An element is called an adsorbate only when both signals agree:

      few    -- it has at most half as many atoms as the most abundant element
                (an adsorbate is a small cluster; an alloy component is not), and
      on top -- every one of its atoms sits above the top of the most abundant
                element's span (a substitutional dopant sits inside it, so it
                stays in the substrate).

    Without a vacuum gap there is no free surface at all, so everything counts
    as substrate. Requiring both signals keeps ordered/layered alloys intact:
    the upper layer of a layered alloy is on top, but it is not few.

    Returns (substrate_elements, adsorbate_elements, is_slab), lists sorted.
    This is a starting point for the user to edit, not a determination: it
    misjudges a structure whose adsorbate rivals the slab in size, or one with
    adsorbates on both faces.
    """
    symbols = [str(sym) for sym in parent.get_chemical_symbols()]
    all_elements = sorted(set(symbols))
    if not symbols:
        return [], [], False

    axis, gap, origin = _largest_vacuum_axis(parent)
    if gap < vacuum_min_gap:
        return all_elements, [], False

    lengths = parent.cell.lengths()
    frac = parent.get_scaled_positions(wrap=True)
    coord = ((frac[:, axis] - origin) % 1.0) * lengths[axis]

    counts = Counter(symbols)
    max_count = max(counts.values())
    few_threshold = max(1, max_count // 2)
    backbone = sorted(all_elements, key=lambda el: (-counts[el], el))[0]
    backbone_top = max(coord[i] for i, sym in enumerate(symbols) if sym == backbone)

    substrate, adsorbate = [], []
    for element in all_elements:
        idx = [i for i, sym in enumerate(symbols) if sym == element]
        few = counts[element] <= few_threshold
        on_top = min(coord[i] for i in idx) > backbone_top - z_tol
        (adsorbate if (few and on_top) else substrate).append(element)
    return substrate, adsorbate, True


def detect_adsorbate_candidates(parent, replace_elements,
                                vacuum_min_gap=VACUUM_MIN_GAP,
                                z_tol=ADSORBATE_Z_TOL):
    """
    Heuristically split the non-pool elements of `parent` into adsorbates and
    substrate-internal species (substitutional dopants, interstitials).

    Method: the replacement pool defines the substrate. Along the detected slab
    normal (the axis with the largest vacuum gap), atoms lying beyond the
    substrate's own extent are on a free surface -> adsorbate; atoms inside that
    extent are part of the substrate -> not an adsorbate.

    This is a *suggestion*, not a determination. It cannot decide anything for a
    bulk cell (no vacuum -> is_slab False), and it can misjudge intercalated
    species or adsorbates that sit level with the top substrate layer. Callers
    must show the evidence and let the user confirm.

    Returns a dict with keys: pool, candidates, suggested, axis, axis_name,
    vacuum_gap, is_slab, substrate_range, details, reason.
    """
    symbols = np.array(parent.get_chemical_symbols())
    replace_set = set(replace_elements)
    pool_idx = [i for i, s in enumerate(symbols) if s in replace_set]
    other_idx = [i for i, s in enumerate(symbols) if s not in replace_set]

    result = {
        # str() so numpy string scalars don't leak into printed output
        "pool": sorted({str(s) for s in symbols[pool_idx]}) if pool_idx else [],
        "candidates": sorted({str(s) for s in symbols[other_idx]}) if other_idx else [],
        "suggested": [],
        "axis": None,
        "axis_name": None,
        "vacuum_gap": None,
        "is_slab": False,
        "substrate_range": None,
        "details": {},
        "reason": "",
    }

    if not other_idx:
        result["reason"] = "치환 풀 밖의 원소가 없습니다 (제거할 흡착물 후보 없음)."
        return result
    if not pool_idx:
        result["reason"] = "치환 풀에 해당하는 원자가 없어 기판을 정의할 수 없습니다."
        return result

    axis, gap, origin = _largest_vacuum_axis(parent)
    lengths = parent.cell.lengths()
    frac = parent.get_scaled_positions(wrap=True)
    coord = ((frac[:, axis] - origin) % 1.0) * lengths[axis]

    result["axis"] = axis
    result["axis_name"] = "xyz"[axis]
    result["vacuum_gap"] = gap
    result["is_slab"] = gap >= vacuum_min_gap

    pool_coord = coord[pool_idx]
    result["substrate_range"] = (float(pool_coord.min()), float(pool_coord.max()))

    for element in result["candidates"]:
        idx = [i for i in other_idx if symbols[i] == element]
        el_coord = coord[idx]
        try:
            min_dist = float(min(
                float(np.min(parent.get_distances(i, pool_idx, mic=True))) for i in idx
            ))
        except Exception:
            min_dist = None
        outside = bool(np.all(
            (el_coord > pool_coord.max() + z_tol) | (el_coord < pool_coord.min() - z_tol)
        ))
        result["details"][element] = {
            "count": len(idx),
            "atom_numbers": [i + 1 for i in idx],
            "coord_min": float(el_coord.min()),
            "coord_max": float(el_coord.max()),
            "min_distance_to_pool": min_dist,
            "outside_substrate": outside,
        }
        if result["is_slab"] and outside:
            result["suggested"].append(element)

    if not result["is_slab"]:
        result["reason"] = (
            f"최대 빈틈이 {gap:.2f} A 뿐이라 진공층이 없는 벌크 구조로 판단됩니다. "
            "표면/내부를 기하학적으로 구분할 수 없으므로 직접 선택해 주세요."
        )
    elif not result["suggested"]:
        result["reason"] = (
            "치환 풀 밖의 원소가 모두 기판 좌표 범위 안에 있어 흡착물로 판단되지 "
            "않았습니다 (치환/도핑된 원소로 보입니다)."
        )
    return result


def print_adsorbate_analysis(parent, info):
    """Print the evidence behind detect_adsorbate_candidates()'s suggestion."""
    counts = Counter(parent.get_chemical_symbols())
    print("\n[흡착물 분석]")
    print(f"  전체 조성          : {dict(counts)}")
    print(f"  치환 풀(기판) 원소 : {info['pool']}")
    if info["axis_name"]:
        verdict = "slab 으로 판단" if info["is_slab"] else "벌크로 판단 (진공층 없음)"
        print(f"  진공축 추정        : {info['axis_name']}축, 최대 빈틈 {info['vacuum_gap']:.2f} A  -> {verdict}")
    if info["substrate_range"]:
        lo, hi = info["substrate_range"]
        print(f"  기판 좌표 범위     : {lo:.2f} ~ {hi:.2f} A  ({info['axis_name']}축 기준)")
    if info["details"]:
        print()
        print("  원소  개수  좌표 범위(A)        최근접 기판거리  판정")
        for element in info["candidates"]:
            d = info["details"][element]
            dist = "  -  " if d["min_distance_to_pool"] is None else f"{d['min_distance_to_pool']:.2f} A"
            if not info["is_slab"]:
                mark = "판단 불가 (벌크)"
            elif d["outside_substrate"]:
                mark = "흡착물로 판단"
            else:
                mark = "기판 내부 (치환/도핑) -> 제외 권장"
            print("  %-4s  %4d  %6.2f ~ %6.2f      %-14s  %s"
                  % (element, d["count"], d["coord_min"], d["coord_max"], dist, mark))
    if info["reason"]:
        print(f"\n  * {info['reason']}")


def interactive_select_surface_elements(parent, replace_elements):
    """
    Show the adsorbate analysis and ask which elements to strip for the
    _surface twin. Returns a comma-joined element string, or None when the user
    declines or there is nothing to strip.
    """
    info = detect_adsorbate_candidates(parent, replace_elements)
    if not info["candidates"]:
        print("\n[surface] " + (info["reason"] or "제거할 후보 원소가 없습니다."))
        return None

    print_adsorbate_analysis(parent, info)
    replace_set = set(replace_elements)

    while True:
        if info["suggested"]:
            suggestion = ",".join(info["suggested"])
            print(f"\n  * 자동 판단 결과: {suggestion}")
            prompt = ("* 이대로 제거할까요? (엔터=예 / 원소 직접 입력 (예: Li,S) / "
                      "n=surface 트윈 안 만듦)\n: ")
        else:
            suggestion = None
            prompt = ("\n* 제거할 원소를 직접 입력해 주세요 (예: Li,S / "
                      "n=surface 트윈 안 만듦)\n: ")
        try:
            answer = input(prompt).strip()
        except EOFError:
            print("\n[surface] 입력이 없어 surface 트윈을 만들지 않습니다.")
            return None

        if answer.lower() in ("n", "no"):
            print("[surface] surface 트윈을 만들지 않습니다.")
            return None
        if not answer:
            if suggestion:
                return suggestion
            print("[입력 오류] 제거할 원소를 입력하거나 n 을 입력해 주세요.")
            continue

        try:
            chosen = parse_element_list(answer)
        except Exception as exc:
            print(f"[입력 오류] {exc}")
            continue
        overlap = set(chosen) & replace_set
        if overlap:
            print(f"[입력 오류] 치환 풀 원소는 제거할 수 없습니다: {sorted(overlap)}")
            continue
        missing = [el for el in chosen if el not in info["candidates"]]
        if missing:
            print(f"[입력 오류] 입력 구조에 없는 원소입니다: {missing} "
                  f"(후보: {info['candidates']})")
            continue
        return ",".join(chosen)


def interactive_select_redox_atoms(parent, replace_elements, max_sets=50):
    """
    List every removable atom (i.e. not part of the replacement pool) with its
    input-file atom number and coordinates, ask which ones are candidates, and
    then how many of them to remove.

    Removing all of the chosen atoms gives a single `_r` folder; removing k of
    n gives every combination its own folder (`_r1`, `_r2`, ...), which is how
    a redox series like Li2S4 -> Li2S2 is enumerated.

    Returns a spec string accepted by parse_redox_spec() (e.g. '19,20,21,22/2'),
    or None when the user declines.
    """
    symbols = parent.get_chemical_symbols()
    frac = parent.get_scaled_positions(wrap=True)
    replace_set = set(replace_elements)
    removable = [i for i, s in enumerate(symbols) if s not in replace_set]

    if not removable:
        print("\n[redox] 치환 풀 밖의 원자가 없어 제거할 수 있는 원자가 없습니다.")
        return None

    info = detect_adsorbate_candidates(parent, replace_elements)
    print("\n[redox 제거 가능 원자]  (치환 풀 원자는 배열 매핑이 깨지므로 제외됨)")
    print("  번호   원소   분율좌표 (a, b, c)              비고")
    for i in removable:
        note = ""
        detail = info["details"].get(symbols[i])
        if detail and info["is_slab"]:
            note = "흡착물로 판단" if detail["outside_substrate"] else "기판 내부 (치환/도핑)"
        print("  #%-5d %-4s  (%.4f, %.4f, %.4f)   %s"
              % (i + 1, symbols[i], frac[i][0], frac[i][1], frac[i][2], note))

    print("\n  [입력 형식]")
    print("    제거 조성 : 'S1'=S 1개  'S2'=S 2개  'Li2,S1'=Li 2개+S 1개 -> 조합마다 폴더")
    print("    특정 원자 : '19'  '19,20'                                -> 그것만, _r 하나")
    print("    후보 중 k : '19,20,21,22/2'                              -> 지정 원자 중 아무 2개")
    print("    * 개수를 꼭 적어 주세요 ('S' 는 안 됩니다). 어떤 원소를 전부 없애는 것은")
    print("      surface 쪽 역할입니다.")

    replace_indices = [i for i, s in enumerate(symbols) if s in replace_set]

    while True:
        try:
            answer = input("\n* 제거할 대상을 입력해 주세요 (n=_r 트윈 안 만듦)\n: ").strip()
        except EOFError:
            print("\n[redox] 입력이 없어 _r 트윈을 만들지 않습니다.")
            return None
        if answer.lower() in ("n", "no", ""):
            print("[redox] _r 트윈을 만들지 않습니다.")
            return None

        try:
            index_sets, _pool, choose = parse_redox_spec(
                parent, answer, replace_sites=replace_indices
            )
        except ValueError as exc:
            print(f"[입력 오류] {exc}")
            continue

        n_sets = len(index_sets)
        if n_sets > max_sets:
            print(f"[입력 오류] 조합이 {n_sets}개로 상한({max_sets})을 넘습니다. "
                  "제거 개수를 줄이거나 후보를 좁혀 주세요.")
            continue

        def _desc(indices):
            return " ".join(f"#{i + 1}{symbols[i]}" for i in indices)

        if n_sets == 1:
            print(f"  -> {_desc(index_sets[0])} 제거, _r 폴더 1개")
        else:
            print(f"  -> 구조마다 {choose}개 제거, 조합 {n_sets}가지 -> _r1 .. _r{n_sets}")
            preview = index_sets if n_sets <= 10 else index_sets[:10]
            for n, indices in enumerate(preview, start=1):
                print("     _r%-3d <- %s" % (n, _desc(indices)))
            if n_sets > len(preview):
                print(f"     ... (총 {n_sets}개)")

        try:
            confirm = input("* 이대로 진행할까요? (엔터=예 / 다시 입력하려면 아무 값 / n=안 만듦)\n: ").strip()
        except EOFError:
            confirm = ""
        if not confirm:
            return answer
        if confirm.lower() in ("n", "no"):
            print("[redox] _r 트윈을 만들지 않습니다.")
            return None


# -----------------------------------------------------------------------------
# CCpy VASP input generation (INCAR / KPOINTS / POTCAR via CCpy.VASP.VASPio)
# -----------------------------------------------------------------------------

def _vasp_input_fingerprint(poscar_path_or_text, is_text=False):
    """
    Return what makes two structures interchangeable as far as INCAR / KPOINTS /
    POTCAR are concerned: the lattice, the species sequence and the atom counts.

    - KPOINTS comes from a reciprocal density on the cell  -> lattice
    - POTCAR is one file per species, concatenated in POSCAR order,
      and ENCUT is the largest ENMAX among them             -> species sequence
    - MAGMOM / LDAU lists are written per species block     -> species + counts

    Anything else in a POSCAR (the coordinates) does not enter those three
    files. Returns None when the POSCAR cannot be read/parsed.
    """
    try:
        if is_text:
            text = poscar_path_or_text
        else:
            with open(poscar_path_or_text) as handle:
                text = handle.read()
        lines = text.splitlines()
        scale = float(lines[1].split()[0])
        lattice = tuple(
            tuple(round(float(v) * scale, 6) for v in lines[row].split()[:3])
            for row in (2, 3, 4)
        )
        species = tuple(lines[5].split())
        counts = tuple(int(v) for v in lines[6].split())
    except (OSError, IndexError, ValueError):
        return None
    if not species or len(species) != len(counts):
        return None
    return lattice, species, counts


def _write_reused_vasp_inputs(cif, template_dir, template_fp):
    """
    Write one structure's VASP inputs by reusing `template_dir`'s INCAR,
    KPOINTS and POTCAR, writing only its own POSCAR.

    Returns False without touching anything when the structure's fingerprint
    does not match the template's - the caller then builds it from scratch.
    Doing the check here (rather than trusting that the generator always keeps
    the cell and composition fixed) is what makes the shortcut safe: a
    mismatched POTCAR would be silently wrong rather than loudly broken.
    """
    from pymatgen.core import Structure

    try:
        structure = Structure.from_file(cif)
    except Exception as exc:
        print(f"[CCpy VASP] {cif}: cannot read ({exc}); generating it separately.")
        return False
    poscar_text = str(structure.to(fmt="poscar"))
    if _vasp_input_fingerprint(poscar_text, is_text=True) != template_fp:
        return False

    dirname = cif[:-4] if cif.lower().endswith(".cif") else cif + "_vasp"
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "POSCAR"), "w") as handle:
        handle.write(poscar_text)
    for name in ("KPOINTS", "POTCAR"):
        shutil.copyfile(os.path.join(template_dir, name),
                        os.path.join(dirname, name))

    # INCAR is identical to the template's except for SYSTEM, which VASPio sets
    # to the folder name. Rewrite just that value and keep the column layout
    # (structure ids are fixed width, so the file stays byte-for-byte aligned).
    with open(os.path.join(template_dir, "INCAR")) as handle:
        incar_lines = handle.read().splitlines(True)
    out_lines = []
    for line in incar_lines:
        key, sep, value_field = line.partition("=")
        if sep and key.strip() == "SYSTEM":
            body = value_field.rstrip("\n")
            current = body.strip()
            newline = "\n" if value_field.endswith("\n") else ""
            body = body.replace(current, dirname, 1) if current else " " + dirname
            line = key + sep + body + newline
        out_lines.append(line)
    with open(os.path.join(dirname, "INCAR"), "w") as handle:
        handle.writelines(out_lines)

    # Mirror VASPio: the source .cif is filed away under structures/.
    if not os.path.isdir("structures"):
        os.mkdir("structures")
    os.replace(cif, os.path.join("structures", os.path.basename(cif)))
    return True


def generate_ccpy_vasp_inputs(
    output_dir,
    preset=None,
    kpoints=False,
    functional="PBE_54",
    pseudo=None,
    single_point=False,
    isif=False,
    vdw=False,
    spin=False,
    mag=False,
    ldau=False,
    batch=False,
    reuse_incar_from=None,
    reuse_inputs=True,
):
    """
    Convert every generated .cif in `output_dir` into a full CCpy VASP input
    folder (POSCAR / INCAR / KPOINTS / POTCAR), using the same VASPInput
    machinery as CCpyVASPInputGen.py.

    Behaviour mirrors `CCpyVASPInputGen.py 1`:
    - The first structure is processed interactively (INCAR review menu),
      unless `batch=True` or `reuse_incar_from` is given.
    - All remaining structures silently reuse the confirmed settings via the
      `.prev_incar.yaml` written by the first call.
    - Each source .cif is moved into `output_dir/structures/` by VASPInput,
      and the inputs are written to `output_dir/<structure_id>/`.

    Parameters largely map 1:1 to the CCpyVASPInputGen.py sub-options:
    preset (-preset, name of a yaml in the CCpy config folder's vasp/, '.yaml'
    optional; vasp_preset_dir_label() prints the resolved path),
    kpoints (-kp, list like [4, 4, 2]), functional (-pot), pseudo (-pseudo,
    list), single_point (-sp), isif (-isif), vdw (-vdw), spin (-spin),
    mag (-mag), ldau (-ldau), batch (-batch).

    reuse_incar_from: path to an existing .prev_incar.yaml (e.g. from the
    main output dir) to apply to every structure without any interaction -
    used for the _surface / _r twin directories.

    reuse_inputs: reuse the first structure's INCAR / KPOINTS / POTCAR for the
    rest of this folder instead of rebuilding them (see the comment on the
    loop below). True by default; -no_reuse turns it off.

    Returns the absolute path of the `.prev_incar.yaml` holding the settings
    that were applied (or None if no .cif files were found).
    """
    # Imported lazily so that pure structure generation keeps working in
    # environments where pymatgen / CCpy config is not available.
    from CCpy.VASP.VASPio import VASPInput
    import yaml as _yaml

    def _normalize_prev_incar(path=".prev_incar.yaml"):
        """
        VASPio dumps `.prev_incar.yaml` from an OrderedDict, which PyYAML
        serializes with a `!!python/object/apply:collections.OrderedDict`
        tag. PyYAML >= 5.1's FullLoader (used by VASPio to read it back)
        refuses that tag, so the reuse path dies on the second structure.
        Rewrite the file as a plain (order-preserving) mapping right after
        each dump so the next FullLoader call can read it.
        """
        if not os.path.isfile(path):
            return
        try:
            with open(path) as handle:
                data = _yaml.load(handle, Loader=_yaml.Loader)
            with open(path, "w") as handle:
                _yaml.dump(dict(data), handle, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            print(f"[CCpy VASP] warning: could not normalize {path}: {exc}")

    preset_yaml = None
    if preset:
        preset_yaml = preset if str(preset).endswith(".yaml") else str(preset) + ".yaml"

    if not os.path.isdir(output_dir):
        print(f"[CCpy VASP] output directory not found: {output_dir}")
        return None

    pwd = os.getcwd()
    os.chdir(output_dir)
    try:
        cifs = sorted(f for f in os.listdir(".") if f.lower().endswith(".cif"))
        if not cifs:
            print(f"[CCpy VASP] no .cif structure files found in: {output_dir}")
            print("[CCpy VASP] (already converted? source files live in ./structures/ after conversion)")
            return None

        n_total = len(cifs)
        print(f"\n[CCpy VASP] Generating VASP inputs for {n_total} structures in: {output_dir}")

        if reuse_incar_from:
            reuse_src = os.path.join(pwd, reuse_incar_from) if not os.path.isabs(reuse_incar_from) else reuse_incar_from
            if os.path.abspath(reuse_src) != os.path.abspath(".prev_incar.yaml"):
                shutil.copy(reuse_src, ".prev_incar.yaml")
            _normalize_prev_incar()

        interactive_first = not (batch or reuse_incar_from)
        template_dir = None          # folder whose INCAR/KPOINTS/POTCAR we reuse
        template_fp = None           # its (lattice, species, counts) fingerprint
        reused = 0
        fell_back = 0
        for i, cif in enumerate(cifs):
            # -- Fast path: every structure in one output folder is the same
            #    cell with the same composition, only the substitution sites
            #    are decorated differently. KPOINTS (from the cell), POTCAR
            #    (from the species sequence) and INCAR (apart from its SYSTEM
            #    line) therefore come out byte-identical for all of them, so
            #    rebuilding them per structure only repeats a POTCAR
            #    read+validate, a k-point density calculation and a full
            #    VASPInput construction. Reuse the first structure's three
            #    files instead and write only POSCAR.
            #
            #    The twins get their own template because they run through
            #    their own generate_ccpy_vasp_inputs() call - that is what
            #    keeps _surface / _r on their own composition (and their own
            #    POTCAR-derived ENCUT) rather than the main folder's.
            if reuse_inputs and template_dir is not None:
                if _write_reused_vasp_inputs(cif, template_dir, template_fp):
                    reused += 1
                    if (i + 1) % 50 == 0:
                        print(f"[CCpy VASP] {i + 1}/{n_total} done")
                    continue
                # Fingerprint mismatch: this structure is not interchangeable
                # with the template after all. Build it from scratch.
                fell_back += 1
                print(f"[CCpy VASP] {cif}: cell/composition differs from the "
                      f"first structure - generating its own inputs.")

            VI = VASPInput(cif, preset_yaml=preset_yaml)
            if i == 0 and not reuse_incar_from:
                # First structure: interactive INCAR confirm (like
                # CCpyVASPInputGen.py 1), or silent when -batch.
                VI.cms_vasp_set(
                    single_point=single_point, isif=isif, vdw=vdw,
                    spin=spin, mag=mag, ldau=ldau,
                    functional=functional, pseudo=pseudo, kpoints=kpoints,
                    get_pre_incar=None, batch=(not interactive_first),
                )
                if n_total > 1:
                    how = ("reusing its INCAR/KPOINTS/POTCAR for"
                           if reuse_inputs else
                           "applying the same INCAR/KPOINTS settings to")
                    print(f"[CCpy VASP] {how} the remaining {n_total - 1} structures...")
            else:
                VI.cms_vasp_set(
                    single_point=single_point, isif=isif, vdw=vdw,
                    spin=spin, mag=mag, ldau=ldau,
                    functional=functional, pseudo=pseudo, kpoints=kpoints,
                    get_pre_incar=".prev_incar.yaml", batch=True,
                )
            # cms_vasp_set re-dumps .prev_incar.yaml after every structure;
            # keep it FullLoader-readable for the next iteration.
            _normalize_prev_incar()

            if reuse_inputs and template_dir is None:
                candidate = VI.dirname
                fingerprint = _vasp_input_fingerprint(
                    os.path.join(candidate, "POSCAR"))
                if fingerprint and all(
                        os.path.isfile(os.path.join(candidate, name))
                        for name in ("INCAR", "KPOINTS", "POTCAR")):
                    template_dir, template_fp = candidate, fingerprint
                else:
                    print("[CCpy VASP] could not read the first structure's inputs; "
                          "generating every structure separately.")
                    reuse_inputs = False

            if (i + 1) % 50 == 0:
                print(f"[CCpy VASP] {i + 1}/{n_total} done")

        print(f"[CCpy VASP] Done. {n_total} VASP input folders written under: {output_dir}")
        if reused:
            print(f"[CCpy VASP] {reused} of them reused {os.path.basename(template_dir)}'s "
                  f"INCAR/KPOINTS/POTCAR (-no_reuse to build each one separately)."
                  + (f" {fell_back} needed their own." if fell_back else ""))
        return os.path.abspath(".prev_incar.yaml")
    finally:
        os.chdir(pwd)


# -----------------------------------------------------------------------------
# Interactive wizard
# -----------------------------------------------------------------------------

def _ask_text(prompt, default=None, required=True):
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        try:
            ans = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            # Piped/redirected stdin ran out (e.g. `printf ... | CCpyAlloyGen.py`).
            # Exit cleanly instead of dumping a traceback.
            print("\n입력이 끝났습니다. 실행을 취소합니다.")
            raise SystemExit(1)
        if not ans and default not in (None, ""):
            return default
        if ans:
            return ans
        if not required:
            return ans
        print("값을 입력해 주세요.")


def _ask_int(prompt, default=None, min_value=None, max_value=None):
    while True:
        ans = _ask_text(prompt, str(default) if default is not None else None)
        try:
            value = int(ans)
        except ValueError:
            print("정수를 입력해 주세요.")
            continue
        if min_value is not None and value < min_value:
            print(f"{min_value} 이상이어야 합니다.")
            continue
        if max_value is not None and value > max_value:
            print(f"{max_value} 이하여야 합니다.")
            continue
        return value


def _ask_float(prompt, default=None, min_value=None):
    while True:
        ans = _ask_text(prompt, str(default) if default is not None else None)
        try:
            value = float(ans)
        except ValueError:
            print("숫자를 입력해 주세요. 예: 1e-3")
            continue
        if min_value is not None and value < min_value:
            print(f"{min_value} 이상이어야 합니다.")
            continue
        return value


def _ask_yes_no(prompt, default=False):
    default_str = "y" if default else "n"
    while True:
        ans = _ask_text(prompt + " (y/n)", default_str).lower()
        if ans in ("y", "yes", "예", "네"):
            return True
        if ans in ("n", "no", "아니오", "아니요"):
            return False
        print("y 또는 n으로 입력해 주세요.")


def _ask_choice(prompt, choices, default=None):
    """choices: list of (key, label, value)."""
    print(prompt)
    for key, label, _ in choices:
        print(f"  [{key}] {label}")
    valid = {str(key): value for key, _, value in choices}
    labels = {str(key): label for key, label, _ in choices}
    while True:
        ans = _ask_text("선택", str(default) if default is not None else None)
        if ans in valid:
            print(f"선택됨: {labels[ans]}")
            return valid[ans]
        print("목록에 있는 번호를 입력해 주세요.")

def _find_base_structure_files():
    """현재 디렉터리에서 *_base_*.cif 구조 파일을 검색한다."""
    base_files = []

    for filename in os.listdir("."):
        if not os.path.isfile(filename):
            continue

        lower_name = filename.lower()

        if "_base_" in lower_name and lower_name.endswith(".cif"):
            base_files.append(filename)

    return sorted(base_files)


def _detect_base_element(input_file):
    """
    CIF 구조에서 base 원소를 자동 인식한다.

    원칙:
    1. 구조에 원소가 하나뿐이면 해당 원소를 선택
    2. 여러 원소가 있으면 가장 많이 존재하는 원소를 base 원소로 선택
    """
    atoms = read(input_file)
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)

    if not counts:
        raise ValueError(f"구조에서 원소를 읽을 수 없습니다: {input_file}")

    # 가장 많이 존재하는 원소
    sorted_counts = counts.most_common()

    if len(sorted_counts) > 1 and sorted_counts[0][1] == sorted_counts[1][1]:
        raise ValueError(
            "가장 많은 원소가 둘 이상이라 base 원소를 자동으로 결정할 수 없습니다. "
            f"구조 조성: {dict(counts)}"
        )

    base_element = sorted_counts[0][0]
    base_count = sorted_counts[0][1]

    return atoms, base_element, base_count, counts


def _select_base_structure():
    """현재 디렉터리의 base CIF 구조를 번호로 선택한다."""
    while True:
        base_files = _find_base_structure_files()

        if not base_files:
            raise FileNotFoundError(
                "현재 디렉터리에서 '*_base_*.cif' 파일을 찾지 못했습니다."
            )

        structure_info = []

        print("\n현재 디렉터리에서 발견된 base 구조:\n")

        for number, filename in enumerate(base_files, start=1):
            try:
                atoms, base_element, base_count, counts = _detect_base_element(
                    filename
                )

                structure_info.append(
                    {
                        "filename": filename,
                        "base_element": base_element,
                        "base_count": base_count,
                        "counts": counts,
                        "n_atoms": len(atoms),
                        "error": None,
                    }
                )

                print(
                    f"  [{number}] {filename:<24} "
                    f"→ base 원소: {base_element:<2}, "
                    f"원자 수: {len(atoms)}"
                )

            except Exception as exc:
                structure_info.append(
                    {
                        "filename": filename,
                        "error": str(exc),
                    }
                )

                print(
                    f"  [{number}] {filename:<24} "
                    f"→ 읽기 오류: {exc}"
                )

        print("  [0] 종료")

        answer = input("\n사용할 base 구조 선택: ").strip()

        if answer.lower() in {"0", "q", "quit", "exit"}:
            raise SystemExit("사용자가 구조 선택을 취소했습니다.")

        try:
            selected_number = int(answer)
        except ValueError:
            print("목록에 표시된 번호를 입력해 주세요.")
            continue

        if not 1 <= selected_number <= len(structure_info):
            print("올바른 번호를 입력해 주세요.")
            continue

        selected = structure_info[selected_number - 1]

        if selected.get("error"):
            print("해당 구조는 정상적으로 읽히지 않아 선택할 수 없습니다.")
            continue

        input_file = selected["filename"]
        replace_element = selected["base_element"]
        n_sites = selected["base_count"]

        print("\n[선택된 base 구조]")
        print(f"  입력 파일        : {input_file}")
        print(f"  자동 인식 원소   : {replace_element}")
        print(f"  전체 조성        : {dict(selected['counts'])}")
        print(f"  치환 가능 자리 수: {n_sites}")

        confirm = input("\n이 구조를 사용하시겠습니까? (y/n): ").strip().lower()

        if confirm in {"y", "yes", "예", "네", ""}:
            return input_file, replace_element

        print("구조를 다시 선택합니다.")

def _read_structure_file(filename):
    """
    CIF, VASP, POSCAR, CONTCAR 구조를 파일명에 맞춰 읽는다.

    POSCAR/CONTCAR 계열은 확장자가 없거나 변형된 이름이어도
    ASE에 VASP 형식임을 명시한다.
    """
    basename = os.path.basename(filename)
    upper_name = basename.upper()
    lower_name = basename.lower()

    is_poscar = (
        upper_name == "POSCAR"
        or upper_name.startswith("POSCAR.")
    )

    is_contcar = (
        upper_name == "CONTCAR"
        or upper_name.startswith("CONTCAR.")
    )

    if is_poscar or is_contcar:
        return read(filename, format="vasp")

    if lower_name.endswith(".vasp"):
        return read(filename, format="vasp")

    if lower_name.endswith(".cif"):
        return read(filename, format="cif")

    return read(filename)


_SKIP_BROWSE_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def _looks_like_vasp_template_dir(path):
    """INCAR나 KPOINTS가 있으면 VASP 템플릿 폴더 후보로 판단한다."""
    try:
        names = {n.upper() for n in os.listdir(path)}
    except OSError:
        return False
    return "INCAR" in names or "KPOINTS" in names


def _find_template_dir_candidates(start_dir=".", max_depth=3, max_scanned=5000):
    """
    start_dir 아래(제한된 깊이까지)에서 INCAR 파일이 있는 폴더를 찾는다.

    .git, __pycache__, node_modules, venv 같은 폴더는 건너뛰고,
    너무 큰 디렉터리 트리에서 오래 걸리지 않도록 스캔 개수에 상한을 둔다.
    """
    start_dir = os.path.abspath(start_dir)
    start_depth = start_dir.rstrip(os.sep).count(os.sep)
    candidates = []
    scanned = 0

    for root, dirs, files in os.walk(start_dir):
        scanned += 1
        if scanned > max_scanned:
            break

        depth = root.rstrip(os.sep).count(os.sep) - start_depth
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in _SKIP_BROWSE_DIR_NAMES and not d.startswith(".")]

        upper_files = {f.upper() for f in files}
        if "INCAR" in upper_files:
            candidates.append(root)

    return sorted(candidates)


def _browse_for_directory(start_dir="."):
    """
    번호로 폴더를 오르내리며 고르는 간단한 탐색기.

    하위 폴더 번호를 입력하면 그 폴더로 들어가고, '..'를 입력하면 상위 폴더로
    나가며, '0'이나 빈 입력은 현재 폴더를 그대로 선택한다. 경로를 직접 입력해도
    된다(탭 자동완성이 안 되는 환경에서도 번호만으로 탐색 가능).
    """
    current = os.path.abspath(start_dir)
    while True:
        print(f"\n[현재 폴더] {current}")
        try:
            entries = sorted(
                e for e in os.listdir(current)
                if os.path.isdir(os.path.join(current, e)) and not e.startswith(".")
            )
        except OSError as exc:
            print(f"폴더를 읽을 수 없습니다: {exc}")
            entries = []

        for number, name in enumerate(entries, start=1):
            sub = os.path.join(current, name)
            hint = " (INCAR/KPOINTS 있음)" if _looks_like_vasp_template_dir(sub) else ""
            print(f"  [{number}] {name}/{hint}")

        here_hint = " (INCAR/KPOINTS 있음)" if _looks_like_vasp_template_dir(current) else ""
        print(f"  [0] 현재 폴더를 템플릿 폴더로 선택: {current}{here_hint}")
        print("  [..] 상위 폴더로 이동")
        print("  (번호 대신 경로를 직접 입력해도 됩니다)")

        answer = input("\n선택 또는 경로 입력: ").strip()

        if answer in ("0", ""):
            return current
        if answer == "..":
            parent = os.path.dirname(current)
            current = parent if parent and parent != current else current
            continue
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(entries):
                current = os.path.join(current, entries[idx - 1])
                continue
            print("올바른 번호를 입력해 주세요.")
            continue

        typed = os.path.expanduser(os.path.expandvars(answer))
        if os.path.isdir(typed):
            current = os.path.abspath(typed)
            continue
        print(f"폴더를 찾을 수 없습니다: {answer}")


def _select_template_dir():
    """
    INCAR가 있는 폴더를 자동으로 찾아 번호로 고르거나, 직접 탐색/입력한다.
    """
    print("\nINCAR 파일이 있는 폴더를 현재 위치 기준으로 검색 중...")
    candidates = _find_template_dir_candidates(".", max_depth=3)

    if candidates:
        print(f"INCAR가 있는 폴더 {len(candidates)}개를 찾았습니다:")
        for number, path in enumerate(candidates, start=1):
            print(f"  [{number}] {path}")
        print("  [0] 직접 폴더 탐색/경로 입력")

        answer = _ask_text("템플릿 폴더 선택 (번호 또는 0)", "1")
        if answer.strip().isdigit():
            idx = int(answer.strip())
            if idx == 0:
                template_dir = _browse_for_directory(".")
            elif 1 <= idx <= len(candidates):
                template_dir = candidates[idx - 1]
            else:
                print("목록에 없는 번호입니다. 직접 탐색으로 전환합니다.")
                template_dir = _browse_for_directory(".")
        else:
            typed = os.path.expanduser(os.path.expandvars(answer.strip()))
            template_dir = typed if typed else _browse_for_directory(".")
    else:
        print("현재 위치 하위에서 INCAR 파일을 찾지 못했습니다. 폴더를 직접 탐색해 주세요.")
        template_dir = _browse_for_directory(".")

    print(f"선택된 템플릿 폴더: {template_dir}")
    return template_dir

def _find_structure_files():
    """
    현재 디렉터리에서 ASE가 읽을 수 있는 일반적인 구조 파일을 찾는다.

    감지 대상:
      - *.cif
      - *.vasp
      - POSCAR
      - CONTCAR
      - POSCAR.*
      - CONTCAR.*
    """
    structure_files = []

    for filename in os.listdir("."):
        if not os.path.isfile(filename):
            continue

        lower_name = filename.lower()
        upper_name = filename.upper()

        is_cif = lower_name.endswith(".cif")
        is_vasp = lower_name.endswith(".vasp")
        is_poscar = upper_name == "POSCAR" or upper_name.startswith("POSCAR.")
        is_contcar = upper_name == "CONTCAR" or upper_name.startswith("CONTCAR.")

        if is_cif or is_vasp or is_poscar or is_contcar:
            structure_files.append(filename)

    return sorted(structure_files)


def _select_structure_file():
    """현재 디렉터리의 구조 파일을 표시하고 번호로 선택한다."""
    while True:
        structure_files = _find_structure_files()

        if not structure_files:
            raise FileNotFoundError(
                "현재 디렉터리에서 구조 파일을 찾지 못했습니다.\n"
                "지원 파일: *.cif, *.vasp, POSCAR, CONTCAR"
            )

        valid_files = []

        print("\n현재 디렉터리에서 발견된 구조 파일:\n")

        for filename in structure_files:
            try:
                atoms = _read_structure_file(filename)
                counts = Counter(atoms.get_chemical_symbols())

                valid_files.append(
                    {
                        "filename": filename,
                        "atoms": atoms,
                        "counts": counts,
                    }
                )

                number = len(valid_files)

                composition_text = " ".join(
                    f"{element}{count}"
                    for element, count in counts.items()
                )

                print(
                    f"  [{number}] {filename:<25} "
                    f"조성: {composition_text:<20} "
                    f"원자 수: {len(atoms)}"
                )

            except Exception as exc:
                print(f"  [읽기 실패] {filename}: {exc}")

        if not valid_files:
            raise ValueError(
                "구조 파일 후보는 발견했지만 ASE로 읽을 수 있는 파일이 없습니다."
            )

        print("  [0] 종료")

        answer = input("\n사용할 구조 파일 선택: ").strip()

        if answer.lower() in {"0", "q", "quit", "exit"}:
            raise SystemExit("사용자가 구조 파일 선택을 취소했습니다.")

        try:
            selected_number = int(answer)
        except ValueError:
            print("목록에 표시된 번호를 입력해 주세요.")
            continue

        if not 1 <= selected_number <= len(valid_files):
            print("올바른 번호를 입력해 주세요.")
            continue

        selected = valid_files[selected_number - 1]
        input_file = selected["filename"]
        atoms = selected["atoms"]
        counts = selected["counts"]

        print("\n[선택된 입력 구조]")
        print(f"  파일명   : {input_file}")
        print(f"  전체 조성: {dict(counts)}")
        print(f"  원자 수  : {len(atoms)}")

        return input_file, atoms, counts

METAL_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Sn", "Pb", "Bi",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

def _select_replace_element(counts):
    """
    입력 구조에서 치환 대상 원소 풀(pool)을 선택한다.

    원자 수가 가장 많은 원소를 기판 원소 후보로 먼저 표시한다.
    Li2S2처럼 소수 원자로 존재하는 흡착종 원소는 별도로 구분한다.

    번호를 하나만 입력하면 기존처럼 단일 원소를 치환 대상으로 선택하고,
    콤마로 여러 번호를 입력하면(예: 1,3,4) 이미 여러 금속이 섞여 있는
    HEA 구조에서도 그 원소들을 하나의 치환 대상 풀로 묶어서 선택할 수 있다.
    이렇게 고르면 Li, S 같이 목록에서 선택하지 않은 흡착물 원소는
    자동으로 치환 대상에서 제외된다.
    """
    elements = sorted(
        counts.keys(),
        key=lambda element: (-counts[element], element),
    )

    if len(elements) == 1:
        replace_element = elements[0]

        print(
            f"\n구조에 원소가 하나만 있으므로 "
            f"{replace_element}를 치환 기준 원소로 자동 선택합니다."
        )
        return [replace_element]

    while True:
        print("\n치환할 원소를 선택하세요 (여러 원소를 하나의 풀로 묶으려면 콤마로 구분, 예: 1,3,4):\n")

        max_count = max(counts.values())

        for number, element in enumerate(elements, start=1):
            label = "기판 후보" if counts[element] == max_count else "유지 후보"

            print(
                f"  [{number}] {element:<2} "
                f"({counts[element]}개 자리, {label})"
            )

        print("  [0] 구조 파일 다시 선택")
        print("  [q] 종료")

        answer = input("\n선택 (예: 2 또는 1,3,4): ").strip()

        if answer.lower() in {"q", "quit", "exit"}:
            raise SystemExit("사용자가 입력을 취소했습니다.")

        if answer == "0":
            return None

        tokens = [t for t in re.split(r"[,\s]+", answer) if t]
        try:
            numbers = [int(t) for t in tokens]
        except ValueError:
            print("목록에 표시된 번호를 콤마로 구분해 입력해 주세요.")
            continue

        if not numbers or any(not (1 <= n <= len(elements)) for n in numbers):
            print("올바른 번호를 입력해 주세요.")
            continue

        if len(set(numbers)) != len(numbers):
            print("같은 원소를 두 번 선택했습니다. 다시 입력해 주세요.")
            continue

        replace_elements = [elements[n - 1] for n in numbers]
        total_sites = sum(counts[e] for e in replace_elements)

        print(
            f"선택된 치환 대상 원소 풀: {', '.join(replace_elements)} "
            f"(합계 {total_sites}개 자리)"
        )

        return replace_elements

    
def _select_input_structure():
    """구조 파일을 고르고 치환 대상 원소 풀까지 결정한다."""
    while True:
        input_file, atoms, counts = _select_structure_file()
        replace_elements = _select_replace_element(counts)

        if replace_elements is None:
            print("\n구조 파일 선택 단계로 돌아갑니다.")
            continue

        return input_file, replace_elements


def _preview_input_structure(input_file, replace_elements, composition=None):
    """
    입력 구조와 치환 대상 원소 풀을 검증하고 요약을 출력한다.

    composition을 None으로 두면(기존 조성 재사용 모드) 현재 구조에서
    replace_elements 자리들의 실제 원소 비율을 그대로 target 조성으로
    자동 계산한다. 이 경우 반환되는 composition으로 실제 사용된 값을
    확인할 수 있다.
    """
    parent = _read_structure_file(input_file)

    counts = Counter(parent.get_chemical_symbols())
    symbols = parent.get_chemical_symbols()

    replace_set = set(replace_elements)
    replace_sites = [
        i for i, symbol in enumerate(symbols)
        if symbol in replace_set
    ]

    if not replace_sites:
        raise ValueError(f"치환 대상 원소 풀 {replace_elements}에 해당하는 자리가 없습니다.")

    if composition is None:
        composition = dict(Counter(symbols[i] for i in replace_sites))

    n_replace = sum(composition.values())

    preserved_counts = {
        element: count
        for element, count in counts.items()
        if element not in replace_set
    }

    expected_counts = Counter(preserved_counts)
    expected_counts.update(composition)

    print("\n[입력 구조 확인]")
    print(f"  전체 조성          : {dict(counts)}")
    print(f"  치환 대상 원소 풀  : {replace_elements}")
    print(f"  치환 대상 자리 수  : {len(replace_sites)}")
    print(f"  입력 HEA 조성      : {composition}")
    print(f"  치환 조성 합       : {n_replace}")
    print(f"  유지되는 원소      : {preserved_counts}")
    print(f"  예상 최종 조성     : {dict(expected_counts)}")

    if len(replace_sites) != n_replace:
        raise ValueError(
            "치환 대상 원소 풀 전체를 치환하려면 치환 조성 합이 "
            "치환 대상 자리 수와 같아야 합니다.\n"
            f"  치환 대상 자리 수 = {len(replace_sites)}\n"
            f"  치환 조성 합 = {n_replace}"
        )

    return parent, len(replace_sites), n_replace, composition


def _suggest_output_dir(input_file, mode, composition, layer_axis="z", view_axis="z"):
    """Build a readable output-folder suggestion after all key settings are known."""
    input_dir = os.path.dirname(str(input_file))
    input_stem = os.path.splitext(os.path.basename(str(input_file)))[0]
    composition_tag = "".join(f"{el}{count}" for el, count in composition.items())
    if mode == "layered":
        mode_tag = f"layered_{layer_axis}"
    elif mode == "domain":
        mode_tag = f"domain_{view_axis}"
    else:
        mode_tag = mode
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{input_stem}_{mode_tag}_{composition_tag}_output",
    ).strip("_.")
    return os.path.join(input_dir, safe_name) if input_dir else safe_name


def _wizard_output_dir(input_file, suggested):
    """Ask for an output directory and reject file/input-destructive choices early."""
    while True:
        output_dir = _ask_text("8) 출력 디렉토리", suggested)
        output_abs = os.path.abspath(os.path.realpath(output_dir))
        input_abs = os.path.abspath(os.path.realpath(input_file))
        if output_abs == input_abs or os.path.isfile(output_abs):
            print(f"[입력 오류] 출력 디렉토리는 파일이 될 수 없습니다: {output_dir}")
            print("새 폴더 이름을 입력해 주세요.")
            continue
        if _is_same_or_inside(input_abs, output_abs):
            print("[입력 오류] 출력 디렉토리 안에 입력 구조 파일이 포함됩니다.")
            print("입력 파일과 별도의 새 폴더를 지정해 주세요.")
            continue
        return output_dir


def run_wizard(initial=None):
    """
    One-screen settings sheet (replaces the old step-by-step wizard).

    `initial` pre-fills sheet keys from the command line, so
    `CCpyAlloyGen.py 4 -n=100` opens the sheet already set to domain mode with
    100 structures. Pre-filled keys count as user input, so the
    structure-derived defaults never overwrite them.

    Every setting is shown at once with its current value. Edit any of them
    with `key=value` (several edits can be chained with commas, e.g.
    `mode=domain,n=100`), then press Enter or type `run` to validate and
    execute. `q` quits without running.

    Hidden advanced keys are accepted the same way even though they are not
    displayed: max_attempts, children, limit, order_tol, order_steps,
    sro_cutoff, sro_tol, sro_weight, bucket, batch, reuse, sp, isif, spin,
    mag, ldau, vdw, pot, pseudo, template, gen_potcar, potcar_lib, potcar_var.
    """
    # ------------------------------ sheet state ------------------------------
    s = {
        # structure generation (visible)
        "input": "",
        "replace": "",          # filled in from the chosen structure
        "comp": "",             # filled in from the chosen structure
        "mode": "random",
        "n": "500",
        "seed": "",
        "fmt": "folder",
        "surface": "",
        "redox": "",
        "output": "",
        "overwrite": "n",
        "symprec": "1e-3",
        # mode detail (visible)
        "axis": "z",
        "view": "z",
        "pattern": "",
        "order": "1,0.75,0.5,0.25,0",
        # CCpy VASP input generation (visible)
        "vasp": "y",
        "preset": "",
        "kp": "",
        # hidden advanced keys (editable via key=value, not displayed)
        "max_attempts": "2000000",
        "children": "",
        "limit": "2000000",
        "order_tol": "0.05",
        "order_steps": "5000",
        "sro_cutoff": "1.20",
        "sro_tol": "0.12",
        "sro_weight": "0.5",
        "bucket": "30",
        "redox_max": "50",
        "batch": "n",
        "reuse": "y",
        "sp": "n",
        "isif": "",
        "spin": "n",
        "mag": "n",
        "ldau": "n",
        "vdw": "",
        "pot": "PBE_54",
        "pseudo": "",
        "template": "",
        "gen_potcar": "n",
        "potcar_lib": "",
        "potcar_var": "",
    }

    # ---------------------- structure files & defaults ----------------------
    candidates = _find_structure_files()
    formulas = {}          # filename -> chemical formula (lazy, cached)
    detected = {}          # filename -> (substrate, adsorbate, is_slab)
    user_set = set()       # keys the user typed, so defaults never overwrite them

    def _formula(filename):
        if filename not in formulas:
            try:
                formulas[filename] = _read_structure_file(filename).get_chemical_formula()
            except Exception:
                formulas[filename] = "(읽기 실패)"
        return formulas[filename]

    def _detect(filename):
        """Substrate/adsorbate split of a structure file, cached."""
        if filename not in detected:
            try:
                detected[filename] = guess_substrate_elements(_read_structure_file(filename))
            except Exception:
                detected[filename] = ([], [], False)
        return detected[filename]

    def _apply_structure_defaults():
        """
        Seed `replace` and `comp` from the chosen structure: the replacement
        pool defaults to the substrate (adsorbates left out) and the target
        composition to what those sites currently hold. Keys the user typed
        are left alone.
        """
        filename = s["input"].strip()
        if not filename or not os.path.isfile(filename):
            return
        substrate, _adsorbate, _is_slab = _detect(filename)
        if not substrate:
            return
        if "replace" not in user_set:
            s["replace"] = ",".join(substrate)
        if "comp" not in user_set:
            try:
                parent = _read_structure_file(filename)
            except Exception:
                return
            pool = set(s["replace"].replace(" ", "").split(","))
            counts = Counter(sym for sym in parent.get_chemical_symbols() if str(sym) in pool)
            if counts:
                s["comp"] = ",".join(f"{el}{counts[el]}" for el in sorted(counts))

    for key, value in (initial or {}).items():
        if key in s and value not in (None, ""):
            s[key] = str(value)
            if key != "input":
                user_set.add(key)

    if s["input"] and os.path.isfile(s["input"]):
        _apply_structure_defaults()          # -i= was given on the command line
    elif candidates:
        # Pick the structure first, the way CCpy's input pickers do, then show
        # the settings sheet for it.
        chosen = select_structure_file_interactively()
        if not chosen:
            return
        s["input"] = chosen
        _apply_structure_defaults()

    def _bool(key):
        return s[key].strip().lower() in ("y", "yes", "true", "1")

    def _row(key, hint=""):
        print("  %-9s = %-22s %s" % (key, s[key], hint))

    def _print_sheet():
        print("\n" + "=" * 74)
        print(" CCpyAlloyGen settings")
        print("=" * 74)
        _row("input", "# 구조 파일 ('input=?' 로 목록에서 다시 선택)")
        _row("replace", "# 치환 풀 원소 (기본=흡착물 제외한 기판 원소)")
        _row("comp", "# 목표 조성 (기본=현재 조성) / 'keep'=기존 조성 재셔플")
        _row("mode", "# random/spread/layered/domain/exhaustive")
        _row("n", "# 목표 구조 수 (exhaustive면 무시)")
        _row("seed", "# 비우면 자동 생성 후 metadata.txt에 기록")
        _row("fmt", "# cif/vasp/folder" + ("  (vasp=y 이므로 최종 출력은 VASP 입력 폴더)"
                                          if _bool("vasp") else ""))
        _row("surface", "# _surface 트윈: 원소 지정 = 그 원소 '전부' 제거 (예: Li,S) / 'auto'=자동감지")
        _row("redox", "# _r 트윈: 제거 조성 '개수 필수' ('S1','S2','Li2,S1') / '35'=그 원자 / 'auto'=선택")
        _row("output", "# 비우면 자동 제안")
        _row("overwrite", "# y면 기존 출력 폴더 삭제 후 생성")
        _row("symprec", "# spglib 대칭 허용 오차")
        print("  --- mode detail (layered/domain) " + "-" * 39)
        _row("axis", "# layered 층 축 (x/y/z)")
        _row("view", "# domain top-view 축 (x/y/z)")
        _row("pattern", "# domain 패턴 (예: Co,Fe/Ni,Cu / 비우면 자동 전수)")
        _row("order", "# 목표 order parameter Q 레벨")
        print("  --- CCpy VASP inputs " + "-" * 51)
        _row("vasp", "# y면 INCAR/KPOINTS/POTCAR 자동 생성 (CCpyVASPInputGen과 동일)")
        _row("preset", "# %s*.yaml 이름 (비우면 default)" % vasp_preset_dir_label())
        _row("kp", "# 예: 4,4,1 (비우면 preset k-density 자동)")
        print("-" * 74)
        print("* 고급 키(미표시)도 key=value로 입력 가능: max_attempts, children, limit,")
        print("  order_tol, order_steps, sro_cutoff, sro_tol, sro_weight, bucket, redox_max, batch,")
        print("  reuse (n = 구조마다 INCAR/KPOINTS/POTCAR 새로 생성),")
        print("  sp, isif, spin, mag, ldau, vdw, pot, pseudo, template, gen_potcar,")
        print("  potcar_lib, potcar_var")

    def _validate_and_run():
        """Validate the sheet; run on success. Returns True when executed."""
        input_file = s["input"].strip()
        if not input_file or not os.path.isfile(input_file):
            print("[검증 실패] 구조 파일을 찾을 수 없습니다: %r  -> input= 로 지정해 주세요." % input_file)
            return False

        mode = s["mode"].strip().lower()
        if mode not in ("random", "spread", "layered", "domain", "exhaustive"):
            print("[검증 실패] mode는 random/spread/layered/domain/exhaustive 중 하나여야 합니다: %r" % s["mode"])
            return False

        try:
            replace_elements = parse_element_list(s["replace"])
        except Exception as exc:
            print("[검증 실패] replace: %s" % exc)
            return False

        keep_composition = s["comp"].strip().lower() == "keep"
        composition = None
        if not keep_composition:
            try:
                composition = parse_composition(s["comp"])
            except Exception as exc:
                print("[검증 실패] comp: %s" % exc)
                return False

        # Structure preview also validates the pool and the composition sum.
        try:
            parent, n_sites, n_replace, composition = _preview_input_structure(
                input_file, replace_elements, composition
            )
        except Exception as exc:
            print("[검증 실패] %s" % exc)
            return False

        if mode == "domain" and len(composition) not in (4, 5):
            print("[검증 실패] domain mode는 4원소(2x2) 또는 5원소(quincunx) 조성만 지원합니다. "
                  "현재 %d원소." % len(composition))
            return False

        surface = s["surface"].strip() or None
        if surface and surface.lower() in ("ask", "auto", "y", "yes"):
            # Interactive picker, with auto-detected adsorbates pre-proposed.
            surface = interactive_select_surface_elements(parent, replace_elements)
            s["surface"] = surface or ""
        elif surface:
            overlap = set(parse_element_list(surface)) & set(replace_elements)
            if overlap:
                print("[검증 실패] surface 원소가 치환 풀과 겹칩니다: %s" % sorted(overlap))
                return False

        redox_spec = None
        redox_n_sets = 0
        redox_val = s["redox"].strip()
        if redox_val and redox_val.lower() in ("ask", "auto", "y", "yes"):
            redox_spec = interactive_select_redox_atoms(
                parent, replace_elements, max_sets=int(s["redox_max"] or 50))
            s["redox"] = redox_spec or ""
            redox_val = redox_spec or ""
        if redox_val:
            # Numbers, element symbols, and the '/k' combination form are all
            # handled by parse_redox_spec(); just preview what it resolved to.
            try:
                redox_sets_preview, redox_pool_preview, redox_choose_preview = parse_redox_spec(
                    parent, redox_val,
                    replace_sites=[i for i, sym in enumerate(parent.get_chemical_symbols())
                                   if sym in set(replace_elements)],
                )
            except ValueError as exc:
                print("[검증 실패] redox: %s" % exc)
                return False
            symbols_all = parent.get_chemical_symbols()
            frac_all = parent.get_scaled_positions()
            n_sets_preview = len(redox_sets_preview)
            if n_sets_preview > 1:
                print("\n[redox] 후보 %d개 중 %d개 제거 -> 조합 %d가지 (_r1 .. _r%d)"
                      % (len(redox_pool_preview), redox_choose_preview, n_sets_preview, n_sets_preview))
                for n, indices in enumerate(redox_sets_preview, start=1):
                    print("  _r%-3d <- " % n + " ".join("#%d%s" % (i + 1, symbols_all[i]) for i in indices))
            else:
                print("\n[redox 제거 원자] 모든 생성 구조에서 아래 원자가 제거되어 _r 트윈에 저장됩니다:")
                for i in redox_sets_preview[0]:
                    print("  #%-4d %-2s  frac=(%.4f, %.4f, %.4f)"
                          % (i + 1, symbols_all[i], frac_all[i][0], frac_all[i][1], frac_all[i][2]))
            redox_spec = redox_val
            redox_n_sets = n_sets_preview

        ccpy_vasp = _bool("vasp")
        out_fmt = s["fmt"].strip().lower()
        vasp_folder = False
        output_format = out_fmt
        if ccpy_vasp:
            if out_fmt not in ("", "cif"):
                print("* vasp=y : 구조는 cif로 생성 후 VASP 입력 폴더로 변환됩니다 (fmt=%s 무시)." % out_fmt)
            output_format = "cif"
        elif out_fmt in ("folder", "poscar_folder"):
            vasp_folder = True
            output_format = "vasp"
        elif out_fmt in ("cif", "vasp", "poscar"):
            output_format = "vasp" if out_fmt == "poscar" else out_fmt
        else:
            print("[검증 실패] fmt는 cif/vasp/folder 중 하나여야 합니다: %r" % s["fmt"])
            return False

        output_dir = s["output"].strip()
        if not output_dir:
            # Ask for the folder name once the settings are settled, showing the
            # name that would be used otherwise. Empty input keeps that default.
            suggested = _suggest_output_dir(
                input_file, mode, composition, layer_axis=s["axis"], view_axis=s["view"]
            )
            try:
                answer = input("\n* folder name ? (엔터 = %s)\n: " % suggested).strip()
            except EOFError:
                answer = ""
            output_dir = answer or suggested
        output_abs = os.path.abspath(os.path.realpath(output_dir))
        input_abs = os.path.abspath(os.path.realpath(input_file))
        if output_abs == input_abs or os.path.isfile(output_abs):
            print("[검증 실패] 출력 디렉토리는 파일이 될 수 없습니다: %s" % output_dir)
            return False
        if _is_same_or_inside(input_abs, output_abs):
            print("[검증 실패] 출력 디렉토리 안에 입력 구조 파일이 포함됩니다. 다른 폴더를 지정해 주세요.")
            return False
        overwrite = _bool("overwrite")
        if os.path.isdir(output_dir) and os.listdir(output_dir) and not overwrite:
            print("[검증 실패] 출력 폴더가 비어 있지 않습니다: %s  (overwrite=y 또는 output= 변경)" % output_dir)
            return False

        try:
            target = int(s["n"])
            symprec = float(s["symprec"])
            seed = int(s["seed"]) if s["seed"].strip() else None
            max_attempts = int(s["max_attempts"])
            children = int(s["children"]) if s["children"].strip() else None
            limit = int(s["limit"])
            order_tol = float(s["order_tol"])
            order_steps = int(s["order_steps"])
            sro_cutoff = float(s["sro_cutoff"])
            sro_tol = float(s["sro_tol"])
            sro_weight = float(s["sro_weight"])
            bucket = int(s["bucket"])
            isif = int(s["isif"]) if s["isif"].strip() else False
        except ValueError as exc:
            print("[검증 실패] 숫자 값을 확인해 주세요: %s" % exc)
            return False

        if s["axis"].strip() not in ("x", "y", "z") or s["view"].strip() not in ("x", "y", "z"):
            print("[검증 실패] axis/view는 x, y, z 중 하나여야 합니다.")
            return False

        kp_tokens = [t for t in s["kp"].replace(" ", "").split(",") if t]
        kp = kp_tokens if kp_tokens else False
        pseudo_tokens = [t for t in s["pseudo"].replace(" ", "").split(",") if t]
        pseudo = pseudo_tokens if pseudo_tokens else None
        pattern = s["pattern"].strip() or None
        preset = s["preset"].strip() or None
        reuse_inputs = s["reuse"].strip().lower() not in ("n", "no", "false")

        # ----------------------------- summary -----------------------------
        print("\n[실행 설정 요약]")
        print("  input=%s  mode=%s  replace=%s" % (input_file, mode, replace_elements))
        print("  composition%s = %s" % (" (keep)" if keep_composition else "", composition))
        fmt_display = "folder+CCpy VASP inputs" if ccpy_vasp else ("folder" if vasp_folder else output_format)
        print("  output=%s  n=%s  fmt=%s  seed=%s  overwrite=%s" % (output_dir, target, fmt_display, seed, overwrite))
        if mode == "layered":
            print("  axis=%s  order=%s" % (s["axis"], s["order"]))
        if mode == "domain":
            print("  view=%s  pattern=%s  order=%s" % (s["view"], pattern, s["order"]))
        if surface:
            print("  surface=%s  ->  %s_surface" % (surface, str(output_dir).rstrip("/\\")))
        if redox_spec:
            suffix_desc = "_r" if redox_n_sets <= 1 else "_r1 .. _r%d" % redox_n_sets
            print("  redox=%s  ->  %s%s" % (redox_spec, str(output_dir).rstrip("/\\"), suffix_desc))
        if ccpy_vasp:
            print("  CCpy VASP: preset=%s  kp=%s  pot=%s" % (preset or "default", kp if kp else "(auto density)", s["pot"]))
        # No extra yes/no here: typing "n" at the sheet already meant "finish".
        result = generate_structures(
            input_file=input_file,
            output_dir=output_dir,
            replace_element=replace_elements,
            composition=composition,
            keep_composition=keep_composition,
            mode=mode,
            target=target if mode != "exhaustive" else 0,
            max_attempts=max_attempts if mode != "exhaustive" else 0,
            symprec=symprec,
            seed=seed,
            output_format=output_format,
            vasp_folder=vasp_folder,
            overwrite=overwrite,
            order_levels=s["order"] if mode in ("layered", "domain") else None,
            order_tolerance=order_tol,
            order_search_steps=order_steps,
            sro_cutoff_factor=sro_cutoff,
            sro_tolerance=sro_tol,
            sro_weight=sro_weight,
            max_trials_per_bucket=bucket,
            template_dir=s["template"].strip() or None,
            exhaustive_limit=limit,
            layer_axis=s["axis"].strip(),
            view_axis=s["view"].strip(),
            domain_pattern=pattern,
            children_per_parent=children,
            generate_potcar=_bool("gen_potcar"),
            potcar_library=s["potcar_lib"].strip() or None,
            potcar_variants=s["potcar_var"].strip() or None,
            adsorbate_elements=surface,
            redox_remove=redox_spec,
            redox_max_sets=int(s["redox_max"] or 50),
        )

        if ccpy_vasp:
            prev_incar = generate_ccpy_vasp_inputs(
                output_dir,
                preset=preset,
                kpoints=kp,
                functional=s["pot"].strip() or "PBE_54",
                pseudo=pseudo,
                single_point=_bool("sp"),
                isif=isif,
                vdw=s["vdw"].strip() or False,
                spin=_bool("spin"),
                mag=_bool("mag"),
                ldau=_bool("ldau"),
                batch=_bool("batch"),
                reuse_inputs=reuse_inputs,
            )
            # created dirs come back from generate_structures, so every _rN
            # twin is covered without guessing folder names
            twin_dirs = []
            if result and result.get("surface_dir"):
                twin_dirs.append(result["surface_dir"])
            if result:
                twin_dirs.extend(result.get("redox_dirs") or [])
            for twin_dir in twin_dirs:
                if os.path.isdir(twin_dir) and prev_incar:
                    print("\n[CCpy VASP] 쌍둥이 폴더에도 같은 설정으로 VASP 입력을 생성합니다: %s" % twin_dir)
                    generate_ccpy_vasp_inputs(
                        twin_dir,
                        preset=preset,
                        kpoints=kp,
                        functional=s["pot"].strip() or "PBE_54",
                        pseudo=pseudo,
                        single_point=_bool("sp"),
                        isif=isif,
                        vdw=s["vdw"].strip() or False,
                        spin=_bool("spin"),
                        mag=_bool("mag"),
                        ldau=_bool("ldau"),
                        reuse_incar_from=prev_incar,
                        reuse_inputs=reuse_inputs,
                    )
        return True

    # ------------------------------ main loop ------------------------------
    while True:
        _print_sheet()
        # Keep taking edits until the user says "n", the same habit as
        # CCpySIESTAInputGen / CCpyVASPInputGen's option menus.
        print('\n* Anything want to modify or add? (ex: mode=domain,n=100, comp=Fe4,Co4,Ni4,Cu4)')
        print('  else, enter "n" to finish     (q = 취소)')
        try:
            ans = input(": ").strip()
        except EOFError:
            print("\n취소했습니다.")
            return
        if ans.lower() in ("q", "quit", "exit"):
            print("취소했습니다.")
            return
        if ans.lower() in ("n", "no"):
            if _validate_and_run():
                return
            continue        # validation failed -> back to editing
        if ans == "":
            continue        # just redraw the sheet
        # key=value edits; commas split pairs only right before the next key=
        pairs = re.split(r",(?=\s*[A-Za-z_]+\s*=)", ans)
        for pair in pairs:
            if "=" not in pair:
                print("[입력 오류] `key=value` 형식이 아닙니다: %r" % pair)
                continue
            key, value = pair.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key not in s:
                print("[입력 오류] 알 수 없는 key: %r  (화면 하단 고급 키 목록 참고)" % key)
                continue
            if key == "input":
                # `input=?` reopens the file list; a number or filename also works.
                if value in ("?", "list", "목록"):
                    chosen = select_structure_file_interactively()
                    if not chosen:
                        continue
                    value = chosen
                elif value.isdigit():
                    n = int(value)
                    if not 1 <= n <= len(candidates):
                        print("[입력 오류] 구조 파일 번호는 1..%d 범위입니다: %s"
                              % (len(candidates), value))
                        continue
                    value = candidates[n - 1]
                elif not os.path.isfile(value):
                    print("[입력 오류] 그런 파일이 없습니다: %r "
                          "(목록의 번호나 정확한 파일명을 입력하세요)" % value)
                    continue
                s["input"] = value
                # A different structure means different sensible defaults.
                _apply_structure_defaults()
                continue
            s[key] = value
            user_set.add(key)


def build_argparser():
    p = argparse.ArgumentParser(
        description="Generate symmetry-unique Pt-based HEA/intermetallic substitution structures."
    )
    p.add_argument("--input", help="Input CIF/POSCAR file")
    p.add_argument("--output", help="Output directory")
    p.add_argument(
        "--replace-element",
        default="Pt",
        help=(
            "Element(s) whose sites form the replaceable pool, e.g. 'Pt' or 'Co'. "
            "Accepts a comma/space-separated list to pool several existing elements "
            "together, e.g. 'Co,Fe,Ni,Cu' when starting from an already-mixed HEA "
            "structure instead of a pure metal."
        ),
    )
    p.add_argument(
        "--composition",
        default="Fe4 Co4 Ni4 Cu4",
        help="Target replacement composition, e.g., 'Fe4 Co4 Ni4 Cu4' or 'Fe:4,Co:4,Ni:4,Cu:4'. Ignored if --keep-composition is set.",
    )
    p.add_argument(
        "--keep-composition",
        action="store_true",
        help=(
            "Reuse the current composition already present on the --replace-element "
            "pool sites instead of a new target composition. Useful for reshuffling "
            "an existing relaxed HEA structure (e.g. resampling active-site "
            "arrangements) without changing its overall composition. --composition "
            "is ignored when this is set."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["random", "spread", "layered", "domain", "exhaustive"],
        default="random",
        help="Generation mode",
    )
    p.add_argument(
        "--layer-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis used to define layers in layered mode",
    )
    p.add_argument(
        "--view-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Viewing axis for domain mode. view-axis z means x-y top view.",
    )
    p.add_argument(
        "--domain-pattern",
        default=None,
        help=(
            "Top-view pattern for domain mode. For a 4-element composition use the "
            "rectangular 2x2 form 'TL,TR/BL,BR' (e.g. 'Co,Fe/Ni,Cu'). For a "
            "5-element composition use the quincunx form 'Center:TL,TR/BL,BR' "
            "(e.g. 'Cu:Co,Fe/Ni,Ti'), where Center is placed in the middle and "
            "the other four are split into equal-count corners around it. If "
            "omitted, all valid patterns are generated and symmetry-equivalent "
            "ones are removed."
        ),
    )
    p.add_argument("--target", type=int, default=500, help="Target number of unique structures")
    p.add_argument("--max-attempts", type=int, default=2_000_000, help="Max attempts for sampling modes")
    p.add_argument("--symprec", type=float, default=1e-3, help="spglib symmetry tolerance")
    p.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, a time-based seed is generated and saved to metadata.txt")
    p.add_argument("--format", choices=["cif", "vasp", "poscar"], default="cif", help="Output format")
    p.add_argument("--vasp-folder", action="store_true", help="Write each structure to conf_XXXX/POSCAR")
    p.add_argument(
        "--template-dir",
        default=None,
        help=(
            "Optional directory to copy INCAR/KPOINTS/POTCAR from into every "
            "structure folder. Only these three exact filenames are copied; "
            "other files in that directory (e.g. from a finished calculation) "
            "are left alone."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    p.add_argument(
        "--order-levels",
        default="1,0.75,0.5,0.25,0",
        help="Target parent-overlap order parameters for layered/domain modes: 1=ordered, 0=random-like",
    )
    p.add_argument(
        "--order-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed absolute difference between target and actual Q",
    )
    p.add_argument(
        "--order-search-steps",
        type=int,
        default=5000,
        help="Internal search iterations used to reach each target Q",
    )
    p.add_argument(
        "--sro-cutoff-factor",
        type=float,
        default=1.20,
        help="First-neighbor cutoff as a multiple of the nearest selected-site distance",
    )
    p.add_argument(
        "--sro-tolerance",
        type=float,
        default=0.12,
        help="Allowed Warren-Cowley SRO RMS above the Q-scaled target",
    )
    p.add_argument(
        "--sro-weight",
        type=float,
        default=0.5,
        help="Weight of SRO reduction in target-Q candidate search",
    )
    p.add_argument(
        "--max-trials-per-bucket",
        type=int,
        default=30,
        help="Maximum candidate searches for each parent/Q quota before moving on",
    )
    p.add_argument(
        "--exhaustive-limit",
        type=int,
        default=2_000_000,
        help="Safety limit for exhaustive enumeration upper bound",
    )
    p.add_argument("--children-per-parent", type=int, default=None, help="For layered/domain modes: number of unique structures per parent and non-parent Q level. If omitted, inferred from target.")
    p.add_argument(
        "--generate-potcar",
        action="store_true",
        help=(
            "Auto-generate a composition-correct POTCAR for each structure by "
            "concatenating per-element POTCAR files from --potcar-library, using "
            "DEFAULT_POTCAR_VARIANTS unless overridden by --potcar-variants. Only "
            "applies together with --vasp-folder."
        ),
    )
    p.add_argument(
        "--potcar-library",
        default=None,
        help=(
            "Path to a potpaw_PBE-style POTCAR library. Only used with "
            "--generate-potcar. If omitted, auto-detected as whichever of "
            + " / ".join(POTCAR_LIBRARY_CANDIDATES) + " exists on this machine."
        ),
    )
    p.add_argument(
        "--potcar-variants",
        default=None,
        help=(
            "Comma-separated Element:Variant overrides on top of "
            "DEFAULT_POTCAR_VARIANTS, e.g. 'Fe:Fe_sv,Co:Co_pv'. Only used with "
            "--generate-potcar."
        ),
    )
    p.add_argument(
        "--adsorbate-elements",
        default=None,
        help=(
            "Comma/space-separated elements to strip when building a surface "
            "(adsorbate-free) twin of every generated structure (e.g. 'Li,S'). "
            "A bare element here means EVERY atom of it -- a clean surface "
            "(--redox-remove instead requires an explicit count, e.g. 'S2'). "
            "Written to a mirrored output_dir + '_surface' tree with matching "
            "structure IDs, so each pair can be used for "
            "E_ads = E(slab+ads) - E(surface) - E(ads reference). "
            "Requires an element that is not part of --replace-element's pool."
        ),
    )
    p.add_argument(
        "--redox-remove",
        default=None,
        help=(
            "Atoms to delete from every generated structure, written to mirrored "
            "output_dir + '_r' trees with matching structure IDs -- for comparing "
            "energies across a redox step. "
            "A removal composition fixes the stoichiometry of what is removed and "
            "gives one folder per distinct choice (_r1, _r2, ...). The count is "
            "mandatory: 'S1' (Li2S4 -> Li2S3) | 'S2' (-> Li2S2) | 'Li2,S1'. "
            "A bare element symbol is rejected on purpose, because "
            "--adsorbate-elements takes a bare element to mean 'strip this species "
            "entirely' -- use that option to remove all of a species. "
            "Bare atom numbers name specific atoms and form a single set: "
            "'35' | '35,36' | 'S1,35'. "
            "An explicit pool with '/k' takes any k of *those* atoms, to restrict "
            "the candidate sites: '19,20,21,22/2' | 'S/2'. "
            "Atom numbers are 1-based in the input file's coordinate order and must "
            "not belong to --replace-element's pool."
        ),
    )
    p.add_argument(
        "--redox-max-sets",
        type=int,
        default=50,
        help=(
            "Safety limit on the number of redox combinations (folders) that "
            "'/k' may expand to. Structure count x combination count folders are "
            "created, so raise this deliberately."
        ),
    )
    p.add_argument("--wizard", action="store_true", help="Start interactive question-answer mode")
    return p


def main():
    args = build_argparser().parse_args()

    # No arguments or explicit --wizard -> interactive mode.
    # This keeps the script easy for first-time users while preserving CLI usage.
    if args.wizard or (args.input is None and args.output is None):
        run_wizard()
        return

    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required in command-line mode. Use --wizard for interactive mode.")

    composition = None if args.keep_composition else parse_composition(args.composition)
    generate_structures(
        input_file=args.input,
        output_dir=args.output,
        replace_element=args.replace_element,
        composition=composition,
        keep_composition=args.keep_composition,
        mode=args.mode,
        target=args.target,
        max_attempts=args.max_attempts,
        symprec=args.symprec,
        seed=args.seed,
        output_format=args.format,
        vasp_folder=args.vasp_folder,
        overwrite=args.overwrite,
        order_levels=args.order_levels,
        order_tolerance=args.order_tolerance,
        order_search_steps=args.order_search_steps,
        sro_cutoff_factor=args.sro_cutoff_factor,
        sro_tolerance=args.sro_tolerance,
        sro_weight=args.sro_weight,
        max_trials_per_bucket=args.max_trials_per_bucket,
        template_dir=args.template_dir,
        exhaustive_limit=args.exhaustive_limit,
        layer_axis=args.layer_axis,
        view_axis=args.view_axis,
        domain_pattern=args.domain_pattern,
        children_per_parent=args.children_per_parent,
        generate_potcar=args.generate_potcar,
        potcar_library=args.potcar_library,
        potcar_variants=args.potcar_variants,
        adsorbate_elements=args.adsorbate_elements,
        redox_remove=args.redox_remove,
        redox_max_sets=args.redox_max_sets,
    )


if __name__ == "__main__":
    main()
