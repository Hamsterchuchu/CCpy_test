# -*- coding: utf-8 -*-
"""Rewrite each configuration's cell to the composition-weighted average (Vegard).

When the two elements' lattice constants differ a lot (Cu 3.627 A vs Ir 3.871
A), a relaxation that starts from one or the other's value does not converge
well. Mixing the two lattice constants by each configuration's own
composition to build the starting cell speeds up convergence.

**This cell is only a starting estimate.** ISIF=3 has to let each
configuration relax to its own volume for the formation energy to be correct.
Measured runs show the volume moving by up to 1.6% after relaxation.

Three problems of the original ``03_cellparam.sh`` are removed here.

1. **Idempotency.** The original read ``$list/POSCAR`` and overwrote the same
   file, so running it twice multiplied the scale factor twice. Here, ``POS``
   -- what CASM writes once and nobody touches afterward -- is read and
   ``POSCAR`` is written, so the result is the same no matter how many times
   this runs.
2. **Argument order.** The original accepted ``03_cellparam.sh Ir Cu`` just as
   readily reversed, and scaled every cell backwards. Here the element order
   is read from PRIM's occupancy; if the user gives an order, it is only
   checked against PRIM's.
3. **Isotropic scaling.** The original multiplied a, b, and c by the same
   factor. Here the ratio of the corresponding length is used per axis. The
   result is the same for a cubic-cubic pair, and only differs for a
   structure like hexagonal where c/a differs.

Lattice constants are looked for in this order.

1. A value given directly through the ``lattice`` argument
2. A structure file given through ``refs``
3. The ``BULK/<element>/CONTCAR`` convention (searched going up the tree)
4. The :data:`LATTICE_TABLE` below

The table is not from the literature -- it is **this project's own optimised
BULK results** (2026-09, PBE, ENCUT 400, EDIFF 1E-06). It is only used as a
starting estimate, so a slightly different setup is not a problem, but a
project's own BULK file always comes first when one exists.
"""

import os

import numpy as np


class CellparamError(ValueError):
    """Raised while processing cell parameters."""


#: Element -> (a, c, structure, POTCAR). Units are A.
#: Cubic structures are filled in with a == c.
LATTICE_TABLE = {
    "Ag": (4.144918, 4.144918, "fcc", "Ag"),
    "Au": (4.161795, 4.161795, "fcc", "Au"),
    "Cd": (3.073154, 5.444719, "hcp", "Cd"),
    "Co": (2.493935, 4.023521, "hcp", "Co"),
    "Cr": (2.835053, 2.835053, "bcc", "Cr"),
    "Cu": (3.627129, 3.627129, "fcc", "Cu"),
    "Fe": (2.828922, 2.828922, "bcc", "Fe"),
    "Hf": (3.194600, 5.051100, "hcp", "Hf"),
    "Ir": (3.870726, 3.870726, "fcc", "Ir"),
    "Mo": (3.152529, 3.152529, "bcc", "Mo"),
    "Nb": (3.308373, 3.308373, "bcc", "Nb_sv"),
    "Ni": (3.511475, 3.511475, "fcc", "Ni"),
    "Os": (2.750344, 4.354639, "hcp", "Os"),
    "Pd": (3.934689, 3.934689, "fcc", "Pd"),
    "Pt": (3.965701, 3.965701, "fcc", "Pt"),
    "Re": (2.777776, 4.468487, "hcp", "Re"),
    "Rh": (3.822364, 3.822364, "fcc", "Rh"),
    "Ru": (2.711545, 4.288621, "hcp", "Ru"),
    "Sc": (3.302217, 5.136205, "hcp", "Sc"),
    "Ta": (3.313380, 3.313380, "bcc", "Ta"),
    "Tc": (2.746784, 4.376789, "hcp", "Tc"),
    "Ti": (2.923401, 4.629729, "hcp", "Ti"),
    "V":  (2.979188, 2.979188, "bcc", "V"),
    "W":  (3.174238, 3.174238, "bcc", "W"),
    "Y":  (3.664575, 5.656533, "hcp", "Y_sv"),
    "Zn": (2.605339, 5.309382, "hcp", "Zn"),
    "Zr": (3.221385, 5.193937, "hcp", "Zr_sv"),
}

TABLE_NOTE = ("built-in table (this project's own BULK optimisation, PBE / "
              "ENCUT 400 / EDIFF 1E-06)")


# ----------------------------------------------------------------------------
# Finding lattice constants
# ----------------------------------------------------------------------------

def _read_lengths(path):
    """Read the 3 lattice vector lengths from a structure file."""
    if not os.path.isfile(path):
        raise CellparamError("Structure file is missing: %s" % path)
    lines = open(path).read().split("\n")
    try:
        scale = float(lines[1].split()[0])
        mat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    except (IndexError, ValueError):
        raise CellparamError("Could not read the lattice from %s." % path)
    if scale < 0:                       # negative scale = volume is given directly
        scale = (abs(scale) / abs(np.linalg.det(mat))) ** (1.0 / 3.0)
    return np.linalg.norm(mat, axis=1) * scale


def _find_bulk(element, roots):
    for root in roots:
        for name in ("CONTCAR", "POSCAR"):
            p = os.path.join(root, element, name)
            if os.path.isfile(p):
                return p
    return None


def element_lengths(element, lattice=None, refs=None, roots=None):
    """Return one element's 3 lattice vector lengths and where they came from.

    Returns
    -------
    (lengths, source)
    """
    if lattice is not None:
        v = np.atleast_1d(np.asarray(lattice, dtype=float))
        if v.size == 1:
            v = np.repeat(v, 3)
        elif v.size == 2:                # given as (a, c)
            v = np.array([v[0], v[0], v[1]])
        if v.size != 3:
            raise CellparamError("Give the lattice constant as 1 value (a), 2 "
                                 "(a c), or 3 (a b c): %r" % (lattice,))
        return v, "given directly"

    if refs and element in refs:
        return _read_lengths(refs[element]), refs[element]

    if roots is None:
        roots = default_bulk_roots()
    found = _find_bulk(element, roots)
    if found:
        return _read_lengths(found), found

    if element in LATTICE_TABLE:
        a, c, _, _ = LATTICE_TABLE[element]
        return np.array([a, a, c]), TABLE_NOTE

    raise CellparamError(
        "Could not find a lattice constant for %s.\n"
        "Place a BULK/%s/CONTCAR, give a path with -ref, or a value with -a.\n"
        "Elements in the built-in table: %s"
        % (element, element, ", ".join(sorted(LATTICE_TABLE))))


def default_bulk_roots(start="."):
    """Places a BULK folder might be, nearest first."""
    out = []
    cur = os.path.abspath(start)
    for _ in range(4):
        out.append(os.path.join(cur, "BULK"))
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


# ----------------------------------------------------------------------------
# Applying
# ----------------------------------------------------------------------------

def elements_from_prim(path="PRIM"):
    """Read the element order from PRIM's occupancy order."""
    from CCpy.CASM.CASMprim import Prim, PrimError
    try:
        prim = Prim.read(path)
    except PrimError as err:
        raise CellparamError("Could not read the PRIM: %s" % err)
    elts = prim.elements
    if len(elts) < 2:
        raise CellparamError(
            "PRIM has only %d element(s) (%s). A composition-weighted "
            "average only makes sense for a binary." % (len(elts), ", ".join(elts) or "none"))
    return elts


def read_concentrations(path="make_dirs"):
    """Read {configuration name: composition} from make_dirs.

    Column 3 is the first element's fraction.
    """
    if not os.path.isfile(path):
        raise CellparamError(
            "%s is missing. Run mainclust's enumeration first." % path)
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        try:
            out[parts[0]] = float(parts[2])
        except ValueError:
            continue
    if not out:
        raise CellparamError("Could not read a composition from %s." % path)
    return out


def apply(root=".", elements=None, lattice=None, refs=None, roots=None,
          isotropic=False, dirs=None):
    """Scale every configuration's POS to the composition-weighted average
    cell and write it as POSCAR.

    Parameters
    ----------
    root : str
    elements : (str, str) or None
        Element order. None reads it from PRIM. If given, it is checked
        against PRIM's.
    lattice : dict or None
        Given directly, e.g. ``{"Cu": 3.6271, "Ir": [3.87, 3.87, 3.87]}``.
    refs : dict or None
        Given as files, e.g. ``{"Cu": "../BULK/Cu/CONTCAR"}``.
    roots : list[str] or None
        Where to look for BULK.
    isotropic : bool
        True scales all three axes by the single a-axis ratio (the original's
        behaviour).
    dirs : list[str] or None

    Returns
    -------
    (records, notes)
    """
    prim_elts = elements_from_prim(os.path.join(root, "PRIM"))
    if elements is None:
        elements = prim_elts[:2]
    else:
        elements = list(elements)
        if len(elements) != 2:
            raise CellparamError("Give exactly 2 elements: %r" % (elements,))
        if elements != prim_elts[:2]:
            raise CellparamError(
                "The order you gave, %s, differs from PRIM's order, %s.\n"
                "Reversing the order flips the composition weighting, so every "
                "cell would start from the wrong volume."
                % (" ".join(elements), " ".join(prim_elts[:2])))
    e1, e2 = elements

    lattice = lattice or {}
    l1, src1 = element_lengths(e1, lattice.get(e1), refs, roots)
    l2, src2 = element_lengths(e2, lattice.get(e2), refs, roots)

    ratio = np.asarray(l2, dtype=float) / np.asarray(l1, dtype=float)
    if isotropic:
        ratio = np.repeat(ratio[0], 3)

    notes = ["%s : %s  (%s)" % (e1, " ".join("%.6f" % v for v in l1), src1),
             "%s : %s  (%s)" % (e2, " ".join("%.6f" % v for v in l2), src2),
             "per-axis scale %s%s" % (" ".join("%.6f" % v for v in ratio),
                              "  (isotropic)" if isotropic else "")]
    if max(ratio) / min(ratio) > 1.02 and not isotropic:
        notes.append("The two structures' c/a differ, so the scale differs "
                     "per axis. Give -iso for the original 03_cellparam.sh's "
                     "isotropic behaviour.")
    spread = abs(ratio[0] - 1.0)
    if spread > 0.10:
        notes.append("Warning: the lattice constants differ by %.1f%%. The "
                     "relaxation may not converge well." % (spread * 100))

    concentrations = read_concentrations(os.path.join(root, "make_dirs"))

    if dirs is None:
        from CCpy.CASM.CASMkpoints import config_dirs
        dirs = config_dirs(root)
    if not dirs:
        raise CellparamError("Could not find any configuration directories (con*).")

    records, skipped = [], []
    for d in dirs:
        name = os.path.basename(d)
        if name not in concentrations:
            skipped.append((name, "not in make_dirs"))
            continue
        src = os.path.join(d, "POS")
        if not os.path.isfile(src):
            skipped.append((name, "no POS (CASM's original structure)"))
            continue
        x = concentrations[name]
        factor = x + ratio * (1.0 - x)
        try:
            _scale_poscar(src, os.path.join(d, "POSCAR"), factor)
        except CellparamError as err:
            skipped.append((name, str(err)))
            continue
        records.append({"name": name, "x": x, "factor": tuple(factor)})

    return records, notes, skipped


def _scale_poscar(src, dst, factor):
    """Multiply POS's lattice vectors by factor, per axis, and write as POSCAR."""
    lines = open(src).read().split("\n")
    if len(lines) < 7:
        raise CellparamError("Structure file is too short: %s" % src)
    out = list(lines)
    for i, f in zip((2, 3, 4), factor):
        parts = lines[i].split()
        if len(parts) < 3:
            raise CellparamError("Line %d is not a lattice vector: %r" % (i + 1, lines[i]))
        vec = [float(v) * f for v in parts[:3]]
        out[i] = "  %15.7f %15.7f %15.7f " % tuple(vec)
    with open(dst, "w") as fh:
        fh.write("\n".join(out))


def describe(records, notes, skipped, limit=4):
    """apply()'s result as sentences for a human to read."""
    lines = list(notes)
    lines.append("Rewrote the cell of %d configuration(s)." % len(records))
    if records:
        xs = [r["x"] for r in records]
        lines.append("  composition range %.4g ~ %.4g" % (min(xs), max(xs)))
        for r in sorted(records, key=lambda r: r["x"])[:2] + \
                sorted(records, key=lambda r: -r["x"])[:1]:
            lines.append("    %-10s x=%-6.4g factor %s"
                         % (r["name"], r["x"],
                            " ".join("%.6f" % v for v in r["factor"])))
    if skipped:
        lines.append("  %d skipped:" % len(skipped))
        for name, why in skipped[:limit]:
            lines.append("    %-10s %s" % (name, why))
        if len(skipped) > limit:
            lines.append("    ... (%d more)" % (len(skipped) - limit))
    lines.append("  Reads POS and writes POSCAR, so running this more than "
                 "once gives the same result.")
    return "\n".join(lines)
