# -*- coding: utf-8 -*-
"""CASM CSPECS file generator.

CSPECS is the file that tells CASM how far out to look for clusters. For each
size (pair, triplet, quadruplet) it states "count every site inside this
radius as one cluster":

    Specifications for lattice
    cluster size      within sphere radius
      2                 3.0
      3                 3.0
      4                 3.0

The radius is set from the actual nearest-neighbour distances of the
structure. FCC Ag (a=4.1449 A), for example, has its 1st NN at 2.9309 A and
2nd NN at 4.1449 A, so 3.0 covers only the 1st shell and 4.5 covers the 2nd
(course notes section 6.2). Widening the radius pulls in more clusters and the
number of inequivalent configurations explodes -- it is a cost/accuracy
trade-off.

This distance used to be measured by building a supercell with VTST's
``scale`` and then running ``neighbors.pl``. This module gets the neighbour
shells straight from pymatgen on the original cell instead. The supercell step
is unnecessary because ``neighbors.pl`` relies on the minimum-image
convention -- the cell has to be enlarged first for that convention to hold --
while pymatgen counts periodic images properly, so that whole step disappears.

Dropping that step also drops the problem that came with it. ``scale``'s 4th
argument is not a vacuum padding (zvac); it is the c-axis multiplier (zscale).
``scale 2 2 2 1``, as used in the course notes and Sym_Alloy.py, grows the
atoms 2x2x2 while leaving the c axis at 1x, so the z-direction images sit
right on top of the original. Counting cms2's Ag/2x2cell for real shows only
16 distinct positions among the 32 atoms (volume 284.84 A^3 / 71.21 A^3 x 4 =
16). Distances themselves are unaffected, but coordination numbers come out
doubled and zero-distance pairs appear. A proper 2x2x2 needs
``scale 2 2 2 2``.
"""

import math

import numpy as np


class CspecsError(ValueError):
    """Raised while building or reading a CSPECS."""


DEFAULT_SIZES = (2, 3, 4)


class Cspecs(object):
    """One CSPECS file.

    Attributes
    ----------
    radii : dict[int, float]
        Cluster size -> radius (A).
    """

    HEADER = "Specifications for lattice"
    COLUMNS = "cluster size      within sphere radius"

    def __init__(self, radii):
        if isinstance(radii, (int, float)):
            radii = dict((s, float(radii)) for s in DEFAULT_SIZES)
        self.radii = dict((int(k), float(v)) for k, v in dict(radii).items())
        if not self.radii:
            raise CspecsError("No cluster sizes were given.")
        for size, r in self.radii.items():
            if size < 2:
                raise CspecsError("Cluster size must be 2 or more: %d" % size)
            if r <= 0:
                raise CspecsError("Radius must be greater than 0: size %d -> %g" % (size, r))

    def to_string(self):
        lines = [self.HEADER, self.COLUMNS]
        for size in sorted(self.radii):
            lines.append("  %-16d%s" % (size, _fmt(self.radii[size])))
        return "\n".join(lines) + "\n"

    def write(self, path="CSPECS"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        radii = {}
        for line in text.split("\n"):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                size, r = int(parts[0]), float(parts[1])
            except ValueError:
                continue                       # header line
            radii[size] = r
        if not radii:
            raise CspecsError(
                "Could not find a 'size radius' line in the CSPECS.\n"
                "There should be a line like '2  3.0' after the two header lines.")
        return cls(radii)

    @classmethod
    def read(cls, path="CSPECS"):
        with open(path) as f:
            return cls.from_string(f.read())

    def __repr__(self):                                # pragma: no cover
        return "<Cspecs %s>" % ", ".join(
            "%d:%.4g" % (s, self.radii[s]) for s in sorted(self.radii))


# ----------------------------------------------------------------------------
# Neighbour shells
# ----------------------------------------------------------------------------

def neighbor_shells(structure, species=None, index=None, rmax=8.0, tol=1e-3):
    """Group neighbours into shells by distance and return them.

    Parameters
    ----------
    structure : str or pymatgen Structure
        A structure file path or a Structure. A Prim is accepted too
        (:class:`CCpy.CASM.CASMprim.Prim`).
    species : str or list[str] or None
        Only look at distances between these elements. Used when the radius
        should be set from one specific element pair, as with Li-C (the
        course notes use the Li-Li distance to get 5.0 A). None means every
        element.
    index : int or None
        The site to use as the centre. None means the first site that
        matches the condition.
    rmax : float
        Search only out to here (A).
    tol : float
        Neighbours within this of each other count as the same shell (A).

    Returns
    -------
    list[(distance, count)]
        Ascending by distance. count is the number of neighbours in that
        shell (the coordination number).
    """
    st = _as_structure(structure)

    if species is not None:
        if isinstance(species, str):
            species = [species]
        species = set(species)
        candidates = [i for i, s in enumerate(st) if s.specie.symbol in species]
        if not candidates:
            raise CspecsError(
                "The structure has no %s. Elements present: %s"
                % (", ".join(sorted(species)),
                   ", ".join(sorted({s.specie.symbol for s in st}))))
    else:
        candidates = list(range(len(st)))

    center = candidates[0] if index is None else index
    if not (0 <= center < len(st)):
        raise CspecsError("Site index is out of range: %s (%d site(s))"
                          % (center, len(st)))
    if species is not None and st[center].specie.symbol not in species:
        raise CspecsError(
            "Site %d is %s, but you asked to look only at %s."
            % (center, st[center].specie.symbol, ", ".join(sorted(species))))

    dists = []
    for nb in st.get_neighbors(st[center], rmax):
        if species is not None and nb.specie.symbol not in species:
            continue
        dists.append(float(nb.nn_distance))
    if not dists:
        raise CspecsError(
            "No neighbours within %.2f A. Try raising rmax." % rmax)

    dists.sort()
    shells = []
    for d in dists:
        if shells and abs(d - shells[-1][0]) <= tol:
            shells[-1][1] += 1
        else:
            shells.append([d, 1])
    return [(d, n) for d, n in shells]


def describe_shells(shells, limit=6):
    """Neighbour shells as a table for a human to read."""
    lines = ["  shell   distance(A)   neighbours"]
    for i, (d, n) in enumerate(shells[:limit], start=1):
        lines.append("  %3d   %10.6f   %6d" % (i, d, n))
    if len(shells) > limit:
        lines.append("  ... (%d more)" % (len(shells) - limit))
    return "\n".join(lines)


def suggest_radius(shells, nshell=1, step=0.5):
    """Suggest a radius that includes up to the n-th shell.

    Picks the next multiple of ``step`` above the shell distance, without
    reaching into the next shell. If it would, the midpoint between the two
    shells is used instead.

    Checked against real values: for FCC Ag (1st 2.9309, 2nd 4.1449) this
    gives 3.0 for the 1st shell and 4.5 for the 2nd, matching the n3/n4
    values of course notes section 6.2.
    """
    if nshell < 1 or nshell > len(shells):
        raise CspecsError("Shell index is out of range: %d (%d shell(s) available)"
                          % (nshell, len(shells)))
    d = shells[nshell - 1][0]
    nxt = shells[nshell][0] if nshell < len(shells) else None

    r = math.ceil((d + 1e-6) / step) * step
    if nxt is not None and r >= nxt:
        r = 0.5 * (d + nxt)
    return round(r, 4)


def make_cspecs(structure, nshell=1, sizes=DEFAULT_SIZES, radius=None,
                species=None, index=None, rmax=8.0):
    """Build a CSPECS from a structure.

    Returns
    -------
    (Cspecs, list[(distance, count)])
        The CSPECS built, and the neighbour shells used to set the radius.
    """
    shells = neighbor_shells(structure, species=species, index=index, rmax=rmax)
    if radius is None:
        radius = suggest_radius(shells, nshell=nshell)
    radius = float(radius)

    sizes = [int(s) for s in sizes]
    if not sizes:
        raise CspecsError("Give at least one cluster size (usually 2, 3, 4).")
    return Cspecs(dict((s, radius) for s in sizes)), shells


# ----------------------------------------------------------------------------

def _fmt(value):
    """3.0 stays '3.0', 4.5 stays '4.5'. Keeps the decimal point."""
    s = ("%.4f" % value).rstrip("0")
    return s + "0" if s.endswith(".") else s


def _as_structure(structure):
    """A path / Prim / Structure, turned into a pymatgen Structure."""
    from pymatgen.core import Structure

    if isinstance(structure, Structure):
        return structure

    if isinstance(structure, str):
        from CCpy.CASM.CASMprim import read_structure
        lattice, coords, species = read_structure(structure)
        return Structure(lattice, species, coords, coords_are_cartesian=False)

    # Prim: a mixed site is reduced to one representative element for the
    # distance measurement only.
    if hasattr(structure, "occupancies") and hasattr(structure, "coords"):
        from CCpy.CASM.CASMprim import VACANCY
        species = []
        for occ in structure.occupancies:
            real = [e for e in occ if e != VACANCY]
            if not real:
                raise CspecsError("A site's occupancy is Vac only.")
            species.append(real[0])
        return Structure(np.asarray(structure.lattice) * structure.scale,
                         species, structure.coords, coords_are_cartesian=False)

    raise CspecsError("Cannot use this as a structure: %s" % type(structure).__name__)
