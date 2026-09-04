# -*- coding: utf-8 -*-
"""CASM PRIM file generation/editing module.

PRIM is the primitive cell definition CASM reads. Its format is almost the
same as POSCAR's, with two differences.

1. There is no element-list line (POSCAR line 6). Only the atom-count line
   remains.
2. Every element that can occupy a site is listed after its coordinates
   (occupancy).

   0.00000000  0.00000000  0.00000000  Cu Ir     <- Cu or Ir
   0.00000000  0.00000000  0.00000000  Li Vac    <- Li or a vacancy
   0.00000000  0.16681600  0.12500000  C         <- C, fixed

This file used to be made by building a supercell with VTST's ``scale``
(csh + awk), then deleting line 6 by hand and attaching the element names.
``scale`` requires a POTCAR, but the values it reads from one (TITEL, RWIGS)
only go into a temporary file that gets deleted at the end, so it is not
actually needed. This module does the same job without POTCAR or csh.

The site-replication order is matched to ``scale``'s (each atom's images are
made in z -> x -> y order). For a binary, where every site's occupancy is the
same, the order does not affect the result, but for something like Li-C where
occupancy differs per site, the order itself carries meaning, so it is kept
matched to the original.
"""

import os
import re

import numpy as np


VACANCY = "Vac"


class PrimError(ValueError):
    """Raised while building or reading a PRIM."""


class Prim(object):
    """One PRIM file.

    Attributes
    ----------
    title : str
    scale : float
    lattice : (3,3) ndarray
        Rows are the lattice vectors a, b, c.
    coords : (N,3) ndarray
        Fractional coordinates.
    occupancies : list[list[str]]
        The elements each site can hold. ``["Cu","Ir"]``, ``["Li","Vac"]``,
        ``["C"]``.
    """

    def __init__(self, title, scale, lattice, coords, occupancies):
        self.title = title
        self.scale = float(scale)
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        self.coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        self.occupancies = [list(o) for o in occupancies]
        if len(self.coords) != len(self.occupancies):
            raise PrimError("Coordinate count (%d) and occupancy count (%d) differ."
                            % (len(self.coords), len(self.occupancies)))

    # -- properties ----------------------------------------------------------

    def __len__(self):
        return len(self.coords)

    @property
    def lengths(self):
        """Lattice vector lengths a, b, c (A). scale already multiplied in."""
        return np.linalg.norm(self.lattice, axis=1) * self.scale

    @property
    def elements(self):
        """Every element that appears (excluding Vac), in first-seen order."""
        seen = []
        for occ in self.occupancies:
            for e in occ:
                if e != VACANCY and e not in seen:
                    seen.append(e)
        return seen

    @property
    def mixed_sites(self):
        """Indices of sites whose occupancy has more than one element."""
        return [i for i, o in enumerate(self.occupancies) if len(o) > 1]

    def groups(self):
        """Group consecutive sites of the same occupancy into
        [(count, occupancy)]."""
        out = []
        for occ in self.occupancies:
            if out and out[-1][1] == occ:
                out[-1][0] += 1
            else:
                out.append([1, list(occ)])
        return [(n, o) for n, o in out]

    # -- I/O -------------------------------------------------------------

    def to_string(self):
        # Uses the same 16 digits as CONTCAR. The original scale was
        # truncated to 8 digits because of awk's OFMT (losing ~1e-9 A in the
        # lattice constant), and there is no reason to imitate that
        # truncation. In a configuration whose supercell volume is 2 or more,
        # this difference shows up in the POS coordinates.
        lines = [self.title,
                 "%20.14f" % self.scale]
        for v in self.lattice:
            lines.append("  %21.16f %21.16f %21.16f" % tuple(v))
        counts = [n for n, _ in self.groups()]
        lines.append(" " + " ".join(str(n) for n in counts))
        lines.append("Direct")
        for xyz, occ in zip(self.coords, self.occupancies):
            lines.append("  %21.16f %21.16f %21.16f  %s"
                         % (xyz[0], xyz[1], xyz[2], " ".join(occ)))
        return "\n".join(lines) + "\n"

    def write(self, path="PRIM"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        raw = text.rstrip("\n").split("\n")
        if len(raw) < 8:
            raise PrimError("PRIM is too short (%d line(s)). At least 8 lines are needed." % len(raw))

        title = raw[0].rstrip()
        try:
            scale = float(raw[1].split()[0])
        except (IndexError, ValueError):
            raise PrimError("Could not read the scale value on line 2: %r" % raw[1])

        lattice = []
        for i in (2, 3, 4):
            parts = raw[i].split()
            if len(parts) < 3:
                raise PrimError("Line %d is not a lattice vector: %r" % (i + 1, raw[i]))
            lattice.append([float(x) for x in parts[:3]])

        counts_line = raw[5].split()
        if not counts_line or not all(p.isdigit() for p in counts_line):
            raise PrimError(
                "Line 6 is not an atom-count line: %r\n"
                "PRIM must not have an element-name line -- please check "
                "whether a POSCAR was used as-is." % raw[5])
        total = sum(int(p) for p in counts_line)

        mode = raw[6].strip()
        if not mode[:1].lower() in ("d", "c", "k"):
            raise PrimError("Line 7 is not a coordinate-mode line (Direct/Cartesian): %r" % raw[6])
        if mode[:1].lower() != "d":
            raise PrimError("Only Direct (fractional) coordinates are supported. Got: %r" % mode)

        coords, occs = [], []
        for i in range(total):
            idx = 7 + i
            if idx >= len(raw):
                raise PrimError("Expected %d coordinate(s) but got only %d." % (total, len(coords)))
            parts = raw[idx].split()
            if len(parts) < 3:
                raise PrimError("Could not read the coordinates on line %d: %r" % (idx + 1, raw[idx]))
            coords.append([float(x) for x in parts[:3]])
            occ = [p for p in parts[3:] if re.match(r"^[A-Za-z]", p)]
            if not occ:
                raise PrimError(
                    "Line %d has no occupancy: %r\n"
                    "Each coordinate must be followed by the elements that "
                    "can occupy that site (e.g. 'Cu Ir', 'Li Vac', 'C')." % (idx + 1, raw[idx]))
            occs.append(occ)

        return cls(title, scale, lattice, coords, occs)

    @classmethod
    def read(cls, path="PRIM"):
        with open(path) as f:
            return cls.from_string(f.read())

    def copy(self):
        return Prim(self.title, self.scale, self.lattice.copy(),
                    self.coords.copy(), [list(o) for o in self.occupancies])

    def __repr__(self):                                # pragma: no cover
        a, b, c = self.lengths
        return ("<Prim %r %d sites %.4f x %.4f x %.4f>"
                % (self.title, len(self), a, b, c))


# ----------------------------------------------------------------------------
# Reading a structure
# ----------------------------------------------------------------------------

def read_structure(path):
    """Read (lattice, coords, species) from a POSCAR / CONTCAR / cif / etc.

    CASM's POS output is VASP4 format with no element-name line, so pymatgen
    looks for a POTCAR in the same folder. When even that is missing, the
    elements cannot be determined, so this says exactly what was missing.
    """
    from pymatgen.core import Structure

    if not os.path.isfile(path):
        raise PrimError("Structure file is missing: %s" % path)
    try:
        st = Structure.from_file(path)
    except Exception as err:
        base = os.path.basename(path)
        hint = ""
        if base.startswith(("POSCAR", "CONTCAR", "POS")):
            hint = ("\nIf this is VASP4 format with no element-name line, a "
                    "POTCAR must be in the same folder for the elements to "
                    "be determined.")
        raise PrimError("Could not read %s: %s%s" % (path, err, hint))

    lattice = np.array(st.lattice.matrix, dtype=float)
    coords = np.array(st.frac_coords, dtype=float)
    species = [str(s.specie.symbol) for s in st.sites]
    return lattice, coords, species


# ----------------------------------------------------------------------------
# supercell + occupancy -> PRIM
# ----------------------------------------------------------------------------

def _replicate(lattice, coords, species, nx, ny, nz):
    """Replicate the cell in the same order as ``scale``.

    Each atom's images are made in z -> x -> y order. The lattice vectors a,
    b, c are multiplied by nx, ny, nz respectively.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise PrimError("Supercell multiples must be 1 or more: (%d, %d, %d)" % (nx, ny, nz))

    new_lat = np.array(lattice, dtype=float).copy()
    new_lat[0] *= nx
    new_lat[1] *= ny
    new_lat[2] *= nz

    new_coords, new_species = [], []
    for (x, y, z), sp in zip(coords, species):
        for iz in range(nz):
            for ix in range(nx):
                for iy in range(ny):
                    new_coords.append([(x + ix) / nx, (y + iy) / ny, (z + iz) / nz])
                    new_species.append(sp)
    return new_lat, np.array(new_coords), new_species


def _normalize_occupancy(occupancy):
    if isinstance(occupancy, str):
        occ = occupancy.replace(",", " ").split()
    else:
        occ = [str(e) for e in occupancy]
    if not occ:
        raise PrimError("Occupancy is empty.")
    return occ


def _resolve_occupancies(occupancy, species):
    """Resolve an occupancy specification into a per-site list.

    Accepted forms
      "Cu Ir" / ["Cu","Ir"]        : the same occupancy on every site
      {"Li": "Li Vac", "C": "C"}   : given per original element
      {0: "Li Vac", 3: "C"}        : given per site index, after the supercell
    """
    n = len(species)

    if isinstance(occupancy, (str, list, tuple)):
        occ = _normalize_occupancy(occupancy)
        return [list(occ) for _ in range(n)]

    if not isinstance(occupancy, dict):
        raise PrimError("occupancy must be a string, a list, or a dict. "
                        "Got: %s" % type(occupancy).__name__)

    by_index = all(isinstance(k, int) for k in occupancy)
    by_symbol = all(isinstance(k, str) for k in occupancy)
    if not (by_index or by_symbol):
        raise PrimError("The keys of the occupancy dict must be all site "
                        "indices (int) or all element symbols (str).")

    out = []
    if by_index:
        unknown = [k for k in occupancy if not (0 <= k < n)]
        if unknown:
            raise PrimError("Site index is out of range: %s (%d site(s))"
                            % (sorted(unknown), n))
        for i, sp in enumerate(species):
            out.append(_normalize_occupancy(occupancy[i]) if i in occupancy else [sp])
    else:
        missing = sorted(set(species) - set(occupancy))
        if missing:
            raise PrimError(
                "The structure has element(s) not in occupancy: %s\n"
                "Give the fixed sites too (e.g. {'C': 'C'})." % ", ".join(missing))
        for sp in species:
            out.append(_normalize_occupancy(occupancy[sp]))
    return out


def make_prim(structure, occupancy, supercell=(1, 1, 1), title=None):
    """Build a PRIM from a structure file.

    Parameters
    ----------
    structure : str or tuple
        A structure file path, or a ``(lattice, coords, species)`` tuple.
    occupancy : str or list or dict
        The elements that can occupy a site. See :func:`_resolve_occupancies`.
    supercell : (int, int, int)
        Multiples along a, b, c. FCC uses ``(2,1,1)``, HCP/BCC use
        ``(2,2,1)`` to give an 8-site cell (course notes section 5.2).
    title : str
        PRIM line 1. If omitted, it is built by joining the element names.

    Returns
    -------
    Prim
    """
    if isinstance(structure, str):
        lattice, coords, species = read_structure(structure)
    else:
        lattice, coords, species = structure
        lattice = np.asarray(lattice, dtype=float)
        coords = np.asarray(coords, dtype=float)
        species = list(species)

    nx, ny, nz = supercell
    lattice, coords, species = _replicate(lattice, coords, species, nx, ny, nz)
    occupancies = _resolve_occupancies(occupancy, species)

    if title is None:
        elems = []
        for occ in occupancies:
            for e in occ:
                if e != VACANCY and e not in elems:
                    elems.append(e)
        title = "-".join(elems) if elems else "PRIM"

    return Prim(title, 1.0, lattice, coords, occupancies)


# ----------------------------------------------------------------------------
# Lowering to P1
# ----------------------------------------------------------------------------

def break_symmetry(prim, amplitude=0.001, sites=None, min_displacement=0.005,
                   strict=True):
    """Perturb the coordinates slightly to remove the cell's symmetry.

    While symmetry is still present, CASM merges symmetry-equivalent
    configurations into one. That is why an 8-site cell's 256 configurations
    come down to 9 (course notes section 5.3). Enumerating them all requires
    lowering to P1 first.

    The perturbation amplitude must clear two thresholds at once (course
    notes section 5.4).
      1) large enough that CASM cannot find the symmetry, and
      2) large enough that VASP cannot restore it with SYMPREC (default 1e-5).

    Because the perturbation is applied in fractional coordinates, a small
    cell can end up with a real displacement that is too small. So the
    displacement in A is computed directly and reported if it is below
    ``min_displacement`` -- a check the original shell script did not have.

    Parameters
    ----------
    prim : Prim
    amplitude : float
        Base amplitude (fractional coordinates). Site i is shifted in x by
        ``(i+1)*amplitude``.
    sites : list[int] or None
        Sites to perturb. Defaults to every site with more than one
        occupancy.
    min_displacement : float
        Minimum acceptable displacement (A).
    strict : bool
        True raises when the displacement is too small. False only warns and
        continues.

    Returns
    -------
    (Prim, list[str])
        The perturbed new Prim, and messages for a human to read.
    """
    out = prim.copy()
    if sites is None:
        sites = out.mixed_sites or list(range(len(out)))
    if not sites:
        raise PrimError("No sites to perturb.")

    for n, i in enumerate(sites):
        if not (0 <= i < len(out)):
            raise PrimError("Site index is out of range: %d" % i)
        out.coords[i][0] += (n + 1) * amplitude
        if n < 2:                       # also shift the first two sites in y, to break axis symmetry
            out.coords[i][1] += (n + 1) * 10 * amplitude

    delta = (out.coords - prim.coords)
    cart = delta.dot(prim.lattice) * prim.scale
    disp = np.linalg.norm(cart, axis=1)
    moved = disp[disp > 0]
    smallest = float(moved.min()) if len(moved) else 0.0
    largest = float(disp.max())

    msgs = ["Perturbed %d site(s), actual displacement %.4f ~ %.4f A" % (len(sites), smallest, largest)]
    if smallest < min_displacement:
        msg = ("A displacement of %.4f A is smaller than %.4f A. VASP could "
               "restore the symmetry via SYMPREC -- raise amplitude, or set "
               "ISYM=0 in the INCAR."
               % (smallest, min_displacement))
        if strict:
            raise PrimError(msg)
        msgs.append("Warning: " + msg)
    return out, msgs
