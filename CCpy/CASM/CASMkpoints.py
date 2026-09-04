# -*- coding: utf-8 -*-
"""KPOINTS generation/distribution module for CASM configuration calculations.

Configurations inside one hull are subtracted from each other to build
formation energies, so if the k-sampling error differs between
configurations, that difference turns into physics. That is why course notes
section 5.6 says "every configuration in one hull must use the same KPOINTS
file" and gives an md5 check alongside it.

This module does three things.

1. **Picks a mesh that fits the cell.** The division count of each axis is
   computed separately so the three axes' k*L come out even against the
   target. Measured runs show all 256 of Cu-Ir's configurations using
   ``6 6 6`` -- a value made for the 4-atom cube (BULK/Cu) that carried over
   unchanged even after the PRIM was grown with ``scale 2 1 1 1``. So k*L
   comes out 22.5 / 22.5 / 45.0, twice as dense on the long axis only.

2. **Picks out hexagonal cells.** In a hexagonal lattice, an even-division
   Monkhorst-Pack has its off-centre grid break the hexagonal symmetry and
   produce the wrong IBZ (course notes section 5.7). An odd division makes MP
   and Gamma-centered the same, so a hexagonal cell is forced to odd. 12 of
   the 27 alloy targets are HCP, so this is not someone else's problem.

3. **Copies one file to every directory and checks they match.** Moves the
   md5sum check the course notes used to do by hand into code.

Not used for Method 2 (symmetric supercell). There each configuration's cell
volume differs (1 vs 2) and mainclust divides the mesh by the SCEL multiplier
to keep the k-density constant (measured k*L 23.6-24.9). Forcing the same
file there would instead throw the density off by a factor of two per
configuration.
"""

import hashlib
import os

import numpy as np


class KpointsError(ValueError):
    """Raised while building or distributing a KPOINTS."""


GAMMA = "Gamma"
MONKHORST = "Monkhorst"

#: Default target for metals. Course notes section 3, prescription 4
#: recommends k*a 35-40 for metals (an insulator is fine around 20).
DEFAULT_TARGET = 35.0


class Kpoints(object):
    """One KPOINTS file (automatic mesh format).

    Attributes
    ----------
    mesh : tuple[int, int, int]
    mode : str
        ``"Gamma"`` or ``"Monkhorst"``. VASP only looks at the first letter.
    comment : str
    shift : tuple[float, float, float] or None
    """

    def __init__(self, mesh, mode=GAMMA, comment="CCpy CASM", shift=(0.0, 0.0, 0.0)):
        mesh = tuple(int(v) for v in mesh)
        if len(mesh) != 3:
            raise KpointsError("mesh needs exactly 3 integers: %r" % (mesh,))
        if min(mesh) < 1:
            raise KpointsError("Mesh divisions must be 1 or more: %r" % (mesh,))
        self.mesh = mesh
        self.mode = GAMMA if str(mode)[:1].upper() == "G" else MONKHORST
        self.comment = comment
        self.shift = tuple(float(v) for v in shift) if shift is not None else None

    @property
    def is_gamma(self):
        return self.mode == GAMMA

    def to_string(self):
        lines = [self.comment, "0", self.mode,
                 " ".join(str(v) for v in self.mesh)]
        if self.shift is not None:
            lines.append(" ".join(("%g" % v) for v in self.shift))
        return "\n".join(lines) + "\n"

    def write(self, path="KPOINTS"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        raw = [l for l in text.rstrip("\n").split("\n")]
        if len(raw) < 4:
            raise KpointsError("KPOINTS is too short (%d line(s)). At least 4 "
                               "lines are needed." % len(raw))
        comment = raw[0].strip()
        try:
            nk = int(raw[1].split()[0])
        except (IndexError, ValueError):
            raise KpointsError("Line 2 is not a number: %r" % raw[1])
        if nk != 0:
            raise KpointsError(
                "Only the automatic mesh format (line 2 is 0) is supported. "
                "Line 2 being %d means an explicit list of k-points." % nk)
        mode = raw[2].strip()
        if not mode[:1].upper() in ("G", "M"):
            raise KpointsError(
                "Line 3 is not Gamma / Monkhorst: %r" % raw[2])
        try:
            mesh = [int(v) for v in raw[3].split()[:3]]
        except ValueError:
            raise KpointsError("Could not read the mesh from line 4: %r" % raw[3])
        if len(mesh) != 3:
            raise KpointsError("Line 4 needs 3 division counts: %r" % raw[3])

        shift = None
        if len(raw) > 4 and raw[4].split():
            try:
                shift = [float(v) for v in raw[4].split()[:3]]
            except ValueError:
                shift = None
        return cls(mesh, mode=mode, comment=comment, shift=shift)

    @classmethod
    def read(cls, path="KPOINTS"):
        with open(path) as f:
            return cls.from_string(f.read())

    def __repr__(self):                                # pragma: no cover
        return "<Kpoints %s %d %d %d>" % ((self.mode,) + self.mesh)


# ----------------------------------------------------------------------------
# Picking a mesh
# ----------------------------------------------------------------------------

def _lattice_matrix(structure):
    """Pull the lattice matrix (3x3, A) out of a structure.

    PRIM has no file extension, so pymatgen cannot recognise its format --
    it is picked out by filename and read with the Prim parser instead.
    """
    if hasattr(structure, "occupancies") and hasattr(structure, "lattice"):
        return np.asarray(structure.lattice, dtype=float) * structure.scale
    if hasattr(structure, "lattice") and hasattr(structure.lattice, "matrix"):
        return np.asarray(structure.lattice.matrix, dtype=float)
    if isinstance(structure, str):
        if os.path.basename(structure).startswith("PRIM"):
            from CCpy.CASM.CASMprim import Prim
            prim = Prim.read(structure)
            return np.asarray(prim.lattice, dtype=float) * prim.scale
        from CCpy.CASM.CASMprim import read_structure
        lattice, _, _ = read_structure(structure)
        return np.asarray(lattice, dtype=float)
    raise KpointsError("Cannot use this as a structure: %s" % type(structure).__name__)


def lattice_lengths(structure):
    """Lattice vector lengths (a, b, c) of a structure, in A."""
    mat = _lattice_matrix(structure)
    return tuple(float(v) for v in np.linalg.norm(mat, axis=1))


def is_hexagonal(structure, angle_tol=1.0, length_tol=0.01):
    """Is this a hexagonal (or trigonal) cell? True when gamma is 120 degrees
    (or 60) and a is approximately b."""
    mat = _lattice_matrix(structure)
    a, b, c = np.linalg.norm(mat, axis=1)
    def ang(u, v):
        return np.degrees(np.arccos(
            np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    alpha, beta, gamma = ang(mat[1], mat[2]), ang(mat[0], mat[2]), ang(mat[0], mat[1])

    ab_close = abs(a - b) < length_tol * max(a, b)
    right = abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol
    hex_gamma = abs(gamma - 120) < angle_tol or abs(gamma - 60) < angle_tol
    return bool(ab_close and right and hex_gamma)


def _to_odd(n):
    """Round to the nearest odd number. An even number rounds up (denser)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def suggest_mesh(structure, target=DEFAULT_TARGET, force_odd=None, min_k=1):
    """Pick a division count per axis so the three axes' k*L match the target.

    Parameters
    ----------
    structure : str or Prim or Structure
    target : float
        Target k*L (A). 35-40 for a metal, around 20 for an insulator.
    force_odd : bool or None
        Whether to force an odd division. None forces it only for a
        hexagonal cell.
    min_k : int
        Lower bound for each axis.

    Returns
    -------
    (mesh, info)
        info is a list of description strings for a human to read.
    """
    if target <= 0:
        raise KpointsError("target must be greater than 0: %g" % target)

    lengths = lattice_lengths(structure)
    hexagonal = is_hexagonal(structure)
    if force_odd is None:
        force_odd = hexagonal

    mesh = []
    for L in lengths:
        k = int(round(target / L))
        k = max(int(min_k), k)
        if force_odd:
            k = _to_odd(k)
        mesh.append(k)
    mesh = tuple(mesh)

    kl = [m * L for m, L in zip(mesh, lengths)]
    info = ["lattice %.4f / %.4f / %.4f A" % lengths,
            "mesh %d x %d x %d  ->  k*L %.1f / %.1f / %.1f (target %.0f)"
            % (mesh + tuple(kl) + (target,))]
    if hexagonal:
        info.append("This is a hexagonal cell. An even-division "
                    "Monkhorst-Pack would break the hexagonal symmetry, so "
                    "an odd division was used.")
    spread = max(kl) - min(kl)
    if spread > 0.35 * target:
        info.append("Warning: the k*L spread between axes is %.1f, which is "
                    "large. Check whether the cell shape is very distorted."
                    % spread)
    return mesh, info


def make_kpoints(structure, target=DEFAULT_TARGET, mode=None, force_odd=None,
                 comment=None, min_k=1):
    """Build a KPOINTS that fits the structure.

    Without a given mode, a hexagonal cell uses Gamma-centered and everything
    else uses Monkhorst.
    """
    mesh, info = suggest_mesh(structure, target=target, force_odd=force_odd,
                              min_k=min_k)
    if mode is None:
        mode = GAMMA if is_hexagonal(structure) else MONKHORST
    if comment is None:
        comment = "CCpy CASM : k*L ~ %g" % target
    return Kpoints(mesh, mode=mode, comment=comment), info


# ----------------------------------------------------------------------------
# Distributing and verifying
# ----------------------------------------------------------------------------

def config_dirs(root=".", prefix="con"):
    """The configuration directories CASM built (con0.0, con1.12, ...)."""
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith(prefix) and os.path.isdir(path) \
                and not name.startswith("config"):
            out.append(path)
    return out


def distribute(kpoints, dirs, filename="KPOINTS"):
    """Write the same KPOINTS to every configuration directory.

    Returns
    -------
    int : number written
    """
    if not dirs:
        raise KpointsError(
            "Could not find any configuration directories. Build con* with "
            "mainclust first.")
    text = kpoints.to_string() if isinstance(kpoints, Kpoints) else str(kpoints)
    for d in dirs:
        if not os.path.isdir(d):
            raise KpointsError("Not a directory: %s" % d)
        with open(os.path.join(d, filename), "w") as f:
            f.write(text)
    return len(dirs)


def verify_uniform(dirs, filename="KPOINTS", compare="mesh"):
    """Check that every configuration uses the same KPOINTS.

    Course notes section 5.6 says ``md5sum con0.*/KPOINTS | sort -u | wc -l``
    should be 1, but **the whole file's md5 is the wrong thing to check.**
    mainclust writes that configuration's directory name into KPOINTS' line
    1, so even with the same mesh the comment differs and the md5s all come
    out different. In practice, Ag-Pd's super2_n3 has 21 distinct md5s but
    only 3 distinct meshes -- ``6 6 6`` / ``6 6 3`` / ``8 5 5``.

    So the default is to **compare the content (mode + mesh + shift)**
    instead. Use ``compare="bytes"`` to look at the whole file.

    Returns
    -------
    (ok, groups)
        groups is {description: [directory, ...]}.
    """
    if compare not in ("mesh", "bytes"):
        raise KpointsError("compare must be 'mesh' or 'bytes': %r" % compare)
    if not dirs:
        raise KpointsError(
            "No configuration directories to check. Build con* with "
            "mainclust first.")

    groups = {}
    missing, unreadable = [], []
    for d in dirs:
        path = os.path.join(d, filename)
        if not os.path.isfile(path):
            missing.append(d)
            continue
        if compare == "bytes":
            with open(path, "rb") as f:
                key = "md5 " + hashlib.md5(f.read()).hexdigest()[:8]
        else:
            try:
                kp = Kpoints.read(path)
            except KpointsError as err:
                unreadable.append((d, err))
                continue
            key = "%s %d %d %d" % ((kp.mode,) + kp.mesh)
            if kp.shift and any(abs(s) > 1e-9 for s in kp.shift):
                key += " shift " + " ".join("%g" % s for s in kp.shift)
        groups.setdefault(key, []).append(d)

    if missing:
        raise KpointsError(
            "%d director(ies) have no %s: %s"
            % (len(missing), filename,
               ", ".join(os.path.basename(m) for m in missing[:5])
               + (" ..." if len(missing) > 5 else "")))
    if unreadable:
        d, err = unreadable[0]
        raise KpointsError("Could not read %s's %s: %s"
                           % (os.path.basename(d), filename, err))
    return (len(groups) == 1), groups


def describe_uniformity(groups, limit=6):
    """verify_uniform's result as sentences for a human to read."""
    total = sum(len(v) for v in groups.values())
    if len(groups) == 1:
        key, dirs = next(iter(groups.items()))
        return "All %d configuration(s) use the same KPOINTS  [%s]" % (len(dirs), key)

    lines = ["%d configuration(s) have %d different KPOINTS -- one hull must "
             "use only one." % (total, len(groups))]
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for key, dirs in ordered[:limit]:
        sample = ", ".join(os.path.basename(d) for d in dirs[:4])
        lines.append("  %-24s %4d  (%s%s)"
                     % (key, len(dirs), sample, " ..." if len(dirs) > 4 else ""))
    if len(ordered) > limit:
        lines.append("  ... (%d more kind(s))" % (len(ordered) - limit))
    return "\n".join(lines)
