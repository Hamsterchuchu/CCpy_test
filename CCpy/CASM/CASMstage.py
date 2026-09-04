# -*- coding: utf-8 -*-
"""Stage progression for CASM configuration calculations -- two relaxations plus a final static.

Until now, each configuration was run just **once** with ``ISIF=3`` and that
was it. Two things are missing.

**Pulay stress.** Changing the volume under a fixed plane-wave basis leaves a
systematic error in the stress. Running once leaves the volume less than
fully converged. Moving CONTCAR to POSCAR and running again effectively
re-solves with a basis matched to the new volume, which removes this error.
Measured runs show Cu-Ir's con0.128 moving 1.6% in volume relative to its
starting cell -- only running once more, and seeing how much further it
moves, can tell whether it has converged.

**Smearing.** ``ISMEAR=1`` is right for relaxation (the tetrahedron method
gives inaccurate forces and stress), but it leaves an artificial entropy term
in the final energy. A final run with ``NSW=0``, ``ISMEAR=-5`` settles the
energy.

This is a direct port of course notes section 3, prescriptions 3, 5 and 7,
and serves both formation-energy accuracy and MLIP labelling.

Convergence judgment is not rebuilt here. The ``01_unconverged_jobs.csv``
that ``CCpyVASPAnal.py 0`` leaves behind is read, and those configurations
are skipped. In practice the "reached required accuracy" phrase alone is not
enough to judge by -- Cu-Ir's con0.50 has that phrase yet was marked
unconverged by zbrent.
"""

import csv
import os
import shutil

import numpy as np

from CCpy.VASP.VASPio import read_incar, update_incar, incar_dict_to_str


class StageError(RuntimeError):
    """Raised while advancing a stage."""


#: Magnetic elements from course notes section 3, prescription 2. ISPIN=2 if
#: any of these is present.
MAGNETIC_ELEMENTS = ("Fe", "Co", "Ni", "Cr", "Mn")

#: Elements whose ground state a plain ferromagnetic initialisation does not give.
AFM_CAUTION = ("Cr", "Mn")

#: Large files VASP regenerates. Deleted when moving on to the next stage.
BULKY_FILES = ("CHG", "CHGCAR", "WAVECAR", "DOSCAR", "EIGENVAL", "PROCAR",
               "PCDAT", "XDATCAR", "IBZKPT", "REPORT", "vasprun.xml")

#: Files kept with a suffix attached when moving on to the next stage.
ARCHIVE_FILES = ("INCAR", "POSCAR", "CONTCAR", "OUTCAR", "OSZICAR", "KPOINTS")

#: INCAR for the relaxation stage (course notes section 3.2).
RELAX_SETTINGS = {
    "PREC": "Accurate",
    "EDIFF": "1E-06",
    "EDIFFG": "-0.02",
    "IBRION": 2,
    "NSW": 200,
    "ISIF": 3,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LREAL": ".FALSE.",
    "LORBIT": 0,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

#: INCAR for the final static stage. ISMEAR is decided from the k-point count.
STATIC_SETTINGS = {
    "IBRION": -1,
    "NSW": 0,
    "EDIFF": "1E-07",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}


# ----------------------------------------------------------------------------
# ISPIN decision
# ----------------------------------------------------------------------------

def suggest_ispin(elements):
    """Decide ISPIN from a list of elements.

    Returns
    -------
    (ispin, notes)
    """
    elements = [str(e) for e in elements]
    found = [e for e in MAGNETIC_ELEMENTS if e in elements]
    if not found:
        return 1, ["No magnetic element, so ISPIN=1 (%s). Leaving it on "
                   "just doubles the computation." % ", ".join(sorted(set(elements)))]
    notes = ["Magnetic element(s) %s present, so ISPIN=2." % ", ".join(found)]
    afm = [e for e in found if e in AFM_CAUTION]
    if afm:
        notes.append("%s is antiferromagnetic, so a plain ferromagnetic "
                     "initialisation is not its ground state. Be sure to "
                     "check OUTCAR's magnetization after the run." % ", ".join(afm))
    notes.append("Once the calculation finishes, check whether a moment "
                 "actually developed. If it converged to 0, ISPIN=1 is fine.")
    return 2, notes


# ----------------------------------------------------------------------------
# Building the INCAR
# ----------------------------------------------------------------------------

def relax_incar(incar, ispin=None, extra=None):
    """Build the relaxation-stage INCAR dict."""
    incar_dict = _as_incar_dict(incar)
    settings = dict(RELAX_SETTINGS)
    if ispin is not None:
        settings["ISPIN"] = int(ispin)
    if extra:
        settings.update(extra)
    return update_incar(incar_dict, settings)


def static_incar(incar, nkpts=None, ispin=None, extra=None):
    """Build the final static-stage INCAR dict.

    ``ISMEAR=-5`` (tetrahedron method + Blochl correction) can only be used
    with 4 or more k-points. If there are fewer, ``ISMEAR=1`` is kept and
    SIGMA is lowered (course notes section 3.2).
    """
    incar_dict = _as_incar_dict(incar)
    settings = dict(STATIC_SETTINGS)

    if nkpts is not None and int(nkpts) < 4:
        settings["ISMEAR"] = 1
        settings["SIGMA"] = 0.05
    else:
        settings["ISMEAR"] = -5

    if ispin is not None:
        settings["ISPIN"] = int(ispin)
    if extra:
        settings.update(extra)
    return update_incar(incar_dict, settings)


def write_incar(incar_dict, path="INCAR"):
    with open(path, "w") as f:
        f.write(incar_dict_to_str(incar_dict))
    return path


def _as_incar_dict(incar):
    if isinstance(incar, dict):
        return dict(incar)
    if isinstance(incar, str):
        if not os.path.isfile(incar):
            raise StageError("INCAR is missing: %s" % incar)
        return read_incar(incar)
    raise StageError("Cannot use this as an INCAR: %s" % type(incar).__name__)


# ----------------------------------------------------------------------------
# Reading state
# ----------------------------------------------------------------------------

def is_finished(directory="."):
    """Judge completion by vasp.done, per CCpy's own convention."""
    return os.path.isfile(os.path.join(directory, "vasp.done"))


def cell_volume(path):
    """Cell volume of a POSCAR / CONTCAR (A^3)."""
    if not os.path.isfile(path):
        raise StageError("Structure file is missing: %s" % path)
    lines = open(path).read().split("\n")
    if len(lines) < 5:
        raise StageError("Structure file is too short: %s" % path)
    try:
        scale = float(lines[1].split()[0])
        mat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    except (IndexError, ValueError):
        raise StageError("Could not read the lattice from %s." % path)
    return abs(float(np.linalg.det(mat))) * (scale ** 3 if scale > 0 else 1.0)


def volume_drift(directory="."):
    """Volume change (%) of CONTCAR (after relaxation) relative to POSCAR
    (starting point).

    Once running one more stage makes this small enough, Pulay stress has
    been removed.
    """
    v0 = cell_volume(os.path.join(directory, "POSCAR"))
    v1 = cell_volume(os.path.join(directory, "CONTCAR"))
    return (v1 - v0) / v0 * 100.0


def _grep_last(path, needle):
    last = None
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            if needle in line:
                last = line
    return last


def nkpts(directory="."):
    """Read the number of irreducible k-points from OUTCAR. None if missing."""
    path = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(path):
        return None
    line = _grep_last(path, "NKPTS =")
    if not line:
        return None
    try:
        return int(line.split("NKPTS =")[1].split()[0])
    except (IndexError, ValueError):
        return None


def natoms(directory="."):
    """Number of atoms in the POSCAR."""
    path = os.path.join(directory, "POSCAR")
    if not os.path.isfile(path):
        raise StageError("POSCAR is missing: %s" % path)
    lines = open(path).read().split("\n")
    for idx in (5, 6):
        parts = lines[idx].split() if idx < len(lines) else []
        if parts and all(p.isdigit() for p in parts):
            return sum(int(p) for p in parts)
    raise StageError("Could not read the atom count from %s." % path)


def entropy_per_atom(directory="."):
    """OUTCAR's last entropy T*S, per atom, in meV.

    Course notes section 3, prescription 3 considers SIGMA appropriate when
    this is under 1 meV per atom. None if missing.
    """
    path = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(path):
        return None
    line = _grep_last(path, "entropy T*S")
    if not line:
        return None
    try:
        value = float(line.split("=")[-1].split()[0])
    except (IndexError, ValueError):
        return None
    return abs(value) / natoms(directory) * 1000.0


def unconverged_dirs(root=".", filename="01_unconverged_jobs.csv"):
    """Read the unconverged list left by CCpyVASPAnal.

    Returns an empty set when the file is missing, and it is the caller's job
    to say so.
    """
    path = os.path.join(root, filename)
    if not os.path.isfile(path):
        return None
    out = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            name = (row.get("Directory") or "").strip()
            if name:
                out.add(name)
    return out


# ----------------------------------------------------------------------------
# Advancing a stage
# ----------------------------------------------------------------------------

def advance(directory, stage="relax", suffix=None, ispin=None,
            clean=True, extra=None):
    """Move one configuration to the next stage.

    Keeps the previous result under a suffix, moves CONTCAR to POSCAR, then
    writes the INCAR for the stage. ``vasp.done`` is removed (the queue
    script recreates it).

    Parameters
    ----------
    directory : str
    stage : str
        ``"relax"`` or ``"static"``.
    suffix : str
        Suffix to attach to the archived files. If omitted, counts up as
        ``_relax1``, ``_relax2``, ...
    ispin : int or None
    clean : bool
        Whether to delete the large output files.
    extra : dict or None
        Values to overlay onto the INCAR.

    Returns
    -------
    dict : what was done (drift, suffix, ismear, ...)
    """
    if stage not in ("relax", "static"):
        raise StageError("stage must be 'relax' or 'static': %r" % stage)
    if not os.path.isdir(directory):
        raise StageError("Directory is missing: %s" % directory)
    if not is_finished(directory):
        raise StageError(
            "%s has no vasp.done. The calculation has either not finished, "
            "or was resubmitted."
            % os.path.basename(directory))

    contcar = os.path.join(directory, "CONTCAR")
    if not os.path.isfile(contcar) or os.path.getsize(contcar) == 0:
        raise StageError("%s's CONTCAR is missing or empty."
                         % os.path.basename(directory))

    info = {"directory": directory, "stage": stage}
    try:
        info["volume_drift"] = volume_drift(directory)
    except StageError:
        info["volume_drift"] = None
    info["entropy_meV_per_atom"] = entropy_per_atom(directory)
    prev_nkpts = nkpts(directory)
    info["nkpts"] = prev_nkpts

    if suffix is None:
        suffix = _next_suffix(directory)
    info["suffix"] = suffix

    for name in ARCHIVE_FILES:
        src = os.path.join(directory, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(directory, name + suffix))

    shutil.copy(os.path.join(directory, "CONTCAR" + suffix),
                os.path.join(directory, "POSCAR"))
    shutil.copy(os.path.join(directory, "KPOINTS" + suffix),
                os.path.join(directory, "KPOINTS"))

    old_incar = os.path.join(directory, "INCAR" + suffix)
    if stage == "relax":
        incar_dict = relax_incar(old_incar, ispin=ispin, extra=extra)
        info["ismear"] = incar_dict.get("ISMEAR")
    else:
        incar_dict = static_incar(old_incar, nkpts=prev_nkpts, ispin=ispin,
                                  extra=extra)
        info["ismear"] = incar_dict.get("ISMEAR")
    write_incar(incar_dict, os.path.join(directory, "INCAR"))

    if clean:
        for name in BULKY_FILES:
            p = os.path.join(directory, name)
            if os.path.isfile(p):
                os.remove(p)
    done = os.path.join(directory, "vasp.done")
    if os.path.isfile(done):
        os.remove(done)

    return info


def _next_suffix(directory):
    n = 1
    while os.path.isfile(os.path.join(directory, "OUTCAR_relax%d" % n)) or \
            os.path.isfile(os.path.join(directory, "OUTCAR_static%d" % n)):
        n += 1
    return "_relax%d" % n


def advance_all(root=".", stage="relax", dirs=None, skip_unconverged=True,
                ispin=None, clean=True, extra=None):
    """Advance several configurations at once.

    Returns
    -------
    (done, skipped)
        done is the list of advance() results, skipped is a list of
        (name, reason).
    """
    if dirs is None:
        from CCpy.CASM.CASMkpoints import config_dirs
        dirs = config_dirs(root)
    if not dirs:
        raise StageError("Could not find any configuration directories.")

    bad = unconverged_dirs(root) if skip_unconverged else set()
    done, skipped = [], []
    for d in dirs:
        name = os.path.basename(d)
        if bad and name in bad:
            skipped.append((name, "CCpyVASPAnal marked it unconverged"))
            continue
        try:
            done.append(advance(d, stage=stage, ispin=ispin, clean=clean,
                                extra=extra))
        except StageError as err:
            skipped.append((name, str(err)))
    return done, skipped


def describe(done, skipped, limit=5):
    """advance_all's result as sentences for a human to read."""
    lines = ["Advanced %d to the next stage." % len(done)]
    if done:
        drifts = [d["volume_drift"] for d in done if d["volume_drift"] is not None]
        if drifts:
            lines.append("  volume change : %+.2f ~ %+.2f%%  (average %+.2f%%)"
                         % (min(drifts), max(drifts),
                            sum(drifts) / len(drifts)))
            big = [d for d in done
                   if d["volume_drift"] is not None and abs(d["volume_drift"]) > 1.0]
            if big:
                lines.append("  %d configuration(s) moved more than 1%% -- run "
                             "one more stage and check whether it settles." % len(big))
        ents = [d["entropy_meV_per_atom"] for d in done
                if d["entropy_meV_per_atom"] is not None]
        if ents:
            worst = max(ents)
            flag = "" if worst < 1.0 else "  <- over 1 meV. Lower SIGMA."
            lines.append("  entropy T*S : max %.3f meV/atom%s" % (worst, flag))
        ismears = sorted({str(d.get("ismear")) for d in done})
        lines.append("  ISMEAR : %s" % ", ".join(ismears))
    if skipped:
        lines.append("%d skipped:" % len(skipped))
        for name, why in skipped[:limit]:
            lines.append("  %-12s %s" % (name, why))
        if len(skipped) > limit:
            lines.append("  ... (%d more)" % (len(skipped) - limit))
    return "\n".join(lines)
