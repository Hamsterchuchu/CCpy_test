# -*- coding: utf-8 -*-
"""INCAR generation for CASM configuration calculations.

mainclust copies the INCAR of the working directory into every configuration. So
the INCAR only has to be built well once, before the calculations, but until now
it was left there by hand or copied over from an older folder. As a result the
items that section 3 of the lecture notes asks to fix differed from folder to
folder -- the 256 configurations of the practice folder had no ENCUT and had
ISPIN=2 while every magnetization was 0.

Here **only the prescriptions of the lecture notes are layered on top of the yaml
defaults CCpy already uses.** The INCAR is not defined anew. That way the format
and the defaults do not drift apart from other calculations built with
CCpyVASPInputGen, and when the user edits ``~/.CCpy_test/vasp/default.yaml`` the
CASM side follows along too.

What is layered on (section 3 of the lecture notes, "the seven things changed
from the original source"):

===========  ==========================================================
ENCUT        largest ENMAX of the POTCAR x 1.3, rounded up to 10
             (CCpy's ENCUT_SCALE / ENCUT_ROUND are used as they are)
ISPIN        2 only when a magnetic element (Fe, Co, Ni, Cr, Mn) exists
ISMEAR/SIGMA relaxation 1 / 0.1, static -5 (1 / 0.05 if few k-points)
ISIF         3
LREAL        .FALSE.
EDIFF        1E-06, final static calculation 1E-07
EDIFFG       -0.02
===========  ==========================================================

**ENCUT is decided by looking at every element of the PRIM.** It must not be
decided per configuration -- a pure end member has only one element, so its ENCUT
comes out low, and then the reference points of the formation energy alone are
computed with a different cutoff. Every configuration on the same hull has to use
the same ENCUT. The structure of keeping a single INCAR in the working directory
and letting mainclust copy it happens to match that requirement.
"""

import os
import re

from CCpy.VASP.VASPio import (ENCUT_SCALE, ENCUT_ROUND, round_encut, num_to_str,
                              load_yaml, read_incar, update_incar,
                              incar_dict_to_str)
from CCpy.CASM.CASMstage import (RELAX_SETTINGS, STATIC_SETTINGS,
                                 suggest_ispin)


class IncarError(RuntimeError):
    """Error raised while generating an INCAR."""


_ENMAX_RE = re.compile(r"ENMAX\s*=\s*([0-9.]+)")


# ----------------------------------------------------------------------------
# ENCUT
# ----------------------------------------------------------------------------

def read_enmax(path):
    """Largest ENMAX (eV) in a POTCAR file. None if absent.

    The largest value is returned even when several elements are concatenated
    into one file.
    """
    try:
        with open(path, errors="ignore") as f:
            found = [float(v) for v in _ENMAX_RE.findall(f.read())]
    except OSError:
        return None
    return max(found) if found else None


def suggest_encut(workdir=".", elements=None, prefix="POTCAR_"):
    """Decide ENCUT by reading the per-element POTCAR files.

    Returns
    -------
    (encut, detail)
        detail is [(element, ENMAX, rounded value)].
    """
    from CCpy.CASM.CASMrun import prim_elements
    if elements is None:
        elements = prim_elements(workdir)

    detail, missing = [], []
    for elt in elements:
        path = os.path.join(workdir, prefix + elt)
        enmax = read_enmax(path)
        if enmax is None:
            missing.append(prefix + elt)
            continue
        detail.append((elt, enmax, round_encut(enmax * ENCUT_SCALE)))

    if missing:
        raise IncarError(
            "Could not read ENMAX: %s\n"
            "  The POTCAR is missing or has a different format. You may also give it directly with -encut=."
            % ", ".join(missing))
    if not detail:
        raise IncarError("Not a single element was found.")

    return max(d[2] for d in detail), detail


# ----------------------------------------------------------------------------
# INCAR
# ----------------------------------------------------------------------------

def base_incar(preset=None):
    """Read CCpy's yaml INCAR defaults (``~/.CCpy_test/vasp/``).

    Given a preset, that yaml is used, otherwise default.yaml. When the config
    folder does not have default.yaml yet, the vasp_default.yaml shipped in the
    package is copied to create it -- the same thing CCpyVASPInputGen does on its
    first run.
    """
    from pathlib import Path
    import shutil
    from CCpy.Tools import CCpyConfig as cfg

    conf = str(cfg.vasp_config_dir())
    if not os.path.isdir(conf):
        os.makedirs(conf, exist_ok=True)
    target = os.path.join(conf, preset if preset else "default.yaml")
    if not os.path.isfile(target):
        if preset:
            raise IncarError("%s does not exist. Give the name of a yaml inside %s."
                             % (target, conf))
        pkg = str(Path(__file__).resolve().parent.parent / "VASP" / "vasp_default.yaml")
        shutil.copy(pkg, target)
        print("* Created new INCAR defaults: %s" % target)

    incar = load_yaml(target, "INCAR")
    if not incar:
        raise IncarError("%s has no INCAR section." % target)
    return incar, target


def casm_settings(stage="relax", elements=None, encut=None, nkpts=None):
    """Build the items that layer the prescriptions of the lecture notes on top.

    Returns
    -------
    (settings, notes)
    """
    if stage not in ("relax", "static"):
        raise IncarError("stage must be 'relax' or 'static': %r" % stage)

    settings = dict(RELAX_SETTINGS)
    notes = []

    if stage == "static":
        settings.update(STATIC_SETTINGS)
        # 사면체법은 k-점이 너무 적으면 쓸 수 없다.
        if nkpts is not None and nkpts < 4:
            settings["ISMEAR"] = 1
            settings["SIGMA"] = 0.05
            notes.append("Only %d k-points, so ISMEAR=1 and SIGMA=0.05 are kept." % nkpts)
        else:
            settings["ISMEAR"] = -5
            settings.pop("SIGMA", None)
            notes.append("Final static calculation: NSW=0, ISMEAR=-5, EDIFF=1E-07")
    else:
        notes.append("Relaxation: ISIF=3, ISMEAR=1/SIGMA=0.1, EDIFF=1E-06, EDIFFG=-0.02")

    if encut is not None:
        settings["ENCUT"] = num_to_str(encut)

    if elements:
        ispin, ispin_notes = suggest_ispin(elements)
        settings["ISPIN"] = ispin
        notes.extend(ispin_notes)

    return settings, notes


def make_incar(workdir=".", elements=None, stage="relax", preset=None,
               encut=None, ispin=None, nkpts=None, extra=None,
               path="INCAR", overwrite=False):
    """Write the INCAR for CASM into the working directory.

    Parameters
    ----------
    elements : list or None
        None reads them from the PRIM.
    encut : int or None
        None decides it from the ENMAX of the POTCAR.
    ispin : int or None
        Given, it overrides the automatic choice.
    overwrite : bool
        False with the file already there does nothing and returns (False, []).

    Returns
    -------
    (written, notes)
    """
    from CCpy.CASM.CASMrun import prim_elements

    full = path if os.path.isabs(path) else os.path.join(workdir, path)
    if os.path.isfile(full) and not overwrite:
        return False, ["Already there, left as it is. Give -incar to build it again."]

    if elements is None:
        elements = prim_elements(workdir)

    notes = []
    if encut is None:
        encut, detail = suggest_encut(workdir, elements)
        notes.append("ENCUT %s  (ENMAX %s times %.1f, rounded up to %d)"
                     % (num_to_str(encut),
                        ", ".join("%s %.1f" % (e, v) for e, v, _ in detail),
                        ENCUT_SCALE, ENCUT_ROUND))
    else:
        notes.append("ENCUT %s  (given directly)" % num_to_str(encut))

    incar, source = base_incar(preset)
    settings, more = casm_settings(stage=stage, elements=elements,
                                   encut=encut, nkpts=nkpts)
    if ispin is not None:
        settings["ISPIN"] = int(ispin)
        more = [m for m in more if "ISPIN" not in m]
        more.append("ISPIN %d was given directly." % int(ispin))
    if extra:
        settings.update(extra)

    incar = update_incar(incar, settings)
    with open(full, "w") as f:
        f.write(incar_dict_to_str(incar))

    notes.append("defaults taken from %s" % source)
    notes.extend(more)
    return True, notes


def describe(written, notes, path="INCAR"):
    head = "Created %s." % path if written else "%s was left untouched." % path
    return "\n".join(["       " + head] + ["       " + n for n in notes])
