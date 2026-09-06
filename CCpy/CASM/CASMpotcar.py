# -*- coding: utf-8 -*-
"""Per-element POTCAR files for a CASM working directory.

mainclust builds each configuration's POTCAR by concatenating the
``POTCAR_<element>`` files it finds in the working directory, in the order the
configuration's species appear. Until now those files had to be put there by
hand, which was the last remaining manual step and the one that failed most
quietly: a missing ``POTCAR_<element>`` makes mainclust write a 0-byte POTCAR
without an error.

This module fills them in. For each element, in order:

1. ``POTCAR_<element>`` already in the working directory -- left alone.
2. the same name found in one of ``search_dirs`` (normally the folder the
   command was started from) -- copied in.
3. built from the POTCAR library, using the element -> variant mapping in the
   ``POTCAR:`` section of the CCpy yaml.

Step 3 reads the same yaml section that CCpyVASPInputGen and AlloyGen's
``-gen_potcar`` read, so the three cannot disagree about which pseudopotential
an element gets. The yaml is read directly rather than through AlloyGen
because AlloyGen imports ase, and CASM should not stop working on a machine
where ase is missing.
"""

import os
import shutil

from CCpy.VASP.VASPio import load_yaml


PREFIX = "POTCAR_"

#: Environment variable pointing at the POTCAR library directory.
LIBRARY_ENV = "CCpy_POTCAR_LIBRARY"


class PotcarError(RuntimeError):
    """Raised when a per-element POTCAR cannot be resolved."""


# ----------------------------------------------------------------------------
# yaml mapping and library
# ----------------------------------------------------------------------------

def variant_map(preset=None):
    """Element -> POTCAR variant name, from the CCpy yaml POTCAR section.

    Looks in the config folder first (``~/.CCpy_test/vasp/<preset or
    default.yaml>``) and falls back to the yaml shipped in the package, so a
    machine whose config folder has not been initialised yet still gets the
    same table CCpy would install there.

    Returns
    -------
    (mapping, source path)
    """
    from pathlib import Path
    from CCpy.Tools import CCpyConfig as cfg

    candidates = [os.path.join(str(cfg.vasp_config_dir()),
                               preset if preset else "default.yaml")]
    if preset:
        candidates.append(os.path.join(str(cfg.vasp_config_dir()), "default.yaml"))
    candidates.append(str(Path(__file__).resolve().parent.parent
                          / "VASP" / "vasp_default.yaml"))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            table = load_yaml(path, "POTCAR")
        except Exception:
            continue
        if table:
            return dict(table), path

    raise PotcarError(
        "No POTCAR section found in the CCpy yaml. Looked in:\n  "
        + "\n  ".join(candidates))


def library_path(explicit=None):
    """Directory holding the POTCAR library (one folder per variant).

    Resolution order: explicit argument, ``$CCpy_POTCAR_LIBRARY``, then the
    candidate list AlloyGen uses. AlloyGen is imported only for that list and
    the failure is tolerated, since it pulls in ase.
    """
    if explicit:
        if not os.path.isdir(explicit):
            raise PotcarError("POTCAR library not found: %s" % explicit)
        return explicit

    env = os.environ.get(LIBRARY_ENV)
    if env:
        if not os.path.isdir(env):
            raise PotcarError("$%s points at a missing directory: %s"
                              % (LIBRARY_ENV, env))
        return env

    tried = []
    try:
        from CCpy.VASP.AlloyGen import POTCAR_LIBRARY_CANDIDATES as cands
        for cand in cands:
            tried.append(cand)
            if os.path.isdir(cand):
                return cand
    except Exception:
        pass

    raise PotcarError(
        "Could not locate a POTCAR library."
        + ("\nLooked in:\n  " + "\n  ".join(tried) if tried else "")
        + "\nPass -potlib=<path>, or set $%s." % LIBRARY_ENV)


def parse_overrides(text):
    """'Fe:Fe_sv,Co:Co_pv' -> {'Fe': 'Fe_sv', 'Co': 'Co_pv'}."""
    out = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise PotcarError("Expected <element>:<variant>, got %r" % part)
        elt, variant = part.split(":", 1)
        out[elt.strip()] = variant.strip()
    return out


# ----------------------------------------------------------------------------
# Filling the working directory
# ----------------------------------------------------------------------------

def _usable(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def ensure(workdir=".", elements=None, search_dirs=(), library=None,
           overrides=None, preset=None, generate=True):
    """Make sure POTCAR_<element> exists in workdir for every element.

    Returns
    -------
    (records, notes)
        records is [(element, how, detail, bytes)] with `how` one of
        "kept", "copied", "built".
    """
    if not elements:
        raise PotcarError("No elements given.")

    records, missing = [], []
    for elt in elements:
        target = os.path.join(workdir, PREFIX + elt)
        if _usable(target):
            records.append((elt, "kept", target, os.path.getsize(target)))
            continue

        found = None
        for d in search_dirs:
            cand = os.path.join(d, PREFIX + elt)
            if _usable(cand) and os.path.abspath(cand) != os.path.abspath(target):
                found = cand
                break
        if found:
            shutil.copy(found, target)
            records.append((elt, "copied", found, os.path.getsize(target)))
            continue

        missing.append(elt)

    if not missing:
        return records, []

    if not generate:
        raise PotcarError(
            "No POTCAR for: %s\n"
            "  Put %s under %s, or drop -nopotcar to build them from the "
            "POTCAR library."
            % (", ".join(missing),
               " ".join(PREFIX + e for e in missing),
               os.path.abspath(workdir)))

    table, source = variant_map(preset)
    table.update(overrides or {})
    unknown = [e for e in missing if e not in table]
    if unknown:
        raise PotcarError(
            "The POTCAR mapping in %s has no entry for: %s\n"
            "  Add it to that yaml's POTCAR section, or give it for this run "
            "with -pot=%s"
            % (source, ", ".join(unknown),
               ",".join("%s:%s" % (e, e) for e in unknown)))

    lib = library_path(library)
    for elt in missing:
        variant = table[elt]
        src = os.path.join(lib, variant, "POTCAR")
        if not os.path.isfile(src):
            available = sorted(
                d for d in os.listdir(lib)
                if (d == elt or d.startswith(elt + "_"))
                and os.path.isfile(os.path.join(lib, d, "POTCAR")))
            raise PotcarError(
                "No POTCAR for %s -> %s in %s.\n  %s"
                % (elt, variant, lib,
                   ("Variants present for %s: %s" % (elt, ", ".join(available)))
                   if available else
                   ("No variant of %s exists in that library." % elt)))
        target = os.path.join(workdir, PREFIX + elt)
        shutil.copy(src, target)
        records.append((elt, "built", variant, os.path.getsize(target)))

    return records, ["variant mapping from %s" % source,
                     "POTCAR library %s" % lib]


def describe(records, notes, indent="       "):
    """One block of human-readable text for the ensure() result."""
    how_text = {"kept": "already there", "copied": "copied from",
                "built": "built from"}
    out = []
    for elt, how, detail, size in records:
        if how == "kept":
            tail = how_text[how]
        elif how == "copied":
            tail = "%s %s" % (how_text[how], os.path.dirname(detail) or ".")
        else:
            tail = "%s %s" % (how_text[how], detail)
        out.append("%s%-12s %-8s %s" % (indent, PREFIX + elt,
                                        "%.1f KB" % (size / 1024.0), tail))
    for n in notes:
        out.append("%s%s" % (indent, n))
    return "\n".join(out)
