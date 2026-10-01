"""
AIMDIncar.py -- INCAR editing for the AIMD (NVT MD loop) job submitter.

The AIMD INCAR is not written by CCpy. It is built on the compute node by
`.AIMDLoop.py` through pymatgen's MITMDSet, and CCpy only supplies a small
override dictionary on top of it:

    user_incar = {"NCORE": 4, "ICHARG": 0, "PREC": "Normal"}

`CCpyJobSubmit.py 9 ... -incar` opens a settings sheet at submit time (on the
login node) that shows the INCAR MITMDSet would actually produce for the
chosen structure, lets the user edit it, and turns the edits back into that
override dictionary. The overrides are written to a YAML file that is shipped
with the job and merged into `user_incar` inside the generated script.

The sheet follows CCpyVASPInputGen's INCAR menu in what it shows (the full
INCAR, `KEY=value` edits, `#KEY=` to drop a tag, "n" to finish) and
CCpyAlloyGen's `run_wizard()` in how it is implemented (one screen, redrawn
after every edit, comma-chained edits, "q" to cancel).

Keys the NVT loop itself controls are locked: TEBEG / TEEND / NSW / SMASS are
rewritten per stage by the loop (heating uses SMASS=-1 with TEBEG=100 ->
TEEND=T, the production run uses SMASS=0 with TEBEG=TEEND=T), so an override
for them would be silently thrown away or would break the convergence logic.
"""

import os
import re
import sys

import yaml

version = sys.version
if version[0] == '3':
    raw_input = input

# -- Rewritten per stage by the NVT loop; an override here cannot survive.
LOCKED_KEYS = ("TEBEG", "TEEND", "NSW", "SMASS")

# -- Must stay in sync with NVTLoopQueScript.py. NCORE must divide the number of
#    MPI ranks or VASP falls back to NCORE=1; 4 divides every core count in use.
NCORE = 4
DEFAULT_USER_INCAR = {"NCORE": NCORE, "ICHARG": 0, "PREC": "Normal"}
SCREEN_USER_INCAR = {"NCORE": NCORE, "ICHARG": 0, "PREC": "Normal", "NELM": 60}

# -- Step counts of the production stage, used only to build a representative
#    INCAR for the sheet (NSW is locked, so its value is never taken from here).
NSW_PRODUCTION = 1000
NSW_PRODUCTION_SCREEN = 1000

OVERRIDE_FILENAME = ".AIMD_incar.yaml"


def default_user_incar(screen=False):
    """The hardcoded override dictionary the loop script starts from."""
    base = SCREEN_USER_INCAR if screen else DEFAULT_USER_INCAR
    return dict(base)


# --------------------------------------------------------------------------- #
#  value handling
# --------------------------------------------------------------------------- #

def _plain(value):
    """numpy scalars / pymatgen wrappers -> plain python, so yaml can dump it."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    for caster in (int, float):
        if isinstance(value, caster):
            return caster(value)
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return _plain(value.item())
    except Exception:
        pass
    return value


_TRUE = ("true", ".true.", "t", "yes", "y")
_FALSE = ("false", ".false.", "f", "no", "n")


def coerce_value(text, reference=None):
    """
    Turn a typed string into an INCAR value.

    `reference` is the value currently held by that key, used to keep the type
    stable (so ISPIN=2 stays an int and LWAVE=False stays a bool rather than
    becoming the string "False", which VASP would not understand the same way).
    """
    text = text.strip()
    if isinstance(reference, bool) or text.lower() in _TRUE + _FALSE:
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
    if "," in text or (" " in text.strip() and not isinstance(reference, str)):
        parts = [p for p in re.split(r"[,\s]+", text) if p != ""]
        if len(parts) > 1:
            return [coerce_value(p) for p in parts]
    if isinstance(reference, bool):
        return text
    try:
        if isinstance(reference, float):
            return float(text)
    except ValueError:
        pass
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def format_value(value):
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple)):
        return " ".join(format_value(v) for v in value)
    return str(value)


# --------------------------------------------------------------------------- #
#  INCAR construction
# --------------------------------------------------------------------------- #

def build_incar(structure_file, temp, user_incar=None, nsw=NSW_PRODUCTION, quiet=False):
    """
    The INCAR MITMDSet produces for the production stage of this run.

    Returns a plain dict, or None when it cannot be built (pymatgen missing,
    unreadable structure). The caller falls back to the override-only sheet.
    """
    try:
        from pymatgen.core.structure import Structure
        from pymatgen.io.vasp.sets import MITMDSet
    except Exception as exc:
        if not quiet:
            print("* pymatgen could not be loaded (%s)." % exc)
        return None
    try:
        structure = Structure.from_file(structure_file)
    except Exception as exc:
        if not quiet:
            print("* Could not read %s (%s)." % (structure_file, exc))
        return None
    try:
        inputset = MITMDSet(structure=structure,
                            start_temp=float(temp), end_temp=float(temp),
                            nsteps=int(nsw),
                            user_incar_settings=dict(user_incar or {}))
        return dict((k, _plain(v)) for k, v in dict(inputset.incar).items())
    except Exception as exc:
        if not quiet:
            print("* MITMDSet could not build the INCAR (%s)." % exc)
        return None


def diff_overrides(baseline, current, removed=()):
    """
    current - baseline, as a user_incar_settings dictionary.

    Keys in `removed` map to None, which is how pymatgen drops a tag that its
    default set would otherwise write.
    """
    overrides = {}
    for key, value in current.items():
        if key not in baseline or baseline[key] != value:
            overrides[key] = _plain(value)
    for key in removed:
        overrides[key] = None
    for key in LOCKED_KEYS:
        overrides.pop(key, None)
    return overrides


def write_overrides(overrides, path=OVERRIDE_FILENAME):
    with open(path, "w") as fo:
        yaml.dump(dict(overrides), fo, default_flow_style=False, sort_keys=False)
    return path


def load_overrides(path=OVERRIDE_FILENAME):
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as fi:
        return yaml.load(fi, Loader=yaml.FullLoader) or {}


# --------------------------------------------------------------------------- #
#  settings sheet
# --------------------------------------------------------------------------- #

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def run_incar_wizard(structure_file, temp, screen=False, structure_files=None):
    """
    Open the INCAR settings sheet and return the resulting override dictionary.

    Returns None when the user cancels (the caller then submits nothing).
    An empty dict is never returned as "cancelled" -- it means the defaults
    were accepted unchanged.

    `structure_files` is the full batch selection, shown so it is clear that
    one sheet is being filled in for every structure and every temperature.
    """
    user_incar = default_user_incar(screen)
    nsw = NSW_PRODUCTION_SCREEN if screen else NSW_PRODUCTION

    baseline = build_incar(structure_file, temp, user_incar={}, nsw=nsw)
    current = build_incar(structure_file, temp, user_incar=user_incar, nsw=nsw, quiet=True)

    full_incar = baseline is not None and current is not None
    if not full_incar:
        # -- Fall back to editing the override dictionary alone. The run itself
        #    is unaffected: the compute node builds the real INCAR either way.
        print(bcolors.WARNING +
              "* The full INCAR cannot be shown here, so only the override values are listed.\n"
              "  (The INCAR itself is still built by MITMDSet on the compute node.)" +
              bcolors.ENDC)
        baseline = {}
        current = dict(user_incar)

    removed = set()
    locked_values = dict((k, current[k]) for k in LOCKED_KEYS if k in current)

    def _sheet():
        print("\n" + "=" * 74)
        print(" CCpy AIMD INCAR settings")
        print("=" * 74)
        if structure_files and len(structure_files) > 1:
            print("  reference structure : %s   (+ %d more, all get these settings)"
                  % (structure_file, len(structure_files) - 1))
        else:
            print("  reference structure : %s" % structure_file)
        print("  temperature         : %s K%s" % (temp, "   (-screen)" if screen else ""))
        print("-" * 74)
        for key in sorted(current):
            if key in LOCKED_KEYS:
                continue
            # -- Without the full INCAR there is nothing to compare against, so the
            #    "differs from the default" mark would mark every line.
            changed = full_incar and (key not in baseline or baseline[key] != current[key])
            mark = " *" if changed else "  "
            print("  %-16s = %-24s%s" % (key, format_value(current[key]), mark))
        if removed:
            print("  --- dropped from INCAR " + "-" * 48)
            for key in sorted(removed):
                print("  # %-14s   (will not be written)" % key)
        if locked_values:
            print("  --- locked: set per stage by -T and the NVT loop " + "-" * 23)
            for key in LOCKED_KEYS:
                if key in locked_values:
                    print("  %-16s = %-24s  (not editable)" % (key, format_value(locked_values[key])))
        print("-" * 74)
        if full_incar:
            print("* marks a value that differs from the MITMDSet default.")
            print("  Heating stage (SMASS=-1, 100K -> %sK) gets the same edits." % temp)
        print('* Edit with KEY=value, several at once with commas (ex: ISPIN=2,ENCUT=400).')
        print('  "#KEY=" drops the tag from INCAR, "KEY=" alone restores the default.')

    while True:
        _sheet()
        print(bcolors.OKGREEN + '\n* Anything want to modify or add? else, enter "n" to submit'
              + '     (q = cancel)' + bcolors.ENDC)
        try:
            answer = raw_input(": ").strip()
        except EOFError:
            print("\nCancelled.")
            return None
        if answer.lower() in ("q", "quit", "exit"):
            print("Cancelled.")
            return None
        if answer.lower() in ("n", "no"):
            return diff_overrides(baseline, current, removed)
        if answer == "":
            continue

        # -- split on commas that are followed by the next `KEY=`, the same rule
        #    CCpyAlloyGen's sheet uses, so values that contain commas survive
        #    (MAGMOM=6*0.6, 2*5.0 stays one edit).
        pairs = re.split(r",(?=\s*#?\s*[A-Za-z_][A-Za-z0-9_]*\s*=)", answer)
        for pair in pairs:
            pair = pair.strip()
            if "=" not in pair:
                print(bcolors.FAIL + "[Input error] Not in `KEY=value` form: %r" % pair + bcolors.ENDC)
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            drop = key.startswith("#")
            key = key.lstrip("#").strip().upper()
            if not key:
                print(bcolors.FAIL + "[Input error] Empty key: %r" % pair + bcolors.ENDC)
                continue
            if key in LOCKED_KEYS:
                print(bcolors.FAIL +
                      "[Input error] %s is set per stage by -T and the NVT loop, so it cannot be edited here."
                      % key + bcolors.ENDC)
                continue
            if drop:
                if not full_incar:
                    print(bcolors.FAIL +
                          "[Input error] Dropping a tag needs the full INCAR, which could not be built."
                          + bcolors.ENDC)
                    continue
                removed.add(key)
                current.pop(key, None)
                continue
            if value == "":
                # restore the MITMDSet default
                removed.discard(key)
                if key in baseline:
                    current[key] = baseline[key]
                else:
                    current.pop(key, None)
                continue
            removed.discard(key)
            current[key] = coerce_value(value, current.get(key, baseline.get(key)))
