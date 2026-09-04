#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advance CASM configuration calculations to the next stage.

Stopping after a single relaxation leaves the volume less than fully
converged because of Pulay stress, and the smearing used for relaxation
leaves an artificial entropy term in the final energy. This command takes a
finished calculation and rewrites it as the next stage's input -- the
calculation itself still needs to be resubmitted.
"""

import os
import sys

version = sys.version
if version[0] == '3':
    raw_input = input


def _help():
    print("\nHow to use : " + os.path.basename(sys.argv[0])
          + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
[options]
1 : Prepare the next relaxation   (CONTCAR -> POSCAR, removes Pulay stress)
2 : Prepare the final static      (settles the energy with NSW=0, ISMEAR=-5)

The finished calculation's output is kept with a suffix attached
(OUTCAR_relax1 ...). Convergence is judged by reading
01_unconverged_jobs.csv, which CCpyVASPAnal.py 0 leaves behind -- run that
once before this command and unconverged configurations are skipped
automatically.


[sub_options]
ex) CCpyCASMStage.py 1 -ispin=1 -y

    -ispin=#  : set ISPIN                  (DEFAULT : keep the existing value)
                1 is enough without a magnetic element (Fe/Co/Ni/Cr/Mn)
    -force    : also proceed with unconverged configurations
    -keep     : don't delete large files such as CHG/CHGCAR/WAVECAR
    -y        : proceed without confirming
    -dir=A,B  : only these configurations   (DEFAULT : all of con*)
--------------------------------------''')
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


def _parse_subopts(argv):
    opt = {}
    for arg in argv:
        if not arg.startswith("-"):
            continue
        if arg.startswith("-ispin="):
            opt["ispin"] = int(arg.split("=", 1)[1])
        elif arg == "-force":
            opt["force"] = True
        elif arg == "-keep":
            opt["keep"] = True
        elif arg == "-y":
            opt["yes"] = True
        elif arg.startswith("-dir="):
            opt["dirs"] = [v.strip() for v in arg.split("=", 1)[1].split(",")]
        else:
            print("\nUnknown option: %s" % arg)
            _help()
    return opt


def run(stage, opt):
    from CCpy.CASM import CASMstage as st
    from CCpy.CASM.CASMkpoints import config_dirs

    dirs = [d for d in opt["dirs"]] if opt.get("dirs") else config_dirs(".")
    if not dirs:
        print("\nCould not find any configuration directories (con*).")
        quit()

    ready = [d for d in dirs if st.is_finished(d)]
    print("\n* %d of %d configuration(s) are finished (by vasp.done)."
          % (len(ready), len(dirs)))
    if len(ready) < len(dirs):
        print("  Skipping %d not yet finished." % (len(dirs) - len(ready)))
    if not ready:
        quit()

    skip_unconverged = not opt.get("force")
    bad = st.unconverged_dirs(".")
    if bad is None:
        print("\n  01_unconverged_jobs.csv is missing.")
        print("  Run CCpyVASPAnal.py 0 first and unconverged configurations will be skipped automatically.")
        if not opt.get("yes"):
            if _ask("  Proceed anyway? (y/n)", "n").lower() not in ("y", "yes"):
                quit()
    elif skip_unconverged:
        print("  Skipping %d configuration(s) marked unconverged." % len(bad))
    else:
        print("  -force given, so proceeding with the %d unconverged configuration(s) too." % len(bad))

    sample = ready[:3]
    print("\n  Current state (first %d):" % len(sample))
    for d in sample:
        try:
            ent = st.entropy_per_atom(d)
            print("    %-10s volume change %+6.2f %%   T*S %-7s meV/atom   NKPTS %s"
                  % (os.path.basename(d), st.volume_drift(d),
                     "%.3f" % ent if ent is not None else "-", st.nkpts(d)))
        except st.StageError as err:
            print("    %-10s %s" % (os.path.basename(d), err))

    if not opt.get("yes"):
        if _ask("\n* Proceed? The previous results are kept with a suffix attached. (y/n)",
                "y").lower() not in ("y", "yes"):
            print("  Did nothing.")
            quit()

    done, skipped = st.advance_all(
        ".", stage=stage, dirs=dirs, skip_unconverged=skip_unconverged,
        ispin=opt.get("ispin"), clean=not opt.get("keep"))
    print("\n" + st.describe(done, skipped))

    if not done:
        return
    print("\n  Now resubmit:")
    print("    CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")
    if stage == "relax":
        print("\n  Once finished, check whether the volume is still moving; once it")
        print("  settles, use option 2 to prepare the final static calculation.")
    else:
        ismears = sorted({str(d.get("ismear")) for d in done})
        if "1" in ismears:
            print("\n  Configuration(s) with fewer than 4 k-points can't use the")
            print("  tetrahedron method (-5), so ISMEAR=1, SIGMA=0.05 was used instead.")
        print("\n  This calculation's OUTCAR is the final energy. CCpyCASMhull.py")
        print("  reads OUTCAR directly, so the hull plots with the static energy without any further changes.")


if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        _help()

    opt = _parse_subopts(sys.argv[2:])
    if option == "1":
        run("relax", opt)
    elif option == "2":
        run("static", opt)
    else:
        print("\nUnknown option: %s" % option)
        _help()
