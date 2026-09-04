#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

from CCpy.Tools.CCpyTools import selectInputs

version = sys.version
if version[0] == '3':
    raw_input = input


def _help():
    print("\nHow to use : " + os.path.basename(sys.argv[0])
          + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
[options]
1 : Generate configurations   one structure file -> the con* directories
                (PRIM . CSPECS . KPOINTS . INCAR -> mainclust enumeration
                 -> directory creation -> cell parameter averaging
                 -> KPOINTS distribution)
2 : Continue / touch up   when 1 was interrupted, or after hand-editing
9 : Generate prim.json    (new-style CASM format)

Calling `CCpyCASMInputGen.py 1` with no arguments walks you through it with
questions; giving even one sub_option skips the questions (for scripts).

Requirements -- the working directory needs a per-element POTCAR_<element>.
INCAR is written when absent (CCpy yaml defaults + the lecture notes, sec. 3);
an INCAR that is already there is left untouched. mainclust is looked for in
the order $CCpy_MAINCLUST / ~/.CCpy_test / working directory / $PATH.


[sub_options]
ex) CCpyCASMInputGen.py 1
    CCpyCASMInputGen.py 1 -str=CONTCAR -sc=2,1,1 -occ=Cu,Ir -y

    < METHOD >   default is Method 1 (asymmetric full enumeration)
    -sym        : Method 2 - keep symmetry and build a CSPECS
                  the default perturbs coordinates down to P1 and skips
                  CSPECS. A symmetric PRIM with a CSPECS changes the
                  enumeration count (an 8-site cell: 9 -> 21)

    < STRUCTURE >   [option 1]
    -str=FILE   : structure file              (DEFAULT : asked)
    -sc=#,#,#   : supercell multiples         (DEFAULT : aims for 8 sites)
                  FCC conventional cell (4 atoms) uses 2,1,1 /
                  BCC/HCP (2 atoms) uses 2,2,1. The configuration count
                  doubles per extra site (8 sites: 256 / 16 sites: 65536)
    -occ=A,B    : elements allowed on a site  (DEFAULT : asked)
                  an empty site is Vac. Per element : -occ=Li:Li,Vac -occ=C:C
    -title=NAME : PRIM line 1                 (DEFAULT : element names)
    -amp=#      : perturbation amplitude for P1        (DEFAULT : 0.001)

    < CSPECS >   [option 1, with -sym]
    -nn=#       : up through which neighbour shell     (DEFAULT : 1)
    -r=#        : specify the radius directly (angstrom)
    -sizes=#,#  : cluster sizes               (DEFAULT : 2,3,4)
    -sp=A,B     : measure distances between just these elements (Li-C: -sp=Li)

    < INCAR >   [option 1]
    -preset=F   : yaml to use                 (DEFAULT : default.yaml)
                  a name inside ~/.CCpy_test/vasp/
    -encut=#    : specify ENCUT directly (eV) (DEFAULT : POTCAR ENMAX x 1.3)
    -ispin=#    : specify ISPIN directly      (DEFAULT : 2 if magnetic)
    -incar      : rewrite INCAR even when one is already there

    < KPOINTS >
    -kl=#       : target k*L (angstrom)       (DEFAULT : 35, insulator 20)
    -kp=#,#,#   : specify the mesh directly

    < CELL PARAM >   [option 2]
    -a=#,#      : the two elements' lattice constants, given directly
    -ref=F,F    : the two elements' structure file paths
                  (DEFAULT : BULK/<element>/CONTCAR, else the built-in table)
    -iso        : scale all three axes by the same factor
                  (matches the original 03_cellparam.sh)

    < RUN >   [option 1]
    -norun      : only create input files, don't run mainclust
    -vol=#      : maximum supercell volume    (DEFAULT : 1)
    -y          : proceed without confirming the configuration count

    < PARTIAL >
    -only=A,B   : [1] prim / cspecs / kpoints / incar  (mainclust is not run)
                  [2] makedirs / cellparam / kpoints
--------------------------------------''')
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


#: Ask for confirmation once the configuration count goes past this. 512 for
#: an 8-site cell / 65536 for a 16-site cell.
CONFIG_WARN = 512


def _suggest_supercell(natoms, target=8):
    """Suggest a multiple that brings the site count to target.

    For the FCC conventional cell (4 atoms) this gives 2 1 1, and for
    BCC/HCP (2 atoms) it gives 2 2 1. If the atom count doesn't divide
    target, 1 1 1 is returned instead -- it is not forced to fit.
    """
    if natoms <= 0 or target % natoms:
        return (1, 1, 1)
    need = target // natoms
    for sc in ((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)):
        if sc[0] * sc[1] * sc[2] == need:
            return sc
    return (need, 1, 1)


def _confirm_size(n, opt):
    """Check whether the configuration count is manageable. It doubles with
    every extra mixed site."""
    if n <= CONFIG_WARN:
        return
    print("\n  ! There are %d configurations. Every extra mixed site doubles this." % n)
    print("    Consider shrinking the cell (-sc=2,1,1), or Method 2 (-sym), which uses symmetry to cut it down.")
    if opt.get("yes") or not opt["interactive"]:
        print("    Proceeding as is.")
        return
    if _ask("    Proceed anyway? (y/n)", "n").lower() not in ("y", "yes"):
        print("    Nothing was created.")
        quit()


def _parse_subopts(argv):
    """Read -key=value / -flag the same way CCpyVASPInputGen does.

    If even one sub_option was given, questions are skipped and defaults are
    used instead, since this needs to be callable from a script.
    """
    opt = {"only": None, "occ": {}, "occ_all": None,
           "interactive": not any(a.startswith("-") for a in argv)}
    for arg in argv:
        if not arg.startswith("-"):
            continue
        if arg.startswith("-str="):
            opt["str"] = arg.split("=", 1)[1]
        elif arg.startswith("-sc="):
            opt["sc"] = arg.split("=", 1)[1]
        elif arg.startswith("-occ="):
            val = arg.split("=", 1)[1]
            if ":" in val:
                elt, occ = val.split(":", 1)
                opt["occ"][elt] = occ.replace(",", " ")
            else:
                opt["occ_all"] = val.replace(",", " ")
        elif arg.startswith("-title="):
            opt["title"] = arg.split("=", 1)[1]
        elif arg == "-sym":
            opt["sym"] = True
        elif arg.startswith("-amp="):
            opt["amp"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-nn="):
            opt["nn"] = int(arg.split("=", 1)[1])
        elif arg.startswith("-r="):
            opt["r"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-sizes="):
            opt["sizes"] = [int(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-sp="):
            opt["sp"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("-kl="):
            opt["kl"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-kp="):
            opt["kp"] = [int(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-preset="):
            opt["preset"] = arg.split("=", 1)[1]
        elif arg.startswith("-encut="):
            opt["encut"] = int(float(arg.split("=", 1)[1]))
        elif arg.startswith("-ispin="):
            opt["ispin"] = int(arg.split("=", 1)[1])
        elif arg == "-incar":
            opt["incar"] = True
        elif arg == "-norun":
            opt["norun"] = True
        elif arg == "-y":
            opt["yes"] = True
        elif arg.startswith("-vol="):
            opt["vol"] = int(arg.split("=", 1)[1])
        elif arg == "-iso":
            opt["iso"] = True
        elif arg.startswith("-a="):
            opt["a"] = [float(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-ref="):
            opt["ref"] = [v.strip() for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-only="):
            opt["only"] = [v.strip().lower() for v in arg.split("=", 1)[1].split(",")]
        elif arg not in ("1", "2", "9"):
            print("\nUnknown option: %s" % arg)
            _help()
    return opt


def _wanted(opt, step):
    return opt["only"] is None or step in opt["only"]


# ---------------------------------------------------------------------------
# [1] Generate CASM inputs
# ---------------------------------------------------------------------------

def input_gen(opt):
    from CCpy.CASM.CASMprim import (Prim, make_prim, read_structure,
                                    break_symmetry, PrimError)
    from CCpy.CASM.CASMcspecs import (make_cspecs, describe_shells, Cspecs,
                                      CspecsError)
    from CCpy.CASM.CASMkpoints import (make_kpoints, Kpoints, config_dirs,
                                       distribute, verify_uniform,
                                       describe_uniformity, KpointsError)

    if opt["only"]:
        unknown = set(opt["only"]) - {"prim", "cspecs", "kpoints", "incar"}
        if unknown:
            print("\n-only has unknown value(s): %s" % ", ".join(sorted(unknown)))
            print("Please choose from prim / cspecs / kpoints / incar.")
            quit()

    # -- reference structure --------------------------------------------------
    src = opt.get("str")
    if src is None:
        if _wanted(opt, "prim") or not os.path.isfile("PRIM"):
            inputs = selectInputs([".xsd", ".cif", "POSCAR", "CONTCAR"], "./")
            if not inputs:
                print("Could not find a structure file (.cif / POSCAR / CONTCAR).")
                quit()
            src = inputs[0]
        else:
            src = "PRIM"
    if not os.path.isfile(src):
        print("%s is missing." % src)
        quit()

    made = []

    # -- PRIM -----------------------------------------------------------------
    if _wanted(opt, "prim"):
        try:
            _, _, species = read_structure(src)
        except PrimError as err:
            print("\n%s" % err)
            quit()
        ordered = []
        for s in species:
            if s not in ordered:
                ordered.append(s)
        print("\n* %s : %d atoms, elements %s" % (src, len(species), ", ".join(ordered)))

        if opt["occ"]:
            occupancy = dict((e, opt["occ"].get(e, e)) for e in ordered)
            missing = [e for e in ordered if e not in opt["occ"]]
            if missing:
                print("  Element(s) not in -occ are left as fixed sites: %s" % ", ".join(missing))
        elif opt["occ_all"]:
            occupancy = opt["occ_all"]
        elif opt["interactive"]:
            print("\n* Enter the elements allowed on each site. An empty site is Vac.")
            print("  (e.g. a binary alloy 'Cu Ir' / a Li site 'Li Vac' / a fixed site 'C')")
            occupancy = dict((e, _ask("  %s site" % e, e)) for e in ordered)
        else:
            print("\nNo -occ given, so every site is left fixed.")
            print("For a binary alloy, give something like -occ=Cu,Ir.")
            occupancy = dict((e, e) for e in ordered)

        guess = " ".join(str(v) for v in _suggest_supercell(len(species)))
        sc = opt.get("sc")
        if sc is None:
            sc = _ask("\n* Supercell multiples a b c"
                      "\n  (this cell has %d atom(s), so %s gives 8 sites)"
                      % (len(species), guess), guess) \
                if opt["interactive"] else guess
        try:
            nx, ny, nz = [int(v) for v in sc.replace(",", " ").split()]
        except ValueError:
            print("The multiples must be 3 integers: %r" % sc)
            quit()

        try:
            prim = make_prim(src, occupancy=occupancy, supercell=(nx, ny, nz),
                             title=opt.get("title"))
        except PrimError as err:
            print("\n%s" % err)
            quit()

        a, b, c = prim.lengths
        print("\n[PRIM] %d site(s) / lattice %.6f %.6f %.6f" % (len(prim), a, b, c))
        print("       " + " / ".join("%d site(s) %s" % (n, " ".join(o))
                                     for n, o in prim.groups()))
        mixed = len(prim.mixed_sites)
        if mixed:
            n_config = 2 ** mixed
            if opt.get("sym"):
                print("       %d mixed site(s) -> up to %d configuration(s) (fewer once symmetry is applied)"
                      % (mixed, n_config))
            else:
                print("       %d mixed site(s) -> %d configuration(s)" % (mixed, n_config))
                _confirm_size(n_config, opt)

        if os.path.isfile("PRIM"):
            os.rename("PRIM", "PRIM_backup")
        prim.write("PRIM")
        made.append("PRIM")

        if not opt.get("sym"):
            try:
                out, msgs = break_symmetry(prim, amplitude=opt.get("amp", 0.001))
            except PrimError as err:
                print("\n%s" % err)
                quit()
            if not os.path.isfile("PRIM_orig"):
                os.rename("PRIM", "PRIM_orig")
            out.write("PRIM")
            for m in msgs:
                print("       " + m)
            print("       Lowered to P1. The original is PRIM_orig.")
            print("       Enumeration uses P1; structure generation uses the original.")
            made.append("PRIM_orig")
        src_for_rest = prim              # measure distances/mesh from the structure before perturbing
    else:
        src_for_rest = Prim.read(src) if os.path.basename(src).startswith("PRIM") \
            else src

    # -- CSPECS -----------------------------------------------------------------
    if _wanted(opt, "cspecs") and opt.get("sym"):
        try:
            cs, shells = make_cspecs(src_for_rest, nshell=opt.get("nn", 1),
                                     sizes=opt.get("sizes", (2, 3, 4)),
                                     radius=opt.get("r"),
                                     species=opt.get("sp"))
        except (CspecsError, PrimError) as err:
            print("\n%s" % err)
            quit()
        if os.path.isfile("CSPECS"):
            os.rename("CSPECS", "CSPECS_backup")
        cs.write("CSPECS")
        radius = cs.radii[sorted(cs.radii)[0]]
        inside = sum(n for d, n in shells if d <= radius)
        print("\n[CSPECS] radius %g angstrom, sizes %s"
              % (radius, ", ".join(str(s) for s in sorted(cs.radii))))
        print(describe_shells(shells, limit=4))
        print("       %d neighbour(s) within the radius" % inside)
        made.append("CSPECS")

    # -- KPOINTS ------------------------------------------------------------
    if _wanted(opt, "kpoints"):
        try:
            kp, info = make_kpoints(src_for_rest, target=opt.get("kl", 35.0))
            if opt.get("kp"):
                kp = Kpoints(opt["kp"], mode=kp.mode, comment=kp.comment)
                info = ["You specified the mesh directly: %s"
                        % " ".join(str(v) for v in kp.mesh)]
        except (KpointsError, PrimError) as err:
            print("\n%s" % err)
            quit()
        if os.path.isfile("KPOINTS"):
            os.rename("KPOINTS", "KPOINTS_backup")
        kp.write("KPOINTS")
        print("\n[KPOINTS] %s %s" % (kp.mode, " ".join(str(v) for v in kp.mesh)))
        for m in info:
            print("       " + m)
        made.append("KPOINTS")

        dirs = config_dirs(".")
        if dirs:
            go = opt.get("dist")
            if go is None:
                go = _ask("\n* Copy this into the %d configuration(s)? (y/n)" % len(dirs), "y") \
                    .lower() in ("y", "yes") if opt["interactive"] else False
                if not opt["interactive"]:
                    print("       There are %d configuration(s). Give -dist to copy into them."
                          % len(dirs))
            if go:
                distribute(kp, dirs)
                ok, groups = verify_uniform(dirs)
                print("       " + describe_uniformity(groups))

    # -- INCAR --------------------------------------------------------------
    if _wanted(opt, "incar"):
        from CCpy.CASM import CASMincar as ic
        from CCpy.CASM.CASMrun import MainclustError
        try:
            written, notes = ic.make_incar(
                ".", preset=opt.get("preset"), encut=opt.get("encut"),
                ispin=opt.get("ispin"), overwrite=opt.get("incar", False))
            print("\n[INCAR]")
            print(ic.describe(written, notes))
            if written:
                made.append("INCAR")
        except (ic.IncarError, MainclustError) as err:
            print("\n[INCAR] %s" % err)
            print("        You can place an INCAR yourself and run again.")

    # -- summary --------------------------------------------------------------
    print("\nFiles created: %s" % ", ".join(made))

    if opt["only"] is not None or opt.get("norun"):
        print("\nmainclust was not run. To continue:")
        print("  CCpyCASMInputGen.py 1     start over from enumeration (regenerates PRIM)")
        print("  CCpyCASMInputGen.py 2     if already enumerated, continue from there")
        return

    build(opt, restore="PRIM_orig" in made)


# ---------------------------------------------------------------------------
# [1-2] Enumerate with mainclust and build the configuration directories
# ---------------------------------------------------------------------------

def _preflight(opt):
    """Check every requirement in one pass before running.

    mainclust doesn't raise an error even when requirements are missing --
    without POTCAR_<element> it makes a 0-byte POTCAR, and without
    INCAR/KPOINTS it skips creating directories entirely while still
    printing "done". Enumeration alone can take several minutes, so check
    everything before starting.
    """
    from CCpy.CASM import CASMrun as run

    try:
        binary = run.resolve_binary(workdir=".")
        potcars = run.check_potcar_sources(".")
        templates = run.check_templates(".")
    except run.MainclustError as err:
        print("\n%s" % err)
        quit()

    def _size(n):
        return "%.1f KB" % (n / 1024.0) if n >= 1024 else "%d B" % n

    print("\n* Requirements")
    print("    %-12s %s" % ("mainclust", binary))
    for name, path, size in templates:
        print("    %-12s %s" % (name, _size(size)))
    for elt, path, size in potcars:
        print("    %-12s %s" % (os.path.basename(path), _size(size)))
    return binary


def build(opt, restore=False):
    """Run mainclust twice to get all the way to con*.

    If restore is True (Method 1), PRIM_orig is restored after enumeration
    finishes. Method 1 enumerates as P1 and generates structures from the
    symmetric one, and skipping this restore step silently builds every
    configuration from the perturbed coordinates instead. In the original
    04_Asym_Alloy.sh this line was commented out, and that is exactly what
    happened.
    """
    from CCpy.CASM import CASMrun as run
    from CCpy.CASM.CASMkpoints import config_dirs

    if config_dirs("."):
        print("\nConfiguration directories (con*) already exist.")
        print("To rebuild, delete them and run again; to just touch things up, use option 2.")
        return

    _preflight(opt)

    if opt["interactive"] and not opt.get("yes"):
        if _ask("\n* Enumerate with mainclust and build the configuration directories? (y/n)",
                "y").lower() not in ("y", "yes"):
            print("  Stopped, leaving only the input files.")
            return

    try:
        print("\n[Enumerate] mainclust ...")
        res = run.enumerate_configurations(".", max_volume=opt.get("vol", 1),
                                           dimension=3)
        print("       %s" % res.summary().replace("\n", "\n       "))

        if restore and os.path.isfile("PRIM_orig"):
            shutil.copy("PRIM_orig", "PRIM")
            print("       Restored PRIM to the symmetric original (structures will be built from it).")

        changed, total = run.set_make_flags("make_dirs")
        print("\n[make_dirs] Set %d / %d to 1 (the original is make_dirs_orig)."
              % (changed, total))

        print("\n[Generate] mainclust ... %d configuration(s)" % total)
        res2 = run.generate_vasp_inputs(".", energy=0, reference=0)
        print("       %s" % res2.summary().replace("\n", "\n       "))
        print(res2.potcar_report)
    except run.MainclustError as err:
        print("\n%s" % err)
        print("\nOnce fixed, you can continue with option 2.")
        quit()

    dirs = config_dirs(".")
    print("\n%d configuration directory(ies)" % len(dirs))
    _finish(opt, dirs)


# ---------------------------------------------------------------------------
# [2] Touch up after enumeration
# ---------------------------------------------------------------------------

def post_enum(opt):
    from CCpy.CASM.CASMrun import set_make_flags, MainclustError
    from CCpy.CASM import CASMcellparam as cp
    from CCpy.CASM.CASMkpoints import (Kpoints, config_dirs, distribute,
                                       verify_uniform, describe_uniformity,
                                       KpointsError)

    if opt["only"]:
        unknown = set(opt["only"]) - {"makedirs", "cellparam", "kpoints"}
        if unknown:
            print("\n-only has unknown value(s): %s" % ", ".join(sorted(unknown)))
            print("Please choose from makedirs / cellparam / kpoints.")
            quit()

    dirs = config_dirs(".")

    # -- make_dirs flags -------------------------------------------------------
    total = None
    if _wanted(opt, "makedirs"):
        try:
            changed, total = set_make_flags("make_dirs")
            print("\n[make_dirs] Set %d / %d to 1 (the original is make_dirs_orig)."
                  % (changed, total))
        except MainclustError as err:
            print("\n[make_dirs] %s" % err)

    # -- if directories don't exist yet, build them here -----------------------
    if not dirs and total:
        from CCpy.CASM import CASMrun as run
        if os.path.isfile("PRIM_orig"):
            shutil.copy("PRIM_orig", "PRIM")
            print("            Restored PRIM to the symmetric original.")
        _preflight(opt)
        go = True
        if opt["interactive"] and not opt.get("yes"):
            go = _ask("\n* Build the %d configuration(s)? (y/n)" % total, "y") \
                .lower() in ("y", "yes")
        if go:
            try:
                print("\n[Generate] mainclust ...")
                res = run.generate_vasp_inputs(".", energy=0, reference=0)
                print("       %s" % res.summary().replace("\n", "\n       "))
                print(res.potcar_report)
            except run.MainclustError as err:
                print("\n%s" % err)
                quit()
            dirs = config_dirs(".")

    if not dirs:
        if _wanted(opt, "cellparam") or _wanted(opt, "kpoints"):
            print("\nNo configuration directories (con*) yet, so the rest is skipped.")
        return

    _finish(opt, dirs)


def _finish(opt, dirs):
    """Touch-up after configuration directories exist -- cell parameter
    averaging and KPOINTS distribution."""
    from CCpy.CASM import CASMcellparam as cp
    from CCpy.CASM.CASMkpoints import (Kpoints, distribute, verify_uniform,
                                       describe_uniformity, KpointsError)

    # -- cell parameter averaging ----------------------------------------------
    if _wanted(opt, "cellparam"):
        lattice, refs = None, None
        elts = None
        try:
            elts = cp.elements_from_prim("PRIM")[:2]
        except cp.CellparamError as err:
            print("\n[cellparam] %s" % err)
            elts = None
        if elts:
            if opt.get("a"):
                if len(opt["a"]) != 2:
                    print("\n-a needs exactly 2 lattice constants, one per element: -a=3.6271,3.8707")
                    quit()
                lattice = dict(zip(elts, opt["a"]))
            if opt.get("ref"):
                if len(opt["ref"]) != 2:
                    print("\n-ref needs exactly 2 structure file paths, one per element.")
                    quit()
                refs = dict(zip(elts, opt["ref"]))
            try:
                rec, notes, skipped = cp.apply(".", lattice=lattice, refs=refs,
                                               isotropic=opt.get("iso", False),
                                               dirs=dirs)
                print("\n[cellparam]")
                print("  " + cp.describe(rec, notes, skipped).replace("\n", "\n  "))
            except cp.CellparamError as err:
                print("\n[cellparam] %s" % err)

    # -- KPOINTS distribution ---------------------------------------------------
    if _wanted(opt, "kpoints"):
        if not os.path.isfile("KPOINTS"):
            print("\n[KPOINTS] KPOINTS is missing. Create it first with option 1.")
        else:
            try:
                kp = Kpoints.read("KPOINTS")
                if opt.get("kp"):
                    kp = Kpoints(opt["kp"], mode=kp.mode, comment=kp.comment)
                    kp.write("KPOINTS")
                distribute(kp, dirs)
                ok, groups = verify_uniform(dirs)
                print("\n[KPOINTS] Copied %s %s into %d configuration(s)."
                      % (kp.mode, " ".join(str(v) for v in kp.mesh), len(dirs)))
                print("  " + describe_uniformity(groups).replace("\n", "\n  "))
            except KpointsError as err:
                print("\n[KPOINTS] %s" % err)

    print("\nNext step -- submit:")
    print("  CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")


# ---------------------------------------------------------------------------
# [9] prim.json (new-style CASM)
# ---------------------------------------------------------------------------

def prim_json_gen():
    from CCpy.CASM.CASMio import CASMInput

    inputs = selectInputs([".xsd", ".cif", "POSCAR", "CONTCAR"], "./")
    for each_input in inputs:
        CASMInput(each_input).primGen()


if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        _help()

    if option == "1":
        input_gen(_parse_subopts(sys.argv[2:]))
    elif option == "2":
        post_enum(_parse_subopts(sys.argv[2:]))
    elif option == "9":
        prim_json_gen()
    else:
        print("\nUnknown option: %s" % option)
        _help()
