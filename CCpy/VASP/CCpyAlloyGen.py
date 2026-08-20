#!/usr/bin/env python
"""
CCpyAlloyGen.py

CCpy-style front-end for alloy / HEA substitution structure generation
(CCpy.VASP.AlloyGen, based on Gen-HEA-pomepaw_v8.py), optionally followed by
full CCpy VASP input generation (INCAR / KPOINTS / POTCAR) for every
generated structure using the same VASPInput machinery as CCpyVASPInputGen.py.
"""
import os
import sys

version = sys.version
if version[0] == '3':
    raw_input = input


# No arguments -> print this usage reference, the same habit as
# CCpyVASPInputGen.py. Pick a mode (1-5 or w) to open the settings sheet.
if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help", "help", "-help"):
    # The preset folder is resolved by CCpy/Tools/CCpyConfig.py, so the help
    # text asks for it instead of hardcoding a path that can go stale.
    from CCpy.VASP.AlloyGen import vasp_preset_dir_label
    _PRESET_DIR = vasp_preset_dir_label()
    print("\nHow to use : " + sys.argv[0].split("/")[-1] + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
[options]   -- every option opens the settings sheet; the number just presets 'mode'
1 : random     (fully random substitution)
2 : spread     (same-element dispersed, spread-biased substitution)
3 : layered    (layer-ordered parent -> disorder controlled by target Q)
4 : domain     (top-view 2x2 domain / 5-element quincunx template)
5 : exhaustive (enumerate all symmetry-unique configurations)
w : wizard     (settings sheet with no mode preset -- same as '1')

(no option) : print this help.  -h does the same.

Every option opens the one-screen settings sheet, where the input file is
picked from a numbered list and every setting is shown at once. Sub-options
pre-fill that sheet rather than running behind your back, so
`CCpyAlloyGen.py 4 -n=100` opens it already set to domain mode with n=100.
Edit anything with key=value, then enter "n" to run. Add -batch to skip every
question and run immediately (then -i= is required).

[sub_options]
ex) CCpyAlloyGen.py 1 -i=Pt32.cif -re=Pt -comp=Fe4,Co4,Ni4,Cu4 -n=500 -vasp -preset=alloy

    < STRUCTURE GENERATION >
    -i=[FILE]      : input structure file (.cif, .vasp, POSCAR/CONTCAR).
                     Omit it and the structures in this directory are listed
                     with numbers to pick from, the way CCpyVASPInputGen does.
    -o=[DIR]       : output directory              (DEFAULT : auto-suggested from input/mode/composition)
    -re=[EL,..]    : element(s) whose sites form the replaceable pool
                     (DEFAULT : the substrate elements of the input structure,
                      i.e. everything except the atoms judged to be adsorbates)
                     ex) -re=Pt        or  -re=Co,Fe,Ni,Cu (pool of an already-mixed HEA)
    -comp=[C]      : target replacement composition
                     (DEFAULT : what those sites currently hold, so the printed
                      value shows the real composition to edit from)
                     ex) -comp=Fe4,Co4,Ni4,Cu4  or  -comp=Fe:4,Co:4,Ni:4,Cu:4
    -keep_comp     : reshuffle the current composition of the -re pool (-comp is ignored)
    -n=#           : target number of unique structures                 (DEFAULT : 500)
    -seed=#        : random seed                   (DEFAULT : auto-generated, saved to metadata.txt)
    -fmt=[F]       : cif | vasp | folder           (DEFAULT : folder)
                     folder : conf dirs with POSCAR (S000001/POSCAR)
                     (ignored with -vasp: the folders get full VASP inputs)
    -overwrite     : remove the output directory first if it exists
    -surface=[EL,..]: also write adsorbate-free surface twins to [DIR]_surface.
                     A bare element here means ALL atoms of it -- a clean surface.
                     ex) -surface=Li,S  removes every Li and every S
    -surface       : same, but pick the elements interactively. Analyzes the input
                     structure (vacuum axis, substrate range, distances), proposes
                     the elements it judges to be adsorbates, and asks to confirm.
    -redox=[SPEC]  : also write redox twins, removing atoms from every generated
                     structure -- for comparing energies across a redox step.
                     Atom numbers are 1-based in the input file's coordinate order.

                     < removal composition >  fixes WHAT is taken away, and writes
                     every distinct choice to its own folder ([DIR]_r1, _r2, ...).
                     The count is REQUIRED here:
                         -redox=S1        remove 1 S      (Li2S4 -> Li2S3)
                         -redox=S2        remove 2 S      (Li2S4 -> Li2S2)
                         -redox=Li2,S1    remove 2 Li and 1 S
                         -redox=Li:2,S:1  same as Li2,S1 (as in -comp=)
                     A bare element (-redox=S) is refused on purpose: in -surface=
                     a bare element means "strip that species entirely", so letting
                     it mean "one atom" here would make two look-alike options do
                     opposite things. To remove every S, use -surface=S.

                     < exactly these atoms >  a single set -> [DIR]_r :
                         -redox=35        atom 35 only
                         -redox=35,36     both atoms
                         -redox=S1,35     atom 35 always, plus any 1 S

                     < any k of a chosen pool >  to restrict the candidate sites:
                         -redox=19,20,21,22/2  any 2 of those 4  -> _r1 .. _r6
                         -redox=S/2            any 2 of every S atom

                     A folder -> removed-atoms map is written to [DIR]/redox_sets.csv
    -redox_max=#   : safety limit on the number of combinations (DEFAULT : 50)
    -redox         : pick interactively -- lists every removable atom, takes a spec
                     in any of the forms above, and previews it before running
    -symprec=#     : spglib symmetry tolerance     (DEFAULT : 1e-3)
    -max_attempts=#: max sampling attempts         (DEFAULT : 2000000)

    < MODE DETAILS >
    -axis=[x|y|z]  : layer stacking axis           (mode 3, DEFAULT : z)
    -view=[x|y|z]  : top-view axis                 (mode 4, DEFAULT : z)
    -pattern=[P]   : domain pattern                (mode 4, DEFAULT : auto-enumerate all)
                     4-element  ex) -pattern=Co,Fe/Ni,Cu
                     5-element  ex) -pattern=Cu:Co,Fe/Ni,Ti
    -order=[Q,..]  : target order parameters Q     (mode 3/4, DEFAULT : 1,0.75,0.5,0.25,0)
    -children=#    : structures per parent & Q     (mode 3/4, DEFAULT : inferred from -n)
    -limit=#       : exhaustive enumeration safety limit (mode 5, DEFAULT : 2000000)

    < ADVANCED Q / SRO (mode 3/4) >
    -order_tol=#   : |Q_target - Q| tolerance      (DEFAULT : 0.05)
    -order_steps=# : Q search iterations           (DEFAULT : 5000)
    -sro_cutoff=#  : first-shell cutoff factor     (DEFAULT : 1.20)
    -sro_tol=#     : SRO RMS tolerance             (DEFAULT : 0.12)
    -sro_weight=#  : SRO weight in Q search        (DEFAULT : 0.5)
    -bucket=#      : max trials per parent/Q bucket (DEFAULT : 30)

    < CCPY VASP INPUT GENERATION >   (same sub-options as CCpyVASPInputGen.py)
    -vasp          : after generation, write INCAR/KPOINTS/POTCAR into every
                     structure folder via CCpy VASPInput (yaml presets in
                     [PRESET_DIR]).
                     The first structure opens the INCAR confirm menu; the same
                     settings are then applied to all remaining structures.
    -preset=[NAME] : [NAME].yaml in [PRESET_DIR]  (DEFAULT : default.yaml)
    -sp            : single point calculation      (NSW = 0)
    -isif=#        : ISIF value
    -spin          : spin polarized calculation
    -mag           : use magnetic moment parameters (values from config file)
    -ldau          : use LDA+U parameters           (values from config file)
    -vdw=[M]       : D2 | D3 | D3damp | dDsC | optb88 | optb86b
    -kp=#,#,#      : Monkhorst-Pack grid           (DEFAULT : preset reciprocal density)
    -pot=[F]       : POTCAR functional             (DEFAULT : PBE_54)
    -pseudo=[..]   : ex) -pseudo=Nb_sv,Ti_sv
    -no_reuse      : build INCAR/KPOINTS/POTCAR separately for every structure.
                     By default only the first structure of each folder goes
                     through the full VASPInput path and the rest reuse its
                     INCAR/KPOINTS/POTCAR (they share the cell and composition,
                     so those three files are identical); use this to switch
                     that off.
    -batch         : ask nothing -- skip the settings sheet and the INCAR confirm
                     menu, and run immediately (-i= is required with it)

    < LEGACY VASP FILES (Gen-HEA style, only with -fmt=folder) >
    -template=[DIR]  : copy INCAR/KPOINTS/POTCAR from DIR into every structure folder
    -gen_potcar      : build POTCAR by concatenating a potpaw_PBE-style library
    -potcar_lib=[DIR]: potpaw library path          (DEFAULT : auto-detect)
    -potcar_var=[..] : variant overrides            ex) -potcar_var=Fe:Fe_sv,Co:Co_pv
--------------------------------------'''.replace("[PRESET_DIR]", _PRESET_DIR)
          )
    quit()


option = sys.argv[1]
mode_map = {"1": "random", "2": "spread", "3": "layered", "4": "domain", "5": "exhaustive",
            "w": "wizard", "wizard": "wizard"}
if option not in mode_map:
    print("Unknown option: %s  (use 1-5 or w; run without arguments for help)" % option)
    quit()

# ---------------------------------------------------------------------------
# Parse sub options (CCpy style: exact match for flags, -key=value for values)
# ---------------------------------------------------------------------------
given = {}                 # sheet keys the user actually supplied
input_file = None
output_dir = None
replace_element = None          # None -> derive from the chosen structure
composition_str = None          # None -> derive from the chosen structure
keep_composition = False
target = 500
seed = None
out_fmt = "folder"
overwrite = False
ASK = "__ASK__"        # sentinel: bare -surface / -redox -> interactive pick
surface = None
redox = None
redox_max_sets = 50
symprec = 1e-3
max_attempts = 2000000

layer_axis = "z"
view_axis = "z"
domain_pattern = None
order_levels = "1,0.75,0.5,0.25,0"
children_per_parent = None
exhaustive_limit = 2000000

order_tolerance = 0.05
order_search_steps = 5000
sro_cutoff_factor = 1.20
sro_tolerance = 0.12
sro_weight = 0.5
max_trials_per_bucket = 30

ccpy_vasp = False
incar_preset = None
single_point = False
isif = False
spin = False
mag = False
ldau = False
vdw = False
kpoints = False
functional = "PBE_54"
pseudo = None
batch = False
reuse_inputs = True

template_dir = None
generate_potcar = False
potcar_library = None
potcar_variants = None

for arg in sys.argv[2:]:
    # flags (exact match)
    if arg == "-keep_comp":
        keep_composition = True
        given["comp"] = "keep"
    elif arg == "-overwrite":
        overwrite = True
        given["overwrite"] = "y"
    elif arg == "-vasp":
        ccpy_vasp = True
        given["vasp"] = "y"
    elif arg == "-sp":
        single_point = True
        given["sp"] = "y"
    elif arg == "-spin":
        spin = True
        given["spin"] = "y"
    elif arg == "-mag":
        mag = True
        given["mag"] = "y"
    elif arg == "-ldau":
        ldau = True
        given["ldau"] = "y"
    elif arg == "-batch":
        batch = True
        given["batch"] = "y"
    elif arg == "-no_reuse":
        reuse_inputs = False
        given["reuse"] = "n"
    elif arg == "-gen_potcar":
        generate_potcar = True
        given["gen_potcar"] = "y"
    elif arg == "-surface":
        surface = ASK          # pick interactively (auto-detect + confirm)
        given["surface"] = "auto"
    elif arg == "-redox":
        redox = ASK            # pick interactively from the atom listing
        given["redox"] = "auto"
    # valued options (-key=value)
    elif arg.startswith("-i="):
        input_file = arg.split("=", 1)[1]
        given["input"] = arg.split("=", 1)[1]
    elif arg.startswith("-o="):
        output_dir = arg.split("=", 1)[1]
        given["output"] = arg.split("=", 1)[1]
    elif arg.startswith("-re="):
        replace_element = arg.split("=", 1)[1]
        given["replace"] = arg.split("=", 1)[1]
    elif arg.startswith("-comp="):
        composition_str = arg.split("=", 1)[1]
        given["comp"] = arg.split("=", 1)[1]
    elif arg.startswith("-n="):
        target = int(arg.split("=", 1)[1])
        given["n"] = arg.split("=", 1)[1]
    elif arg.startswith("-seed="):
        seed = int(arg.split("=", 1)[1])
        given["seed"] = arg.split("=", 1)[1]
    elif arg.startswith("-fmt="):
        out_fmt = arg.split("=", 1)[1].lower()
        given["fmt"] = arg.split("=", 1)[1]
    elif arg.startswith("-surface="):
        surface = arg.split("=", 1)[1]
        given["surface"] = arg.split("=", 1)[1]
    elif arg.startswith("-redox_max="):
        redox_max_sets = int(arg.split("=", 1)[1])
        given["redox_max"] = arg.split("=", 1)[1]
    elif arg.startswith("-redox="):
        # spec string: numbers / element symbols / optional '/k'
        redox = arg.split("=", 1)[1]
        given["redox"] = arg.split("=", 1)[1]
    elif arg.startswith("-symprec="):
        symprec = float(arg.split("=", 1)[1])
        given["symprec"] = arg.split("=", 1)[1]
    elif arg.startswith("-max_attempts="):
        max_attempts = int(arg.split("=", 1)[1])
        given["max_attempts"] = arg.split("=", 1)[1]
    elif arg.startswith("-axis="):
        layer_axis = arg.split("=", 1)[1]
        given["axis"] = arg.split("=", 1)[1]
    elif arg.startswith("-view="):
        view_axis = arg.split("=", 1)[1]
        given["view"] = arg.split("=", 1)[1]
    elif arg.startswith("-pattern="):
        domain_pattern = arg.split("=", 1)[1]
        given["pattern"] = arg.split("=", 1)[1]
    elif arg.startswith("-order="):
        order_levels = arg.split("=", 1)[1]
        given["order"] = arg.split("=", 1)[1]
    elif arg.startswith("-children="):
        children_per_parent = int(arg.split("=", 1)[1])
        given["children"] = arg.split("=", 1)[1]
    elif arg.startswith("-limit="):
        exhaustive_limit = int(arg.split("=", 1)[1])
        given["limit"] = arg.split("=", 1)[1]
    elif arg.startswith("-order_tol="):
        order_tolerance = float(arg.split("=", 1)[1])
        given["order_tol"] = arg.split("=", 1)[1]
    elif arg.startswith("-order_steps="):
        order_search_steps = int(arg.split("=", 1)[1])
        given["order_steps"] = arg.split("=", 1)[1]
    elif arg.startswith("-sro_cutoff="):
        sro_cutoff_factor = float(arg.split("=", 1)[1])
        given["sro_cutoff"] = arg.split("=", 1)[1]
    elif arg.startswith("-sro_tol="):
        sro_tolerance = float(arg.split("=", 1)[1])
        given["sro_tol"] = arg.split("=", 1)[1]
    elif arg.startswith("-sro_weight="):
        sro_weight = float(arg.split("=", 1)[1])
        given["sro_weight"] = arg.split("=", 1)[1]
    elif arg.startswith("-bucket="):
        max_trials_per_bucket = int(arg.split("=", 1)[1])
        given["bucket"] = arg.split("=", 1)[1]
    elif arg.startswith("-preset="):
        incar_preset = arg.split("=", 1)[1]
        given["preset"] = arg.split("=", 1)[1]
    elif arg.startswith("-isif="):
        isif = int(arg.split("=", 1)[1])
        given["isif"] = arg.split("=", 1)[1]
    elif arg.startswith("-vdw="):
        vdw = arg.split("=", 1)[1]
        given["vdw"] = arg.split("=", 1)[1]
    elif arg.startswith("-kp="):
        kpoints = arg.split("=", 1)[1].split(",")
        given["kp"] = arg.split("=", 1)[1]
    elif arg.startswith("-pot="):
        functional = arg.split("=", 1)[1]
        given["pot"] = arg.split("=", 1)[1]
    elif arg.startswith("-pseudo="):
        pseudo = arg.split("=", 1)[1].split(",")
        given["pseudo"] = arg.split("=", 1)[1]
    elif arg.startswith("-template="):
        template_dir = arg.split("=", 1)[1]
        given["template"] = arg.split("=", 1)[1]
    elif arg.startswith("-potcar_lib="):
        potcar_library = arg.split("=", 1)[1]
        given["potcar_lib"] = arg.split("=", 1)[1]
    elif arg.startswith("-potcar_var="):
        potcar_variants = arg.split("=", 1)[1]
        given["potcar_var"] = arg.split("=", 1)[1]
    else:
        print("Unknown sub_option: %s  (run without arguments for help)" % arg)
        quit()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mode = mode_map[option]

# The option number picks the generation mode and then hands over to the
# settings sheet, the same way CCpyVASPInputGen shows its INCAR menu before
# writing anything: sub-options given here just pre-fill the sheet. `-batch`
# means "ask nothing" and runs straight through instead.
if not batch:
    from CCpy.VASP.AlloyGen import run_wizard
    if mode != "wizard":
        given["mode"] = mode
    run_wizard(initial=given)
    quit()

from collections import Counter

from CCpy.VASP.AlloyGen import (
    generate_structures,
    guess_substrate_elements,
    generate_ccpy_vasp_inputs,
    parse_composition,
    parse_element_list,
    interactive_select_surface_elements,
    interactive_select_redox_atoms,
    select_structure_file_interactively,
    _read_structure_file,
    _suggest_output_dir,
)

if not input_file:
    # -batch means no questions, so there is nothing to pick from here.
    print("With -batch the input file must be given: -i=[FILE]")
    print("(drop -batch to get the settings sheet with a file chooser)")
    quit()
if not os.path.isfile(input_file):
    print("Input structure file not found: %s" % input_file)
    quit()

# -- defaults taken from the structure itself, so a Pt-free CONTCAR never
#    silently gets "-re=Pt" applied to it
if replace_element is None or composition_str is None:
    seed_parent = _read_structure_file(input_file)
    if replace_element is None:
        substrate, adsorbate, _is_slab = guess_substrate_elements(seed_parent)
        if not substrate:
            print("Could not work out the replacement pool from the structure; give -re=")
            quit()
        replace_element = ",".join(substrate)
        note = "  (adsorbate: %s)" % ",".join(adsorbate) if adsorbate else ""
        print("* Replacement pool (auto): %s%s" % (replace_element, note))
    if composition_str is None:
        pool = set(parse_element_list(replace_element))
        counts = Counter(str(sym) for sym in seed_parent.get_chemical_symbols() if str(sym) in pool)
        composition_str = ",".join("%s%d" % (el, counts[el]) for el in sorted(counts))
        print("* Target composition (auto, = current): %s" % composition_str)

composition = None if keep_composition else parse_composition(composition_str)

# -- bare -surface / -redox : pick interactively before touching anything
if surface == ASK or redox == ASK:
    preview_parent = _read_structure_file(input_file)
    preview_pool = parse_element_list(replace_element)
    if surface == ASK:
        surface = interactive_select_surface_elements(preview_parent, preview_pool)
    if redox == ASK:
        redox = interactive_select_redox_atoms(
            preview_parent, preview_pool, max_sets=redox_max_sets
        )

if not output_dir:
    if keep_composition:
        print("With -keep_comp, the output directory must be assigned: -o=[DIR]")
        quit()
    output_dir = _suggest_output_dir(input_file, mode, composition,
                                     layer_axis=layer_axis, view_axis=view_axis)
    print("* Output directory (auto): %s" % output_dir)

# -vasp works on the CIF files written by the generator, then converts each
# one into a full VASP input folder. Force CIF output in that case.
vasp_folder = False
output_format = out_fmt
if ccpy_vasp:
    if out_fmt != "cif":
        print("* -vasp : structures are first written as .cif then converted; -fmt=%s is ignored." % out_fmt)
    if template_dir or generate_potcar:
        print("* -vasp : -template/-gen_potcar (legacy, -fmt=folder only) are ignored.")
        template_dir = None
        generate_potcar = False
    output_format = "cif"
elif out_fmt in ("folder", "poscar_folder"):
    vasp_folder = True
    output_format = "vasp"
elif out_fmt in ("cif", "vasp", "poscar"):
    output_format = "vasp" if out_fmt == "poscar" else out_fmt
else:
    print("Unknown -fmt value: %s (use cif | vasp | folder)" % out_fmt)
    quit()

result = generate_structures(
    input_file=input_file,
    output_dir=output_dir,
    replace_element=replace_element,
    composition=composition,
    keep_composition=keep_composition,
    mode=mode,
    target=target if mode != "exhaustive" else 0,
    max_attempts=max_attempts if mode != "exhaustive" else 0,
    symprec=symprec,
    seed=seed,
    output_format=output_format,
    vasp_folder=vasp_folder,
    overwrite=overwrite,
    order_levels=order_levels if mode in ("layered", "domain") else None,
    order_tolerance=order_tolerance,
    order_search_steps=order_search_steps,
    sro_cutoff_factor=sro_cutoff_factor,
    sro_tolerance=sro_tolerance,
    sro_weight=sro_weight,
    max_trials_per_bucket=max_trials_per_bucket,
    template_dir=template_dir,
    exhaustive_limit=exhaustive_limit,
    layer_axis=layer_axis,
    view_axis=view_axis,
    domain_pattern=domain_pattern,
    children_per_parent=children_per_parent,
    generate_potcar=generate_potcar,
    potcar_library=potcar_library,
    potcar_variants=potcar_variants,
    adsorbate_elements=surface,
    redox_remove=redox,
    redox_max_sets=redox_max_sets,
)

if ccpy_vasp:
    prev_incar = generate_ccpy_vasp_inputs(
        output_dir,
        preset=incar_preset,
        kpoints=kpoints,
        functional=functional,
        pseudo=pseudo,
        single_point=single_point,
        isif=isif,
        vdw=vdw,
        spin=spin,
        mag=mag,
        ldau=ldau,
        batch=batch,
        reuse_inputs=reuse_inputs,
    )
    # generate_structures reports the dirs it created, so every _rN twin is
    # covered without guessing folder names
    twin_dirs = []
    if result and result.get("surface_dir"):
        twin_dirs.append(result["surface_dir"])
    if result:
        twin_dirs.extend(result.get("redox_dirs") or [])
    for twin_dir in twin_dirs:
        if os.path.isdir(twin_dir) and prev_incar:
            print("\n* Generating the same VASP inputs for the twin directory: %s" % twin_dir)
            generate_ccpy_vasp_inputs(
                twin_dir,
                preset=incar_preset,
                kpoints=kpoints,
                functional=functional,
                pseudo=pseudo,
                single_point=single_point,
                isif=isif,
                vdw=vdw,
                spin=spin,
                mag=mag,
                ldau=ldau,
                reuse_incar_from=prev_incar,
                reuse_inputs=reuse_inputs,
            )
