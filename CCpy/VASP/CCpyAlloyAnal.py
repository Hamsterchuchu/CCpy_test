#!/usr/bin/env python
"""
CCpyAlloyAnal.py

CCpy-style front-end for analysing the folder sets that CCpyAlloyGen.py
produced (CCpy.VASP.AlloyAnal): whether every folder finished and converged,
where each adsorbate atom ended up sitting, the adsorption energy of every
folder against the adsorbate-free (_surface) twin, and the redox reaction
energy of every redox twin (_r, _r1, _r2 ...) against the main folder.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd

# Above this many rows a table is left to the file instead of the screen.
SITE_PRINT_LIMIT = 40
STATUS_PRINT_LIMIT = 60

version = sys.version
if version[0] == '3':
    raw_input = input


if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help", "help", "-help"):
    print("\nHow to use : " + sys.argv[0].split("/")[-1] + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
Analyse the output folders of CCpyAlloyGen.py after the VASP jobs are done.
Run it in the directory that HOLDS the output folder(s); the folders are
listed with numbers to pick from, the way CCpyVASPAnal.py does it. The
_surface / _r* twins are not listed as targets -- they are paired with their
main set automatically, by structure ID (S000001 of one folder against
S000001 of the twin).

[options]
0 : Convergence check. One row per FOLDER -- main, _surface and every redox
    twin of every structure -- because a folder is the level at which a VASP
    run fails. Options 2-4 carry the same verdict as a flag on the structure's
    row, which names the twin that failed but not what went wrong in it.
    ex) CCpyAlloyAnal.py 0

    Two checks, either one alone enough to fail a folder:
      custodian    : VaspErrorHandler over vasp.out plus the max-ionic pattern
                     built from NSW -- the CCpyVASPAnal.py option 0 verdict.
                     'Unknown' when the folder keeps no vasp.out; that is not
                     a failure.
      SCF (NELM)   : did the last closed ionic block of OSZICAR stay under
                     NELM. Custodian's error list has no entry for an SCF that
                     simply ran out of steps, and in a single-point folder
                     (-sp, so NSW=0) its max-ionic pattern can never fire
                     either -- '0 F=' is a line VASP does not print. So this
                     column is the only thing standing between an SCF that
                     stopped at NELM and the energy tables of options 2-4.
      NELM         : the limit that check compared against, taken from OUTCAR's
                     echo before the INCAR, so a folder resubmitted with a
                     raised NELM is judged against the raised one.
    'Job end' is vasp.done: without it the job has not run yet, or what lies
    in the folder is the output of an earlier attempt that was resubmitted.
    Written to 04_[SET]_Convergence.csv / .txt.

1 : Adsorption sites. For every adsorbate atom, which substrate atom it sits
    on top of, which two it bridges, or which three/four form the hollow
    (a 3-fold hollow is also reported as fcc or hcp).
    ex) CCpyAlloyAnal.py 1

    On a substituted surface the geometry is only half the answer: a hollow of
    three Ni is not the site a hollow of two Ni and one Fe is. So every atom
    also gets an 'ensemble' column -- the element composition of the site it
    occupies, written canonically (Fe1Ni2) so rows can be grouped by it -- and
    the counts are rolled up per (site, ensemble) into
    04_[SET]_SiteEnsembles.csv, which is the table to read to see which
    element combination the adsorbate actually prefers.

    The redox twins are assigned too, not just the main folder: whatever is
    left after a redox step usually relaxes into a different site, and that
    move is the comparison worth having. The site table gets one row per
    adsorbate atom PER FOLDER, marked in the 'folder' column (main, r1, r2 ..).
    Atom numbers restart in every folder, so 'atom' carries the number the
    same atom has in the MAIN folder and 'local_no' the number it has in its
    own -- read down the 'atom' column to follow one atom across the redox
    steps. The map is read off the unrelaxed POSCARs, which are identical apart
    from the removed atoms, so it is exact; without both POSCARs the column is
    left blank rather than guessed. Turn it all off with -no_twins.

2 : Adsorption energy, per structure. Every folder that still carries
    adsorbate is measured against the SAME reference, the clean surface:
        dE_surface     = E(set)    - E(set_surface)
        dE_surface_r1  = E(set_r1) - E(set_surface),  dE_surface_r2 = ...
    ex) CCpyAlloyAnal.py 2

    One shared reference is what makes the row a ladder: read across it and
    each column is what is still adsorbed after that redox step.

    NOTE  dE_surface is only the difference between the two folders. The
    reference energy of the adsorbate itself is NOT subtracted, so it is not
    a complete adsorption energy.

3 : Redox reaction energy, per structure. Each redox twin against the MAIN
    folder -- what that redox step itself cost:
        dE_r1 = E(set) - E(set_r1),  dE_r2 = E(set) - E(set_r2), ...
    ex) CCpyAlloyAnal.py 3

    Option 2 and option 3 answer different questions about the same folders:
    2 puts every folder on the clean surface, 3 differences each redox twin
    against the state it came from.

4 : Options 1, 2 and 3 in one run, one table. Option 3's columns are skipped
    for a set that has no redox twin next to it.
    ex) CCpyAlloyAnal.py 4

    Energies are read the same way as CCpyVASPAnal.py option 2: 'free  energy
    TOTEN' of OUTCAR and nothing else. OSZICAR's E0 is a different reference
    and is not used, so a folder with no readable OUTCAR is left blank.
    The column 'E source' says where each value came from.

[sub_options]
-i=[DIR]       : analyse this set directly (comma-separated for several),
                 skipping the folder question
                 ex) -i=Pt36_HEA        ex) -i=set_a,set_b
-all           : take every set found here without asking
-ads=[EL,..]   : adsorbate elements (DEFAULT : taken from the _surface twin,
                 i.e. whatever the main structure has and the twin has not;
                 then metadata.txt; then AlloyGen's geometric detection)
                 ex) -ads=Li,S
-pool=[EL,..]  : substrate elements -- restricts BOTH the geometric adsorbate
                 guess and which atoms count as the surface when the site is
                 assigned (DEFAULT : everything that is not the adsorbate,
                 so a substituted HEA surface keeps all of its elements)
                 ex) -pool=Pt,Fe,Co,Ni,Cu
-nocheck       : skip the convergence check inside options 2-4 (it reads
                 OSZICAR, vasp.out and the INCAR of every folder; option 0 is
                 that check on its own and ignores this flag). Without it the
                 tables carry a 'Converged' column for the main folder and an
                 'unconverged' column naming any twin of that row that failed
                 -- a difference is only as good as the worse of its two
                 folders, and an unconverged _surface poisons dE_surface and
                 every dE_surface_rN built on it. The two checks a folder has
                 to pass are the ones option 0 tabulates. A folder with no
                 vasp.out is 'Unknown' rather than a failure.
                 Two more columns come with it: 'Finished' says whether the
                 main folder holds vasp.done, and 'unfinished' names any folder
                 of that row without it. A folder without vasp.done has either
                 not run yet or is holding the output of an earlier attempt
                 that was resubmitted (CCpy deletes vasp.done on resubmission
                 but leaves CONTCAR/OUTCAR/OSZICAR in place), so its energy is
                 kept in the table but left out of the ensemble fit.
-no_twins      : assign sites only in the main folder, not in the redox twins
-noprogress    : do not draw the "[  n /  N  ]" folder counter (for a run whose
                 output is piped into a file)
-poscar        : read POSCAR instead of CONTCAR (DEFAULT : CONTCAR, i.e. the
                 relaxed result). Without it, a folder with no CONTCAR is one
                 VASP has not run yet: it is left out of the tables and counted
                 in a note, rather than filling a row with blanks. With
                 -poscar the unrelaxed input is what was asked for, so nothing
                 is skipped.
-tol=#         : nearest-neighbour window for the distance method, as a
                 fraction of the shortest distance   (DEFAULT : 0.15)
-layer_tol=#   : thickness of one substrate layer in A (DEFAULT : 1.2)
                 Raise it for a relaxed, buckled surface: when atoms of one
                 physical layer fall on both sides of this window the top
                 layer comes out incomplete, which is the one thing that
                 really breaks the projection method.
-hcp_tol=#     : in-plane distance in A within which a second-layer atom counts
                 as sitting under a hollow -> hcp instead of fcc (DEFAULT : 0.8)
-main=[M]      : which method fills the single 'site' column and the per
                 structure summary: proj | dist   (DEFAULT : proj)

Two independent methods are used, but only ONE answer is tabulated:
  site      : the -main answer (projection by default), one column
  agree     : same / DIFF / unresolved -- did the other method concur
The other method's answer is not carried through the site file. It is written
only for the atoms where the two disagree, to 04_[SET]_SiteDiff.csv, with both
answers, the barycentric weights and d_min side by side -- those few rows are
the ones a human has to settle, and burying them among identical columns for
every other atom is what makes them invisible.

  Projection leads because its barycentric test is scale-free, so a surface
  whose elements have different atomic radii does not shift the answer the way
  a distance window can. It has one blind spot by construction: height is not
  part of it, so an adsorbate that drifted off the surface still gets a
  confident label. 'd_min (A)' and 'height (A)' are the columns that catch it.

  Columns are one per fact: 'atom' is the number the atom carries in the MAIN
  folder (the same row means the same atom in every folder), 'local_no' is its
  number inside its own folder's CONTCAR. 'side' appears only when something
  is actually adsorbed on the underside of the slab.

  sub_site / sub_neighbors : the same classification against the SECOND
  substrate layer. The adsorbate does not bond to that layer, so this is a
  descriptor of what the site sits on, not a site of its own. On an ideal
  fcc(111) slab it just repeats fcc/hcp (hcp <=> sub_site 'top'); it earns its
  place by naming the SUBSURFACE elements under the site, which is what makes
  two geometrically identical sites of a substituted surface differ in energy.

  The 'dE vs group avg' column of the roll-up needs BOTH the sites and the
  adsorption energies, so it appears under option 4 and not under option 1 on
  its own. The redox twins get their own fit, against dE_surface_rN, so a
  twin's rows carry contributions too -- read down one folder to see whether
  the preference survives the redox step. The redox reaction energy of option 3
  is not split this way: it is a difference between two adsorbate-carrying
  folders, so it is not a sum over either one's sites.

[output files]
04_[SET]_Convergence.csv / .txt      : one row per folder (option 0)
04_[SET]_AlloyAnal.csv / .txt        : one row per structure (option 2, 3, 4)
04_[SET]_AdsorptionSites.csv / .txt  : one row per adsorbate atom (option 1, 4)
04_[SET]_SiteEnsembles.csv / .txt    : how often each (site, element
                                       composition) is occupied
04_[SET]_SiteDiff.csv / .txt         : only the atoms the two methods disagree
                                       on, both answers side by side

The per-structure table carries energies only. The sites are in their own
files above, so the same answer is not printed twice in two shapes.
''')
    quit()

from CCpy.VASP.AlloyAnal import (select_alloy_sets, analyze_set,
                                 analyze_convergence, find_twins, write_tables,
                                 DEFAULT_DIST_TOL, DEFAULT_LAYER_TOL,
                                 DEFAULT_HCP_TOL, DEFAULT_MAIN_METHOD)

option = sys.argv[1]
if option not in ("0", "1", "2", "3", "4"):
    print("Unknown option: %s   (use 0, 1, 2, 3, 4, or -h)" % option)
    quit()

do_convergence = option == "0"
do_sites = option in ("1", "4")
do_ads_energy = option in ("2", "4")
do_redox_energy = option in ("3", "4")

chosen, ads, pool = None, None, None
prefer_poscar = "-poscar" in sys.argv
do_twin_sites = "-no_twins" not in sys.argv
check_errors = "-nocheck" not in sys.argv
show_progress = "-noprogress" not in sys.argv
take_all = "-all" in sys.argv
dist_tol, layer_tol, hcp_tol = DEFAULT_DIST_TOL, DEFAULT_LAYER_TOL, DEFAULT_HCP_TOL
main = DEFAULT_MAIN_METHOD

for arg in sys.argv[2:]:
    if arg.startswith("-i="):
        chosen = [p for p in arg.split("=", 1)[1].split(",") if p]
    elif arg.startswith("-ads="):
        ads = [e for e in arg.split("=", 1)[1].split(",") if e]
    elif arg.startswith("-pool="):
        pool = [e for e in arg.split("=", 1)[1].split(",") if e]
    elif arg.startswith("-tol="):
        dist_tol = float(arg.split("=", 1)[1])
    elif arg.startswith("-layer_tol="):
        layer_tol = float(arg.split("=", 1)[1])
    elif arg.startswith("-hcp_tol="):
        hcp_tol = float(arg.split("=", 1)[1])
    elif arg.startswith("-main="):
        main = arg.split("=", 1)[1].strip().lower()
        if main not in ("proj", "dist"):
            print("-main= takes 'proj' or 'dist', not '%s'" % main)
            quit()

sets = select_alloy_sets("./", ask=not take_all, chosen=chosen)
if not sets:
    quit()

pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

if do_convergence:
    for set_dir in sets:
        table, info = analyze_convergence(set_dir, show_progress=show_progress)
        if table is None or not len(table):
            print("  no folder to check in this set.")
            continue
        print("")
        if len(table) <= STATUS_PRINT_LIMIT:
            print(table.to_string())
        else:
            print("%d folder(s) checked -- see 04_%s_Convergence.txt"
                  % (len(table), info["set_name"]))
        print("")
        print("* converged %d   failed %d   unknown %d   (of %d folder(s))"
              % (info["n_converged"], info["n_failed"], info["n_unknown"],
                 info["n_folders"]))
        if info["failed"]:
            print("* these folders did not converge:")
            for path, why in info["failed"][:20]:
                print("    %-50s %s" % (path, why))
            if len(info["failed"]) > 20:
                print("    ... and %d more" % (len(info["failed"]) - 20))
        if info["unfinished"]:
            print("* %d folder(s) have no vasp.done -- not run yet, or holding "
                  "the output of an earlier attempt that was resubmitted: %s"
                  % (len(info["unfinished"]), ", ".join(info["unfinished"][:5])
                     + (" ..." if len(info["unfinished"]) > 5 else "")))
        for warning in info["warnings"]:
            print("* %s" % warning)
        written = write_tables(table, None, info["set_name"], out_dir="./",
                               do_sites=False, kind="Convergence")
        if written:
            print("Analysis files have been saved: "
                  + ", ".join(os.path.basename(f) for f in written))
    quit()

for set_dir in sets:
    # Option 4 asks for the redox columns, but a set with no redox twin has
    # none to fill: asking for them would only add a column of blanks.
    set_redox = do_redox_energy
    if set_redox:
        _surface_dir, _redox_dirs = find_twins(set_dir)
        if not _redox_dirs:
            set_redox = False
            if option == "4":
                print("\n* %s has no redox twin, so the redox reaction "
                      "energies are skipped for it." % set_dir)
            else:
                print("\n* %s has no redox twin, so there is nothing to "
                      "measure a redox reaction energy against." % set_dir)

    table, site_table, diff_table, info = analyze_set(
        set_dir, do_sites=do_sites, do_ads_energy=do_ads_energy,
        do_redox_energy=set_redox,
        prefer_poscar=prefer_poscar, ads_override=ads, pool_override=pool,
        dist_tol=dist_tol, layer_tol=layer_tol, hcp_tol=hcp_tol, main=main,
        do_twin_sites=do_twin_sites, check_errors=check_errors,
        show_progress=show_progress)

    if table is None or not len(table):
        print("  nothing to report in this folder.")
        continue

    if len(table.columns) > 1:
        print("")
        print(table.to_string())
    if do_sites and site_table is not None and len(site_table):
        # A long site table scrolls the useful part off the screen; the file
        # holds it either way, so past a screenful only the summary is printed.
        print("")
        if len(site_table) <= SITE_PRINT_LIMIT:
            print(site_table.to_string())
        else:
            print("%d adsorbate atom row(s) -- see 04_%s_AdsorptionSites.txt"
                  % (len(site_table), info["set_name"]))
        ensembles = info.get("ensemble_table")
        if ensembles is not None and len(ensembles):
            print("")
            print(ensembles.to_string())
            fits = ensembles.attrs.get("fit") or []
            if fits:
                print("* 'dE vs group avg' is a least-squares split of one energy "
                      "column over the adsorbate sites of each structure, done "
                      "per folder:")
                for fit in fits:
                    print("    %-6s %-26s %d structures, %d terms, R2=%s, "
                          "residual RMS %s eV vs a spread of %s eV%s"
                          % (fit["folder"], fit["column"], fit["n_structures"],
                             fit["n_terms"], fit["r2"], fit["rms"], fit["spread"],
                             "   (%d left out)" % fit["dropped"] if fit["dropped"] else ""))
                print("  Compare values only WITHIN one folder + element + site "
                      "group: each is relative to the average ensemble of its own "
                      "group, because the absolute level of a group is not "
                      "identifiable when every structure holds the same number of "
                      "each site type.")
        disagreed = 0 if diff_table is None else len(diff_table)
        print("")
        print("* site methods: %d of %d atom(s) disagree"
              % (disagreed, info["n_atoms_checked"]))
        if disagreed:
            print("  (these are the positions that are genuinely between two "
                  "sites -- worth a look by hand)")
            print(diff_table.to_string())

    for warning in info["warnings"]:
        print("* %s" % warning)
    for twin, ids in info["missing_twin_ids"].items():
        print("* %s has no folder for %d structure(s): %s"
              % (twin, len(ids), ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")))
    if do_ads_energy and info["surface_dir"]:
        print("* the difference is between the two folders only; the adsorbate "
              "reference energy is not included.")
    if do_ads_energy and not info["surface_dir"]:
        print("* no _surface twin next to this set, so there is nothing to "
              "measure the adsorption energies against.")

    written = write_tables(table, site_table, info["set_name"], out_dir="./",
                           do_sites=do_sites, kind="AlloyAnal",
                           diff_table=diff_table,
                           ensemble_table=info.get("ensemble_table"))
    if written:
        print("Analysis files have been saved: " + ", ".join(os.path.basename(f) for f in written))
