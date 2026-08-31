#!/usr/bin/env python
"""
CCpyAlloyAnal.py

CCpy-style front-end for analysing the folder sets that CCpyAlloyGen.py
produced (CCpy.VASP.AlloyAnal): where each adsorbate atom ended up sitting,
and how the energy of each structure compares with its adsorbate-free
(_surface) and redox (_r, _r1, _r2 ...) twins.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd

# Above this many rows the site table is left to the file instead of the screen.
SITE_PRINT_LIMIT = 40

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

2 : Energy differences against the twins, per structure:
        dE_surface = E(set) - E(set_surface)
        dE_r1      = E(set) - E(set_r1),  dE_r2 = ...
    ex) CCpyAlloyAnal.py 2

3 : Both of the above in one table.
    ex) CCpyAlloyAnal.py 3

4 : Every twin against the SURFACE twin, per structure:
        main - _surface = E(set)     - E(set_surface)
        _r1  - _surface = E(set_r1)  - E(set_surface),  _r2 - _surface = ...
    ex) CCpyAlloyAnal.py 4

    Option 2 differences each redox twin against the main folder, which
    answers "what did this redox step cost". Option 4 puts every twin on ONE
    reference, the clean surface, so the column of a redox twin is what is
    still adsorbed after that step -- the same quantity as dE_surface, one
    redox state further along. Read down a row and it is a ladder.
    Written to 04_[SET]_RedoxVsSurface.csv / .txt, so it does not overwrite
    the table of option 1-3.

    NOTE  dE_surface is only the difference between the two folders. The
    reference energy of the adsorbate itself is NOT subtracted, so it is not
    a complete adsorption energy.

    Energies are read the same way as CCpyVASPAnal.py option 2: 'free  energy
    TOTEN' of OUTCAR, and the last E0 of OSZICAR only when OUTCAR is missing.
    The column 'E source' says which file each value came from.

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
-nocheck       : skip the convergence check (it reads OSZICAR, vasp.out and
                 the INCAR of every folder). Without it the tables carry a
                 'Converged' column for the main folder and an 'unconverged'
                 column naming any twin of that row that failed -- a difference
                 is only as good as the worse of its two folders, and an
                 unconverged _surface poisons dE_surface and every
                 _rN - _surface built on it. A folder fails on either of two
                 counts: the CCpyVASPAnal option 0 verdict (custodian over
                 vasp.out + the max-ionic check from NSW), or an SCF loop that
                 hit NELM. The second is read from OSZICAR and is what option 0
                 cannot see at all -- most of all in a single-point folder
                 (NSW=0), where its max-ionic pattern can never fire and an
                 unsettled energy used to be reported as converged. A folder
                 with no vasp.out is 'Unknown' rather than a failure.
-no_twins      : assign sites only in the main folder, not in the redox twins
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

  The redox twins get their own fit, against `_rN - _surface` (option 4's
  column, or the same number derived from option 2's two columns), so a twin's
  rows carry contributions too -- read down one folder to see whether the
  preference survives the redox step.

[output files]
04_[SET]_AlloyAnal.csv / .txt        : one row per structure (option 1-3)
04_[SET]_AdsorptionSites.csv / .txt  : one row per adsorbate atom (option 1, 3)
04_[SET]_SiteEnsembles.csv / .txt     : how often each (site, element
                                       composition) is occupied
04_[SET]_SiteDiff.csv / .txt         : only the atoms the two methods disagree
                                       on, both answers side by side

The per-structure table of option 2 / 3 carries energies only. The sites are
in their own files above, so the same answer is not printed twice in two
shapes.
04_[SET]_RedoxVsSurface.csv / .txt   : one row per structure (option 4)
''')
    quit()

from CCpy.VASP.AlloyAnal import (select_alloy_sets, analyze_set, write_tables,
                                 DEFAULT_DIST_TOL, DEFAULT_LAYER_TOL,
                                 DEFAULT_HCP_TOL, DEFAULT_MAIN_METHOD)

option = sys.argv[1]
if option not in ("1", "2", "3", "4"):
    print("Unknown option: %s   (use 1, 2, 3, 4, or -h)" % option)
    quit()

do_sites = option in ("1", "3")
do_energy = option in ("2", "3")
do_redox_surface = option == "4"

chosen, ads, pool = None, None, None
prefer_poscar = "-poscar" in sys.argv
do_twin_sites = "-no_twins" not in sys.argv
check_errors = "-nocheck" not in sys.argv
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

for set_dir in sets:
    table, site_table, diff_table, info = analyze_set(
        set_dir, do_sites=do_sites, do_energy=do_energy,
        prefer_poscar=prefer_poscar, ads_override=ads, pool_override=pool,
        dist_tol=dist_tol, layer_tol=layer_tol, hcp_tol=hcp_tol, main=main,
        do_redox_surface=do_redox_surface, do_twin_sites=do_twin_sites,
        check_errors=check_errors)

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
    if (do_energy or do_redox_surface) and info["surface_dir"]:
        print("* the difference is between the two folders only; the adsorbate "
              "reference energy is not included.")
    if do_redox_surface and not info["surface_dir"]:
        print("* no _surface twin next to this set, so there is nothing to "
              "measure the twins against.")

    written = write_tables(table, site_table, info["set_name"], out_dir="./",
                           do_sites=do_sites, do_energy=do_energy,
                           kind="RedoxVsSurface" if do_redox_surface else "AlloyAnal",
                           diff_table=diff_table,
                           ensemble_table=info.get("ensemble_table"))
    if written:
        print("Analysis files have been saved: " + ", ".join(os.path.basename(f) for f in written))
