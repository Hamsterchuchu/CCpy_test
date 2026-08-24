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

2 : Energy differences against the twins, per structure:
        dE_surface = E(set) - E(set_surface)
        dE_r1      = E(set) - E(set_r1),  dE_r2 = ...
    ex) CCpyAlloyAnal.py 2

3 : Both of the above in one table.
    ex) CCpyAlloyAnal.py 3

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
-poscar        : read POSCAR instead of CONTCAR (DEFAULT : CONTCAR, i.e. the
                 relaxed result; POSCAR is used anyway when CONTCAR is missing)
-tol=#         : nearest-neighbour window for the distance method, as a
                 fraction of the shortest distance   (DEFAULT : 0.15)
-layer_tol=#   : thickness of one substrate layer in A (DEFAULT : 1.0)
-hcp_tol=#     : in-plane distance in A within which a second-layer atom counts
                 as sitting under a hollow -> hcp instead of fcc (DEFAULT : 0.8)

Two independent methods are always reported side by side:
  site_dist : nearest-neighbour counting (how many substrate atoms share the
              shortest distance, within -tol)
  site_proj : geometric projection onto the surface plane, located in the
              Delaunay triangulation of the top layer (vertex / edge / inside)
The 'agree' column flags where they differ -- that is where the position is
genuinely ambiguous and worth looking at by hand.

[output files]
04_[SET]_AlloyAnal.csv / .txt        : one row per structure
04_[SET]_AdsorptionSites.csv / .txt  : one row per adsorbate atom
''')
    quit()

from CCpy.VASP.AlloyAnal import (select_alloy_sets, analyze_set, write_tables,
                                 DEFAULT_DIST_TOL, DEFAULT_LAYER_TOL, DEFAULT_HCP_TOL)

option = sys.argv[1]
if option not in ("1", "2", "3"):
    print("Unknown option: %s   (use 1, 2, 3, or -h)" % option)
    quit()

do_sites = option in ("1", "3")
do_energy = option in ("2", "3")

chosen, ads, pool = None, None, None
prefer_poscar = "-poscar" in sys.argv
take_all = "-all" in sys.argv
dist_tol, layer_tol, hcp_tol = DEFAULT_DIST_TOL, DEFAULT_LAYER_TOL, DEFAULT_HCP_TOL

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

sets = select_alloy_sets("./", ask=not take_all, chosen=chosen)
if not sets:
    quit()

pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

for set_dir in sets:
    table, site_table, info = analyze_set(
        set_dir, do_sites=do_sites, do_energy=do_energy,
        prefer_poscar=prefer_poscar, ads_override=ads, pool_override=pool,
        dist_tol=dist_tol, layer_tol=layer_tol, hcp_tol=hcp_tol)

    if table is None or not len(table):
        print("  nothing to report in this folder.")
        continue

    print("")
    print(table.to_string())
    if do_sites and site_table is not None and len(site_table):
        print("")
        print(site_table.to_string())

    for warning in info["warnings"]:
        print("* %s" % warning)
    for twin, ids in info["missing_twin_ids"].items():
        print("* %s has no folder for %d structure(s): %s"
              % (twin, len(ids), ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")))
    if do_energy and info["surface_dir"]:
        print("* dE_surface is the difference between the two folders only; the "
              "adsorbate reference energy is not included.")

    written = write_tables(table, site_table, info["set_name"], out_dir="./",
                           do_sites=do_sites, do_energy=do_energy)
    if written:
        print("Analysis files have been saved: " + ", ".join(os.path.basename(f) for f in written))
