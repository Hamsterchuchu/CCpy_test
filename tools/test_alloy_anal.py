#!/usr/bin/env python
"""Self-check for CCpyAlloyAnal.py (AlloyGen result analysis).

Run from the repo root:

    python3 tools/test_alloy_anal.py

Everything happens in a temporary directory, so nothing outside it is touched.
The point of this test is that the adsorption sites have a known answer: ase
builds fcc(111) / fcc(100) slabs and places the adsorbate on the site it is
asked for, so the label the command prints can be compared with the site that
was actually built. The energies are fake OUTCAR / OSZICAR values, so the
differences are known exactly too.

Checks
  1. fcc(111): ontop / bridge / fcc / hcp are labelled correctly, by BOTH
     the distance method and the projection method
  2. fcc(100): the four-fold hollow is not mistaken for a bridge
     (the Delaunay diagonal trap), and a real bridge / top still work
  3. the adsorbate is taken from the _surface twin without any heuristic
  4. dE_surface and dE_r1 / dE_r2 match the fake energies exactly
  5. twin folders are not offered as analysis targets of their own
  6. an adsorbate on the bottom face is reported as such
  7. a folder with no OUTCAR / OSZICAR yields a blank energy, not a crash
  8. the csv / txt files are written
  9. the 'site' column follows -main (projection by default, distance when
     asked), and falls back only when the primary method cannot resolve
 10. sub_site -- the same classification against the second layer -- matches
     the stacking that was built: an hcp hollow sits on a second-layer ATOM
     ('top'), an fcc hollow on a second-layer hollow, and the four-fold hollow
     of fcc(100) sits on a second-layer atom
 11. option 4 puts every twin on the _surface reference
     (main - _surface, _r1 - _surface, _r2 - _surface) and writes its own file
 12. a surface buckled by more than -layer_tol is REPORTED (that split is the
     one real failure mode of the projection method), and raising -layer_tol
     puts the layer back together
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "CCpy", "VASP", "CCpyAlloyAnal.py")

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((bool(cond), name, extra))
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  <- " + str(extra)) if (extra and not cond) else ""))


def skip(name, why):
    RESULTS.append((None, name, why))
    print("  [SKIP] %s  (%s)" % (name, why))


try:
    import numpy as np
    from ase.build import fcc100, fcc111, add_adsorbate
    from ase.io import write as ase_write
except Exception as error:                                  # pragma: no cover
    skip("ase available", str(error))
    print("\nPASS 0 / FAIL 0 / SKIP 1")
    sys.exit(0)


def fake_energy(directory, energy, use_outcar=True):
    if use_outcar:
        with open(os.path.join(directory, "OUTCAR"), "w") as handle:
            handle.write("  free  energy   TOTEN  =      %.8f eV\n" % (energy - 1.0))
            handle.write("  free  energy   TOTEN  =      %.8f eV\n" % energy)
    else:
        with open(os.path.join(directory, "OSZICAR"), "w") as handle:
            handle.write("   1 F= -1.0E+02 E0= %.5E  d E =0.0\n" % energy)


def put(directory, atoms, energy=None, use_outcar=True):
    os.makedirs(directory, exist_ok=True)
    ase_write(os.path.join(directory, "CONTCAR"), atoms, format="vasp")
    ase_write(os.path.join(directory, "POSCAR"), atoms, format="vasp")
    if energy is not None:
        fake_energy(directory, energy, use_outcar=use_outcar)


def read_csv(path):
    rows = []
    import csv as _csv
    with open(path) as handle:
        for row in _csv.DictReader(handle):
            rows.append(row)
    return rows


def build(root):
    """Build the test sets and return the list of (set, structure, site) truths."""
    truth = []

    # -- set A: fcc(111), four known sites, plus a _surface and two redox twins
    slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    symbols = slab.get_chemical_symbols()
    for i in (0, 5, 11):
        symbols[i] = "Fe"
    slab.set_chemical_symbols(symbols)
    sites = ["ontop", "bridge", "fcc", "hcp"]
    expected = {"ontop": "top", "bridge": "bridge", "fcc": "hollow3-fcc", "hcp": "hollow3-hcp"}
    # What the site sits ON, one layer down. fcc(111) is ABC stacked, so an hcp
    # hollow has a second-layer atom right under it and an fcc hollow does not;
    # an atom of layer 1 itself sits over a layer-2 hollow.
    expected_sub = {"ontop": "hollow3", "bridge": "", "fcc": "hollow3", "hcp": "top"}
    for number, site in enumerate(sites, start=1):
        sid = "S%06d" % number
        main = slab.copy()
        add_adsorbate(main, "Li", 2.0, site)
        put(os.path.join(root, "A_set", sid), main, -300.0 - number)
        put(os.path.join(root, "A_set_surface", sid), slab, -280.0)
        put(os.path.join(root, "A_set_r1", sid), slab, -290.0 - number)
        put(os.path.join(root, "A_set_r2", sid), slab, -295.0 - number, use_outcar=False)
        truth.append(("A_set", sid, expected[site], expected_sub[site]))
    with open(os.path.join(root, "A_set_surface", "metadata.txt"), "w") as handle:
        handle.write("%-22s = %s\n" % ("replace_elements", "['Pt']"))
        handle.write("%-22s = %s\n" % ("adsorbate_elements_removed", "['Li']"))

    # -- set B: fcc(100) hollow / bridge / top, and one folder with no energy
    square = fcc100("Pt", size=(3, 3, 4), vacuum=10.0)
    # fcc(100): layer 2 sits under the four-fold hollows of layer 1, so the
    # hollow site sits on an atom and an ontop site sits on a hollow.
    for number, (site, label, sub) in enumerate([("hollow", "hollow4", "top"),
                                                 ("bridge", "bridge", ""),
                                                 ("ontop", "top", "hollow4")], start=1):
        sid = "S%06d" % number
        main = square.copy()
        add_adsorbate(main, "O", 1.5, site)
        put(os.path.join(root, "B_set", sid), main, None if number == 3 else -100.0 - number)
        put(os.path.join(root, "B_set_surface", sid), square, -90.0)
        truth.append(("B_set", sid, label, sub))

    # -- set C: adsorbate under the slab (bottom face)
    base = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    positions = base.get_positions()
    zmin = positions[:, 2].min()
    bottom = [i for i in range(len(base)) if abs(positions[i, 2] - zmin) < 0.1]
    triangle = sorted(bottom, key=lambda i: (positions[i, 0], positions[i, 1]))[:3]
    centre = positions[triangle].mean(axis=0)
    under = base.copy()
    under.append("Li")
    under.positions[-1] = [centre[0], centre[1], zmin - 2.0]
    put(os.path.join(root, "C_set", "S000001"), under, -100.0)
    put(os.path.join(root, "C_set_surface", "S000001"), base, -90.0)
    truth.append(("C_set", "S000001", "hollow3-hcp", "top"))

    # -- set D: top layer buckled by more than the default -layer_tol (1.2 A).
    #    Relaxation does this, and it is what tears the top layer in half: the
    #    projection method would then triangulate only the atoms that stayed
    #    up. The command has to say so instead of answering quietly.
    rough = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    coords = rough.get_positions()
    top_z = coords[:, 2].max()
    for i in range(len(rough)):
        if abs(coords[i, 2] - top_z) < 0.1 and i % 2 == 0:
            rough.positions[i, 2] -= 1.6
    add_adsorbate(rough, "Li", 2.0, "fcc")
    put(os.path.join(root, "D_set", "S000001"), rough, -100.0)
    put(os.path.join(root, "D_set_surface", "S000001"),
        rough[[i for i in range(len(rough)) if rough[i].symbol != "Li"]], -90.0)
    return truth


def run(work, extra=(), option="3"):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = REPO + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, BIN, option, "-all"] + list(extra),
                          cwd=work, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)


work = tempfile.mkdtemp(prefix="ccpy_alloyanal_")
try:
    truth = build(work)
    process = run(work)
    output = process.stdout
    check("command runs without a traceback", "Traceback" not in output and process.returncode == 0,
          output[-800:])

    sites = {}
    for name in ("A_set", "B_set", "C_set", "D_set"):
        path = os.path.join(work, "04_%s_AdsorptionSites.csv" % name)
        if os.path.exists(path):
            for row in read_csv(path):
                sites[(name, row["Structure"], row["element"])] = row

    check("site files written",
          all(os.path.exists(os.path.join(work, "04_%s_AdsorptionSites.csv" % n))
              for n in ("A_set", "B_set", "C_set")))
    check("energy files written",
          all(os.path.exists(os.path.join(work, "04_%s_AlloyAnal.csv" % n))
              for n in ("A_set", "B_set", "C_set")))

    for set_name, sid, expected_label, expected_sub in truth:
        element = "Li" if set_name in ("A_set", "C_set") else "O"
        row = sites.get((set_name, sid, element))
        if row is None:
            check("%s %s -> %s" % (set_name, sid, expected_label), False, "no row for this atom")
            continue
        check("%s %s distance method -> %s" % (set_name, sid, expected_label),
              row["site_dist"] == expected_label, row["site_dist"])
        # The projection method does not tell fcc from hcp any better than the
        # distance one does, so only the site type is compared there.
        want = expected_label.split("-")[0]
        check("%s %s projection method -> %s" % (set_name, sid, want),
              row["site_proj"].split("-")[0] == want, row["site_proj"])
        check("%s %s 'site' column follows projection by default" % (set_name, sid),
              row["site"] == row["site_proj"], "%s vs %s" % (row["site"], row["site_proj"]))
        if expected_sub:
            check("%s %s sub_site -> %s" % (set_name, sid, expected_sub),
                  row["sub_site"].split("-")[0] == expected_sub, row["sub_site"])
            check("%s %s sub_neighbors not empty" % (set_name, sid),
                  bool(row["sub_neighbors"]), repr(row["sub_neighbors"]))

    energies = {name: read_csv(os.path.join(work, "04_%s_AlloyAnal.csv" % name))
                for name in ("A_set", "B_set", "C_set")}

    first = energies["A_set"][0]
    check("dE_surface exact", abs(float(first["dE_surface (eV)"]) - (-301.0 + 280.0)) < 1e-6,
          first.get("dE_surface (eV)"))
    check("dE_r1 exact", abs(float(first["dE_r1 (eV)"]) - (-301.0 + 291.0)) < 1e-6,
          first.get("dE_r1 (eV)"))
    check("dE_r2 exact (OSZICAR twin)",
          abs(float(first["dE_r2 (eV)"]) - (-301.0 + 296.0)) < 1e-6, first.get("dE_r2 (eV)"))
    check("energy source reported as OUTCAR", first["E source"] == "OUTCAR", first["E source"])

    check("adsorbate came from the _surface twin",
          "_surface twin composition" in output)
    check("twins are not analysed as targets",
          output.count("# ---------- ") == 4 and "_surface ---" not in output,
          output.count("# ---------- "))
    check("bottom-face adsorbate reported as bottom",
          sites[("C_set", "S000001", "Li")]["side"] == "bottom",
          sites[("C_set", "S000001", "Li")]["side"])

    blank = [row for row in energies["B_set"] if row["Structure"] == "S000003"][0]
    check("folder without OUTCAR/OSZICAR leaves a blank energy",
          blank["Energy (eV)"] == "", repr(blank["Energy (eV)"]))

    # -main=dist has to move the 'site' column over to the distance answer.
    process = run(work, ["-main=dist"])
    check("-main=dist runs without a traceback",
          "Traceback" not in process.stdout and process.returncode == 0,
          process.stdout[-800:])
    rows = read_csv(os.path.join(work, "04_A_set_AdsorptionSites.csv"))
    check("-main=dist puts the distance answer in 'site'",
          all(row["site"] == row["site_dist"] for row in rows),
          [(row["site"], row["site_dist"]) for row in rows])
    check("-main= rejects an unknown method",
          "takes 'proj' or 'dist'" in run(work, ["-main=xyz"]).stdout)

    # Option 4: every twin against the clean surface.
    #   A_set S000001: main -301, _surface -280, _r1 -291, _r2 -296
    process = run(work, option="4")
    check("option 4 runs without a traceback",
          "Traceback" not in process.stdout and process.returncode == 0,
          process.stdout[-800:])
    path = os.path.join(work, "04_A_set_RedoxVsSurface.csv")
    check("option 4 writes its own file (option 1-3 table not overwritten)",
          os.path.exists(path) and os.path.exists(os.path.join(work, "04_A_set_AlloyAnal.csv")))
    if os.path.exists(path):
        first = read_csv(path)[0]
        check("option 4: main - _surface exact",
              abs(float(first["main - _surface (eV)"]) - (-301.0 + 280.0)) < 1e-6,
              first.get("main - _surface (eV)"))
        check("option 4: _r1 - _surface exact",
              abs(float(first["_r1 - _surface (eV)"]) - (-291.0 + 280.0)) < 1e-6,
              first.get("_r1 - _surface (eV)"))
        check("option 4: _r2 - _surface exact (OSZICAR twin)",
              abs(float(first["_r2 - _surface (eV)"]) - (-296.0 + 280.0)) < 1e-6,
              first.get("_r2 - _surface (eV)"))
        check("option 4: the shared reference energy is reported",
              abs(float(first["E_surface (eV)"]) + 280.0) < 1e-6, first.get("E_surface (eV)"))

    # The buckled slab: the warning has to appear, and -layer_tol has to fix it.
    default_run = run(work).stdout
    check("buckled surface is reported at the default -layer_tol",
          "may be buckled by more than -layer_tol" in default_run,
          default_run[-600:])
    wide = run(work, ["-layer_tol=2.0"]).stdout
    check("raising -layer_tol stops the warning",
          "may be buckled by more than -layer_tol" not in wide, wide[-600:])
    check("buckled surface still gets a site, not a crash",
          "Traceback" not in wide)
finally:
    shutil.rmtree(work, ignore_errors=True)

fails = [r for r in RESULTS if r[0] is False]
skips = [r for r in RESULTS if r[0] is None]
print("\nPASS %d / FAIL %d / SKIP %d"
      % (len([r for r in RESULTS if r[0] is True]), len(fails), len(skips)))
for _ok, name, extra in fails:
    print("  FAIL: %s  <- %s" % (name, extra))
sys.exit(1 if fails else 0)
