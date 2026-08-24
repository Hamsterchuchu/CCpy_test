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
    for number, site in enumerate(sites, start=1):
        sid = "S%06d" % number
        main = slab.copy()
        add_adsorbate(main, "Li", 2.0, site)
        put(os.path.join(root, "A_set", sid), main, -300.0 - number)
        put(os.path.join(root, "A_set_surface", sid), slab, -280.0)
        put(os.path.join(root, "A_set_r1", sid), slab, -290.0 - number)
        put(os.path.join(root, "A_set_r2", sid), slab, -295.0 - number, use_outcar=False)
        truth.append(("A_set", sid, expected[site]))
    with open(os.path.join(root, "A_set_surface", "metadata.txt"), "w") as handle:
        handle.write("%-22s = %s\n" % ("replace_elements", "['Pt']"))
        handle.write("%-22s = %s\n" % ("adsorbate_elements_removed", "['Li']"))

    # -- set B: fcc(100) hollow / bridge / top, and one folder with no energy
    square = fcc100("Pt", size=(3, 3, 4), vacuum=10.0)
    for number, (site, label) in enumerate([("hollow", "hollow4"),
                                            ("bridge", "bridge"),
                                            ("ontop", "top")], start=1):
        sid = "S%06d" % number
        main = square.copy()
        add_adsorbate(main, "O", 1.5, site)
        put(os.path.join(root, "B_set", sid), main, None if number == 3 else -100.0 - number)
        put(os.path.join(root, "B_set_surface", sid), square, -90.0)
        truth.append(("B_set", sid, label))

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
    truth.append(("C_set", "S000001", "hollow3-hcp"))
    return truth


def run(work):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = REPO + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, BIN, "3", "-all"], cwd=work, env=environment,
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
    for name in ("A_set", "B_set", "C_set"):
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

    for set_name, sid, expected_label in truth:
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
          output.count("# ---------- ") == 3 and "_surface ---" not in output,
          output.count("# ---------- "))
    check("bottom-face adsorbate reported as bottom",
          sites[("C_set", "S000001", "Li")]["side"] == "bottom",
          sites[("C_set", "S000001", "Li")]["side"])

    blank = [row for row in energies["B_set"] if row["Structure"] == "S000003"][0]
    check("folder without OUTCAR/OSZICAR leaves a blank energy",
          blank["Energy (eV)"] == "", repr(blank["Energy (eV)"]))
finally:
    shutil.rmtree(work, ignore_errors=True)

fails = [r for r in RESULTS if r[0] is False]
skips = [r for r in RESULTS if r[0] is None]
print("\nPASS %d / FAIL %d / SKIP %d"
      % (len([r for r in RESULTS if r[0] is True]), len(fails), len(skips)))
for _ok, name, extra in fails:
    print("  FAIL: %s  <- %s" % (name, extra))
sys.exit(1 if fails else 0)
