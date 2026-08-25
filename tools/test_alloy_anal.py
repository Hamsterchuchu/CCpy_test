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
     methods -- projection in the default run, distance under -main=dist
  2. fcc(100): the four-fold hollow is not mistaken for a bridge
     (the Delaunay diagonal trap), and a real bridge / top still work
  3. the adsorbate is taken from the _surface twin without any heuristic
  4. dE_surface and dE_r1 / dE_r2 match the fake energies exactly
  5. twin folders are not offered as analysis targets of their own
  6. an adsorbate on the bottom face is reported as such
  7. a folder with no OUTCAR / OSZICAR yields a blank energy, not a crash
  8. the csv / txt files are written
  9. the 'site' column follows -main (projection by default, distance when
     asked), and the site file carries ONE answer -- the other method's
     columns are not in it, they are in 04_[SET]_SiteDiff.csv, which appears
     only when the two disagree
 9b. 'side' is left out unless something really sits on the underside
 10. sub_site -- the same classification against the second layer -- matches
     the stacking that was built: an hcp hollow sits on a second-layer ATOM
     ('top'), an fcc hollow on a second-layer hollow, and the four-fold hollow
     of fcc(100) sits on a second-layer atom
 11. option 4 puts every twin on the _surface reference
     (main - _surface, _r1 - _surface, _r2 - _surface) and writes its own file
 12. sites are assigned in the redox twins too, and 'atom' maps a twin atom
     back to the number it has in the main folder even when the removed atom
     shifted the numbering (-no_twins turns the whole thing off); when that
     cross-reference cannot be made (no matching POSCAR position) 'atom' is
     blank, never the twin's own local number mistaken for a main-folder one
 13. an energy that sits far outside the rest of the set is reported by
     structure id (median-absolute-deviation, so the outliers cannot hide
     themselves), and an ordinary spread is NOT reported
 13b. the convergence columns are dropped, with one note, when the verdict
     could not be made for any folder (no vasp.out, or no custodian). The
     True/False path needs custodian and a vasp.out, so it is not covered here
 14. a surface buckled by more than -layer_tol is REPORTED (that split is the
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

    # -- set E: two S adsorbates and a redox twin for each of them, so the
    #    remaining atom's number SHIFTS in one twin but not the other. That is
    #    the case a naive per-folder numbering gets wrong.
    plain = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    both = plain.copy()
    add_adsorbate(both, "S", 1.8, "fcc")     # becomes S#37 in the main folder
    add_adsorbate(both, "S", 1.8, "hcp")     # becomes S#38
    put(os.path.join(root, "E_set", "S000001"), both, -300.0)
    put(os.path.join(root, "E_set_surface", "S000001"), plain, -280.0)
    keep_hcp = both[[i for i in range(len(both) - 2)] + [len(both) - 1]]
    put(os.path.join(root, "E_set_r1", "S000001"), keep_hcp, -290.0)   # S#38 stays
    keep_fcc = both[[i for i in range(len(both) - 1)]]
    put(os.path.join(root, "E_set_r2", "S000001"), keep_fcc, -291.0)   # S#37 stays

    # -- set F: same two-adsorbate layout as E, but the main folder's POSCAR is
    #    gone and its CONTCAR has moved (a relaxed job that only kept CONTCAR).
    #    map_to_main_labels() then falls back to that CONTCAR and the positions
    #    no longer match either twin atom within its 1e-3 tolerance, so the
    #    mapping must come back blank -- not the twin's own local number
    #    mistaken for a main-folder one.
    f_both = plain.copy()
    add_adsorbate(f_both, "S", 1.8, "fcc")    # S#37 in the main folder
    add_adsorbate(f_both, "S", 1.8, "hcp")    # S#38
    put(os.path.join(root, "F_set", "S000001"), f_both, -300.0)
    f_relaxed = f_both.copy()
    f_relaxed.positions += 0.03
    ase_write(os.path.join(root, "F_set", "S000001", "CONTCAR"), f_relaxed, format="vasp")
    os.remove(os.path.join(root, "F_set", "S000001", "POSCAR"))
    put(os.path.join(root, "F_set_surface", "S000001"), plain, -280.0)
    f_keep_hcp = f_both[[i for i in range(len(f_both) - 2)] + [len(f_both) - 1]]
    put(os.path.join(root, "F_set_r1", "S000001"), f_keep_hcp, -290.0)   # S#38 stays

    # -- set G: NO POSCAR anywhere, and CONTCARs that happen to sit exactly on
    #    the generated coordinates. Matching those would "work" -- and that is
    #    the trap: CONTCAR coordinates have relaxed, and on a real run the
    #    substrate atoms move too, so a match found there is a nearest-neighbour
    #    guess wearing an identity's clothes. The map must refuse and say why.
    g_both = plain.copy()
    add_adsorbate(g_both, "S", 1.8, "fcc")
    add_adsorbate(g_both, "S", 1.8, "hcp")
    put(os.path.join(root, "G_set", "S000001"), g_both, -300.0)
    os.remove(os.path.join(root, "G_set", "S000001", "POSCAR"))
    put(os.path.join(root, "G_set_surface", "S000001"), plain, -280.0)
    g_keep_hcp = g_both[[i for i in range(len(g_both) - 2)] + [len(g_both) - 1]]
    put(os.path.join(root, "G_set_r1", "S000001"), g_keep_hcp, -290.0)

    # -- set H: energies in a narrow band, with ONE surface twin 35 eV off.
    #    That is what a job which ended in a different state looks like, and it
    #    is invisible in the site tables: the difference just comes out wrong.
    for number in range(1, 8):
        sid = "S%06d" % number
        one = plain.copy()
        add_adsorbate(one, "S", 1.8, "fcc")
        put(os.path.join(root, "H_set", sid), one, -300.0 - 0.01 * number)
        put(os.path.join(root, "H_set_surface", sid),
            plain, -280.0 - 0.01 * number + (35.0 if number == 3 else 0.0))

    # -- set I: exercises check_converged()'s True/False path against real
    #    custodian, plus the one edge case its own docstring does not spell
    #    out: a folder that has OUTCAR and vasp.out but no vasp.done (a job
    #    killed mid-run -- walltime, a dead node -- leaves exactly this). Only
    #    the branch of vasp_status() that ALSO requires vasp.done fills in a
    #    real True/False; without it the field stays at its unset default,
    #    which is neither "True" nor "False" and must not be read as either.
    incar_text = "NSW = 0\n"
    outcar_text = "  free  energy   TOTEN  =  -100.0 eV\n reached required accuracy\n"
    vaspout_clean = "running on 1 total cores\nWriting wavefunctions\n"
    vaspout_error = "ZBRENT: fatal error in bracketing\nplease rerun with smaller EDIFF\n"

    def put_vasp_run(directory, atoms, energy, vasp_out, mark_done):
        put(directory, atoms, energy)
        with open(os.path.join(directory, "INCAR"), "w") as f:
            f.write(incar_text)
        with open(os.path.join(directory, "OUTCAR"), "w") as f:
            f.write(outcar_text)
        with open(os.path.join(directory, "vasp.out"), "w") as f:
            f.write(vasp_out)
        if mark_done:
            open(os.path.join(directory, "vasp.done"), "w").close()

    i_main = plain.copy()
    add_adsorbate(i_main, "S", 1.8, "fcc")
    put_vasp_run(os.path.join(root, "I_set", "S000001"), i_main, -300.0,
                 vaspout_clean, mark_done=True)
    put_vasp_run(os.path.join(root, "I_set_surface", "S000001"), plain, -280.0,
                 vaspout_clean, mark_done=False)   # the vasp.done-less crash
    put_vasp_run(os.path.join(root, "I_set_r1", "S000001"), i_main, -290.0,
                 vaspout_error, mark_done=True)     # a real custodian error

    # -- set J: ONLY the vasp.done-less edge case, alone, so its verdict is the
    #    single one across the whole set. That makes the "all folders unknown"
    #    cleanup (column dropped + one note) the observable proxy for whether
    #    this case actually normalizes to "Unknown": if it fell through as the
    #    unset default instead, {"Unknown"} would not match and the column
    #    would stay -- with that blank sitting in it unexplained.
    j_main = plain.copy()
    add_adsorbate(j_main, "S", 1.8, "fcc")
    put_vasp_run(os.path.join(root, "J_set", "S000001"), j_main, -300.0,
                 vaspout_clean, mark_done=False)

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

    # The distance method no longer has a column of its own in the site file,
    # so its answer is checked from a run that asks for it as the main one.
    run(work, ["-main=dist"])
    dist_sites = {}
    for name in ("A_set", "B_set", "C_set"):
        path = os.path.join(work, "04_%s_AdsorptionSites.csv" % name)
        if os.path.exists(path):
            for row in read_csv(path):
                if row.get("folder", "main") == "main":
                    dist_sites[(name, row["Structure"], row["atom"].split("#")[0])] = row
    run(work)

    sites = {}
    for name in ("A_set", "B_set", "C_set", "D_set"):
        path = os.path.join(work, "04_%s_AdsorptionSites.csv" % name)
        if os.path.exists(path):
            for row in read_csv(path):
                if row.get("folder", "main") == "main":
                    element = row["atom"].split("#")[0]
                    sites[(name, row["Structure"], element)] = row

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
        dist_row = dist_sites.get((set_name, sid, element), {})
        check("%s %s distance method -> %s" % (set_name, sid, expected_label),
              dist_row.get("site") == expected_label, dist_row.get("site"))
        # The projection method does not tell fcc from hcp any better than the
        # distance one does, so only the site type is compared there.
        want = expected_label.split("-")[0]
        check("%s %s projection method -> %s" % (set_name, sid, want),
              row["site"].split("-")[0] == want, row["site"])
        check("%s %s both methods agree here" % (set_name, sid),
              row["agree"] == "same", row["agree"])
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
          output.count("# ---------- ") == 10 and "_surface ---" not in output,
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
          rows[0]["site"] == "top" and rows[3]["site"] == "hollow3-hcp",
          [row["site"] for row in rows])
    run(work)
    check("-main= rejects an unknown method",
          "takes 'proj' or 'dist'" in run(work, ["-main=xyz"]).stdout)

    # The site file must carry one answer, not both, and no bookkeeping columns.
    header = list(read_csv(os.path.join(work, "04_A_set_AdsorptionSites.csv"))[0])
    for gone in ("site_dist", "site_proj", "neighbors_dist", "neighbors_proj",
                 "element", "atom_no", "main_atom", "d_min_sub (A)"):
        check("site file drops the '%s' column" % gone, gone not in header, header)
    for kept in ("atom", "local_no", "site", "ensemble", "neighbors",
                 "sub_site", "sub_ensemble", "agree"):
        check("site file keeps the '%s' column" % kept, kept in header, header)
    check("'side' is left out when nothing is on the underside",
          "side" not in header, header)
    check("'side' appears where an atom IS on the underside",
          "side" in list(read_csv(os.path.join(work, "04_C_set_AdsorptionSites.csv"))[0]))
    # The element composition of the site, and its roll-up.
    a_rows = read_csv(os.path.join(work, "04_A_set_AdsorptionSites.csv"))
    check("ensemble names the elements of the site",
          all(row["ensemble"] and row["ensemble"][0].isalpha() for row in a_rows),
          [row["ensemble"] for row in a_rows])
    check("ensemble size matches the site geometry",
          all(sum(int(c) for c in row["ensemble"] if c.isdigit())
              == {"top": 1, "bridge": 2, "hollow3": 3, "hollow4": 4}[row["site"].split("-")[0]]
              for row in a_rows),
          [(row["site"], row["ensemble"]) for row in a_rows])
    ens_path = os.path.join(work, "04_A_set_SiteEnsembles.csv")
    check("ensemble roll-up file is written", os.path.exists(ens_path))
    if os.path.exists(ens_path):
        ens = read_csv(ens_path)
        check("roll-up counts every adsorbate atom once",
              sum(int(row["atoms"]) for row in ens) == len(a_rows),
              (sum(int(row["atoms"]) for row in ens), len(a_rows)))
        check("roll-up counts structures as well as atoms",
              all(int(row["structures"]) <= int(row["atoms"]) for row in ens))
    # F_set is the set whose twin cannot be mapped, so its 'atom' is blank.
    # The roll-up must still file those atoms under their real element.
    f_ens = os.path.join(work, "04_F_set_SiteEnsembles.csv")
    if os.path.exists(f_ens):
        check("a blank 'atom' does not lose the element in the roll-up",
              all(row["element"] for row in read_csv(f_ens)),
              [row["element"] for row in read_csv(f_ens)])

    # No POSCAR -> the cross-reference is refused outright, even though the
    # CONTCARs would have matched. Blank column, and a stated reason.
    g_rows = [row for row in read_csv(os.path.join(work, "04_G_set_AdsorptionSites.csv"))
              if row["folder"] != "main"]
    check("no POSCAR -> the twin's atom stays blank instead of matching CONTCARs",
          g_rows and all(row["atom"] == "" for row in g_rows),
          [row["atom"] for row in g_rows])
    check("and the reason is printed, not left as a silent blank",
          "cannot be cross-referenced" in output, output[-900:])

    check("no diff file when the methods never disagree",
          not os.path.exists(os.path.join(work, "04_A_set_SiteDiff.csv")))

    # A wide -tol makes the distance method over-count, so the two disagree and
    # the diff file has to appear with BOTH answers in it.
    wide_tol = run(work, ["-tol=0.6"])
    check("a disagreement is announced on screen",
          "site methods:" in wide_tol.stdout and "of" in wide_tol.stdout,
          wide_tol.stdout[-400:])
    diff_path = os.path.join(work, "04_A_set_SiteDiff.csv")
    check("diff file is written when the methods disagree", os.path.exists(diff_path))
    if os.path.exists(diff_path):
        diff_rows = read_csv(diff_path)
        check("diff file carries both answers",
              all(row["site_proj"] and row["site_dist"] and row["agree"] != "same"
                  for row in diff_rows),
              diff_rows[:1])
        check("diff file holds only the disagreeing atoms",
              len(diff_rows) < len(read_csv(os.path.join(work, "04_A_set_AdsorptionSites.csv"))),
              len(diff_rows))
    run(work)

    # Sites in the redox twins, and the number map back to the main folder.
    twin_rows = read_csv(os.path.join(work, "04_E_set_AdsorptionSites.csv"))
    by_folder = {row["folder"]: row for row in twin_rows if row["folder"] != "main"}
    check("redox twins get their own site rows",
          set(by_folder) == {"r1", "r2"}, sorted(by_folder))
    if set(by_folder) == {"r1", "r2"}:
        # _r1 kept the hcp atom: folder-local #37, main #38 -- the shifted case.
        check("the atom keeps its main-folder identity in _r1, with the "
              "folder's own number beside it",
              by_folder["r1"]["atom"] == "S#38" and by_folder["r1"]["local_no"] == "37",
              (by_folder["r1"]["atom"], by_folder["r1"]["local_no"]))
        check("_r1 keeps the hcp site", by_folder["r1"]["site"] == "hollow3-hcp",
              by_folder["r1"]["site"])
        # _r2 kept the fcc atom, whose number did not move.
        check("identity unchanged where nothing shifted",
              by_folder["r2"]["atom"] == "S#37" and by_folder["r2"]["local_no"] == "37",
              (by_folder["r2"]["atom"], by_folder["r2"]["local_no"]))
        check("_r2 keeps the fcc site", by_folder["r2"]["site"] == "hollow3-fcc",
              by_folder["r2"]["site"])
    e_row = [row for row in read_csv(os.path.join(work, "04_E_set_AlloyAnal.csv"))][0]
    check("the per-structure table carries energies only, not sites",
          not any(key.startswith("Sites") for key in e_row), list(e_row))

    # Set F: the main folder's POSCAR is gone and its CONTCAR has moved, so the
    # cross-reference cannot match either twin atom. 'atom' has to come back
    # blank -- the whole point of the fix is that it must NOT silently show
    # the twin's own local number as if it were a main-folder identity.
    f_rows = read_csv(os.path.join(work, "04_F_set_AdsorptionSites.csv"))
    f_twin = [row for row in f_rows if row["folder"] != "main"]
    check("a redox twin whose atoms cannot be cross-referenced gets a blank "
          "'atom', not its own local number mistaken for one",
          len(f_twin) == 1 and f_twin[0]["atom"] == "" and f_twin[0]["local_no"] == "37",
          f_twin)

    no_twins = run(work, ["-no_twins"])
    check("-no_twins runs without a traceback",
          "Traceback" not in no_twins.stdout and no_twins.returncode == 0,
          no_twins.stdout[-800:])
    check("-no_twins leaves only the main folder",
          all(row["folder"] == "main"
              for row in read_csv(os.path.join(work, "04_E_set_AdsorptionSites.csv"))))
    run(work)   # restore the full tables for whatever follows

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

    # The energy that does not belong to the band, named by structure.
    check("an energy far outside the set is reported, by structure id",
          "far from the rest of the set" in output and "S000003" in output,
          [line for line in output.splitlines() if "far from the rest" in line])
    check("an ordinary spread of energies is not reported as an outlier",
          sum(1 for line in output.splitlines() if "far from the rest" in line) == 1,
          [line for line in output.splitlines() if "far from the rest" in line])
    check("no vasp.out anywhere -> the convergence columns are dropped, "
          "with one note",
          "Converged" not in list(read_csv(os.path.join(work, "04_A_set_AlloyAnal.csv"))[0])
          and "convergence was not checked" in output)

    # The True/False path needs real custodian; skip gracefully without it,
    # same as the module-wide ase check at the top of this file.
    try:
        import custodian  # noqa: F401
        have_custodian = True
    except Exception:
        have_custodian = False
    if not have_custodian:
        skip("custodian available (check_converged True/False/edge-case path)",
             "custodian not installed")
    else:
        i_row = read_csv(os.path.join(work, "04_I_set_AlloyAnal.csv"))[0]
        check("a cleanly finished main folder is Converged = True",
              i_row.get("Converged") == "True", i_row.get("Converged"))
        check("a twin with a real custodian-caught error is named in "
              "'unconverged'",
              i_row.get("unconverged") == "_r1", i_row.get("unconverged"))
        check("a twin with OUTCAR+vasp.out but no vasp.done is not counted "
              "as a plain failure (that would hide a genuinely bad twin "
              "behind a wrong-looking pass)",
              "_surface" not in (i_row.get("unconverged") or "").split(","),
              i_row.get("unconverged"))

        # J_set isolates that same vasp.done-less folder as the ONLY verdict
        # in its set: the column-drop cleanup only fires on an exact
        # {"Unknown"} match, so this is what actually proves the value came
        # out as "Unknown" and not some other blank standing in for it.
        j_row = read_csv(os.path.join(work, "04_J_set_AlloyAnal.csv"))[0]
        check("OUTCAR+vasp.out without vasp.done normalizes to 'Unknown' "
              "(seen via: the all-Unknown column drop fires for it too, same "
              "as the no-vasp.out case)",
              "Converged" not in j_row,
              j_row)

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
