#!/usr/bin/env python
"""Self-check for CCpyAlloyGen reuse mode (option 6) and the substrate guess.

Run from the repo root:

    python3 tools/test_alloy_reuse.py

Everything happens in a temporary directory. The reference sets are built here
with ase, so the right answer is known exactly.

Checks
  1. guess_substrate_elements: an adsorbate with more H atoms than any single
     metal of a mixed slab (CHP-like C9H18O on a 5-element slab) is still an
     adsorbate -- it used to become the "backbone" and land in the pool
  2. layered alloys, H* and a clean metal are judged as before
  3. reuse copies the arrangement ID by ID onto a new adsorbate; coordinates
     and adsorbate of the new input are untouched, IDs (with gaps) are kept
  4. a relaxed reference (noise, 1.5% in-plane strain, slab moved along z,
     scrambled atom order, CONTCAR) maps to the same arrangement
     4b. a reference of the same slab with a thicker vacuum layer, wrapped
     across the cell boundary, maps exactly (z is compared in Angstrom)
  5. a reference slid sideways by about half a site or more is refused
     instead of being mapped onto the neighbouring sites (a slide by an exact
     surface lattice vector is the same structure translated and cannot be
     told apart -- that limitation is documented, not tested)
  6. a missing atom / mixed compositions in the reference are refused
  7. flat S000001.cif references work, and the _surface twin is written
"""
import os
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ase import Atoms
from ase.build import fcc111, add_adsorbate, molecule
from ase.io import read, write

from CCpy.VASP.AlloyGen import generate_structures, guess_substrate_elements

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("PASS  " + name)
    else:
        FAIL += 1
        print("FAIL  " + name + ("  -- " + detail if detail else ""))


METALS = ["Fe", "Co", "Ni", "Cu", "Pt"]


def hea_slab(seed):
    slab = fcc111("Pt", (4, 4, 5), a=3.64, vacuum=8.0)
    elems = np.repeat(METALS, 16)
    np.random.default_rng(seed).shuffle(elems)
    slab.set_chemical_symbols(list(elems))
    return slab


def put(slab, mol, height=2.2):
    mol = mol.copy()
    mol.positions -= mol.positions.mean(0)
    c = slab.cell.array
    top = slab.positions[:, 2].max()
    mol.positions += [*(c[0] + c[1])[:2] / 2, top + height - mol.positions[:, 2].min()]
    out = slab + mol
    out.cell = slab.cell
    out.pbc = True
    return out


def metal_species_by_position(atoms):
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in METALS]
    order = np.lexsort(np.round(atoms.positions[idx], 4).T[::-1])
    return [atoms.get_chemical_symbols()[idx[k]] for k in order]


def run(**kw):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return generate_structures(**kw)


with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)

    # 1-2. substrate guess -------------------------------------------------
    chp = Atoms("C9H18O", positions=np.random.default_rng(0).normal(0, 1.2, (28, 3)))
    sub, ads, _ = guess_substrate_elements(put(hea_slab(1), chp, 2.5))
    check("1  H-rich adsorbate on a 5-element slab is an adsorbate",
          sub == sorted(METALS) and ads == ["C", "H", "O"], f"{sub} / {ads}")
    s = fcc111("Ni", (3, 3, 4), vacuum=8.0)
    z = np.round(s.positions[:, 2], 2)
    layers = sorted(set(z))
    for i in range(len(s)):
        if layers.index(z[i]) % 2:
            s[i].symbol = "Cu"
    check("2a layered NiCu stays all substrate", guess_substrate_elements(s)[:2] == (["Cu", "Ni"], []))
    h = fcc111("Pt", (3, 3, 4), vacuum=8.0)
    add_adsorbate(h, "H", 0.9, "fcc")
    check("2b H* on Pt is an adsorbate", guess_substrate_elements(h)[:2] == (["Pt"], ["H"]))

    # 3. exact reuse ----------------------------------------------------------
    ref_ids = [2, 7, 11]
    os.makedirs("ref", exist_ok=True)
    refs = {}
    for k, sid in enumerate(ref_ids):
        a = put(hea_slab(10 + k), chp, 2.6)
        refs[sid] = a
        os.makedirs(f"ref/S{sid:06d}")
        write(f"ref/S{sid:06d}/POSCAR", a, format="vasp", direct=True, sort=True, vasp5=True)
    pt = fcc111("Pt", (4, 4, 5), a=3.64, vacuum=8.0)
    new = put(pt, molecule("C6H6"), 2.2)
    new = new[np.random.default_rng(3).permutation(len(new))]
    write("new.vasp", new, format="vasp", direct=True, vasp5=True)
    res = run(input_file="new.vasp", output_dir="out", replace_element="Pt", composition=None,
              mode="reuse", ref_dir="ref", vasp_folder=True, output_format="vasp",
              adsorbate_elements="C,H")
    ok = res["unique_structures_saved"] == 3
    for sid in ref_ids:
        o = read(f"out/S{sid:06d}/POSCAR", format="vasp")
        ok &= metal_species_by_position(o) == metal_species_by_position(refs[sid])
        mol_o = sorted(map(tuple, np.round(o.positions[[i for i, s in enumerate(o.get_chemical_symbols()) if s in "CH"]], 6)))
        mol_n = sorted(map(tuple, np.round(new.positions[[i for i, s in enumerate(new.get_chemical_symbols()) if s in "CH"]], 6)))
        ok &= mol_o == mol_n
    check("3  arrangement copied ID by ID, new adsorbate untouched", ok)
    check("3b IDs with gaps are kept", sorted(os.listdir("out")) [:3] == [f"S{i:06d}" for i in ref_ids])
    check("7b _surface twin written", sorted(d for d in os.listdir("out_surface") if d.startswith("S")) == [f"S{i:06d}" for i in ref_ids])

    # 4. relaxed / strained / moved along z ------------------------------------
    rng = np.random.default_rng(5)
    for sid, a in refs.items():
        b = a.copy()
        m = [i for i, s in enumerate(b.get_chemical_symbols()) if s in METALS]
        b.positions[m] += rng.normal(0, 0.12, (len(m), 3))
        b.set_cell(b.cell.array * np.array([1.015, 1.015, 1.0])[:, None], scale_atoms=True)
        b.positions[:, 2] -= 3.0
        b.wrap()
        b = b[rng.permutation(len(b))]
        os.makedirs(f"refR/S{sid:06d}")
        write(f"refR/S{sid:06d}/CONTCAR", b, format="vasp", direct=True, vasp5=True)
    run(input_file="new.vasp", output_dir="outR", replace_element="Pt", composition=None,
        mode="reuse", ref_dir="refR", ref_file="CONTCAR", vasp_folder=True, output_format="vasp")
    same = all(
        read(f"outR/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        == read(f"out/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        for sid in ref_ids
    )
    check("4  relaxed/strained/z-shifted CONTCAR reference -> same arrangement", same)

    # 4b. same slab, more vacuum, split across the cell boundary --------------
    for sid, a in refs.items():
        b = a.copy()
        cell = b.cell.array.copy()
        cell[2] += [0.0, 0.0, 8.0]          # 8 A more vacuum, same slab
        b.set_cell(cell, scale_atoms=False)
        b.positions[:, 2] += 0.5 * cell[2, 2]
        b.wrap()                            # slab now wraps through z = 0
        os.makedirs(f"refV/S{sid:06d}")
        write(f"refV/S{sid:06d}/POSCAR", b, format="vasp", direct=True, sort=True, vasp5=True)
    run(input_file="new.vasp", output_dir="outV", replace_element="Pt", composition=None,
              mode="reuse", ref_dir="refV", vasp_folder=True, output_format="vasp")
    same = all(
        read(f"outV/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        == read(f"out/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        for sid in ref_ids
    )
    import csv
    with open("outV/reuse_map.csv") as fh:
        worst = max(float(r["max_dev_A"]) for r in csv.DictReader(fh))
    check("4b reference with thicker vacuum (wrapped) -> same arrangement, ~0 A",
          same and worst < 1e-3, f"max_dev {worst:.4f} A")

    # 5. slid sideways ---------------------------------------------------------
    nn = 3.64 / np.sqrt(2)
    for sid, a in refs.items():
        b = a.copy()
        b.positions[:, 0] += 0.6 * nn
        b.positions[:, 1] += 0.3 * nn
        b.wrap()
        os.makedirs(f"refS/S{sid:06d}")
        write(f"refS/S{sid:06d}/POSCAR", b, format="vasp", direct=True, vasp5=True)
    try:
        run(input_file="new.vasp", output_dir="outS", replace_element="Pt", composition=None,
            mode="reuse", ref_dir="refS", vasp_folder=True, output_format="vasp", ref_tol=0.8)
        check("5  sideways slide refused", False)
    except ValueError:
        check("5  sideways slide refused", not os.path.exists("outS"))

    # 6. bad references --------------------------------------------------------
    a = refs[2].copy()
    del a[[i for i, s in enumerate(a.get_chemical_symbols()) if s in METALS][0]]
    os.makedirs("refM/S000001")
    write("refM/S000001/POSCAR", a, format="vasp", direct=True, vasp5=True)
    try:
        run(input_file="new.vasp", output_dir="outM", replace_element="Pt", composition=None,
            mode="reuse", ref_dir="refM", vasp_folder=True, output_format="vasp")
        check("6a missing atom refused", False)
    except ValueError:
        check("6a missing atom refused", not os.path.exists("outM"))
    for sid, a in refs.items():
        b = a.copy()
        if sid == 7:
            i = [k for k, s in enumerate(b.get_chemical_symbols()) if s == "Fe"][0]
            b[i].symbol = "Co"
        os.makedirs(f"refC/S{sid:06d}")
        write(f"refC/S{sid:06d}/POSCAR", b, format="vasp", direct=True, vasp5=True)
    try:
        run(input_file="new.vasp", output_dir="outC", replace_element="Pt", composition=None,
            mode="reuse", ref_dir="refC", vasp_folder=True, output_format="vasp")
        check("6b mixed compositions refused", False)
    except ValueError:
        check("6b mixed compositions refused", True)

    # 7. flat cif reference ----------------------------------------------------
    os.makedirs("refF")
    for sid, a in refs.items():
        write(f"refF/S{sid:06d}.cif", a)
    run(input_file="new.vasp", output_dir="outF", replace_element="Pt", composition=None,
        mode="reuse", ref_dir="refF", vasp_folder=True, output_format="vasp")
    same = all(
        read(f"outF/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        == read(f"out/S{sid:06d}/POSCAR", format="vasp").get_chemical_symbols()
        for sid in ref_ids
    )
    check("7  flat S######.cif reference", same)
    os.chdir("/")

print(f"\n{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
