#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CCpy.Tools.CCpyTools import find_convex_hull  # noqa: F401
from CCpy.VASP.VASPio import VASPOutput

from pymatgen.core.structure import IStructure
from pymatgen.core.composition import Composition


class CASMhull:
    def __init__(self, base=None, tot_base=None, structure_sets=None,
                 output_dir="Data", count_tol=1e-3, verbose=False):
        pwd = os.getcwd()
        self.pwd, self.base, self.tot_base = pwd, base, tot_base
        if "Data" not in os.listdir("./"):
            os.mkdir("Data")

        self.structure_sets = structure_sets or [(None, ".")]
        self.count_tol = count_tol
        self.verbose = verbose
        self.data_dir = os.path.join(self.pwd, output_dir)
        self.warnings = []

    def parsingData(self):
        records = self._scan_all()
        if not records:
            raise ValueError("No structures found (check structure_sets).")

        groups = {}
        for r in records:
            groups.setdefault(self._group_key(r), []).append(r)

        rows = []
        for gkey, group_records in groups.items():
            cap_info = self._resolve_capacity(group_records)
            capacity, n_fu = cap_info["capacity"], cap_info["n_fu"]

            for r in group_records:
                dirpath = os.path.join(self.pwd, r["rel_path"])
                outcar_path = os.path.join(dirpath, "OUTCAR")
                if not os.path.isfile(outcar_path):
                    self._warn("{}: no OUTCAR (skipped)".format(r["rel_path"]))
                    continue
                try:
                    e = self._read_toten(outcar_path)
                except Exception as err:
                    self._warn("{}: failed to parse energy ({})".format(r["rel_path"], err))
                    continue

                n_base = float(r["elts"].get(self.base, 0.0))
                con = round(n_base / capacity, 6)
                e_fu = e / n_fu

                rows.append({
                    "Concentration": con, "Energy": e, "Energy/f.u.": e_fu,
                    "Number of base atom": n_base, "Directory": r["dirname"],
                    "Set": r["set"], "Path": r["rel_path"], "SCEL_group": gkey,
                    "Mode": cap_info["mode"],
                })

        if not rows:
            raise ValueError("No structure parsed successfully all the way to energy.")

        df = pd.DataFrame(rows)
        con0_energy, con0_c = pick_endpoint_energy(
            df["Concentration"], df["Energy/f.u."], 0.0, self.count_tol)
        con1_energy, con1_c = pick_endpoint_energy(
            df["Concentration"], df["Energy/f.u."], 1.0, self.count_tol)

        if con0_energy is None or con1_energy is None:
            raise ValueError(
                "Could not find a structure near concentration 0.0 or 1.0 (tol={}). "
                "con0 candidates={}, con1 candidates={}".format(
                    self.count_tol, len(con0_c), len(con1_c))
            )
        self._log("con=0.0 reference energy: {} (lowest of {} candidates)".format(con0_energy, len(con0_c)))
        self._log("con=1.0 reference energy: {} (lowest of {} candidates)".format(con1_energy, len(con1_c)))

        self.df = df
        # keep original's con1_energy / con_energy(=con0) naming for compat
        self.con1_energy, self.con_energy = con1_energy, con0_energy
        self.con0_energy = con0_energy
        return df

    def getFormData(self):
        df = self.df
        con1_energy, con0_energy = self.con1_energy, self.con_energy

        df['Formation energy'] = formation_energy(
            df['Energy/f.u.'], df['Concentration'], con0_energy, con1_energy)
        df = df.sort_values(by='Concentration').reset_index(drop=True)
        df.to_csv(self._outpath("01_formation_energy.csv"))
        print("Data saved : " + self._outpath("01_formation_energy.csv"))
        if self.warnings:
            print("\n[summary] {} warning(s)/failure(s) during processing "
                  "(see self.warnings).".format(len(self.warnings)))

        self.df = df
        return df

    def makeHull(self):
        df = self.df
        points = df[["Concentration", "Formation energy"]].to_numpy()
        hull_idx = lower_convex_hull_indices(points)

        hull_df = df.iloc[hull_idx].copy()
        bad = hull_df[hull_df["Formation energy"] > 1e-6]
        if len(bad):
            self._warn("{} lower-hull point(s) with Formation energy > 0 found "
                       "(possible numerical error, please check)".format(len(bad)))

        cols = ["Concentration", "Formation energy", "Energy", "Energy/f.u.",
                "Number of base atom", "Directory", "Set", "Path", "Mode"]
        hull_df = hull_df[cols].sort_values('Concentration').reset_index(drop=True)

        hull_df.to_csv(self._outpath("02_convex_hull_points.csv"))
        print("Data saved : " + self._outpath("02_convex_hull_points.csv"))
        print("\n* Hull points")
        pd.set_option('expand_frame_repr', False)
        print(hull_df)

        self.hull_df = hull_df
        return hull_df

    def plotHull(self):
        df, hull_df = self.df, self.hull_df
        plt.figure()

        sets = df["Set"].unique().tolist()
        if len(sets) <= 1:
            plt.scatter(df['Concentration'], df['Formation energy'],
                        marker="D", color='b', s=10)
        else:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, s in enumerate(sets):
                sub = df[df["Set"] == s]
                plt.scatter(sub["Concentration"], sub["Formation energy"],
                            marker="D", s=20, alpha=0.4,
                            color=colors[i % len(colors)], label=str(s))
            plt.legend()

        plt.plot(hull_df['Concentration'], hull_df['Formation energy'],
                  marker='o', color="r", alpha=0.7, ms=8)
        plt.axhline(y=0, color='gray', lw=1, ls='--')
        plt.xlim(0.0, 1.0)
        plt.xlabel(str(self.base) + " Concentration", fontsize=20)
        plt.ylabel("Formation energy (eV/f.u.)", fontsize=20)
        plt.tight_layout()

        figname = self._outpath("03_convexhull.png")
        jpg = figname.replace(".png", ".jpg")
        plt.savefig(figname)
        print("\nFigure saved : " + figname)
        try:
            plt.savefig(jpg)
            print("Figure saved : " + jpg)
        except Exception as err:
            self._warn("failed to save jpg (Pillow may not be installed): {}".format(err))
        plt.close()

    def getHullPointStructures(self):
        hull_df = self.hull_df
        if hull_df is None or len(hull_df) == 0:
            return

        print("Hull point structures")
        cwd_before = os.getcwd()
        for _, row in hull_df.iterrows():
            dirpath = os.path.join(self.pwd, row["Path"])
            set_tag = (str(row["Set"]) + "_") if row["Set"] else ""
            # must pass target_name explicitly (default is the literal string
            # "None", so every hull point would overwrite the same file)
            target_name = "hull_{}{}_con{:.4f}.cif".format(
                set_tag, row["Directory"], row["Concentration"])
            try:
                os.chdir(dirpath)
                VO = VASPOutput()
                VO.getFinalStructure(target_name=target_name,
                                      path=os.path.join(self.data_dir, ""))
            except Exception as err:
                self._warn("{}: failed to extract structure ({})".format(row["Path"], err))
            finally:
                os.chdir(cwd_before)

    def getVoltageProfile(self, chempot):
        df = self.hull_df
        cons = df['Concentration'].tolist()
        energies = df['Energy'].tolist()
        n_of_atoms = df['Number of base atom'].tolist()

        x, y = [], []
        for i in range(len(energies) - 1):
            j = i + 1
            e1, e2 = energies[i], energies[j]
            n1, n2 = n_of_atoms[i], n_of_atoms[j]
            diff_base = n2 - n1
            if diff_base == 0:
                continue
            pot = ((e2 - e1) / diff_base) - chempot
            vol = -pot

            if i == 0:
                x.append(cons[i])
            else:
                x.append(cons[i])
                x.append(cons[i])
            y.append(vol)
            y.append(vol)
        x.append(1.0)

        voltage_df = pd.DataFrame({'Concentration': x, 'Voltage': y})
        voltage_df.to_csv(self._outpath("04_voltage_profile.csv"))
        print("\nData saved : " + self._outpath("04_voltage_profile.csv"))

        plt.figure()
        plt.plot(x, y, lw=2)
        plt.xlim(0.0, 1.0)
        plt.xlabel(str(self.base) + " Concentration", fontsize=20)
        plt.ylabel("Voltage (V)", fontsize=20)
        figname = self._outpath("05_voltage_profile.png")
        plt.savefig(figname)
        print("\nFigure saved : " + figname)
        try:
            plt.savefig(figname.replace(".png", ".jpg"))
            print("Figure saved : " + figname.replace(".png", ".jpg"))
        except Exception as err:
            self._warn("failed to save jpg: {}".format(err))
        plt.close()
        return voltage_df

    def mainFlow(self, chempot=None):
        self.parsingData()
        self.getFormData()
        self.makeHull()
        self.plotHull()
        self.getHullPointStructures()
        if self.base == "Li":
            self.getVoltageProfile(chempot)

    # -- internal utils (new) ------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _warn(self, msg):
        self.warnings.append(msg)
        print("[warning] " + msg, file=sys.stderr)

    def _outpath(self, filename):
        return os.path.join(self.data_dir, filename)

    @staticmethod
    def _parse_scel_index(dirname):
        if not dirname.startswith("con"):
            return None
        head = dirname.split(".")[0]
        try:
            return int(head[len("con"):])
        except ValueError:
            return None

    @staticmethod
    def _read_scel_vols(root_dir):
        path = os.path.join(root_dir, "SCEL")
        if not os.path.isfile(path):
            return None
        vols = {}
        idx = 0
        with open(path) as f:
            for line in f:
                if "volume" in line.lower():
                    vols[idx] = float(line.split()[-1])
                    idx += 1
        return vols or None

    @staticmethod
    def _read_toten(outcar_path):
        last = None
        with open(outcar_path, "r", errors="ignore") as f:
            for line in f:
                if "free  energy   TOTEN" in line:
                    last = line
        if last is None:
            raise RuntimeError("TOTEN not found in OUTCAR (calculation may be incomplete)")
        return float(last.split()[-2])

    def _scan_set(self, set_name, root_dir):
        records = []
        full_root = os.path.join(self.pwd, root_dir)
        if not os.path.isdir(full_root):
            self._warn("set '{}': directory not found -> {}".format(set_name, full_root))
            return records

        sub_ds = sorted(
            sd for sd in os.listdir(full_root)
            if os.path.isdir(os.path.join(full_root, sd)) and "con" in sd
        )
        self._log("[{}] found {} con* directories".format(set_name or root_dir, len(sub_ds)))

        for sd in sub_ds:
            sub_path = os.path.join(full_root, sd)
            poscar_path = os.path.join(sub_path, "POSCAR")
            if not os.path.isfile(poscar_path):
                self._warn("{}: no POSCAR (skipped)".format(os.path.join(root_dir, sd)))
                continue
            try:
                poscar = IStructure.from_file(poscar_path)
                species = poscar.species
                elts_dict = {}
                for sp in poscar.types_of_specie:
                    elts_dict[str(sp)] = species.count(sp)
                records.append({
                    "set": set_name, "root_dir": root_dir, "dirname": sd,
                    "rel_path": os.path.join(root_dir, sd),
                    "scel_idx": self._parse_scel_index(sd), "elts": elts_dict,
                })
            except Exception as err:
                self._warn("{}: failed to parse POSCAR ({})".format(os.path.join(root_dir, sd), err))
        return records

    def _scan_all(self):
        records = []
        for set_name, root_dir in self.structure_sets:
            records.extend(self._scan_set(set_name, root_dir))
        return records

    @staticmethod
    def _group_key(record):
        return record["scel_idx"] if record["scel_idx"] is not None else "__single__"

    def _resolve_capacity(self, group_records):
        elts_dicts = [r["elts"] for r in group_records]
        mode, info = detect_capacity_mode(elts_dicts, self.base, self.count_tol)

        scel_idx = group_records[0]["scel_idx"]
        vol = None
        for set_name, root_dir in self.structure_sets:
            vols = self._read_scel_vols(root_dir)
            if vols and scel_idx in vols:
                vol = vols[scel_idx]
                break

        if mode == "host_guest":
            host_dict = info
            if self.tot_base is not None and vol is not None:
                capacity = self.tot_base * vol
                self._log("  [capacity] host_guest, using SCEL: tot_base({})*vol({})={}"
                          .format(self.tot_base, vol, capacity))
            elif self.tot_base is not None:
                capacity = self.tot_base
                self._log("  [capacity] host_guest, using tot_base directly: {}".format(capacity))
            else:
                capacity = max(ed.get(self.base, 0.0) for ed in elts_dicts)
                self._log("  [capacity] host_guest, auto-detected reference structure: capacity={} (host={})"
                          .format(capacity, host_dict))
            if capacity <= 0:
                raise ValueError(
                    "host_guest capacity computed as <= 0 (scel_idx={}). "
                    "The fully-lithiated reference structure may be missing from the data, "
                    "or tot_base may be wrong.".format(scel_idx)
                )

            # n_fu (formula units) must come from the actual atom counts of the
            # fully-populated reference structure (GCD reduction), not from the
            # SCEL volume field -- using SCEL volume here was the original bug.
            ref_elts = max(elts_dicts, key=lambda ed: ed.get(self.base, 0.0))
            full_comp = Composition(ref_elts)
            _, n_fu = full_comp.get_reduced_composition_and_factor()

            return {"mode": "host_guest", "capacity": capacity, "n_fu": n_fu, "detail": host_dict}

        elif mode == "alloy":
            capacity = info
            self._log("  [capacity] alloy, self-contained total atom count={}".format(capacity))
            return {"mode": "alloy", "capacity": capacity, "n_fu": capacity, "detail": "atoms"}

        else:
            if self.tot_base is not None:
                capacity = self.tot_base * vol if vol is not None else self.tot_base
                self._warn(
                    "scel_idx={}: could not auto-detect host_guest/alloy -> falling back to "
                    "SCEL/tot_base (capacity={})".format(scel_idx, capacity)
                )
                return {"mode": "fallback_scel", "capacity": capacity,
                        "n_fu": vol if vol is not None else capacity, "detail": None}
            raise ValueError(
                "Group scel_idx={} is neither host_guest nor alloy, and no SCEL/tot_base "
                "is available, so concentration cannot be computed. Please specify "
                "tot_base directly.".format(scel_idx)
            )


# -- pure functions (new, no self - imported directly by unit tests) ------------
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lower_convex_hull_indices(points):
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    order = sorted(range(n), key=lambda i: (points[i, 0], points[i, 1]))
    lower = []
    for i in order:
        p = points[i]
        while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], p) <= 0:
            lower.pop()
        lower.append(i)
    return lower


def detect_capacity_mode(elts_dicts, base, count_tol=1e-3):
    # host_guest: an element other than base has a constant count -> host_guest
    # alloy: total atom count is constant -> alloy
    if not elts_dicts:
        return "unknown", None

    all_elements = set()
    for ed in elts_dicts:
        all_elements.update(ed.keys())
    all_elements.discard(base)

    host_dict = {}
    for el in sorted(all_elements):
        counts = [ed.get(el, None) for ed in elts_dicts]
        if any(c is None for c in counts):
            continue
        if max(counts) - min(counts) <= count_tol:
            host_dict[el] = counts[0]

    if host_dict:
        return "host_guest", host_dict

    totals = [sum(ed.values()) for ed in elts_dicts]
    if max(totals) - min(totals) <= count_tol:
        return "alloy", totals[0]

    return "unknown", None


def formation_energy(e_fu, con, con0_energy, con1_energy):
    return e_fu - con * con1_energy - (1.0 - con) * con0_energy


def pick_endpoint_energy(cons, energies, target, tol):
    candidates = [(c, e) for c, e in zip(cons, energies) if abs(c - target) <= tol]
    if not candidates:
        return None, []
    best = min(candidates, key=lambda ce: ce[1])
    return best[1], candidates


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) >= 2:
        # non-interactive (ex: python CCpyCASMhull.py Li 9 [other_dir])
        base = argv[0]
        tot_base = None if argv[1].lower() == "auto" else float(argv[1])
        other_dir = argv[2] if len(argv) >= 3 else None
    else:
        # -- Get info ------------
        other_dir = argv[0] if len(argv) == 1 else None
        base = input("\n* Element name which be changed (ex: Li) : ")  # base atom
        tot_base_in = input(
            "* Number of " + base + " when full (ex: 9) in the primitive cell "
            "(leave blank to auto-detect): ")  # the number of base atoms in unit cell
        tot_base = float(tot_base_in) if tot_base_in.strip() else None

    if other_dir:
        there = os.path.basename(other_dir.rstrip("/")) or other_dir
        if there.startswith("AB"):
            here = "AA" + there[2:]
        else:
            here = os.path.basename(os.getcwd().rstrip(os.sep)) or "."
        structure_sets = [(here, "."), (there, other_dir)]
    else:
        structure_sets = None

    ch = CASMhull(base=base, tot_base=tot_base, structure_sets=structure_sets)
    if base == "Li":
        get_chempot = input("Chemical potential of Li, just enter to use the default value (-1.886)\n: ")
        if len(get_chempot) == 0:
            chempot = -1.886
        else:
            chempot = float(get_chempot)
        ch.mainFlow(chempot=chempot)
    else:
        ch.mainFlow()
