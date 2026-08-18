#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCpySIESTABandSubmit.py
========================

Queue-submit script for the SIESTA Band / FatBand / DOS workflow
(see siesta_band_workflow.py, which must sit in the same directory or be
found via --script-dir).

This is a companion to CCpyJobSubmit.py / CCpyJobControl.py from the CCpy
lab framework: it reuses the same ~/.CCpy/queue_config.yaml and
$CCpy_SCHEDULER_CONFIG partition/node handling (via subclassing
CCpy.Queue.CCpyJobControl.JobSubmit), but is dedicated to this one
pipeline instead of the general software menu (Gaussian/VASP/ATK/...).

Usage:
    CCpySIESTABandSubmit.py <mode> <queue> [suboptions]

Modes (what gets computed - see MODES below):
    1 : full workflow  - genfdf + band-calc + fatband + dos-calc + all plots
    2 : band only       - genfdf + band-calc + plot-band                   (skip fatband/DOS)
    3 : dos only         - genfdf + dos-calc + plot-dos                     (skip band/fatband)
    4 : fatband          - genfdf + band-calc + fatband + plot-fatband       (skip plain band plot/DOS)

Mode 4 is self-contained: it submits its own SIESTA run (band-calc), so it
works from scratch and is also how you redo fatbands after changing
-ef-window=. If a valid band-calc for this system already exists and you
only want to redo the fat/eigfat2plot post-processing (e.g. new -moiety=
or -fat-cmap=) without recomputing, override with
-steps=fatband,plot-fatband.

Modes double as "just redo this part": rerunning the same mode after
changing e.g. -dos-emin=/-dos-emax= (mode 3) or -ef-window= (mode 2/4) redoes
only that piece, and re-submitting the same mode after a failed job just
retries it. genfdf is idempotent (it always regenerates the fdf from the
original base input + current settings), so re-running it is safe.
For a pure re-plot with no recomputation (e.g. just changing -plot-emin=/
-plot-emax=), override with -steps=plot-band or -steps=plot-dos directly.

Example:
    CCpySIESTABandSubmit.py 1 I5 -n=24 \\
        -car=CAR_POSCAR/SWNT7-6W.car -moiety=Tube:1-508 -moiety=Ads:700-786 \\
        -ef-window=15 -bandpath=1d -axis=c

    CCpySIESTABandSubmit.py 3 I5 -n=24 -config=siesta_band_config.yaml -a -dos-emin=-10 -dos-emax=10

    CCpySIESTABandSubmit.py 4 I5 -n=24 -a -ef-window=20 -moiety=Tube:1-508 -moiety=Ads:700-786

Selecting which system directories to submit:
    - No suboption: interactive numbered prompt (comma / range, e.g. 1-3,5).
    - -a            : select all found systems, no prompt.
    - -systems=1-3,5: select specific numbers non-interactively (same numbering
                      as the interactive prompt would show), no prompt.

Batch submission (-batch):
    By default every selected system is submitted as its OWN SLURM job. Add
    -batch to instead submit ALL selected systems as a SINGLE job that runs
    them one after another in the same allocation (like CCpyJobControl's
    vasp_batch) - useful for many small/quick systems where per-job queue
    overhead matters more than wall-clock time. Each system still gets its
    own genfdf/band_config.yaml settings; only -n=/-node=/-time= are shared
    across the whole batch (they describe the one job, not any single
    system). Use -jobname=NAME to name the batch job, or leave it out to be
    prompted for one interactively (default if left blank: SiestaBand_batch<N>).
    A failure in one system does not stop the rest.

    ex) CCpySIESTABandSubmit.py 1 I5 -batch -systems=1-3,7 -n=24 -time=3-00:00:00

Interactive settings review (before submitting):
    Unless -a is given (or stdin isn't a real terminal, e.g. under cron), you
    get a preview box of the settings about to be submitted and a prompt:
      "Anything want to modify? (ex: ef_window=20,dos_emin=-10) else, enter
      'n' to finish" - it keeps re-showing the box and asking again after
      every change (blank input just asks again) until you type "n". This
      mirrors CCpySIESTAInputGen.py's review loop. Any change here is applied
      on top of every selected system (not just the first one).

Fatband moiety groups (-moiety=NAME:IDXSEL):
    Atom indices for each moiety are entered manually, e.g.
    -moiety=Tube:1-508 -moiety=Ads:509-511. To look up the indices, run
    `python siesta_band_workflow.py fatband --list-atoms` directly inside
    Band-DOS/BANDS (after a band-calc has completed). (Automatic bond-based
    detection was tried and dropped: chemisorbed adsorbates often sit close
    enough to the surface to end up in the same bonded fragment as the
    framework, making auto-detection unreliable.)

Per-system settings (car file, moiety atom indices, ef-window, DOS/plot
ranges, ...) can be supplied either as suboptions on the command line, or
via YAML. Without -config=PATH, two YAML layers are checked in order:
  1) the shared, lab-wide DEFAULT_SHARED_CONFIG (CCpy/SIESTA/siesta_band_config.yaml
     next to this script) - applied to every system as a common base.
  2) a per-system `siesta_band_config.yaml` inside that system's own
     directory, if present - overrides just the keys it sets, on top of (1).
-config=PATH, if given, replaces both of the above entirely.
CLI suboptions always win over any YAML value.
`steps` itself is controlled by the mode argument, not by the YAML file.

Run with no arguments for the full suboption list.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from CCpy.Queue.CCpyJobControl import JobSubmit as JS

if sys.version[0] == '3':
    raw_input = input


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# -----------------------------------------------------------------------------
# Mode -> default --steps preset (see siesta_band_workflow.py PIPELINE_STEPS
# for the full ordered list: genfdf, band-calc, fatband, plot-band,
# plot-fatband, dos-calc, plot-dos).
# -----------------------------------------------------------------------------
MODES = {
    "1": dict(steps="all",
              desc="full workflow: genfdf + band-calc + fatband + dos-calc + all plots"),
    "2": dict(steps="genfdf,band-calc,plot-band",
              desc="band only: genfdf + band-calc + plot-band (skip fatband/DOS)"),
    "3": dict(steps="genfdf,dos-calc,plot-dos",
              desc="dos only: genfdf + dos-calc + plot-dos (skip band/DOS)"),
    "4": dict(steps="genfdf,band-calc,fatband,plot-fatband",
              desc="fatband: genfdf + band-calc + fatband + plot-fatband (self-contained, "
                   "skip plain band plot/DOS; submits its own SIESTA run - use -steps="
                   "fatband,plot-fatband to skip the recompute if a band-calc already exists)"),
}

# SLURM job name for single (non-batch) submissions: "S" + system directory
# name + a mode suffix, e.g. mode 1 on ./Ti2CTx_O_NT_2 -> "STi2CTx_O_NT_2",
# mode 2 -> "STi2CTx_O_NT_2_Band".
MODE_JOBNAME_SUFFIX = {"1": "", "2": "_Band", "3": "_DOS", "4": "_FatBand"}


def mode_jobname(mode: str, dirname: str) -> str:
    return f"S{dirname}{MODE_JOBNAME_SUFFIX.get(mode, '')}"


# Where the mpi.sh.o/e SLURM logs should end up for each mode: mode 1 runs
# both band+dos so there's no single obvious subfolder - keep it at
# Band-DOS/ itself. Modes 2/4 only touch BANDS (band-calc / fatband), mode
# 3 only touches DOS, so their logs go one level deeper alongside the
# files they actually describe.
MODE_LOG_SUBDIR = {"1": "Band-DOS", "2": "Band-DOS/BANDS", "3": "Band-DOS/DOS", "4": "Band-DOS/BANDS"}


def mode_log_subdir(mode: str) -> str:
    return MODE_LOG_SUBDIR.get(mode, "Band-DOS")


def submit_and_show_jobid(qsub_cmd: str) -> None:
    """Run the qsub/sbatch command and print just the numeric job ID (e.g.
    '226444'), instead of the scheduler's full 'Submitted batch job ...'
    text. Falls back to the raw output if no number is found in it."""
    result = subprocess.run(qsub_cmd, shell=True, capture_output=True, text=True)
    out = ((result.stdout or "") + (result.stderr or "")).strip()
    m = re.search(r"\d+", out)
    print(m.group(0) if m else out)


# Shared, lab-wide siesta_band_config.yaml - used whenever a system directory
# doesn't have its own siesta_band_config.yaml and -config=PATH wasn't given.
# Lets everyone default to one common set of settings instead of copying the
# yaml into every system directory by hand.
#
# Resolved next to this module, so it follows wherever CCpy is installed. The
# old value was a literal path pinning python3.8 and a .egg layout; under any
# other interpreter it simply pointed at a file that does not exist, and
# _apply_yaml() skips missing files silently - so the shared defaults were
# dropped without any warning.
DEFAULT_SHARED_CONFIG = Path(__file__).resolve().parent / "siesta_band_config.yaml"

# -----------------------------------------------------------------------------
# Default pipeline settings ("steps" itself comes from the mode argument, see
# MODES above - it is intentionally not settable via siesta_band_config.yaml).
# Overridden (in order) by: mode  <-  DEFAULT_SHARED_CONFIG  <-  a per-system
# siesta_band_config.yaml in the system dir (or -config=PATH)  <-  CLI
# suboptions.
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    car=None,
    moiety=[],
    element=True,
    bandpath="1d",
    axis=None,  # None = let siesta_band_workflow.py auto-detect from kgrid_Monkhorst_Pack
    ef_window=15,
    extra_bands=50,
    dos_emin=-25.0, dos_emax=25.0, dos_broad=0.05, dos_npts=2000,
    plot_emin=-2.0, plot_emax=2.0,
    lw_band=1.0,
    fat_cmap="inferno",
    bin_dir="/opt/siesta/siesta-5.4.2/siesta-mkl-mpi/bin",
    time="7-00:00:00",
)

# Types for the interactive settings-review loop (maybe_review_settings) and
# for coercing '-key=value' suboptions/YAML values consistently. "moiety" is
# a list, edited as a single ';'-separated string (e.g. "Tube:1-508;Ads:700-786").
EDITABLE_TYPES = {
    "car": str, "bandpath": str, "axis": str, "fat_cmap": str, "bin_dir": str,
    "time": str, "siesta_bin": str, "steps": str,
    "ef_window": int, "extra_bands": int, "dos_npts": int,
    "dos_emin": float, "dos_emax": float, "dos_broad": float,
    "plot_emin": float, "plot_emax": float, "lw_band": float,
    "element": bool,
    "moiety": list,
}


def _coerce(key: str, raw: str):
    t = EDITABLE_TYPES.get(key)
    if t is None:
        return raw
    if t is bool:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if t is list:
        return [x.strip() for x in raw.split(";") if x.strip()]
    if raw.strip().lower() in ("none", "null", "") and key in ("car", "siesta_bin"):
        return None
    return t(raw)


# -----------------------------------------------------------------------------
# Suboption parsing: '-key=value' / '--key=value', CCpyJobSubmit.py style.
# Repeatable keys (e.g. -moiety=) collect into a list. Bare flags (e.g. -a)
# go into `flags`.
# -----------------------------------------------------------------------------
def parse_dash_kv_args(argv):
    kv = {}
    flags = set()
    for tok in argv:
        m = re.match(r"^-{1,2}([A-Za-z][\w-]*)=(.*)$", tok)
        if m:
            key = m.group(1).replace("-", "_")
            val = m.group(2)
            if key in kv:
                if isinstance(kv[key], list):
                    kv[key].append(val)
                else:
                    kv[key] = [kv[key], val]
            else:
                kv[key] = val
        elif re.match(r"^-{1,2}[A-Za-z][\w-]*$", tok):
            flags.add(tok.lstrip("-").replace("-", "_"))
    return kv, flags


def find_siesta_systems(root="."):
    """
    Find candidate SIESTA system directories: any directory containing a
    base '<label>.fdf' (before Band-DOS/ is generated). Includes the
    current directory itself if it qualifies.
    """
    root = Path(root)
    systems = []

    if list(root.glob("*.fdf")):
        systems.append(root)

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in ("Band-DOS", ".git", "__pycache__"):
            continue
        if list(d.glob("*.fdf")):
            systems.append(d)

    return systems


def parse_index_selection(sel: str, n: int):
    """'1-3,5' -> [1,2,3,5] (1-based). '0' -> all indices 1..n."""
    sel = sel.strip()
    if sel == "0":
        return list(range(1, n + 1))
    idxs = []
    for tok in sel.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            idxs.extend(range(int(a), int(b) + 1))
        else:
            idxs.append(int(tok))
    return idxs


def select_systems(systems, ask=True, preselect=None):
    """
    preselect: optional '-systems=' string (same '1-3,5' syntax as the
    interactive prompt) to pick systems non-interactively without -a
    (which would otherwise select everything).
    """
    if not systems:
        print("No SIESTA systems (*.fdf) found under this directory.")
        quit()

    if preselect is not None:
        idxs = parse_index_selection(preselect, len(systems))
        return [systems[i - 1] for i in idxs]

    if not ask:
        return systems

    for i, d in enumerate(systems):
        print(f"{i + 1} : {d}")
    print("0 : All directories")
    while True:
        sel = raw_input("choose file : ").strip()
        if not sel:
            continue
        try:
            idxs = parse_index_selection(sel, len(systems))
            return [systems[i - 1] for i in idxs]
        except (ValueError, IndexError) as exc:
            print(f"  invalid selection ({exc}), try again")


def detect_label(system_dir: Path) -> str:
    """Parse SystemLabel from the base .fdf; fall back to the .fdf stem."""
    fdfs = sorted(system_dir.glob("*.fdf"))
    for f in fdfs:
        try:
            for line in f.read_text(errors="ignore").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                m = re.match(r"(?i)^\s*SystemLabel\s+(\S+)", s)
                if m:
                    return m.group(1)
        except Exception:
            continue
    if fdfs:
        return fdfs[0].stem
    return system_dir.name


def load_system_config(system_dir: Path, mode: str, config_path=None):
    cfg = dict(DEFAULTS)
    cfg["moiety"] = list(DEFAULTS["moiety"])
    cfg["steps"] = MODES[mode]["steps"]

    def _apply_yaml(yml_path: Path) -> None:
        if not yml_path.exists():
            return
        with open(yml_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        if "steps" in user_cfg:
            print(f"[warn] 'steps' in {yml_path} is ignored - steps come from the mode "
                  f"argument (currently mode {mode}: {MODES[mode]['desc']}). "
                  f"Use -steps=... on the command line to override.")
            user_cfg.pop("steps")
        cfg.update(user_cfg)
        print(f"[info] loaded config: {yml_path}")

    if config_path:
        # Explicit -config=PATH takes over the whole lookup chain - only this.
        _apply_yaml(Path(config_path))
    else:
        # Shared lab-wide config first (base layer for everyone), then a
        # per-system siesta_band_config.yaml on top of it if the system
        # directory has one - per-system keys override the shared ones,
        # anything not set there keeps the shared value.
        _apply_yaml(DEFAULT_SHARED_CONFIG)
        _apply_yaml(system_dir / "siesta_band_config.yaml")

    return cfg


class SiestaBandJobSubmit(JS):
    """
    Extends CCpy's JobSubmit with the SIESTA Band/FatBand/DOS pipeline.
    __init__ is inherited unchanged: it loads ~/.CCpy/queue_config.yaml and
    sets self.partition_name / self.pe_request / self.node_assign /
    self.allot_node / self.qsub / self.python_path / self.mpi_run /
    self.siesta_path from it (see CCpyJobControl.py).
    """

    def __init__(self, inputfile, queue, n_of_cpu, node=None, script_dir=None):
        super().__init__(inputfile, queue, n_of_cpu, node=node)
        self.script_dir = script_dir or str(Path(__file__).resolve().parent)

    def _build_pipeline_cmd(self, label: str, cfg: dict) -> str:
        """Build the 'python siesta_band_workflow.py pipeline ...' command line for one system."""
        pyscript = str(Path(self.script_dir) / "siesta_band_workflow.py")

        moiety = cfg.get("moiety") or []
        if isinstance(moiety, str):
            moiety = [moiety]
        moiety_flags = " ".join(f'--moiety "{m}"' for m in moiety)

        siesta_bin = cfg.get("siesta_bin") or f'{cfg["bin_dir"]}/siesta'

        pipeline_argv = [
            # -u: unbuffered stdout/stderr. Without this, python fully-buffers
            # its own print() output when stdout is redirected to a file (as
            # SLURM does for mpi.sh.o<jobid>), so [info]/[cmd]/[ERROR] lines
            # can appear late, out of order, or missing entirely if the job
            # is killed before the buffer flushes - very confusing to debug.
            f'{self.python_path} -u {pyscript} pipeline',
            f'--label {label}',
            f'--steps {cfg["steps"]}',
            f'--bandpath {cfg["bandpath"]}',
            f'--ef-window {cfg["ef_window"]}',
            f'--extra-bands {cfg["extra_bands"]}',
            f'--dos-emin {cfg["dos_emin"]} --dos-emax {cfg["dos_emax"]}',
            f'--dos-broad {cfg["dos_broad"]} --dos-npts {cfg["dos_npts"]}',
            f'--plot-emin {cfg["plot_emin"]} --plot-emax {cfg["plot_emax"]}',
            f'--lw-band {cfg["lw_band"]}',
            f'--fat-cmap {cfg["fat_cmap"]}',
            f'--bin-dir {cfg["bin_dir"]}',
            f'--siesta-bin {siesta_bin}',
            f'--mpi-run "{self.mpi_run}"',
        ]
        pipeline_argv.append('--element' if cfg.get("element", True) else '--no-element')
        if cfg.get("axis"):
            pipeline_argv.append(f'--axis {cfg["axis"]}')
        if cfg.get("car"):
            pipeline_argv.append(f'--carfile {cfg["car"]}')
        if moiety_flags:
            pipeline_argv.append(moiety_flags)

        return " \\\n    ".join(pipeline_argv)

    def siesta_band_fatband_dos(self, workdir: Path, cfg: dict, jobname: Optional[str] = None,
                                 log_subdir: str = "Band-DOS"):
        """Submit a single system as its own SLURM job."""
        label = self.inputfile
        jobname = jobname or f"{label}_BandFatDOS"
        pipeline_cmd = self._build_pipeline_cmd(label, cfg)

        mpi = f'''#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -p {self.partition_name}
{self.allot_node}
{self.node_assign}
{self.pe_request}
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time={cfg["time"]}

echo "===== START: $(date) ====="
echo "WORKDIR: $(pwd)"

# Best-effort: activate the siesta conda env if conda is available in this
# (non-interactive) shell - helps the siesta/fat/gnubands binaries find their
# runtime libs. Not a hard requirement: siesta_band_workflow.py auto-detects
# and relaunches itself under the right python for numpy/matplotlib/sisl
# regardless of what's active here, so we never hard-fail the job over this.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate siesta 2>/dev/null
fi

cd "{workdir}"

{pipeline_cmd}

rc=$?
echo "siesta_band_workflow.py exit code: $rc"
echo "===== END: $(date) ====="

# Move the SLURM stdout/stderr logs into {log_subdir}/ now that it's
# guaranteed to exist (genfdf, run above via the pipeline, creates it) -
# moving an already-open log file is safe on Linux (rename() doesn't break
# the open fd, so this line itself still lands correctly). Kept at
# top-level instead of pre-creating the target dir before submission,
# since that made -o/-e brittle (SLURM refuses to redirect into a
# directory that doesn't exist yet at submit time). Falls back to
# Band-DOS/ (and then just leaves the logs at top-level) if the more
# specific mode subfolder isn't there for some reason.
# Use $SLURM_JOB_NAME (not the {jobname} we asked for via -J) to build the
# filename: on this cluster %x has been observed to fall back to the
# submitted script's own name ("mpi.sh") instead of the -J value, so the
# actual log files were "mpi.sh.o<jobid>"/"mpi.sh.e<jobid>", not
# "{jobname}.o<jobid>". $SLURM_JOB_NAME always reflects whatever %x really
# expanded to, so this matches the real file regardless of why -J didn't
# take effect.
if [ -d "{log_subdir}" ]; then
    target="{log_subdir}"
elif [ -d "Band-DOS" ]; then
    target="Band-DOS"
else
    target=""
fi
if [ -n "$target" ]; then
    for f in "${{SLURM_JOB_NAME}}.o${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}.e${{SLURM_JOB_ID}}" \\
             "mpi.sh.o${{SLURM_JOB_ID}}" "mpi.sh.e${{SLURM_JOB_ID}}"; do
        [ -f "$f" ] && mv -f "$f" "$target/" 2>/dev/null
    done
fi

exit $rc
'''
        pwd = os.getcwd()
        os.chdir(workdir)
        Path("mpi.sh").write_text(mpi)
        submit_and_show_jobid(self.qsub + " mpi.sh")
        os.chdir(pwd)

    def siesta_band_fatband_dos_batch(self, jobs, jobname: str, time: str):
        """
        Submit several systems as ONE SLURM job, run sequentially in the
        same allocation (like CCpyJobControl.vasp_batch). `jobs` is a list
        of (workdir: Path, label: str, cfg: dict). A failure in one system
        does not stop the rest (no `set -e`); each system's own exit status
        is echoed so it's easy to grep the log afterwards.
        """
        blocks = []
        for workdir, label, cfg in jobs:
            pipeline_cmd = self._build_pipeline_cmd(label, cfg)
            blocks.append(
                f'echo "--- [{label}] {workdir} ---"\n'
                f'cd "{workdir}"\n'
                f'{pipeline_cmd}\n'
                f'echo "--- [{label}] exit code: $? ---"\n'
            )
        body = "\n".join(blocks)

        mpi = f'''#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -p {self.partition_name}
{self.allot_node}
{self.node_assign}
{self.pe_request}
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time={time}

echo "===== START: $(date) ====="

# Best-effort: activate the siesta conda env if conda is available in this
# (non-interactive) shell - see comment in siesta_band_fatband_dos(). Never
# hard-fails the job over this; siesta_band_workflow.py's own bootstrap
# re-exec covers the python-package side regardless.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate siesta 2>/dev/null
fi

{body}

echo "===== END: $(date) ====="
'''
        Path("mpi.sh").write_text(mpi)
        submit_and_show_jobid(self.qsub + " mpi.sh")


def print_help_and_quit():
    modes_txt = "\n".join(f"    {k} : {v['desc']}" for k, v in MODES.items())
    print(f"""
--------------------------------------------------------------------------------------
    How to use: CCpySIESTABandSubmit.py [Mode] [I5/opa] [Suboptions]
--------------------------------------------------------------------------------------
< Mode >
{modes_txt}

    ex) CCpySIESTABandSubmit.py 1 I5 -n=24 -car=CAR_POSCAR/SWNT7-6W.car \\
            -moiety=Tube:1-508 -moiety=Ads:700-786 -ef-window=15
    ex) CCpySIESTABandSubmit.py 3 I5 -n=24 -config=siesta_band_config.yaml -a -dos-emin=-10 -dos-emax=10
    ex) CCpySIESTABandSubmit.py 4 I5 -n=24 -a -ef-window=20 -moiety=Tube:1-508 -moiety=Ads:700-786
            (mode 4 submits its own band-calc - works from scratch, or to redo fatbands
             with a different -ef-window=. To skip recompute and only redo the
             fat/eigfat2plot post-processing + plot, add -steps=fatband,plot-fatband)

    Re-running the same mode is the normal way to: change a setting and redo just
    that part (e.g. -dos-emin=/-dos-emax= with mode 3, -ef-window= with mode 4), or
    resubmit after a failed job. For a pure re-plot with no recomputation, override
    directly, e.g. -steps=plot-dos.

[Suboptions]
    -n=[int]           number of CPUs (default: queue's core count)
    -node=[name]       assign specific node
    -a                 select all found systems without asking
    -systems=[N-M,...] select specific system numbers without asking (same numbering
                       as the interactive prompt)
    -batch             submit all selected systems as ONE shared SLURM job
                       (sequential, like vasp_batch) instead of one job each
    -jobname=[name]    batch job name (only used with -batch; if omitted you'll be
                       prompted for one interactively unless -a is given;
                       default if left blank: SiestaBand_batch<N>)
    -config=[path]     YAML config (default: shared CCpy/SIESTA/siesta_band_config.yaml,
                       then a per-system siesta_band_config.yaml layered on top if
                       present in that system's own dir; -config= replaces both)
    -steps=[list]      override the mode's default step list, e.g. -steps=plot-dos
                       (full ordered list: genfdf,band-calc,fatband,
                                            plot-band,plot-fatband,dos-calc,plot-dos)
    -car=[path]        Materials Studio .car file (optional - only needed for
                       -bandpath=seekpath with seekpath's symmetry detection;
                       1d/hex/manual/plain-seekpath-fallback all derive the
                       lattice directly from the system's own fdf instead)
    -moiety=[N:SEL]    moiety NAME:INDEX-SELECTION (1-based), repeatable - look up
                       indices with `siesta_band_workflow.py fatband --list-atoms`
                       ex) -moiety=Tube:1-508 -moiety=Ads:700-786
    -bandpath=[..]     seekpath | 1d | hex | manual        (default: 1d)
    -axis=[a|b|c]      1d bandpath axis  (default: auto-detected from the base fdf's
                       kgrid_Monkhorst_Pack - the axis with >1 k-point; falls back to
                       'c' only if that block is missing/ambiguous - override if wrong,
                       e.g. after symmetrization moves the tube axis to b)
    -ef-window=[int]   bands window around EF for fatbands  (default: 15)
    -dos-emin=/-dos-emax=/-dos-broad=/-dos-npts=  DOS window (default: -25 25 0.05 2000)
    -plot-emin=/-plot-emax=                       plot energy window (default: -2 2)
    -lw-band=[float]   band linewidth                       (default: 1.0)
    -fat-cmap=[name]   fatband colormap                     (default: inferno)
    -bin-dir=[path]    dir with siesta/fat/eigfat2plot/gnubands
                       (default: /opt/siesta/siesta-5.4.2/siesta-mkl-mpi/bin)
    -time=[D-HH:MM:SS] SLURM walltime                       (default: 7-00:00:00)

    Suboptions override siesta_band_config.yaml, which overrides the built-in defaults.
--------------------------------------------------------------------------------------
""")
    home = os.getenv("HOME")
    print(bcolors.OKGREEN + f"    *** Queue config file: {home}/.CCpy/queue_config.yaml ***" + bcolors.ENDC)
    print("""    - Modify software version / binary paths there (siesta_path, python_path, mpi_run, qsub, ...).
    - This file is created automatically the first time any CCpy*.py script runs.
""")
    quit()


def apply_cli_overrides(cfg: dict, kv: dict, flags: set) -> dict:
    """CLI suboptions win over whatever came from mode/YAML."""
    for key in ("steps", "car", "bandpath", "axis", "bin_dir", "time", "fat_cmap", "siesta_bin"):
        if key in kv:
            cfg[key] = kv[key]
    for key in ("ef_window", "extra_bands", "dos_npts"):
        if key in kv:
            cfg[key] = int(kv[key])
    for key in ("dos_emin", "dos_emax", "dos_broad", "plot_emin", "plot_emax", "lw_band"):
        if key in kv:
            cfg[key] = float(kv[key])
    if "moiety" in kv:
        cfg["moiety"] = kv["moiety"] if isinstance(kv["moiety"], list) else [kv["moiety"]]
    if "element" in flags:
        cfg["element"] = True
    if "no_element" in flags:
        cfg["element"] = False
    return cfg


# -----------------------------------------------------------------------------
# Interactive settings-review loop, before submitting - same UX as
# test_CCpySIESTAInputGen.py's "Anything want to modify or add?" prompt:
# keep re-showing the current settings and asking again after every change,
# until "n" is typed (blank input just asks again, it does not finish).
# Only runs when stdin is a real terminal and -a was not given, so
# batch/Slurm/cron invocations never block on input().
# -----------------------------------------------------------------------------
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_COLOR_OK = sys.stdout.isatty()


def _colorize(text: str) -> str:
    return f"{_ANSI_YELLOW}{text}{_ANSI_RESET}" if _COLOR_OK else text


def format_settings_preview(cfg: dict, mode: str, highlight: set = None) -> str:
    highlight = highlight or set()

    def row(label: str, key: str) -> str:
        text = f"{label:<16s}: {cfg.get(key)}"
        return _colorize(text) if key in highlight else text

    lines = ["", "# " + "-" * 56 + " #", "#           Band / FatBand / DOS settings              #", "# " + "-" * 56 + " #"]
    lines.append(f"{'mode':<16s}: {mode} ({MODES[mode]['desc']})")
    lines.append(f"{'steps':<16s}: {cfg.get('steps')}")
    lines.append(row("car", "car"))
    lines.append(row("bandpath", "bandpath"))
    axis_text = f"{'axis':<16s}: {cfg.get('axis') or 'auto (from kgrid)'}"
    lines.append(_colorize(axis_text) if "axis" in highlight else axis_text)
    lines.append(row("ef_window", "ef_window"))
    lines.append(row("extra_bands", "extra_bands"))
    lines.append(row("element", "element"))
    lines.append(row("moiety", "moiety"))
    lines.append(row("dos_emin", "dos_emin"))
    lines.append(row("dos_emax", "dos_emax"))
    lines.append(row("dos_broad", "dos_broad"))
    lines.append(row("dos_npts", "dos_npts"))
    lines.append(row("plot_emin", "plot_emin"))
    lines.append(row("plot_emax", "plot_emax"))
    lines.append(row("lw_band", "lw_band"))
    lines.append(row("fat_cmap", "fat_cmap"))
    lines.append(row("bin_dir", "bin_dir"))
    lines.append(row("time", "time"))
    lines.append("# " + "-" * 56 + " #")
    return "\n".join(lines)


def maybe_review_settings(cfg: dict, mode: str, ask: bool) -> dict:
    """
    Preview the settings that are about to be submitted and let the user
    tweak them (KEY=VALUE, comma-separated, repeatable) until "n" is typed.
    Returns only the keys actually changed, to be layered on top of every
    selected system's own config - so per-system car/moiety stay untouched
    unless explicitly edited here.
    """
    if not ask or not sys.stdin.isatty():
        return {}

    working = dict(cfg)
    changed: dict = {}
    while True:
        print(format_settings_preview(working, mode, highlight=set(changed)))
        try:
            raw = input(
                'Anything want to modify? (ex: ef_window=20,dos_emin=-10,dos_emax=10,'
                'plot_emin=-1,plot_emax=1) else, enter "n" to finish\n'
                '(moiety uses ";" between entries, e.g. moiety=Tube:1-508;Ads:700-786)\n: '
            ).strip()
        except EOFError:
            break

        if raw.lower() == "n":
            break
        if not raw:
            continue  # nothing typed - ask again, only "n" finishes

        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                print(f"  (ignored: expected KEY=VALUE, got {chunk!r})")
                continue
            key, val = chunk.split("=", 1)
            key = key.strip().replace("-", "_")
            val = val.strip()
            if key not in EDITABLE_TYPES:
                print(f"  (ignored: unknown setting {key!r}; known: {', '.join(sorted(EDITABLE_TYPES))})")
                continue
            try:
                working[key] = _coerce(key, val)
                changed[key] = working[key]
            except Exception as exc:
                print(f"  (ignored: {key}={val!r}: {exc})")

    return changed


def main():
    # -- initiate ~/.CCpy/queue_config.yaml if missing (same as CCpyJobSubmit.py)
    JS(None, None, None, init_only=True)

    if len(sys.argv) < 3:
        print_help_and_quit()

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"Mode '{mode}' not in {list(MODES.keys())}\n")
        print_help_and_quit()

    try:
        CCpy_SCHEDULER_CONFIG = os.environ['CCpy_SCHEDULER_CONFIG']
    except KeyError:
        print('''Error while loading $CCpy_SCHEDULER_CONFIG file.
Please check the example scheduler config file at https://github.com/91bsjun/CCpy/tree/master/CCpy/Queue''')
        quit()

    queue_info = yaml.safe_load(open(CCpy_SCHEDULER_CONFIG, 'r'))
    queues = list(queue_info.keys())

    queue = sys.argv[2]
    if queue not in queues:
        print(f"{queue} not in {queues}")
        quit()

    kv, flags = parse_dash_kv_args(sys.argv[3:])

    n_of_cpu = int(kv["n"]) if "n" in kv else None
    node = kv.get("node")
    ask = "a" not in flags
    config_path = kv.get("config")
    batch = "batch" in flags

    systems = find_siesta_systems(".")
    systems = select_systems(systems, ask=ask, preselect=kv.get("systems"))

    print(f"[info] mode {mode}: {MODES[mode]['desc']}")
    print(f"[info] {len(systems)} system(s) selected"
          + (" -> submitting as ONE batch job" if batch else " -> submitting one job per system"))

    jobs = []
    for system_dir in systems:
        label = detect_label(system_dir)
        cfg = apply_cli_overrides(load_system_config(system_dir, mode, config_path=config_path), kv, flags)
        jobs.append([system_dir.resolve(), label, cfg])

    if not jobs:
        print("Nothing to submit.")
        quit()

    # -- interactive settings review (skipped if -a was given, or stdin isn't a tty)
    review_overrides = maybe_review_settings(jobs[0][2], mode, ask)
    if review_overrides:
        print(f"[info] applying reviewed change(s) to all {len(jobs)} selected system(s): {review_overrides}")
        for _, _, cfg in jobs:
            cfg.update(review_overrides)

    if batch:
        for workdir, label, cfg in jobs:
            print(f"  - {workdir.name}")

        default_jobname = f"SiestaBand_batch{len(jobs)}"
        jobname = kv.get("jobname")
        if not jobname:
            if ask and sys.stdin.isatty():
                entered = raw_input(f"Jobname for this job \n: ").strip()
                jobname = entered or default_jobname
            else:
                jobname = default_jobname
        batch_time = kv.get("time") or DEFAULTS["time"]
        myJS = SiestaBandJobSubmit(jobname, queue, n_of_cpu, node=node)
        myJS.siesta_band_fatband_dos_batch([(w, l, c) for w, l, c in jobs], jobname=jobname, time=batch_time)
    else:
        for workdir, label, cfg in jobs:
            jobname = mode_jobname(mode, workdir.name)
            myJS = SiestaBandJobSubmit(label, queue, n_of_cpu, node=node)
            myJS.siesta_band_fatband_dos(workdir, cfg, jobname=jobname, log_subdir=mode_log_subdir(mode))


if __name__ == "__main__":
    main()
