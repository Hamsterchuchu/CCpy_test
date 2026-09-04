# -*- coding: utf-8 -*-
"""Driver module for CASM's external binary (mainclust).

mainclust only works interactively, and which questions come up depends on
which files are already in the working directory. For example, on a first
run there's no eci.in yet, so that question never appears at all, and
answering the supercells question with read(1) skips the whole max
volume / dimension question that would otherwise follow.

So "push answers in a fixed order" is dangerous. If the file state differs
from what was expected by even a little, the answers get shifted, the wrong
value lands on the wrong question, and mainclust accepts it silently. This
module instead **recognizes the question text on screen with a regex and
supplies the answer that matches that question.**

mainclust block-buffers stdout when it's a pipe, so questions don't show up
in time. A pseudo-terminal (pty) is attached instead, to get the same
line-buffering as running it interactively.
"""

import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

try:
    import pty
    import termios
    _HAS_PTY = True
except ImportError:                                   # pragma: no cover
    _HAS_PTY = False


#: Executable name. By convention it used to be copied into every system's
#: folder, but now that resolve_binary() searches several locations in
#: order, one copy in one place is enough.
DEFAULT_BINARY = "mainclust"

#: Environment variable for pointing directly at the executable path (same
#: style as CCpy_SERVER).
BINARY_ENV = "CCpy_MAINCLUST"

#: Filename prefix for the per-element POTCARs mainclust concatenates.
POTCAR_PREFIX = "POTCAR_"

#: Templates mainclust copies into each configuration's directory. Without
#: them it exits normally without creating any directories at all (see
#: check_templates below).
REQUIRED_TEMPLATES = ("INCAR", "KPOINTS")

#: Question key -> (regex, human-readable description). Text extracted from
#: the mainclust binary. The order here is the order they actually appear
#: in, but matching itself is order-independent.
PROMPT_SPECS = (
    ("eci",
     r"generate\s*\(0\)\s*or\s*keep\s*\(1\)\s*eci\.in",
     "whether to regenerate eci.in (0) or keep it (1)"),
    ("supercells",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*supercells",
     "whether to generate supercells (0) or read them (1)"),
    ("max_volume",
     r"enter the maximum supercell volume",
     "maximum supercell volume"),
    ("dimension",
     r"generate 3-dimensional or 2-dimensional supercells",
     "whether to generate 3-dimensional (3) or 2-dimensional (2) supercells"),
    ("configuration",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*configuration and configuration\.corr",
     "whether to generate configuration (0) or read it (1)"),
    ("make_dirs",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*make_dirs",
     "whether to generate make_dirs (0) or read it (1)"),
    ("energy",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*energy and corr\.in",
     "whether to generate energy / corr.in (0) or read them (1)"),
    ("reference",
     r"generate\s*\(0\)\s*or\s*keep\s*\(1\)\s*reference",
     "whether to regenerate reference (0) or keep it (1)"),
)

_COMPILED = tuple((key, re.compile(pat), desc) for key, pat, desc in PROMPT_SPECS)

#: Text that is always printed at the end of a normal run.
_DONE_RE = re.compile(r"CASM code has completed successfully")

#: Normal-progress marker. Which checkpoint number was reached tells us
#: where things stopped.
_CHECKPOINT_RE = re.compile(r"Checkpoint\s+(\d+)")

#: Messages that print even on a normal run. Collected here so they aren't
#: mistaken for errors.
BENIGN_MESSAGES = (
    "No SPECIES file in the current directory",
    "No CSPECS file in the current directory",
    "cannot open file custom_structures",
    "No configurations available to determine convex hull",
    "quitting assemble_hull()",
    "No cluster expansion is present",
)


class MainclustError(RuntimeError):
    """An error that occurred while driving mainclust."""

    def __init__(self, message, transcript=None):
        super(MainclustError, self).__init__(message)
        self.transcript = transcript or ""


class MainclustResult(object):
    """The result of one mainclust run."""

    def __init__(self, returncode, transcript, answered, unused, checkpoints, completed):
        self.returncode = returncode
        self.transcript = transcript
        #: [(question key, answer)] -- the questions that actually appeared
        #: and the answers supplied, in the order they appeared.
        self.answered = answered
        #: Question keys that were prepared but never appeared.
        self.unused = unused
        #: List of checkpoint numbers passed.
        self.checkpoints = checkpoints
        #: Whether "CASM code has completed successfully" was seen.
        self.completed = completed

    @property
    def ok(self):
        return self.returncode == 0 and self.completed

    def summary(self):
        seq = " ".join("%s=%s" % (k, v) for k, v in self.answered)
        return ("exit=%s completed=%s checkpoint=%s\n  answered: %s"
                % (self.returncode, self.completed,
                   max(self.checkpoints) if self.checkpoints else "-",
                   seq or "(none)"))

    def __repr__(self):                                # pragma: no cover
        return "<MainclustResult %s>" % self.summary().replace("\n", " ")


# ----------------------------------------------------------------------------
# Standard answer sets
# ----------------------------------------------------------------------------

def answers_enumerate(max_volume=1, dimension=3):
    """A fresh enumeration. Use this after changing PRIM and re-running.

    If a previous result exists it is regenerated entirely (0); if not,
    that question simply doesn't appear.
    """
    return {
        "eci": 0,
        "supercells": 0,
        "max_volume": int(max_volume),
        "dimension": int(dimension),
        "configuration": 0,
        "make_dirs": 0,
        "energy": 0,
        "reference": 0,
    }


def answers_reuse(energy=0, reference=0):
    """Read back an existing enumeration to build VASP input directories.

    Flip make_dirs's make flags to 1 and run again with these answers, and
    the con* directories get created. This is the "1 1 1 1 0 0" from the
    course notes.

    energy / reference use 0 (regenerate) or 1 (read), depending on the system.
    """
    return {
        "eci": 1,
        "supercells": 1,
        "configuration": 1,
        "make_dirs": 1,
        "energy": int(energy),
        "reference": int(reference),
    }


# ----------------------------------------------------------------------------
# Locating the executable
# ----------------------------------------------------------------------------

def _config_dir():
    """Config folder path as a string. Falls back to the conventional path
    if CCpy's config module can't be imported."""
    try:
        from CCpy.Tools.CCpyConfig import config_home
        return str(config_home())
    except Exception:                                  # pragma: no cover
        return os.path.join(os.path.expanduser("~"), ".CCpy_test")


def _executable(path):
    """Absolute path if it's an executable file, None if it doesn't exist.
    Raises if it exists but isn't permitted to execute."""
    if not path:
        return None
    full = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    if not os.path.isfile(full):
        return None
    if not os.access(full, os.X_OK):
        raise MainclustError(
            "%s is not executable. Please chmod +x it." % full)
    return full


def resolve_binary(binary=None, workdir="."):
    """Find the absolute path to the mainclust executable.

    mainclust is a compiled external binary of unclear origin and license,
    so it is not kept in the repository. But copying 674 KB into every
    system's folder means it becomes unclear, as folders pile up, which copy
    is current. So it is searched for in the following order -- once found,
    later locations are not checked.

    1. The path given in ``binary`` (relative to ``workdir`` if not absolute)
    2. The environment variable ``$CCpy_MAINCLUST``
    3. The config folder ``~/.CCpy_test/mainclust`` (movable via ``$CCpy_HOME``)
    4. Inside the working directory (the old convention, kept so folders
       people already used this way don't break)
    5. ``$PATH``

    Only case 1 is an exception: if a path was given directly and it's
    wrong, this fails immediately rather than silently falling back to some
    other binary.
    """
    name = str(binary) if binary else DEFAULT_BINARY
    tried = []

    def _try(desc, path):
        if not path:
            return None
        tried.append((desc, str(path)))
        return _executable(path)

    if binary and (os.sep in name or name.startswith("~") or name.startswith(".")):
        expanded = os.path.expanduser(os.path.expandvars(name))
        base = expanded if os.path.isabs(expanded) else os.path.join(workdir, expanded)
        found = _try("given path", base)
        if found:
            return found
        raise MainclustError("Could not find mainclust at: %s" % tried[-1][1])

    found = _try("environment variable $%s" % BINARY_ENV, os.environ.get(BINARY_ENV))
    if found:
        return found

    found = _try("config folder", os.path.join(_config_dir(), name))
    if found:
        return found

    found = _try("working directory", os.path.join(workdir, name))
    if found:
        return found

    hit = shutil.which(name)
    tried.append(("$PATH", "which %s" % name))
    if hit:
        return os.path.abspath(hit)

    raise MainclustError(
        "Could not find mainclust. Looked in the following places, in order:\n"
        + "\n".join("  %-14s %s" % (d, p) for d, p in tried)
        + "\n\nmainclust is an external binary not included in the repository. "
          "Fetch it once and place it at\n"
          "  %s\n"
          "or set $%s to its path, so it doesn't need to be copied into every "
          "system's folder."
          % (os.path.join(_config_dir(), DEFAULT_BINARY), BINARY_ENV))


# ----------------------------------------------------------------------------
# Checking POTCARs
# ----------------------------------------------------------------------------

def prim_elements(workdir=".", prim="PRIM"):
    """Elements appearing in PRIM, in order. Raises MainclustError if it
    can't be read."""
    path = prim if os.path.isabs(prim) else os.path.join(workdir, prim)
    if not os.path.isfile(path):
        raise MainclustError("%s is missing." % path)
    try:
        from CCpy.CASM.CASMprim import Prim
    except ImportError as err:                         # pragma: no cover
        raise MainclustError("Could not import CASMprim: %s" % err)
    return Prim.read(path).elements


def check_potcar_sources(workdir=".", elements=None, prim="PRIM"):
    """Check that a per-element POTCAR is properly present, before creating con*.

    mainclust concatenates ``POTCAR_<element>`` to build each configuration's
    POTCAR, and if that file is missing it **does not raise an error -- it
    silently makes a 0-byte POTCAR and moves on.** This actually happened to
    a configuration in a working folder. It's the kind of failure you only
    find out about after submitting 256 calculations, so it's caught here
    before anything is built.

    Returns
    -------
    [(element, path, bytes)]
    """
    if elements is None:
        elements = prim_elements(workdir, prim)
    if not elements:
        raise MainclustError("Could not find any elements in PRIM.")

    ok, missing, empty = [], [], []
    for elt in elements:
        path = os.path.join(workdir, POTCAR_PREFIX + elt)
        if not os.path.isfile(path):
            missing.append(path)
        elif os.path.getsize(path) == 0:
            empty.append(path)
        else:
            ok.append((elt, path, os.path.getsize(path)))

    if missing or empty:
        lines = ["Per-element POTCARs are not ready."]
        if missing:
            lines.append("  missing : " + ", ".join(os.path.basename(p) for p in missing))
        if empty:
            lines.append("  0 bytes : " + ", ".join(os.path.basename(p) for p in empty))
        lines.append("")
        lines.append("  mainclust makes a 0-byte POTCAR without an error even "
                     "when this file is missing.")
        lines.append("  Please place %s in %s and run again."
                     % (" ".join(POTCAR_PREFIX + e for e in elements),
                        os.path.abspath(workdir)))
        raise MainclustError("\n".join(lines))
    return ok


def _titel_element(line):
    """Just the element symbol from a POTCAR's TITEL line. None if it can't
    be read.

    ``TITEL  = PAW_PBE Cu_pv 22Jun2005`` -> ``Cu``
    """
    try:
        parts = line.split("=", 1)[1].split()
    except IndexError:                                 # pragma: no cover
        return None
    if not parts:
        return None
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.split("_", 1)[0]


def _pos_counts(path):
    """The per-element count line of a POS/POSCAR, as a list of ints. None
    if it isn't clear.

    The POS that CASM uses is VASP4 format, so there is no element-name
    line. The integer line right before the coordinate-system line
    (Direct/Cartesian) is the count line.
    """
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except OSError:                                    # pragma: no cover
        return None
    for i, line in enumerate(lines):
        if i < 5:                     # can't come before scale(1) + lattice(3) + counts(1)
            continue
        if line.strip()[:1].upper() not in ("D", "C", "S"):
            continue
        for j in range(i - 1, -1, -1):
            parts = lines[j].split()
            if not parts:
                continue
            try:
                return [int(v) for v in parts]
            except ValueError:
                return None
        return None
    return None


def check_templates(workdir=".", names=REQUIRED_TEMPLATES):
    """Check that the templates mainclust will copy exist, before creating con*.

    This is also a silent failure. If INCAR / KPOINTS are missing from the
    working directory, mainclust **exits normally without creating a single
    configuration directory.** It still prints exit 0, checkpoint 6, and
    "CASM code has completed successfully". Confirmed in a container with
    Cu-Ir -- 0 directories without the templates, 21 with them.

    Returns
    -------
    [(name, path, bytes)]
    """
    ok, missing, empty = [], [], []
    for name in names:
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            missing.append(name)
        elif os.path.getsize(path) == 0:
            empty.append(name)
        else:
            ok.append((name, path, os.path.getsize(path)))

    if missing or empty:
        lines = ["Templates for mainclust to copy into configuration directories are missing."]
        if missing:
            lines.append("  missing : " + ", ".join(missing))
        if empty:
            lines.append("  0 bytes : " + ", ".join(empty))
        lines.append("")
        lines.append("  Left as is, mainclust will exit normally without an error")
        lines.append("  and without creating a single configuration directory.")
        lines.append("  Please place the file(s) in %s and run again." % os.path.abspath(workdir))
        raise MainclustError("\n".join(lines))
    return ok


def check_generated_potcars(workdir=".", dirs=None, elements=None, prim="PRIM"):
    """Check that the con*/POTCAR files that got created are intact.

    The number of TITELs in a POTCAR must equal **the number of elements
    actually present in that configuration**. A pure endpoint (all Cu, all
    Ir) has only one element and so only one TITEL. So this cannot be
    compared against PRIM's element count -- it's checked against POS's
    count line instead.

    An element named in a TITEL that isn't in PRIM means a wrong POTCAR_ was
    placed there, so that is also flagged.

    Returns
    -------
    (number checked, [(directory, description of the problem)])
    """
    from CCpy.CASM.CASMkpoints import config_dirs
    if dirs is None:
        dirs = config_dirs(workdir)
    if elements is None:
        try:
            elements = prim_elements(workdir, prim)
        except MainclustError:
            elements = None

    bad = []
    for d in dirs:
        path = os.path.join(d, "POTCAR")
        if not os.path.isfile(path):
            bad.append((d, "POTCAR missing"))
            continue
        if os.path.getsize(path) == 0:
            bad.append((d, "0 bytes"))
            continue
        try:
            with open(path) as f:
                titels = [_titel_element(ln) for ln in f
                          if "TITEL" in ln and "=" in ln]
        except OSError as err:                         # pragma: no cover
            bad.append((d, "could not read: %s" % err))
            continue
        titels = [t for t in titels if t]
        if not titels:
            bad.append((d, "no TITEL lines (not POTCAR format)"))
            continue

        counts = _pos_counts(os.path.join(d, "POS"))
        if counts is not None and len(titels) != len(counts):
            bad.append((d, "%d TITEL(s) (POS has %d element(s))" % (len(titels), len(counts))))
            continue
        if elements:
            odd = [t for t in titels if t not in elements]
            if odd:
                bad.append((d, "element(s) not in PRIM: %s" % ", ".join(odd)))

    return len(dirs), bad


def describe_potcars(total, bad, limit=6):
    """check_generated_potcars's result as a single human-readable block."""
    if not bad:
        return "  Checked %d POTCAR(s) -- all fine." % total
    out = ["  %d of %d POTCAR(s) have a problem:" % (len(bad), total)]
    for d, why in bad[:limit]:
        out.append("    %-12s %s" % (os.path.basename(d), why))
    if len(bad) > limit:
        out.append("    ... (%d more)" % (len(bad) - limit))
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

class MainclustDriver(object):
    """Drives mainclust without a human present.

    Parameters
    ----------
    workdir : str
        Directory to run mainclust in. Inputs like PRIM must already be here.
    binary : str or None
        Path to the executable. If None, :func:`resolve_binary` searches
        environment variable -> config folder -> working directory -> PATH,
        in that order.
    idle_timeout : float
        If no output at all appears for this many seconds, it's treated as
        stuck. Checkpoint 1 (symmetry analysis) can take several minutes on
        a large cell, so allow plenty of room.
    total_timeout : float or None
        Overall time limit for the run. None means unlimited.
    echo : bool
        If True, mainclust's output is streamed to the screen as-is.
    """

    def __init__(self, workdir=".", binary=None,
                 idle_timeout=600.0, total_timeout=None, echo=False):
        self.workdir = os.path.abspath(workdir)
        self.binary = binary
        self.idle_timeout = float(idle_timeout)
        self.total_timeout = total_timeout
        self.echo = echo

    # -- paths ----------------------------------------------------------------

    def resolve_binary(self):
        """Return the absolute path to the executable. Raises MainclustError
        if it can't be found."""
        return resolve_binary(self.binary, self.workdir)

    # -- run --------------------------------------------------------------------

    def run(self, answers, strict=True):
        """Run mainclust once.

        Parameters
        ----------
        answers : dict
            question key -> answer. See :func:`answers_enumerate` /
            :func:`answers_reuse`.
        strict : bool
            If True, stop immediately when a question with no prepared
            answer appears. If False, feed that question a blank line and
            keep going (not recommended).

        Returns
        -------
        MainclustResult
        """
        unknown = set(answers) - set(k for k, _, _ in _COMPILED)
        if unknown:
            raise MainclustError(
                "Unknown question key(s): %s\navailable: %s"
                % (", ".join(sorted(unknown)),
                   ", ".join(k for k, _, _ in _COMPILED)))

        if not _HAS_PTY:                               # pragma: no cover
            raise MainclustError(
                "pty is not available on this platform, so mainclust cannot "
                "be driven. Please run this on Linux.")

        binary = self.resolve_binary()
        return self._run_pty(binary, answers, strict)

    def _run_pty(self, binary, answers, strict):
        master, slave = pty.openpty()
        self._disable_echo(slave)

        proc = subprocess.Popen(
            [binary],
            cwd=self.workdir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(slave)

        chunks = []
        pending = ""          # tail not yet matched to a question
        answered = []
        checkpoints = []
        completed = False
        started = time.time()
        last_output = started

        try:
            while True:
                if self.total_timeout is not None and \
                        time.time() - started > self.total_timeout:
                    self._kill(proc)
                    raise MainclustError(
                        "Stopped: exceeded the overall time limit of %.0f seconds."
                        % self.total_timeout, "".join(chunks))

                rlist, _, _ = select.select([master], [], [], 1.0)

                if rlist:
                    try:
                        data = os.read(master, 8192)
                    except OSError:
                        data = b""
                    if not data:
                        break
                    text = data.decode("utf-8", "replace")
                    chunks.append(text)
                    if self.echo:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    pending += text
                    last_output = time.time()

                    for num in _CHECKPOINT_RE.findall(text):
                        n = int(num)
                        if n not in checkpoints:
                            checkpoints.append(n)
                    if _DONE_RE.search(text):
                        completed = True

                    pending = self._consume(pending, answers, answered,
                                            master, proc, strict, chunks)
                else:
                    if proc.poll() is not None:
                        # Drain whatever output is left after exit.
                        tail = self._drain(master)
                        if tail:
                            chunks.append(tail)
                            if self.echo:
                                sys.stdout.write(tail)
                            for num in _CHECKPOINT_RE.findall(tail):
                                n = int(num)
                                if n not in checkpoints:
                                    checkpoints.append(n)
                            if _DONE_RE.search(tail):
                                completed = True
                        break
                    if time.time() - last_output > self.idle_timeout:
                        self._kill(proc)
                        raise MainclustError(
                            "Stopped: no output for %.0f seconds. mainclust may "
                            "be stuck on a question it has no answer for.\n"
                            "last output:\n%s"
                            % (self.idle_timeout, _tail("".join(chunks))),
                            "".join(chunks))
        finally:
            try:
                os.close(master)
            except OSError:
                pass
            if proc.poll() is None:
                self._kill(proc)

        returncode = proc.wait()
        transcript = "".join(chunks)
        used = set(k for k, _ in answered)
        unused = [k for k in answers if k not in used]

        return MainclustResult(returncode, transcript, answered, unused,
                               checkpoints, completed)

    # -- internals -----------------------------------------------------------

    def _consume(self, pending, answers, answered, master, proc, strict, chunks):
        """Handle every question findable in pending, and return the
        remaining tail."""
        while True:
            best = None
            for key, rx, desc in _COMPILED:
                m = rx.search(pending)
                if m and (best is None or m.start() < best[1].start()):
                    best = (key, m, desc)
            if best is None:
                # Keep a tail since a question's text may be split across
                # two chunks.
                return pending[-400:] if len(pending) > 400 else pending

            key, m, desc = best
            if key in answers:
                value = str(answers[key])
            elif strict:
                self._kill(proc)
                raise MainclustError(
                    "A question with no prepared answer appeared: '%s' (%s)\n"
                    "Please add '%s' to answers."
                    % (m.group(0).strip(), desc, key),
                    "".join(chunks))
            else:
                value = ""

            os.write(master, (value + "\n").encode("utf-8"))
            answered.append((key, value))
            pending = pending[m.end():]

    @staticmethod
    def _disable_echo(fd):
        """Turn off pty echo, so the answers we write don't get matched
        again when they come back in the output."""
        try:
            attrs = termios.tcgetattr(fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception:                              # pragma: no cover
            pass

    @staticmethod
    def _drain(fd, limit=1 << 20):
        out = []
        total = 0
        while total < limit:
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                break
            try:
                data = os.read(fd, 8192)
            except OSError:
                break
            if not data:
                break
            out.append(data.decode("utf-8", "replace"))
            total += len(data)
        return "".join(out)

    @staticmethod
    def _kill(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:                                # pragma: no cover
            try:
                proc.terminate()
            except OSError:
                pass
        for _ in range(30):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:                                           # pragma: no cover
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass


def _tail(text, lines=15):
    parts = text.rstrip().split("\n")
    return "\n".join(parts[-lines:])


# ----------------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------------

def enumerate_configurations(workdir=".", max_volume=1, dimension=3,
                             binary=None, echo=False, **kwargs):
    """Enumeration only (no directories are created). Produces make_dirs."""
    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    return drv.run(answers_enumerate(max_volume=max_volume, dimension=dimension))


def generate_vasp_inputs(workdir=".", energy=0, reference=0,
                         binary=None, echo=False, check_potcar=True, **kwargs):
    """Read make_dirs and create the con* directories.

    If ``check_potcar`` is True, checks that ``POTCAR_<element>`` exists
    before building, and that each configuration's POTCAR is non-empty
    afterward. mainclust silently lets both slip through, so without this
    check a 0-byte POTCAR gets submitted for calculation.

    Returns
    -------
    MainclustResult
        Carries a ``potcar_report`` attribute with the check results (when checked).
    """
    if check_potcar:
        check_potcar_sources(workdir)
        check_templates(workdir)

    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    result = drv.run(answers_reuse(energy=energy, reference=reference))

    if check_potcar and result.ok:
        total, bad = check_generated_potcars(workdir)
        if total == 0:
            raise MainclustError(
                "mainclust exited normally but created no configuration "
                "directories at all.\n"
                "  Either make_dirs's make flags are all 0, or INCAR / KPOINTS\n"
                "  were missing and mainclust silently skipped everything.",
                result.transcript)
        result.potcar_report = describe_potcars(total, bad)
        if bad:
            raise MainclustError(
                "con* were created, but the POTCARs are not intact.\n"
                + result.potcar_report
                + "\n\n  Submitting like this will make VASP compute the wrong "
                  "thing or die immediately.",
                result.transcript)
    return result


# ----------------------------------------------------------------------------
# make_dirs flags
# ----------------------------------------------------------------------------

def set_make_flags(path="make_dirs", value=1, backup=True):
    """Bulk-change make_dirs's make column (replaces 02_ModiMake.sh).

    mainclust only builds a VASP input directory for configurations where
    this column is 1. Right after enumeration it's all 0, so it needs to be
    flipped to 1.

    The original shell script relied on whitespace count, like
    ``sed -e "s/0     0/1     0/g"``, and silently did nothing the moment
    mainclust's output width shifted even slightly. Here the line is split
    into columns and rewritten instead.

    Returns
    -------
    (changed, total)
    """
    if not os.path.isfile(path):
        raise MainclustError("%s is missing. Run mainclust's enumeration first." % path)

    lines = open(path).read().split("\n")
    header, rows, changed = None, [], 0
    for line in lines:
        if not line.strip():
            continue
        if header is None and line.lstrip().startswith("#"):
            header = line
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[1] != str(value):
            changed += 1
        rows.append((parts[0], str(value), parts[2]))

    if not rows:
        raise MainclustError("Could not find any configuration rows in %s." % path)

    if backup:
        orig = path + "_orig"
        if not os.path.isfile(orig):
            shutil.copy(path, orig)

    out = [header] if header else ["#    name      make      concentrations  "]
    out += ["%s  %s     %s   " % r for r in rows]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return changed, len(rows)
