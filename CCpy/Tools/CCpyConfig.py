"""Single place that decides where the CCpy personal config folder (`~/.CCpy_test`) lives.

CCpy does not read the yaml bundled in the package directly; on the first run it copies
it once into a config folder in the user home and then reads that copy. This repo
(CCpy_test) is an upgrade working copy, so sharing `~/.CCpy` (used by the production
CCpy) as-is contaminates the settings of both sides:

  - production reads the default.yaml / queue_config.yaml made by the test version.
  - conversely, yaml keys added in the repo are ignored because of an old home copy
    (the copy is created only once, on the first run).
  - `python_path` in queue_config.yaml points at the production interpreter, so a job
    submitted with the test version runs on the production python on the compute node.
    (this is because that path was hardcoded in the package template. So when the file
     is first created, `python_path` is recorded as the absolute path of the python at
     that moment -- see ensure_queue_config() below.)

After the file is created CCpy does not modify the config yaml again. Per-server /
per-user values are edited by the user directly, and the code only reads them.

So this repo uses `~/.CCpy_test`. To avoid scattering the path string over many files
it is defined only here, and the rest of the code calls the functions below.

Only the standard library is imported (importing CCpyTools etc. would be a circular import).
"""

import os
import re
import sys
from pathlib import Path

# -- when promoting this repo to production, revert only this one line to ".CCpy".
DEFAULT_CONFIG_DIRNAME = ".CCpy_test"

# -- environment variable to move the config folder elsewhere (same form as CCpy_SERVER).
CONFIG_HOME_ENV = "CCpy_HOME"


def config_home():
    """Config folder path (Path). If $CCpy_HOME is set, that one is used."""
    env = os.environ.get(CONFIG_HOME_ENV)
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env)))
    return Path(os.path.expanduser("~")) / DEFAULT_CONFIG_DIRNAME


def ensure_config_home():
    """Create the config folder if it does not exist. Does nothing if it exists."""
    d = config_home()
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path(*parts):
    """Build a path under the config folder. ex) config_path("g09_input.json")"""
    return config_home().joinpath(*parts)


def vasp_config_dir():
    """VASP preset yaml folder (`~/.CCpy_test/vasp`)."""
    return config_home() / "vasp"


def queue_config_path():
    """`~/.CCpy_test/queue_config.yaml`"""
    return config_home() / "queue_config.yaml"


def package_queue_config_template():
    """queue_config.yaml template bundled in the package (`CCpy/Queue/queue_config.yaml`)."""
    return Path(__file__).resolve().parent.parent / "Queue" / "queue_config.yaml"


PYTHON_PATH_KEY = "python_path"


def set_python_path(text, python_path):
    """Replace the python_path value in the queue_config.yaml text with the given path.

    Returns (changed text, recorded value). Instead of rewriting with yaml, it replaces
    line by line to keep the comments and key order. If the key is missing, it is added at the top.
    """
    out = []
    done = False
    for line in text.splitlines(True):
        if not done and re.match(r"\s*%s\s*:" % PYTHON_PATH_KEY, line):
            out.append("%s: %s\n" % (PYTHON_PATH_KEY, python_path))
            done = True
            continue
        out.append(line)
    if not done:
        out.insert(0, "%s: %s\n" % (PYTHON_PATH_KEY, python_path))
    return "".join(out), python_path


def ensure_queue_config(template=None):
    """Create queue_config.yaml from the package template if missing, and return its path.

    When creating it, only `python_path` is filled with the absolute path of the python
    currently running CCpy (sys.executable). Running a CCpy command once in your own
    virtual environment records the python of that environment, so there is no need to
    hardcode a specific environment path in the template or fix it by hand from the start.

    If the file already exists, nothing is done. **After it is created CCpy does not
    modify this file again** -- per-server / per-user paths (vasp/lammps/siesta etc.) are
    edited by the user directly and the code only reads them.

    There is no inheriting of an existing `~/.CCpy/queue_config.yaml`. It was dropped
    because production values (especially a python_path pointing at Python 3.8) tagged along.

    template : path of the queue_config.yaml bundled in the package. If omitted,
               package_queue_config_template() is used.
    """
    if template is None:
        template = package_queue_config_template()
    target = queue_config_path()
    if target.is_file():
        return target

    ensure_config_home()
    text, recorded = set_python_path(Path(template).read_text(encoding="utf-8"), sys.executable)
    target.write_text(text, encoding="utf-8")
    print("* queue_config.yaml was newly created: %s" % target)
    print("    python_path: %s   (the python currently running CCpy)" % recorded)
    print("  - edit this file directly for per-server / per-user paths (vasp/lammps/siesta etc.).")
    print("    CCpy does not modify this file after it is created.")
    return target
