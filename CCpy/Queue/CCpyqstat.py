#!/usr/bin/env python
"""
Per-server qstat implementation selector (dispatcher).

cms2 and node99 read the queue status in completely different ways.

    cms2   : parse PBS-style 'qstat -f' output    -> CCpyqstat_cms2.py
    node99 : parse SLURM 'squeue' output          -> CCpyqstat_node99.py

Both implementations expose the same interface, CCpyqstat / get_empty_nodes /
get_waiting_nodes, so here we pick the one matching the server and re-export it.
(CCpy/bin/CCpyqstat.py is a symbolic link pointing to this file.)

For the server detection rules see CCpy/Queue/server_profile.py.
  1) environment variable $CCpy_SERVER  ("cms2" / "node99")
  2) hostname guess
  3) DEFAULT_SERVER
"""

from CCpy.Queue.server_profile import get_server_name

_server = get_server_name()

if _server == "node99":
    from CCpy.Queue import CCpyqstat_node99 as _impl
else:
    from CCpy.Queue import CCpyqstat_cms2 as _impl

# -- common interface
CCpyqstat = _impl.CCpyqstat
get_empty_nodes = _impl.get_empty_nodes
get_waiting_nodes = _impl.get_waiting_nodes

# -- feature that exists only in the cms2 version
if hasattr(_impl, "chk_load"):
    chk_load = _impl.chk_load


if __name__ == "__main__":
    import runpy

    runpy.run_module(_impl.__name__, run_name="__main__")
