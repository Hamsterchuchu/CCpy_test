"""
Per-server (cluster) hardware profile definitions.

cms2 and node99 run the same CCpy code, but their node names / partition layout
differ. Previously each of the two servers patched CCpyJobControl.py and
CCpyqstat.py on its own, and because of that the code of the two servers diverged.

Here the settings of both servers are kept together, and the right one is picked
at run time depending on which server it is. The server selection order follows.

  1) environment variable $CCpy_SERVER  ("cms2" or "node99")   <- recommended.
     Just add one line like    export CCpy_SERVER=cms2    to .bashrc.
  2) guess from hostname
  3) DEFAULT_SERVER if both of the above fail

To add a new server, just put one more entry into NODE_PROFILES.
"""

import os
import socket

# Default value used when the server could not be detected
DEFAULT_SERVER = "cms2"

NODE_PROFILES = {
    # ------------------------------------------------------------------ #
    "cms2": {
        "default_partition": "72core",
        # partition name -> list of nodes belonging to that partition
        "node_partitions": {
            "8core":   ["node00", "node01"],
            "72core":  ["node02", "node03", "node04", "node05", "node06"],
            "128core": ["node07"],
        },
        # Partition to map to when only a core count is given instead of a node name (like -n 8).
        # In this case the job goes to the whole partition without a specific node.
        "partition_alias": {
            "8":   "8core",
            "128": "128core",
        },
    },
    # ------------------------------------------------------------------ #
    "node99": {
        "default_partition": "48core",
        "node_partitions": {
            "48core":  ["node01", "node02", "node03", "node04",
                        "node05", "node06", "node07", "node08"],
            "96core":  ["node09", "node10", "node11", "node12"],
            "64core":  ["node13", "node14", "node15", "node16",
                        "node17", "node18"],
            "256core": ["node19"],
        },
        "partition_alias": {
            "96":  "96core",
            "64":  "64core",
            "256": "256core",
        },
    },
}


def get_server_name():
    """Return the name of the server this is running on."""
    name = os.environ.get("CCpy_SERVER", "").strip()
    if name in NODE_PROFILES:
        return name

    try:
        host = socket.gethostname().lower()
    except Exception:
        host = ""
    for key in NODE_PROFILES:
        if key in host:
            return key

    return DEFAULT_SERVER


def get_node_profile(server=None):
    """Return the server profile dict."""
    if server is None:
        server = get_server_name()
    return NODE_PROFILES[server]
