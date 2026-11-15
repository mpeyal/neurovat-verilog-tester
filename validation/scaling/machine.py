"""Machine description recorded with every C3 timing (no host or user names).

Single-threaded on the Python side: the BLAS / OpenMP thread counts are set
to 1 before numpy is imported by any scaling script (import this module
first).
"""
import os
import platform
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for ln in f:
                if ln.lower().startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def info():
    import numpy as np
    return {"cpu": cpu_model(), "logical_cpus": os.cpu_count(),
            "os": platform.system(), "python": sys.version.split()[0],
            "numpy": np.__version__, "threads": 1, "loadavg_1_5_15": load()}
