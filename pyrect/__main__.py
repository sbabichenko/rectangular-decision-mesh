"""python3 -m pyrect [seed=7]

The same command line and DMESH_* environment as build-release/rect_mesh, the same dump
files, and the same numbers to the tolerance pyrect/tests/test_port.py checks. See README.md
beside this file."""
import sys

from .estimator.engine import Engine
from .dataset import build_dataset
from .run_config import parse_run_config


def _tune_allocator():
    """Keep glibc from returning the fit's large, short-lived numpy arrays (the moment blocks run
    to tens of MB) to the kernel after every use: with its default, dynamic threshold each one
    is a fresh mmap whose pages are zeroed and faulted in again, which cost the benchmark fits
    about 20% (measured 2026-09-22; peak memory is lower, not higher). No effect elsewhere."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3
        libc.mallopt(M_MMAP_THRESHOLD, 64 << 20)
        libc.mallopt(M_TRIM_THRESHOLD, 128 << 20)
    except (OSError, AttributeError):
        pass


def main(argv=None):
    _tune_allocator()
    argv = sys.argv if argv is None else argv
    try:
        config = parse_run_config(argv)
    except RuntimeError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.stderr.write("usage: python -m pyrect [seed=7]\n")
        return 2
    try:
        dataset = build_dataset(config)
        return Engine(config, dataset).run()
    except (RuntimeError, OSError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
