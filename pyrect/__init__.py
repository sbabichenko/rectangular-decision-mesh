"""A Python implementation of core/, the adaptive rectangular mesh, in numpy and scipy: the same
command line, environment and dump files as the binary, and the same numbers to a relative
tolerance (pyrect/tests/test_port.py). Nothing is compiled.

    python3 -m pyrect [seed]      # the same CLI and DMESH_* environment as the binary
    python3 -m pyrect.compare <design.csv> --binary build-release/rect_mesh ...   # run both, diff every dump

    from pyrect import fit
    result = fit("design.csv", "out/run", options={"DMESH_SPLIT": "1"})  # a fit in a fresh process, dumps returned by path
"""
__version__ = "0.1.0"

from .api import FitResult, fit   # noqa: E402,F401
