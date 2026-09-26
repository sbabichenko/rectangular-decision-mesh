"""Run the C++ binary and the Python port on one design under one environment and compare
every output file and every printed line.

    python3 -m pyrect.compare <design.csv> [--binary build-release/rect_mesh] [--out DIR]
                              [--env KEY=VALUE ...] [--args 7] [--tolerance 1e-9]

Exit status 0 when every dump agrees to the relative tolerance (0 for byte identity) and the
normalized stdout matches.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from .api import DUMPS as DUMP_NAMES

DUMPS = tuple("_" + name for name in DUMP_NAMES)     # the six dumps as prefix suffixes


def normalize_stdout(text: str) -> str:
    return re.sub(r"^TOTAL [0-9]+\.[0-9]+ s$", "TOTAL [T]", re.sub(r"\[[a-z]* *[0-9]+\.[0-9]+ s\]", "[T]", text), flags=re.M)


def normalize_stderr(text: str) -> str:
    """The usage line names the program (argv[0] in the binary, `python -m pyrect` in the port), and
    the empirical null's iteration count is a convergence diagnostic that rounding can move by a
    step or two (the fitted null it reports agrees)."""
    text = re.sub(r"^usage: .*? \[seed", "usage: [PROGRAM] [seed", text, flags=re.M)
    return re.sub(r"lindsey-status (\S+) iter \d+", r"lindsey-status \1 iter [N]", text)


def run(cmd, env, out_dir, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    from .api import child_environment
    e = child_environment(os.environ, inherit_pyrect=True)   # the package importable by the child from any cwd
    e.update(env)
    e["DMESH_DUMP"] = str(out_dir / prefix)
    with open(out_dir / "stdout.log", "w") as so, open(out_dir / "stderr.log", "w") as se:
        rc = subprocess.run(cmd, env=e, stdout=so, stderr=se).returncode
    return rc


def first_difference(a: str, b: str):
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return i + 1, x, y
    if len(la) != len(lb):
        return min(len(la), len(lb)) + 1, "<%d lines>" % len(la), "<%d lines>" % len(lb)
    return None


def within_tolerance(a: str, b: str, tolerance: float):
    """Line by line, token by token (commas and whitespace): numbers agree to the relative
    tolerance (absolute below one), everything else exactly. Returns None or the first
    differing (line number, a, b)."""
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return (min(len(la), len(lb)) + 1, "<%d lines>" % len(la), "<%d lines>" % len(lb))
    split = re.compile(r"[,\s]+")
    for i, (x, y) in enumerate(zip(la, lb), 1):
        if x == y:
            continue
        tx, ty = split.split(x), split.split(y)
        if len(tx) != len(ty):
            return (i, x, y)
        for u, v in zip(tx, ty):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return (i, x, y)
            if not (fu == fv or abs(fu - fv) <= tolerance * max(1.0, abs(fu), abs(fv))):
                return (i, x, y)
    return None


def compare(design: Path, binary: Path, out: Path, env: dict, args: list[str], logs: bool = True,
            tolerance: float | None = None) -> bool:
    """tolerance None: every dump byte-identical. A number: the
    dumps agree token by token to that relative tolerance (the centered method's standard since
    2026-09-21; the engines are the same algorithm, not the same arithmetic order)."""
    env = dict(env); env["DMESH_DATA"] = str(design)
    cpp = out / "cpp"; py = out / "py"
    rc1 = run([str(binary), *args], env, cpp, "run")
    rc2 = run([sys.executable, "-m", "pyrect", *args], env, py, "run")
    ok = True
    print(f"exit codes: cpp {rc1}, py {rc2}")
    ok &= rc1 == rc2
    for suffix in DUMPS:
        a, b = cpp / ("run" + suffix), py / ("run" + suffix)
        if not a.exists() or not b.exists():
            if not a.exists() and not b.exists():
                print(f"{suffix}: absent on both sides"); continue      # both refused the run
            print(f"{suffix}: missing ({a.exists()}, {b.exists()})"); ok = False; continue
        ta, tb = a.read_text(), b.read_text()
        if ta == tb:
            print(f"{suffix}: identical ({len(ta.splitlines())} lines)")
        elif tolerance is not None and within_tolerance(ta, tb, tolerance) is None:
            print(f"{suffix}: within {tolerance:g} ({len(ta.splitlines())} lines)")
        else:
            d = first_difference(ta, tb) if tolerance is None else within_tolerance(ta, tb, tolerance)
            print(f"{suffix}: DIFFER at line {d[0]}\n   cpp: {d[1][:160]}\n   py : {d[2][:160]}")
            ok = False
    for name in ("stdout.log", "stderr.log") if logs else ():
        ta, tb = (cpp / name).read_text(), (py / name).read_text()
        if name == "stdout.log":
            ta, tb = normalize_stdout(ta), normalize_stdout(tb)
        else:
            ta, tb = normalize_stderr(ta), normalize_stderr(tb)
        if ta == tb:
            print(f"{name}: identical")
        elif tolerance is not None and within_tolerance(ta, tb, tolerance) is None:
            print(f"{name}: within {tolerance:g}")
        else:
            d = first_difference(ta, tb) if tolerance is None else within_tolerance(ta, tb, tolerance)
            print(f"{name}: differ at line {d[0]}\n   cpp: {d[1][:200]}\n   py : {d[2][:200]}")
            ok = False
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design", type=Path)
    ap.add_argument("--binary", type=Path, required=True, help="the C++ rect_mesh to compare against")
    ap.add_argument("--out", type=Path, default=Path("pyrect-compare"))
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--args", nargs="*", default=["7"])
    ap.add_argument("--tolerance", type=float, default=1e-9, help="relative tolerance per token; 0 for byte identity")
    a = ap.parse_args(argv)
    env = dict(kv.split("=", 1) for kv in a.env)
    ok = compare(a.design, a.binary, a.out, env, a.args, tolerance=a.tolerance or None)
    print(("AGREE" if a.tolerance else "BIT-IDENTICAL") if ok else "DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
