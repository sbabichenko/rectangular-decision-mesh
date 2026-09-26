"""A fit as a function call.

The estimator is configured through the `DMESH_*` environment and keeps process-global state
(construction ids), exactly as the binary does, so one fit is one process.
`fit` runs the port in a fresh interpreter with a clean environment and returns the paths of
its dumps. The overlay package launches the engine through `run_process` below with the same
process-group handling; it strips the caller's `DMESH_*` settings and passes `PYRECT_*` ones
through, where `fit` strips both and takes its worker count as an argument."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DUMPS = ("model.json", "obs.csv", "mesh.csv", "hier_vertices.csv", "surface.csv", "pools.csv", "candidates.csv")


class FitError(RuntimeError):
    """The fit did not complete; the message carries the tail of its stderr."""


@dataclass
class FitResult:
    prefix: Path
    dumps: dict = field(default_factory=dict)      # name -> path, for the dumps that exist
    stdout: str = ""
    stderr: str = ""


def package_parent_for_child() -> str | None:
    """The directory to put on the child's PYTHONPATH so `-m pyrect` imports this package: the
    repository root when running from a checkout; nothing when installed (site-packages is on
    the child's path already, and PYTHONPATH would precede the standard library)."""
    parent = Path(__file__).resolve().parents[1]
    if any(part in ("site-packages", "dist-packages") for part in parent.parts):
        return None
    return str(parent)


# PYRECT_* settings that only affect how a fit is executed, never what it estimates. Everything
# else is dropped when building a child's environment, so a packaged fit cannot inherit a
# modelling choice from whatever happened to be exported in the operator's shell.
PYRECT_OPERATIONAL = ("PYRECT_TIMEOUT",)


def child_environment(base: dict, inherit_pyrect: bool) -> dict:
    """`base` without the caller's DMESH_* settings, with the package importable by the child.

    `inherit_pyrect` passes through the operational PYRECT_* settings only.
    """
    def keep(k):
        if k.startswith("DMESH_"):
            return False
        if k.startswith("PYRECT_"):
            return inherit_pyrect and k in PYRECT_OPERATIONAL
        return True

    env = {k: v for k, v in base.items() if keep(k)}
    parent = package_parent_for_child()
    if parent is not None:
        env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_process(cmd: list, env: dict, stdout, stderr, timeout: float | None) -> tuple[int, bool]:
    """Run `cmd` in its own process group; on timeout kill the whole group (the port's forked
    workers included). Returns (returncode, timed_out)."""
    kwargs = {"start_new_session": True} if hasattr(os, "setsid") else {}
    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, **kwargs)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        proc.wait()
        return proc.returncode, True


def fit(design: str | Path, dump_prefix: str | Path, options: dict | None = None, *,
        seed: int = 7,
        timeout: float | None = None) -> FitResult:
    """Fit the design at `design` (the binary's CSV format) and write the dumps under
    `dump_prefix`. `options` are the `DMESH_*` settings of the run (see pyrect/README.md), for
    example {"DMESH_SPLIT": "1"}. The caller's own DMESH_*/PYRECT_* variables are not
    inherited."""
    design = Path(design); prefix = Path(dump_prefix)
    if not design.exists():
        raise FileNotFoundError(f"design not found: {design}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env = child_environment(os.environ, inherit_pyrect=False)
    for k, v in (options or {}).items():
        if not k.startswith("DMESH_"):
            raise ValueError(f"options are DMESH_* settings; got {k!r}")
        env[k] = str(v)
    env["DMESH_DATA"] = str(design); env["DMESH_DUMP"] = str(prefix)
    cmd = [sys.executable, "-m", "pyrect", str(seed)]
    out_path = prefix.parent / f"{prefix.name}.stdout.log"; err_path = prefix.parent / f"{prefix.name}.stderr.log"
    with open(out_path, "w") as out, open(err_path, "w") as err:
        returncode, timed_out = run_process(cmd, env, out, err, timeout)
    stdout = out_path.read_text(errors="replace"); stderr = err_path.read_text(errors="replace")
    if timed_out:
        raise FitError(f"the fit exceeded {timeout} s and was killed; last lines of its stderr:\n" + "".join(stderr.splitlines(keepends=True)[-12:]))
    if returncode != 0:
        raise FitError(f"the fit exited with status {returncode}; last lines of its stderr:\n" + "".join(stderr.splitlines(keepends=True)[-12:]))
    dumps = {name: prefix.parent / f"{prefix.name}_{name}" for name in DUMPS}
    return FitResult(prefix, {k: p for k, p in dumps.items() if p.exists()}, stdout, stderr)
