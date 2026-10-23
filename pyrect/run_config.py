"""core/fit/config.cpp: the command line and the environment, validated the same way."""
from __future__ import annotations

import os
import re


class FitConfig:
    def __init__(self):
        self.heldout_split = False
        self.baseline = "training"
        self.fdr_level = 0.40      # the gate's false discovery rate (DMESH_Q); 0.10 before 2026-09-29


class OutputConfig:
    def __init__(self):
        self.prefix = None


class RunConfig:
    """The run's configuration: the one positional argument (the seed) and the DMESH_* environment."""
    def __init__(self):
        self.data_file = None; self.seed = 7; self.fit = FitConfig(); self.output = OutputConfig()


def _env_flag(name):
    v = os.environ.get(name)
    if v is None:
        return False
    return v == "" or (v != "0" and v != "false" and v != "FALSE")


def _env_string(name):
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return v


_INTEGER = re.compile(r"^[+-]?\d+$", re.ASCII)


def _parse_int(text, label):
    """The binary's parse_number<int>: strtol on the whole token, within the int range."""
    token = text.lstrip(" \t\n\r\x0b\x0c")          # strtol skips leading whitespace, not trailing
    if not _INTEGER.match(token):
        raise RuntimeError(f"invalid {label}: {text}")
    v = int(token, 10)
    if v < -2147483648 or v > 2147483647:
        raise RuntimeError(f"invalid {label}: {text}")
    return v


_DECIMAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", re.ASCII)


def _parse_float(text, label):
    """The binary's parse_number<double>: strtod on the whole token, finite (decimal forms)."""
    token = text.lstrip(" \t\n\r\x0b\x0c")          # strtod skips leading whitespace, not trailing
    if not _DECIMAL.match(token):
        raise RuntimeError(f"invalid {label}: {text}")
    v = float(token)
    if v != v or v in (float("inf"), float("-inf")):
        raise RuntimeError(f"invalid {label}: {text}")
    return v


def parse_run_config(argv):
    config = RunConfig()
    args = argv[1:]
    if len(args) > 1:
        raise RuntimeError("the engine takes one optional argument, the seed (the five-argument form "
                           "'sigma_u q scale points_per_face seed' was retired on 2026-09-22)")
    if len(args) > 0:
        seed = _parse_int(args[0], "seed")
        if seed < 0:
            raise RuntimeError("seed must be nonnegative")
        config.seed = seed
    config.data_file = _env_string("DMESH_DATA")
    config.fit.heldout_split = _env_flag("DMESH_SPLIT")
    config.fit.baseline = _env_string("DMESH_BASELINE") or config.fit.baseline
    if config.fit.baseline not in ("training", "external"):
        raise RuntimeError("DMESH_BASELINE must be training or external")
    config.output.prefix = _env_string("DMESH_DUMP")
    if config.output.prefix is not None:
        parent = os.path.dirname(config.output.prefix) or "."
        if not os.path.isdir(parent):
            raise RuntimeError(f"DMESH_DUMP directory does not exist: {parent}")
        if not os.access(parent, os.W_OK):
            raise RuntimeError(f"DMESH_DUMP directory is not writable: {parent}")
    q = _env_string("DMESH_Q")
    if q is not None:
        config.fit.fdr_level = _parse_float(q, "DMESH_Q")
        if not (0.0 < config.fit.fdr_level < 1.0):
            raise RuntimeError("DMESH_Q must be strictly between 0 and 1")
    return config
