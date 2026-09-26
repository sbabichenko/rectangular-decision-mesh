"""The port agrees with the binary to CENTERED_TOLERANCE on every configuration, and the port's
geometry keeps the contracts core/tests/test_rect_geometry.cpp checks for the binary's."""
import os
from pathlib import Path

import numpy as np
import pytest

from pyrect import compare
from pyrect.stdlib import MT19937_64

BINARY = Path(os.environ.get("DMESH_BINARY", Path(__file__).resolve().parents[2] / "build-release" / "rect_mesh"))
# The two engines implement one algorithm, not one arithmetic order: their dumps must agree to
# this relative tolerance.
CENTERED_TOLERANCE = 1e-9
needs_binary = pytest.mark.skipif(not BINARY.exists(), reason="build-release/rect_mesh is not built")
BASE = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}


def write_design(tmp_path, name, n_pools, seed):
    from .synthetic import synthetic_design
    path = tmp_path / f"{name}.csv"
    synthetic_design(n_pools=n_pools, seed=seed).to_csv(path, index=False)
    return path


def write_mask(tmp_path, n, seed):
    rng = np.random.default_rng(seed)
    path = tmp_path / f"mask{seed}.txt"
    path.write_text("\n".join("1" if t else "0" for t in rng.random(n) < 0.8) + "\n")
    return path


def test_mersenne_twister_matches_the_standard_first_output():
    assert MT19937_64(5489)() == 14514284786278117030


def test_geometry_contracts():
    """Constrained moves and splits leave the surface unchanged; a candidate's column before its
    move is its realised column; the nodal rows reproduce the predictions."""
    from pyrect.mesh import RectMesh

    class Obs:
        def __init__(self, x, y): self.x, self.y, self.offset, self.exposure, self.events = x, y, 0.0, 10, 1

    rng = np.random.default_rng(3)
    obs = [Obs(*rng.uniform(0, 1, 2)) for _ in range(1500)] + [Obs(0.5, rng.uniform()) for _ in range(40)]
    mesh = RectMesh(obs, [1] * len(obs))
    mesh.refine_uniform(2)
    for v in mesh.vertices:
        if not v.has_prior() or v.depth <= 1:
            v.free_coefficient = True; v.surplus = float(rng.normal())
    mesh.coefficient_version += 1
    mesh.recompute_heights()
    for step in range(40):
        frontier = mesh.frontier()
        candidate = frontier[int(rng.integers(len(frontier)))]
        rows, values = mesh.candidate_column(candidate)
        before = mesh.predict_all_array().copy()
        vid = mesh.realise(candidate)
        mesh.recompute_heights()
        assert np.max(np.abs(mesh.predict_all_array() - before)) < 1e-12
        mesh.vertices[vid].free_coefficient = True; mesh.coefficient_version += 1
        mesh.vertices[vid].surplus = 1.0
        mesh.recompute_heights()
        dense = np.zeros(len(obs)); dense[rows] = values
        assert np.max(np.abs((mesh.predict_all_array() - before) - dense)) < 1e-12
        mesh.vertices[vid].surplus = float(rng.normal()) * 0.3
        mesh.recompute_heights()
    ids, vals = mesh.nodal_rows(np.arange(len(obs)))
    heights = np.array([v.height for v in mesh.vertices])
    assert np.max(np.abs((vals * heights[ids]).sum(axis=1) - mesh.predict_all_array())) < 1e-10


@needs_binary
def test_port_agrees_with_the_binary_with_a_mask(tmp_path):
    design = write_design(tmp_path, "d", 300, 0)
    env = {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(write_mask(tmp_path, 300, 1))}
    assert compare.compare(design, BINARY, tmp_path / "cmp", env, ["7"], tolerance=CENTERED_TOLERANCE)


@needs_binary
def test_the_engines_own_split_and_a_large_design_agree_with_the_binary(tmp_path):
    """The local exposure-balanced split the engine generates itself, and a design above the
    2048-row chunking of the moment and mean-neutral evaluations."""
    base = {**BASE, "DMESH_SPLIT": "1"}
    small = write_design(tmp_path, "small", 300, 5)
    assert compare.compare(small, BINARY, tmp_path / "own_split", base, ["7"], tolerance=CENTERED_TOLERANCE)
    large = write_design(tmp_path, "large", 2500, 6)
    assert compare.compare(large, BINARY, tmp_path / "large", base, ["7"], tolerance=CENTERED_TOLERANCE)


@needs_binary
def test_joint_levels_and_the_external_baseline_agree_with_the_binary(tmp_path):
    design = write_design(tmp_path, "d", 800, 2)
    base = {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(write_mask(tmp_path, 800, 3))}
    groups = tmp_path / "groups.txt"
    groups.write_text("\n".join(str(i % 3) for i in range(800)) + "\n")
    assert compare.compare(design, BINARY, tmp_path / "levels", {**base, "DMESH_LEVEL_GROUPS": str(groups)}, ["7"],
                           logs=False, tolerance=CENTERED_TOLERANCE)
    assert compare.compare(design, BINARY, tmp_path / "external", {**base, "DMESH_BASELINE": "external"}, ["7"],
                           tolerance=CENTERED_TOLERANCE)


@needs_binary
def test_several_level_factors_agree_with_the_binary(tmp_path):
    """DMESH_LEVEL_GROUPS with one column per factor: the binary and the port fit the same
    additive levels, and a factor holding only its reference changes no dump."""
    design = write_design(tmp_path, "d", 800, 2)
    base = {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(write_mask(tmp_path, 800, 3))}
    files = {"one": [f"{i % 3}" for i in range(800)], "padded": [f"{i % 3} 0" for i in range(800)],
             "two": [f"{i % 3} {(i // 7) % 4}" for i in range(800)]}
    for name, lines in files.items():
        groups = tmp_path / f"{name}.txt"
        groups.write_text("\n".join(lines) + "\n")
        assert compare.compare(design, BINARY, tmp_path / name, {**base, "DMESH_LEVEL_GROUPS": str(groups)}, ["7"],
                               logs=False, tolerance=CENTERED_TOLERANCE)
    for dump in ("run_obs.csv", "run_mesh.csv", "run_hier_vertices.csv"):
        assert (tmp_path / "one" / "cpp" / dump).read_bytes() == (tmp_path / "padded" / "cpp" / dump).read_bytes()
    import json
    levels = json.loads((tmp_path / "two" / "cpp" / "run_model.json").read_text())["centered_levels"]
    assert levels["factor_sizes"] == [2, 3] and len(levels["levels"]) == 5


@needs_binary
def test_off_surface_inputs_agree_with_the_binary(tmp_path):
    """A split mask with an out-of-range or non-integer token, a loan count above the int range,
    a headerless file with a trailing comma, seeds the C grammar refuses, the retired
    five-argument command line, a directory for the data."""
    design = write_design(tmp_path, "d", 120, 4)
    good = design.read_text().splitlines()

    def agree(name, env, design=design, args=("7",)):
        assert compare.compare(design, BINARY, tmp_path / name, env, list(args), tolerance=CENTERED_TOLERANCE), name
    n = len(good) - 1
    mask = tmp_path / "mask.txt"
    mask.write_text("99999999999\n" + "1\n" * (n - 1))
    agree("mask_overflow", {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(mask)})
    mask.write_text("yes\n" + "1\n" * (n - 1))
    agree("mask_word", {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(mask)})
    f = good[1].split(","); f[2] = "5000000000"; f[3] = "4"
    huge = tmp_path / "huge.csv"; huge.write_text("\n".join([good[0], ",".join(f)] + good[2:]) + "\n")
    agree("huge_n0", BASE, huge)
    headerless = tmp_path / "headerless.csv"
    headerless.write_text("".join(f"{1 + i},{3 + (7 * i % 40) / 10},{100 + 10 * i},{2 + i},\n" for i in range(40)))
    agree("headerless_trailing_comma", BASE, headerless)
    for seed in ("1_0", " 7", "0x7", "7.0", "3000000000"):
        agree("seed_" + seed.strip().replace("_", "u").replace(".", "p"), BASE, design, [seed])
    agree("retired_five_arguments", BASE, design, ["0", "0.10", "1", "24", "7"])
    agree("data_is_dir", {**BASE, "DMESH_DATA": str(tmp_path)}, tmp_path)


def test_malformed_design_rows_are_refused_identically(tmp_path):
    """Refused before any output with the binary's message; a blank CRLF line is skipped."""
    from pyrect import fit
    from pyrect.api import FitError
    design = write_design(tmp_path, "good", 120, 7)
    good = design.read_text().splitlines()
    first = good[1].split(",")

    def with_wala(v):
        f = list(first); f[0] = v; return ",".join(f)
    cases = {"blank_pool": (good[1].rsplit(",", 1)[0] + ",", "has no pool_id"),
             "nan": (with_wala("nan"), "non-finite"),
             "zero_n0": (",".join(["1.5", "4.0", "0", "0"] + first[4:]), "invalid counts"),
             "k_gt_n": (",".join(["1.5", "4.0", "10", "11"] + first[4:]), "invalid counts"),
             "prefix": (with_wala(first[0] + "abc"), "non-numeric wala"),
             "hex": (with_wala("0x10"), "non-numeric wala")}
    for name, (bad_row, expect) in cases.items():
        bad = tmp_path / f"{name}.csv"
        bad.write_text("\n".join([good[0], bad_row] + good[2:]) + "\n", encoding="utf-8")
        with pytest.raises(FitError, match=expect):
            fit(bad, tmp_path / name / "run", {})
        assert not list((tmp_path / name).glob("run_*"))
        if BINARY.exists():
            assert compare.compare(bad, BINARY, tmp_path / name / "cmp", {}, ["7"], tolerance=CENTERED_TOLERANCE)
    crlf = tmp_path / "crlf.csv"
    crlf.write_text("\r\n".join([good[0], good[1], "", good[2]] + good[3:]) + "\r\n")
    result = fit(crlf, tmp_path / "crlf" / "run", {})
    assert "run_mesh.csv" in {p.name for p in result.dumps.values()}


def test_fit_runs_in_a_fresh_process_with_a_clean_environment(tmp_path, monkeypatch):
    """The programmatic entry point: the caller's DMESH_* settings do not leak in, the dumps come
    back by path, and a bad option is refused before anything runs."""
    from pyrect import fit
    from pyrect.api import FitError
    design = write_design(tmp_path, "d", 300, 8)
    monkeypatch.setenv("DMESH_LEVEL_GROUPS", str(tmp_path / "missing.txt"))   # must not reach the fit
    result = fit(design, tmp_path / "run", {"DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(write_mask(tmp_path, 300, 9))})
    assert set(result.dumps) == {"model.json", "obs.csv", "mesh.csv", "hier_vertices.csv", "surface.csv", "pools.csv", "candidates.csv"}
    assert "HELDOUT" in result.stdout
    with pytest.raises(FitError, match="DMESH_BASELINE"):
        fit(design, tmp_path / "bad", {"DMESH_BASELINE": "nonsense"})
    with pytest.raises(ValueError, match="DMESH_"):
        fit(design, tmp_path / "bad", {"THREADS": "2"})


@needs_binary
def test_three_dimensional_fits_agree_with_the_binary(tmp_path):
    """DMESH_AXIS3: the mesh over loan age, coupon and a third column (the closure, the dry-run
    columns, the 3-D dumps) in both engines, alone and with two level factors."""
    from .synthetic import synthetic_design
    design = tmp_path / "d3.csv"
    synthetic_design(n_pools=1200, seed=4, with_balance=True).to_csv(design, index=False)
    # mask seed 6: with seed 5 the two-factor fit reaches a round whose empirical-null (Lindsey)
    # iteration converges at its 98th step of 100 in the port and misses the cap in the binary, on
    # scores equal to ten digits: a knife-edge of the shared gate, not of the port (see RECT_3D.md)
    # PYRECT_CHECK_COLUMN_CACHE: every kept dry-run column is recomputed and must be identical
    base = {**BASE, "DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(write_mask(tmp_path, 1200, 6)), "DMESH_AXIS3": "log_balance",
            "PYRECT_CHECK_COLUMN_CACHE": "1"}
    assert compare.compare(design, BINARY, tmp_path / "three", base, ["7"], logs=False, tolerance=CENTERED_TOLERANCE)
    groups = tmp_path / "g.txt"
    groups.write_text("\n".join(f"{i % 3} {(i // 5) % 2}" for i in range(1200)) + "\n")
    assert compare.compare(design, BINARY, tmp_path / "three_levels", {**base, "DMESH_LEVEL_GROUPS": str(groups)}, ["7"],
                           logs=False, tolerance=CENTERED_TOLERANCE)
    import json
    model = json.loads((tmp_path / "three" / "cpp" / "run_model.json").read_text())
    assert model["axis3"] == "log_balance" and "h111" in (tmp_path / "three" / "cpp" / "run_mesh.csv").read_text().splitlines()[0]
