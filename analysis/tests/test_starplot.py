"""A star page from a fit through the Python port: the tool reads the rectangular dumps (the binary
writes the same files), draws every picker's choice, and its card agrees with the vertex table."""
import numpy as np
import pandas as pd

from analysis.starplot import Run


def test_star_pages_from_a_port_fit(tmp_path):
    from pyrect import fit
    from pyrect.tests.synthetic import synthetic_design
    design = tmp_path / "design.csv"
    synthetic_design(n_pools=1500, seed=11).to_csv(design, index=False)
    rng = np.random.default_rng(12)
    mask = tmp_path / "mask.txt"; mask.write_text("\n".join("1" if t else "0" for t in rng.random(1500) < 0.8) + "\n")
    result = fit(design, tmp_path / "run" / "run", {"DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(mask)})
    run_dir = result.dumps["mesh.csv"].parent
    run = Run(run_dir, design)
    ranked = run.rank()
    assert len(ranked) > 0, "no gate admission to draw"
    v = pd.read_csv(run_dir / "run_hier_vertices.csv").set_index("id")
    for pick in ("max_share", "keff_share"):
        vid = int(ranked.sort_values(pick).iloc[-1].id)
        c = run.page(vid, tmp_path / f"{pick}.png", show_held_out=True)
        assert (tmp_path / f"{pick}.png").stat().st_size > 10_000
        assert abs(c["surplus_logit"] - v.loc[vid, "surplus"]) < 1e-9
        assert c["pools"] > 0 and np.isfinite(c["data_move"])
        # the hat is the coefficient's column: one at the vertex, and within [0, 1]
        pools = run.star_pools(vid)
        assert pools.hat.max() <= 1 + 1e-9 and pools.hat.min() > 0
    # the rectangular facts: every admission recorded its move and the far ends of its cut, which
    # lie on the cut (across the segment, through the vertex); a rival for the same cells sits on
    # their other midline, level with the midpoint of a chord half
    admitted = v[v.gate_admitted == 1]
    assert (admitted.admit_move >= 0).all()
    assert ((admitted.admit_move == 1) == (admitted.admit_splits == 0)).all()
    rivals = 0
    for vid, u in admitted.iterrows():
        across = "y" if u.axis == 0 else "x"; along = "x" if u.axis == 0 else "y"
        for end in (int(u.admit_chord_lo), int(u.admit_chord_hi)):
            if end < 0: continue
            assert abs(v.loc[end, along] - u[along]) < 1e-12 and v.loc[end, across] != u[across]
        g = run.geometry(vid, run.star_pools(vid))
        if "rival" in g:
            rivals += 1
            wala = run.wala_range[0] + run.months(np.array([v.loc[int(u.parent0), "x"], v.loc[int(u.parent1), "x"]]))
            wac = run.wac_range[0] + run.points(np.array([v.loc[int(u.parent0), "y"], v.loc[int(u.parent1), "y"]]))
            on = wala if u.axis == 0 else wac
            assert np.min(np.abs(on - g["rival"]["at"][0 if u.axis == 0 else 1])) < 1e-9
    assert rivals > 0


def test_three_dimensional_pages_from_a_port_fit(tmp_path):
    """A 3-D run (a log_balance axis): the star page with its two slices and the grid figure draw,
    and the card agrees with the vertex table."""
    from pyrect import fit
    from pyrect.tests.synthetic import synthetic_design
    from analysis.make_2x2_logodds import main as grid
    design = tmp_path / "design3.csv"
    d = synthetic_design(n_pools=1500, seed=11, with_balance=True); d.to_csv(design, index=False)
    rng = np.random.default_rng(12)
    mask = tmp_path / "mask.txt"; mask.write_text("\n".join("1" if t else "0" for t in rng.random(1500) < 0.8) + "\n")
    result = fit(design, tmp_path / "run" / "run", {"DMESH_SPLIT": "1", "DMESH_SPLIT_MASK": str(mask), "DMESH_AXIS3": "log_balance"})
    run_dir = result.dumps["mesh.csv"].parent
    run = Run(run_dir, design)
    assert run.dim == 3
    ranked = run.rank()
    assert len(ranked) > 0
    v = pd.read_csv(run_dir / "run_hier_vertices.csv").set_index("id")
    vid = int(ranked.sort_values("max_share").iloc[-1].id)
    c = run.page3d(vid, tmp_path / "star3.png", show_held_out=True)
    assert (tmp_path / "star3.png").stat().st_size > 10_000
    assert abs(c["surplus_logit"] - v.loc[vid, "surplus"]) < 1e-9
    grid([str(design), str(run_dir / "run"), "3-D", str(tmp_path / "grid3.png")])
    assert (tmp_path / "grid3.png").stat().st_size > 10_000
