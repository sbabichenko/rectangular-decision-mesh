from types import SimpleNamespace

import numpy as np
import pytest

from pyrect.gate import LfdrResult, lindsey_lfdr, _damped_poisson_retry


# Training-only candidate histogram, May 2020 seed 2 round 15. Full-step IRLS
# oscillated for 100 iterations. Counts, rather than outcomes, pin the solver.
COUNTS = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          2,6,8,4,13,18,23,33,47,44,41,45,52,47,23,28,22,16,11,7,
          8,5,3,2,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


def test_failed_full_steps_are_recovered_without_changing_density_objective():
    mids = -9.5 + (np.arange(72)+.5)*19/72
    result = lindsey_lfdr(np.repeat(mids, COUNTS).tolist(), True)
    assert result.used_lindsey and result.status == 'valid'
    assert 100 < result.iterations <= 200
    assert result.empirical_sd == pytest.approx(1.1474031327320064, abs=1e-8)
    B = np.vander((mids-mids.mean())/mids.std(ddof=1), 7, increasing=True)
    beta, _ = _damped_poisson_retry(B.tolist(), COUNTS)
    beta = np.asarray(beta[:7]); mu = np.exp(B@beta)
    score = B.T@(np.asarray(COUNTS)-mu)-1e-8*beta
    H = B.T@(mu[:,None]*B)+1e-8*np.eye(7)
    assert score@np.linalg.solve(H,score) < 1e-9
    assert abs(mu.sum()/sum(COUNTS)-1) < 1e-7


def test_degenerate_histogram_is_still_refused():
    result = lindsey_lfdr([0.]*100, True)
    assert not result.used_lindsey
    assert result.lfdr == [1.]*100


@pytest.mark.parametrize('status,admitted', [('nonconverged',0), ('invalid_density',0),
                                          ('mass_out_of_range',0), ('invalid_central_fit',100)])
def test_gate_defers_numerical_failure_but_preserves_structural_fallback(monkeypatch,status,admitted):
    """A numerical failure of the empirical null admits nothing (the round stalls and refits);
    a converged density without a central peak falls back to direct BH."""
    import pyrect.refine as R
    frontier = [SimpleNamespace(vertex=-1)] * 100
    mesh = SimpleNamespace(leaf_count=lambda: 1, recompute_tau_sq=lambda: None, frontier=lambda: frontier)
    result=LfdrResult(); result.status=status
    monkeypatch.setattr(R,'lindsey_lfdr',lambda *a:result)
    decisions=[]
    monkeypatch.setattr(R,'report_calibration',lambda rnd, scored, lfdr_result, fallback, kk, engine: decisions.append(kk))
    monkeypatch.setattr(R,'admit_selected_candidates',lambda *a:None)
    class Engine:
        def refit(self, mesh): pass
        def score(self, frontier, context):
            for q in range(100): context.push(q, 5., 1., None)
    monkeypatch.setattr(R,'K_MAX_ROUNDS',1)
    R.run_gate_rounds(mesh, 1000, Engine(), R.CandidateCensus())
    assert decisions==[admitted]
