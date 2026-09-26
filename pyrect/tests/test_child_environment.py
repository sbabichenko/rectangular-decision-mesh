"""A packaged fit must not inherit a modelling choice from the ambient environment: DMESH_*
settings never pass to a child, and PYRECT_* ones only when operational and asked for."""
from pyrect.api import child_environment


def _base():
    return {"PYRECT_TIMEOUT": "60", "PYRECT_SOMETHING_ELSE": "1", "DMESH_DATA": "/somewhere/design.csv", "PATH": "/usr/bin"}


def test_unknown_pyrect_settings_never_reach_a_child():
    for inherit in (False, True):
        env = child_environment(_base(), inherit_pyrect=inherit)
        assert "PYRECT_SOMETHING_ELSE" not in env


def test_operational_settings_pass_only_when_asked():
    assert child_environment(_base(), inherit_pyrect=True)["PYRECT_TIMEOUT"] == "60"
    assert "PYRECT_TIMEOUT" not in child_environment(_base(), inherit_pyrect=False)


def test_dmesh_is_always_dropped_and_unrelated_kept():
    env = child_environment(_base(), inherit_pyrect=True)
    assert "DMESH_DATA" not in env
    assert env["PATH"] == "/usr/bin"
