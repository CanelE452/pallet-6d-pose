"""Private driver bindings and state-independent authorization; no real IO."""
from types import SimpleNamespace
import pytest

from . import smoke_geometry_guided_runtime_v4_30 as S


def test_actual_calls_require_GO_before_metadata_io(monkeypatch):
    monkeypatch.setattr(S, '_verify', lambda: pytest.fail('No IO without GO'))
    with pytest.raises(PermissionError):
        S.runtime()
    with pytest.raises(PermissionError):
        S.independent()


def test_runtime_facade_rebinds_every_versioned_dependency_without_parent_mutation(monkeypatch):
    prep, cal = SimpleNamespace(), SimpleNamespace()
    before = dict(S.M.PARENT.__dict__)
    monkeypatch.setattr(S.M, '_modules', lambda: (prep, cal))
    facade, p, c = S._facade()
    assert facade.G is S.M.G and facade.AUTH is S.M.AUTH
    assert facade._problem is S.M._problem and facade._run is S.M._run
    assert facade.ARMS == S.M.ARMS and facade.PREP is p is prep and facade.CAL is c is cal
    assert facade.BOUNDED is S.M.PARENT.BOUNDED and facade.CONTAIN is S.M.PARENT.CONTAIN
    assert all(S.M.PARENT.__dict__[k] is v for k, v in before.items())


def test_independent_driver_uses_frozen_manual_reconstruction_not_new_consumer():
    function = S._clone(S.INDEPENDENT.main, dict(OBS='new-observation-path'))
    assert function.__globals__['A'] is S.INDEPENDENT.A
    assert function.__globals__['A'].recompute_native is S.INDEPENDENT.A.recompute_native
    assert function.__globals__['OBS'] == 'new-observation-path'
    assert S.INDEPENDENT.OBS != 'new-observation-path'
