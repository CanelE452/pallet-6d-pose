"""Private runtime wiring/storage/version contracts, no real observations."""
from copy import deepcopy
import gzip
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from . import evaluate_geometry_guided_edges_v4_30 as M
from .test_geometry_guided_edge_problem_v4_29 import fixture


def test_private_clone_keeps_defaults_and_parent_namespace():
    source = M.PARENT._models
    before = dict(source.__globals__)
    clone = M._clone(source, dict(G=SimpleNamespace(marker='new')))
    assert clone.__globals__['G'].marker == 'new'
    assert all(source.__globals__[key] is value for key, value in before.items())
    assert clone.__defaults__ == source.__defaults__
    assert clone.__kwdefaults__ == source.__kwdefaults__


def test_spawn_worker_is_top_level_picklable_and_arms_are_distinct():
    assert pickle.loads(pickle.dumps(M._run)) is M._run
    assert M.ARMS[0] == M.PARENT.ARMS[0] == 'SAME_FROZEN_E'
    assert not set(M.ARMS[1:]) & set(M.PARENT.ARMS[1:])
    assert set(M.POLICY_ARMS) == set(M.G.SOURCE_POLICIES)


def test_storage_compression_changes_no_json_value_or_row_order(tmp_path):
    values = [dict(id='한글:1', q=np.array([1., -2.5]), invalid=float('inf')),
        dict(id='case:2', states=['S', 'U', 'B'], selected=[])]
    old, new = tmp_path / 'old.gz', tmp_path / 'new.gz'
    M.C.save_rows(old, values)
    M._save_rows(new, values)
    assert gzip.decompress(old.read_bytes()) == gzip.decompress(new.read_bytes())
    assert list(M.C.rows(old)) == list(M.C.rows(new))
    with pytest.raises(FileExistsError):
        M._save_rows(new, values)


def test_unavailable_new_noise_abstains_uniformly_with_new_profile_not_old_model(monkeypatch):
    base, _, _, R, t = fixture(partial=True)
    family = dict(name='ORIGINAL_WD')
    prep = SimpleNamespace(native_baseline=lambda task, family: base)
    monkeypatch.setattr(M, '_modules', lambda: (prep, None))
    marker = object()
    monkeypatch.setattr(M.PARENT.OLD.V.PARENT, '_problem', lambda *args: (marker, None))
    task = dict(v29_observation=dict(association_configuration=dict(normal_search_radius_px=43.)))
    models = dict(V29={p: dict(available=False, old_model_substituted=False)
        for p in M.G.SOURCE_POLICIES}, NMS=dict(OLD={}))
    before = deepcopy(models)
    for policy in M.G.SOURCE_POLICIES:
        problem, bundle = M._problem(task, family, models, policy)
        assert bundle['noise_unavailable_abstention'] and not bundle['groups']
        assert bundle['selection_profile'] == M.G.SELECTION_PROFILE
        assert problem.tangent_gamma == 0. and not problem.plan['new_owner_ids']
        assert problem.reference_start_problem is marker
        p = M.G.V._pose_vector(R, t, base.scale)
        assert problem.objective(p)[0] == base.objective(p)[0]
    assert models == before


def test_worker_facade_uses_new_uniform_arms_without_changing_parent(monkeypatch):
    # Globals referenced by this fake frozen body are supplied by the private
    # namespace, just as in the real parent controller; no solver is invoked.
    def frozen_body(task, models):
        return dict(arms=list(ARMS), policies=dict(POLICY_ARMS), factory=_problem)
    monkeypatch.setattr(M.PARENT, '_run', frozen_body)
    result = M._run({}, {})
    assert result['arms'] == list(M.ARMS)
    assert result['policies'] == M.POLICY_ARMS
    assert result['factory'] is M._problem
    assert result['selection_profile'] == M.G.SELECTION_PROFILE
    assert M.PARENT.ARMS != M.ARMS


def test_predict_rejects_wrong_profile_before_private_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(M, '_verify_plan', lambda: {})
    M.C.write(tmp_path / 'PROTOCOL.json', dict(selection_profile='OLD_WINNER'))
    with pytest.raises(ValueError, match='V30 full238'):
        M.predict(tmp_path)


def complete_observation_metadata(folder):
    ids = ['CAL:' + str(i) for i in range(46)] + ['TEST:' + str(i) for i in range(49)] \
        + ['REMAINDER:' + str(i) for i in range(143)]
    protocol = dict(ids=ids, selection_profile=M.G.SELECTION_PROFILE,
        raw_observation_generation=M.G.RAW_OBSERVATION_GENERATION)
    M.C.write(folder / M.OBSERVATION_PROTOCOL, protocol)
    seals = {}
    for name, part in (('CAL', ids[:46]), ('REST', ids[46:])):
        p = folder / (name + '_OBSERVATION_SEAL.json')
        M.C.write(p, dict(rows=len(part), ordered_ids=part))
        seals[name] = M.C.sha(p)
    M.C.write(folder / 'OBSERVATION_SEAL.json', dict(rows=238, ordered_ids=ids,
        protocol_sha256=M.C.sha(folder / M.OBSERVATION_PROTOCOL), partition_seals=seals))
    return ids


def test_full_observation_requires_separate_final_protocol_and_both_partition_seals(tmp_path):
    ids = complete_observation_metadata(tmp_path)
    protocol, seal = M._full_observation_metadata(tmp_path, ids)
    assert protocol['ids'] == seal['ordered_ids'] == ids
    M.C.write(tmp_path / 'CAL_OBSERVATION_SEAL.json', dict(rows=45, ordered_ids=ids[:45]))
    with pytest.raises(ValueError, match='CAL46'):
        M._full_observation_metadata(tmp_path, ids)


def test_old_CAL_protocol_cannot_stand_in_for_uniform_all238_closure(tmp_path):
    ids = complete_observation_metadata(tmp_path)
    # The original CAL protocol is preserved even when a valid final closure
    # exists. Changing only that original file does not rebind the final seal.
    M.C.write(tmp_path / 'PROTOCOL.json', dict(selection_profile='PARENT_CAL_ONLY'))
    assert M._full_observation_metadata(tmp_path, ids)[1]['rows'] == 238
    (tmp_path / M.OBSERVATION_PROTOCOL).unlink()
    with pytest.raises(FileNotFoundError):
        M._full_observation_metadata(tmp_path, ids)
