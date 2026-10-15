import ast
import inspect
from copy import deepcopy
import numpy as np
import pytest

from . import calibrate_geometry_guided_edge_noise_v4_30 as D
from . import calibrate_geometry_guided_edge_noise_v4_29 as PARENT
from . import test_calibrate_geometry_guided_edge_noise_v4_29 as FIXTURE
from .test_geometry_guided_edge_noise_calibration_v4_30 import records

M = D.M


def prepared():
    row = FIXTURE.prepared()
    row.update(observation_generation=M.OBSERVATION_FAMILY,
        raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        noise_population=M.POPULATION, selection_profile=M.SELECTION_PROFILE)
    for profile in row['profiles'].values():
        profile['selection_profile'] = M.SELECTION_PROFILE
    return row


def test_selected_axes_floors_and_profile_validation_preserve_archive_bytes():
    row = prepared(); before = deepcopy(row)
    result = D.validate_prepared_row(row)
    assert row == before and result['selection_profile'] == M.SELECTION_PROFILE
    assert result['noise_population'] == M.POPULATION
    for profile in result['profiles'].values():
        assert profile['selection_profile'] == M.SELECTION_PROFILE
        assert profile['tangents'][0]['original_variance_axis_px2'] == 9.
        assert profile['tangents'][0]['selected_line_tangent_unit_xy'] == [0., 1.]


@pytest.mark.parametrize('mode', ['oldrow', 'raw', 'profile', 'policyprofile', 'population'])
def test_old_or_mixed_profile_metadata_rejected(mode):
    row = prepared()
    if mode == 'oldrow': row = FIXTURE.prepared()
    elif mode == 'raw': row['raw_observation_generation'] = M.OBSERVATION_FAMILY
    elif mode == 'profile': row.pop('selection_profile')
    elif mode == 'policyprofile': row['profiles']['ALL_BI_REPLACE']['selection_profile'] = 'old'
    else: row['noise_population'] = M.PARENT.POPULATION
    with pytest.raises(ValueError): D.validate_prepared_row(row)


def test_actual_fit_still_requires_separate_GO_before_IO(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'read', lambda path: pytest.fail('IO before separate GO'))
    with pytest.raises(PermissionError): D.fit(tmp_path/'absent')


def test_full_driver_numerical_fit_and_LOGO_body_are_parent_exact_except_metadata():
    metadata = {'selection_profile', 'observation_generation',
        'raw_observation_generation', 'noise_population'}
    class RemoveMetadata(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [k for k in node.keywords if k.arg not in metadata]
            return node
    def normalized(fn):
        return ast.dump(RemoveMetadata().visit(ast.parse(inspect.getsource(fn))),
            include_attributes=False)
    assert normalized(D.fit) == normalized(PARENT.fit)
    assert normalized(D._heldout_description) == normalized(PARENT._heldout_description)
    assert normalized(D._fit_or_status) == normalized(PARENT._fit_or_status)


def test_new_source_ancestry_and_central_declaration_paths_bound_without_data_read():
    sources = D._sources()
    for module in (M, M.PARENT, M.CORE, M.CORE.ASSOCIATION):
        relative = str(D.Path(module.__file__).resolve().relative_to(D.C.ROOT))
        assert sources[relative] == D.C.sha(module.__file__)
    assert D.CENTRAL_PLAN.parent.name == 'observed_bi_associated_edges_plan_v4_30'
    assert D.CENTRAL_PLAN_SHA256 == '1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'
    assert D.OUTPUT.name == 'CAL_geometry_guided_edge_noise_v4_30'
    assert D.PREPARED.name == 'geometry_guided_native_evidence_v4_30_checks'


def test_zero_support_no_old_substitution_and_LOGO_uses_new_profile():
    rows = records()
    for row in rows: row['groups'] = []; row['tangents'] = []
    status = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'line', {})
    assert status['model'] is None and not status['old_model_substituted']
    rows = records()
    line = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'line', {}, 'b')
    tangent = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'tangent', {}, 'b')
    report = D._heldout_description(rows, 'b', line, tangent)
    assert line['model']['fit_ids'] == ['0', '1']
    assert line['model']['selection_profile'] == M.SELECTION_PROFILE
    assert report['heldout_supported_line_frames'] == 2
    assert np.isfinite(report['heldout_line_NLL'])
    assert np.isfinite(report['heldout_tangent_NLL'])


def test_central_all_three_maps_are_verified_before_use(monkeypatch):
    central=dict(source_sha256={'source.py':'source-sha'},
        input_sha256={'input.json':'input-sha'},protected_user_sha256={'protected.json':'protected-sha'})
    calls=[]
    def fake_sha(path):
        calls.append(path)
        if path==D.CENTRAL_PLAN:return D.CENTRAL_PLAN_SHA256
        return {D.C.ROOT/'source.py':'source-sha',D.C.ROOT/'input.json':'input-sha',
            D.C.ROOT/'protected.json':'protected-sha'}[path]
    monkeypatch.setattr(D.C,'sha',fake_sha)
    monkeypatch.setattr(D.C,'read',lambda path:deepcopy(central))
    assert D._central_declaration()==central
    assert set(calls)=={D.CENTRAL_PLAN,D.C.ROOT/'source.py',
        D.C.ROOT/'input.json',D.C.ROOT/'protected.json'}


def test_ancestry_map_conflicts_rejected_and_inputs_unchanged():
    one={'same-path':'first','a':'a-sha'};two={'same-path':'different','b':'b-sha'}
    before=deepcopy((one,two))
    with pytest.raises(ValueError,match='ancestry conflict'):D._union_maps(one,two)
    assert (one,two)==before
    assert D._union_maps(one,{'same-path':'first','b':'b-sha'})=={
        'same-path':'first','a':'a-sha','b':'b-sha'}
