"""Synthetic metadata gates and runtime-population ledger; no data/label reads."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import prepare_prepartial_RGB_evidence_v4_26 as P
from .common import finite
from .test_prepartial_RGB_point_preservation_v4_26 import partial_fixture


def pair(source='B_ACTUAL_BOUNDARY_RECOVERY'):
    base, obs, kw, ids, R, t = partial_fixture(source)
    # The shared fixture returns already normalized base.boundary. Recreate
    # an explicit typed producer input so the real constructor can validate
    # it again, exactly as it does for a serialized runtime input.
    boundary=deepcopy(base.boundary)
    k=ids[0]
    m=next(r for r in kw['visible_measurements'] if r['native_corner']==k)
    b=boundary[0]
    b.pop('legacy_B_tangent_prediction_archive',None)
    b.update(replaces_complete_BI_likelihood=True,explicit_complete_BI_likelihood_revocation=True)
    b['association_provenance'].update(original_measurement_source=m['source'],
        original_observed_xy=deepcopy(m['xy']),original_observed_covariance_px2=deepcopy(m['covariance_px2']),
        original_effective_fit_covariance_px2=deepcopy(m['fit_covariance_px2']),
        original_fit_reliability_weight=m['fit_reliability_weight'],
        revocation_reason='no_unique_face_supported_junction',display_coordinate_is_full_2D_measurement=False,
        normal_projected_display_xy=deepcopy(m['xy']))
    geometry = dict(native_xyz=base.xyz.tolist(), physical_registration_Q=base.Q.tolist(),
        K=base.K.tolist(), image_hw=list(base.image_hw),
        initial_self_hidden=kw['initial_self_hidden'], initial_projection_xy=kw['initial_projection_xy'])
    row = dict(id='synthetic:1', partition='CAL', session='synthetic', initial_geometry=geometry,
        states=base.states, visible_measurements=kw['visible_measurements'],
        u_regions=[], domains=[], image_sha256=kw['image_binding']['sha256'],
        chain_observation=obs, junction_proposals=kw['junction_proposals'])
    spec = dict(states=base.states, K=base.K.tolist(), image_hw=list(base.image_hw),
        visible=kw['visible_measurements'], domains=[], config=dict(prior_weight=0., analytic_jacobian=True))
    task = dict(id=row['id'], partition='CAL', session=row['session'], spec=spec,
        states=base.states, initial_self_hidden=kw['initial_self_hidden'], u_regions=[],
        image_sha256=row['image_sha256'], families={'ORIGINAL_WD':dict(native_xyz=base.xyz.tolist(),
            physical_registration_Q=base.Q.tolist(), coarse_pose=dict(R=R.tolist(),t=t.tolist(),
                physical_registration_Q=base.Q.tolist()))}, boundary_merged=boundary,
        new_chain_observation=obs, complete_BI_evidence={'junction_proposals':kw['junction_proposals'],
            'initial_projection_xy':kw['initial_projection_xy']},
        O_localization_abstention={'records':[]})
    # The actual producer/consumer contract crosses JSON, including tuple cfg.
    return json.loads(json.dumps(finite(task))), json.loads(json.dumps(finite(row))), ids


class Forbidden(dict):
    def __getitem__(self, key):
        raise AssertionError('Forbidden coordinate/metadata access: ' + str(key))

    def __deepcopy__(self, memo):
        raise AssertionError('Forbidden coordinate copy')


def test_metadata_uses_restored_runtime_population_and_keeps_inventory_separate():
    task, row, (k, _, _) = pair()
    before = deepcopy((task, row))
    item, meta, modes, sources, totals = P.prepare_one(task, row)
    assert (task, row) == before
    assert item['partition'] == 'CAL' and item['preparation_only_no_reference_decode']
    assert item['replacement_plan']['restored_prepartial_complete_ids'] == [k]
    assert item['authentication_inventory_groups']  # Observed edge remains archived.
    assert item['authenticated_group_bundle']['groups'] == []  # Its sole cost owner was restored.
    assert item['replacement_plan']['active_groups'] == item['authenticated_group_bundle']['groups']
    assert modes['FULL_Q_RETAINED'] == meta['complete_count_after_prepartial_restore']
    assert totals['restored_prepartial_complete'] == 1 and totals['active_groups'] == 0
    assert meta['restored_source_counts'] == {'B_ACTUAL_BOUNDARY_RECOVERY':1}
    assert sources['B_ACTUAL_BOUNDARY_RECOVERY'] == 1
    assert meta['ledger']['restored_owner_old_normal_costs'] == 0
    assert meta['ledger']['hard_guard_partial_record_count'] == 1


def test_exact_fallback_keeps_partial_owned_runtime_edge_and_no_fake_restore():
    task, row, _ = pair(P.AUTH.FALLBACK_SOURCE)
    item, meta, modes, _, totals = P.prepare_one(task, row)
    assert item['replacement_plan']['restored_prepartial_complete_ids'] == []
    assert len(item['authenticated_group_bundle']['groups']) == 1
    assert totals['active_groups'] == 1 and totals['restored_prepartial_complete'] == 0
    assert modes['FULL_Q_RETAINED'] == meta['original_complete_count']
    assert meta['ledger']['hard_guard_partial_record_count'] == 1


@pytest.mark.parametrize('partition',['TEST','REMAINDER'])
def test_nonCAL_partition_rejected_before_spec_or_geometry(partition):
    with pytest.raises(ValueError, match='CAL only'):
        P._validate_row_bindings(dict(partition=partition,spec=Forbidden()),
            dict(partition='CAL',initial_geometry=Forbidden()))


@pytest.mark.parametrize('kind',['S','U'])
@pytest.mark.parametrize('location',['visible','partial'])
def test_S_and_nonBI_gates_precede_payload_comparison_or_copy(kind, location):
    task, row, _ = pair()
    k = next(i for i,s in enumerate(task['states']) if s == 'S')
    if kind == 'U':
        task['states'][k] = row['states'][k] = task['spec']['states'][k] = 'U'
        task['initial_self_hidden'][k] = row['initial_geometry']['initial_self_hidden'][k] = False
    record = dict(native_corner=k,state=kind,xy=Forbidden(),lines=Forbidden())
    (task['spec']['visible'] if location == 'visible' else task['boundary_merged']).append(record)
    with pytest.raises(ValueError,match='initial-nonS BI'):
        P._validate_row_bindings(task,row)


def test_prefix_iterator_never_requests_first_primary_record(monkeypatch):
    def records(_):
        yield dict(partition='CAL',id='a')
        yield dict(partition='CAL',id='b')
        raise AssertionError('TEST/REMAINDER decoded beyond CAL prefix')
    monkeypatch.setattr(P.C,'rows',records)
    assert [r['id'] for r in P._cal_task_prefix('unused',['a','b'])] == ['a','b']
    with pytest.raises(ValueError,match='ordered CAL-only'):
        list(P._cal_task_prefix('unused',['wrong']))


def test_only_frozen_canvas_O_transform_allowed_and_original_archive_exact():
    task, row, (k, _, _) = pair()
    # Move an unused nonS mate to O. No coordinate/source record is queried for O.
    omitted = next(i for i,s in enumerate(task['states']) if s != 'S' and i != k)
    for states in (task['states'],task['spec']['states'],row['states']):states[omitted]='O'
    task['spec']['visible']=[m for m in task['spec']['visible'] if m['native_corner']!=omitted]
    row['visible_measurements']=[m for m in row['visible_measurements'] if m['native_corner']!=omitted]
    h,w=task['spec']['image_hw']
    row['domains']=[dict(native_corner=omitted,state='O',shape='ellipse',center_xy=[100.,100.],
        half_axes_px=[12.,12.],angle_deg=0.,canvas_hw=[h,w],localization_available=True,
        halfplanes=[[-1.,0.,0.],[1.,0.,float(w)],[0.,-1.,0.],[0.,1.,float(h)]])]
    old = deepcopy(row['domains'])
    adapted=P.O_ABSTENTION.abstain_external_o_localization(old,task['states'],[h,w],
        initial_self_hidden=task['initial_self_hidden'])
    task['spec']['domains']=adapted['domains']
    task['O_localization_abstention']['records']=adapted['records']
    P._validate_row_bindings(task,row)
    assert row['domains']==old and row['domains'][0]['shape']=='ellipse'
    task['spec']['domains'][0]['bounds_xyxy'][2]-=1.
    with pytest.raises(ValueError,match='v19 O-domain'):
        P._validate_row_bindings(task,row)


def test_geometry_chain_and_q_mismatches_do_not_silently_change_population():
    for where in ('Q','q','chain','guide'):
        task,row,_=pair()
        if where=='Q':task['families']['ORIGINAL_WD']['physical_registration_Q'][0][0]=.9
        elif where=='q':task['spec']['visible'][0]['xy'][0]+=.25
        elif where=='chain':task['new_chain_observation']['edge_records'][0]['chains'][0]['offset']+=.25
        else:task['complete_BI_evidence']['initial_projection_xy'][0][0]+=.25
        with pytest.raises(ValueError,match='Same frozen'):
            P._validate_row_bindings(task,row)


def test_root_declaration_arm_mismatch_gates_before_archive_decode(tmp_path,monkeypatch):
    reads=[]
    def read(path):
        reads.append(path)
        if len(reads)>1:raise AssertionError('Archive accessed before declaration gate')
        return {'arms':['SAME_FROZEN_E']}
    monkeypatch.setattr(P.C,'read',read)
    with pytest.raises(ValueError,match='seven-arm declaration'):
        P.main(tmp_path/'new',declaration=tmp_path/'declaration.json')
    assert not (tmp_path/'new').exists() and len(reads)==1
