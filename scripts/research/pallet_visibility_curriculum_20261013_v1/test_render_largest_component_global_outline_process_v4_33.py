"""Pure synthetic display/source/metadata gates; no saved image/row/ref IO."""
from copy import deepcopy
from pathlib import Path
import inspect

import numpy as np
import pytest

from . import render_largest_component_global_outline_process_v4_33 as R
from .test_evaluate_largest_component_global_outline_poses_v4_33 import toy
from .test_score_clean_fresh_BI_poses_v4_32 import prediction, LEGACY


def forbidden(*args,**kwargs):
    raise AssertionError('Actual image/reference/row/NN/fit/observation path reached')


def test_ROOT_GOs_and_explicit_all_stage_bindings_precede_IO(monkeypatch):
    monkeypatch.setattr(R.C,'read',forbidden);monkeypatch.setattr(R.C,'sha',forbidden)
    monkeypatch.setattr(Path,'exists',forbidden)
    with pytest.raises(PermissionError):R.declare()
    with pytest.raises(PermissionError):R.render()
    with pytest.raises(ValueError,match='before IO'):
        R.declare(root_metadata_GO=True)


def test_fixed_same_three_cases_old_draw_and_ROI_no_new_error_or_GT_routing():
    assert R.FIXED_IDS==('wood_night_01:031344','wood_night_01:031426','eval_noapril:1775201428535961600')
    assert R.observed_model_ROI is R.ROI.observed_model_ROI
    assert R.final_projection is R.OLD.final_projection
    assert R.OUTPUT!=R.PARENT
    assert R.C.sha(R.OLD.__file__)==R.OLD_RENDERER_SHA256
    assert R.C.sha(R.SCORE.__file__)==R.SCORER_SHA256
    source=inspect.getsource(R.render)
    for text in ('choose_cases(', 'reference_from_document(', 'pose_metric(', 'solve_worker_packet(', 'refine_', 'generate_fresh_', 'observe_union_and_largest_component('):
        assert text not in source
    assert "data['fixed_reference_projection_xy']" in source


def mask_fixture(*,empty=False):
    packet,outline=toy(empty=empty)
    h,w=packet['physics']['image_hw'];mask=np.zeros((h,w),np.float32)
    if not empty:mask[100:250,200:400]=1.;mask[20:25,30:35]=1.
    return packet,outline,mask


@pytest.mark.parametrize('empty',[False,True])
def test_exact_display_pixel_replay_matches_sealed_toy_digests_without_sampler(empty,monkeypatch):
    _,outline,mask=mask_fixture(empty=empty)
    before=mask.tobytes();mask.setflags(write=False)
    monkeypatch.setattr(R.LCC,'observe_union_and_largest_component',forbidden)
    monkeypatch.setattr(R.LCC,'frozen_union_observer',forbidden)
    union,selected,audit=R.verify_display_mask(mask,outline)
    assert mask.tobytes()==before
    assert union.dtype==selected.dtype==bool
    assert np.all(selected<=union)
    assert audit==outline['largest_component']['foreground_component_selection']
    assert selected.sum()==(0 if empty else 30000)
    assert audit['semantic_pallet_membership_proved'] is False


def test_changed_pixels_or_saved_component_audit_cannot_be_silently_redrawn():
    _,outline,mask=mask_fixture()
    wrong=mask.copy();wrong[100,200]=0
    with pytest.raises(ValueError,match='replay'):
        R.verify_display_mask(wrong,outline)
    wrong=deepcopy(outline);wrong['original_union']['mask_array_sha256']='0'*64
    with pytest.raises(ValueError,match='raster'):
        R.verify_display_mask(mask,wrong)


def archived(packet):
    source=packet['clean_input']
    return dict(states=deepcopy(source['states']),original_native_xyz=deepcopy(source['native_xyz']),
        original_Q=deepcopy(source['physical_registration_Q']),K=deepcopy(source['K']),
        own_independent_coarse=deepcopy(source['initial_coarse_pose']),
        original_BI_archives=R.OLD.archived_BI(packet),U_ranges=deepcopy(packet['physics']['u_regions']),
        O_regions=deepcopy(packet['regularizers']['o_regions']))


def test_archived_context_same_original_q_source_cov_state_coarse_and_ranges():
    packet,_,_=mask_fixture();data=archived(packet);before=deepcopy(packet)
    assert R.verify_archived_context(packet,data)==data and packet==before
    data['original_BI_archives'][0]['source']='WRONG'
    with pytest.raises(ValueError,match='archive'):
        R.verify_archived_context(packet,data)


def test_S_and_U_observed_payload_is_never_used_for_context_or_marker():
    packet,_,_=mask_fixture()
    source=packet['clean_input'];source['states'][0]='U';source['states'][6]='S';source['initial_self_hidden'][6]=True
    class StateOnly(dict):
        def __getitem__(self,key):
            if key=='native_corner':return super().__getitem__(key)
            raise AssertionError('S/U point payload read')
        def __deepcopy__(self,memo):raise AssertionError('S/U point payload copied')
    source['visible_measurements']=[record for record in source['visible_measurements'] if record['native_corner'] not in (0,6)]
    source['visible_measurements'] += [StateOnly(native_corner=0),StateOnly(native_corner=6)]
    data=archived(packet)
    context=R.verify_archived_context(packet,data)
    assert {0,6}.isdisjoint(record['native_corner'] for record in context['original_BI_archives'])


def test_ROI_is_unchanged_by_GT_final_pose_or_score_fields():
    data=dict(coarse_projection_xy=[[100.,100.],[130.,120.]],
        original_BI_archives=[dict(xy=[110.,110.])],active_cost_plans={'native':{'active_groups':[dict(actual_inlier_xy=[[115.,105.],[125.,115.]])]}})
    roi=R.observed_model_ROI(data,(480,640))
    changed=deepcopy(data);changed['fixed_reference_projection_xy']=[[0.,0.],[639.,479.]]
    changed['final_projection_xy']={'e':[[1e9,-1e9]]};changed['saved_metrics']={'T':1e99}
    assert R.observed_model_ROI(changed,(480,640))==roi
    assert roi==[40.,190.,53.5,166.5]


def test_final_forward_projection_only_uses_stored_pose_and_known_physical_body():
    packet,_,_=mask_fixture();arm=R.ARMS[1]
    # Genuine stored-pose conversion; no PnP or metric/GT call.
    entry=prediction(arm,family='SWAPPED_WD')
    original=entry['new_geometry_edge_cost_plans']['SWAPPED_WD']['initial_model_ray_binding']['geometry']
    packet['clean_input']['native_xyz']=original['native_xyz']
    packet['clean_input']['physical_registration_Q']=original['physical_registration_Q']
    projected=R.final_projection(packet,entry,arm,LEGACY)
    assert projected.shape==(8,2) and np.isfinite(projected).all()
    none=dict(accepted=False,pose_available=False)
    assert R.final_projection(packet,none,arm,LEGACY) is None


def test_fixed_packet_stream_keeps_all_cases_once_and_refuses_missing_or_duplicate(monkeypatch):
    packets=[]
    for identifier in ('other',*reversed(R.FIXED_IDS)):
        packets.append(dict(identifiers={'id':identifier}))
    monkeypatch.setattr(R.C,'rows',lambda path:iter(packets))
    result=list(R._fixed_packets(Path('SYNTHETIC')))
    assert [row['identifiers']['id'] for row in result]==list(reversed(R.FIXED_IDS))
    packets.append(deepcopy(packets[-1]))
    with pytest.raises(ValueError,match='Duplicate'):list(R._fixed_packets(Path('SYNTHETIC')))
    packets.pop();packets.pop()
    with pytest.raises(ValueError,match='All three'):list(R._fixed_packets(Path('SYNTHETIC')))


def completed_fixture(monkeypatch):
    docs={};hashes={}
    protocol=dict(protected_user_sha256={})
    score=dict(prediction_seal={'path':'SYNTHETIC/PREDICTION_SEAL.json','sha256':'b'*64},
        scoring_protocol_sha256='a'*64,outline_selection_profile=R.LCC.SELECTION_PROFILE,
        native_CAL32_models_relabelled=False,all_rows_kept=True,default='SAME_FROZEN_E unchanged',
        score_sha256={part:'a'*64 for part in (*R.M.PARTITIONS,'PRIMARY')})
    scoring=dict(central_plan_sha256='c'*64,prediction_seal_sha256='b'*64,
        outline_selection_profile=R.LCC.SELECTION_PROFILE,native_CAL32_models_relabelled=False,
        source_sha256={},input_sha256={},protected_user_sha256={})
    parent=dict(cases=[dict(id=identifier,folder='case'+str(i)) for i,identifier in enumerate(R.FIXED_IDS)],
        assets={f'case{i}/{name}':'a'*64 for i in range(3) for name in ('PROCESS_DATA.json','source_rgb.png')})
    docs[R.M.OUTPUT/'SCORE_SEAL.json']=score
    docs[R.M.OUTPUT/'SCORING_PROTOCOL.json']=scoring
    docs[R.PARENT/'RENDER_SEAL.json']=parent
    hashes[R.M.OUTPUT/'SCORE_SEAL.json']='d'*64;hashes[R.PARENT/'RENDER_SEAL.json']=R.PARENT_RENDER_SEAL_SHA256
    monkeypatch.setattr(R,'_pins',lambda:None)
    monkeypatch.setattr(R.SCORE,'_prediction_gate',lambda *args:protocol)
    monkeypatch.setattr(R.M,'_binding',lambda path:dict(path='SYNTHETIC/PREDICTION_SEAL.json',sha256='b'*64))
    monkeypatch.setattr(R.M,'_verify',lambda *args:None)
    monkeypatch.setattr(R.C,'read',lambda path:docs[path])
    monkeypatch.setattr(R.C,'sha',lambda path:hashes.get(Path(path),'a'*64))
    return score,parent


def test_completed_current_score_and_old_sealed_artifacts_are_required_before_rows(monkeypatch):
    score,parent=completed_fixture(monkeypatch)
    _,_,cases,inputs=R._completed('c'*64,'b'*64,'d'*64)
    assert set(cases)==set(R.FIXED_IDS) and len(inputs)==7
    score['native_CAL32_models_relabelled']=True
    with pytest.raises(ValueError,match='fixed-reference'):
        R._completed('c'*64,'b'*64,'d'*64)
    score['native_CAL32_models_relabelled']=False
    parent['cases'].pop()
    with pytest.raises(ValueError,match='Same three'):
        R._completed('c'*64,'b'*64,'d'*64)
