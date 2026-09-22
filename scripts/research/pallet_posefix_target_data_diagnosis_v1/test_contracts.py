import ast
import subprocess
import numpy as np
import pytest
import torch
from . import common as C
from .probe import corrupt, controlled_occ, descriptors

@pytest.fixture(scope='module')
def protocol():return C.read(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json')
@pytest.fixture(scope='module')
def rows():return C.read(C.RAW/'TRAIN_CORNER_CENSUS.json')
@pytest.fixture(scope='module')
def freeze():return C.read(C.DOC/'PREDICTIONS_FROZEN.json')

def test_current_head():assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==C.read(C.DOC/'INPUT_BINDINGS.json')['HEAD']
def test_accepted253_exact():assert len(C.read(C.B.E.RAW/'PSEUDOLABEL_MANIFEST.json'))==253
def test_candidate264_exact():assert len(C.read(C.B.E.DOC/'DATA_ROLE_MANIFEST.json')['train_candidates'])==264
def test_real_pair_bindings():
    for r in C.real_records():C.verify(r['old']);C.verify(r['pair'])
def test_original_xy_roundtrip():assert C.read(C.DOC/'PREPARATION_CHECKS.json')['roundtrip_max']<1e-4
def test_clean_R0_preserved(rows):
    for r in rows:np.testing.assert_array_equal(r['clean_input'],C.pair(r['index'])['pair']['CLEAN']['original_points'][r['corner_id']])
def test_OCC_R0_preserved(rows):
    for r in rows:np.testing.assert_array_equal(r['occ_input'],C.pair(r['index'])['pair']['OCC']['original_points'][r['corner_id']])
def test_pseudo_target_preserved(rows):
    for r in rows:np.testing.assert_array_equal(r['pseudo_target'],C.pair(r['index'])['metadata']['target_original'][r['corner_id']])
def test_real_order_exact():
    a=C.read(C.P.DOC/'TRAIN_INPUT_AUDIT.json');C.verify(a['real_order']);o=torch.load(C.ROOT/a['real_order']['path'],weights_only=True);assert o.shape==(300,8)
def test_no_eval_train_overlap():
    r=C.read(C.B.E.DOC/'DATA_ROLE_MANIFEST.json');assert r['eval_image_hash_overlap']==r['eval_recording_overlap']==0
def test_center8_excluded(rows):assert all(0<=r['corner_id']<8 for r in rows)
def test_error_band_boundaries():assert [C.band(x) for x in (0,5,5.01,10,10.01,20,20.01,40,40.01)]==[0,0,1,1,2,2,3,3,4]
def test_strict_threshold_fixed(protocol):assert protocol['strict_threshold']==.025
def test_no_eval_GT_probe_generation(protocol):
    assert protocol['GT_eval_input'] is False
    text=(C.HERE/'probe.py').read_text()
    for name in ('E1_FRAME_METRICS','canonical_errors','truth_for_display_only'):assert name not in text
def test_controlled_corruption_original_coords():
    item=C.pair(0)['pair']['CLEAN'];q=corrupt(item,2,30,90)
    np.testing.assert_allclose(C.D.transform_points(q['points'],np.linalg.inv(q['matrix'])),q['original_points'],atol=1e-4,rtol=0)
def test_controlled_radius_exact():
    x=C.pair(0)['pair']['CLEAN']
    for r in (10,20,30,40):
        for a in (0,90,180,270):assert abs(np.linalg.norm(corrupt(x,3,r,a)['original_points'][3]-x['original_gt'][3])-r)<1e-9
def test_frozen_model_no_parameter_change(freeze):assert all(v['before']==v['after'] for v in freeze['hashes'].values())
def test_no_optimizer_created(freeze):assert freeze['no_optimizer_created'] and freeze['optimizer_steps']==freeze['checkpoint_updates']==0
def test_prediction_freeze_before_DEV_analysis(freeze):
    assert freeze['all_before_DEV_analysis'];p=C.DOC/'PREDICTIONS_FROZEN.json'
    for x in C.DOC.glob('ANALYSIS_START*.json'):assert x.stat().st_mtime>=p.stat().st_mtime
def test_primary_denominator_unchanged():assert C.read(C.DOC/'TRAIN_DEV_DISTRIBUTION.json')['DEV_MATCHED']['n']==659
def test_symmetry_contract():
    b=C.read(C.P.DOC/'INPUT_LOCK.json')['symmetry'];C.verify(b)
    for o in C.read(C.ROOT/b['path'])['objects']:
        for p in o['permutations']:assert sorted(p)==list(range(9)) and p[8]==8
def test_existing_artifacts_unchanged():
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['protected']:C.verify(b)
def test_no_overwrite():
    p=C.DOC/'PURPOSE_AND_PLAN.md';b=C.bind(p)
    with pytest.raises(FileExistsError):C.save(p,'should never be written')
    C.verify(b)
def test_RGB_only_P1_P2():
    e=C.pair(0);x=e['pair']['CLEAN'];y=controlled_occ(e,x,1.25)
    for k in ('box','matrix','points','original_points','target','valid','target_valid'):np.testing.assert_array_equal(x[k],y[k])
def test_probe_count_and_eligibility(protocol,rows,freeze):
    index={(r['index'],r['corner_id']):r for r in rows}
    for s in protocol['selected']:
        r=index[(s['index'],s['corner_id'])];assert r['strict'] and r['clean_error_px']<=10 and r['target_supported_125'] and r['target_supported_150']
    n=len(list(descriptors(protocol)));assert all(v['examples']==n for v in freeze['hashes'].values())
def test_nullable_eight_corner_metric_adapter():
    errors=np.array([1,None,2,3,4,5,6,7],float);valid=np.isfinite(errors);assert len(valid)==8 and np.isfinite(errors[np.flatnonzero(valid)]).all()
