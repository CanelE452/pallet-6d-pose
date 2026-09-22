import inspect
import json
import numpy as np
import pytest
import torch
from . import common as C
from . import data as D

@pytest.fixture(scope='module')
def audit():return C.read(C.DOC/'TRAIN_INPUT_AUDIT.json')

def test_current_head_recorded():assert len(C.read(C.DOC/'INPUT_LOCK.json')['HEAD'])==40
def test_existing_full_checkpoint_binding():C.verify(C.read(C.DOC/'INPUT_LOCK.json')['FULL_fit']['checkpoint'])
def test_prior1_binding():C.verify(C.protocol()['same_as_FULL']['base'])
def test_crop125_baseline_exact(audit):assert all(r['baseline_tensors_exact'] for r in C.read(C.DOC/'ORIGINAL_COORDINATE_PARITY.json')['real'])

def test_crop150_center_same():
    for b in ([0,0,500,200],[-50,20,140,450],[30,-500,170,150]):
        center=np.array(b).reshape(2,2).mean(0)[None]
        for e in (1.25,1.50):np.testing.assert_allclose(D.transform_points(center,D.matrix(b,e)),[[144,192]],atol=1e-9,rtol=0)

def test_bbox_exact_same():assert all(r['initial_xy_bbox_exact'] for r in C.read(C.DOC/'ORIGINAL_COORDINATE_PARITY.json')['real'])
def test_matrix_roundtrip():
    q=np.random.default_rng(12).normal(size=(9,2))*1000
    for e in (1.25,1.50):
        m=D.matrix([18,33,441,120],e);np.testing.assert_allclose(D.transform_points(D.transform_points(q,m),np.linalg.inv(m)),q,atol=1e-9,rtol=0)

def test_output_support_math():
    for r in C.read(C.DOC/'SUBSET_LOCK.json')['groups']['H163']:
        for a in ('C0','C1'):
            m=np.array(r[a]['matrix']);g=np.array(r[a]['reference_crop']);p=D.transform_points(np.clip(g,[0,0],[284,380])[None],np.linalg.inv(m))[0]
            np.testing.assert_allclose(np.linalg.norm(p-r['reference']),r[a]['minimum_distance_px'],atol=1e-9,rtol=0)

def test_fixed_subsets_exact():
    s=C.read(C.DOC/'SUBSET_LOCK.json')['groups'];old=C.S.lock()
    keys=lambda rr:{(r['id'],r['canonical']) for r in rr}
    assert keys(s['H163'])==keys(old['fixed_hard']);assert keys(s['U29'])==keys(old['fixed_old_unreachable'])
    assert keys(s['N14'])|keys(s['U15'])==keys(s['U29']);assert not keys(s['N14'])&keys(s['U15'])
    assert keys(s['I11'])|keys(s['E3'])==keys(s['N14'])

def test_original_real_xy_same():assert all(r['all_original_xy_exact'] for r in C.read(C.DOC/'ORIGINAL_COORDINATE_PARITY.json')['real'])
def test_original_source_xy_same():assert C.read(C.DOC/'ORIGINAL_COORDINATE_PARITY.json')['source_original_points_GT_exact']
def test_original_corruption_same():
    a=C.read(C.DOC/'SOURCE_CORRUPTION_PARITY.json');assert a['all_300_trace_hashes_exact'] and len(a['steps'])==C.protocol()['same_as_FULL']['updates'];C.verify(a['original_points'])

def test_real_order_same(audit):assert audit['real_order']==C.read(C.B.DOC/'INPUT_LOCK.json')['real_order'];C.verify(audit['real_order'])
def test_source_order_same(audit):assert audit['source_orders']==C.read(C.B.DOC/'INPUT_LOCK.json')['source_orders'];C.verify(audit['source_orders'])
def test_occ_plan_same():assert all(r['occlusion_RGB_plan_exact'] for r in C.read(C.DOC/'ORIGINAL_COORDINATE_PARITY.json')['real'])
def test_target_trust_same(audit):assert audit['semantic_mask_recovered'] and 'original confidence' in audit['mask_rule']
def test_support_recomputed_only_from_crop(audit):
    assert not audit['eval_GT_dependency']
    for r in audit['real_masks']+audit['source_masks']:assert r['lost']==0 and np.all(np.array(r['old_mask'])<=np.array(r['new_mask']))

def test_full_trainables_same():
    m=C.B.model('FULL');assert C.B.A.trainable_contract(m)==C.read(C.B.DOC/'TRAINABLE_CONTRACTS.json')['FULL']
    for module in m.modules():
        if isinstance(module,torch.nn.BatchNorm2d):assert not module.weight.requires_grad and not module.bias.requires_grad

def test_no_eval_gt_training():
    from . import train
    for mod in (D,train):
        text=inspect.getsource(mod)
        for token in ('truth_for_display_only','E1_FRAME_METRICS','canonical_errors'):assert token not in text
    assert not C.read(C.DOC/'TRAIN_INPUT_AUDIT.json')['eval_GT_dependency']

def test_no_eval_during_train():
    from . import train
    assert 'evaluate' not in inspect.getsource(train)

def test_symmetry_contract():
    groups=C.read(C.ROOT/C.read(C.DOC/'INPUT_LOCK.json')['symmetry']['path'])['objects']
    for g in groups:
        for p in g['permutations']:assert sorted(p)==list(range(9)) and p[8]==8
        for p in g['permutations']:
            for q in g['permutations']:assert np.array(p)[q].tolist() in g['permutations']

def test_identity_mapping():
    lock=C.read(C.DOC/'INPUT_LOCK.json');rec={r['id']:r for r in lock['eval_records']}
    sy={g['object_type']:g['permutations'] for g in C.read(C.ROOT/lock['symmetry']['path'])['objects']}
    metrics=C.read(C.F.RAW/'FRAME_METRICS.json')['FULL_PRESERVE']
    for r in C.read(C.DOC/'SUBSET_LOCK.json')['groups']['H163']:assert sy[rec[r['id']]['object_type']][metrics[r['id']]['branch']][r['fixed_channel']]==r['canonical']

def test_no_overwrite():
    with pytest.raises(FileExistsError):C.save(C.DOC/'PROTOCOL.json',{})
    with pytest.raises(AssertionError):C.save(C.S.DOC/'never_write.json',{})

def test_existing_artifacts_unchanged():
    for b in C.read(C.DOC/'INPUT_LOCK.json')['protected']:C.verify(b)

def test_bn_frozen():
    if not (C.DOC/'FIT_C.json').exists():pytest.skip('post-training check')
    assert C.read(C.DOC/'FIT_C.json')['BN_frozen']
    h=C.read(C.DOC/'TRAIN_START.json')['BN_sha']
    assert all(json.loads(s)['BN_sha']==h for s in (C.RAW/'TRACE_C.jsonl').read_text().splitlines())

def test_last_checkpoint_only():
    if not (C.DOC/'FIT_C.json').exists():pytest.skip('post-training check')
    fit=C.read(C.DOC/'FIT_C.json');assert fit['updates']==300 and fit['new_train_runs']==1 and fit['final_only'];C.verify(fit['checkpoint'])
    assert len(list(C.RAW.glob('*last*.pt')))==1

def test_prediction_freeze_before_scoring():
    if not (C.DOC/'PREDICTION_LOCK.json').exists():pytest.skip('post-inference check')
    p=C.read(C.DOC/'PREDICTION_LOCK.json');assert p['all_frozen_before_scoring']
    for x in p['arms'].values():C.verify(x['predictions'])

def test_failed_match_retained():
    if not (C.RAW/'FRAME_METRICS.json').exists():pytest.skip('post-scoring check')
    m=C.read(C.RAW/'FRAME_METRICS.json');ids=C.read(C.DOC/'INPUT_LOCK.json')['populations']['PRIMARY_OCC96']
    old=C.read(C.F.DOC/'REAL_RESULTS.json')['summary']['PRIMARY_OCC96']['FULL']
    for a in C.ARMS:
        assert sum(m[a][i]['corners'] for i in ids)==old['corners']
        assert sum(m[a][i]['matched'] for i in ids)==old['matched']
        assert all(e==800 for i in ids if not m[a][i]['matched'] for e in m[a][i]['errors'])

def test_center8():
    if not (C.DOC/'PREDICTION_LOCK.json').exists():pytest.skip('post-inference check')
    raw=C.read(C.B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0']
    for a,x in C.read(C.DOC/'PREDICTION_LOCK.json')['arms'].items():
        for fid,p in C.read(C.ROOT/x['predictions']['path'])['predictions'].items():
            if fid not in {r['id'] for r in C.read(C.DOC/'INPUT_LOCK.json')['eval_records']}:continue
            assert C.B.E.P.top(p)['keypoints_xy'][8]==C.B.E.P.top(raw[fid])['keypoints_xy'][8]

def test_bbox_score_candidate_pass_through():
    if not (C.DOC/'PREDICTION_LOCK.json').exists():pytest.skip('post-inference check')
    raw=C.read(C.B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0']
    for x in C.read(C.DOC/'PREDICTION_LOCK.json')['arms'].values():
        for fid,p in C.read(C.ROOT/x['predictions']['path'])['predictions'].items():C.B.E.assert_preserved(raw[fid],p)
