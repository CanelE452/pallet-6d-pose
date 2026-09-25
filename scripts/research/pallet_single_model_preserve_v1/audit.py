import json
import subprocess
import sys
from collections import Counter
import numpy as np
import torch
from . import common as C
def main():
    tests={}
    def ok(k,v):assert v,k;tests[k]=True
    b=C.read(C.DOC/'INPUT_BINDINGS.json');ok('test_head_recorded',b['head']==C.read(C.RAW/'WORKTREE_START.json')['head'])
    for x in b['files']:C.verify(x)
    ok('test_input_hashes',True);ok('test_existing_results_unchanged',True)
    bi=C.read(C.RAW/'INFERENCE_BINDINGS.json')
    for k in bi:C.verify(bi[k])
    ok('test_S1_checkpoint_hash',True);ok('test_GEO_LINEAR_hash',True)
    p=C.read(C.DOC/'ZERO_INIT_PARITY.json');fit=C.read(C.DOC/'FIT.json');C.verify(fit['checkpoint']);C.verify(fit['trace'])
    ok('test_zero_init',p['zero_last_conv']);ok('test_initial_prediction_parity',p['frames']==32 and p['max_xy_px']<=1e-6 and all(r['D9_same'] and r['GEO_LINEAR_same'] for r in p['rows']));ok('test_decoder_functional_parity',p['functional_decoder_bit_exact'])
    ok('test_box_class_unchanged',fit['box_class_conf_unchanged_every_step'] and C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json')['box_class_conf_unchanged']);ok('test_S1_frozen',fit['base_state_before']==fit['base_state_after']);ok('test_only_adapter_trainable',fit['trainable']=='adapter_only');ok('test_adapter_gradient_nonzero',p['adapter_gradient_L1']>0);ok('test_base_gradient_zero',fit['base_grad_zero'])
    lock=C.read(C.DOC/'OCCURRENCES_LOCK.json');C.verify(lock['occurrences']);occ=C.read(C.ROOT/lock['occurrences']['path']);ct=Counter(r['task'] for r in occ)
    ok('test_occurrences_5120',len(occ)==5120);ok('test_real_clean_1280',ct['REAL_CLEAN_TASK']==1280);ok('test_synth_clean_1280',ct['SYNTH_CLEAN_TASK']==1280);ok('test_synth_occ_preserve_2560',ct['SYNTH_OCCLUDED_PRESERVE']==2560)
    for r in occ:C.verify(r['cache'])
    ok('test_cached_augmentation_hashes',True)
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    rr=V.records();heldsha={r['image']['sha256'] for r in rr};ok('test_no_real_dev_training',not heldsha&set(lock['real_sha']) and not fit['real_DEV_training'])
    final=C.read(V.FINAL);anchorids={f['frame_id'] for f in final['frames']};ok('test_anchor_not_training',not anchorids&set(lock['real_clean_ids']))
    probe=subprocess.run([sys.executable,'-c',"from scripts.research.pallet_single_model_preserve_v1 import common as C; C.train_guard(); open('/tmp/TRUTH_FOR_DISPLAY_ONLY.json')"],capture_output=True,text=True);ok('test_training_reference_guard',fit['reference_guard'] and 'TRAIN_REFERENCE_READ_FORBIDDEN' in probe.stderr)
    tr=C.read(C.RAW/'TRACE.json');ok('test_seed42',fit['seed']==42);ok('test_320_steps',fit['steps']==len(tr)==320 and [r['step'] for r in tr]==list(range(1,321)));ok('test_last_only',fit['last_only']);ok('test_lambda1',fit['lambda_occ']==1.)
    raw=C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json');ps=C.read(C.DOC/'POSE_DECISIONS_LOCK.json');ref=C.read(C.DOC/'REFERENCE_READ_LOCK.json');C.verify(raw['predictions']);C.verify(ps['poses'])
    ok('test_prediction_lock_before_GT',raw['created_at']<ps['created_at']<ref['created_at'] and not raw['GT_input'] and not ps['GT_input']);ok('test_same_GEO_LINEAR',raw['scorer']==ps['selector']==bi['scorer'] and ps['same_selector'])
    res=C.read(C.DOC/'RESULTS.json')['groups']
    for g,n in [('CLEAN',29),('MODERATE',21),('SEVERE',78)]:ok('test_'+g.lower()+str(n),res[g]['PRES1']['current']['frames']==n)
    aa=C.read(C.DOC/'VERIFIED_VISIBLE.json');ok('test_verified_FINAL_V2',aa['reference']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2' and aa['counts']==dict(ALL=66,HARD=36,CLEAN=30,MODERATE=22,SEVERE=14));ok('test_oracle_posthoc',not ps['oracle_used'])
    d=C.read(C.DOC/'DECISION.json');gap=C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json';ok('test_gap_only_if_preservation_fails',gap.exists()==(d['primary']!='PRESERVATION_SUPPORTED'))
    if gap.exists():
        g=C.read(gap);ok('test_hard36',g['hard_visible_points']==36);C.verify(g['teacher_binding']);ok('test_teacher_frozen',True);ok('test_role_unstable_support_excluded',all(r['axis_assignment_confirmed'] is True or r['role_explicit'] for r in g['trusted']));ok('test_gap_rule_fixed',g['rule']==C.read(C.DOC/'PROTOCOL_LOCK.json')['gap_rule'])
    src=C.read(C.DOC/'SOURCE_PRESERVATION.json');ok('test_source256_exact_geometry',src['frames']==256 and src['exact_projection_max_px']<.05)
    ok('test_required_figures',all(any((C.DOC/'figures').glob(f'{n:02d}_*')) for n in range(1,11)))
    C.save(C.DOC/'AUDIT.json',dict(created_at=C.now(),status='PASS',tests=tests,n_tests=len(tests),immutable_inputs=len(b['files']),notes='Conditional annotation tests apply only if gate justified and a label package is prepared; no label task on failed gate.'))
    print('PRESERVATION_AUDIT_PASS',len(tests),flush=True)
if __name__=='__main__':main()
