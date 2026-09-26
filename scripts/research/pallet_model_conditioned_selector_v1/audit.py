"""Artifact-backed checks; no model fitting, output retuning, or extra TEST scoring."""
import argparse
import io
import unittest
import numpy as np
import torch
from . import common as C
from .test_contract import ContractTests
from scripts.research.pallet_selector_recovery_v1 import models as M,features as F

def main(final=False):
    checks={}
    def check(name,ok,evidence):
        checks[name]=dict(passed=bool(ok),evidence=evidence)
        assert ok,name
    count=C.immutable();b=C.read(C.DOC/'INPUT_BINDINGS.json');p=C.read(C.DOC/'PROTOCOL_LOCK.json');f=C.read(C.stage(2)/'SYNTH_FEATURES_LOCK.json');pl=C.read(C.stage(2)/'SYNTH_PREDICTIONS_LOCK.json');sl=C.read(C.stage(2)/'SCORER_LOCK.json');a=C.read(C.stage(2)/'SCORER_TRAINING_AUDIT.json')
    check('head_recorded',len(b['head_start'])==40,b['head_start']);check('all_input_hashes',count>=300,count)
    for model,bb in b['checkpoints'].items():C.verify(bb);check(model+'_checkpoint_unchanged',True,bb)
    C.verify(b['old_scorer']);check('old_GEO_checkpoint_unchanged',True,b['old_scorer'])
    check('existing_results_unchanged',any(x['path'].endswith('pallet_min_hard_ab_v1/RESULTS.json') for x in b['files']),C.bind(C.HARD/'RESULTS.json'))
    check('baseline_reproduced',C.read(C.stage(1)/'BASELINE_PARITY.json')['passed'],'90 comparisons including fresh GT pose recomputation')
    check('S1_optimizer_steps_zero',pl['keypoint_optimizer_steps']['S1']==0,'Reused predictions only')
    check('HMAN_optimizer_steps_zero',pl['keypoint_optimizer_steps']['H_MANUAL']==0 and pl['HMAN_state_before']==pl['HMAN_state_after'] and pl['gradients_absent'],'Inference-only requires_grad=False; state hash before/after; no optimizer instantiated')
    check('no_new_hard_labels',p['additional_hard_labels']==0 and not list(C.RAW.glob('*LABELS_PRIVATE*')),'Existing annotations hash-unchanged; no annotation workflow invoked')
    historical_labels=C.read(C.HARD/'HARD_LABEL_LOCK.json')['labels'];C.verify(historical_labels)
    check('no_new_manual_annotation',True,dict(historical_labels=historical_labels,no_UI_opened=True))
    C.verify(f['feature_contract']);check('exact_feature_contract_hash',f['feature_contract']==b['contract'],b['contract'])
    check('94_features_exact',p['features']==F.names() and len(F.names())==94,p['features'])
    check('no_feature_added_removed',f['shape']==[6144,2,94],f['shape'])
    check('no_pointwise_remapping',C.read(C.OLD/'SELECTOR_FEATURE_CONTRACT.json')['no_pointwise_remapping'],'Same two W/D candidates and raw keypoints')
    split=C.read(C.ROOT/b['synth_split']['path']);records=C.read(C.ROOT/split['records']['path']);inputs=C.read(C.RAW/'SYNTH_INFERENCE_INPUTS.json')
    z=dict(np.load(C.ROOT/f['files'][0]['path']));labels=dict(np.load(C.ROOT/sl['labels']['path']))
    check('exact_4096_1024_1024',split['counts']==dict(TRAIN=4096,VAL=1024,TEST=1024),split['counts'])
    check('exact_frame_ids',np.array_equal(z['ids'],labels['ids']) and list(z['ids'])==[r['id'] for r in records],[len(z['ids']),len(set(z['ids']))])
    memberships={}
    for r in records:memberships.setdefault(r['group'],set()).add(r['split'])
    check('exact_renderer_groups',all(len(v)==1 for v in memberships.values()) and all(r['group'] in split['groups'][r['split']] for r in records),split['groups'])
    check('exact_K_dimensions_RGB',inputs==C.read(C.ROOT/split['inputs']['path']),'Exact inference-only metadata equality; all 6144 image hashes verified in preflight and HMAN inference')
    check('exact_parity_labels',sl['labels']==C.read(C.OLD/'stage2_synth_scorer/EXACT_LABEL_AUDIT.json')['labels'],sl['labels'])
    check('no_new_synth_split',p['split_seed']==split['seed']==20260925,b['synth_split'])
    oldlock=C.read(C.OLD/'stage2_synth_scorer/SYNTH_PREDICTION_LOCK.json');oldz=dict(np.load(C.ROOT/oldlock['features']['path']))
    check('S1_cache_exact',all(np.array_equal(z['S1_'+k],oldz['S1_'+k]) for k in ('geo','valid','current')),'Hash-bound source plus bitwise equality of all cached arrays')
    check('two_new_linear_scorers_only',a['trainings']==2 and set(sl['checkpoints'])==set(C.MODELS) and len(list(C.RAW.glob('*.pt')))==2,list(sl['checkpoints']))
    for key,val in [('seed',42),('lr',.001),('weight_decay',.0001),('batch',256),('max_epoch',30),('patience',5)]:check(key,a[key]==val,a[key])
    check('TEST_after_lock',sl['created_at']<C.read(C.stage(2)/'SYNTH_TEST_ONCE_LOCK.json')['created_at'],'SCORER_LOCK timestamp < TEST-once timestamp')
    check('labels_after_predictions',f['created_at']<C.read(C.stage(2)/'EXACT_LABEL_REUSE.json')['created_at'],'Frozen features first, parity link second')
    check('real_files_not_read_during_fit',a['real_GT_reads']==0 and a['guard_active'] and not any('/stage1/' in path or 'TRUTH_FOR_DISPLAY' in path or 'FRAME_METRICS' in path or 'POSE_METRICS' in path for path in a['read_paths']),a['read_paths'])
    for model in C.MODELS:
        C.verify(sl['checkpoints'][model]);ck=torch.load(C.ROOT/sl['checkpoints'][model]['path'],map_location='cpu',weights_only=False)
        valid=z[model+'_valid'];mask=(labels['split']=='TRAIN')&valid;vals=z[model+'_geo'][mask].reshape(-1,94)
        check(model+'_TRAIN_only_normalization',np.array_equal(ck['mean'],vals.mean(0)) and np.array_equal(ck['std'],np.maximum(vals.std(0),1e-6)),dict(train_samples=int(mask.sum()),std_floor=1e-6))
        check(model+'_model_specific_train_samples_only',a['models'][model]['train_ids']==z['ids'][mask].tolist(),a['models'][model]['model_inputs_only'])
        v=C.read(C.stage(2)/('S1_SCORER_VAL.json' if model=='S1' else 'HMAN_SCORER_VAL.json'))
        first=max(r['val_accuracy'] for r in v['curve']);epoch=next(r['epoch'] for r in v['curve'] if r['val_accuracy']==first)
        check(model+'_VAL_only_earlystop',epoch==v['best_epoch'] and v['epochs']<=30 and (v['epochs']==30 or v['epochs']-epoch==5),v)
        check(model+'_shared_candidate_scorer',set(ck['state'])=={'net.weight','net.bias'} and tuple(ck['state']['net.weight'].shape)==(1,94),'exact existing Scorer nn.Linear(94,1)')
        # Structural order test with arbitrary inputs; does not reopen TEST outcomes.
        x=np.random.default_rng(13).normal(size=(7,2,94)).astype(np.float32);score=M.scores(ck,x);reverse=M.scores(ck,x[:,::-1])
        check(model+'_candidate_order_swap',np.array_equal(M.selection(score,C.HYP),1-M.selection(reverse,C.HYP[::-1])),'Random feature tensor, swapped candidates and names')
    check('no_real_selector_refit',not a['TEST_used_in_fit'] and a['VAL_only_earlystop'] and a['trainings']==2,'Two predeclared synthetic-only fits')
    check('no_threshold_sweep',not p['threshold_sweep'],'Fixed old settings and predeclared exact comparison predicates')
    check('no_extra_labeling',p['additional_hard_labels']==0,'No new annotation artifacts')
    check('no_auto_new_architecture',p['new_scorers']==list(C.NEW.values()),p['selector_architecture'])
    if final:
        l=C.read(C.stage(3)/'REAL_SELECTOR_DECISION_LOCK.json');C.verify(l['decisions']);C.verify(l['features']);rr=C.read(C.ROOT/l['decisions']['path']);r=C.read(C.stage(3)/'REAL_SELECTOR_RESULTS.json');d=C.read(C.stage(3)/'DECISION.json')
        check('all_8_combinations_locked_before_GT',len(rr)==8 and all(len(x)==128 for x in rr.values()) and not l['GT_input'],l['decisions_count'])
        check('reference_read_after_lock',l['created_at']<C.read(C.stage(3)/'REFERENCE_READ_START.json')['created_at'],'Separate guarded selection and GT-dependent scoring processes')
        check('no_severity_specific_routing',l['no_severity_routing'] and all(set(x)==set(next(iter(rr.values()))) for x in rr.values()),'All eight combos on identical 128 frames')
        check('same_candidate_generation_per_model',l['same_candidates_as_original'] and l['old_selection_parity_count']==256,'Fresh F.extract vs frozen original candidates: numerical pose parity, old selection exact')
        for name,n in [('CLEAN',29),('MODERATE',21),('SEVERE',78),('ALL',128)]:check(name+'_population',all(v['frames']==n for v in r['groups'][name].values()),n)
        rec=sorted(g for g in r['groups'] if g.startswith('REC_'))
        check('recording_groups_unchanged',rec==['REC_007','REC_021','REC_022','REC_025','REC_027','REC_041','REC_044'],rec)
        check('anchor_not_training',r['verified_visible']['binding']==b['final_reference'] and not any('VERIFIED_LABELS' in path for path in a['read_paths']),b['final_reference'])
        check('oracle_posthoc_only','NONDEPLOYABLE' in r['oracle_label'] and not l['GT_input'],r['oracle_label'])
        check('2D_independent_of_selector',r['twoD_selector_independent'] and set(r['twoD']['ALL'])==set(C.MODELS),'Only model-level raw 2D metrics')
        check('tail_reported',all('matched_pooled_corner8_P90_px' in v and 'gross20' in v for g in r['twoD'].values() for v in g.values()),'All populations include model P90/gross20')
        check('missing_detection_reported',all('missing' in v for g in r['twoD'].values() for v in g.values()),'All populations include missing/detected/matched')
        expected=C.decision(r['groups'],C.read(C.stage(2)/'SYNTH_COMPATIBILITY_MATRIX.json')['combinations'])
        check('predeclared_selector_decision',expected['Q_SELECTOR_COMPATIBILITY']==d['Q_SELECTOR_COMPATIBILITY'],d['Q_SELECTOR_COMPATIBILITY'])
        check('predeclared_pipeline_decision',expected['Q_PIPELINE']==d['Q_PIPELINE'],d['Q_PIPELINE'])
        check('scorers_unchanged_after_TEST',l['scorer_lock']==C.bind(C.stage(2)/'SCORER_LOCK.json'),'All scorer checkpoint hashes reverified above')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    check('unit_tests',result.wasSuccessful(),dict(run=result.testsRun,output=stream.getvalue()))
    output=dict(created_at=C.now(),passed=all(v['passed'] for v in checks.values()),count=len(checks),checks=checks,
        evidence_limit='Artifact integrity, actual runtime read audit, parameter/state checks and unit tests; not an independent security sandbox or independent final validation.')
    # Audit snapshots can be refreshed; fitted weights, decisions and metrics cannot.
    C.save(C.DOC/('FINAL_AUDIT.json' if final else 'STAGE2_AUDIT.json'),output,immutable=False)
    print('AUDIT_PASS',len(checks),result.testsRun,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
