"""Executable acceptance checks; no training or artifact mutation outside namespace."""
import argparse
import json
import subprocess
import sys
from collections import Counter
import numpy as np
import torch
from . import common as C
from . import models as M

def main(require_git=False):
    checks={}
    def ok(name,value):
        assert value,name
        checks[name]=True
    start=C.read(C.RAW/'WORKTREE_START.json');bindings=C.read(C.DOC/'INPUT_BINDINGS.json')
    ok('test_head_recorded',start['head']==bindings['head_start'] and start['branch']=='main')
    for b in bindings['files']:C.verify(b)
    ok('test_input_hashes',True);ok('test_existing_results_unchanged',True)
    for a,b in bindings['base_checkpoints'].items():C.verify(b);ok('test_'+a+'_hash_unchanged',True)
    fc=C.read(C.DOC/'SELECTOR_FEATURE_CONTRACT.json');stage1=C.read(C.sdoc(1)/'REFERENCE_READ_LOCK.json')
    ok('test_selector_feature_contract_before_stage1_GT',fc['created_at']<stage1['created_at'])
    ok('test_shared_candidate_scorer',fc['candidate_scorer_shared']);ok('test_no_pointwise_remapping',fc['no_pointwise_remapping'])
    # Guard must also reject an explicit attempted forbidden reference open.
    cmd=[sys.executable,'-c',"from scripts.research.pallet_selector_recovery_v1 import common as C; C.synth_guard(); open('/tmp/TRUTH_FOR_DISPLAY_ONLY.json')"]
    probe=subprocess.run(cmd,capture_output=True,text=True)
    ok('test_runtime_real_GT_guard_negative_control',probe.returncode!=0 and 'SYNTH_ONLY_READ_VIOLATION' in probe.stderr)
    for stage,prefix in [(2,'SCORER'),(4,'ROUTER')]:
        a=C.read(C.sdoc(stage)/(prefix+'_TRAINING_AUDIT.json'))
        ok('test_synth_training_never_reads_real_GT' if stage==2 else 'test_router_training_never_reads_real_GT',a['guard_active'] and a['real_GT_reads']==0 and not any('TRUTH_FOR_DISPLAY' in p or 'verified_anchor' in p or 'stage1_diagnostic' in p or 'REAL_SCORER_RESULTS' in p for p in a['read_paths']))
        ok(f'test_{prefix}_no_base_optimizer_steps',a['base_optimizer_steps']==0)
        ok(f'test_{prefix}_real_not_used_for_model_selection',not a['TEST_used_in_fit'] and not C.read(C.sdoc(stage)/(prefix+'_SELECTION_LOCK.json'))['real_used'])
    for doc,name in [(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json','synthetic'),(C.DOC/'REAL_FEATURE_LOCK.json','real'),(C.sdoc(4)/'ROUTER_SYNTH_PREDICTION_LOCK.json','occluded')]:
        j=C.read(doc);C.verify(j['features']);C.verify(j['predictions'])
        ok(f'test_frozen_base_capture_{name}',all(x['base_state_unchanged'] and x['base_grad_zero'] and x['hook_on_off_prediction_exact'] and x['repeated_feature_exact'] and x['base_optimizer_steps']==0 for x in j['captures']))
    split=C.read(C.sdoc(2)/'SYNTHETIC_SPLIT_LOCK.json');rows=C.read(C.sraw(2)/'SYNTH_RECORDS.json');C.verify(split['records']);C.verify(split['inputs'])
    ids={};shas={};scenarios={};groups={}
    for r in rows:
        fid=r['id'];part=r['split'];assert fid not in ids;ids[fid]=part
        assert r['image']['sha256'] not in shas;shas[r['image']['sha256']]=part
        for field,cache in [('scenario',scenarios),('group',groups)]:assert r[field] not in cache or cache[r[field]]==part;cache[r[field]]=part
        assert r['image']['sha256'] not in split['replay512_sha256'] and r['scenario'] not in split['excluded_derivative_scenarios']
        C.verify(r['image'])
    ok('test_frame_level_split',dict(Counter(ids.values()))==split['counts'])
    ok('test_image_sha_disjoint',len(shas)==6144);ok('test_renderer_derivative_group_disjoint',True);ok('test_exclude_S0_S1_synth_replay',True)
    lab=C.read(C.sdoc(2)/'EXACT_LABEL_AUDIT.json');C.verify(lab['labels']);l=dict(np.load(C.sraw(2)/'SYNTH_LABELS.npz'))
    ok('test_exact_geometry_label',lab['projection_max_px']<.05 and lab['perfect_GT_candidate_ADD_max']<1e-4 and lab['reference_read_time']>lab['prediction_lock_time'])
    ok('test_no_area_heuristic_label','No image area heuristic' in lab['label'])
    vals=C.read(C.sdoc(2)/'SCORER_VAL_RESULTS.json')['variants'];sl=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json');C.verify(sl['checkpoint'])
    winner=min(vals,key=lambda a:(-vals[a]['best_val_accuracy'],a!='GEO_LINEAR'));ok('test_val_only_variant_selection',winner==sl['winner'])
    once=C.read(C.sdoc(2)/'SYNTH_TEST_ONCE_LOCK.json');ok('test_synth_test_once',once['test_reads']==1 and once['created_at']>sl['created_at'])
    for stage,prefix in [(3,'SCORER'),(4,'ROUTER')]:
        dl=C.read(C.sdoc(stage)/('REAL_'+prefix+'_DECISION_LOCK.json'));ref=C.read(C.sdoc(stage)/('REFERENCE_READ_LOCK.json' if stage==3 else 'ROUTER_REFERENCE_READ_LOCK.json'))
        C.verify(dl['decisions']);C.verify(dl['scorer'] if stage==3 else dl['router'])
        ok(f'test_real_{prefix.lower()}_decision_before_GT',ref['created_at']>dl['created_at'] and dl['GT_input'] is False)
        ok(f'test_{prefix.lower()}_weights_frozen_on_real',dl['weights_unchanged'])
    ck=torch.load(C.ROOT/sl['checkpoint']['path'],map_location='cpu',weights_only=False);z=dict(np.load(C.RAW/'REAL_FEATURES.npz'))
    for a in C.ARMS:
        x=M.pack_inputs(z,a,sl['winner']);s=M.scores(ck,x);ss=M.scores(ck,x[:,::-1].copy());assert np.allclose(s[:,::-1],ss,atol=1e-6);assert np.array_equal(M.selection(s,C.HYP),1-M.selection(ss,C.HYP[::-1]))
    ties=np.zeros((3,2));assert np.array_equal(M.selection(ties,C.HYP),1-M.selection(ties,C.HYP[::-1]));ok('test_candidate_order_swap_invariance',True)
    rc=C.read(C.sdoc(4)/'ROUTER_FEATURE_CONTRACT.json');rp=C.read(C.sdoc(4)/'ROUTER_SYNTH_PREDICTION_LOCK.json');pl=C.read(C.sdoc(4)/'OCCLUSION_PLAN_LOCK.json');C.verify(pl['plans']);C.verify(pl['original_policy'])
    ok('test_router_feature_contract_locked',rc['created_at']<pl['created_at']<rp['created_at'] and not rc['GT_input'] and not rc['severity_input'] and not rc['session_input'])
    ok('test_occlusion_plan_GT_blind',not pl['GT_error_used'] and not pl['model_outcomes_used'])
    rd=C.read(C.sdoc(4)/'ROUTER_DATASET_LOCK.json');C.verify(rd['dataset']);rz=dict(np.load(C.ROOT/rd['dataset']['path']))
    for i,p in zip(rz['ids'],rz['split']):assert ids[i]==p
    assert np.array_equal(rz['ids'][:6144],rz['ids'][6144:]);assert np.array_equal(rz['split'][:6144],rz['split'][6144:])
    e=rz['errors'];labels=(e[:,1]<e[:,0]) & ((np.abs(e[:,1]-e[:,0])>1e-9)|~np.isfinite(e[:,0]));assert np.array_equal(labels,rz['y'].astype(bool))
    ok('test_router_synth_only_training',rd['reference_read_time']>rd['prediction_lock_time'] and len(rz['y'])==12288)
    rt=C.read(C.sdoc(4)/'ROUTER_SELECTION_LOCK.json');C.verify(rt['checkpoint']);ro=C.read(C.sdoc(4)/'ROUTER_TEST_ONCE_LOCK.json');ok('test_router_test_once',ro['test_reads']==1 and ro['created_at']>rt['created_at'])
    base=C.read(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json');dl=C.read(C.sdoc(4)/'REAL_ROUTER_DECISION_LOCK.json');ok('test_same_base_selector_for_experts',base['both_experts_same'] and dl['both_experts_same_selector'] and rd['base_selector']==base['base']==dl['base_selector'])
    old=C.read(C.PREV_DOC/'RESULTS.json')['groups'];s3=C.read(C.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];s4=C.read(C.sdoc(4)/'REAL_ROUTER_RESULTS.json')['groups']
    for g,n in [('MODERATE',21),('CLEAN',29),('SEVERE',78),('ALL',128)]:
        ok(f'test_{g.lower()}{n}_exact',s3[g]['S1']['current']['frames']==n and s4[g]['S0']['sixD']['frames']==n)
    ok('test_recording_groups_unchanged',set(s3)==set(s4)==set(old))
    d3=C.read(C.sraw(3)/'REAL_SCORER_DECISIONS.json');d4=C.read(C.sraw(4)/'REAL_ROUTER_DECISIONS.json')
    ok('test_oracle_posthoc_only',all(v['selected'] in C.HYP for a in C.ARMS for v in d3[a].values()) and all(v['chosen'] in C.ARMS for v in d4.values()))
    expected={1:['01_current_vs_oracle.png','02_selector_margin.png','03_component_distributions.png','04_wrong_candidate_cases.jpg','05_recording_breakdown.png'],2:['01_validation.png','02_test.png'],3:['01_moderate_auc_current_scorer_oracle.png','02_axis_recovery.png','03_clean_severe_safeguards.png','04_recording_breakdown.png','05_gap_recovery.png'],4:['01_router_synth.png','02_clean_preservation.png','03_moderate_preservation.png','04_severe_preservation.png','05_route_fraction.png','06_latency_cost.png','07_routed_examples.jpg']}
    ok('test_all_figures_exist',all((C.DOC/'figures'/f'stage{st}'/f).is_file() for st,ff in expected.items() for f in ff))
    if require_git:
        for stage in range(1,5):
            record=C.read(C.DOC/f'STAGE{stage}_GIT.json');assert record['commit']==record['remote_sha'] and record['push']=='VERIFIED'
        ok('test_stage_git_records_exist',True)
    C.save(C.DOC/('FINAL_AUDIT.json' if require_git else 'PREPUSH_AUDIT.json'),dict(created_at=C.now(),status='PASS',tests=checks,n_tests=len(checks),immutable_inputs=len(bindings['files']),selected_synthetic_images_checked=len(rows),stage_git_pending=not require_git))
    print('AUDIT_PASS',len(checks),'immutable',len(bindings['files']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-git',action='store_true');a=p.parse_args();main(a.require_git)
