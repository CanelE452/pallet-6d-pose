"""Assert measured completion separately from prior/data readiness."""
import ast
import numpy as np
import torch
from env import *
def run():
    verify();required=['CURRENT_RESULT','SOURCE_BINDING','METHOD_FREEZE','POPULATION_AND_METRIC_LOCK','P_VS_R0_PAIRED','D_PROTOCOL_LOCK','D_TRAINING_AUDIT','D_SELECTION','UNIFIED_DEV_RESULTS','P_VS_D_PAIRED','PREDECLARED_PRIOR_BASELINE_PROTOCOL','PRIOR_BASELINE_READINESS','RUNTIME_PROTOCOL','RUNTIME_PANEL','REGRESSION_TESTS','FINAL_STATUS']
    assert all((DOC/(name+'.json')).is_file() for name in required)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    lock=read(DOC/'D_PROTOCOL_LOCK.json');reg=read(DOC/'REGRESSION_TESTS.json');assert reg['PASS'] and reg['smoke_updates']<=32
    for name,h in lock['code'].items():assert sha(HERE/name)==h,name
    audit=read(DOC/'D_TRAINING_AUDIT.json');assert audit['actual_main_updates']==18000 and audit['fits']==3
    for s in (1,2,3):
        c=read(DOC/f'D{s}_COMPLETION.json');p=RAW/f'runs/D{s}/last.pt';assert sha(p)==c['checkpoint_sha256']
        ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['step']==6000 and ck['complete'] and ck['seed']==s
        assert ck['protocol_sha256']==sha(DOC/'D_PROTOCOL_LOCK.json') and ck['baseline_checkpoint_sha256']==C.R0_SHA
        assert len(ck['step_metrics'])==6000 and [r['step'] for r in ck['step_metrics']]==list(range(1,6001))
        assert all(np.isfinite(r['loss']) and r['post_step_parameter_norm']>0 for r in ck['step_metrics'])
        order=np.load(BRAW/f'order_seed{s}.npy');assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()==c['batch_order_sha256']==read(B/'INIT_ORDER_PARITY.json')['seeds'][str(s)]['batch_order_sha256']
        inf=read(DOC/f'D{s}_INFERENCE_AUDIT.json');assert inf['PASS'] and inf['actual_forwards']==3008
        assert sha(RAW/f'evaluation/D{s}/IMAGE_PREDICTIONS.json')==inf['image_predictions_sha256']
    sel=read(DOC/'D_SELECTION.json');assert sel['no_real_selection'] and sel['temperature'] is None and sel['rule_count']==10
    assert set(sel['checkpoints'])=={'1','2','3'}
    m=read(DOC/'UNIFIED_DEV_RESULTS.json');assert len(m['methods'])==10 and m['common_support_exact']
    for name,v in m['methods'].items():
        assert v['matched_frames']==311 and v['supervised_points']==2756 and v['gt_denominator']==2818
        assert v['pose']['n']+len(v['pose']['excluded_frame_ids'])==319
        if not name.startswith('D'):assert v['pose']['n']==319
    errors=read(RAW/'DEV_FULL_PRECISION_ERRORS.json')
    for name,rr in errors.items():
        e=np.array([e for r in rr for e in r['errors']]);assert abs(np.median(e)-m['methods'][name]['median_px'])<1e-12
        for t in (5,10,20):assert abs((e<=t).sum()/2818-m['methods'][name]['ALL_GT_PCK'][str(t)])<1e-12
    for f in ('P_VS_R0_PAIRED','P_VS_D_PAIRED'):
        r=read(DOC/(f+'.json'))['session'];assert r['resamples']==10000 and r['units']==13
    runtime=read(DOC/'RUNTIME_PANEL.json');assert runtime['complete'] and runtime['accuracy_parity'];assert len(runtime['records'])==10*26*5
    assert not runtime['before']['foreign_compute'] and not runtime['after']['foreign_compute']
    assert read(DOC/'CONFIRMATION_READINESS.json')['available_new_confirmed_frames']==0
    assert read(DOC/'PRIOR_BASELINE_READINESS.json')['full_training']==0
    assert read(DOC/'FINAL_STATUS.json')['PAPER']=='NEEDS_BOTH'
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    historical_changes=subprocess.check_output(['git','diff','--name-only','--','_docs/experiments/pallet_final_ml_contribution_test_v1','scripts/research/pallet_final_ml_contribution_test_v1','scripts/research/pallet_line_pose_v1','_docs/experiments/pallet_task_risk_active_v1','_docs/experiments/pallet_active_learning_v1','_docs/experiments/pallet_transfer_replay_control_v1','_docs/paper/final'],text=True).strip()
    assert not historical_changes,historical_changes
    code={p.name:sha(p) for p in HERE.glob('*.py')}
    write(DOC/'RAW_EVIDENCE_MANIFEST.json',dict(checkpoints=[bound(RAW/f'runs/D{s}/last.pt') for s in (1,2,3)],
        predictions=[bound(RAW/f'evaluation/D{s}/PREDICTIONS.json') for s in (1,2,3)],
        full_precision=bound(RAW/'DEV_FULL_PRECISION_ERRORS.json'),runtime=bound(DOC/'RUNTIME_PANEL.json'),history_manifest=bound(RAW/'confirmation/HISTORY_IMAGES.json')))
    write(DOC/'FINAL_AUDIT.json',dict(integrity='PASS',measured_execution='D3_FITS_DEV_AND_RUNTIME_COMPLETE',paper_readiness='NEEDS_BOTH',
        preserved_core_sources=len(read(DOC/'SOURCE_BINDING.json')['files']),cache_arrays_size_mtime_unchanged=True,D_updates=18000,smoke_updates=reg['smoke_updates'],D_resumes=sum(r['resumes'] for r in audit['runs']),
        sample_order_exact=True,actual_parameter_updates_all_steps=True,full_precision_PCK_and_median_recomputed=True,canonical_pose_frame_binding_bijective=True,
        detection_contract_actual_D_images=3*3008,all_runtime_samples=1300,code_sha256=code,prior_network_unmeasured=True,independent_data_unavailable=True,
        historical_verdicts_unchanged=True,no_P_R0_L_retraining=True,original_paper_final_unchanged=True,git_publication='verify after final commit/push; unrelated user changes retained'))
    print('FINAL_AUDIT_PASS_NEEDS_BOTH',flush=True)
if __name__=='__main__':run()
