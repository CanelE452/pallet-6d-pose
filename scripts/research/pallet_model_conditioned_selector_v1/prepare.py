from . import common as C

def main():
    assert C.git('branch','--show-current')=='main'
    assert C.git('ls-remote','origin','refs/heads/main').split()[0]==C.git('rev-parse','HEAD')
    if (C.DOC/'INPUT_BINDINGS.json').exists():print('VERIFIED',C.immutable());return
    files=set()
    for folder in (C.HARD,C.OLD,C.ROOT/'_docs/experiments/pallet_recording_disjoint_transfer_v1'):
        files.update(p for p in folder.rglob('*') if p.is_file())
    for module in ('pallet_selector_recovery_v1','pallet_min_hard_ab_v1','pallet_recording_disjoint_transfer_v1','pallet_clean19_pose_mismatch_v1','pallet_single_model_preserve_v1'):
        files.update((C.ROOT/'scripts/research'/module).glob('*.py'))
    files.update([C.ROOT/'challenge/evaluation_v2/pnp_selector.py',C.ROOT/'challenge/evaluation_v2/pose_metrics.py'])
    for f in ('RAW_PREDICTIONS.json','POSE_DECISIONS.json','INFERENCE_INPUTS.json','FRAME_METRICS.json','POSE_METRICS.json','ANCHOR_METRICS.json'):
        files.add(C.HARDRAW/f)
    for f in ('SYNTH_INPUTS.json','SYNTH_RECORDS.json','FEATURES_CLEAN.npz','PREDICTIONS_CLEAN.json','SYNTH_LABELS.npz','GEO_LINEAR.pt'):
        files.add(C.OLDRAW/'stage2_synth_scorer'/f)
    split=C.read(C.OLD/'stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json')
    for k in ('inputs','records','manifest','geometry'):C.verify(split[k]);files.add(C.ROOT/split[k]['path'])
    inputs=C.read(C.ROOT/split['inputs']['path'])
    assert len(inputs)==6144 and split['counts']==dict(TRAIN=4096,VAL=1024,TEST=1024) and split['seed']==20260925
    for r in inputs:C.verify(r['image'])
    rawlock=C.read(C.HARD/'RAW_PREDICTIONS_LOCK.json');pose=C.read(C.HARD/'POSE_DECISIONS_LOCK.json')
    for k in ('predictions','metadata'):C.verify(rawlock[k])
    C.verify(pose['poses']);C.verify(pose['raw_lock'])
    checkpoints={a:rawlock['checkpoints'][C.ARM[a]] for a in C.MODELS}
    for b in checkpoints.values():C.verify(b);files.add(C.ROOT/b['path'])
    oldbase=C.read(C.OLDRAW/'BASE_INPUTS.json');assert checkpoints['S1']==oldbase['checkpoints']['S1'];files.add(C.OLDRAW/'BASE_INPUTS.json')
    oldlock=C.read(C.OLD/'stage2_synth_scorer/SYNTH_PREDICTION_LOCK.json')
    for k in ('features','predictions','feature_contract'):C.verify(oldlock[k])
    assert oldlock['feature_contract']==C.bind(C.OLD/'SELECTOR_FEATURE_CONTRACT.json')
    contract=C.read(C.OLD/'SELECTOR_FEATURE_CONTRACT.json');assert contract['n_features']==94
    files.add(C.ROOT/'_docs/experiments/pallet_clean19_structured_easyhard_v1/FIT_PLASTIC_S1.json')
    held=C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json';files.add(held)
    records=C.read(held)['heldout'];assert len(records)==128
    final=C.ROOT/'data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json';files.add(final)
    qa=C.ROOT/'_docs/experiments/pallet_verified_anchor_v1/METADATA_QA_FINAL.json';files.add(qa)
    assert C.sha(final)==C.read(qa)['final_reference_sha256']
    output=dict(created_at=C.now(),head_start=C.git('rev-parse','HEAD'),branch='main',files=[C.bind(p) for p in sorted(files)],checkpoints=checkpoints,
                old_scorer=C.bind(C.OLDRAW/'stage2_synth_scorer/GEO_LINEAR.pt'),contract=C.bind(C.OLD/'SELECTOR_FEATURE_CONTRACT.json'),
                synth_inputs=split['inputs'],synth_split=C.bind(C.OLD/'stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'),
                images_verified=6144,S1_cache_reusable=True,heldout_records=records,final_reference=C.bind(final),existing_results_immutable=True)
    C.save(C.DOC/'INPUT_BINDINGS.json',output)
    C.save(C.RAW/'SYNTH_INFERENCE_INPUTS.json',inputs)
    C.save(C.DOC/'PROTOCOL_LOCK.json',dict(created_at=C.now(),contract=output['contract'],train=contract['train'],features=contract['features'],
        models=list(C.MODELS),selectors=list(C.SELECTORS),new_scorers=list(C.NEW.values()),matrix=8,split=split['counts'],split_seed=20260925,
        selector_architecture='existing models.Scorer(94,linear=True); shared nn.Linear(94,1)',normalization='own model synthetic TRAIN only; both candidates; floor 1e-6',
        additional_hard_labels=0,keypoint_optimizer_steps=0,real_GT_fit=False,threshold_sweep=False,no_severity_routing=True,
        decision_rules='common.decision: exact predeclared severity deltas, no effect-size cutoff',
        directive_sha256=C.sha('/home/minjae/.codex/attachments/5b6a52be-bbf7-4d15-b6e8-cd3536d9c8cd/pasted-text.txt')))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f'# Preflight\n\nHEAD `{output["head_start"]}`, main/remote equal.\n\n{len(files)} upstream artifacts hash-bound; 6,144 synthetic RGB hashes verified. S1 cache reusable.\n\nFeature contract SHA256 `{output["contract"]["sha256"]}` (current repository truth; document reference is not substituted).\n\nNo keypoint re-fit, annotation, new split, real-GT selector fitting or architecture search. Old selector was trained on pooled S0/S1.\n\nReal HELDOUT128 and synthetic TEST are previously viewed development-heldouts, not independent final tests.\n')
    print('PREFLIGHT',len(files),output['head_start'],flush=True)

if __name__=='__main__':main()
