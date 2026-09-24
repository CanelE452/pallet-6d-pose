"""Validate original pair, every cached input, and unchanged recording split."""
import json
import subprocess
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_clean19_structured_easyhard_v1 import augmentation as A

def main():
    assert not (C.DOC / 'INPUT_BINDINGS.json').exists(), 'Do not overwrite an existing preflight lock'
    for p in (C.DOC, C.RAW, C.OUT):
        p.mkdir(parents=True, exist_ok=True)
    work = dict(head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'], text=True).strip(),
                status=subprocess.check_output(['git','status','--short','--branch'], text=True), time=C.now())
    C.save(C.RAW/'WORKTREE_BEFORE.json', work)
    E = C.E
    files = set()
    for folder in (E.C.DOC, E.DOC, C.ANCHOR, C.CONTRACT):
        files.update(p for p in folder.iterdir() if p.suffix in ('.json','.jsonl','.md'))
    for folder in ('pallet_clean19_structured_easyhard_v1','pallet_existing_data_transfer_v1',
                   'pallet_clean19_pose_mismatch_v1','pallet_clean19_pose_sensitive_diag_v1',
                   'pallet_dim_conditioned_p_v1','pallet_verified_anchor_v1','pallet_011067_corner_contract_v1'):
        files.update((C.ROOT/'scripts/research'/folder).glob('*.py'))
    files.update([Path(E.D.Selector.__file__), Path(E.D.Pose.__file__), E.V.RAW/'INFERENCE_METADATA.json',
                  E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json', E.RAW/'TARGETS.json', E.RAW/'DIAGNOSTICS.json',
                  E.RAW/'REFERENCE_PREDICTIONS.json', E.V.RAW/'PREDICTIONS.json', C.FINAL,
                  C.ROOT/'data/pallet/results/pallet_verified_anchor_v1/keypoints_status_review/ANCHOR_SELECTION.json',
                  E.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json', E.D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json',
                  C.ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py'])
    for b in C.read(E.C.DOC/'PROTOCOL.json')['bindings']:
        C.verify(b); files.add(C.ROOT/b['path'])
    fits = {a: C.read(E.C.DOC/f'FIT_PLASTIC_{a}.json') for a in ('S0','S1')}
    for fit in fits.values():
        for key in ('checkpoint','trace','protocol'):
            C.verify(fit[key]); files.add(C.ROOT/fit[key]['path'])
        assert fit['complete'] and fit['steps'] == 320 and len(fit['epochs']) == 5
    assert fits['S0']['initial_state'] == fits['S1']['initial_state']
    assert fits['S0']['trainable_inventory'] == fits['S1']['trainable_inventory']
    r0 = E.C.H.C.N.E.R0
    r0_binding = next(b for b in C.read(E.C.DOC/'INPUT_BINDINGS.json')['files'] if b['path']==str(r0.relative_to(C.ROOT)))
    C.verify(r0_binding); files.add(r0)
    state = torch.load(r0, map_location='cpu', weights_only=False)['model'].float().state_dict()
    for key, value in state.items():
        if key in fits['S0']['initial_state']:
            assert E.C.digest(value.numpy()) == fits['S0']['initial_state'][key], key
    args = C.read(E.C.DOC/'PREFLIGHT.json')['args']
    assert {k: args[k] for k in ('epochs','batch','nbs','imgsz','seed','optimizer','lr0','lrf','cos_lr')} == dict(epochs=5,batch=16,nbs=16,imgsz=640,seed=42,optimizer='AdamW',lr0=.0001,lrf=.1,cos_lr=True)
    traces = {a: C.read(C.ROOT/fits[a]['trace']['path']) for a in fits}
    plans = [r for r in map(json.loads,(E.C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC']
    targets = [r for r in C.read(E.C.DOC/'TARGETS.json') if r['material']=='PLASTIC']
    assert len(plans)==len(traces['S0'])==len(traces['S1'])==5120 and len(targets)==10
    changed = 0
    for i,r in enumerate(plans):
        assert r['epoch']*1024+r['slot']==i
        C.verify(r['cache']); files.add(C.ROOT/r['cache']['path'])
        with np.load(C.ROOT/r['cache']['path']) as z:
            img=z['img']; kp=z['keypoints']; box=z['bboxes']
        assert E.C.digest(img)==r['base_RGB_sha256'] and E.C.digest(kp)==r['target_sha256'] and E.C.digest(box)==r['box_sha256']
        for a in fits:
            t=traces[a][i]
            assert t['occ']==i and t['real']==r['real']
            for k in ('base_RGB_sha256','target_sha256','box_sha256'):
                assert t[k]==r[k]
            assert t['input_sha256']==E.C.digest(A.apply(img,r['plan'],a))
            assert t['applied']==(a=='S1' and r['plan']['applied'])
        if traces['S0'][i]['input_sha256']!=traces['S1'][i]['input_sha256']:
            assert r['real'] and r['plan']['applied']; changed+=1
        if not r['real']:
            assert not r['plan']['applied']
        if i%1024==0: print('CACHE_TRACE_VERIFIED',i,flush=True)
    assert sum(r['real'] for r in plans)==2560
    assert len({r['image'] for r in plans if not r['real']})==512
    for a in fits:
        p=E.C.RAW/f'EVAL_PLASTIC_{a}.json'
        b=next(b for b in C.read(E.C.DOC/'EVALUATION_PREDICTIONS_LOCK.json')['files'] if b['path']==str(p.relative_to(C.ROOT)))
        C.verify(b); files.add(p)
        j=C.read(p); assert j['checkpoint']==fits[a]['checkpoint'] and not j['GT_input']
        files.add(E.C.RAW/f'DIAGNOSTICS_PLASTIC_{a}.json')
    for b in C.read(E.DOC/'PREDICTION_LOCK.json')['files']:
        C.verify(b); files.add(C.ROOT/b['path'])
    teacher_lock=C.read(C.ROOT/'_docs/experiments/pallet_visible_refine_hidden_pnp_v1/PREDICTION_LOCK.json')
    for k in ('predictions','metadata'):
        C.verify(teacher_lock[k])
    files.add(C.ROOT/'_docs/experiments/pallet_visible_refine_hidden_pnp_v1/PREDICTION_LOCK.json')
    qa=C.read(C.ANCHOR/'METADATA_QA_FINAL.json')
    assert qa['status']=='HUMAN_METADATA_QA_COMPLETE' and C.sha(C.FINAL)==qa['final_reference_sha256']

    split=C.read(E.DOC/'SPLIT_LOCK.json')
    lookup={r['id']:r for r in split['train']+split['heldout']}
    train=[lookup[t['id']] for t in targets]; held=split['heldout']
    assert sorted(t['id'] for t in targets)==split['c0_ids']
    mapping_path=C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
    files.add(mapping_path); mapping=C.read(mapping_path)
    aliases={s['session_key']:g['recording_id'] for g in mapping['groups'] if not g['is_collection'] for s in g['sessions']}
    parent={g:g for g in set(aliases.values())}
    def root(g):
        while parent[g]!=g: g=parent[g]
        return g
    for m in split['recording_merges']: parent[root(m['b'])]=root(m['a'])
    for r in train+held:
        assert aliases[str(Path(r['image']['path']).parent.parent)]==r['recording']
        assert root(r['recording'])==r['recording_group']
        for k in ('image','annotation'):
            C.verify(r[k]); files.add(C.ROOT/r[k]['path'])
    trainrecs=sorted({r['recording_group'] for r in train}); heldrecs=sorted({r['recording_group'] for r in held})
    overlap=sorted(set(trainrecs)&set(heldrecs))
    shas=sorted({r['image']['sha256'] for r in train}&{r['image']['sha256'] for r in held})
    def thumb(r):
        im=cv2.imread(str(C.ROOT/r['image']['path']))
        return cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(float)
    ta=[thumb(r) for r in train]; ha=[thumb(r) for r in held]
    mad=np.array([[np.abs(a-b).mean() for b in ha] for a in ta])
    near=[dict(train=train[i]['id'],heldout=held[j]['id'],MAD=float(mad[i,j])) for i,j in zip(*np.where(mad<=2.))]
    disjoint=not overlap and not shas and not near
    audit=dict(status='RECORDING_DISJOINTNESS_VERIFIED' if disjoint else 'RECORDING_DISJOINTNESS_FAILED',
               train_frames=len(train),heldout_frames=len(held),train_ids=[r['id'] for r in train],heldout_ids=[r['id'] for r in held],
               train_recordings=trainrecs,heldout_recordings=heldrecs,recording_intersection=overlap,image_sha_intersection=shas,
               near_duplicate=dict(rule='existing grayscale64x48 MAD<=2/255; not scene independence guarantee',minimum=float(mad.min()),flagged=near),
               counts=dict(Counter(r['severity'] for r in held)),recording_counts=dict(Counter(r['recording_group'] for r in held)),
               train_recording_counts=dict(Counter(r['recording_group'] for r in train)),split=C.bind(E.DOC/'SPLIT_LOCK.json'),
               independent_test=False,already_viewed=True)
    C.save(C.DOC/'RECORDING_DISJOINT_AUDIT.json',audit)
    C.save(C.DOC/'PAIR_INTEGRITY.json',dict(status='CAUSAL_PAIR_VERIFIED',same_init=True,same_train_images=True,same_targets=True,same_masks=True,
        same_source_replay=True,same_order=True,same_optimizer=True,same_lr=True,same_seed=True,same_updates=True,same_base_RGB=True,
        only_random_occlusion_difference=True,actual_trace_cache_verified=5120,changed_real_inputs=changed,real_exposures=2560,source_exposures=2560,
        source_unique_images=512,real_unique_images=10,args=args,checkpoint_rule='last',
        policy='Existing S1 conditional random rectangular occlusion: valid supervised coverage + matched S2 placement feasibility gate; NOT unconstrained random erasing. S2 is not evaluated or trained here.',
        reconstruction='Every S0/S1 actual input digest re-created with frozen cache and original augmentation.apply; labels/masks/boxes unchanged.'))
    C.save(C.DOC/'CHECKPOINT_PROVENANCE.json',dict(checkpoints={a:fits[a]['checkpoint'] for a in fits},R0=r0_binding,original_init_parameters_verified=True,
        fit_bindings={a:C.bind(E.C.DOC/f'FIT_PLASTIC_{a}.json') for a in fits},training_code_verified=True))
    C.save(C.DOC/'REPRODUCTION_STATUS.json',dict(checkpoint_reused=True,new_training_steps=0,reproduction_fits=0,new_raw_inference=0))
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(head_before=work['head'],branch=work['branch'],created_at=work['time'],files=[C.bind(p) for p in sorted(files)]))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f"# Preflight\n\nHEAD `{work['head']}` / {work['branch']}.\n\nS0/S1 last 체크포인트 해시 확인. 신규 학습0. 캐시/실제 trace5120개와 RGB 재구성 모두 통과. 추가 가림으로 바뀐 실사 입력 {changed}/2560.\n\n{audit['status']}; train {trainrecs}, heldout {heldrecs}. 이미지 SHA 겹침 {len(shas)}, MAD2 근접중복 {len(near)}.\n\n기존 변경/미추적 파일은 건드리지 않으며 전체 시작 상태는 private WORKTREE_BEFORE.json에 보존.\n")
    print('PREFLIGHT_COMPLETE',audit['counts'], 'changed',changed,flush=True)

if __name__=='__main__': main()
