import copy
from collections import Counter
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as E

def split():
    assert not E.DOC.exists() and not E.RAW.exists()
    E.DOC.mkdir(parents=True);E.RAW.mkdir(parents=True)
    E.save(E.RAW/'WORKTREE_BEFORE.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','status','--short','--branch'],text=True),stage=subprocess.check_output(['git','diff','--cached','--name-only'],text=True)))
    old=E.read(E.P.DOC/'SPLIT.json');records=old['train']+old['evaluation'];assert len(records)==len({r['id'] for r in records})==319
    typepath=E.ROOT/'_docs/experiments/pallet_replay_by_type_v1/plastic';ts=E.read(typepath/'SPLIT.json');fit=E.read(typepath/'FIT.json');E.verify(fit['checkpoint']);support=E.read(typepath/'TRAIN_SUPPORT.json')
    assert {r['id'] for r in ts['train']}=={r['id'] for r in old['train'] if r['object_type']=='plastic'}=={r['id'] for r in support['records']}
    assert E.read(typepath/'PROTOCOL.json')['old_mixed_checkpoint_used_as_init'] is False
    for b in E.read(typepath/'INPUT_LOCK.json')['files']:E.verify(b)
    prior=E.P.C.PRIOR_DOC;priorck=E.P.C.PRIOR_CK
    assert E.bind(priorck)['sha256']==E.read(prior/'PRIOR1_COMPLETE.json')['checkpoint_sha256']==E.read(prior/'PRIOR_SELECTION.json')['checkpoints']['1']
    r0p=E.read(prior/'R0_PRETRAINING_PROVENANCE.json');E.verify(r0p['baseline']);assert r0p['baseline']==E.bind(E.C.H.C.N.E.R0)
    groupspath=E.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json';gd=E.read(groupspath)
    aliases={s['session_key']:g['recording_id'] for g in gd['groups'] if not g['is_collection'] for s in g['sessions']}
    parent={g:g for g in set(aliases.values())}
    def root(g):
        while parent[g]!=g:g=parent[g]
        return g
    merges=[]
    for pair in gd['partial_overlap_pairs']:
        a=aliases.get(pair['session_a']);b=aliases.get(pair['session_b'])
        if a and b and root(a)!=root(b):parent[root(b)]=root(a);merges.append(dict(a=a,b=b,reason='existing_partial_overlap'))
    thumbs=[]
    for r in records:
        E.verify(r['image']);E.verify(r['annotation']);session=str(Path(r['image']['path']).parent.parent);assert aliases[session]==r['recording']
        im=cv2.imread(str(E.ROOT/r['image']['path']));thumbs.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32))
    # Same pre-existing near-duplicate rule, before reading model outputs.
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if root(a['recording'])==root(b['recording']):continue
            if a['image']['sha256']==b['image']['sha256'] or np.abs(thumbs[i]-thumbs[j]).mean()<=2.:
                parent[root(b['recording'])]=root(a['recording']);merges.append(dict(a=a['recording'],b=b['recording'],reason='SHA_or_existing_MAD2'))
    c0=ts['train'];trainrecs={root(r['recording']) for r in c0};plast=[dict(r,recording_group=root(r['recording']),old_role='TRAIN' if r['id'] in {x['id'] for x in old['train']} else 'EVAL') for r in records if r['object_type']=='plastic']
    train=[r for r in plast if r['recording_group'] in trainrecs];held=[r for r in plast if r['recording_group'] not in trainrecs]
    assert held and train and not {r['image']['sha256'] for r in train}&{r['image']['sha256'] for r in held}
    pools={};fitids={r['id'] for r in c0};fitsha={r['image']['sha256'] for r in c0}
    for rec in sorted(trainrecs):
        pp=[r for r in train if r['recording_group']==rec and r['id'] not in fitids and r['image']['sha256'] not in fitsha and E.manual(r)[1].any()]
        pools[rec]={label:sorted([r for r in pp if (r['severity']=='CLEAN')==(label=='E')],key=lambda r:E.key(r['id'])) for label in ('E','H')}
    E.save(E.DOC/'SPLIT_LOCK.json',dict(head_before=E.read(E.RAW/'WORKTREE_BEFORE.json')['head'],population319=True,train=train,heldout=held,c0_ids=sorted(fitids),train_recordings=sorted(trainrecs),heldout_recordings=sorted({r['recording_group'] for r in held}),recording_merges=merges,old_eval_reassigned_train=sum(r['old_role']=='EVAL' for r in train),heldout_counts=dict(Counter(r['severity'] for r in held)),recording_disjoint_fitting=True,independent_test=False,history='All are repeatedly viewed reused DEV; selection history is not fitting exposure',teacher_checkpoint=fit['checkpoint'],teacher_parent=E.bind(priorck),teacher_manual_coordinates=support['supported'],R0_provenance=r0p,upstream_caveat='R0 pallet fitting synthetic-only; upstream COCO-pose pretraining exists, not these recordings'))
    E.save(E.RAW/'CANDIDATES_LOCK.json',pools)
    files=[E.P.DOC/'SPLIT.json',groupspath,typepath/'SPLIT.json',typepath/'FIT.json',typepath/'TRAIN_SUPPORT.json',typepath/'PROTOCOL.json',prior/'PRIOR_PROTOCOL_LOCK.json',prior/'PRIOR1_COMPLETE.json',prior/'R0_PRETRAINING_PROVENANCE.json',E.C.DOC/'TARGETS.json',E.C.DOC/'AUGMENTATION_PLAN.jsonl',E.C.DOC/'PREFLIGHT.json',E.C.DOC/'LOSS_TEST.json',E.C.H.C.N.E.R0,priorck,E.ROOT/fit['checkpoint']['path'],Path(E.V.__file__),Path(E.D.Pose.__file__),Path(E.D.Selector.__file__),E.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py']
    files += [E.ROOT/r[k]['path'] for r in records for k in ('image','annotation')]
    E.save(E.DOC/'INPUT_LOCK.json',dict(files=[E.bind(p) for p in sorted(set(files))],split=E.bind(E.DOC/'SPLIT_LOCK.json'),candidates=E.bind(E.RAW/'CANDIDATES_LOCK.json')))
    E.save(E.DOC/'PROTOCOL.md','# Existing-data hard exposure and pseudo-target quality\n\nPlastic only, original R0 each, three fits×320 updates max960; fixed teacher, no new annotation/capture. Recording-disjoint reused DEV, not independent unseen TEST. T0 C0+E pseudo; T1 C0+H pseudo; T2 same H/manual and same support. C0 and replacement each1280 exposures; source2560. Same S1 random rectangular fill policy/area/aspect/frequency, predeclared normalized-box positions shared across paired occurrences; no structural S2 placement or visibility-count selection. TrueIgnore unchanged. Existing R0 includes upstream COCO human pretraining, but no pallet-heldout-recording fitting; teacher synthetic PRIOR1 then plastic Clean10 only. No old EVAL300 score update. Last only, no evaluation during fitting or rescue search.\n')
    print('SPLIT_LOCKED',len(train),len(held),{r:{k:len(v) for k,v in p.items()} for r,p in pools.items()},'merges',merges,flush=True)

@torch.no_grad()
def teacher():
    E.immutable();E.G.deterministic();E.C.guard();split=E.read(E.DOC/'SPLIT_LOCK.json');pools=E.read(E.RAW/'CANDIDATES_LOCK.json')
    detector=YOLO(str(E.C.H.C.N.E.R0),task='pose');refiner=E.P.C.load_model();refiner.load_state_dict(torch.load(E.ROOT/split['teacher_checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict']);refiner.eval()
    pp={};candidates={r['id']:r for pool in pools.values() for rows in pool.values() for r in rows}
    for fid,r in candidates.items():
        im=cv2.imread(str(E.ROOT/r['image']['path']));raw=E.C.predict(detector,im,100);refined=E.P.C.predict(refiner,im,raw,None);cam=E.read(E.ROOT/r['annotation']['path'])['camera_data'];k=cam['intrinsics'];K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]])
        dims,_=E.V.registry_input(E.C.H.C.TYPES['PLASTIC']);initial=E.D.Pose.infer(E.D.points(raw),K,dims[[0,2,1]],False);final,visible,info=E.V.pipeline(raw,refined,initial,K,im.shape[:2]);q=E.D.points(final);manual,mask=E.manual(r)
        available=np.zeros(9,bool) if q is None else np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<im.shape[1])&(q[:,1]>=0)&(q[:,1]<im.shape[0]);available[8]=False
        pp[fid]=dict(record=r,raw=raw,teacher=final,decision=info,manual=np.nan_to_num(manual,nan=-1).tolist(),manual_mask=mask.tolist(),teacher_available=available.tolist(),hw=list(im.shape[:2]))
    E.save(E.RAW/'TRAIN_TEACHER_ONCE.json',pp)
    # Stable greedy within-recording matching, followed by recording round-robin.
    local={};rejected=[]
    for rec,pool in pools.items():
        remaining=list(pool['H']);local[rec]=[]
        for e in pool['E']:
            for h in list(remaining):
                a,b=pp[e['id']],pp[h['id']];mask=np.array(a['manual_mask'])&a['teacher_available']&b['manual_mask']&b['teacher_available']
                if mask.any():local[rec].append(dict(E=e['id'],H=h['id'],mask=mask.tolist(),recording=rec));remaining.remove(h);break
                rejected.append(dict(E=e['id'],H=h['id'],reason='zero_common_valid_channel'))
    pairs=[]
    for i in range(max([len(v) for v in local.values()]+[0])):
        for rec in sorted(local):
            if i<len(local[rec]) and len(pairs)<10:pairs.append(local[rec][i])
    assert pairs,'m=0 — STOP input contract'
    targets=[];oldtargets=E.read(E.C.DOC/'TARGETS.json')
    for r in oldtargets:
        if r['material']=='PLASTIC':targets.append(dict(r,role='C0'))
    supervision=0
    for i,pair in enumerate(pairs):
        for role in ('E','H'):
            a=pp[pair[role]];r=a['record'];c=E.P.C.selected(a['teacher']);assert c is not None
            targets.append(dict(id=r['id'],image=r['image'],annotation=r['annotation'],hw=a['hw'],role=role,pair=i,mask=pair['mask'],target=c['keypoints_xy'],manual=a['manual'],bbox=E.P.C.selected(a['raw'])['box_xyxy'],severity=r['severity']))
        supervision+=sum(pair['mask'])
    E.save(E.RAW/'TARGETS.json',targets)
    E.save(E.DOC/'TARGET_AND_PAIR_AUDIT.json',dict(C0=10,E=len(pairs),H=len(pairs),pairs=pairs,rejected_technical=rejected,teacher_candidates=len(pp),teacher_no_detection=sum(E.P.C.selected(v['teacher']) is None for v in pp.values()),teacher_manual_coordinates=split['teacher_manual_coordinates'],extra_unique_manual_coordinates_T2=supervision,selection='sha256(existing-hard-v1+id), within-recording first valid H, recording round-robin; no teacher/manual residual read',H_severity=dict(Counter(pp[p['H']]['record']['severity'] for p in pairs)),no_new_annotation=True,T2_label='EXISTING_MANUAL_SUPERVISION_CONTROL',teacher_filter='unchanged pipeline fallback reasons recorded; no extra rejection or threshold tuning'))
    print('TARGETS_READY',len(pairs),supervision,flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
