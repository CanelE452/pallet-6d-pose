"""Conditional data-only pilot: same frozen pseudo recipe on both sessions."""
import copy
import gc
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from . import common as C
B=C.B;E=B.E;P=B.P
ARMS=('SINGLE32','MULTI32')
DAY='data/evaluation/pallet_eval_v1/incoming/sessions/real_unlabeled_day_20260830'
NIGHT='data/pallet/raw_data/night/capturenight01'

def tensor_save(path,obj):
    assert path.resolve().is_relative_to(C.RAW);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(obj,f)

def prepare():
    C.protocol();gate=C.read(C.DOC/'DIAGNOSTIC_GATE.json');assert gate['status']=='PROCEED_MATCHED32'
    assert not (C.DOC/'DATA_PREPARATION_PROTOCOL.json').exists(),'Single preparation only'
    day=C.read(E.DOC/'DATA_ROLE_MANIFEST.json');pool=C.read(C.ROOT/'_docs/experiments/pallet_type_selftrain_v1/POOL.json')
    night=[r for r in pool['records'] if r['session']==NIGHT];assert len(night)>=16
    records=[dict(r,K=day['K'],recording_id='REC_001') for r in day['train_candidates']]+night
    assert all(r['object_type']=='plastic_standard_110x130x11' for r in records)
    inv=C.read(C.F.DOC/'DATA_DIVERSITY_INVENTORY.json');inventory={r['session']:r for r in inv['session_inventory']}
    assert not any(inventory[s]['excluded_evaluation_recording'] for s in (DAY,NIGHT))
    assert inventory[DAY]['recording_id']!=inventory[NIGHT]['recording_id']
    assert {r['image']['sha256'] for r in records}.isdisjoint(r['image']['sha256'] for r in C.read(C.DOC/'INPUT_LOCK.json')['eval_records'])
    for binding in C.read(E.DOC/'PSEUDO_PROTOCOL.json')['bindings']:C.verify(binding)
    script_sources=[C.bind(C.HERE/'data.py'),C.bind(Path(P.__file__)),C.bind(Path(E.__file__))]
    C.save(C.DOC/'DATA_PREPARATION_PROTOCOL.json',dict(candidate_count=len(records),sessions=[DAY,NIGHT],
        recipe='Same existing E2: R0 .85/min6kp.5 -> raw flip+LOO .05 -> frozen Replay -> unchanged self-occlusion PnP -> all8LOO.05',
        no_new_geometry=True,pseudo_GT_used=False,teacher_training_day_recording=True,night_teacher_training=False,
        selection='After identical frozen filters and paired-input eligibility, path-sort; 32 midpoint DAY; multi=16 even-index members of single32 +16 midpoint NIGHT. No evaluation error selection.',
        real_corruption='Existing E2 RGB patch augmentation, actual OCC R0 rerun, unchanged clean pseudo target, intersection mask.',
        augmentation_seed=902106,order='300x8 balanced32 repeated75 shuffledseed6401; identical index order across arms',
        training=dict(arms=list(ARMS),init='PRIOR1',seed=1,updates=300,lr=1e-4,optimizer='TFAdam',real_batch=8,source_batch=8,microbatch=2,BN='all frozen',preserve=False),
        screening='>=32 DAY and>=16 NIGHT after identical recipe+pair preparation or STOP. No replacements based on evaluation.',
        known_limits='DAY approximate clean interval, NIGHT clean/occlusion quality unknown. Session change also changes viewpoint/light/background and teacher exposure; not independent confirmation.',
        sources=script_sources,inputs=[C.bind(E.DOC/'DATA_ROLE_MANIFEST.json'),C.bind(C.ROOT/'_docs/experiments/pallet_type_selftrain_v1/POOL.json')]))
    B.L.setup('cuda');E.N.setup();cv2.setNumThreads(1)
    extractor=E.N.E.old('features').FrozenYoloFeatures(E.N.E.R0);model=E.N.load_model();teacher_hash=C.state_hash(model.state_dict())
    accepted=[];decisions=[];rng=np.random.default_rng(902106)
    try:
        for i,r in enumerate(records):
            C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));assert image.shape[:2]==(480,640)
            K=np.array(r['K']);raw=E.raw_infer(extractor,image);candidate=E.P.top(raw);reason='raw_confidence'
            refined=None;s1=None;s2=None;info=None;pair=None;plans=None
            if candidate and candidate['score']>=.85 and E.P.valid_points(candidate)[:8].sum()>=6:
                flip=E.P.top(E.raw_infer(extractor,cv2.flip(image,1)));reason='flip_missing'
                if flip:
                    s1=E.scores(candidate,K,flip);reason='raw_flip_LOO'
                    if E.P.passed(s1,['s_remove','s_flip']):
                        torch.backends.cudnn.allow_tf32=False;refined=E.N.C.predict(model,image,raw);cc=E.P.top(refined)
                        initial=E.pose.infer(cc['keypoints_xy'],K,np.array([1.1,.11,1.3]));q,info=E.V.S.correct(cc,initial,K)
                        cc['keypoints_xy']=q.tolist();E.assert_preserved(raw,refined);reason='refined_missing'
                        if E.P.valid_points(cc,False)[:8].all():
                            s2=E.scores(cc,K);reason='refined_all8_LOO'
                            if E.P.passed(s2,['s_remove']):
                                reason='accepted';mask=E.P.valid_points(candidate)&np.isfinite(q).all(1)&(q>=0).all(1)&(q[:,0]<640)&(q[:,1]<480);mask[8]=False
                                occ,plans=P.augmented(image,candidate,q,mask,rng,r['id']);occraw=E.raw_infer(extractor,occ)
                                pair=P.paired_items(image,occ,raw,occraw,q,mask)
                                if pair is None:reason='no_pair_support'
            row=dict(**r,reason=reason,accepted=reason=='accepted',stage1=s1,stage2=s2,raw=raw,refined=refined,pnp=info)
            C.save(C.RAW/'pseudo_frames'/f'{i:04d}.json',row);decisions.append(dict(id=r['id'],session=r['session'],reason=reason))
            if reason=='accepted':
                for item in pair.values():item['id']=r['id']
                path=C.RAW/'paired_inputs'/f'{i:04d}.pt';tensor_save(path,dict(pair=pair,metadata=dict(id=r['id'],plans=plans,raw=raw,occraw=occraw,
                    target=q.tolist(),target_sha=P.array_sha(q),mask_sha=P.array_sha(pair['OCC']['target_valid']),actual_OCC_R0=True)))
                accepted.append(dict(id=r['id'],session=r['session'],image=r['image'],recording_id=r['recording_id'],pair=C.bind(path),
                    pseudo=C.bind(C.RAW/'pseudo_frames'/f'{i:04d}.json'),support=int(pair['OCC']['target_valid'].sum()),patches=bool(plans)))
            if (i+1)%40==0:B.L.gpu_guard();print('UNIFIED_PSEUDO',i+1,dict(Counter(x['session'] for x in accepted)),flush=True)
        assert C.state_hash(model.state_dict())==teacher_hash
    finally:extractor.close();del model,extractor;gc.collect();torch.cuda.empty_cache()
    dayrows=sorted([r for r in accepted if r['session']==DAY],key=lambda r:r['image']['path'])
    nightrows=sorted([r for r in accepted if r['session']==NIGHT],key=lambda r:r['image']['path'])
    C.save(C.DOC/'POOL_AUDIT.json',dict(candidates=len(records),accepted_by_session=dict(Counter(r['session'] for r in accepted)),
        reasons=dict(Counter(r['reason'] for r in decisions)),teacher_unchanged=True,eval_GT_used=False,decisions=decisions))
    assert len(dayrows)>=32 and len(nightrows)>=16,'Insufficient fixed-filter paired data: no threshold relaxation'
    def midpoint(rr,n):return [rr[((2*i+1)*len(rr))//(2*n)] for i in range(n)]
    single=midpoint(dayrows,32);multi=single[::2]+midpoint(nightrows,16);selected=dict(SINGLE32=single,MULTI32=multi)
    for rr in selected.values():assert len({x['image']['sha256'] for x in rr})==32
    order=np.random.default_rng(6401).permutation(np.tile(np.arange(32),75)).reshape(300,8)
    tensor_save(C.RAW/'REAL_ORDER.pt',torch.tensor(order));source=C.read(B.DOC/'INPUT_LOCK.json')['source_orders'];C.verify(source)
    C.save(C.DOC/'TRAIN_INPUT_LOCK.json',dict(selected=selected,real_order=C.bind(C.RAW/'REAL_ORDER.pt'),source_orders=source,
        unique32_each=True,each_image_exposures=75,overlap_images=16,same_recipe=True,
        masks={a:sum(r['support'] for r in rr) for a,rr in selected.items()},augmented={a:sum(r['patches'] for r in rr) for a,rr in selected.items()}))
    C.save(C.DOC/'TRAIN_PROTOCOL.json',dict(arms=list(ARMS),updates=300,seed=1,init=B.protocol()['base'],lr=1e-4,
        trainable_contract=C.read(B.DOC/'TRAINABLE_CONTRACTS.json')['FULL'],BN='all running/affine frozen',source_corruption_rng=7103,
        real_exposures=2400,source_exposures=2400,real_batch=8,source_batch=8,microbatch=2,preserve=False,
        regularizer=B.protocol()['regularization'],inputs=C.bind(C.DOC/'TRAIN_INPUT_LOCK.json'),final_only=True,no_rescue=True,
        success_screen='MULTI vs SINGLE primary PCK10>=+0.5pp, B3>=SINGLE, BASE-correct loss<=SINGLE, GREEN>=SINGLE-0.5pp, source clean>=SINGLE-1pp. Exploratory one seed, no independent causal proof.',
        data_difference_includes='16 clean DAY examples replaced by NIGHT examples. Natural occlusion/viewpoint/teacher exposure not matched.',
        eval=C.read(C.DOC/'INPUT_LOCK.json')['populations']))
    C.save(C.DOC/'TRAIN_CODE_LOCK.json',dict(protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),files=[C.bind(C.HERE/'train.py'),C.bind(C.HERE/'data.py')]))
    print('PAIRED_32_READY',len(dayrows),len(nightrows),flush=True)

if __name__=='__main__':prepare()
