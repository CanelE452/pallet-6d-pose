"""Four fixed refiner-only arms; paired targets, real OCC-R0 inputs, source control."""
import argparse
import copy
import gc
import hashlib
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from . import run as C
from scripts.research.pallet_cad8_occlusion_v1 import augmentation as A
from scripts.research.pallet_posefix_large_error_v1 import train as O
from scripts.research.pallet_posefix_replay_v1 import train as T
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam,expectation


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def augmented(image,candidate,target,mask,rng,fid):
    h,w=image.shape[:2]
    q=np.c_[target/[w,h],np.where(mask,2.,1.)]
    box=np.array(candidate['box_xyxy']);center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    batch=dict(img=torch.from_numpy(image[:,:,::-1].copy().transpose(2,0,1)).float()[None]/255,
        keypoints=torch.from_numpy(q.astype(np.float32))[None],bboxes=torch.tensor([np.r_[center/[w,h],size/[w,h]]],dtype=torch.float32),
        batch_idx=torch.tensor([0]),im_file=['eval_cad__adapter_'+fid+'.png'])
    # Adapter name enables the old recipe's real-image selector, not evaluation membership.
    plans=A.plan(batch,rng);before=batch['keypoints'].clone();A.apply(batch['img'],plans)
    assert torch.equal(before,batch['keypoints'])
    out=np.clip(np.rint(batch['img'][0].numpy().transpose(1,2,0)[:,:,::-1]*255),0,255).astype(np.uint8)
    return out,plans


def paired_items(image,occ,raw,occraw,target,mask):
    clean=C.N.C.prepare_input(image,raw);difficult=C.N.C.prepare_input(occ,occraw)
    if clean is None or difficult is None:return None
    common=mask.copy();transformed=[]
    for item in (clean,difficult):
        y=C.N.C.transform_points(target,item['matrix']).astype(np.float32)
        common &= np.isfinite(y).all(-1)&(y>=0).all(-1)&(y[:,0]<288)&(y[:,1]<384)
        transformed.append(y)
    common[8]=False
    if not common[:8].any():return None
    for item,y in zip((clean,difficult),transformed):
        back=C.N.C.transform_points(y,np.linalg.inv(item['matrix']))
        np.testing.assert_allclose(back[common],target[common],atol=1e-4,rtol=0)
        item.update(target=y,target_valid=common.copy(),original_gt=target.copy(),original_gt_valid=common.copy())
    assert np.array_equal(clean['target_valid'],difficult['target_valid'])
    return dict(CLEAN=clean,OCC=difficult)


def prepare():
    C.N.setup();C.gpu();assert C.read(C.DOC/'E1_PSEUDOLABEL_POOL.json')['pool_gate']
    rows=C.read(C.RAW/'PSEUDOLABEL_MANIFEST.json');extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0)
    rng=np.random.default_rng(A.SEED);accepted=[];excluded=[];locks=[];parity=None
    try:
        for i,r in enumerate(rows):
            dest=C.RAW/'paired_inputs'/f'{r["id"]}.pt'
            assert not dest.exists(),'Paired preparation is immutable; do not partially overwrite'
            C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));raw=r['raw'];q=np.array(C.P.top(r['refined'])['keypoints_xy'])
            original=C.P.top(raw);mask=C.P.valid_points(original)&np.isfinite(q).all(1)&(q>=0).all(1)&(q[:,0]<640)&(q[:,1]<480);mask[8]=False
            if parity is None:
                actual=C.raw_infer(extractor,image);C.V.O.assert_same(actual,raw,'R0 coordinate parity',atol=1e-4);parity=r['id']
            occ,plans=augmented(image,original,q,mask,rng,r['id'])
            occraw=C.raw_infer(extractor,occ)  # NEVER substitute the clean prediction.
            pair=paired_items(image,occ,raw,occraw,q,mask)
            if pair is None:
                excluded.append(dict(id=r['id'],reason='no_OCC_detection_or_common_crop_target'));continue
            c=C.P.top(occraw)
            shift=float(np.max(np.linalg.norm(np.array(c['keypoints_xy'])[:8]-np.array(original['keypoints_xy'])[:8],axis=1)))
            metadata=dict(id=r['id'],image=r['image'],plans=plans,clean_RGB_sha=array_sha(image),occluded_RGB_sha=array_sha(occ),
                raw_prediction=raw,occluded_R0_prediction=occraw,R0_actually_rerun=True,
                input_points_max_change_px=shift,target_original=q.tolist(),mask=pair['CLEAN']['target_valid'].tolist(),
                target_sha=array_sha(q),mask_sha=array_sha(pair['CLEAN']['target_valid']))
            for item in pair.values():item['id']=r['id']
            C.tensor_save(dest,dict(pair=pair,metadata=metadata));accepted.append(metadata);locks.append(C.bind(dest))
            if len(accepted)<=8:
                preview=C.OUT/'augmentation'/f'{r["id"]}.png';preview.parent.mkdir(parents=True,exist_ok=True)
                assert not preview.exists();assert cv2.imwrite(str(preview),np.concatenate([image,occ],axis=1))
            if (i+1)%60==0:print('PAIR_PREPARE',i+1,len(accepted),C.gpu(),flush=True)
    finally:extractor.close();del extractor;gc.collect();torch.cuda.empty_cache()
    assert len(accepted)>=8,('Stop: paired input gate',len(accepted))
    # Uniform fixed RNG order; used unchanged by all four arms, no extra real source slots.
    order=np.random.default_rng(6401).integers(0,len(accepted),size=(300,8))
    C.tensor_save(C.RAW/'REAL_ORDER.pt',torch.from_numpy(order))
    role=C.read(C.DOC/'DATA_ROLE_MANIFEST.json');teacher_sessions=set(C.V.read(C.V.DOC/'DATA_ROLE_MANIFEST.json')['teacher_sessions'])
    evaluation=[r for r in role['eval_records'] if r['session'] not in teacher_sessions]
    ids={r['id'] for r in evaluation}
    populations={name:[fid for fid in p['ids'] if fid in ids] for name,p in role['populations'].items()}
    populations={k:v for k,v in populations.items() if v}
    oldprotocol=C.read(C.N.DOC/'PROTOCOL.json');assert oldprotocol['steps']==300 and oldprotocol['optimizer']['lr']==1e-4
    source_lock=C.read(C.N.DOC/'INPUT_LOCK.json');C.verify(source_lock['orders'])
    C.freeze(C.DOC/'E2_INPUT_LOCK.json',dict(status='[확인]',paired_unique=len(accepted),excluded=excluded,records=accepted,paired_files=locks,
        real_order=C.bind(C.RAW/'REAL_ORDER.pt'),R0_clean_parity_frame=parity,source_orders=source_lock['orders'],
        augmented_images=sum(bool(r['plans']) for r in accepted),all_OCC_inputs_actually_inferred=True,
        unaugmented_25pct_is_intentional=True,source_metadata=C.bind(C.N.DOC/'INPUT_LOCK.json')))
    C.freeze(C.DOC/'E2_PROTOCOL.json',dict(status='[확인]',arms=C.ARMS,seed=1,updates=300,real_batch=8,source_batch=8,microbatch=2,
        real_exposures=2400,optimizer='TFAdam',lr=1e-4,BN='frozen running buffers; affine trainable',final_only=True,
        initialization=role['checkpoint_bindings']['POSEFIX_SYNTH'],
        common_targets='Frozen Replay+selfocclusionPnP original-image coordinates; exact same common crop-supported mask across arms',
        common_target_adaptation='Clean/OCC crop intersection fixed before training, no GT. Crop transforms differ but inverse targets agree within1e-4px.',
        augmentation='Unchanged cad8 recipe on original 640x480 RGB, same seed902106; one fixed variant per unique real frame. File prefix adapter only enables recipe for these real candidates.',
        normal_vs_occ='A00/A10 clean RGB+cleanR0; A01/A11 augmentedRGB+actual augmentedR0. 75% patch probability; same y*.',
        source='Existing verified source ORDERS and normal/stress corruption RNG7103, heldout256; source loss absent in A00/A01, not replaced by real samples.',
        compute_matched=False,eval_records=evaluation,populations=populations,teacher_sessions_excluded=sorted(teacher_sessions),
        no_GT_in_training_or_inference=True,no_eval_filtering=True,headroom_only_prior_selection=False,
        reference_gate='Occlusion prior best PCK10 amongN2/N3/Replay; tie priorityN2. Clean vsN2. Source vsinitialsyntheticPoseFix.',
        input_lock=C.bind(C.DOC/'E2_INPUT_LOCK.json'),source_protocol=C.bind(C.N.DOC/'PROTOCOL.json'),code=C.bind(Path(__file__))))
    print('PAIRS_READY',len(accepted),'augmented',sum(bool(r['plans']) for r in accepted),'eval',len(evaluation),{k:len(v) for k,v in populations.items()},flush=True)


def inputs():
    lock=C.read(C.DOC/'E2_INPUT_LOCK.json');rows=[]
    for b in lock['paired_files']:C.verify(b);rows.append(torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False)['pair'])
    C.verify(lock['real_order']);order=torch.load(C.ROOT/lock['real_order']['path'],weights_only=True).numpy()
    return rows,order


@torch.no_grad()
def source_probe(model,held):
    # Original probe exactly, with additional PCK20 and gross20 metrics.
    result={};rng=np.random.default_rng(7104);model.eval()
    for mode in ('clean','stress'):
        rows=held if mode=='clean' else [O.corrupted(x,rng,True) for x in held]
        before=[];after=[];frame_metrics=[]
        for i in range(0,len(rows),2):
            xs=rows[i:i+2];b=O.batch(xs);q=expectation(model(b['rgb'],b['points'],b['valid'])).cpu().numpy()
            for x,p in zip(xs,q):
                m=x['target_valid'];scale=x['matrix'][0,0]
                a=np.linalg.norm(x['points'][m]-x['target'][m],axis=1)/scale
                z=np.linalg.norm(p[m]-x['target'][m],axis=1)/scale;before.extend(a.tolist());after.extend(z.tolist())
                frame_metrics.append(dict(id=x['id'],mean_px=float(z.mean()) if len(z) else None,errors=z.tolist()))
        a=np.array(before);z=np.array(after);good=a<5;hard=a>20
        result[mode]=dict(corners=len(z),PCK10=float(np.mean(z<=10)),PCK20=float(np.mean(z<=20)),median_px=float(np.median(z)),
            P90_px=float(np.quantile(z,.9)),gross20_count=int((z>20).sum()),gross20_rate=float((z>20).mean()),
            good=int(good.sum()),damaged=int((good&(z>10)).sum()),hard=int(hard.sum()),recovered=int((hard&(z<=10)).sum()),
            frames=frame_metrics,scope='Matched supported fixed heldout256, fixed index, preparedRGB pixels; not new full detection benchmark')
    return result


def train(arm):
    C.N.setup();C.gpu();protocol=C.read(C.DOC/'E2_PROTOCOL.json');C.verify(protocol['input_lock']);C.verify(protocol['initialization']);C.verify(protocol['code'])
    fitfile=C.DOC/f'FIT_{arm}.json'
    if fitfile.exists():C.verify(C.read(fitfile)['checkpoint']);print('REUSE_COMPLETED',arm);return
    assert not (C.RAW/f'{arm}_last300.pt').exists(),'Do not overwrite incomplete final checkpoint'
    real,order=inputs();source=SourceData();old=np.load(C.N.RAW/'ORDERS.npz')
    held=[source.item(int(i)) for i in old['held_rows']]
    assert not set(old['held_rows'])&set(old['source_rows'].ravel())
    if not (C.DOC/'SOURCE_BEFORE.json').exists():
        before=C.N.C.load_model();stats=source_probe(before,held)
        # Match old synthetic-only probe; different ordering cannot change errors.
        prior=C.read(C.N.DOC/'BEFORE.json')['PRIOR_SOURCE']
        for mode in stats:
            assert abs(stats[mode]['PCK10']-prior[mode]['PCK10'])<1e-9
            assert abs(stats[mode]['P90_px']-prior[mode]['P90_px'])<.001
        C.freeze(C.DOC/'SOURCE_BEFORE.json',stats);del before;torch.cuda.empty_cache()
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=C.N.C.load_model().requires_grad_(True);initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=TFAdam(model.parameters(),lr=1e-4);model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eval()
    buffers={k:v.clone() for k,v in model.named_buffers()};rng=np.random.default_rng(7103)
    source_on=arm in ('A10','A11');mode='OCC' if arm in ('A01','A11') else 'CLEAN'
    history=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        pending=[workers.submit(source.item,int(i)) for i in old['source_rows'][0]] if source_on else []
        for step in range(300):
            realrows=[real[int(i)][mode] for i in order[step]]
            branches={'real':realrows}
            if source_on:
                items=[f.result() for f in pending]
                pending=[workers.submit(source.item,int(i)) for i in old['source_rows'][step+1]] if step<299 else []
                branches['source']=[O.corrupted(x,rng) for x in items]
            optimizer.zero_grad(set_to_none=True);losses={}
            for branch,rr in branches.items():
                parts=[]
                for j in range(0,8,2):
                    b=O.batch(rr[j:j+2]);q=model(b['rgb'],b['points'],b['valid']);_,detail=model.losses(q,b['target'],b['target_valid'])
                    loss=T.combined_micro_loss(detail,branch);assert torch.isfinite(loss);loss.backward()
                    parts.append({k:float(v.detach()) for k,v in detail.items()});del b,q,detail,loss
                losses[branch]={k:float(np.mean([p[k] for p in parts])) for k in parts[0]}
            update=float(optimizer.step());assert np.isfinite(update) and update>0
            history.append(dict(step=step+1,real_ids=[x['id'] for x in realrows],real_supervised=sum(int(x['target_valid'].sum()) for x in realrows),loss=losses,update=update))
            if step==0 or (step+1)%50==0:print('TRAIN',arm,step+1,'seconds',round(time.monotonic()-start,1),C.gpu(),flush=True)
    assert all(torch.equal(buffers[k],v) for k,v in model.named_buffers())
    changed=sum(not torch.equal(initial_state[k],v.detach().cpu()) for k,v in model.state_dict().items())
    assert changed>0
    dest=C.RAW/f'{arm}_last300.pt';C.tensor_save(dest,dict(model_state_dict=model.state_dict(),step=300,arm=arm,protocol_sha256=C.bind(C.DOC/'E2_PROTOCOL.json')['sha256']))
    C.freeze(C.RAW/f'TRACE_{arm}.json',history)
    stats=source_probe(model,held);C.freeze(C.DOC/f'SOURCE_{arm}.json',stats)
    C.freeze(fitfile,dict(status='[확인]',complete=True,arm=arm,checkpoint=C.bind(dest),real_exposures=2400,source_exposures=2400 if source_on else 0,
        updates=300,BN_buffers_exact=True,changed_state_tensors=changed,seed=1,seconds=time.monotonic()-start,
        real_order=C.bind(C.RAW/'REAL_ORDER.pt'),source=C.bind(C.DOC/f'SOURCE_{arm}.json'),trace=C.bind(C.RAW/f'TRACE_{arm}.json'),
        final_only=True,detector_training=False,system_changes=False))
    del model,optimizer,real,held,source,initial_state;gc.collect();torch.cuda.empty_cache();print('TRAIN_DONE',arm,flush=True)


@torch.no_grad()
def infer(arm):
    C.N.setup();C.gpu();protocol=C.read(C.DOC/'E2_PROTOCOL.json');fit=C.read(C.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint'])
    dest=C.RAW/f'PREDICTIONS_{arm}.json'
    if dest.exists():print('PREDICTIONS_COMPLETE',arm);return
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    model=C.N.C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict']);model=model.cuda().eval().requires_grad_(False);del ck
    baseline=C.read(C.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0'];result={}
    for i,r in enumerate(protocol['eval_records']):
        C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));raw=baseline[r['id']];before=copy.deepcopy(raw)
        q=C.N.C.predict(model,image,raw);assert raw==before;C.assert_preserved(raw,q);result[r['id']]=q
        if (i+1)%75==0:print('EVAL_INFER',arm,i+1,C.gpu(),flush=True)
    C.freeze(dest,dict(predictions=result,checkpoint=fit['checkpoint'],GT_input=False,evaluation_filtering=0))
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','all',*C.ARMS]);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        for a in C.ARMS if args.stage=='all' else [args.stage]:train(a);infer(a)
