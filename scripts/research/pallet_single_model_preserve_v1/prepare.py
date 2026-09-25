"""Lock model-blind occurrences, preserve old augmentation caches, sanitize fit inputs."""
import json
from collections import Counter
import numpy as np
from . import common as C
from scripts.research.pallet_selector_recovery_v1.build_router_dataset import policy

def main():
    old=[r for r in map(json.loads,(C.STRUCT/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC'];targets={r['index']:r for r in C.read(C.STRUCT/'TARGETS.json') if r['material']=='PLASTIC'}
    assert len(targets)==10
    split=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json');assert {r['id'] for r in targets.values()}==set(split['c0_ids']);heldsha={r['image']['sha256'] for r in split['heldout']}
    assert not heldsha&{r['image']['sha256'] for r in targets.values()}
    # Only clean training targets/masks in fit inputs. Never copy legacy/manual/anchor coordinates.
    occ=[];a,apath=policy()
    for epoch in range(5):
        source=[r for r in old if r['epoch']==epoch and not r['real']];real=[r for r in old if r['epoch']==epoch and r['real']]
        assert len(source)==len(real)==512 and len({r['image'] for r in source})==512
        order=lambda r:C.key(f"{epoch}:{r['slot']}:{r['image']}")
        sc=sorted(source,key=order)[:256];rc=sorted(real,key=order)[:256];so=sorted(source,key=order)
        for batch in range(64):
            items=[('REAL_CLEAN_TASK',r) for r in rc[batch*4:batch*4+4]]+[('SYNTH_CLEAN_TASK',r) for r in sc[batch*4:batch*4+4]]+[('SYNTH_OCCLUDED_PRESERVE',r) for r in so[batch*8:batch*8+8]]
            for task,r in items:
                p=dict(applied=False,reason='no_extra_occlusion',scheduled=False)
                if task=='SYNTH_OCCLUDED_PRESERVE':
                    with np.load(C.ROOT/r['cache']['path']) as z:
                        kp=z['keypoints'][0];box=z['bboxes'][0]*640;xyxy=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]
                        p=a['plan'](kp[:,:2]*640,kp[:,2]==2,xyxy,[0,0,640,640],int(C.key(f'42:occ:{epoch}:{r["slot"]}')[:8],16))
                occ.append(dict(index=len(occ),epoch=epoch,batch=batch,task=task,cache=r['cache'],base_RGB_sha256=r['base_RGB_sha256'],target_sha256=r['target_sha256'],
                    image=r['image'],original_slot=r['slot'],target_id=targets[r['target_index']]['id'] if r['real'] else None,plan=p))
    path=C.RAW/'OCCURRENCES.json';C.freeze(path,occ)
    C.freeze(C.DOC/'OCCURRENCES_LOCK.json',dict(created_at=C.now(),occurrences=C.bind(path),counts=dict(Counter(r['task'] for r in occ)),total=len(occ),steps=320,epochs=5,batch=16,
        real_clean_ids=sorted(r['id'] for r in targets.values()),real_sha=sorted(r['image']['sha256'] for r in targets.values()),heldout_overlap=0,old_plan=C.bind(C.STRUCT/'AUGMENTATION_PLAN.jsonl'),occlusion_policy=C.bind(apath),
        applied_by_epoch={e:sum(r['plan']['applied'] for r in occ if r['epoch']==e) for e in range(5)},selection='Fixed SHA occurrence ranking within each original epoch; 4/4/8 per batch; no model outputs read',
        augmentation_caveat='Original cached YOLO geometry retained. Synthetic rectangle native canvas=640 training cache, not an extra geometric augmentation. Original policy schedule/paired feasibility retained.'))
    # Prediction-only inference metadata; no GT fields copied.
    meta={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json')}
    real=[dict(id=r['id'],image=r['image'],K=meta[r['id']]['K'],dims=meta[r['id']]['xyz']) for r in split['heldout']]
    sources=C.read(C.STRUCT/'SOURCE_PROBE_PLAN.json')['records'];table=np.load(C.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz');stems=table['stems'];ix={str(v):i for i,v in enumerate(stems)};ks=table['K'];dims=table['dims'];sr=[]
    for r in sources:
        i=ix[r['id']];fx,fy,cx,cy=ks[i];sr.append(dict(id=r['id'],image=r['image'],hw=r['hw'],table_index=i,K=[[fx,0,cx],[0,fy,cy],[0,0,1]],dims=dims[i].tolist()))
    assert len(sr)==256;C.freeze(C.RAW/'INFERENCE_INPUTS.json',dict(real=real,source=sr));C.freeze(C.DOC/'INFERENCE_INPUTS_LOCK.json',dict(created_at=C.now(),inputs=C.bind(C.RAW/'INFERENCE_INPUTS.json'),source_plan=C.bind(C.STRUCT/'SOURCE_PROBE_PLAN.json')))
    C.freeze(C.DOC/'SANITY32_LOCK.json',dict(created_at=C.now(),source_ids=[r['id'] for r in sorted(sr,key=lambda r:C.key(r['id']))[:32]],selection='First32 SHA ranked fixed source probe; inference parity only, never fit',GT_input=False))
    print('OCCURRENCES',len(occ),dict(Counter(r['task'] for r in occ)),flush=True)
if __name__=='__main__':main()
