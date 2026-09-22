"""Original-coordinate locked training adapter. Does not read evaluation truth."""
import copy
from functools import lru_cache
import json
import numpy as np
import cv2
import torch
from . import common as C
from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix,transform_points
from scripts.research.pallet_cad8_occlusion_v1.augmentation import apply as apply_occlusion
B=C.B;CORE=B.E.N.C

def matrix(box,expansion):
    assert expansion in (C.BASE_EXPANSION,C.EXPANSION)
    return axis_aligned_crop_matrix(box,expansion=expansion)

def crop(image,box,points,valid,expansion):
    m=matrix(box,expansion)
    rgb=cv2.warpAffine(image,m[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[:,:,::-1].astype(np.float32)-CORE.MEAN
    return dict(rgb=rgb.transpose(2,0,1),points=transform_points(np.where(valid[:,None],points,0),m).astype(np.float32),
        valid=valid.copy(),matrix=m,original_points=points.copy(),box=np.array(box),bbox_diagonal=float(np.linalg.norm(np.array(box[2:])-box[:2])))

def prepare_input(image,pred,expansion):
    q=CORE.selected(pred)
    if q is None or q.get('keypoints_xy') is None:return None
    pts=np.array(q['keypoints_xy'],float);box=np.array(q['box_xyxy'],float)
    if not np.isfinite(box).all() or not (box[2:]>box[:2]).all():return None
    valid=np.isfinite(pts).all(-1)&~(pts==-1).all(-1)
    return crop(image,box,pts,valid,expansion)

def supported(target,valid):return valid&np.isfinite(target).all(-1)&(target>=0).all(-1)&(target[:,0]<288)&(target[:,1]<384)

def target_item(x,gt,sem):
    x=dict(x);x['target']=transform_points(np.where(sem[:,None],gt,0),x['matrix']).astype(np.float32)
    x['target_valid']=supported(x['target'],sem);x['target_valid'][8]=False
    x['original_gt']=gt.copy();x['original_gt_valid']=sem.copy();return x

def counts(old,new):
    a=old['target_valid'];b=new['target_valid'];t=new['target']
    return dict(old=int(a.sum()),new=int(b.sum()),newly=int((~a&b).sum()),lost=int((a&~b).sum()),
        old_mask=a.tolist(),new_mask=b.tolist(),newly_mask=(~a&b).tolist(),
        target_valid_but_outside_expectation=int((b&((t[:,0]>284)|(t[:,1]>380))).sum()))

class Source:
    def __init__(self):self.base=B.L.SourceData()
    @lru_cache(maxsize=24)
    def pair(self,row):
        old=self.base.item(int(row));image=cv2.imread(str(C.ROOT/old['image_binding']['path']))
        check=crop(image,old['box'],old['original_points'],old['valid'],C.BASE_EXPANSION)
        check=target_item(check,old['original_gt'],old['original_gt_valid'])
        for k in ('rgb','points','valid','matrix','target','target_valid'):np.testing.assert_array_equal(check[k],old[k])
        new=crop(image,old['box'],old['original_points'],old['valid'],C.EXPANSION)
        new=target_item(new,old['original_gt'],old['original_gt_valid'])
        for k in ('id','partition','image_binding','row','source_id','source_partition'):new[k]=old[k]
        assert not (old['target_valid']&~new['target_valid']).any()
        return old,new

def original_corrupted(old,rng,force=False):
    perturbed=B.L.corrupted(old,rng,force)
    original=transform_points(perturbed['points'],np.linalg.inv(old['matrix']))
    return perturbed,original

def with_original_points(item,points,valid):
    out=dict(item);out['points']=transform_points(points,item['matrix']).astype(np.float32);out['valid']=valid.copy();return out

def prepare():
    C.protocol();B.L.setup();cv2.setNumThreads(1);lock=C.read(C.DOC/'INPUT_LOCK.json');oldlock=lock['previous_inputs']
    pool={r['id']:r for r in C.read(B.E.RAW/'PSEUDOLABEL_MANIFEST.json')};paired=C.read(B.E.DOC/'E2_INPUT_LOCK.json')['records']
    real=[];parity=[];masks=[]
    evalsha={r['image']['sha256'] for r in lock['eval_records']}
    for i,(binding,meta) in enumerate(zip(oldlock['paired_files'],paired)):
        C.verify(binding);C.verify(meta['image']);assert meta['image']['sha256'] not in evalsha
        entry=torch.load(C.ROOT/binding['path'],map_location='cpu',weights_only=False);assert entry['metadata']==meta
        row=pool[meta['id']];assert row['accepted'] and not row['GT_input'];assert row['raw']==meta['raw_prediction']
        image=cv2.imread(str(C.ROOT/meta['image']['path']));assert B.P.array_sha(image)==meta['clean_RGB_sha']
        tensor=torch.from_numpy(image[:,:,::-1].copy().transpose(2,0,1)).float()[None]/255
        apply_occlusion(tensor,copy.deepcopy(meta['plans']))
        occ=np.clip(np.rint(tensor[0].numpy().transpose(1,2,0)[:,:,::-1]*255),0,255).astype(np.uint8)
        assert B.P.array_sha(occ)==meta['occluded_RGB_sha']
        gt=np.array(meta['target_original']);np.testing.assert_array_equal(gt,np.array(B.E.P.top(row['refined'])['keypoints_xy']))
        sem=B.E.P.valid_points(B.E.P.top(row['raw']))&np.isfinite(gt).all(1)&(gt>=0).all(1)&(gt[:,0]<image.shape[1])&(gt[:,1]<image.shape[0]);sem[8]=False
        pairs={};errors=[]
        for expansion in (C.BASE_EXPANSION,C.EXPANSION):
            items={};common=sem.copy()
            for mode,img,pred in [('CLEAN',image,meta['raw_prediction']),('OCC',occ,meta['occluded_R0_prediction'])]:
                item=prepare_input(img,pred,expansion);assert item is not None
                # Legacy paired_items transforms every finite pseudo coordinate, including unsupervised ones.
                target=transform_points(gt,item['matrix']).astype(np.float32)
                common &= supported(target,sem);item.update(target=target,id=meta['id'],original_gt=gt.copy(),original_gt_valid=sem.copy())
                items[mode]=item
            for mode,item in items.items():
                item['target_valid']=common.copy()
                err=float(np.abs(transform_points(item['target'],np.linalg.inv(item['matrix']))-gt).max());errors.append(err);assert err<1e-4
                if expansion==C.BASE_EXPANSION:
                    for key in ('rgb','points','valid','matrix','box','original_points','target','target_valid'):np.testing.assert_array_equal(item[key],entry['pair'][mode][key])
                else:
                    np.testing.assert_array_equal(item['original_points'],entry['pair'][mode]['original_points'])
                    np.testing.assert_array_equal(item['box'],entry['pair'][mode]['box'])
            pairs[str(expansion)]=items
        q=counts(entry['pair']['OCC'],pairs[str(C.EXPANSION)]['OCC']);assert not q['lost']
        dest=C.RAW/'real'/f'{i:04d}.pt'
        if dest.exists():
            cached=torch.load(dest,map_location='cpu',weights_only=False);assert cached['metadata']==meta
            np.testing.assert_array_equal(cached['semantic_mask'],sem)
            for mode,item in pairs[str(C.EXPANSION)].items():
                for key,value in item.items():np.testing.assert_array_equal(cached['pair'][mode][key],value)
        else:C.tensor_save(dest,dict(pair=pairs[str(C.EXPANSION)],metadata=meta,semantic_mask=sem))
        real.append(dict(id=meta['id'],session=row['session'],image=meta['image'],pair=C.bind(dest),old=binding))
        masks.append(dict(id=meta['id'],session=row['session'],**q));parity.append(dict(id=meta['id'],all_original_xy_exact=True,initial_xy_bbox_exact=True,occlusion_RGB_plan_exact=True,baseline_tensors_exact=True,roundtrip_max_px=max(errors)))
        if (i+1)%70==0:print('REAL_REBUILD',i+1,flush=True)
    C.verify(oldlock['real_order']);C.verify(oldlock['source_orders'])
    order=torch.load(C.ROOT/oldlock['real_order']['path'],weights_only=True).numpy();orders=np.load(C.ROOT/oldlock['source_orders']['path'])
    source=Source();ids=np.unique(orders['source_rows']);assert np.isin(ids,source.base.train_rows).all();assert not np.intersect1d(ids,orders['held_rows']).size
    srcmask=[]
    for i,row in enumerate(np.r_[ids,orders['held_rows']]):
        old,new=source.pair(int(row));q=counts(old,new)
        np.testing.assert_array_equal(old['original_gt'],new['original_gt']);np.testing.assert_array_equal(old['original_points'],new['original_points'])
        srcmask.append(dict(row=int(row),id=old['id'],partition=old['partition'],**q))
        if (i+1)%300==0:print('SOURCE_MASK',i+1,flush=True)
    trace=[json.loads(s) for s in (B.RAW/'TRACE_FULL.jsonl').read_text().splitlines()];rng=np.random.default_rng(7103);corrupt=[];cv=[];repro=[]
    for step,indices in enumerate(orders['source_rows']):
        points=[];valid=[];legacy=[];names=[];maximum=0.
        for j in indices:
            old,new=source.pair(int(j));perturbed,original=original_corrupted(old,rng)
            np.testing.assert_allclose(transform_points(original,old['matrix']),perturbed['points'],atol=1e-9,rtol=0)
            mapped=with_original_points(new,original,perturbed['valid'])
            maximum=max(maximum,float(np.abs(transform_points(mapped['points'],np.linalg.inv(new['matrix']))-original).max()))
            points.append(original);valid.append(perturbed['valid']);legacy.append(perturbed['points']);names.append(old['id'])
        sha=B.P.array_sha(np.stack(legacy));assert sha==trace[step]['source_corrupted_points_sha'],('source corruption mismatch',step)
        assert names==trace[step]['source_ids'];assert indices.tolist()==trace[step]['source_rows']
        assert [real[int(i)]['id'] for i in order[step]]==trace[step]['real_ids'];assert maximum<1e-4
        corrupt.append(points);cv.append(valid);repro.append(dict(step=step+1,source_sha125=sha,roundtrip_max_px=maximum,ids=names))
        if (step+1)%75==0:print('CORRUPTION_PARITY',step+1,flush=True)
    dest=C.RAW/'CORRUPTION_ORIGINAL.npz'
    if dest.exists():
        frozen=np.load(dest);np.testing.assert_array_equal(frozen['points'],np.array(corrupt));np.testing.assert_array_equal(frozen['valid'],np.array(cv))
    else:
        with dest.open('xb') as f:np.savez_compressed(f,points=np.array(corrupt),valid=np.array(cv))
    def aggregate(rr):
        return dict(unique_images=len(rr),**{k:sum(r[k] for r in rr) for k in ('old','new','newly','lost','target_valid_but_outside_expectation')},
            per_corner={key:np.array([r[key] for r in rr],int).sum(0).tolist() for key in ('old_mask','new_mask','newly_mask')})
    smap={r['row']:r for r in srcmask};realexpo=[masks[int(i)] for i in order.ravel()];srcexpo=[smap[int(i)] for i in orders['source_rows'].ravel()]
    summary=dict(real=aggregate(masks),source_train=aggregate([r for r in srcmask if r['partition']=='train']),source_heldout=aggregate([r for r in srcmask if r['partition']=='heldout']),
        real_exposures=aggregate(realexpo),source_exposures=aggregate(srcexpo),real_sessions={s:aggregate([r for r in masks if r['session']==s]) for s in {r['session'] for r in masks}})
    realprobe=sorted(set(list(np.linspace(0,len(real)-1,16,dtype=int))+[i for i,r in enumerate(masks) if r['newly']][:32]))
    srcprobe=sorted(set(ids[:16].tolist()+[r['row'] for r in srcmask if r['partition']=='train' and r['newly']][:32]))
    C.save(C.DOC/'TRAIN_INPUT_AUDIT.json',dict(PASS=True,real=real,real_order=oldlock['real_order'],source_orders=oldlock['source_orders'],corruption=C.bind(dest),
        semantic_mask_recovered=True,mask_rule='original confidence/finite/in-frame semantic mask AND clean/OCC crop intersection',
        counts=summary,real_probe_indices=realprobe,source_probe_rows=srcprobe,source_masks=srcmask,real_masks=masks,
        eval_GT_dependency=False,real_image_hash_overlap=0,no_new_pseudo_teacher=True))
    C.save(C.DOC/'ORIGINAL_COORDINATE_PARITY.json',dict(PASS=True,real=parity,source_original_points_GT_exact=True,baseline_source_tensors_exact=True))
    C.save(C.DOC/'SOURCE_CORRUPTION_PARITY.json',dict(PASS=True,baseline_trace= C.bind(B.RAW/'TRACE_FULL.jsonl'),steps=repro,
        original_points=C.bind(dest),same_seed_only=False,all_300_trace_hashes_exact=True,original_fixed_before_crop=True,newly_supervised_not_newly_corrupted=True))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f'# Preflight PASS\n\nHEAD `{lock["HEAD"]}`. 기존FULL/PRIOR1 checkpoint·contract, protected 입력 해시 확인. 실사{len(real)}장의 원영상RGB/고정가림RGB/초기점/bbox/pseudo XY 및1.25 tensor exact parity. 원래 semantic mask 복원 후동일clean/OCC 교집합 적용.\n\nsource TRAIN/heldout 분리. 원래1.25교란을300step 전부재생성해 기존trace hash exact일치 후 원영상좌표동결. 신규support에 따라RNG소비 변경없음. real/source row순서일치. 평가GT는학습입력에미사용. 제어crop baseline 불변. 신규학습전 필수테스트 필요.\n')
    print('DATA_READY',summary,flush=True)

def load_real():
    a=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json');out=[]
    for r in a['real']:
        C.verify(r['pair']);out.append(torch.load(C.ROOT/r['pair']['path'],map_location='cpu',weights_only=False)['pair']['OCC'])
    return out,torch.load(C.ROOT/a['real_order']['path'],weights_only=True).numpy(),np.load(C.ROOT/a['source_orders']['path']),np.load(C.ROOT/a['corruption']['path'])

if __name__=='__main__':prepare()
