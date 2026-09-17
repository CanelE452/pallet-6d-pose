"""Phase 3: known-object conditional deployment; GT stays outside inference."""
import argparse,copy
import numpy as np
import torch,cv2
import dcp_env as E
from inference import load_head,predict_captured,registry_input,preservation,serial
from eval_math import measure,summary

def population_metadata():
    pe=E.old('paper_evaluation');meta={r['frame_id']:r for r in E.read(pe.POS)['items']}
    return pe,[(item,meta[item.frame_id]) for item in pe.population().positive.items]

@torch.no_grad()
def cache():
    assert E.read(E.DOC/'SYNTH_HELDOUT_RESULTS.json')['complete'];E.gpu();fx=E.old('features');extractor=fx.FrozenYoloFeatures(E.R0)
    pe,pop=population_metadata();baseline_payload=E.read(E.LINE/'baseline/FULL_CANDIDATES.json');baseline=baseline_payload['frames'];records=[];dims_by_type={}
    for item,meta in pop:
        key=pe.canonical_key(item.image);dims,order=registry_input(meta['object_type']);dst=E.RAW/f'DEV_cache/{item.frame_id.replace(":","__")}.pt';dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            im=cv2.imread(str(E.ROOT/item.image));assert im is not None
            assert E.sha(E.ROOT/item.image)==baseline_payload['frame_metadata'][key]['image_sha256']
            cap=extractor.predict(im);before=baseline[key];after=cap['candidates'];assert len(before)==len(after)
            for a,b in zip(before,after):
                for k in ['score','box_xyxy','keypoints_xy']:assert np.array_equal(np.array(a[k]),np.array(b[k])),(key,k)
            cap={k:v.cpu() if torch.is_tensor(v) else v for k,v in cap.items()}
            torch.save(dict(captured=cap,id=item.frame_id,key=key,raw_hw=im.shape[:2],dimensions=dims,order=order,object_type=meta['object_type'],session=meta['session_id'],GT_input=False,
              image_sha256=E.sha(E.ROOT/item.image)),dst)
        records.append(dict(id=item.frame_id,path=str(dst.relative_to(E.ROOT)),sha256=E.sha(dst),key=key,object_type=meta['object_type']))
        value=dims.tolist()
        if meta['object_type'] in dims_by_type:assert dims_by_type[meta['object_type']]==value
        dims_by_type[meta['object_type']]=value
    extractor.close();assert len(records)==319
    E.write(E.DOC/'DEV_CACHE_COMPLETE.json',dict(complete=True,frames=319,records=records,canonical_dimensions_by_type=dims_by_type,R0_candidates_exact=True,
      metadata_bindings=[E.bound(pe.POS),E.bound(E.LINE/'baseline/FULL_CANDIDATES.json'),E.bound(E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')],
      source='PAPER_EVAL_ALL_POS.object_type -> CHALLENGE_OBJECT_GEOMETRY_REGISTRY physical XYZ; no target/pose/axis hypothesis used',
      known_object_type_assumption=True,metadata_availability_verified_in_live_deployment=False,
      unknown_type_scenario='NOT_ESTIMATED: explicit rejection; no object type predictor trained',no_WD_GT_swap=True))
    print('DEV_GT_FREE_CACHE_COMPLETE',flush=True)

@torch.no_grad()
def infer(arms=None,temperatures=None):
    cache=E.read(E.DOC/'DEV_CACHE_COMPLETE.json');assert cache['complete'];E.gpu()
    sel=E.read(E.DOC/'CALIBRATION_AND_SELECTION.json');norm=E.read(E.DOC/'DIM_NORMALIZATION_LOCK.json');oldsel=E.read(E.C.B/'P_SELECTION.json')
    arms=['OLD_P',*E.ARMS] if arms is None else arms;files={};checked=0
    for arm in arms:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';dst=E.RAW/f'predictions/REAL_DEV/{name}.json'
            if dst.exists():files[name]=E.bound(dst);continue
            head,_=load_head(arm,seed);T=(oldsel['temperatures'][str(seed)]['temperature'] if arm=='OLD_P' else (temperatures or sel['temperatures'])[name]['temperature']);preds=[]
            for r in cache['records']:
                row=torch.load(E.ROOT/r['path'],map_location='cpu',weights_only=False);cap=row['captured']
                for k in ['p3','p4']:cap[k]=cap[k].cuda()
                pred,diag=predict_captured(head,arm,cap,row['dimensions'],row['order'],T,sel['rule'],row['raw_hw'],norm)
                pred.update(id=row['id'],key=row['key']);preds.append(pred);checked+=1
                if diag is not None:
                    dd=E.RAW/f'DEV_logits/{name}';dd.mkdir(parents=True,exist_ok=True)
                    np.savez(dd/(row['id'].replace(':','__')+'.npz'),**diag)
            E.write(dst,dict(complete=True,records=preds,GT_input=False,known_object_type_assumption=True,checkpoint=E.bound(E.RAW/f'runs/{name}/last.pt') if arm!='OLD_P' else E.bound(E.C.BRAW/f'runs/seed{seed}/last.pt')))
            files[name]=E.bound(dst);del head;torch.cuda.empty_cache();print('DEV_INFERENCE',name,319,flush=True)
    E.write(E.DOC/('DEV_INFERENCE_COMPLETE.json' if arms==['OLD_P',*E.ARMS] else 'MIXED_DEV_INFERENCE_COMPLETE.json'),dict(complete=True,files=files,contract_checked_this_execution=checked,GT_input=False))

def iou(a,b):
    a=np.array(a);b=np.array(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1e-12))

def evaluate(arms=None):
    assert E.read(E.DOC/'DEV_INFERENCE_COMPLETE.json')['complete'];pe,pop=population_metadata();groups={r['object_type']:r for r in E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']};cache=E.read(E.DOC/'DEV_CACHE_COMPLETE.json')
    cmeta={r['id']:torch.load(E.ROOT/r['path'],map_location='cpu',weights_only=False) for r in cache['records']}
    targets={}
    for item,m in pop:
        t=pe.E._legacy_forbidden_target(item);targets[item.frame_id]=dict(gt=np.array(t.keypoints_xy),valid=np.array(t.keypoint_supervision_mask),box=np.array(t.box_xyxy),meta=m)
    arms=['OLD_P',*E.ARMS] if arms is None else arms;allrows={};summaries={}
    for arm in arms:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';pred=E.read(E.RAW/f'predictions/REAL_DEV/{name}.json');assert pred['complete'] and not pred['GT_input'];rows=[]
            for p in pred['records']:
                t=targets[p['id']];m=t['meta'];idx=p['selected_index'];candidate=None if idx is None else p['candidates'][idx];matched=candidate is not None and iou(candidate['box_xyxy'],t['box'])>=.5
                points=np.full((9,2),np.nan) if candidate is None else candidate['keypoints_xy'];g=groups[m['object_type']];d=cmeta[p['id']]['dimensions'];ratio=max(d[0]/d[1],d[1]/d[0])
                row=measure(points,t['gt'],t['valid'],g['permutations'],cmeta[p['id']]['raw_hw'],matched,idx is not None)
                row.update(id=p['id'],session=m['session_id'],group='C'+str(g['group_order']),object=m['object_type'],ratio_bin='<=1.05' if ratio<=1.05 else '<=1.2' if ratio<=1.2 else '>1.2');rows.append(row)
            allrows[name]=rows;summaries[name]=summary(rows);print('DEV_RESULT',name,summaries[name],flush=True)
    E.write(E.RAW/('REAL_DEV_METRICS.json' if arms==['OLD_P',*E.ARMS] else 'MIXED_DEV_METRICS.json'),allrows)
    E.write(E.DOC/('REAL_DEV_RESULTS.json' if arms==['OLD_P',*E.ARMS] else 'MIXED_DEV_RESULTS.json'),dict(complete=True,summary=summaries,known_object_type_assumption=True,metadata_availability_verified_in_live_deployment=False,
      independent_confirmation=False,physical_occlusion_subgroup='NOT_REPORTED: no verified physical-occlusion contract adopted; annotation-valid is not occlusion',unknown_object_type='not a supported deployment scenario'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','infer','evaluate']);args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);globals()[args.phase]()
