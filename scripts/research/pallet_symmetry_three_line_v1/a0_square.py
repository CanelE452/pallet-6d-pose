"""Actual historical checkpoint inference, serialized before separate GT scoring."""
import math, csv, copy
import numpy as np
import cv2, torch
from ultralytics import YOLO
import env as E
from audit_math import symmetry_error

def stats(x):
    a=np.array(x,float);return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90))) if len(a) else dict(n=0)
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    g=E.gpu();assert torch.cuda.is_available();print(g,flush=True)
    membership=E.read(E.RAW/'A/square_membership.json')['val']['records']
    p=np.array(E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations'])
    arms={a:E.TRACK/a/'weights/best.pt' for a in ['F0','F1','F2','clean_label/stage1_indexed','clean_label/stage2_c4']}
    allrows=[];summaries={};bindings=[];inference_count=0
    for arm,path in arms.items():
        tag=arm.replace('/','_');predpath=E.RAW/f'A/A0/{tag}_predictions.json'
        if not path.exists():summaries[arm]=dict(status='MISSING_CHECKPOINT');continue
        bindings.append(E.bound(path))
        if not predpath.exists():
            model=YOLO(str(path));model.model.requires_grad_(False).eval();outputs=[]
            for i,r in enumerate(membership):
                im=cv2.imread(str(E.ROOT/r['image']))
                pred=model.predict(im,imgsz=640,conf=.25,verbose=False,device=0,save=False)[0]
                outputs.append(dict(id=r['id'],image_sha256=r['image_sha256'],
                  boxes=pred.boxes.xyxy.cpu().tolist(),scores=pred.boxes.conf.cpu().tolist(),
                  keypoints=pred.keypoints.data.cpu().tolist() if pred.keypoints is not None else []))
                if i%50==0:print('A0 inference',arm,i,len(membership),flush=True)
            E.write(predpath,dict(checkpoint=E.bound(path),GT_input=False,images=len(outputs),records=outputs))
            del model;torch.cuda.empty_cache()
        payload=E.read(predpath);assert payload['checkpoint']['sha256']==E.sha(path)
        inference_count+=len(payload['records'])
        rows=[];pooled=[]
        for rec,r in zip(payload['records'],membership):
            assert rec['id']==r['id'];im_hw=np.array(r['raw_hw'])+200
            # Old F0/1/2 used live_gt_v4 labels; clean arms use v6 labels.
            ds=E.ROOT/'challenge/yolo_pose_one_model/datasets/live_gt_v4' if arm in ['F0','F1','F2'] else E.SQUARE
            lp=ds/'labels/val'/f"{r['id']}.txt"
            target=np.array(lp.read_text().split(),float)[5:].reshape(9,3)
            gt=target[:,:2]*im_hw[::-1]-100;valid=target[:,2]>0
            pred=np.array(rec['keypoints'][0])[:,:2]-100 if rec['keypoints'] else np.full((9,2),np.nan)
            diagonal=math.hypot(*r['raw_hw']);s=symmetry_error(pred,gt,valid,p,diagonal)
            if not s['evaluable']:continue
            if rec['keypoints']:
                buggy=np.array([np.linalg.norm(pred[valid]-gt[q][valid],axis=-1).mean() for q in p])
                correct9=np.array([np.linalg.norm(pred[valid[q]]-gt[q][valid[q]],axis=-1).mean() for q in p])
                buggy_value=float(buggy.min());correct9_value=float(correct9.min())
                fixed9=float(buggy[0]);affected=abs(correct9_value-buggy_value)>1e-8
            else:buggy_value=correct9_value=fixed9=None;affected=False
            d=dict(arm=arm,id=r['id'],detected=bool(rec['keypoints']),legacy_population=str(ds.relative_to(E.ROOT)),
              E_fixed=s['fixed_mean']/diagonal,E_sym=s['equivalent_mean']/diagonal,
              fixed_mean_px=s['fixed_mean'],sym_mean_px=s['equivalent_mean'],branch=s['branch'],
              annotated_corners=s['n_corners'],legacy_fixed9_px=fixed9,legacy_buggy9_px=buggy_value,
              corrected9_px=correct9_value,mask_bug_changed_metric=affected,
              mask_noninvariant=bool(np.any(valid[p]!=valid)),
              mask_changed_slots=int(np.sum(valid[p]!=valid)),
              rescue=s['fixed_mean']>20 and s['equivalent_mean']<5,true_collapse=s['equivalent_mean']>20)
            pooled.extend(s['errors'][:8][s['gt_mask'][:8]].tolist());rows.append(d)
        allrows+=rows
        summaries[arm]=dict(status='COMPLETE',frames=len(rows),detections=sum(r['detected'] for r in rows),
          primary_E_sym=stats([r['E_sym'] for r in rows]),frame_corner_mean_px=stats([r['sym_mean_px'] for r in rows]),
          pooled_corner_px=stats(pooled),mask_bug_affected_frames=sum(r['mask_bug_changed_metric'] for r in rows),
          mask_noninvariant_frames=sum(r['mask_noninvariant'] for r in rows),
          prediction_binding=E.bound(predpath))
        print(arm,summaries[arm],flush=True)
    path=E.DOC/'A/evaluation_correction.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]),lineterminator='\n');w.writeheader();w.writerows(allrows)
    E.write(E.DOC/'A/A0_SQUARE.json',dict(status='COMPLETE',arms=summaries,checkpoint_bindings=bindings,
      actual_inferences=inference_count,actual_training_updates=0,
      historical_matching='same historical conf0.25 top1; no added GT IoU gate',
      score_change_is_not_model_improvement=True,old_staged_training_not_matched_budget=True,
      original_results_modified=False,paper_population_mixed=False,FINAL_access=False))
if __name__=='__main__':main()
