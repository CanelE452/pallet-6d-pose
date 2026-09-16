"""Frozen last-weight detector inference; serialized coordinates before GT scoring."""
import time
import cv2,numpy as np,torch
from ultralytics import YOLO
import env as E
from a_data import records
from a_train import configure,verify_binding

def populations():
    result={}
    for cohort,name in [('RECT','RECT_SYNTH_VAL'),('SQUARE','SQUARE_DEV')]:
        result[name]=[dict(id=r['id'],image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],already_padded=True) for r in records(cohort,'val')]
    pe=E.C.old('paper_evaluation');base=E.read(E.C.LINE/'baseline/FULL_CANDIDATES.json')
    result['RECT_DEV']=[]
    for item in pe.population().positive.items:
        key=pe.canonical_key(item.image)
        result['RECT_DEV'].append(dict(id=item.frame_id,image=key,image_sha256=base['frame_metadata'][key]['image_sha256'],already_padded=False))
    return result

def models(cohort):
    return [('R0',E.R0)]+[(f'{arm}_seed{s}',E.RAW/f'A/runs/{cohort}_{arm}_seed{s}/last.pt') for s in [1,2,3] for arm in ['INDEXED','EQUIV']]

def main():
    configure(1);E.gpu()
    audit=E.read(E.DOC/'A/training_audit.json');assert audit['main_updates']==24000 and audit['source_and_locked_code_hashes_unchanged']
    pops=populations()
    E.freeze(E.DOC/'A/EVALUATION_LOCK.json',dict(time=E.now() if not (E.DOC/'A/EVALUATION_LOCK.json').exists() else E.read(E.DOC/'A/EVALUATION_LOCK.json')['time'],
      populations={k:len(v) for k,v in pops.items()},last_only=True,GT_for_inference=False,
      confidence={'RECT_SYNTH_VAL':.001,'RECT_DEV':.001,'SQUARE_DEV':.25},
      confidence_basis='Existing paper/source .001 and historical square A0 .25; no new threshold tuning.',
      prediction_selection='highest detector confidence; GT IoU never selects candidate',
      evaluation_match='selected candidate IoU>=0.5 against the single GT bbox; otherwise full diagonal penalty',
      source_and_square='already reflect100 padded; no second padding',paper_real='reflect100 once',
      imgsz=640,rect=True,half=False,TF32=False,inference_batch=1,
      missing_annotation='no corner GT => explicit non-evaluable frame, not zero error',
      statistics='paired three-seed primary E_sym difference, 10000 bootstrap, real session/source frame',
      groups='verified C1/C2 source and task-C4 square',FINAL_access=False,
      weights=[E.bound(p) for cohort in ['RECT','SQUARE'] for _,p in models(cohort)]))
    allruns=[]
    for cohort in ['RECT','SQUARE']:
        splits=['RECT_SYNTH_VAL','RECT_DEV'] if cohort=='RECT' else ['SQUARE_DEV']
        for arm,path in models(cohort):
            binding=E.bound(path);model=None
            for split in splits:
                dst=E.RAW/f'A/predictions/{split}/{arm}.json'
                if dst.exists():
                    old=E.read(dst);assert old['complete'] and old['checkpoint']==binding and len(old['records'])==len(pops[split])
                    allruns.append(E.bound(dst));continue
                if model is None:
                    E.gpu();model=YOLO(str(path),task='pose');model.model.requires_grad_(False).eval()
                out=[];start=time.monotonic();conf=.25 if split=='SQUARE_DEV' else .001
                for i,r in enumerate(pops[split]):
                    im=cv2.imread(str(E.ROOT/r['image']));assert im is not None and E.sha(E.ROOT/r['image'])==r['image_sha256']
                    raw_hw=r['raw_hw'] if r['already_padded'] else im.shape[:2]
                    if not r['already_padded']:im=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
                    with torch.inference_mode():
                        pred=model.predict(im,conf=conf,imgsz=640,rect=True,augment=False,half=False,device=0,verbose=False,save=False)[0]
                    assert not model.model.training
                    candidates=[]
                    if pred.boxes is not None:
                        for j in range(len(pred.boxes)):
                            candidates.append(dict(score=float(pred.boxes.conf[j]),box_xyxy=(pred.boxes.xyxy[j].cpu().numpy()-100).tolist(),
                              keypoints_xy=(pred.keypoints.xy[j].cpu().numpy()-100).tolist(),keypoints_conf=pred.keypoints.conf[j].cpu().tolist()))
                    out.append(dict(id=r['id'],image=r['image'],image_sha256=r['image_sha256'],raw_hw=list(raw_hw),candidates=candidates))
                    if (i+1)%500==0:print('A INFER',split,arm,i+1,'/',len(pops[split]),round(time.monotonic()-start,1),'sec',flush=True)
                E.write(dst,dict(complete=True,checkpoint=binding,GT_input=False,records=out,new_detector_forward_images=len(out),elapsed_seconds=time.monotonic()-start))
                allruns.append(E.bound(dst));E.write(E.DOC/'A/INFERENCE_PROGRESS.json',dict(last_completed=f'{split}/{arm}',completed_prediction_files=len(allruns)))
                print('A INFER COMPLETE',split,arm,len(out),flush=True)
            if model is not None:del model;torch.cuda.empty_cache()
    E.write(E.DOC/'A/INFERENCE_COMPLETE.json',dict(complete=True,prediction_files=allruns,
      populations={k:len(v) for k,v in pops.items()},new_detector_forward_images=7*sum(len(v) for v in pops.values()),
      GT_inference_inputs=0,new_optimizer_updates=0,FINAL_access=False))
if __name__=='__main__':main()
