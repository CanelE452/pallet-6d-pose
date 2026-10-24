"""Score already-sealed SAM predictions with each source's original mask rule.

This repairs source-GT decoding only: SYNTH >127 versus YCB >0. Prediction
masks, query coordinates, selected SAM alternatives and logit0 stay frozen.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def lines(path):return[json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]

def csvrows(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))

def dumpcsv(path,rows):
    with Path(path).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def score():
    from PIL import Image
    lock=json.loads((HERE/'PREPARATION_LOCK.json').read_text());seal=json.loads((HERE/'PREDICTION_SEAL.json').read_text())
    assert lock['scoring_mask_manifest_sha256']==sha(HERE/'SCORING_ONLY_MASKS.jsonl')
    for b in seal['files']:assert sha(b['path'])==b['sha256']
    targets={r['id']:r for r in lines(HERE/'TARGET_PREDICTIONS.jsonl')};truth={r['id']:r for r in lines(HERE/'SCORING_ONLY_MASKS.jsonl')}
    masks={};ious=[];rows=[]
    for ident,r in truth.items():
        mask_threshold=127 if r['source']=='SYNTH' else 0
        v=np.array(Image.open(r['mask_vis']))>mask_threshold;f=np.array(Image.open(r['mask_full']))>mask_threshold;assert v.shape==f.shape and not(v&~f).any()
        p=targets[ident];assert sha(p['mask_path'])==p['mask_sha256']
        with np.load(p['mask_path'])as z:m=z['mask'].astype(bool)
        assert m.shape==v.shape;intersection=int((m&v).sum());union=int((m|v).sum())
        ious.append(dict(id=ident,source=r['source'],intersection=intersection,union=union,visible_mask_IoU=intersection/union if union else None))
        masks[ident]=v,f
    for q in csvrows(HERE/'QUERY_PREDICTIONS.csv'):
        if q['query_valid']!='True':continue
        v,f=masks[q['id']];h,w=v.shape;px,py,rad=int(q['pixel_x']),int(q['pixel_y']),int(q['radius'])
        patch=(slice(max(0,py-rad),min(h,py+rad+1)),slice(max(0,px-rad),min(w,px+rad+1)))
        vy,fy=bool(v[patch].any()),bool(f[patch].any())
        state='VISIBLE_TARGET_SUPPORT'if vy else'EXTERNAL_OCCLUDED_SUPPORT'if fy else'OUTSIDE_TARGET_SUPPORT'
        rows.append(dict(**q,target_visible_support=vy,source_support_state=state))
    metrics=[];conf=[]
    for domain in['SYNTH','YCB']:
        for rad in[0,1,2]:
            a=[r for r in rows if r['source']==domain and int(r['radius'])==rad]
            tp=sum(r['target_visible_support']and r['pred_visible_support']=='True'for r in a);fp=sum(not r['target_visible_support']and r['pred_visible_support']=='True'for r in a)
            fn=sum(r['target_visible_support']and r['pred_visible_support']=='False'for r in a);tn=sum(not r['target_visible_support']and r['pred_visible_support']=='False'for r in a)
            ratio=lambda x,y:x/y if y else None
            metrics.append(dict(source=domain,radius=rad,target_name='exact_query_pixel_presence'if rad==0 else f'clipped_square_radius{rad}_any_visible_support',N=len(a),TP=tp,FP=fp,FN=fn,TN=tn,visible_precision=ratio(tp,tp+fp),visible_recall=ratio(tp,tp+fn),absent_precision=ratio(tn,tn+fn),absent_recall=ratio(tn,tn+fp)))
            for state in['VISIBLE_TARGET_SUPPORT','EXTERNAL_OCCLUDED_SUPPORT','OUTSIDE_TARGET_SUPPORT']:
                z=[r for r in a if r['source_support_state']==state];pv=sum(r['pred_visible_support']=='True'for r in z)
                conf.append(dict(source=domain,radius=rad,source_support_state=state,N=len(z),pred_visible_support=pv,pred_absent=len(z)-pv))
    dumpcsv(HERE/'SOURCE_QUERY_SCORED.csv',rows);dumpcsv(HERE/'SOURCE_METRICS.csv',metrics);dumpcsv(HERE/'SOURCE_SUPPORT_CONFUSION.csv',conf);dumpcsv(HERE/'VISIBLE_MASK_IOU.csv',ious)
    result=dict(scope='source-only frozen box-prompted general MobileSAM; no current private eval',code_sha256=sha(__file__),prediction_seal_sha256=sha(HERE/'PREDICTION_SEAL.json'),metrics=metrics,mean_visible_mask_IoU_by_source={d:float(np.mean([r['visible_mask_IoU']for r in ious if r['source']==d]))for d in['SYNTH','YCB']},semantic_self='not supplied by target mask; no semantic self performance claim',new_PnP_calls=0,source_GT_masks_opened_only_in_score=True,source_mask_thresholds=dict(SYNTH='>127',YCB='>0'),score_repair='SYNTH nonzero background artifact values1/2 are absent under the preserved v61 mask contract; no model/prediction/input/threshold change')
    write(HERE/'SOURCE_EVALUATION.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':score()
