"""Exact-pixel source scoring with external-support/outside-support separation.

Inputs are source cache state manifest and frozen source-output NPZ only.
No current pallet evaluation images/annotations, model, training or PnP import.
"""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rate(a,b):return a/b if b else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--states',required=True);ap.add_argument('--outputs',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output-prefix',required=True);args=ap.parse_args()
    out=HERE/args.output_prefix;assert out.resolve().is_relative_to(HERE)
    freeze=json.loads(Path(args.freeze).read_text());threshold=float(freeze['threshold'])
    with Path(args.states).open(newline='')as f:states=list(csv.DictReader(f))
    assert [int(r['cache_row_index'])for r in states]==list(range(len(states)))
    with np.load(args.outputs,allow_pickle=False)as z:
        indices=z['index'].astype(int);probs=z['probability'].astype(float);ys=z['y'].astype(int)
    assert len(set(indices))==len(indices) and len(indices)==len(probs)==len(ys)
    assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
    rows=[]
    for i,p,y in zip(indices,probs,ys):
        r=states[int(i)];assert r['valid']=='True'and int(r['exact_presence'])==y
        rows.append(dict(**r,probability=float(p),pred_visible=bool(p>=threshold)))
    metrics=[];confusions=[]
    for domain in ['ALL',*sorted({r['source']for r in rows})]:
        z=[r for r in rows if domain=='ALL'or r['source']==domain]
        for band in ['ALL','BOUNDARY','NONBOUNDARY']:
            a=[r for r in z if band=='ALL'or(r['visible_boundary_uncertain']=='True')==(band=='BOUNDARY')]
            tp=sum(r['pred_visible']and int(r['exact_presence'])==1 for r in a);fp=sum(r['pred_visible']and int(r['exact_presence'])==0 for r in a)
            tn=sum(not r['pred_visible']and int(r['exact_presence'])==0 for r in a);fn=sum(not r['pred_visible']and int(r['exact_presence'])==1 for r in a)
            metrics.append(dict(model=args.model,source=domain,boundary_band=band,N=len(a),TP=tp,FP=fp,FN=fn,TN=tn,presence_precision=rate(tp,tp+fp),presence_recall=rate(tp,tp+fn),absence_precision=rate(tn,tn+fn),absence_recall=rate(tn,tn+fp),threshold=threshold))
            for state in ['VISIBLE_TARGET','EXTERNAL_OCCLUDED_SUPPORT','OUTSIDE_TARGET_SUPPORT']:
                s=[r for r in a if r['source_state']==state];pv=sum(r['pred_visible']for r in s);pa=len(s)-pv
                confusions.append(dict(model=args.model,source=domain,boundary_band=band,source_state=state,N=len(s),pred_visible=pv,pred_absent=pa,state_correct_recall=rate(pv if state=='VISIBLE_TARGET'else pa,len(s)),false_presence_rate=rate(pv,len(s))if state!='VISIBLE_TARGET'else None))
    for suffix,rr in [('EXACT_PIXEL_METRICS.csv',metrics),('SUPPORT_STATE_CONFUSION.csv',confusions)]:
        with Path(str(out)+'_'+suffix).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    result=dict(schema='SOURCE_EXACT_PIXEL_SUPPORT_STATE_EVALUATION_V1',model=args.model,threshold=threshold,N=len(rows),state_counts=dict(Counter(r['source_state']for r in rows)),
                target='visible target-instance ownership at exact nearest query pixel',negative_state_interpretation='External-support and outside-support are source GT strata; binary absent output alone does not predict the mechanism.',semantic_self='UNAVAILABLE, not evaluated as mask absence',
                no_current_private_eval_read=True,no_new_training_model_inference_or_PnP=True,source_sampling_caution='Source-mask sampled query strata are intentionally nonnatural proportions; report each source/domain and state denominator.',
                input_bindings=[dict(path=str(Path(p).resolve()),sha256=sha(p))for p in [args.states,args.outputs,args.freeze,Path(__file__)]])
    Path(str(out)+'_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(model=args.model,N=len(rows),out=str(out))))


if __name__=='__main__':main()
