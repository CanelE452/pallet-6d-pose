"""Saved source-only comparison after C14 freeze; no model execution or selection."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parent
C13=V.parent/'query_visibility_same_material_query_head_20261023_v118'

def bind(p):
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def summarize(rows):
    result={}
    for name in ['ALL','TRAIN','VALIDATION','VALIDATION_WOOD16','VALIDATION_OTHER']:
        rr=[r for r in rows if name=='ALL' or r['split']==name or
            (name=='VALIDATION_WOOD16' and r['split']=='VALIDATION' and r['donor_object_id']==16) or
            (name=='VALIDATION_OTHER' and r['split']=='VALIDATION' and r['donor_object_id']!=16)]
        if not rr: continue
        group=dict(N=len(rr),base_queries=len({r['original_query_id'] for r in rr}),
                   base_images=len({r['base_image_sha256'] for r in rr}),donors=len({r['donor_id'] for r in rr}),models={})
        for model in ['C11','C12','C13','C14']:
            if model not in rr[0]:continue
            pr=np.array([r[model]['pred_BEFORE_COVER_SHAM'] for r in rr])
            group['models'][model]=dict(before_D=int((pr[:,0]==0).sum()),cover_E=int((pr[:,1]==1).sum()),
                sham_D=int((pr[:,2]==0).sum()),all3_correct=int((pr==[0,1,0]).all(1).sum()),
                U_branch_responses=int((pr<0).sum()),margin_ge2=sum(r[model]['margin_ge2'] for r in rr),
                mean_cover_E_BCE=float(np.mean([np.logaddexp(0.,-r[model]['COVER']) for r in rr])))
        result[name]=group
    return result

def run(score_path,out):
    assert not out.exists(), 'Preserve existing saved report.'
    with np.load(score_path,allow_pickle=False) as f:
        z=f['query_logit'].copy();a=f['query_available'].copy();state=f['raw_teacher_state'].copy()
        q=f['query_xy_raw'].copy();L=f['L_raw'].copy();split=f['split'].copy()
    assert z.shape==a.shape==(46108,) and np.isfinite(z).all()
    result_rows=[]
    for source_name in ['ALL194_OLD_WITH_C13.jsonl','ALL62_NEW_WITH_C13.jsonl']:
        rows=[json.loads(x) for x in (C13/'public_source_comparison'/source_name).read_text().splitlines()]
        for r in rows:
            ix=np.asarray(r['triple_global_indices'])
            assert np.array_equal(state[ix],['D','E','D'])
            assert (split[ix]==r['split']).all()
            assert np.array_equal(q[ix],np.tile(q[ix[0]],(3,1))) and np.array_equal(L[ix],np.repeat(L[ix[0]],3))
            zz=z[ix].astype(np.float64);pr=np.where(a[ix],(zz>=0).astype(int),-1)
            r['C14']=dict(BEFORE=float(zz[0]),COVER=float(zz[1]),SHAM=float(zz[2]),
                pred_BEFORE_COVER_SHAM=pr.tolist(),cover_minus_sham=float(zz[1]-zz[2]),
                margin_ge2=bool(a[ix[1:]].all() and np.float32(z[ix[1]]-z[ix[2]])>=2),
                cover_E=bool(pr[1]==1),before_D=bool(pr[0]==0),sham_D=bool(pr[2]==0),
                all3_correct=bool(np.array_equal(pr,[0,1,0])),U=int((pr<0).sum()))
        result_rows.append(rows)
    old,new=result_rows
    assert len(old)==194 and len(new)==62
    val=[r for r in old if r['split']=='VALIDATION']
    summary=dict(completed=True,scope='Frozen saved source report; not a selector or natural-transfer claim.',
        scores=bind(score_path),old194=summarize(old),new62_TRAIN_only=summarize(new),
        C13_to_C14_VAL_cover_fixes=sum(r['C13']['pred_BEFORE_COVER_SHAM'][1]==0 and r['C14']['cover_E'] for r in val),
        C13_to_C14_VAL_cover_regressions=sum(r['C13']['pred_BEFORE_COVER_SHAM'][1]==1 and not r['C14']['cover_E'] for r in val),
        VAL_margin_pass_but_cover_D=sum(r['C14']['margin_ge2'] and not r['C14']['cover_E'] for r in val),
        original_q_L_labels_folds_preserved=True,new_model_forward_fit_current_V112_PnP=0,
        source_proxy_is_not_natural_or_native_gold=True,natural_transfer_proven=False)
    out.mkdir(parents=True)
    for name,rows in [('ALL194_OLD_WITH_C14.jsonl',old),('ALL62_NEW_WITH_C14.jsonl',new)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    (out/'RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,oldVAL=summary['old194']['VALIDATION'],woodVAL=summary['old194']['VALIDATION_WOOD16']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.scores,a.output)
