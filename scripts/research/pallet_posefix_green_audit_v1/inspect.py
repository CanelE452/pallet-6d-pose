"""Read-only green error audit and frozen-model heatmap diagnostics, no fitting."""
import gc
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_large_error_v1 import core as C
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation

ROOT=N.ROOT;DOC=ROOT/'_docs/experiments/pallet_posefix_green_audit_v1'
RAW=ROOT/'data/pallet/results/pallet_posefix_green_audit_v1'
CASES={'worst':('capture_20260902__009323',1),
       'truncated':('capture_20260902_kimjihoon__003249',2),
       'nontruncated':('capture_20260902_kimjihoon__006349',0),
       'improved':('capture_20260902_kimjihoon__007309',4)}


def write(path,obj):
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists(): assert json.loads(path.read_text())==obj
    else: path.write_text(content)


def selected(p):return p['candidates'][p['selected_index']]


def inputs():
    N.verify()
    snapshot=N.E.read(C.R.GREEN)
    records={r['id']:r for r in snapshot['records']}
    metrics=N.E.read(N.RAW/'PER_FRAME_METRICS.json')['GREEN150_MANUAL']
    predictions=N.E.read(N.RAW/'PREDICTIONS.json')['GREEN150']
    models={'R0':{},'N2':{},'PRIOR':{},'REAL_ONLY':{},'REPLAY':{},'REPLAY_CAP':{}}
    scores={k:{} for k in models}
    for label,arm in [('R0','R0'),('N2','A_N2'),('REPLAY','POSEFIX_RAW'),('REPLAY_CAP','POSEFIX_CAP8')]:
        models[label]={r['id']:r['predictions'][arm] for r in predictions}
        scores[label]={r['id']:r for r in metrics[arm]}
    for label,prefix in [('PRIOR','EXISTING'),('REAL_ONLY','FINETUNED')]:
        preds=N.E.read(C.RAW/f'{prefix}_PREDICTIONS.json')['GREEN150']
        mets=N.E.read(C.RAW/f'{prefix}_PER_FRAME_METRICS.json')['GREEN150_MANUAL']
        models[label]={r['id']:r['predictions']['POSEFIX_RAW'] for r in preds}
        scores[label]={r['id']:r for r in mets['POSEFIX_RAW']}
    contract=N.E.read(N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    perms=np.array(next(o for o in contract['objects'] if o['object_type']=='plastic_standard_110x110x15')['permutations'])
    return records,models,scores,perms


def audit():
    records,models,scores,perms=inputs();rows=[]
    for ident,record in records.items():
        N.F.verify(record['image']);N.F.verify(record['annotation'])
        im=cv2.imread(str(ROOT/record['image']['path']));inp=C.prepare_input(im,models['R0'][ident])
        points=inp['points'][:8];in_crop=(points[:,0]>=0)&(points[:,0]<288)&(points[:,1]>=0)&(points[:,1]<384)
        corners=C.transform_points(np.array([[0,0],[288,384]],float),np.linalg.inv(inp['matrix']))
        valid=scores['N2'][ident]['canonical_valid'];a=scores['N2'][ident]['canonical_errors'];b=scores['REPLAY'][ident]['canonical_errors']
        lost=[i for i,v in enumerate(valid) if v and a[i]<=10 and b[i]>10]
        gained=[i for i,v in enumerate(valid) if v and a[i]>10 and b[i]<=10]
        row=dict(id=ident,groups=record['groups'],session=record['session'],matched=scores['N2'][ident]['matched'],
                 canonical_valid=valid,lost=lost,gained=gained,
                 lost_native=[int(np.flatnonzero(perms[scores['REPLAY'][ident]['branch']]==i)[0]) for i in lost],
                 native_R0_outside_crop=np.flatnonzero(~in_crop).tolist(),crop_xyxy=corners.ravel().tolist(),
                 branch_N2=scores['N2'][ident]['branch'],branch_replay=scores['REPLAY'][ident]['branch'],
                 scores={k:v[ident] for k,v in scores.items()})
        rows.append(row)
    subsets={
        'all':lambda r:True,
        'handheld':lambda r:'handheld_multiview' in r['groups'],
        'forklift':lambda r:'handheld_multiview' not in r['groups'],
        'truncation':lambda r:'truncation' in r['groups'],
        'no_truncation_tag':lambda r:'truncation' not in r['groups'],
        'R0_points_outside_crop':lambda r:bool(r['native_R0_outside_crop']),
        'R0_all_points_inside_crop':lambda r:not r['native_R0_outside_crop']}
    sums={}
    for name,fn in subsets.items():
        rr=[r for r in rows if fn(r)]
        totals={}
        for model in models:
            e=[e for r in rr for e,v in zip(r['scores'][model]['canonical_errors'],r['canonical_valid']) if v]
            obs=[e for r in rr if r['matched'] for e,v in zip(r['scores'][model]['canonical_errors'],r['canonical_valid']) if v]
            totals[model]=dict(frames=len(rr),corners=len(e),correct10=sum(e0<=10 for e0 in e),
                PCK10=sum(e0<=10 for e0 in e)/len(e) if e else None,matched_corners=len(obs),
                mean_matched=float(np.mean(obs)) if obs else None,median_matched=float(np.median(obs)) if obs else None,
                P90_matched=float(np.quantile(obs,.9)) if obs else None)
        sums[name]=dict(models=totals,lost=sum(len(r['lost']) for r in rr),gained=sum(len(r['gained']) for r in rr),
                        branch_changes=sum(r['branch_N2']!=r['branch_replay'] for r in rr))
    bindings=[N.E.bound(p) for p in (C.R.GREEN,N.RAW/'PREDICTIONS.json',N.RAW/'PER_FRAME_METRICS.json',
        C.RAW/'EXISTING_PREDICTIONS.json',C.RAW/'FINETUNED_PREDICTIONS.json',C.RAW/'EXISTING_PER_FRAME_METRICS.json',C.RAW/'FINETUNED_PER_FRAME_METRICS.json')]
    write(DOC/'NUMERIC_AUDIT.json',dict(complete=True,subsets=sums,rows=rows,bindings=bindings,
        GT_used_only_for_diagnostics=True,new_training=False,new_pseudo_labels=False,source_code=N.E.bound(Path(__file__))))
    print('SUBSETS',json.dumps(sums,ensure_ascii=False),flush=True)


def peaks(prob,matrix,gt):
    p=prob.copy();peaks=[]
    for _ in range(4):
        y,x=np.unravel_index(np.argmax(p),p.shape)
        crop=np.array([[x*4,y*4]],float)
        q=C.transform_points(crop,np.linalg.inv(matrix))[0]
        region=prob[max(0,y-3):min(96,y+4),max(0,x-3):min(72,x+4)]
        peaks.append(dict(image_xy=q.tolist(),probability=float(prob[y,x]),mass_7x7=float(region.sum()),
                          error_to_manual_px=float(np.linalg.norm(q-gt))))
        p[max(0,y-5):min(96,y+6),max(0,x-5):min(72,x+6)]=-1
    return peaks


@torch.no_grad()
def heatmaps():
    N.setup();N.E.gpu();records,models,scores,perms=inputs();prepared={}
    for tag,(ident,k) in CASES.items():
        record=records[ident];N.F.verify(record['image']);N.F.verify(record['annotation'])
        im=cv2.imread(str(ROOT/record['image']['path']));x=C.prepare_input(im,models['R0'][ident])
        annotation=N.E.read(ROOT/record['annotation']['path']);gt,valid=C.R.annotation_arrays(annotation,True)
        assert valid[k]
        prepared[tag]=(ident,k,gt,x,record)
    entries={tag:dict(id=ident,canonical_corner=k,manual_xy=gt[k].tolist(),crop_matrix=x['matrix'].tolist(),
                      image=record['image'],annotation=record['annotation'],models={})
             for tag,(ident,k,gt,x,record) in prepared.items()}
    arrays={}
    for label,load in [('PRIOR',C.load_model),('REAL_ONLY',C.load_finetuned),('REPLAY',N.load_model)]:
        model=load().eval().requires_grad_(False)
        for tag,(ident,k,gt,x,record) in prepared.items():
            args=[torch.as_tensor(x[n],device='cuda')[None] for n in ('rgb','points','valid')]
            logits=model(*args);q_crop=expectation(logits)[0].cpu().numpy()
            q=C.transform_points(q_crop,np.linalg.inv(x['matrix']))
            saved=np.array(selected(models[label][ident])['keypoints_xy'])
            np.testing.assert_allclose(q[:8],saved[:8],atol=.005,rtol=0)
            branch=scores[label][ident]['branch'];native=int(np.flatnonzero(perms[branch]==k)[0])
            prob=logits[0,native].flatten().softmax(0).reshape(96,72).cpu().numpy()
            probs=peaks(prob,x['matrix'],gt[k])
            ent=float(-(prob*np.log(np.maximum(prob,1e-30))).sum())
            entries[tag]['models'][label]=dict(native=native,branch=branch,expectation_xy=q[native].tolist(),
                expectation_error_px=float(np.linalg.norm(q[native]-gt[k])),
                saved_error_px=scores[label][ident]['canonical_errors'][k],
                entropy_nats=ent,effective_support_cells=float(np.exp(ent)),peaks=probs,
                stored_prediction_parity_max_abs_px=float(np.max(np.abs(q[:8]-saved[:8]))))
            arrays[tag+'__'+label]=prob
        del model;gc.collect();torch.cuda.empty_cache()
    RAW.mkdir(parents=True,exist_ok=True);path=RAW/'HEATMAPS.npz'
    assert not path.exists(),'Never overwrite a completed diagnostic'
    np.savez(path,**arrays)
    write(DOC/'HEATMAP_DIAGNOSTIC.json',dict(complete=True,cases=entries,heatmaps=N.E.bound(path),
        models=[N.E.bound(C.PRIOR_CK),N.E.read(C.DOC/'FIT.json')['checkpoint'],N.E.read(N.DOC/'FIT.json')['checkpoint']],
        code=N.E.bound(Path(__file__)),new_training=False,modified_crops=False,checkpoint_or_hyperparameter_selection=False,
        note='Frozen existing models, original crop unchanged. Mode vs expectation is diagnosis only, not an argmax deployment proposal.'))
    for tag,row in entries.items():
        print(tag,{k:dict(expect_error=v['expectation_error_px'],peak_error=v['peaks'][0]['error_to_manual_px'],peak_xy=v['peaks'][0]['image_xy']) for k,v in row['models'].items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','heatmaps'));args=p.parse_args()
    audit() if args.stage=='audit' else heatmaps()
