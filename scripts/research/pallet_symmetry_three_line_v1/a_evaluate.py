"""Paired A evaluation on immutable serialized predictions, no model selection."""
import csv,math
from collections import Counter
import cv2,numpy as np
import env as E
from a_data import records,target
from metrics import measure,summarize,contrast
from audit_math import symmetry_error

ARMS=['R0']+[f'{a}_seed{s}' for s in [1,2,3] for a in ['INDEXED','EQUIV']]

def targets():
    result={}
    for cohort,split in [('RECT','RECT_SYNTH_VAL'),('SQUARE','SQUARE_DEV')]:
        table={}
        for r in records(cohort,'val'):
            a=target(r);assert len(a)==1
            a=a[0].astype(float);size=np.array(r['raw_hw'])[::-1]+200;kp=a[5:].reshape(9,3)
            xy=kp[:,:2]*size-100;c=a[1:3]*size-100;half=a[3:5]*size/2
            table[r['id']]=dict(points=xy,valid=kp[:,2]>0,box=np.r_[c-half,c+half],raw_hw=r['raw_hw'],
              perms=np.array(r['permutations']),group_order=r['group_order'],
              session=r['id'].split('__')[0] if cohort=='SQUARE' else None,
              source=r.get('source','square_task'),asset=r.get('asset','square_task'))
        result[split]=table
    pe=E.C.old('paper_evaluation');meta={r['frame_id']:r for r in E.read(pe.POS)['items']}
    perms=np.array(E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations'])
    table={}
    for item in pe.population().positive.items:
        t=pe.E._legacy_forbidden_target(item);im=cv2.imread(str(E.ROOT/item.image));assert im is not None
        table[item.frame_id]=dict(points=np.asarray(t.keypoints_xy),valid=np.asarray(t.keypoint_supervision_mask),box=np.asarray(t.box_xyxy),
          raw_hw=im.shape[:2],perms=perms,group_order=2,session=meta[item.frame_id]['session_id'],source='paper_DEV',asset=meta[item.frame_id].get('object_type','paper_C2'))
    result['RECT_DEV']=table
    return result

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);area=lambda x:float(np.maximum(x[2:]-x[:2],0).prod())
    inter=float(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod())
    return inter/max(area(a)+area(b)-inter,1e-12)

def confidence_selected_match(candidates,gt_box):
    top=max(candidates,key=lambda x:x['score']) if candidates else None
    return top,top is not None and iou(top['box_xyxy'],gt_box)>=.5

def summarize_all(rows):
    valid=[r for r in rows if r['evaluable']];summary=summarize(valid)
    fixed=np.concatenate([r['fixed_errors'] for r in valid])
    summary.update(total_frames=len(rows),non_evaluable_frame_ids=[r['id'] for r in rows if not r['evaluable']],
      detected=sum(r['detected'] for r in rows),matched=sum(r['matched'] for r in rows),detection_coverage=sum(r['matched'] for r in rows)/len(rows),
      fixed_pooled_median_px=float(np.median(fixed)),fixed_pooled_P90_px=float(np.quantile(fixed,.9)),
      evaluation_orbit_counts=dict(Counter(r['branch'] for r in valid)),outside_raw_corners=sum(r['outside_raw_corners'] for r in rows))
    return summary

def main():
    assert E.read(E.DOC/'A/INFERENCE_COMPLETE.json')['complete']
    gt=targets();allrows={};summaries={};contrasts={};subgroups={};damage={};flat=[]
    for split,table in gt.items():
        allrows[split]={};summaries[split]={};subgroups[split]={};damage[split]={}
        for arm in ARMS:
            payload=E.read(E.RAW/f'A/predictions/{split}/{arm}.json');assert payload['complete'] and not payload['GT_input']
            rows=[]
            for p in payload['records']:
                t=table[p['id']];top,matched=confidence_selected_match(p['candidates'],t['box'])
                pred=np.asarray(top['keypoints_xy']) if matched else np.full((9,2),np.nan)
                score=measure(pred,t['points'],t['valid'],t['perms'],t['raw_hw'])
                raw=np.asarray(top['keypoints_xy'])[:8] if top is not None else np.empty((0,2));h,w=t['raw_hw']
                score.update(id=p['id'],detected=top is not None,matched=matched,group_order=t['group_order'],session=t['session'],
                  outside_raw_corners=int(((raw[:,0]<0)|(raw[:,0]>=w)|(raw[:,1]<0)|(raw[:,1]>=h)).sum()))
                if score['evaluable']:
                    fixed=symmetry_error(pred,t['points'],t['valid'],np.arange(9)[None],math.hypot(*t['raw_hw']))
                    score['fixed_errors']=fixed['errors'][:8][fixed['gt_mask'][:8]].tolist()
                rows.append(score)
                flat.append(dict(split=split,arm=arm,id=p['id'],evaluable=score['evaluable'],detected=score['detected'],matched=matched,
                  C=t['group_order'],session=t['session'],E_sym=score.get('E_sym'),E_fixed=score.get('E_fixed'),
                  frame_mean_px=score.get('frame_mean_px'),branch=score.get('branch'),annotated_corners=score.get('n_corners'),
                  center_error_px=score.get('center_error_px'),outside_raw_corners=score['outside_raw_corners']))
            assert len(rows)==len(table)
            allrows[split][arm]=rows;summaries[split][arm]=summarize_all(rows)
            subgroups[split][arm]={f'C{k}':summarize_all([r for r in rows if r['group_order']==k]) for k in sorted({r['group_order'] for r in rows})}
        valid_ids=[r['id'] for r in allrows[split]['R0'] if r['evaluable']]
        evaluated={a:[r for r in rows if r['evaluable']] for a,rows in allrows[split].items()}
        assert all([r['id'] for r in v]==valid_ids for v in evaluated.values())
        clusters=None if split=='RECT_SYNTH_VAL' else [r['session'] for r in evaluated['R0']]
        primary=contrast([[r['E_sym'] for r in evaluated[f'EQUIV_seed{s}']] for s in [1,2,3]],
                         [[r['E_sym'] for r in evaluated[f'INDEXED_seed{s}']] for s in [1,2,3]],clusters)
        lo,hi=primary['CI95'];gain='SUPPORTED' if hi<0 and primary['improved_seeds']>=2 else 'WORSENED' if lo>0 else 'UNRESOLVED'
        comparisons=dict(EQUIV_minus_INDEXED=primary)
        for arm in ['INDEXED','EQUIV']:
            comparisons[f'{arm}_minus_R0']=contrast([[r['E_sym'] for r in evaluated[f'{arm}_seed{s}']] for s in [1,2,3]],
              [[r['E_sym'] for r in evaluated['R0']]]*3,clusters)
        contrasts[split]=dict(geometry_gain=gain,definition_integrity='PASS',**comparisons)
        for seed in [1,2,3]:
            b=evaluated[f'INDEXED_seed{seed}'];n=evaluated[f'EQUIV_seed{seed}'];delta=np.array([y['frame_mean_px']-x['frame_mean_px'] for x,y in zip(b,n)])
            be=np.concatenate([r['errors'] for r in b]);ne=np.concatenate([r['errors'] for r in n]);good=be<5
            damage[split][seed]=dict(delta_mean_px=float(delta.mean()),delta_median_px=float(np.median(delta)),delta_P90_px=float(np.quantile(delta,.9)),
              improved_frames=int((delta<-1e-9).sum()),harmed_frames=int((delta>1e-9).sum()),unchanged_frames=int((abs(delta)<=1e-9).sum()),
              good5_to_bad10=int((good&(ne>10)).sum()),good_baseline_points=int(good.sum()),
              evaluation_branch_changed=sum(x['branch']!=y['branch'] for x,y in zip(b,n)))
        print('A RESULT',split,gain,primary,flush=True)
    E.freeze(E.DOC/'A/history/RESULTS_BEFORE_MAIN.json',E.read(E.DOC/'A/results_and_intervals.json'))
    E.write(E.RAW/'A/EVALUATION_FULL_METRICS.json',allrows)
    with (E.DOC/'A/per_frame_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    E.write(E.DOC/'A/results_and_intervals.json',dict(status='2D_COMPLETE_POSE_PENDING',fits=12,updates=24000,summary=summaries,contrasts=contrasts,
      C1_C2_C4_subgroups=subgroups,damage=damage,pose='PENDING',checkpoint_selection='last only, unchanged',
      source='all original source val, previously exposed to network development',real='reused DEV; square session interleave',
      primary='EQUIV minus INDEXED; negative is better; three paired seeds',secondary='vs R0 exploratory, no multiplicity adjustment',
      no_noninferiority_or_overall_safety_claim=True))
if __name__=='__main__':main()
