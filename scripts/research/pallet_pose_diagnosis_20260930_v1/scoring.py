"""Reference-dependent diagnostics; all model predictions are read-only."""
import argparse
import copy
import time
from collections import Counter
import numpy as np
from . import run as C
from scripts.research.pallet_posefix_large_error_v1 import evaluate as EV, core as CORE
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as TWO

def references():
    pe,pop=EV.population_metadata();targets={item.frame_id:(item,meta) for item,meta in pop}
    split={r['id']:r for r in C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['heldout']}
    protocol=C.read(C.L.DOC/'EXPERIMENT_PROTOCOL.json')
    perms={r['object_type']:r['permutations'] for r in C.read(C.ROOT/protocol['symmetry']['path'])['objects']}
    out={}
    for row in C.metadata():
        i=row['id'];item,meta=targets[i];t=pe.E._legacy_forbidden_target(item)
        out[i]=dict(gt=np.asarray(t.keypoints_xy),box=np.asarray(t.box_xyxy),valid=np.asarray(t.keypoint_supervision_mask),
            permutations=perms[meta['object_type']],annotation=split[i]['annotation'])
    return out

def two_metric(i,pred,t,hw):
    c=CORE.selected(pred);detected=c is not None;matched=detected and EV.iou(c['box_xyxy'],t['box'])>=.5
    q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
    return dict(id=i,**TWO.measure(q,t['gt'],t['valid'],t['permutations'],hw,matched,detected))

def detection():
    start,cpu=time.monotonic(),time.process_time();rows=C.metadata();refs=references();ck=C.geo_checkpoint();_,gt=C.O.D.Pose.metadata('REAL_DEV')
    preds=C.read(C.RAW/'INPUT_PREDICTIONS.json');allrows={};metrics={}
    baseline=C.read(C.RAW/'E1_POSE_METRICS.json');twobase=C.read(C.RAW/'E1_2D_METRICS.json')
    # Independent parity against pre-existing same-GEO controls.
    old=C.read(C.ROOT/'data/pallet/results/pallet_clean_pose_minimal_v1/CURRENT_GEO_FRAME_METRICS_PRIVATE.json')
    for a,oldarm in [('identity','R0_GEO'),('OLD_REF217','OLD_REF_GEO')]:
        for r in rows:
            for key in ('translation_cm','rotation_deg'):
                np.testing.assert_allclose(baseline[a][r['id']][key],old[oldarm][r['id']][key],atol=1e-7,rtol=1e-7)
    for arm in ('identity','FULL125','OLD_REF217'):
        allrows[arm]={};metrics[arm]={}
        for r in rows:
            i=r['id'];p=preds[arm][i];t=refs[i];top=p['selected_index'];details=[]
            m=two_metric(i,p,t,r['hw']);assert (m['matched'],m['detected'])==(twobase[arm][i]['matched'],twobase[arm][i]['detected'])
            for ix,c in enumerate(p['candidates']):
                changed=copy.deepcopy(p);changed['selected_index']=ix
                rec=C.candidates(changed,r,ck)
                details.append(dict(index=ix,score=c['score'],IoU=EV.iou(c['box_xyxy'],t['box']),
                    GEO_name=rec['GEO_name'],GEO=C.metric(i,rec['GEO_pose'],gt[i]),
                    hypotheses=[dict(name=h['name'],metric=C.metric(i,h['pose'],gt[i])) for h in rec['hypotheses']]))
            best=min(details,key=lambda d:(-d['IoU'],d['index'])) if details else None
            metrics[arm][i]=best['GEO'] if best else dict(id=i,available=False)
            allrows[arm][i]=dict(id=i,recording=r['recording'],severity=r['severity'],selected_index=top,detected=m['detected'],matched=m['matched'],
                category='NO_DETECTION' if not details else 'MATCHED_OBJECT' if m['matched'] else 'CORRECT_BOX_IN_POOL_WRONG_SELECTION' if best['IoU']>=.5 else 'NO_MATCHING_BOX_IN_STORED_POOL',
                selected_IoU=details[top]['IoU'] if top is not None else None,best_IoU_index=best['index'] if best else None,boxes=details)
    groups=C.E.group_ids(rows);summary={}
    for g,ids in groups.items():
        rr=[r for r in rows if r['id'] in ids];summary[g]={}
        for a in allrows:
            tail=sorted(ids,key=lambda i:-(baseline[a][i].get('translation_cm',float('inf'))))[:max(1,int(np.ceil(len(ids)*.1)))]
            summary[g][a]=dict(categories=dict(Counter(allrows[a][i]['category'] for i in ids)),tail_largest_T_ids=tail,
                tail_categories=dict(Counter(allrows[a][i]['category'] for i in tail)),
                best_box_oracle_summary=C.summarize(metrics[a][i] for i in ids),best_box_minus_native=C.paired(baseline[a],metrics[a],rr,g in ('NATURAL99','CLEAN29')))
    C.save(C.RAW/'E2_BOX_DIAGNOSTIC.json',allrows);C.save(C.RAW/'E2_BOX_ORACLE_METRICS.json',metrics)
    C.save(C.DOC/'E2_DETECTION_SUMMARY.json',dict(groups=summary,old_GEO_metric_parity_frames=256,
        limitation='IoU>=.5 is association diagnosis only, never a pose gate. Reference best box changes selected detection; no new proposals. Top10 percent T tail is descriptive.'))
    C.ledger('E2_boxes',start,cpu)

def visibility():
    start,cpu=time.monotonic(),time.process_time();rows=C.metadata();refs=references();ck=C.geo_checkpoint()
    base=C.read(C.RAW/'INPUT_PREDICTIONS.json')['identity'];basecand=C.read(C.RAW/'E1_CANDIDATES.json')['identity'];base2=C.read(C.RAW/'E1_2D_METRICS.json')['identity']
    human={};bindings=[]
    for name in ('DAYTIME_VISIBILITY_AMENDMENTS.json','COVERAGE_GAP_AMENDMENTS.json'):
        p=C.ROOT/'data/evaluation/pallet_eval_v1/review'/name;d=C.read(p);bindings.append(C.bind(p));assert 'prediction-blinded' in d['protocol']
        for fid,rr in d['frames'].items():
            for j,state in rr.items():
                if j.isdigit():
                    key=(fid.replace('__',':',1),int(j));assert key not in human or human[key]==state;human[key]=state
    outputs={kind:{} for kind in ('baseline','occluded','visible','both')};support={};two_outputs={kind:{} for kind in outputs}
    for r in rows:
        i=r['id'];t=refs[i];q=np.asarray(CORE.selected(base[i])['keypoints_xy']);b=base2[i]['branch'];perm=np.array(t['permutations'][b])
        ann=C.read(C.ROOT/t['annotation']['path'])['objects'][0]
        sets={'occluded':[],'visible':[]};reviewed=Counter();excluded=[]
        for j in range(8):
            state=human.get((i,j),'unknown');reviewed[state]+=1
            if state not in ('o','v'):continue
            entry=ann['keypoint_annotations'][j];expected={'v':(2,'visible'),'o':(1,'occluded')}[state]
            assert (entry['visibility'],entry['reason'])==expected
            predidx=int(np.flatnonzero(perm==j)[0])
            if not base2[i]['matched'] or not t['valid'][j] or not np.isfinite(q[predidx]).all() or (q[predidx]==-1).all():
                excluded.append(dict(canonical=j,predicted=predidx,state=state,reason='unmatched_or_invalid'));continue
            sets['occluded' if state=='o' else 'visible'].append((predidx,j))
        support[i]=dict(recording=r['recording'],severity=r['severity'],reviewed=dict(reviewed),replacement_indices=sets,
            canonical_branch=b,canonical_permutation=perm.tolist(),excluded=excluded,matched=base2[i]['matched'])
        for kind in outputs:
            pp=copy.deepcopy(base[i]);new=q.copy()
            replace=sets['occluded']+sets['visible'] if kind=='both' else sets.get(kind,[])
            for predidx,j in replace:new[predidx]=t['gt'][j]
            CORE.selected(pp)['keypoints_xy']=new.tolist();np.testing.assert_array_equal(q[8],new[8])
            outputs[kind][i]=C.candidates(pp,r,ck)
            two_outputs[kind][i]=two_metric(i,pp,t,r['hw'])
    C.save(C.RAW/'E4_REFERENCE_INTERVENTION_CANDIDATES.json',outputs);C.save(C.RAW/'E4_VISIBILITY_SUPPORT.json',support)
    C.save(C.RAW/'E4_REVIEW_2D_METRICS.json',two_outputs)
    _,gt=C.O.D.Pose.metadata('REAL_DEV');metrics={}
    for kind,values in outputs.items():
        for mode in ('GEO','held'):
            mm={}
            for i,rec in values.items():
                p=rec['GEO_pose'] if mode=='GEO' else next((h['pose'] for h in rec['hypotheses'] if h['name']==basecand[i]['GEO_name']),dict(available=False))
                mm[i]=C.metric(i,p,gt[i])
            metrics[kind+'_'+mode]=mm
    groups=C.E.group_ids(rows);summaries={}
    for group,ids in groups.items():
        rr=[r for r in rows if r['id'] in ids];review=Counter()
        for i in ids:review.update(support[i]['reviewed'])
        summaries[group]=dict(frames=len(ids),total_corner_slots=8*len(ids),reviewed=dict(review),
            replacement_corners={k:sum(len(support[i]['replacement_indices'][k]) for i in ids) for k in ('visible','occluded')},
            affected_frames={k:sum(bool(support[i]['replacement_indices'][k]) for i in ids) for k in ('visible','occluded')},
            models={a:C.summarize(v[i] for i in ids) for a,v in metrics.items()},
            paired={a:C.paired(metrics['baseline_'+a.rsplit('_',1)[1]],v,rr,group in ('NATURAL99','CLEAN29')) for a,v in metrics.items() if not a.startswith('baseline')})
    C.save(C.RAW/'E4_METRICS.json',metrics);C.save(C.DOC/'E4_SUMMARY.json',dict(groups=summaries,inputs=bindings,
        limitations=['Generic human occluded; external/self not separated','Reference 2D replaced only on matched object, existing whole-object canonical branch','Effects are conditional interventions, not additive cause fractions or total refiner upper bounds','Unknown points and center8 retained']))
    C.ledger('E4',start,cpu)

def realft():
    start,cpu=time.monotonic(),time.process_time();rows=C.metadata();pp=C.read(C.RAW/'E6_PREDICTIONS.json');ck=C.geo_checkpoint()
    rec={r['id']:C.candidates(pp[r['id']],r,ck) for r in rows};C.save(C.RAW/'E6_CANDIDATES.json',rec)
    refs=references();_,gt=C.O.D.Pose.metadata('REAL_DEV');mm={i:C.metric(i,v['GEO_pose'],gt[i]) for i,v in rec.items()}
    two={r['id']:two_metric(r['id'],pp[r['id']],refs[r['id']],r['hw']) for r in rows};base=C.read(C.RAW/'E1_POSE_METRICS.json');groups={}
    for group,ids in C.E.group_ids(rows).items():
        rr=[r for r in rows if r['id'] in ids]
        groups[group]=dict(pose=C.summarize(mm[i] for i in ids),twoD=TWO.summary([two[i] for i in ids]),
            paired={a:C.paired(base[a],mm,rr,group in ('NATURAL99','CLEAN29')) for a in ('identity','FULL125','OLD_REF217')})
    C.save(C.RAW/'E6_POSE_METRICS.json',mm);C.save(C.RAW/'E6_2D_METRICS.json',two);C.save(C.DOC/'E6_SUMMARY.json',groups)
    C.ledger('E6_score',start,cpu)

def e3():
    start,cpu=time.monotonic(),time.process_time();rows=[r for r in C.metadata() if r['severity']=='CLEAN'];ck=C.geo_checkpoint()
    qc=C.read(C.RAW/'E3_CPU_QC.json');qo=C.read(C.RAW/'E3_R0_PREDICTIONS.json');plans=C.read(C.RAW/'E3_MASK_PLANS.json')
    refs=references();_,gt=C.O.D.Pose.metadata('REAL_DEV');byid={r['id']:r for r in rows}
    pred={kind:{a:{cond:{} for cond in ('CC','CO','OC','OO','nativeOO')} for a in ('identity','PRIOR1','FULL125')} for kind in ('cover','avoid')}
    pairmeta={};coverage={}
    from .inference import masked
    from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import cover
    for ix,r in enumerate(rows):
        i=r['id'];coverage[i]={};pairmeta[i]={}
        for a in ('PRIOR1','FULL125'):
            d=C.read(C.RAW/'e3_frames'/a/f'{ix:03d}.json');assert d['id']==i
            for kind,v in d['conditions'].items():
                pairmeta[i][kind]={k:v[k] for k in ('same_object_pair','pair_box_iou')}
                for cond,p in v['predictions'].items():pred[kind][a][cond][i]=p
        im=C.cv2.imread(str(C.ROOT/r['image']['path']))
        for kind,native in qo[i].items():
            changed=copy.deepcopy(qc[i]);nc=CORE.selected(native)
            if pairmeta[i][kind]['same_object_pair']:
                CORE.selected(changed)['keypoints_xy']=copy.deepcopy(nc['keypoints_xy'])
                for cond in ('CO','OO'):pred[kind]['identity'][cond][i]=changed
            for cond in ('CC','OC'):pred[kind]['identity'][cond][i]=qc[i]
            pred[kind]['identity']['nativeOO'][i]=native
            rect=plans[i]['rectangles'][kind]['rectangle'];q=np.array(CORE.selected(qc[i])['keypoints_xy'])
            mask=np.isfinite(q).all(1)&~(q==-1).all(1);mask[8]=False
            occ=masked(im,plans[i],kind)
            coverage[i][kind]=dict(rectangle=rect,covered_CPU_qC=np.flatnonzero(cover(q,rect)&mask).tolist(),
                covered_reference_corner_slots=np.flatnonzero(cover(refs[i]['gt'][:8],rect)).tolist(),
                masked_BGR_sha256=C.hashlib.sha256(occ.tobytes()).hexdigest(),mask_area_pixels=rect[2]*rect[3],
                qO_minus_qC_corner_mean_px=float(np.linalg.norm(np.array(nc['keypoints_xy'])[:8]-q[:8],axis=1).mean()) if nc else None)
    records={};metrics={};two={};following={}
    for kind,arms in pred.items():
        records[kind]={};metrics[kind]={};two[kind]={};following[kind]={}
        for a,conditions in arms.items():
            records[kind][a]={};metrics[kind][a]={};two[kind][a]={};following[kind][a]={}
            for cond,pp in conditions.items():
                rr={i:C.candidates(p,byid[i],ck) for i,p in pp.items()};records[kind][a][cond]=rr
                metrics[kind][a][cond]={r['id']:C.metric(r['id'],rr[r['id']]['GEO_pose'],gt[r['id']]) if r['id'] in rr else dict(id=r['id'],available=False) for r in rows}
                two[kind][a][cond]={i:two_metric(i,p,refs[i],byid[i]['hw']) for i,p in pp.items()}
                following[kind][a][cond]={}
                for i,p in pp.items():
                    c=CORE.selected(p);q=np.array(CORE.selected(qc[i])['keypoints_xy']);v=np.isfinite(q).all(1)&~(q==-1).all(1);v[8]=False
                    z=np.array(c['keypoints_xy']) if c else np.full_like(q,np.nan);valid=v&np.isfinite(z).all(1)&~(z==-1).all(1)
                    following[kind][a][cond][i]=dict(clean_prediction_restoration_mean_px=float(np.linalg.norm(z[valid]-q[valid],axis=1).mean()) if valid.any() else None,
                        eligible_corners=int(v.sum()),available_corners=int(valid.sum()),is_training_target=False)
    C.save(C.RAW/'E3_CANDIDATES.json',records);C.save(C.RAW/'E3_POSE_METRICS.json',metrics);C.save(C.RAW/'E3_2D_METRICS.json',two)
    C.save(C.RAW/'E3_CLEAN_PREDICTION_RESTORATION.json',following);C.save(C.RAW/'E3_MASK_COVERAGE.json',coverage);C.save(C.RAW/'E3_PAIR_STATUS.json',pairmeta)
    groups={'CLEAN29':[r['id'] for r in rows]};groups.update({'recording:'+rec:[r['id'] for r in rows if r['recording']==rec] for rec in sorted({r['recording'] for r in rows})})
    summary={}
    for kind in pred:
        summary[kind]={}
        for group,ids in groups.items():
            rr=[r for r in rows if r['id'] in ids];s={};pairs={}
            for a in pred[kind]:
                s[a]={}
                for cond,mm in metrics[kind][a].items():
                    s[a][cond]=dict(pose=C.summarize(mm[i] for i in ids),twoD=TWO.summary([two[kind][a][cond][i] for i in ids if i in two[kind][a][cond]]),
                        missing_controlled_rows=[i for i in ids if i not in two[kind][a][cond]],
                        clean_prediction_restoration=C.M.distribution(following[kind][a][cond][i]['clean_prediction_restoration_mean_px'] for i in ids if i in following[kind][a][cond] and following[kind][a][cond][i]['clean_prediction_restoration_mean_px'] is not None))
                    if a!='identity':pairs[a+'_'+cond+'-minus-identity']=C.paired(metrics[kind]['identity'][cond],mm,rr,group=='CLEAN29')
                if a!='identity':
                    for before,after in [('CO','OO'),('OO','nativeOO'),('CC','OC'),('CC','OO')]:
                        pairs[a+'_'+after+'-minus-'+before]=C.paired(metrics[kind][a][before],metrics[kind][a][after],rr,group=='CLEAN29')
            summary[kind][group]=dict(models=s,paired=pairs,full_denominator=len(ids),same_object_pairs=sum(pairmeta[i][kind]['same_object_pair'] for i in ids),
                stress_qO_minus_qC_px=C.M.distribution(coverage[i][kind]['qO_minus_qC_corner_mean_px'] for i in ids if coverage[i][kind]['qO_minus_qC_corner_mean_px'] is not None))
    C.save(C.DOC/'E3_SUMMARY.json',summary);C.ledger('E3_score',start,cpu)

def stress():
    start,cpu=time.monotonic(),time.process_time();rows=[r for r in C.metadata() if r['severity']=='CLEAN'];ck=C.geo_checkpoint()
    pred=C.read(C.RAW/'E5_STRESS_PREDICTIONS.json');base=C.read(C.RAW/'E3_CPU_QC.json');plans=C.read(C.RAW/'E5_STRESS_PLAN.json')
    refs=references();_,gt=C.O.D.Pose.metadata('REAL_DEV');metrics={};twod={};restoration={};candidates={}
    for a,kinds in pred.items():
        metrics[a]={};twod[a]={};restoration[a]={};candidates[a]={}
        for kind,pp in kinds.items():
            metrics[a][kind]={};twod[a][kind]={};restoration[a][kind]={};candidates[a][kind]={}
            for r in rows:
                i=r['id'];p=pp[i];rec=C.candidates(p,r,ck);candidates[a][kind][i]=rec;metrics[a][kind][i]=C.metric(i,rec['GEO_pose'],gt[i])
                twod[a][kind][i]=two_metric(i,p,refs[i],r['hw'])
                q=np.array(CORE.selected(base[i])['keypoints_xy']);z=np.array(CORE.selected(p)['keypoints_xy']);ix=plans[i]['corners']
                restoration[a][kind][i]=dict(corners=ix,errors_px=np.linalg.norm(z[ix]-q[ix],axis=1),
                    unchanged_input_corner_errors_px=np.linalg.norm(z[[j for j in range(8) if j not in ix]]-q[[j for j in range(8) if j not in ix]],axis=1))
    C.save(C.RAW/'E5_STRESS_CANDIDATES.json',candidates);C.save(C.RAW/'E5_STRESS_POSE_METRICS.json',metrics);C.save(C.RAW/'E5_STRESS_2D_METRICS.json',twod);C.save(C.RAW/'E5_STRESS_RESTORATION.json',restoration)
    groups={'CLEAN29':[r['id'] for r in rows]};groups.update({'recording:'+rec:[r['id'] for r in rows if r['recording']==rec] for rec in sorted({r['recording'] for r in rows})})
    summary={}
    for group,ids in groups.items():
        rr=[r for r in rows if r['id'] in ids];models={};pairs={}
        for a in pred:
            models[a]={}
            for kind in pred[a]:
                errors=[e for i in ids for e in restoration[a][kind][i]['errors_px']]
                models[a][kind]=dict(pose=C.summarize(metrics[a][kind][i] for i in ids),twoD=TWO.summary([twod[a][kind][i] for i in ids]),
                    clean_R0_restoration=dict(distribution=C.M.distribution(errors),recovered30to10=sum(e<=10 for e in errors),N=len(errors)),
                    unchanged_input_corner_damage=C.M.distribution(e for i in ids for e in restoration[a][kind][i]['unchanged_input_corner_errors_px']))
                if a!='identity':pairs[a+'_'+kind+'-minus-identity']=C.paired(metrics['identity'][kind],metrics[a][kind],rr,group=='CLEAN29')
            pairs[a+'_correlated-minus-independent']=C.paired(metrics[a]['independent'],metrics[a]['correlated'],rr,group=='CLEAN29')
        summary[group]=dict(models=models,paired=pairs)
    C.save(C.DOC/'E5_STRESS_SUMMARY.json',summary);C.ledger('E5_stress_score',start,cpu)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['detection','visibility','realft','e3','stress']);a=p.parse_args()
    C.torch.set_num_threads(2);C.cv2.setNumThreads(1)
    {'detection':detection,'visibility':visibility,'realft':realft,'e3':e3,'stress':stress}[a.stage]()

if __name__=='__main__':main()
