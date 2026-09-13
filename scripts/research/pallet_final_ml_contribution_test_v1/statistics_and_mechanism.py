"""Same-seed contrasts; same paired resample across seeds, never n*3 frames."""
import csv
import numpy as np
from common import *

def load():
    pe=old('paper_evaluation');ag=old('aggregate_results')
    pair=pe.population();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i in pair.positive.items}
    metadata={i['frame_id']:i for i in read(pe.POS)['items']}
    directories={'R0':LINE/'evaluation/R0',**{f'L{s}':LINE/f'evaluation/image_line_only_seed{s}' for s in (1,2,3)},
                 **{f'P{s}':BRAW/f'evaluation/P{s}' for s in (1,2,3)}}
    baseline=read(LINE/'baseline/FULL_CANDIDATES.json')['frames']
    stores={};poses={};preds={};summaries={};bindings={}
    for name,path in directories.items():
        if name=='R0':
            frames=baseline;csv_path=pe.OLDCSV;pose_path=POSE/'POSE_PER_FRAME_BY_ARM.json';pose_arm='R0';metric_path=pe.OLD2D
        else:
            frames=read(path/'PREDICTIONS.json')['frames'];csv_path=path/'PAPER_2D_per_frame.csv';pose_path=path/'POSE_PER_FRAME_BY_ARM.json'
            pose_arm=name if name.startswith('P') else f'image_line_only_seed{name[1]}';metric_path=path/'PAPER_2D.json'
        bindings[name]={str(p.relative_to(ROOT)):sha(p) for p in [csv_path,pose_path,metric_path]}
        csv_rows=ag.load_keypoint_rows(csv_path);store={};prediction={}
        for item in pair.positive.items:
            key=item.frame_id;target=targets[key];candidates=frames[pe.canonical_key(item.image)]
            top=max(candidates,key=lambda c:c['score']) if candidates else None
            valid=top is not None and pe.E._box_iou(np.asarray(top['box_xyxy']),target.box_xyxy)>=.5 and top['keypoints_xy'] is not None
            points=np.asarray(top['keypoints_xy'],float) if valid else None
            errors=np.linalg.norm(points-target.keypoints_xy,axis=-1)[target.keypoint_supervision_mask] if valid else np.empty(0)
            assert np.isfinite(errors).all() and np.allclose(errors,csv_rows[key]['errors'],atol=5.01e-7,rtol=0)
            store[key]=dict(frame_id=key,session_id=metadata[key]['session_id'],errors=errors)
            prediction[key]=points
        pose_rows=read(pose_path)['per_frame'][pose_arm]
        ps={r['frame_id']:r for r in pose_rows};assert len(ps)==len(pose_rows)
        summaries[name]=summarize(store,ps,targets)
        # Use full-precision predictions+canonical targets, not CSV rounding.
        canonical=read(metric_path)['metrics']['box_and_keypoint_2d']
        for metric in ('keypoint_location_median_px','keypoint_location_p90_px'):
            assert abs(summaries[name]['2d'][metric]-canonical[metric])<1e-10,(name,metric)
        stores[name]=store;poses[name]=ps;preds[name]=prediction
    write(B/'REAL_METRIC_SOURCE_BINDING.json',bindings)
    write(B/'PER_SEED_REAL.json',dict(complete=True,methods=summaries,positive=319,negative=2689,
        role='reused DEV',full_precision_2d_recomputed_canonical_exact=True,
        Proj_definition='fraction of supervised keypoints within threshold pixels; exact established paper definition, not frame-mean thresholding',
        coverage_definition='matched frames and available supervised points / all319 annotated supervised points; not every frame has all9 supervised',
        lateral_depth='Not available in frozen per-frame pose scorer. No new pose metric or recomputation introduced.'))
    return stores,poses,preds,targets,metadata,summaries


def summarize(store,poses,targets):
    ag=old('aggregate_results');rows=list(store.values());e=np.concatenate([r['errors'] for r in rows]);fm=np.array([r['errors'].mean() for r in rows if len(r['errors'])])
    kp=ag.keypoint_summary(rows);kp.update(frame_mean_px=float(fm.mean()),gross20=float(np.mean(e>20)),
        Proj_at={str(t):float(np.mean(e<=t)) for t in (5,10,20)},frame_mean_success_at={str(t):float(np.mean(fm<=t)) for t in (5,10,20)},
        supervised_point_coverage=len(e)/sum(int(t.keypoint_supervision_mask.sum()) for t in targets.values()))
    pose=ag.pose_summary(list(poses.values()));pose['yaw_median_deg']=float(np.median([r['yaw_error_deg'] for r in poses.values()]));pose['coverage']=len(poses)/319
    return {'2d':kp,'6d':pose}


def paired(left,right,metric):
    ag=old('aggregate_results');allstores=left+right
    keys=sorted(set.intersection(*(set(s) for s in allstores)))
    # Coverage disagreement independently fails G4. For downstream metrics,
    # report the explicitly conditional common-frame contrast and exclusions.
    # The 2D stores retain all319 even if a frame has no supervised errors.
    excluded=[len(s)-len(keys) for s in allstores]
    if metric in ag.KEYPOINT_METRICS:assert not any(excluded)
    sessions=sorted({left[0][k]['session_id'] for k in keys});assert len(sessions)==13
    for k in keys:assert len({s[k]['session_id'] for s in allstores})==1
    result={};observed=None
    for scheme in ('frame_level','session_cluster'):
        lookup={k:i for i,k in enumerate(keys)} if scheme=='frame_level' else {k:sessions.index(left[0][k]['session_id']) for k in keys}
        n=len(keys) if scheme=='frame_level' else len(sessions)
        series=[ag.prepare_series(s,keys,metric,lookup) for s in allstores]
        ones=np.ones((1,n),np.int32)
        values=[ag.weighted_statistic(v,g,ones,metric,d)[0] for v,g,d in series]
        if observed is None:observed=float(np.mean(np.array(values[:3])-np.array(values[3:])))
        rng=np.random.default_rng(STAT_SEED);draws=np.empty(BOOT)
        for start in range(0,BOOT,128):
            count=min(128,BOOT-start);weights=rng.multinomial(n,np.ones(n)/n,size=count).astype(np.int32)
            v=np.stack([ag.weighted_statistic(v,g,weights,metric,d) for v,g,d in series])
            draws[start:start+count]=(v[:3]-v[3:]).mean(0)
        result[scheme]=ag.interval(draws,metric in ag.LOWER)
    return dict(difference=observed,paired_frames=len(keys),paired_sessions=len(sessions),resamples=BOOT,seed=STAT_SEED,
        excluded_frames_by_seed_left=excluded[:3],excluded_frames_by_seed_right=excluded[3:],coverage_exact=not any(excluded),
        estimand='mean_s(metric(L_s,resampled IDs)-metric(P_s,same IDs)); same resample across all3 seeds; R0 not replicated as independent data',**result)


def statistics(stores,poses,summaries):
    ag=old('aggregate_results');results={}
    for metric in ag.METRICS:
        source=stores if metric in ag.KEYPOINT_METRICS else poses
        results[metric]=paired([source[f'L{s}'] for s in (1,2,3)],[source[f'P{s}'] for s in (1,2,3)],metric)
        print('PAIRED_STAT',metric,results[metric]['difference'],results[metric]['session_cluster'],flush=True)
    write(B/'REAL_PAIRED_STATISTICS.json',dict(complete=True,comparisons=results,no_independent_seed_inflation=True,
        source_sha256=sha(B/'REAL_METRIC_SOURCE_BINDING.json'),full_precision_not_CSV_rounding=True))
    mean=lambda arm,section,key:float(np.mean([summaries[f'{arm}{s}'][section][key] for s in (1,2,3)]))
    m='keypoint_location_median_px';r=results[m]
    direction={k:mean('L','6d',k)<mean('P','6d',k) if k in ag.LOWER else mean('L','6d',k)>mean('P','6d',k) for k in ag.POSE_METRICS}
    harm={k:results[k]['session_cluster']['low']>0 if k in ag.LOWER else results[k]['session_cluster']['high']<0 for k in ag.POSE_METRICS}
    coverage=all(set(poses[f'L{s}'])==set(poses[f'P{s}'])==set(poses['R0']) for s in (1,2,3))
    gates=dict(G1=r['difference']<0 and r['session_cluster']['high']<0,
        G2=mean('L','2d','keypoint_location_p90_px')<=mean('P','2d','keypoint_location_p90_px'),
        G3=mean('L','2d','gross20')<=mean('P','2d','gross20'),G4=coverage,
        G5=sum(direction.values())>=3 and not any(harm.values()))
    reverse_direction={k:mean('P','6d',k)<mean('L','6d',k) if k in ag.LOWER else mean('P','6d',k)>mean('L','6d',k) for k in ag.POSE_METRICS}
    reverse_harm={k:results[k]['session_cluster']['high']<0 if k in ag.LOWER else results[k]['session_cluster']['low']>0 for k in ag.POSE_METRICS}
    reverse=(r['session_cluster']['low']>0 and mean('P','2d','keypoint_location_p90_px')<=mean('L','2d','keypoint_location_p90_px')
        and mean('P','2d','gross20')<=mean('L','2d','gross20') and coverage and sum(reverse_direction.values())>=3 and not any(reverse_harm.values()))
    local=all(mean(arm,'2d',m)<summaries['R0']['2d'][m] and sum(summaries[f'{arm}{s}']['2d'][m]<summaries['R0']['2d'][m] for s in (1,2,3))>=2 for arm in ('L','P'))
    verdict='LINE_BIAS_DEVELOPMENT_SIGNAL' if all(gates.values()) else 'GENERIC_POINT_REFINER_DOMINATES' if reverse else 'LOCAL_REFINEMENT_ONLY_SIGNAL' if local else 'NO_REFINEMENT_SIGNAL'
    write(B/'VERDICT.json',dict(verdict=verdict,gates=gates,downstream_directions_L_better=direction,downstream_clear_L_harm=harm,
        reverse_safety_pass=bool(reverse),both_local_improve_R0=bool(local),development_only=True,independent_confirmation=False,
        novelty_claim=False,further_line_rescue_authorized=False,statistic_sha256=sha(B/'REAL_PAIRED_STATISTICS.json')))


def distribution(values):
    x=np.asarray(values,float)
    return dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),p90=float(np.quantile(x,.9))) if len(x) else dict(n=0,median=None,mean=None,p90=None)


def mechanism(preds,targets,metadata,stores):
    edges=old('model').SIDE_EDGES;records=[]
    for frame,target in targets.items():
        gt=target.keypoints_xy;valid=target.keypoint_supervision_mask
        base_error=stores['R0'][frame]['errors'];base_mean=float(base_error.mean()) if len(base_error) else None
        difficulty='UNAVAILABLE' if base_mean is None else 'easy' if base_mean<=5 else 'mid' if base_mean<=10 else 'hard'
        item=metadata[frame];payload=read(ROOT/item['gt_v2_path']);annotations=payload['objects'][0]['keypoint_annotations']
        for role,(a,b) in enumerate(edges):
            if not valid[a] or not valid[b]:continue
            vec=gt[b]-gt[a];length=np.linalg.norm(vec)
            if length<2:continue
            tangent=vec/length;normal=np.array([-tangent[1],tangent[0]])
            # Numeric visibility labels remain descriptive. Their physical
            # observability provenance is not validated by this audit.
            vis='ANNOTATION_2' if target.visibility[a]==target.visibility[b]==2 else 'ANNOTATION_1_OR_MIXED'
            values={}
            for name,prediction in preds.items():
                points=prediction[frame]
                if points is None:continue
                err=points[[a,b]]-gt[[a,b]]
                values[name]=dict(normal=float(np.abs(err@normal).mean()),tangent=float(np.abs(err@tangent).mean()))
            records.append(dict(frame=frame,role=str(role),session=item['session_id'],material=item['object_type'],day_night=item['domain'],
                difficulty=difficulty,visibility=vis,values=values,endpoint_indices=[a,b]))
    groups={'overall':{'all':records}}
    for field in ('role','session','material','day_night','difficulty','visibility'):
        groups[field]={v:[r for r in records if r[field]==v] for v in sorted({r[field] for r in records})}
    report={}
    for field,groupset in groups.items():
        report[field]={}
        for group,rows in groupset.items():
            methods={m:{axis:distribution([r['values'][m][axis] for r in rows if m in r['values']]) for axis in ('normal','tangent')} for m in preds}
            delta={axis:{stat:float(np.mean([methods[f'L{s}'][axis][stat]-methods[f'P{s}'][axis][stat] for s in (1,2,3)])) for stat in ('median','mean','p90')} for axis in ('normal','tangent')}
            report[field][group]=dict(edge_frames=len(rows),methods=methods,L_minus_P=delta)
    write(BRAW/'MECHANISM_PER_EDGE.json',dict(records=records,GT_assisted_diagnostic_only=True))
    write(B/'MECHANISM_NORMAL_TANGENT.json',dict(complete=True,groups=report,GT_assisted_diagnostic_only=True,
        endpoints_averaged_within_each_edge=True,shared_endpoint_repeated_across_roles=True,
        visibility_scope='Annotation visibility codes only; physical observed/occluded provenance not independently validated, so no physical occlusion mechanism claim.',
        subgroup_selection=False,primary_gate_not_overridden=True,rows_sha256=sha(BRAW/'MECHANISM_PER_EDGE.json')))


if __name__=='__main__':
    stores,poses,preds,targets,metadata,summaries=load()
    statistics(stores,poses,summaries);mechanism(preds,targets,metadata,stores)
