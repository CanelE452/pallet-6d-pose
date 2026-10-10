"""Saved-row statistics for the frozen easy/medium corrected-source replay.

No detector/head, optimizer, pose solver, ray, renderer or source-test selection.
The existing 10,000 session multiplicity rows are reused, never regenerated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

NAME = 'pallet_kp_corrected_supervision_20261010_v1'
OLD = 'pallet_observation_refiner_20261009_v1'
DIFFICULTY = 'pallet_kp_difficulty_20261010_v1'
HEADS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
NEW_METHODS = HEADS + ('IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE')
SIX = ('NO_MASK_STANDARD','NO_MASK_ROBUST','GEOM_NOSELF_STANDARD','GEOM_NOSELF_ROBUST',
       'ORACLE_NOSELF_ROBUST','ORACLE_VISIBLE_ROBUST')
METRICS = {'translation_cm':('translation_cm',1.,'cm'),
           'rotation_deg':('rotation_deg',1.,'degree'), 'ADDsym_cm':('ADDsym_m',100.,'cm')}
SCOPES = ('common_operational','candidate_new_pose','both_new_pose')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def binding(path, root):
    path = Path(path)
    return dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                sha256=sha(path),bytes=path.stat().st_size)


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if hasattr(value,'tolist'): return clean(value.tolist())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write(path, value):
    assert not path.exists(), 'Preserve completed statistics: ' + str(path)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_rows(path, rows):
    assert not path.exists(), 'Preserve completed rows: ' + str(path)
    with path.open('wb') as handle:
        with gzip.GzipFile(fileobj=handle,mode='wb',filename='',mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(clean(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())


def read_rows(path):
    with gzip.open(path,'rt') as handle:
        return [json.loads(line) for line in handle]


def protected(root, document):
    manifest = json.loads(document.read_text())
    for record in manifest['files']:
        p=root/record['path']
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'], record['path']
    return dict(files=len(manifest['files']),all_unchanged=True,manifest=binding(document,root))


def distribution(values, unit):
    a=np.asarray(values,dtype=np.float64)
    assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        sample_variance=float(a.var(ddof=1)) if len(a)>1 else None,
        sample_std=float(a.std(ddof=1)) if len(a)>1 else None,
        median=float(np.median(a)) if len(a) else None,
        P90=float(np.quantile(a,.9)) if len(a) else None,max=float(a.max()) if len(a) else None,
        unit=unit,ddof=1,quantile_method='numpy linear')


def summary(rows):
    available=[r for r in rows if r['pose']['available']]
    fresh=[r for r in available if r.get('new_pose_estimated')]
    assert all(bool(r.get('no_pose')) != bool(r['pose']['available']) for r in rows)
    return dict(total_frames=len(rows),denominator=len(rows),pose_available=len(available),
        new_pose_estimated=len(fresh),fallback_used=sum(bool(r.get('fallback_used')) for r in rows),
        no_pose=len(rows)-len(available),fixed_control_outputs=sum(r.get('output_status')=='HISTORICAL_FIXED_CONTROL' for r in rows),
        hidden_reprojected=sum(bool(r.get('hidden_reprojected')) for r in rows),
        hidden_set_changed=sum(bool(r.get('hidden_set_changed')) for r in rows),
        available_ids=[r['id'] for r in available],new_pose_ids=[r['id'] for r in fresh],
        fallback_ids=[r['id'] for r in rows if r.get('fallback_used')],
        no_pose_ids=[r['id'] for r in rows if not r['pose']['available']],
        output_status_counts=dict(Counter(r['output_status'] for r in rows)),
        failure_reasons=dict(Counter(r.get('solver',{}).get('reason','historical_fixed') for r in rows if not r.get('new_pose_estimated'))),
        metrics={scope:{m:distribution([r['pose'][field]*factor for r in rr],unit)
                    for m,(field,factor,unit) in METRICS.items()}
                 for scope,rr in (('operational',available),('new_pose',fresh))},
        denominator_note='Every eligible frame retained. Available outputs including fallback enter operational moments; complete failures remain separate, without invented finite error penalties.')


def paired(a,b,ids,inverse,draws,scope):
    indices=[i for i,fid in enumerate(ids) if a[fid]['pose']['available'] and b[fid]['pose']['available']
             and (scope=='common_operational' or a[fid].get('new_pose_estimated'))
             and (scope!='both_new_pose' or b[fid].get('new_pose_estimated'))]
    indices=np.asarray(indices,dtype=int)
    counts=np.bincount(inverse[indices],minlength=draws.shape[1])
    denominator=draws@counts
    keep=denominator>0
    paired_ids=[ids[i] for i in indices]
    record=dict(denominator=len(ids),common_frames=len(indices),common_ids=paired_ids,pair_ids=paired_ids,
        excluded_ids=[fid for fid in ids if fid not in set(paired_ids)],scope=scope,
        bootstrap_nonempty_resamples=int(keep.sum()),metrics={})
    for m,(field,factor,unit) in METRICS.items():
        delta=np.asarray([(a[ids[i]]['pose'][field]-b[ids[i]]['pose'][field])*factor for i in indices])
        totals=np.bincount(inverse[indices],weights=delta,minlength=draws.shape[1])
        samples=(draws@totals)[keep]/denominator[keep]
        record['metrics'][m]={**distribution(delta,unit),
            'mean_delta':float(delta.mean()) if len(delta) else None,
            'CI95':np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
            'improved_frames':int((delta < -1e-9).sum()),'worsened_frames':int((delta > 1e-9).sum()),
            'unchanged_frames':int((np.abs(delta)<=1e-9).sum()),
            'interpretation':'Candidate minus comparator; session bootstrap CI of paired mean, not error dispersion.'}
    record['new_marginals']=summary([a[fid] for fid in paired_ids])['metrics']['operational']
    record['comparator_marginals']=summary([b[fid] for fid in paired_ids])['metrics']['operational']
    return record


def cube(dims):
    a,b,c=np.asarray(dims,float)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def image_shape(q):
    q=np.asarray(q,float).reshape(-1,2)
    s=np.linalg.svd(q-q.mean(0),compute_uv=False) if len(q) else np.array([])
    return dict(singular_values=s,numerical_rank=int(np.sum(s>s[0]*1e-10)) if len(s) and s[0]>0 else 0,
                relative_singular_values=s/s[0] if len(s) and s[0]>0 else s)


def outcome(t,r,new,available):
    if not available:return 'no_pose'
    if not new:return 'fallback'
    if t is not None and r is not None and t < -1e-9 and r < -1e-9:return 'both_improved'
    if t is not None and r is not None and t > 1e-9 and r > 1e-9:return 'both_worsened'
    return 'mixed_or_equal'


def posthoc(rows,controls,targets,old_audit,cohort,input_metadata):
    result=[];labels={f['id']:f['label'] for f in cohort['frames']}
    for alias,row in rows:
        arm='N3_SUBPIX' if alias.startswith('N3_SUBPIX') else 'BASE'
        fid=row['id'];baseline=controls[arm][fid];target=targets[fid]
        inherited=old_audit[(arm+'_NO_MASK_ROBUST',fid)]
        phase=np.asarray(target['permutations'][baseline['corner'].get('branch',0)][:8],int)
        assert phase.tolist()==inherited['permutation_native_to_canonical']
        states=inherited['human_states_native'];reference=np.asarray(target['gt'],float)[phase]
        valid=np.asarray(target['valid'],bool)[phase] & np.isfinite(reference).all(1) & ~(reference==-1).all(1) & bool(target['matched'])
        def errors(points):
            q=np.asarray(points,float)[:8];known=valid & np.isfinite(q).all(1) & ~(q==-1).all(1)
            e=np.linalg.norm(q-reference,axis=1);e[~known]=np.nan
            return e,known
        input_points=row.get('input_points',row['native_points'])
        e,known=errors(input_points);before,bknown=errors(baseline['native_points']);after,aknown=errors(row['native_points'])
        correct=set(np.flatnonzero(known & (e<=8.)).tolist());known_ids=set(np.flatnonzero(known).tolist())
        solver=row.get('solver',{});pool=set(solver.get('used',[]));local=bool(row.get('local_point_line_refinement') or alias.endswith('POINT_LINE'))
        inliers=set(solver.get('final_inliers',solver.get('inliers',[]))) if not local else set()
        hidden=set(row.get('hidden_initial',[]));human_hidden={i for i,s in enumerate(states) if s=='SELF_OCCLUDED'}
        direct={i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'};label_known={i for i,s in enumerate(states) if s!='UNANNOTATED'}
        mask_applied='NO_MASK' not in alias and alias not in ('BASE','N3_SUBPIX')
        wrong_mask=bool((hidden ^ human_hidden) & label_known)
        dims=(row.get('actual_pose') or {}).get('cf_extents') or row.get('xyz') or (baseline.get('actual_pose') or {}).get('cf_extents') or input_metadata[fid]['xyz']
        layout=cube(dims)[sorted(pool & correct)]
        singular=np.linalg.svd(layout-layout.mean(0),compute_uv=False) if len(layout) else np.array([])
        current,prior=row['pose'],baseline['pose']
        t=current['translation_cm']-prior['translation_cm'] if current['available'] and prior['available'] else None
        r=current['rotation_deg']-prior['rotation_deg'] if current['available'] and prior['available'] else None
        record=dict(id=fid,session=row['session'],method=alias,raw_method=row['method'],label=labels[fid],
            reference_kind='GEOMETRIC_PROXY',GT_used_after_geometry_seal=True,frozen_baseline_arm=arm,
            permutation_native_to_canonical=phase.tolist(),human_states_native=states,reference_matched=bool(target['matched']),
            reference_native_points_px=reference,reference_valid_native_ids=np.flatnonzero(valid).tolist(),
            frozen_BASE_native_points=controls['BASE'][fid]['native_points'],frozen_initial_native_points=baseline['native_points'],
            input_native_points=input_points,output_native_points=row['native_points'],raw_hw=row.get('raw_hw',baseline['raw_hw']),
            accuracy_threshold_px=8.,reference_error_input_native_px=e,reference_error_initial_native_px=before,
            reference_error_output_native_px=after,reference_valid_input_ids=sorted(known_ids),
            paired_corner_valid_ids=np.flatnonzero(bknown & aknown).tolist(),correct_input_native_ids=sorted(correct),
            pool_ids=sorted(pool),correct_pool_ids=sorted(correct & pool),correct_pool_count=len(correct & pool),
            wrong_pool_ids=sorted((pool & known_ids)-correct),unknown_pool_ids=sorted(pool-known_ids),
            final_inlier_ids=sorted(inliers),correct_final_inlier_ids=sorted(correct & inliers),correct_final_inlier_count=len(correct & inliers),
            wrong_final_inlier_ids=sorted((inliers & known_ids)-correct),unknown_final_inlier_ids=sorted(inliers-known_ids),
            human_direct_pool_ids=sorted(direct & pool),human_direct_correct_pool_ids=sorted(direct & pool & correct),
            human_direct_correct_final_inlier_ids=sorted(direct & inliers & correct),
            false_excluded_direct_ids=sorted(hidden & direct),false_excluded_accurate_ids=sorted(hidden & correct),
            false_retained_human_self_ids=sorted((pool & human_hidden)-hidden),
            correct_pool_layout_m=layout,correct_pool_shape_singular_values=singular,
            correct_pool_shape_rank=int(np.sum(singular>singular[0]*1e-10)) if len(singular) and singular[0]>0 else 0,
            correct_pool_image_layout=image_shape([input_points[i] for i in sorted(pool & correct)]),
            solver_local_jacobian=solver.get('geometry',{}).get('jacobian'),
            normalized_jacobian_condition=(solver.get('geometry',{}).get('jacobian') or {}).get('condition_number'),
            mask_applied=mask_applied,mask_wrong_on_known=wrong_mask,initial_hidden_ids=sorted(hidden),
            hidden_reprojected_ids=row.get('reprojected_ids',[]),hidden_set_changed=bool(row.get('hidden_set_changed')),
            point_PnP_inliers_applicable=not local,local_point_line_refinement=local,point_line_rank=solver.get('rank') if local else None,
            retained_line_edges=solver.get('line_edges',[]) if local else [],
            new_pose_estimated=row.get('new_pose_estimated',False),fallback_used=row.get('fallback_used',False),
            no_pose=not current['available'],output_status=row['output_status'],translation_cm=current.get('translation_cm'),
            rotation_deg=current.get('rotation_deg'),ADDsym_m=current.get('ADDsym_m'),translation_delta_cm=t,rotation_delta_deg=r,
            paired_pose_outcome_vs_same_coordinate_initial=outcome(t,r,row.get('new_pose_estimated'),current['available']))
        assert set(record['correct_pool_ids'])|set(record['wrong_pool_ids'])|set(record['unknown_pool_ids'])==pool
        result.append(record)
    return clean(result)


def correspondence_summary(records):
    def subset(rr):
        return dict(frames=len(rr),new_pose=sum(r['new_pose_estimated'] for r in rr),fallback=sum(r['fallback_used'] for r in rr),
            no_pose=sum(r['no_pose'] for r in rr),pose_outcomes=dict(Counter(r['paired_pose_outcome_vs_same_coordinate_initial'] for r in rr)),
            correct_pool_count_histogram=dict(Counter(str(r['correct_pool_count']) for r in rr)),
            correct_final_inlier_count_histogram=dict(Counter(str(r['correct_final_inlier_count']) for r in rr)),
            direct_correct_pool_count_histogram=dict(Counter(str(len(r['human_direct_correct_pool_ids'])) for r in rr)),
            translation_cm=distribution([r['translation_cm'] for r in rr if r['translation_cm'] is not None],'cm'),
            rotation_deg=distribution([r['rotation_deg'] for r in rr if r['rotation_deg'] is not None],'degree'))
    ge4=[r for r in records if r['correct_pool_count']>=4]
    return dict(all=subset(records),correct_pool_lt4=subset([r for r in records if r['correct_pool_count']<4]),
        correct_pool_ge4=subset(ge4),correct_pool_ge4_object_rank_counts=dict(Counter(str(r['correct_pool_shape_rank']) for r in ge4)),
        false_excluded_direct_total=sum(len(r['false_excluded_direct_ids']) for r in records),
        false_excluded_accurate_total=sum(len(r['false_excluded_accurate_ids']) for r in records),
        final_reference_inaccurate_inliers_total=sum(len(r['wrong_final_inlier_ids']) for r in records),
        final_unknown_inliers_total=sum(len(r['unknown_final_inlier_ids']) for r in records),
        point_PnP_inliers_applicable=all(r['point_PnP_inliers_applicable'] for r in records),
        mask_known_relation={
            'wrong_mask':subset([r for r in records if r['mask_wrong_on_known']]),
            'matching_known_human_self_states':subset([r for r in records if not r['mask_wrong_on_known']])}
            if records and records[0]['mask_applied'] else {'applicable':False,'reason':'No mask applied; human-label mismatch is not a classifier error.'})


def corner_summary(records,only_new=False):
    pairs=defaultdict(list);frame_means=defaultdict(list)
    for row in records:
        if only_new and not row['new_pose_estimated']:continue
        local=defaultdict(list)
        for i in row['paired_corner_valid_ids']:
            categories=[row['human_states_native'][i]]
            if i in row['hidden_reprojected_ids']:categories.append('ALGORITHM_REPROJECTED_IDS')
            if i in row['false_excluded_direct_ids']:categories.append('DIRECT_VISIBLE_FALSE_EXCLUDED')
            values=(row['reference_error_initial_native_px'][i],row['reference_error_output_native_px'][i])
            for category in categories:pairs[category].append(values);local[category].append(values)
        for category,values in local.items():frame_means[category].append(tuple(np.mean(values,axis=0)))
    result={}
    for category,values in pairs.items():
        a,b=np.asarray(values).T;fa,fb=np.asarray(frame_means[category]).T
        result[category]=dict(corners=len(a),frames=len(fa),before=distribution(a,'px'),after=distribution(b,'px'),
            paired_delta=distribution(b-a,'px'),before_frame_mean=distribution(fa,'px'),after_frame_mean=distribution(fb,'px'),
            improved=int((b<a-1e-9).sum()),worsened=int((b>a+1e-9).sum()),
            good5_to_bad10=int(((a<5)&(b>10)).sum()),bad20_to_good10=int(((a>20)&(b<=10)).sum()))
    return result


def curve_summary(doc):
    logs=read_rows(doc/'TRAIN_LOGS.jsonl.gz');completion=json.loads((doc/'TRAINING_COMPLETION.json').read_text())
    formal=[r for r in logs if r['kind']=='formal'];curves=[r for r in logs if r['kind']=='source_curve']
    assert completion['complete'] and completion['formal_updates']==9000 and completion['throwaway_updates']==0
    assert len(curves)==9 and {(r['arm'],r['step']) for r in curves}=={(a,s) for a in HEADS for s in (1000,2000,3000)}
    return dict(schema='corrected_source_learning_curves_v1',complete=True,
        initial_probes={r['arm']:r['initial_probe'] for r in completion['checkpoints']},
        formal_logged_rows=formal,source_curves=curves,final_source_test={r['arm']:r['source_test'] for r in curves if r['step']==3000},
        formal_updates=9000,formal_exposures=144000,seed=1,batch16=True,
        decoder='unchanged 66-way MAP; integer within-stripe candidate argmax; no match-mass decoder',
        model_selection=False,geometry_only_keeps_geometric_role_channels=True,
        loss='image-average soft-target cross-entropy; no-match single bin and position soft interpolation',
        coordinate_error='Conditional accepted positive candidates only. Adoption and false no-match acceptance reported separately.',
        transfer_caveat='Existing synthetic P0 heldout family scores are not real pose accuracy or independent real truth.')


def freeze_protocol(root):
    doc=root/'_docs/experiments'/NAME;cohort=json.loads((doc/'COHORT.json').read_text())
    p=doc/'STATISTICS_PROTOCOL.json'
    value=dict(schema='corrected_easy_medium_saved_row_statistics_protocol_v1',
        cohort=binding(doc/'COHORT.json',root),primary='CORRECTED_IMAGE_ROLE_minus_N3_SUBPIX',
        strata=dict(combined=245,easy=153,medium=92),all_eligible_outputs_retained=True,
        metrics=METRICS,sample_variance_ddof=1,quantile_method='numpy linear',
        statistics=['mean','sample_variance','sample_std','median','P90','max'],
        marginal_scopes=['operational','new_pose'],paired_scopes=list(SCOPES),
        bootstrap=binding(root/'_docs/experiments'/DIFFICULTY/'BOOTSTRAP_SESSION_DRAWS.json.gz',root),
        bootstrap_reuse='exact existing10000x13 session multiplicities; eligible frame weights recomputed, no new draws',
        methods=['BASE','N3_SUBPIX']+[arm+'_'+s for arm in ('BASE','N3_SUBPIX') for s in SIX]
            +['OLD_'+a for a in HEADS+('IMAGE_ROLE_POINT_LINE',)]+['CORRECTED_'+a for a in NEW_METHODS],
        reference_phase='Same-coordinate frozen initial BASE/N3 branch only, inherited human states; geometric-proxy references',
        mask_error_is_not_frame_failure=True,correct_correspondence_threshold_px=8.,
        direct_visible_damage_and_self_hidden_reprojection_reported_separately=True,
        historical_real_stress='filter old two real stress arms by frozen eligible IDs; newexecution0; absent 2/both real arms remain absent',
        historical_analytic_stress='Old128 analytical scenes separate from real cohort; no new execution or claim of245real coverage',
        selection_uses_new_accuracy=False,new_detector_head_fit_optimizer_ray_render_calls=0)
    assert cohort['count']==245
    if p.exists():assert json.loads(p.read_text())==clean(value)
    else:write(p,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve();doc=root/'_docs/experiments'/NAME
    protocol=freeze_protocol(root)
    if args.freeze_only:print('STATISTICS_PROTOCOL frozen; no corrected scores read');return
    assert args.source_root,'--source-root is needed for post-seal saved reference coordinates'
    started=time.monotonic();old=root/'_docs/experiments'/OLD
    outputs=('METRICS.json','METRICS.csv','POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz','REAL_MASK_STRESS_SUMMARY.json',
             'HISTORICAL_FILTERED_ROWS.jsonl.gz','SOURCE_CURVES.json','STATISTICS_RECEIPT.json')
    assert not any((doc/name).exists() for name in outputs),'Preserve completed statistics'
    before=protected(root,doc/'PRIOR_PUBLICATION_BINDINGS.json')
    cohort=json.loads((doc/'COHORT.json').read_text());ids=cohort['ids'];allowed=set(ids)
    assert len(ids)==len(allowed)==245
    scored_path=doc/'LEARNED_PREDICTIONS.jsonl.gz';sealed_path=doc/'LEARNED_GEOMETRY_SEALED.jsonl.gz'
    execution_path=doc/'LEARNED_POSE_EXECUTION.json';execution=json.loads(execution_path.read_text())
    assert execution['complete'] and sha(sealed_path)==execution['final_sealed']['sha256']
    raw={(r['method'],r['id']):r for r in read_rows(sealed_path)}
    fresh=read_rows(scored_path)
    assert len(raw)==len(fresh)==245*6
    for row in fresh:
        original=raw[(row['method'],row['id'])]
        for key,value in original.items():assert row[key]==value,(row['method'],row['id'],key)
    historical_specs=[(old/'FIXED_CONTROLS.jsonl.gz',{'BASE':'BASE','N3_SUBPIX':'N3_SUBPIX'}),
        (old/'PREDICTIONS.jsonl.gz',{a+'_'+s:a+'_'+s for a in ('BASE','N3_SUBPIX') for s in SIX}),
        (old/'LEARNED_PREDICTIONS.jsonl.gz',{a:'OLD_'+a for a in HEADS+('IMAGE_ROLE_POINT_LINE',)})]
    methods=defaultdict(dict);sources={};historical=[]
    for path,mapping in historical_specs:
        for row in read_rows(path):
            if row['id'] not in allowed or row['method'] not in mapping:continue
            alias=mapping[row['method']];assert row['id'] not in methods[alias]
            methods[alias][row['id']]=row;sources[alias]=dict(path=binding(path,root)['path'],raw_method=row['method'],new_execution=0)
            historical.append(dict(alias=alias,**row))
    for row in fresh:
        alias='CORRECTED_'+row['method'];assert row['id'] not in methods[alias]
        methods[alias][row['id']]=row;sources[alias]=dict(path=binding(scored_path,root)['path'],raw_method=row['method'],new_execution=None)
    assert set(methods)==set(protocol['methods'])
    for alias,data in methods.items():assert set(data)==allowed,alias
    draw_path=root/'_docs/experiments'/DIFFICULTY/'BOOTSTRAP_SESSION_DRAWS.json.gz'
    with gzip.open(draw_path,'rt') as handle:draw_record=json.load(handle)
    draws=np.asarray(draw_record['counts'],dtype=np.dtype('<u2'))
    assert draws.shape==(10000,13) and (draws.sum(1)==13).all()
    assert hashlib.sha256(draws.tobytes(order='C')).hexdigest()==draw_record['serialized_raw_sha256']
    session_index={s:i for i,s in enumerate(draw_record['sessions'])}
    cohort_frames={r['id']:r for r in cohort['frames']}
    strata={'combined':ids,'easy':[fid for fid in ids if cohort_frames[fid]['label']=='clean'],
            'medium':[fid for fid in ids if cohort_frames[fid]['label']=='moderate']}
    contrasts=[('CORRECTED_'+a,b) for a in NEW_METHODS for b in ('BASE','N3_SUBPIX')]
    contrasts += [('CORRECTED_'+a,'OLD_'+a) for a in HEADS+('IMAGE_ROLE_POINT_LINE',)]
    contrasts += [('CORRECTED_'+a,'CORRECTED_IMAGE_ROLE') for a in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE')]
    for a in ('BASE','N3_SUBPIX'):
        contrasts += [(a+'_'+s,a) for s in SIX]
        contrasts += [(a+'_GEOM_NOSELF_'+s,a+'_NO_MASK_'+s) for s in ('STANDARD','ROBUST')]
        contrasts += [(a+'_NO_MASK_ROBUST',a+'_NO_MASK_STANDARD'),(a+'_GEOM_NOSELF_ROBUST',a+'_GEOM_NOSELF_STANDARD')]
    assert len(contrasts)==len(set(contrasts))
    strata_result={}
    for label,cohort_ids in strata.items():
        inverse=np.asarray([session_index[cohort_frames[fid]['session']] for fid in cohort_ids],dtype=int)
        strata_result[label]=dict(count=len(cohort_ids),ids=cohort_ids,
            methods={a:summary([data[fid] for fid in cohort_ids]) for a,data in methods.items()},
            contrasts={a+'_minus_'+b:{scope:paired(methods[a],methods[b],cohort_ids,inverse,draws,scope) for scope in SCOPES} for a,b in contrasts})
    # First private reference access occurs only after complete geometry/scored joins.
    target_path=args.source_root.resolve()/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json'
    targets=json.loads(target_path.read_text())
    old_audit={(r['method'],r['id']):r for r in read_rows(old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz') if r['id'] in allowed}
    input_metadata={f['id']:f for f in json.loads((old/'INPUTS.json').read_text())['frames']}
    annotations=posthoc([(alias,data[fid]) for alias,data in methods.items() for fid in ids],methods,targets,old_audit,cohort,input_metadata)
    ann_groups=defaultdict(list)
    for row in annotations:ann_groups[row['method']].append(row)
    for label,cohort_ids in strata.items():
        scope_set=set(cohort_ids)
        strata_result[label]['posthoc_correspondence']={a:correspondence_summary([r for r in rr if r['id'] in scope_set]) for a,rr in ann_groups.items()}
        strata_result[label]['visibility_damage']={a:{scope:corner_summary([r for r in rr if r['id'] in scope_set],scope=='new_pose')
            for scope in ('operational','new_pose')} for a,rr in ann_groups.items()}
    stress_path=old/'REAL_MASK_STRESS.jsonl.gz'
    stress=[r for r in read_rows(stress_path) if r['id'] in allowed]
    assert len(stress)==490 and len({r['method'] for r in stress})==2
    stress_ann=posthoc([(r['method'],r) for r in stress],methods,targets,old_audit,cohort,input_metadata)
    stress_summary=dict(schema='historical_easy_medium_real_mask_stress_v1',complete=True,new_execution=0,
        source=binding(stress_path,root),rows=490,cohort=binding(doc/'COHORT.json',root),
        missing_real_variants=['drop_two_correct','keep_two_wrong','drop_keep_one_each','drop_keep_two_each'],
        supplied_real_conditions=['DROP_ONE_VISIBLE','KEEP_ONE_HIDDEN'],
        warning='Historical changes are human-visibility errors; retained hidden coordinates are not automatically inaccurate. Reference accuracy and correct-pool geometry are reported separately.',
        strata={label:{m:dict(summary=summary([r for r in stress if r['id'] in set(cids) and r['method']==m]),
            correspondence=correspondence_summary([r for r in stress_ann if r['id'] in set(cids) and r['method']==m]),
            transformations_possible=sum(bool(r['transformation_possible']) for r in stress if r['id'] in set(cids) and r['method']==m))
            for m in sorted({r['method'] for r in stress})} for label,cids in strata.items()},
        separate_analytic_diagnostic=binding(old/'GEOMETRY_STRESS_SUMMARY.json',root),
        analytic_is_not_real_selected_cohort=True)
    curves=curve_summary(doc)
    combined=strata_result['combined'];primary=combined['contrasts']['CORRECTED_IMAGE_ROLE_minus_N3_SUBPIX']['common_operational']
    verdict=dict(method='CORRECTED_IMAGE_ROLE',fixed_comparator='N3_SUBPIX',denominator=245,
        new_pose_estimated=combined['methods']['CORRECTED_IMAGE_ROLE']['new_pose_estimated'],
        fallback_used=combined['methods']['CORRECTED_IMAGE_ROLE']['fallback_used'],
        both_operational_means_lower=all(primary['metrics'][m]['mean_delta'] is not None and primary['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg')),
        both_paired_CI95_strictly_below_zero=all(primary['metrics'][m]['CI95'] is not None and primary['metrics'][m]['CI95'][1]<0 for m in ('translation_cm','rotation_deg')),
        geometric_proxy_reference=True,independent_heldout_confirmation=False)
    save_rows(doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz',annotations)
    save_rows(doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz',historical+[dict(alias=r['method'],historical_stress=True,**r) for r in stress])
    write(doc/'REAL_MASK_STRESS_SUMMARY.json',stress_summary);write(doc/'SOURCE_CURVES.json',curves)
    metrics=dict(schema='corrected_easy_medium_metrics_v1',complete=True,population=dict(frames=245,easy=153,medium=92,severe_excluded=74,sessions=13,new_rows=len(fresh)),
        reference='Unchanged geometric-proxy references; no independent physical pose truth or new annotations.',
        methods=combined['methods'],contrasts=combined['contrasts'],posthoc_correspondence=combined['posthoc_correspondence'],
        visibility_damage=combined['visibility_damage'],strata=strata_result,method_sources=sources,
        verdicts=dict(primary=verdict),cohort=binding(doc/'COHORT.json',root),statistics_protocol=binding(doc/'STATISTICS_PROTOCOL.json',root),
        bootstrap=dict(source=binding(draw_path,root),resamples=10000,sessions=draw_record['sessions'],
            serialized_raw_sha256=draw_record['serialized_raw_sha256'],new_draws_generated=0,
            session_frame_counts=cohort['session_counts'],CI_quantiles=[.025,.975],multiplicity_correction=False),
        paired_scope_contract=dict(common_operational='Both output poses available including fallback.',
            candidate_new_pose='Candidate new pose; comparator available. Fixed controls remain fixed, not new.',
            both_new_pose='Both newly estimated poses; empty for fixed BASE/N3 by definition.'),
        uncertainty='Sample SD describes error dispersion; the shared session bootstrap CI describes paired mean uncertainty. Neither is seed variability or independent physical truth.',
        historical_methods_recomputed_from_selected_original_rows=True,
        reference_annotations=binding(doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz',root),
        visibility_contract='Frozen same-coordinate initial corner phase. DIRECT and SELF errors separate; unobserved inputs remain unknown. Point+line has no thresholded point consensus.',
        zero_new_pose_in_statistics=True)
    write(doc/'METRICS.json',metrics)
    with (doc/'METRICS.csv').open('w',newline='') as handle:
        columns=['stratum','method','scope','metric','n','mean','sample_variance','sample_std','median','P90','max','unit']
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for stratum,result in strata_result.items():
            for method,data in result['methods'].items():
                for scope,rr in data['metrics'].items():
                    for metric,values in rr.items():writer.writerow(dict(stratum=stratum,method=method,scope=scope,metric=metric,**{k:values[k] for k in columns[4:]}))
    after=protected(root,doc/'PRIOR_PUBLICATION_BINDINGS.json')
    receipt=dict(schema='corrected_easy_medium_statistics_receipt_v1',complete=True,script=binding(Path(__file__).resolve(),root),
        inputs=[binding(p,root) for p in [doc/'COHORT.json',doc/'STATISTICS_PROTOCOL.json',scored_path,sealed_path,execution_path,draw_path,
            old/'FIXED_CONTROLS.jsonl.gz',old/'PREDICTIONS.jsonl.gz',old/'LEARNED_PREDICTIONS.jsonl.gz',old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz',stress_path,
            doc/'TRAIN_LOGS.jsonl.gz',doc/'TRAINING_COMPLETION.json']],
        target_reference=binding(target_path,args.source_root.resolve()),GT_inputs_opened_after_seal=True,
        eligible_ids_identical_all24methods=True,geometry_raw_fields_unchanged_in_scored=True,
        rows=dict(new_scored=len(fresh),posthoc=len(annotations),historical_filtered=len(historical),historical_stress=len(stress)),
        original331protected_before=before,original331protected_after=after,
        new_detector_head_calls=0,new_PnP_optimizer_calls=0,new_training_updates=0,new_RGB_or_ray_renders=0,new_bootstrap_draws=0,
        outputs=[binding(doc/name,root) for name in outputs if name!='STATISTICS_RECEIPT.json'],wall_seconds=time.monotonic()-started)
    write(doc/'STATISTICS_RECEIPT.json',receipt)
    print(json.dumps(dict(complete=True,methods=len(methods),cohort=245,posthoc=len(annotations),verdict=verdict,wall_seconds=receipt['wall_seconds'])))


if __name__=='__main__':main()
