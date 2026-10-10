"""Post-seal numeric audit for the single frozen original-ROLE line ablation.

Run with python -m scripts.research.pallet_kp_supervision_gate_20261010_v1.line_statistics.
Imports no model/renderer/pose solver; the optional private reference JSON is
opened only after the complete geometry and score receipts have been verified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from ..pallet_kp_difficulty_20261010_v1.statistics import (
    bind, clean, delta_outcome, distribution, paired, save_rows, summary, write)

NAME = 'pallet_kp_supervision_gate_20261010_v1'
OLD = 'pallet_observation_refiner_20261009_v1'
KP = 'pallet_kp_difficulty_20261010_v1'
METHOD = 'IMAGE_ROLE_ORIGINAL_ALL_LINES'
COMPARATORS = ('BASE', 'N3_SUBPIX', 'IMAGE_ROLE_POINT_LINE')
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def protected(root, snapshot):
    failures = []
    for entry in snapshot['protected_files']:
        path = root/entry['path']
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise AssertionError('Original published files changed: '+repr(failures))
    return dict(files=len(snapshot['protected_files']),all_SHA256_and_bytes_match=True,
                protected_commit=snapshot['commit'])


def annotate(methods, ids, controls, targets, human):
    records = []
    for method, by_id in methods.items():
        for fid in ids:
            row,base = by_id[fid],controls[fid]
            target = targets[fid]
            inherited = human[fid]
            phase = target['permutations'][base['corner'].get('branch',0)][:8]
            assert phase==inherited['permutation_native_to_canonical']
            states = inherited['human_states_native']
            ref = np.asarray(target['gt'],float)[phase]
            valid = (np.asarray(target['valid'],bool)[phase]&np.isfinite(ref).all(1)
                     &~np.all(ref==-1,axis=1)&bool(target['matched']))
            before = np.asarray(base['native_points'],float)[:8]
            after = np.asarray(row['native_points'],float)[:8]
            def errors(points):
                known = valid&np.isfinite(points).all(1)&~np.all(points==-1,axis=1)
                value = np.linalg.norm(points-ref,axis=1)
                value[~known] = np.nan
                return value,known
            eb,kb = errors(before);ea,ka = errors(after)
            paired_known = kb&ka
            H = set(row.get('hidden_initial',[]))
            new = bool(row.get('new_pose_estimated'))
            direct = {i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}
            human_hidden = {i for i,s in enumerate(states) if s=='SELF_OCCLUDED'}
            known_states = {i for i,s in enumerate(states) if s!='UNANNOTATED'}
            wrong = bool((H^human_hidden)&known_states)
            reproj = set(row.get('reprojected_ids',[]))
            selected = set(row.get('selected_corner_ids',[])) if method==METHOD else set()
            native_updates = {i for i in range(8) if np.isfinite(before[i]).all() and np.isfinite(after[i]).all() and np.max(np.abs(before[i]-after[i]))>1e-9}
            outcomes = {}
            for comparator in COMPARATORS:
                p,b = row['pose'],methods[comparator][fid]['pose']
                td = p['translation_cm']-b['translation_cm'] if p['available'] and b['available'] else None
                rd = p['rotation_deg']-b['rotation_deg'] if p['available'] and b['available'] else None
                outcomes[comparator] = dict(translation_delta_cm=td,rotation_delta_deg=rd,
                    classification=delta_outcome(td,rd,new,p['available']))
            records.append(dict(id=fid,session=row['session'],method=method,
                reference_kind='GEOMETRIC_PROXY',physical_GT_independently_validated=False,
                reference_read_after_complete_geometry_and_score_seal=True,
                frozen_baseline_phase_arm='BASE',permutation_native_to_canonical=phase,
                human_states_native=states,reference_matched=bool(target['matched']),
                reference_native_points_px=ref,reference_valid_native_ids=np.flatnonzero(valid).tolist(),
                frozen_BASE_native_points=before,output_native_points=after,
                reference_error_initial_native_px=eb,reference_error_output_native_px=ea,
                paired_corner_valid_ids=np.flatnonzero(paired_known).tolist(),
                mask_applied_to_output=method in (METHOD,'IMAGE_ROLE_POINT_LINE'),
                mask_used_for='Hidden output replacement only; line factors are not excluded by this corner mask.' if method==METHOD else 'Historical comparator policy.',
                mask_wrong_on_known=wrong if method in (METHOD,'IMAGE_ROLE_POINT_LINE') else None,
                initial_hidden_ids=sorted(H),hidden_reprojected_ids=sorted(reproj),
                false_excluded_direct_ids=sorted(H&direct),
                human_self_not_marked_hidden_ids=sorted(human_hidden-H),
                selected_nonhidden_output_corner_ids=sorted(selected-H) if new else [],
                native_changed_corner_ids=sorted(native_updates),
                direct_visible_changed_ids=sorted(native_updates&direct),
                direct_visible_reprojected_ids=sorted(reproj&direct),
                direct_visible_observed_corner_updated_ids=sorted((selected-H)&direct) if new else [],
                new_pose_estimated=new,fallback_used=bool(row.get('fallback_used')),no_pose=bool(row.get('no_pose')),
                output_status=row['output_status'],pose=row['pose'],paired_pose_outcomes=outcomes,
                point_PnP_inliers_applicable=False if method==METHOD else None,
                line_factor_edges=row.get('selected_line_edges',[]),
                actual_local_model_geometry_rank=row.get('solver',{}).get('model_geometry_observability',{}).get('rank'),
                actual_local_model_geometry_condition=row.get('solver',{}).get('model_geometry_observability',{}).get('condition_number'),
                local_rank_is_not_global_unique_pose=True))
    return clean(records)


def visibility(records, fresh_only=False):
    corner_pairs = defaultdict(list);frame_pairs = defaultdict(list)
    for row in records:
        if fresh_only and not row['new_pose_estimated']:
            continue
        local = defaultdict(list)
        for i in row['paired_corner_valid_ids']:
            groups = [row['human_states_native'][i]]
            if i in row['hidden_reprojected_ids']:groups.append('ALGORITHM_REPROJECTED_IDS')
            if i in row['false_excluded_direct_ids']:groups.append('DIRECT_VISIBLE_FALSE_EXCLUDED')
            if i in row['direct_visible_observed_corner_updated_ids']:groups.append('DIRECT_VISIBLE_OBSERVED_CORNER_UPDATED')
            a,b = row['reference_error_initial_native_px'][i],row['reference_error_output_native_px'][i]
            for group in groups:
                corner_pairs[group].append((a,b));local[group].append((a,b))
        for group,values in local.items():
            frame_pairs[group].append(tuple(np.mean(values,axis=0)))
    result = {}
    for group,values in corner_pairs.items():
        a,b=np.asarray(values).T;fa,fb=np.asarray(frame_pairs[group]).T
        result[group]=dict(corners=len(a),frames=len(fa),before=distribution(a,'px'),after=distribution(b,'px'),
            paired_delta=distribution(b-a,'px'),before_frame_mean=distribution(fa,'px'),after_frame_mean=distribution(fb,'px'),
            improved=int((b<a-1e-9).sum()),worsened=int((b>a+1e-9).sum()),unchanged=int((np.abs(b-a)<=1e-9).sum()),
            good5_to_bad10=int(((a<5)&(b>10)).sum()),bad20_to_good10=int(((a>20)&(b<=10)).sum()))
    return result


def mask_relation(records):
    result={}
    for relation,rr in [('wrong_on_known',[r for r in records if r['mask_wrong_on_known']]),
                        ('matches_known_self_states',[r for r in records if not r['mask_wrong_on_known']])]:
        result[relation]=dict(frames=len(rr),new_pose=sum(r['new_pose_estimated'] for r in rr),
            fallback=sum(r['fallback_used'] for r in rr),no_pose=sum(r['no_pose'] for r in rr),
            outcomes={c:dict(Counter(r['paired_pose_outcomes'][c]['classification'] for r in rr)) for c in COMPARATORS})
    result['interpretation']='Mask mismatch concerns hidden coordinate output replacement, not rejection of line factors. Mask classification is not a pose success gate.'
    return result


def scientific_figure(output, metrics, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['BASE','N3_SUBPIX','IMAGE_ROLE_POINT_LINE',METHOD]
    labels=['Base','N3→SubPix','Original point+line','Same ROLE: all lines']
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for axis,metric,title in [(axes[0,0],'translation_cm','Whole operational population: position [cm]'),
                              (axes[0,1],'rotation_deg','Whole operational population: rotation [degree]')]:
        vals=[metrics['methods'][m]['metrics']['operational'][metric]['mean'] for m in names]
        bars=axis.bar(range(4),vals,color=['#64748b','#2563eb','#a78bfa','#ef4444'])
        axis.set_xticks(range(4),labels,rotation=12);axis.set_ylim(0,max(vals)*1.2);axis.set_title(title)
        for bar,value in zip(bars,vals):axis.text(bar.get_x()+bar.get_width()/2,value+.4,f'{value:.2f}',ha='center')
        axis.grid(axis='y',alpha=.2)
    axis=axes[1,0];fresh=np.array([metrics['methods'][m]['new_pose_estimated'] for m in names]);fallback=np.array([metrics['methods'][m]['fallback_used'] for m in names]);historical=np.array([319-metrics['methods'][m]['new_pose_estimated']-metrics['methods'][m]['fallback_used']-metrics['methods'][m]['no_pose'] for m in names]);fail=np.array([metrics['methods'][m]['no_pose'] for m in names])
    bottom=np.zeros(4)
    for values,label,color in [(fresh,'New pose','#16a34a'),(fallback,'Fallback','#f59e0b'),(historical,'Historical fixed output','#64748b'),(fail,'No output','#dc2626')]:
        axis.bar(range(4),values,bottom=bottom,label=label,color=color);bottom+=values
    axis.set_xticks(range(4),labels,rotation=12);axis.set_ylim(0,360);axis.set_title('Status counts: all 319 IDs retained');axis.legend(fontsize=8)
    for i,n in enumerate(fresh):axis.text(i,325,f'new={n}',ha='center',fontsize=9)
    axis=axes[1,1];c=source['counts'];exact=[49,90];observed=[50,91]
    assert exact==[source['factor_rank_histograms']['current_factor_topology_with_exact_source_lines'].get('6',0),c['all_exact_source_lines_finite_difference_rank6']]
    x=np.arange(2);axis.bar(x-.18,exact,.35,label='Exact source line geometry',color='#2563eb');axis.bar(x+.18,observed,.35,label='Quantized target TLS lines',color='#94a3b8');axis.set_xticks(x,['Corner + unused line','All lines, once/edge']);axis.set_ylim(0,140);axis.set_title('Source-only local rank 6: denominator 128');axis.legend(fontsize=8)
    for xx,values in [(x-.18,exact),(x+.18,observed)]:
        for a,b in zip(xx,values):axis.text(a,b+2,str(b),ha='center')
    axis.text(.02,.02,'Rank 6 does not prove unique or correct pose.\nOne parallel-line case gains false rank from quantization.',transform=axis.transAxes,fontsize=9)
    visible=metrics['visibility_damage'][METHOD]['operational'];cats=['DIRECT_VISIBLE','SELF_OCCLUDED','ALGORITHM_REPROJECTED_IDS'];catlabels=['Direct visible','Human self occluded','Algorithm reprojected H']
    axis=axes[2,0];x=np.arange(3)
    before=[visible.get(k,{}).get('before',{}).get('mean',0) for k in cats];after=[visible.get(k,{}).get('after',{}).get('mean',0) for k in cats]
    axis.bar(x-.18,before,.35,label='Frozen Base input',color='#94a3b8');axis.bar(x+.18,after,.35,label='All-lines operational output',color='#ef4444');axis.set_xticks(x,catlabels,rotation=10);axis.set_ylabel('Mean per-corner proxy error [raw px]');axis.set_title('Corner error: paired known corners; overlapping categories');axis.legend(fontsize=8)
    for i,k in enumerate(cats):axis.text(i,max(before+after)*1.02,f"n={visible.get(k,{}).get('corners',0)}",ha='center',fontsize=8)
    axis=axes[2,1];damage=[visible.get(k,{}).get('good5_to_bad10',0) for k in cats];axis.bar(range(3),damage,color='#ef4444');axis.set_xticks(range(3),catlabels,rotation=10);axis.set_ylabel('Corner count');axis.set_title('Initially <5 px → output >10 px: damage')
    for i,n in enumerate(damage):axis.text(i,n+.2,str(n),ha='center')
    fig.suptitle('Fixed original IMAGE_ROLE observations: one line-only ablation, no retraining',fontsize=15)
    fig.text(.02,.018,'Position/rotation use existing GEOMETRIC_PROXY DEV319; source rank is a separate local derivative diagnostic.\nFallback is counted separately. No new real RGB, model forward, GT selection, or deployment timing measurement.',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.965]);(output/'figures').mkdir(exist_ok=True)
    path=output/'figures/01_line_ablation.png';assert not path.exists();fig.savefig(path,dpi=150);plt.close(fig)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reference-targets',type=Path,required=True)
    args=parser.parse_args();began=time.monotonic();root=args.root.resolve()
    old=root/'_docs/experiments'/OLD;kp=root/'_docs/experiments'/KP;doc=root/'_docs/experiments'/NAME
    output=(args.output_dir or doc).resolve();output.mkdir(parents=True,exist_ok=True)
    outputs=['METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz','STATISTICS_RECEIPT.json','FIGURE_BINDING.json']
    assert not any((output/n).exists() for n in outputs),'Preserve completed statistics'
    protocol=json.loads((doc/'PROTOCOL.json').read_text());execution=json.loads((doc/'LINE_EXECUTION.json').read_text())
    snapshot=json.loads((doc/'PRIOR_PUBLICATION_BINDINGS.json').read_text());before=protected(root,snapshot)
    assert protocol['new_methods']==[METHOD] and protocol['primary_comparator']=='N3_SUBPIX'
    assert execution['complete'] and execution['geometry_sealed_before_GT'] and execution['score_after_seal']
    for key in ('geometry','scored'):
        assert sha(root/execution[key]['path'])==execution[key]['sha256']
    new=read_rows(doc/'PREDICTIONS.jsonl.gz');sealed=read_rows(doc/'GEOMETRY_SEALED.jsonl.gz')
    assert len(new)==len(sealed)==319
    assert all({k:v for k,v in a.items() if k not in ('pose','corner')}==b for a,b in zip(new,sealed))
    frames=json.loads((old/'INPUTS.json').read_text())['frames'];ids=[r['id'] for r in frames]
    assert len(ids)==len(set(ids))==319
    controls=read_rows(old/'FIXED_CONTROLS.jsonl.gz');learned=read_rows(old/'LEARNED_PREDICTIONS.jsonl.gz')
    methods=defaultdict(dict)
    for r in controls+learned+new:
        if r['method'] not in (METHOD,)+COMPARATORS:continue
        assert r['id'] not in methods[r['method']];methods[r['method']][r['id']]=r
    assert set(methods)==set((METHOD,)+COMPARATORS)
    assert all(set(rr)==set(ids) for rr in methods.values())
    sessions,inverse=np.unique([r['session'] for r in frames],return_inverse=True);assert len(sessions)==13
    with gzip.open(kp/'BOOTSTRAP_SESSION_DRAWS.json.gz','rt') as stream:draw_data=json.load(stream)
    draws=np.asarray(draw_data['counts'],dtype='<u2')
    assert draws.shape==(10000,13) and draw_data['sessions']==sessions.tolist() and np.all(draws.sum(1)==13)
    draw_sha=hashlib.sha256(draws.tobytes(order='C')).hexdigest();assert draw_sha==draw_data['serialized_raw_sha256']
    original_metrics=json.loads((old/'METRICS.json').read_text());assert draw_sha==original_metrics['bootstrap']['draw_sha256']
    summaries={m:summary([data[fid] for fid in ids]) for m,data in methods.items()}
    assert all(summaries[m]['metrics']==original_metrics['methods'][m]['metrics'] for m in COMPARATORS)
    contrasts={METHOD+'_minus_'+c:{scope:paired(methods[METHOD],methods[c],ids,inverse,draws,scope) for scope in SCOPES} for c in COMPARATORS}
    # Only now read existing private proxy reference coordinates: no inference
    # imports occur, and no pose optimization function exists in this module.
    targets=json.loads(args.reference_targets.read_text())
    old_human={r['id']:r for r in read_rows(old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz') if r['method']=='IMAGE_ROLE'}
    records=annotate(methods,ids,methods['BASE'],targets,old_human)
    assert len(records)==1276
    grouped={m:[r for r in records if r['method']==m] for m in methods}
    visible={m:{'operational':visibility(rr),'new_pose':visibility(rr,True)} for m,rr in grouped.items()}
    source=json.loads((doc/'SOURCE_OBSERVATION_GATE.json').read_text())
    candidate=summaries[METHOD]['metrics']['operational'];baseline=summaries['N3_SUBPIX']['metrics']['operational']
    ci=contrasts[METHOD+'_minus_N3_SUBPIX']['common_operational']['metrics']
    metrics=dict(schema='original_ROLE_all_lines_metrics_v1',complete=True,new_method=METHOD,fixed_comparators=list(COMPARATORS),
        population=dict(frames=319,sessions=13,scored_rows=319,summary_arms=4,posthoc_rows=1276),
        primary_comparator='N3_SUBPIX',methods=summaries,contrasts=contrasts,
        verdict=dict(both_operational_means_lower=all(candidate[m]['mean']<baseline[m]['mean'] for m in ('translation_cm','rotation_deg')),
            both_paired_CI95_strictly_below_zero=all(ci[m]['CI95'][1]<0 for m in ('translation_cm','rotation_deg')),
            not_independent_heldout_confirmation=True),
        bootstrap=dict(resamples=10000,sessions=sessions.tolist(),session_frame_counts=dict(Counter(r['session'] for r in frames)),
            draw_sha256=draw_sha,serialized_dtype='uint16_little_endian',new_draws_generated=0,
            source=bind(kp/'BOOTSTRAP_SESSION_DRAWS.json.gz',root),seed_of_existing_matrix=20260917,
            algorithm='Existing frozen session multiplicities; retained-frame-weighted paired means; empty conditional draws omitted',
            quantiles=[.025,.975],multiple_comparison_correction=False),
        paired_scope_contract=dict(common_operational='Both outputs available, including fallback.',
            candidate_new_pose='Candidate has a new pose; comparator output available.',
            both_new_pose='Both methods have new poses; empty vs historical Base/N3.',
            denominator='All319 retained in population; each conditional/common set explicitly lists included and excluded IDs.'),
        visibility_damage=visible,mask_known_relation=mask_relation(grouped[METHOD]),
        reference='Existing GEOMETRIC_PROXY DEV319; no independent physical ground truth.',
        corner_contract='All error annotations use frozen BASE phase and initial coordinates; known corner pairs only; mean corner and mean frame separate; categories overlap. Hidden references may themselves be geometrically derived.',
        source_local_sensitivity=dict(families=128,exact_source_current_rank6=49,exact_source_all_lines_rank6=90,
            observed_TLS_current_rank6=50,observed_TLS_all_lines_rank6=91,
            geometry_diagnostic_only=True,global_unique_pose_proven=False,source_pose_not_used_in_real_fit=True,
            gate=bind(doc/'SOURCE_OBSERVATION_GATE.json',root)),
        uncertainty='Sample variance/SD are error dispersion (ddof1), not mean CI. Paired13-session CI uses existing fixed draws, not new seeds or independent heldout validation.',
        actual_new_calls=dict(PnP=0,optimizers=0,detectors=0,heads=0,training_updates=0,rays=0,RGB_reads=0),
        posthoc=bind(output/'POSTHOC_ROWS.jsonl.gz',root) if (output/'POSTHOC_ROWS.jsonl.gz').exists() else None,
        runtime='No new pipeline benchmark: cached accuracy solve seconds must not be described as deployment latency.')
    save_rows(output/'POSTHOC_ROWS.jsonl.gz',records);metrics['posthoc']=bind(output/'POSTHOC_ROWS.jsonl.gz',root)
    write(output/'METRICS.json',metrics)
    with (output/'METRICS.csv').open('x',newline='') as stream:
        fields=['method','scope','metric','unit','n','mean','sample_variance','sample_std','median','P90','max','total_frames','new_pose_estimated','fallback_used','no_pose']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for m,data in summaries.items():
            for scope,by_metric in data['metrics'].items():
                for metric,stats in by_metric.items():
                    writer.writerow(dict(method=m,scope=scope,metric=metric,**{k:stats[k] for k in ['unit','n','mean','sample_variance','sample_std','median','P90','max']},**{k:data[k] for k in ['total_frames','new_pose_estimated','fallback_used','no_pose']}))
    figure=scientific_figure(output,metrics,source)
    write(output/'FIGURE_BINDING.json',dict(schema='scientific_numeric_line_ablation_figure_v1',file=bind(figure,root),
        sources=[bind(output/'METRICS.json',root),bind(doc/'SOURCE_OBSERVATION_GATE.json',root)],
        presentation_only=True,pose_outputs_unchanged=True,new_RGB_reads=0,
        panels=['all319 meanT cm','all319 meanR degree','all319 statuses','source128 exact versus quantized localrank','known-corner before/after error','direct/self/H good5-to-bad10 damage'],
        source_rank_is_not_pose_success=True,reference_kind='GEOMETRIC_PROXY'))
    after=protected(root,snapshot)
    inputs=[doc/'PROTOCOL.json',doc/'LINE_EXECUTION.json',doc/'GEOMETRY_SEALED.jsonl.gz',doc/'PREDICTIONS.jsonl.gz',
        old/'INPUTS.json',old/'FIXED_CONTROLS.jsonl.gz',old/'LEARNED_PREDICTIONS.jsonl.gz',old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz',
        old/'METRICS.json',kp/'BOOTSTRAP_SESSION_DRAWS.json.gz',doc/'SOURCE_OBSERVATION_GATE.json',doc/'PRIOR_PUBLICATION_BINDINGS.json']
    receipt=dict(schema='original_ROLE_all_lines_statistics_receipt_v1',complete=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        inputs=[bind(p,root) for p in inputs],sealed_geometry_hash_verified_before_private_reference_read=True,
        score_and_geometry_records_exact_except_score_fields=True,source_private_reference=dict(path=args.reference_targets.name,origin='external_readonly_dependency',sha256=sha(args.reference_targets),bytes=args.reference_targets.stat().st_size),
        raw_counts=dict(new_scored=319,posthoc=1276,summary_arms=4,paired_contrasts=3,scopes_per_contrast=3),
        original_comparator_metric_moments_exact=True,existing_draw_matrix_exact=True,new_bootstrap_draws=0,
        protected_before=before,protected_after=after,
        execution='NumPy arithmetic and Matplotlib scientific plotting only; direct post-seal JSON reference read, no model/pose/renderer imports.',
        actual_calls=metrics['actual_new_calls'],outputs=[bind(output/p,root) for p in outputs if p!='STATISTICS_RECEIPT.json']+[bind(figure,root)],
        code=bind(Path(__file__).resolve(),root),pure_math_dependency=bind(Path(__file__).resolve().parents[1]/KP/'statistics.py',root),wall_seconds=time.monotonic()-began)
    write(output/'STATISTICS_RECEIPT.json',receipt)
    print(json.dumps(dict(complete=True,method=METHOD,new_pose=summaries[METHOD]['new_pose_estimated'],fallback=summaries[METHOD]['fallback_used'],
        primary_operational=candidate,primary_delta=ci,verdict=metrics['verdict'],wall_seconds=receipt['wall_seconds']),ensure_ascii=False))


if __name__=='__main__':
    main()
