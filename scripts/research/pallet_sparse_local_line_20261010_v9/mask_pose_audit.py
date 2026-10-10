"""Supplement the frozen V9 mask-label diagnosis without changing any score.

V8.annotate determines mask_applied from its own method registry. V9's new
LOCAL method is absent from that registry, although the geometry packet's
initial_H_applied is true and H is excluded from its fit. Preserve that saved
false flag and independently group existing human native states and saved
N3-phase pose errors using the actual V9 packet policy. No production module,
private truth, image, model, scorer, PnP or local optimizer is opened here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
METHOD = 'ROLE_BOUNDARY_LOCAL_POINT_LINE'
COMPARATOR = 'ROLE_BOUNDARY_H_ROBUST'
FILES = ('MASK_POSE_AUDIT_PROTOCOL.json','MASK_POSE_AUDIT_STARTED.json',
    'MASK_POSE_AUDIT_CHECKS.json','MASK_POSE_AUDIT_ROWS.jsonl.gz')
METRICS = ('translation_cm','rotation_deg','ADDsym_cm')
OUTCOMES = ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')
MASKS = ('WRONG_ON_KNOWN','MATCHES_ON_KNOWN','NO_KNOWN_VISIBILITY','MASK_NOT_APPLIED')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def no_symlink(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink: '+str(path))


def read(path):
    with Path(path).open('r',encoding='utf-8') as stream:
        return json.load(stream)


def binding(path):
    path = Path(path)
    no_symlink(path)
    require(path.is_file(),'missing audit input: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    absolute = path.absolute()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
        bytes=path.stat().st_size,sha256=h.hexdigest())


def same(left,right):
    return all(left[key] == right[key] for key in ('bytes','sha256'))


def bound(path,expected,label):
    require(same(binding(path),expected),'byte binding differs: '+label)


def public_path(item):
    p = Path(item['path'])
    require(item.get('origin') == 'public_repository' and not p.is_absolute() and '..' not in p.parts,
        'public input required')
    path = REPO/p
    no_symlink(path)
    return path


def write_new(path,value):
    no_symlink(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def guard(args,stage):
    folder,output = args.input.absolute(),args.output.absolute()
    no_symlink(folder);no_symlink(output)
    require(folder.is_dir(),'evidence input directory missing')
    output.mkdir(parents=True,exist_ok=True)
    for name in FILES if stage == 'freeze' else FILES[1:]:
        require(not (output/name).exists(),'preserve existing '+name)
    require(not (output/'PENDING_MASK_POSE_AUDIT_ROWS.jsonl.gz').exists(),'preserve pending audit prefix')
    if stage == 'run':
        require((output/FILES[0]).is_file(),'own freeze required')
    return folder,output


def paths(folder):
    names = ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json',
        'VALIDATION_CHECKS.json','SCORING_RECEIPT.json','SCORING_PARITY.json','VERIFICATION.json',
        'METRICS.json','PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz')
    result = {n:folder/n for n in names}
    protocol = read(result['PROTOCOL.json'])
    for name in ('cohort','evaluator.py','reused_v8:evaluator.py'):
        result[name] = public_path(protocol['inputs'][name])
    result['audit_code'] = Path(__file__).resolve()
    return result


def metadata(files):
    protocol,seal,control,validation,score,parity,verified,metrics = (read(files[name]) for name in
        ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json',
         'SCORING_RECEIPT.json','SCORING_PARITY.json','VERIFICATION.json','METRICS.json'))
    require(protocol['schema'] == 'fixed_same_observation_local_C2_protocol_v9' and
        protocol['methods'] == [METHOD] and protocol['primary'] == METHOD and protocol['frames'] == protocol['rows'] == 245,
        'frozen one-arm245 protocol required')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True and seal['frames'] == seal['rows'] == seal['ledger_rows'] == 245,
        'GT-free complete geometry and cleanup required')
    require(control['complete'] is True and control['error'] is None and control['cleanup_error'] is None,
        'local receipt incomplete')
    bound(files['PROTOCOL.json'],seal['protocol'],'seal protocol')
    bound(files['CONTROL_RECEIPT.json'],seal['control_receipt'],'seal control')
    require(validation['complete'] is True and validation['passed'] is True,'independent geometry PASS required')
    bound(files['VALIDATION_PROTOCOL.json'],validation['own_protocol'],'independent geometry protocol')
    require(score['complete'] is True and score['frames'] == 245 and score['methods'] == [METHOD] and
        score['scored_method_rows'] == score['comparator_rows'] == 245 and score['fixed_control_rows'] == 490 and
        score['actual_total_scored_rows'] == 980 and score['GT_access_only_after_complete_seal_cleanup_and_validation'] is True and
        score['new_pose_fits'] == score['new_local_optimizers'] == score['new_head_forwards'] == 0 and
        score['forbidden_entries_attempted'] == 0,'completed fit-free980 scoring required')
    for key,name in (('protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),
        ('control_receipt','CONTROL_RECEIPT.json'),('predictions','PREDICTIONS.jsonl.gz'),
        ('comparator_predictions','COMPARATOR_PREDICTIONS.jsonl.gz'),('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz'),
        ('parity','SCORING_PARITY.json')):
        bound(files[name],score[key],'scoring '+key)
    require(parity['passed'] is True and parity['total_rows'] == 735 and
        parity['saved_scores_only_read_after_complete_seal_cleanup_validation'] is True and
        parity['old_scores_used_for_geometry'] is False,'saved parity required')
    require(verified['schema'] == 'independent_sparse_local_point_line_statistics_checks_v9' and
        verified['complete'] is True and verified['passed'] is True and verified['standard_library_only'] is True and
        verified['production_modules_imported'] is False and verified['actual_counts']['moment_scalars_checked'] == 648 and
        verified['actual_counts']['metric_CI_slots_checked'] == 81 and verified['actual_counts']['local_diagnostic_groups_checked'] == 33,
        'completed independent fixed statistics verification required')
    for name in ('PROTOCOL.json','SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz',
        'FIXED_PREDICTIONS.jsonl.gz','METRICS.json'):
        bound(files[name],verified['inputs'][name],'verified '+name)
    require(metrics['complete'] is True and metrics['primary_method'] == METHOD,'saved main statistics required')
    for name in ('cohort','evaluator.py','reused_v8:evaluator.py'):
        bound(files[name],protocol['inputs'][name],'frozen public '+name)
    return protocol


def freeze(args):
    folder,output = guard(args,'freeze')
    files = paths(folder);metadata(files)
    write_new(output/FILES[0],dict(schema='supplemental_actual_H_mask_pose_audit_protocol_v9',
        inputs={k:binding(p) for k,p in files.items()},folder=str(folder),frames=245,scored_rows=980,
        method=METHOD,comparison='LOCAL minus fixed N3_SUBPIX saved pose metrics',
        actual_policy_field='initial_H_applied',historical_wrong_flag_preserved='mask_audit.mask_applied',
        cause='V8.annotate method-registry membership does not include V9 LOCAL label',
        grouping_fields=['initial_H_applied','hidden_initial','human_states_native','semantic_known_ids','eligible'],
        semantic_known_rule="human states excluding both UNANNOTATED and UNKNOWN",
        legacy_known_rule="saved raw known_ids excludes only UNANNOTATED; preserved for parity, not semantic truth",
        unknown_states_do_not_certify_self_occlusion_mismatch=True,
        classification_difference_is_not_pose_failure=True,original_core_or_rows_modified=False,
        new_GT_image_model_scoring_PnP_local_optimizer_training_RGB_draw_seed_calls=0,
        conditional_moments_produced_from_existing_scores=True,
        conditional_moments_not_cross_checked_by_a_second_implementation=True,
        frozen_before_own_arithmetic=True,no_automatic_retry=True))
    print('V9_MASK_AUDIT_FROZEN',binding(output/FILES[0])['sha256'],flush=True)


def stream_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            require(line.strip(),'blank saved row')
            yield json.loads(line)


def scalar_pose(row):
    pose = row['pose']
    require(type(pose['available']) is bool and pose['available'] == row['pose_available'], 'saved pose status')
    result = {m:float(pose['ADDsym_m'])*100 if m == 'ADDsym_cm' else float(pose[m])
        for m in METRICS} if pose['available'] else {}
    require(all(math.isfinite(x) for x in result.values()),'nonfinite available saved score')
    return result


def distribution(values):
    n=len(values);mean=math.fsum(values)/n if n else None
    var=math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None
    ordered=sorted(values)
    def q(p):
        if not n:return None
        pos=(n-1)*p;lo,hi=math.floor(pos),math.ceil(pos)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
    return dict(n=n,mean=mean,sample_variance=var,sample_std=math.sqrt(var) if var is not None else None,
        median=q(.5),P90=q(.9),max=ordered[-1] if n else None)


def summary(selected):
    return dict(frames=len(selected),ids=[r['id'] for r in selected],
        statuses=dict(Counter(r['status'] for r in selected)),NEW=sum(r['new'] for r in selected),
        fallback=sum(r['fallback'] for r in selected),no_pose=sum(not r['available'] for r in selected),
        actual_saved_pose={m:distribution([r['values'][m] for r in selected if r['available']]) for m in METRICS},
        paired_delta_vs_N3={m:distribution([r['delta_vs_N3'][m] for r in selected if r['delta_vs_N3']]) for m in METRICS},
        point_count_histogram=dict(Counter(r['actual_point_count'] for r in selected)),
        unused_line_count_histogram=dict(Counter(r['actual_unused_line_count'] for r in selected)),
        false_excluded_visible_count_histogram=dict(Counter(len(r['false_excluded_visible']) for r in selected)),
        false_retained_self_count_histogram=dict(Counter(len(r['false_retained_self']) for r in selected)))


def run(args):
    folder,output=guard(args,'run');files=paths(folder)
    own=read(output/FILES[0]);current={k:binding(p) for k,p in files.items()}
    require(own['schema']=='supplemental_actual_H_mask_pose_audit_protocol_v9' and
        own['folder']==str(folder) and own['inputs']==current,'own frozen evidence/helper differs')
    metadata(files)
    write_new(output/FILES[1],dict(protocol=binding(output/FILES[0]),inputs=current,actual_saved_arithmetic_runs=1,
        original_core_or_rows_modified=False,new_GT_image_model_scoring_PnP_local_optimizer_calls=0,no_automatic_retry=True))
    began=time.perf_counter()
    result=dict(schema='supplemental_actual_H_mask_pose_audit_checks_v9',complete=False,passed=False,
        protocol=binding(output/FILES[0]),inputs=current,standard_library_only=True,production_modules_imported=False,
        original_core_or_rows_modified=False,original_wrong_mask_applied_flag_preserved=True,
        main_pose_scores_648moments_81CIs_or_33local_groups_changed=False,
        conditional_moments_not_cross_checked_by_a_second_implementation=True,
        new_GT_image_model_scoring_PnP_local_optimizer_training_RGB_draw_seed_calls=0,
        no_automatic_retry=True,failure_count=0,failures=[])
    counts=Counter();pending=output/'PENDING_MASK_POSE_AUDIT_ROWS.jsonl.gz'
    try:
        cohort=read(files['cohort']);ids=cohort['ids'];frames={r['id']:r for r in cohort['frames']}
        require(len(ids)==len(set(ids))==len(frames)==245 and [r['id'] for r in cohort['frames']]==ids,
            'complete ordered245 cohort')
        require(Counter(r['label'] for r in frames.values())=={'clean':153,'moderate':92},'difficulty scope')
        fixed={}
        for row in stream_rows(files['FIXED_PREDICTIONS.jsonl.gz']):
            key=(row['method'],row['id'])
            require(row['method'] in ('BASE','N3_SUBPIX') and row['id'] in frames and key not in fixed,
                'fixed identity duplicate/unknown')
            require(row['session']==frames[row['id']]['session'],'fixed session')
            fixed[key]=scalar_pose(row);counts['fixed_saved_rows']+=1
        require(counts['fixed_saved_rows']==490,'complete fixed490')
        point_ids=[]
        for row in stream_rows(files['COMPARATOR_PREDICTIONS.jsonl.gz']):
            require(row['method']==COMPARATOR and row['id'] in frames and
                row['session']==frames[row['id']]['session'],'point identity/session')
            scalar_pose(row);point_ids.append(row['id'])
        require(point_ids==ids,'ordered point245')
        counts['point_saved_rows']=len(point_ids)
        records=[]
        for row in stream_rows(files['PREDICTIONS.jsonl.gz']):
            fid=row['id'];require(row['method']==METHOD and fid in frames and row['session']==frames[fid]['session'],
                'local identity/session')
            audit=row['mask_audit'];states=audit['human_states_native'];H=set(row['hidden_initial'])
            require(len(states)==8 and all(v in ('DIRECT_VISIBLE','SELF_OCCLUDED','EXTERNAL_OCCLUDED',
                'OUT_OF_FRAME','UNKNOWN','UNANNOTATED') for v in states),
                'existing human native visibility states')
            require(type(row['initial_H_applied']) is bool and row['initial_H_applied'] is True,
                'actual LOCAL packet H policy differs')
            require(audit['mask_applied'] is False and audit['mask_wrong_on_known'] is None,
                'preserve identified frozen inherited mask-label bug')
            legacy_known={k for k,v in enumerate(states) if v!='UNANNOTATED'}
            known={k for k,v in enumerate(states) if v not in ('UNANNOTATED','UNKNOWN')}
            humanH={k for k,v in enumerate(states) if v=='SELF_OCCLUDED'}
            require(sorted(legacy_known)==audit['known_ids'] and H<=set(range(8)),'native H/legacy-known identities')
            legacy_wrong=bool((H^humanH)&legacy_known)
            wrong=bool((H^humanH)&known)
            require(audit['excluded_set_differs_from_human_self_on_known']==legacy_wrong,'saved raw legacy mask difference witness')
            mask='NO_KNOWN_VISIBILITY' if not known else 'WRONG_ON_KNOWN' if wrong else 'MATCHES_ON_KNOWN'
            solver=row['solver'];eligible=set(solver.get('eligible',range(8)));used=list(solver.get('used',[]))
            require(H.isdisjoint(solver.get('fit_input_ids',[])),'hidden initial2D entered fit')
            require(row['reprojections_reused_as_observations'] is False and solver['global_uniqueness_proven'] is False,
                'H was refit or global uniqueness claimed')
            new,fallback=bool(row['new_pose_estimated']),bool(row['fallback_used'])
            require(not(new and fallback),'NEW/fallback overlap')
            if new:
                require(solver['available'] is True and solver['local_refinement_estimated'] is True and
                    solver['initial_pose_residual_prior'] is False and solver['inlier_geometry_is_acceptance_gate'] is False,
                    'accepted LOCAL policy')
            values=scalar_pose(row);n3=fixed[('N3_SUBPIX',fid)]
            delta={m:values[m]-n3[m] for m in METRICS} if values and n3 else {}
            category='POSE_UNAVAILABLE' if not delta else 'BOTH_BETTER' if delta['translation_cm']<0 and delta['rotation_deg']<0 else (
                'BOTH_WORSE' if delta['translation_cm']>0 and delta['rotation_deg']>0 else 'MIXED_OR_UNCHANGED')
            false_excluded=sorted(H&{k for k,v in enumerate(states) if v=='DIRECT_VISIBLE'})
            false_retained=sorted((humanH-H)&eligible)
            require(false_excluded==audit['false_excluded_visible'] and false_retained==audit['false_retained_self'],
                'original raw correspondence error IDs differ')
            records.append(dict(id=fid,session=row['session'],difficulty=frames[fid]['label'],method=METHOD,
                status=row['output_status'],new=new,fallback=fallback,available=bool(values),values=values,
                fixed_N3_values=n3,delta_vs_N3=delta,actual_H_applied=True,
                original_saved_mask_applied=False,original_saved_mask_wrong_on_known=None,
                corrected_mask_wrong_on_known=wrong if known else None,mask_category=mask,pose_category_vs_N3=category,
                H=sorted(H),human_self_hidden_ids=sorted(humanH),semantic_known_ids=sorted(known),
                legacy_known_ids=sorted(legacy_known),legacy_raw_mask_difference=legacy_wrong,
                unknown_ids=[k for k,v in enumerate(states) if v=='UNKNOWN'],human_states_native=states,
                unknown_states_do_not_certify_self_occlusion_mismatch=True,
                false_excluded_visible=false_excluded,false_retained_self=false_retained,
                actual_point_count=len(used),actual_unused_line_count=len(solver['line_edges']),
                actual_point_ids=used,fit_input_ids=solver.get('fit_input_ids',[]) if new else [],
                fit_line_edges=solver.get('fit_line_edges',[]) if new else [],
                diagnostic_inlier_point_ids=solver['final_inliers'],diagnostic_inlier_line_edges=solver['final_line_inliers'],
                diagnostic_inlier_support_rank6=solver.get('diagnostic_inlier_support_rank6',False),
                hidden_reprojected=row['hidden_reprojected'],reprojected_ids=row['reprojected_ids'],
                classification_difference_is_not_pose_failure=True,numeric_NEW_is_not_accuracy_success=True))
            counts['local_saved_rows']+=1;counts['original_false_mask_flag']+=1
        require([r['id'] for r in records]==ids and counts['local_saved_rows']==245,'ordered local245')
        strata={}
        for name,labels in (('combined',{'clean','moderate'}),('easy',{'clean'}),('medium',{'moderate'})):
            selected=[r for r in records if r['difficulty'] in labels]
            groups={mask+'__'+outcome:summary([r for r in selected if r['mask_category']==mask and
                r['pose_category_vs_N3']==outcome]) for mask in MASKS for outcome in OUTCOMES}
            require(sum(g['frames'] for g in groups.values())==len(selected),'mask/outcome partitions all frames')
            strata[name]=dict(frames=len(selected),ids=[r['id'] for r in selected],groups=groups,
                mask_categories=dict(Counter(r['mask_category'] for r in selected)),
                actual_H_applied_frames=len(selected),original_false_flag_frames=len(selected))
            counts['mask_pose_groups_produced']+=len(groups)
        with gzip.open(pending,'xt',encoding='utf-8') as stream:
            for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        with pending.open('rb') as stream:os.fsync(stream.fileno())
        require({k:binding(p) for k,p in files.items()}==current,'frozen inputs changed during read-only audit')
        os.link(pending,output/FILES[3]);pending.unlink()
        counts['rows_written']=len(records)
        result.update(complete=True,passed=True,frames=245,scored_rows_streamed=980,strata=strata,
            corrected_posthoc_rows=binding(output/FILES[3]),actual_counts=dict(counts),
            original_frozen_method_label_bug_not_pose_fit_bug=True,classification_difference_is_not_pose_failure=True)
    except Exception as error:
        result.update(failure_count=1,failures=[dict(type=type(error).__name__,message=str(error))],actual_counts=dict(counts),
            pending_prefix_preserved=pending.exists())
    result['elapsed_seconds']=time.perf_counter()-began
    write_new(output/FILES[2],result)
    print('V9_MASK_POSE_AUDIT','PASS' if result['passed'] else 'FAIL',dict(counts),flush=True)
    if not result['passed']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--input',type=Path,default=DOC)
    parser.add_argument('--output',type=Path,default=DOC,help='new supplemental own receipt/rows directory')
    args=parser.parse_args();(freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':
    main()
