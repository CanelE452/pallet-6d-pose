"""Independent public V8 identity, receipt, byte and moment review.

Freeze this supplement before its own arithmetic. Full original gzip hashes
are checked before streaming; only small scalar/identity fields are retained.
No experiment module, private weights, source GT, head, PnP or fresh scoring
is imported or executed. Fixed bootstrap CIs belong to verify.py and are not
recalculated here; its separately completed receipt can be byte-bound.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_same_observation_controls_20261010_v8'
FILES = ('PUBLIC_REVIEW_PROTOCOL.json','PUBLIC_REVIEW_STARTED.json','PUBLIC_REVIEW_CHECKS.json')
METHODS = ('ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_H_STANDARD','ROLE_BOUNDARY_NO_MASK_ROBUST')
ALL_METHODS = ('BASE','N3_SUBPIX',*METHODS)
METRICS = ('translation_cm','rotation_deg','ADDsym_cm')
ABS_TOL, REL_TOL = 1e-9, 1e-12


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    with Path(path).open('r',encoding='utf-8') as stream:
        return json.load(stream)


def reject_symlink(path):
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink: ' + str(path))


def binding(path):
    path = Path(path)
    reject_symlink(path)
    require(path.is_file(),'missing public input ' + str(path))
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
    require(same(binding(path),expected),'byte binding differs: ' + label)


def public_path(item):
    relative = Path(item['path'])
    require(item.get('origin') == 'public_repository' and not relative.is_absolute() and
        '..' not in relative.parts,'nonpublic dependency')
    result = REPO/relative
    reject_symlink(result)
    return result


def write_new(path,value):
    reject_symlink(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def guard(args,stage):
    folder,output = args.input.absolute(),args.output.absolute()
    reject_symlink(folder)
    reject_symlink(output)
    require(folder.is_dir(),'input evidence directory missing')
    output.mkdir(parents=True,exist_ok=True)
    for name in FILES if stage == 'freeze' else FILES[1:]:
        require(not (output/name).exists(),'preserve existing ' + name)
    if args.export_csv:
        reject_symlink(args.export_csv)
        require(not args.export_csv.exists(),'preserve per-frame CSV')
    return folder,output


def inputs(args,folder):
    names = ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json',
        'GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','VALIDATION_PROTOCOL.json',
        'VALIDATION_CHECKS.json','SCORING_RECEIPT.json','SCORING_PARITY.json',
        'PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz','METRICS.json','POSTHOC_ROWS.jsonl.gz')
    files = {name:folder/name for name in names}
    files.update({'PROTOCOL.json':args.protocol,'checker_code':Path(__file__).resolve(),
        'archive_code':Path(__file__).resolve().with_name('archive_evidence.py'),
        'restore_code':Path(__file__).resolve().with_name('restore_archives.py')})
    protocol = read(args.protocol)
    external = []
    for name,item in protocol['inputs'].items():
        if item.get('origin') == 'public_repository':
            files['public_input:' + name] = public_path(item)
        else:
            external.append(name)
    files['cohort'] = public_path(protocol['inputs']['cohort'])
    if args.statistics_verification:
        files['statistics_verification'] = args.statistics_verification
    return files,sorted(external)


def metadata(files):
    protocol,seal,control,score,parity,metrics,validation_protocol,validation = (
        read(files[name]) for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json',
        'SCORING_RECEIPT.json','SCORING_PARITY.json','METRICS.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json'))
    require(protocol['schema'] == 'fixed_same_observation_C3_protocol_v8' and
        protocol['frames'] == 245 and protocol['rows'] == 735 and protocol['methods'] == list(METHODS) and
        protocol['primary'] == METHODS[0] and len(protocol['contrasts']) == 8,'fixed V8 scope differs')
    expected_contrasts = ([[m,b] for m in METHODS for b in ('BASE','N3_SUBPIX')] +
        [[METHODS[1],METHODS[0]],[METHODS[2],METHODS[0]]])
    require(protocol['contrasts'] == expected_contrasts,'eight fixed contrast definitions differ')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True and seal['no_scored_or_human_inputs'] is True and
        seal['frames'] == 245 and seal['rows'] == 735 and seal['ledger_rows'] == 245 and
        seal['methods'] == list(METHODS) and seal['primary'] == METHODS[0],'complete GT-free seal required')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),
        ('ledgers','CONTROL_LEDGERS.jsonl.gz'),('control_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        bound(files[name],seal[key],'seal:' + key)
    require(control['complete'] is True and control['error'] is None and control['cleanup_error'] is None and
        control['actual_complete_frames'] == 245 and control['GT_access_during_controls'] is False and
        control['environment_and_monkeypatch_cleanup_completed'] is True and
        not control['source_asset_or_GT_attempted_reads'],'control completion/cleanup contract')
    require(control['method_rows'] == {m:245 for m in METHODS} and
        control['actual_counts']['geometry_rows'] == 735 and control['actual_counts']['ledger_rows'] == 245,
        'complete control row counters')
    bound(files['PROTOCOL.json'],control['protocol'],'control protocol')
    for name,count in (('geometry',735),('ledgers',245)):
        stream = control['serialization']['streams'][name]
        require(stream['rows'] == count and stream['published'] is True and not stream['close_errors'] and
            not stream['interruption_preservation_errors'],'control stream completion: ' + name)
    require(validation_protocol['schema'] == 'supplemental_same_sparse_ROLE_scalar_protocol_v8' and
        validation['schema'] == 'supplemental_same_sparse_ROLE_scalar_checks_v8' and
        validation['complete'] is True and validation['passed'] is True and
        validation['inputs'] == validation_protocol['inputs'],'independent geometry validation PASS required')
    bound(files['VALIDATION_PROTOCOL.json'],validation['protocol'],'independent validation protocol')
    for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','GEOMETRY_SEALED.jsonl.gz',
        'CONTROL_LEDGERS.jsonl.gz','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json'):
        bound(files[name],validation_protocol['inputs'][name],'validated:' + name)
    bound(files['public_input:validation_checks.py'],validation_protocol['inputs']['checker_code'],'validated checker')
    bound(files['public_input:validation_checks.py'],validation['checker'],'successful checker')
    require(score['complete'] is True and score['frames'] == 245 and score['methods'] == list(METHODS) and
        score['scored_method_rows'] == 735 and score['fixed_control_rows'] == 490 and
        score['GT_access_only_after_complete_seal_cleanup_and_validation'] is True and
        score['scoring_fitting_and_model_entries_forbidden'] is True and score['forbidden_entries_attempted'] == 0,
        'complete unchanged postseal scoring required')
    require(all(score[key] == 0 for key in ('new_detector_forwards','new_N3_forwards',
        'new_head_forwards','new_pose_fits','new_rays')),'scoring unexpectedly executed inference')
    for key,name in (('protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),
        ('control_receipt','CONTROL_RECEIPT.json'),('predictions','PREDICTIONS.jsonl.gz'),
        ('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz'),('parity','SCORING_PARITY.json')):
        bound(files[name],score[key],'scoring:' + key)
    for key,name in (('protocol','VALIDATION_PROTOCOL.json'),('receipt','VALIDATION_CHECKS.json')):
        bound(files[name],score['independent_geometry_validation'][key],'scoring validation:' + key)
    require(parity['passed'] is True and parity['fixed_rows'] == 490 and parity['parent_primary_rows'] == 245 and
        parity['old_scores_only_used_after_complete_seal_and_validation'] is True and
        parity['old_scores_used_for_geometry'] is False,'complete parent score parity required')
    require(metrics['complete'] is True and metrics['primary_method'] == METHODS[0] and
        metrics['new_accuracy_primary'] is False and metrics['population']['frames'] == 245 and
        metrics['new_detector_N3_head_PnP_optimizer_ray_training_RGB_scoring_calls'] == 0,'complete saved-row statistics')
    for key,name in (('new_scored_rows','PREDICTIONS.jsonl.gz'),('fixed_scored_rows','FIXED_PREDICTIONS.jsonl.gz'),
        ('scoring_receipt','SCORING_RECEIPT.json'),('geometry_seal','GEOMETRY_SEAL.json'),
        ('control_receipt','CONTROL_RECEIPT.json'),('control_ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('scoring_parity','SCORING_PARITY.json'),('validation_protocol','VALIDATION_PROTOCOL.json'),
        ('validation_receipt','VALIDATION_CHECKS.json'),('protocol','PROTOCOL.json'),('cohort','cohort')):
        bound(files[name],metrics['bindings'][key],'statistics:' + key)
    bound(files['POSTHOC_ROWS.jsonl.gz'],metrics['posthoc_rows'],'posthoc original')
    for key,name in (('observations','parent:OBSERVATIONS.jsonl.gz'),('bootstrap','bootstrap_draws'),
        ('borrowed_statistics_code','v7_statistics'),('statistics_code','statistics.py')):
        bound(files['public_input:' + name],metrics['bindings'][key],'statistics public:' + key)
    for name,item in protocol['inputs'].items():
        if item.get('origin') == 'public_repository':
            bound(files['public_input:' + name],item,'public core:' + name)
    if 'statistics_verification' in files:
        verifier = read(files['statistics_verification'])
        require(verifier['schema'] == 'independent_same_observation_statistics_checks_v8' and
            verifier['complete'] is True and verifier['passed'] is True and
            verifier['standard_library_only'] is True and verifier['production_modules_imported'] is False and
            verifier['actual_counts']['metric_CI_slots_checked'] == 216 and
            verifier['actual_counts']['moment_scalars_checked'] == 810,'completed separate statistics verifier')
        for name in ('PROTOCOL.json','SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz','METRICS.json'):
            bound(files[name],verifier['inputs'][name],'separate statistics verifier:' + name)
    return protocol,metrics


def freeze(args):
    folder,output = guard(args,'freeze')
    files,external = inputs(args,folder)
    protocol,_ = metadata(files)
    write_new(output/FILES[0],dict(schema='supplemental_same_observation_public_review_protocol_v8',
        inputs={name:binding(path) for name,path in files.items()},input_folder=str(folder),
        frames=245,scored_rows=1225,geometry_rows=735,ledger_rows=245,posthoc_rows=735,
        methods=list(ALL_METHODS),strata={'combined':245,'easy':153,'medium':92},moment_scalars=810,
        contrast_definitions=protocol['contrasts'],public_core_input_count=sum(n.startswith('public_input:') for n in files),
        external_frozen_dependencies_not_opened=external,frozen_before_own_arithmetic=True,
        scalar_absolute_tolerance=ABS_TOL,scalar_relative_tolerance=REL_TOL,
        CI_arithmetic_repeated_here=False,separate_CI_receipt_bound='statistics_verification' in files,
        no_automatic_retry=True,new_model_PnP_training_RGB_GT_scoring_draw_seed_calls=0))
    print(json.dumps(dict(frozen=True,protocol=binding(output/FILES[0]))),flush=True)


def stream_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            require(line.strip(),'blank line in complete raw stream')
            yield json.loads(line)


def compact_scored(row):
    pose = row['pose']
    require(all(type(x) is bool for x in (pose['available'],row['pose_available'],
        row['new_pose_estimated'],row['fallback_used'])),'saved status booleans required')
    require(row['pose_available'] == pose['available'] and
        not (row['new_pose_estimated'] and row['fallback_used']),'saved operational status contract')
    values = {key:(float(pose['ADDsym_m'])*100 if key == 'ADDsym_cm' else float(pose[key]))
        for key in METRICS} if pose['available'] else {}
    require(all(math.isfinite(x) for x in values.values()),'finite scored metrics required')
    solver = row.get('solver') or {}
    return dict(id=row['id'],session=row['session'],method=row['method'],status=row['output_status'],
        available=pose['available'],new=row['new_pose_estimated'],fallback=row['fallback_used'],values=values,
        H=row['hidden_initial'],fit_ids=solver.get('fit_input_ids',[]),inliers=solver.get('final_inliers',[]),
        selected_ids=row.get('selected_corner_ids',[]),mask_applied=row['mask_audit']['mask_applied'],
        mask_wrong_on_known=row['mask_audit']['mask_wrong_on_known'],hidden_reprojected=row['hidden_reprojected'],
        reprojected_ids=row['reprojected_ids'])


def quantile(values,p):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values)-1)*p
    lo,hi = math.floor(offset),math.ceil(offset)
    return values[lo]+(values[hi]-values[lo])*(offset-lo)


def distribution(values):
    n = len(values)
    mean = math.fsum(values)/n if n else None
    variance = math.fsum((v-mean)**2 for v in values)/(n-1) if n > 1 else None
    return dict(n=n,mean=mean,sample_variance=variance,
        sample_std=math.sqrt(variance) if variance is not None else None,
        median=quantile(values,.5),P90=quantile(values,.9),max=max(values) if values else None)


def run(args):
    folder,output = guard(args,'run')
    own = read(output/FILES[0])
    require(own['schema'] == 'supplemental_same_observation_public_review_protocol_v8' and
        own['input_folder'] == str(folder),'own public review root/schema differs')
    files,external = inputs(args,folder)
    current = {name:binding(path) for name,path in files.items()}
    require(current == own['inputs'],'frozen public evidence/helper bytes changed')
    write_new(output/FILES[1],dict(protocol=binding(output/FILES[0]),inputs=current,
        actual_public_saved_arithmetic_runs=1,new_model_PnP_training_RGB_GT_scoring_draw_seed_calls=0))
    start = time.perf_counter()
    counts = Counter()
    max_difference = 0.0
    result = dict(schema='supplemental_same_observation_public_review_checks_v8',complete=False,passed=False,
        protocol=binding(output/FILES[0]),inputs=current,standard_library_only=True,
        production_modules_imported=False,external_frozen_dependencies_not_opened=external,
        full_original_raw_SHA_checked_before_streaming=True,original_serialized_rows_modified=False,
        retained_fields='identity, status, saved pose metrics, H/fit/inlier/display IDs',
        CI_arithmetic_repeated_here=False,separate_CI_receipt_bound='statistics_verification' in files,
        source_calibration_or_physical_truth_recomputed=False,independent_GPU_or_physical_accuracy_rerun=False,
        new_model_PnP_training_RGB_GT_scoring_draw_seed_calls=0,failure_count=0,failures=[],no_automatic_retry=True)
    try:
        protocol,metrics = metadata(files)
        cohort = read(files['cohort'])
        ids = cohort['ids']
        frames = {row['id']:row for row in cohort['frames']}
        require(len(ids) == len(set(ids)) == len(frames) == len(cohort['frames']) == 245 and
            [row['id'] for row in cohort['frames']] == ids,'unique ordered cohort')
        require(Counter(row['label'] for row in frames.values()) == {'clean':153,'moderate':92},'difficulty cohort')
        groups = {m:{} for m in ALL_METHODS}
        scored = []
        for name,expected in (('PREDICTIONS.jsonl.gz',735),('FIXED_PREDICTIONS.jsonl.gz',490)):
            population = Counter()
            for original in stream_rows(files[name]):
                row = compact_scored(original)
                require(row['method'] in groups and row['id'] in frames and
                    row['id'] not in groups[row['method']],'unknown/duplicate scored identity')
                require(row['session'] == frames[row['id']]['session'],'scored session join')
                require((row['method'] in METHODS) == (name == 'PREDICTIONS.jsonl.gz'),'scored stream method contract')
                groups[row['method']][row['id']] = row
                scored.append(row)
                population[row['method']] += 1
            require(sum(population.values()) == expected,'complete scored stream count')
            counts[name] = expected
        require(all(set(group) == set(ids) for group in groups.values()),'all five methods retain245')
        geometry_keys = set()
        for row in stream_rows(files['GEOMETRY_SEALED.jsonl.gz']):
            key = (row['method'],row['id'])
            require(row['method'] in METHODS and row['id'] in frames and key not in geometry_keys and
                row['session'] == frames[row['id']]['session'],'unique geometry identity/session')
            require('pose' not in row and 'evaluation_reference' not in row and 'mask_audit' not in row,
                'sealed geometry already scored')
            saved = groups[row['method']][row['id']]
            require(row['new_pose_estimated'] == saved['new'] and row['fallback_used'] == saved['fallback'] and
                row['output_status'] == saved['status'],'scoring changed geometry status')
            require(row['reprojections_reused_as_observations'] is False and
                row['missing_sparse_filled_for_numeric_fit'] is False,'geometry observation contract')
            if row['new_pose_estimated'] and row['method'] != METHODS[2]:
                require(set(row['hidden_initial']).isdisjoint(row['solver']['fit_input_ids']),
                    'masked initial H entered accepted fit')
            geometry_keys.add(key)
        require(len(geometry_keys) == 735,'complete geometry stream')
        counts['geometry_rows'] = len(geometry_keys)
        ledger_ids = []
        same_U_ids = []
        removed_H = Counter()
        for row in stream_rows(files['CONTROL_LEDGERS.jsonl.gz']):
            require(row['id'] in frames and row['session'] == frames[row['id']]['session'] and
                row['parent_replay_parity']['passed'] is True and row['parent_unchanged'] is True,
                'control ledger identity/parent parity')
            ledger_ids.append(row['id'])
            if row['H_and_no_H_used_U_equal']:
                same_U_ids.append(row['id'])
            removed_H.update(row['actual_H_removed_sparse_ids'])
        require(ledger_ids == ids,'complete ordered245 control ledgers')
        require(metrics['same_observation_diagnostics']['same_U_ids'] == same_U_ids and
            metrics['same_observation_diagnostics']['H_and_no_H_same_used_U_frames'] == len(same_U_ids) and
            metrics['same_observation_diagnostics']['actual_H_removed_sparse_ID_counts'] == dict(removed_H),
            'ledger-derived same-U bookkeeping')
        counts['ledger_rows'] = len(ledger_ids)
        posthoc_keys = set()
        for row in stream_rows(files['POSTHOC_ROWS.jsonl.gz']):
            key = (row['method'],row['id'])
            require(key in geometry_keys and key not in posthoc_keys and
                row['posthoc_kernel_label_adapter_only'] is True and
                row['standard_diagnostic_inlier_count_is_not_a_NEW_acceptance_gate'] is True,
                'complete unchanged-kernel posthoc identity/label contract')
            posthoc_keys.add(key)
        require(posthoc_keys == geometry_keys,'all735 posthoc identities')
        counts['posthoc_rows'] = len(posthoc_keys)
        for stratum,labels in (('combined',{'clean','moderate'}),('easy',{'clean'}),('medium',{'moderate'})):
            selected_ids = [fid for fid in ids if frames[fid]['label'] in labels]
            require(metrics['strata'][stratum]['ids'] == selected_ids,'stratum ID order')
            for method in ALL_METHODS:
                rows = [groups[method][fid] for fid in selected_ids]
                block = metrics['strata'][stratum]['methods'][method]
                scopes = {'operational':[r for r in rows if r['available']],
                    'new_pose':[r for r in rows if r['available'] and r['new']],
                    'fallback':[r for r in rows if r['available'] and r['fallback']]}
                expected = dict(denominator=len(rows),operational=len(scopes['operational']),
                    new_pose=len(scopes['new_pose']),fallback=len(scopes['fallback']),
                    no_pose=sum(not r['available'] for r in rows),
                    fixed_control=sum(r['status'] == 'FRESH_FIXED_CONTROL' for r in rows),
                    statuses=dict(Counter(r['status'] for r in rows)))
                require(all(block[key] == value for key,value in expected.items()),'operational status summary')
                counts['method_stratum_status_groups'] += 1
                for scope,selected in scopes.items():
                    for key in METRICS:
                        actual = block['metrics'][scope][key]
                        expected = distribution([r['values'][key] for r in selected])
                        require(actual['n'] == expected['n'],'moment denominator')
                        counts['moment_denominators_checked'] += 1
                        for field in ('mean','sample_variance','sample_std','median','P90','max'):
                            a,b = actual[field],expected[field]
                            if b is None:
                                require(a is None,'empty moment remains None')
                            else:
                                require(type(a) in (float,int) and math.isfinite(a),'finite moment scalar')
                                max_difference = max(max_difference,abs(a-b))
                                require(math.isclose(a,b,rel_tol=REL_TOL,abs_tol=ABS_TOL),'moment scalar differs')
                            counts['moment_scalars_checked'] += 1
        require(counts['moment_scalars_checked'] == 810 and counts['moment_denominators_checked'] == 135 and
            counts['method_stratum_status_groups'] == 15,'complete public moment/status review')
        require({name:binding(path) for name,path in files.items()} == current,'public inputs changed during review')
        if args.export_csv:
            args.export_csv.parent.mkdir(parents=True,exist_ok=True)
            with args.export_csv.open('x',newline='',encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(('id','session','difficulty','method','status','new_pose','fallback','available',
                    'T_cm','R_deg','ADDsym_cm','H','accepted_fit_ids','diagnostic_inlier_ids','selected_boundary_ids',
                    'mask_applied','mask_wrong_on_known','hidden_reprojected','reprojected_ids'))
                for row in scored:
                    writer.writerow([row['id'],row['session'],frames[row['id']]['label'],row['method'],row['status'],
                        row['new'],row['fallback'],row['available'],*[row['values'].get(k) for k in METRICS],
                        json.dumps(row['H']),json.dumps(row['fit_ids'] if row['new'] else []),json.dumps(row['inliers']),
                        json.dumps(row['selected_ids']),row['mask_applied'],row['mask_wrong_on_known'],
                        row['hidden_reprojected'],json.dumps(row['reprojected_ids'])])
                stream.flush()
                os.fsync(stream.fileno())
            result['export_csv'] = binding(args.export_csv)
        result.update(complete=True,passed=True,frames=245,scored_rows=1225,methods=list(ALL_METHODS),
            public_core_input_bindings_checked=sum(n.startswith('public_input:') for n in files),
            same_U_frames=len(same_U_ids),actual_H_removed_sparse_ID_counts=dict(removed_H))
    except Exception as error:
        result.update(failure_count=1,failures=[dict(type=type(error).__name__,message=str(error))])
    result.update(actual_counts=dict(counts),max_numeric_difference=max_difference,
        absolute_tolerance=ABS_TOL,relative_tolerance=REL_TOL,elapsed_seconds=time.perf_counter()-start)
    write_new(output/FILES[2],result)
    print(json.dumps({key:result.get(key) for key in ('complete','passed','frames','scored_rows',
        'actual_counts','public_core_input_bindings_checked','max_numeric_difference','failure_count')}),flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--input',type=Path,default=DOC)
    parser.add_argument('--protocol',type=Path,default=DOC/'PROTOCOL.json')
    parser.add_argument('--output',type=Path,default=DOC,help='new supplemental protocol/receipt directory')
    parser.add_argument('--statistics-verification',type=Path,help='optional completed verify.py VERIFICATION.json; bind only')
    parser.add_argument('--export-csv',type=Path,help='optional new all1225 per-frame CSV')
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__':
    main()
