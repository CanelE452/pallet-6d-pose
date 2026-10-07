"""Verify saved native rows, denominator, preservation and the real execution budget."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT, DOC, OUTPUT, ARMS, read, write, sha, digest, iter_rows


def verify():
    assert not (DOC/'CHECKS.json').exists(), 'Retain the first completed verification receipt'
    protocol=read(DOC/'PROTOCOL.json'); receipt=read(DOC/'PREDICTIONS.json')
    assert receipt['complete'] and receipt['protocol_sha256']==sha(DOC/'PROTOCOL.json')
    assert receipt['execution']['new_F_accuracy']==1276 and receipt['execution']['new_F_BASE_parity']==26
    assert receipt['raw_rows_sha256']==sha(DOC/receipt['raw_rows_file'])
    assert read(DOC/'CHECKS_METHODS.json')['status']=='PASS'
    assert read(DOC/'BASE_PARITY.json')['PASS'] and read(DOC/'BASE_PARITY.json')['calls']==26
    for b in protocol['code']:
        assert sha(ROOT/b['path'])==b['sha256']
    for b in protocol['dependencies']:
        assert sha(ROOT/b['path'])==b['sha256']
    manifest={r['id']:r for r in protocol['input_manifest']}
    rows=defaultdict(dict)
    for row in iter_rows():
        assert row['id'] not in rows[row['method']]
        rows[row['method']][row['id']]=row
    assert set(rows)==set(ARMS)
    source_checks=0
    for fid,b in manifest.items():
        for key in ('image','cache'):
            assert sha(ROOT/b[key])==b[key+'_sha256'],b[key]
            source_checks+=1
        raw=np.array(b['initial_points'],float); support=np.array(b['prediction_support'],bool)
        usable=support[:8]&np.isfinite(raw[:8]).all(-1)&~(raw[:8]==-1).all(-1)
        for algorithm in ('SUBPIX','CVRANK'):
            native=rows[algorithm+'_NATIVE'][fid]; capped=rows[algorithm+'_CAP1'][fid]
            assert native['correction']['diagnostics']==capped['correction']['diagnostics']
            q=np.array(native['native_points'],float); z=np.array(capped['native_points'],float)
            h,w=b['raw_hw']; cap=.01*np.hypot(w,h)
            expected=raw.copy();d=q[:8][usable]-raw[:8][usable]
            length=np.linalg.norm(d,axis=-1);factor=np.ones(len(d));moving=length>0
            factor[moving]=np.minimum(1.,cap/length[moving])
            expected[:8][usable]=raw[:8][usable]+d*factor[:,None]
            assert np.array_equal(expected,z,equal_nan=True)
            for r,p in ((native,q),(capped,z)):
                assert np.array_equal(raw,np.array(r['RAW_native_points'],float),equal_nan=True)
                assert np.array_equal(support,np.array(r['prediction_support'],bool))
                assert np.array_equal(raw[8],p[8],equal_nan=True)
                assert np.array_equal(raw[~support],p[~support],equal_nan=True)
                assert np.isfinite(p[support]).all()
                assert r['fixed_metadata']['preserved']
                assert r['fixed_metadata']['selected_index']==b['selected_index']==r['selected_index']
                assert digest(r['fixed_metadata']['metadata'])==digest(b['candidate_metadata'])
                assert np.allclose(np.linalg.norm(p[:8]-raw[:8],axis=-1),r['correction']['displacement_px8'],rtol=0,atol=0)
                assert all(flag==np.array_equal(p[k],raw[k],equal_nan=True) for k,flag in enumerate(r['correction']['corner_unchanged8']))
    population={}
    counts=Counter()
    for arm, values in rows.items():
        assert set(values)==set(manifest)
        ordered=[values[i] for i in protocol['population']['frame_ids']]
        summary=dict(frames=len(ordered),reference_corners=sum(len(r['corner']['errors']) for r in ordered),
            observed_corners=sum(len(r['corner']['observed_errors']) for r in ordered),
            matched=sum(r['corner']['matched'] for r in ordered),
            pose_available=sum(r['pose']['available'] for r in ordered),
            failures=sum(not r['pose']['available'] for r in ordered),
            actual_NoOp_frames=sum(all(r['correction']['corner_unchanged8']) for r in ordered))
        assert (summary['frames'],summary['reference_corners'],summary['observed_corners'],summary['matched'])==(319,2499,2445,311)
        population[arm]=summary
        for r in ordered:counts.update(r['PnP_counts'])
    assert dict(counts)==receipt['execution']['accuracy_PnP_counts']
    runtime=read(DOC/'RUNTIME.json')
    runtime_counts=runtime['execution']
    assert runtime_counts['pipeline_calls_started']<=1050
    assert runtime_counts['final_F_calls_started']<=1050
    assert runtime_counts['final_F_calls_complete']<=runtime_counts['final_F_calls_started']
    if runtime['complete']:
        assert runtime['status']=='DONE'
        consumed=runtime['warmup_accounting']['BASE']['failed_consumed']
        assert consumed in (0,1)
        assert runtime_counts['pipeline_calls_started']==runtime_counts['detector_calls']==runtime_counts['final_F_calls_complete']==1050
        assert runtime_counts['pipeline_calls_complete']==1050-consumed
        with gzip.open(runtime['raw_rows']['path'],'rt') as f: timed=[json.loads(line) for line in f]
        assert len(timed)==1050-consumed
        assert Counter(r['arm'] for r in timed if r['phase']=='warmup')=={a:20-(consumed if a=='BASE' else 0) for a in runtime['arms']}
        assert Counter(r['arm'] for r in timed if r['phase']=='measured')=={a:130 for a in runtime['arms']}
        assert runtime_counts['full_measured_rows']==910
        for r in timed:
            assert r['parity_status']=='PASS' and r['RAW_cached_bitexact'] and r['detection_and_center_missing_preserved']
            assert r['full_ms']>=0
            assert np.isclose(r['full_ms'],r['detector_ms']+(r['stage_only_ms'] or 0)+r['F_ms'],atol=1)
    for b in runtime['input_bindings']:
        assert sha(b['path'])==b['sha256'],b['path']
    square_status='NOT_AVAILABLE'
    square_rows=0
    if (DOC/'SQUARE_PREDICTIONS.json').exists():
        square=read(DOC/'SQUARE_PREDICTIONS.json');square_status=square.get('status','DONE' if square.get('complete') else 'BLOCKED')
        assert square['complete'] and square['direct_input_verified']
        assert square['square_protocol_sha256']==sha(DOC/'SQUARE_PROTOCOL.json')
        assert square['raw_rows_sha256']==sha(ROOT/square['raw_rows_file'])
        assert square['input_preservation_after_actual_sha']=='PASS'
        for b in square['source_bindings']:
            assert sha(ROOT/b['path'])==b['sha256'],b['path']
        square_rows=square['rows'];sqcounts=square['counts']
        assert square_rows==sqcounts['new_2d_scored_rows']==476
        assert sqcounts['native_algorithm_calls']==sqcounts['cap_pure_calls']==238
        assert sqcounts['native_SUBPIX_calls']==sqcounts['native_CVRANK_calls']==119
        assert sqcounts['new_2d_mode_measure_calls']==952
        assert square['new_F_calls']==sqcounts['final_F_calls']==sqcounts['detector_calls']==sqcounts['refiner_calls']==sqcounts['optimizer_updates']==0
        with gzip.open(ROOT/square['raw_rows_file'],'rt') as f:sqrows=[json.loads(line) for line in f]
        assert Counter(r['method'] for r in sqrows)=={a:119 for a in ARMS}
        for arm in ARMS:
            rr=[r for r in sqrows if r['method']==arm]
            assert len({r['id'] for r in rr})==119
            for mode,denominator in square['reference'].items():
                assert sum(len(r['corners'][mode]['errors']) for r in rr)==denominator
    total_F=receipt['execution']['new_F_BASE_parity']+receipt['execution']['new_F_accuracy']+runtime_counts['final_F_calls_started']
    assert total_F<=2352 and square_rows<=476
    initial=read(OUTPUT/'INITIAL_STATE.json')
    for p,h in initial['tracked_sha256'].items():
        assert sha(ROOT/p)==h,'User file changed: '+p
    current_flags=subprocess.check_output(['git','ls-files','-v'],cwd=ROOT).decode().splitlines()
    assert current_flags==initial['skip_flags'],'Tracked file flags changed'
    old_untracked={p for p in initial['untracked'] if p and 'pallet_training_free_compare_20261007_v1' not in p}
    assert all((ROOT/p).exists() for p in old_untracked),'Existing untracked path removed'
    changed=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).decode().split('\0')
    assert {p for p in changed if p}==set(initial['tracked_sha256']),'Unrequested tracked source/manuscript changed'
    result=dict(status='PASS',schema='training_free_saved_row_verification_v1',
        protocol_sha256=sha(DOC/'PROTOCOL.json'),row_sha256=sha(DOC/'PREDICTIONS.jsonl.gz'),
        checks=dict(method_tests=True,BASE_parity_26_actual_F=True,IDs_unique_full319=True,
            reference2499_matched311_observed2445=True,native_same_initial_9points=True,
            center_box_score_confidence_selection_masks_preserved=True,
            CAP1_exact_same_native_formula_and_zero=True,no_GT_into_correction=True,
            original_images_caches_used_references_unchanged=True,runtime_budget_accounted=True,
            current_user_changes_preserved=True,existing_untracked_paths_preserved=True,
            tracked_file_flags_preserved=True,no_paper_latex_pdf_bibliography_changes=True),
        population=population,source_file_checks=source_checks,
        execution=dict(new_F_accuracy=1276,new_F_BASE_parity=26,
            new_F_runtime_started=runtime_counts['final_F_calls_started'],
            new_F_runtime_complete=runtime_counts['final_F_calls_complete'],new_F_total=total_F,
            new_F_total_ceiling=2352,accuracy_PnP_counts=dict(counts),
            parity_PnP_counts=read(DOC/'BASE_PARITY.json')['PnP_counts'],
        runtime_PnP_counts={k:runtime_counts[k] for k in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')},
            runtime_detector_calls=runtime_counts['detector_calls'],runtime_head_forwards=runtime.get('execution_model_forwards'),
            accuracy_detector_head_calls=0,new_training=0,new_annotations=0,parameter_search=0,
            optional_square_new_2D_rows=square_rows,optional_square_F=0),
        square_status=square_status,runtime_status=runtime['status'],
        runtime_warmup_accounting=runtime.get('warmup_accounting'),
        runtime_failed_attempts=dict(setup=read(DOC/'RUNTIME_SETUP_ATTEMPT.json')['reason'],
            consumed_first_warmup=read(DOC/'RUNTIME_WARMUP_ATTEMPT.json')['reason'],
            failed_warmup_F_calls=read(DOC/'RUNTIME_WARMUP_ATTEMPT.json')['execution']['final_F_calls_started'],
            failed_warmup_included_in_total_F=True,failed_or_fallback_timing_pooled=False),
        setup_attempts=[dict(stage='pre_protocol_preparation',reason='readonly a22 worktree has no TARGETS file; saved baseline target SHA is used instead',F=0,NN=0),
            dict(stage='pre_protocol_write',reason='root disk full; preserved temporary browser caches in RAM, then sealed protocol before evaluation',F=0,NN=0),
            dict(stage='square_pre_protocol_compile',reason='unmatched parenthesis corrected before any input/evaluation execution',F=0,NN=0)],
        storage=dict(temporary_root=str(OUTPUT),browser_cache_relocation=read(OUTPUT/'TEMP_STORAGE_RECEIPT.json'),
            source_data_or_scientific_results_deleted=False),
        method_configuration_changes_after_results=0,additional_F_for_this_verification=0,
        limitations=['Only used source files were hashed; existing unrelated multi-gigabyte caches were not exhaustively rehashed.',
            'Original CVRANK collector suppresses individual-family exceptions; no zero-hidden-error claim.',
            'Existing real pose references are reconstructed, not newly measured independent GT.'])
    write(DOC/'CHECKS.json',result)
    print('CHECKS_PASS',result['execution'])
    return result


if __name__=='__main__':
    verify()
