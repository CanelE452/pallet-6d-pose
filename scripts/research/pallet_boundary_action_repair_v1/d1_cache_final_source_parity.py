"""Bind the final frozen memo source to the already solved fixed CAL-free row.

This technical replay never executes a new neural forward, PnP, or optimizer.
Its actual-e comparator is the preserved first-REMAINDER technical execution.
"""
from __future__ import annotations
import argparse,time
from pathlib import Path

from . import d1_stage_memoization as M
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,rows,sha,write


def replay(private,old_run,original_run,typed_run):
    started,cpu=time.monotonic(),time.process_time();root=Path(private)
    choose=read(root/'D1_CACHE_PARITY_FIXED_SELECTION.json');index=choose['index']
    frame=list(rows(root/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz'))[index]
    if frame['id']!=choose['id']:raise ValueError('Preselected original input-order row changed')
    lock=read(root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json')
    if lock['additional_source_sha256']['scripts/research/pallet_boundary_action_repair_v1/d1_stage_memoization.py']!=sha(M.__file__):
        raise ValueError('Final frozen memo source changed')
    actual=root/'d1_cache_parity_actual/frames'/f'{index:04d}'
    destination=root/'d1_cache_parity_memoized_finalsource/frames'/f'{index:04d}'
    seal=M.solver_job(frame,root/'d1_real_observations_v1/frames'/f'{index:04d}',destination,old_run,
        M.load_entry(frame['id'],original_run,typed_run))
    a=next(rows(actual/'E_RUN.jsonl.gz'));b=next(rows(destination/'E_RUN.jsonl.gz'))
    fa=next(rows(actual/'FEATURES.jsonl.gz'));fb=next(rows(destination/'FEATURES.jsonl.gz'))
    essential=('candidate_id','source','family','R','t','R_physical','centroid')
    endpoints=('stage','family','kind','R','t')
    checks=dict(all_candidate_62features_pose_aliases_eligible_exact=fa['candidates']==fb['candidates'],
        feature_names_exact=fa['feature_names']==fb['feature_names'],fallback_N0_exact=fa['fallback']==fb['fallback'],
        E0_pose_exact=a['e0_regenerated']==b['e0_regenerated'],
        raw_candidate_numeric_and_identity_exact=[{k:r.get(k)for k in essential}for r in a['candidates']]==[{k:r.get(k)for k in essential}for r in b['candidates']],
        all_endpoint_numeric_identity_and_order_exact=[{k:r.get(k)for k in endpoints}for r in a['endpoint_journal']]==[{k:r.get(k)for k in endpoints}for r in b['endpoint_journal']],
        final_frozen_module_source_bound=b['proof']['exact_stage_memoization']['source_sha256']==sha(M.__file__),
        all_actual_stage_inputs_validated=all(r['exact']for r in b['proof']['exact_stage_memoization']['input_checks']),
        new_numeric_calls_zero=not any(seal['operations'].values()))
    result=dict(schema='D1_FINAL_FROZEN_CACHE_SOURCE_PARITY_V1',status='PASS'if all(checks.values())else'FAIL',
        id=frame['id'],index=index,checks=checks,reference_or_accuracy_reads=0,new_neural_forwards=0,
        actual_e_executed_again=False,current_source_frame_seal_sha256=sha(destination/'FRAME_SEAL.json'),
        preserved_actual_frame_seal_sha256=sha(actual/'FRAME_SEAL.json'),
        earlier_full_numeric_diagnostic_parity_sha256=sha(root/'D1_EXACT_MEMOIZATION_ACTUAL_PARITY.json'),
        technical_amendment_lock_sha256=sha(root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json'),
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,source_sha256=sha(__file__))
    write(root/'D1_FINAL_SOURCE_MEMO_PARITY.json',result)
    if result['status']!='PASS':raise ValueError('Final frozen-source exact memo parity failed; original outputs retained')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--old-run',required=True)
    p.add_argument('--original-run',required=True);p.add_argument('--typed-run',required=True);a=p.parse_args()
    print(replay(a.private,a.old_run,a.original_run,a.typed_run))
