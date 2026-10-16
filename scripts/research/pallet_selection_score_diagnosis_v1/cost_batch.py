"""Stream original fixed-candidate costs with two bounded CPU workers.

Dense outline arrays stay in per-frame private journals.  The compact joined
journal retains every candidate/observation scalar and an exact pointer to
those arrays, without recomputing or rounding any value.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import time
import traceback

from .common import read, rows, sha, write
from .fixed_cost import evaluate, prepare
from .runtime_guard import FixedPoseOnly

ARRAY_FIELDS={'sample_xy','residual_xy_px','standardized_radius','robust_losses',
              'arc_or_sample_weights','weighted_sample_contributions'}


def emit(stream,value):
    # evaluate() already converts ndarray/scalars to finite native JSON types.
    stream.write((json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())


def compact_cost(value,frame_path,line_number):
    out={**value,'dense_outline_arrays':False,
         'dense_outline_stored_in':dict(path=frame_path,line_number=line_number,
                                       candidate_id=value['candidate_id'])}
    out['per_observation']=[{k:v for k,v in row.items() if k not in ARRAY_FIELDS}
                            for row in value.get('per_observation',[])]
    return out


def frame_job(output,index,packet,candidates):
    started,cpu=time.monotonic(),time.process_time();output=Path(output)
    full_rel=f'frames/{index:03d}_COSTS.jsonl.gz';compact_rel=f'compact_frames/{index:03d}_COSTS.jsonl.gz'
    full_path,compact_path=output/full_rel,output/compact_rel
    contexts={};guard=FixedPoseOnly();errors=[];counts=Counter();maxima=Counter();rows_written=0
    with full_path.open('xb') as full_raw,compact_path.open('xb') as compact_raw:
        with gzip.GzipFile(fileobj=full_raw,mode='wb',mtime=0,compresslevel=3) as full_stream, \
             gzip.GzipFile(fileobj=compact_raw,mode='wb',mtime=0,compresslevel=3) as compact_stream:
            for candidate in candidates:
                identity=dict(candidate_id=candidate['candidate_id'],id=packet['id'],index=index,
                              partition=packet['partition'],session=packet['session'],
                              stage=candidate['stage'],family=candidate['family'])
                try:
                    with guard:
                        key=(candidate['stage'],candidate['family'])
                        if key not in contexts:
                            contexts[key]=prepare(packet,*key,scalar_only=True)
                        result=evaluate(packet,*key,candidate,context=contexts[key],dense_outline=True,strict=True)
                    result={**result,**identity,'status':'COMPLETE'}
                    counts['complete']+=1
                    for name,value in result['stored_component_differences'].items():
                        maxima['stored_component:'+name]=max(maxima['stored_component:'+name],abs(value))
                    for name,value in result['observation_component_differences'].items():
                        maxima['observation_component:'+name]=max(maxima['observation_component:'+name],abs(value))
                    maxima['stored_objective']=max(maxima['stored_objective'],abs(result['stored_objective_difference'] or 0.))
                    maxima['additive_sum']=max(maxima['additive_sum'],abs(result['additive_difference']))
                    if result['stored_objective_parity'] is not True:
                        counts['stored_objective_unavailable']+=1
                    if result['stored_component_parity'] is None:
                        counts['stored_components_unavailable']+=1
                except Exception as error:
                    result={**identity,'status':'COST_EVALUATION_UNRESOLVED','reason':str(error),
                            'error_type':type(error).__name__,'objective':None,'weighted_components':None,
                            'per_observation':[],'traceback':traceback.format_exc(),
                            'new_pose_solve_calls':0,'new_candidates':0}
                    errors.append({k:result[k] for k in ('candidate_id','reason','error_type')})
                    counts['unresolved']+=1
                rows_written+=1
                emit(full_stream,result)
                emit(compact_stream,compact_cost(result,full_rel,rows_written))
    receipt=dict(id=packet['id'],index=index,partition=packet['partition'],candidates_expected=len(candidates),
                 candidates_written=rows_written,counts=dict(counts),max_absolute_differences=dict(maxima),
                 dense_path=full_rel,dense_sha256=sha(full_path),dense_bytes=full_path.stat().st_size,
                 compact_path=compact_rel,compact_sha256=sha(compact_path),compact_bytes=compact_path.stat().st_size,
                 runtime_guard=guard.receipt(),errors=errors,elapsed_seconds=time.monotonic()-started,
                 worker_cpu_seconds=time.process_time()-cpu,new_pose_calls=0,new_model_forwards=0)
    write(output/f'frames/{index:03d}_RECEIPT.json',receipt)
    return receipt


def run(old_root,selection_directory,pilot_receipt,output,*,workers=2):
    if workers not in (1,2):raise ValueError('At most two workers authorized for this stage')
    started,cpu=time.monotonic(),time.process_time()
    old_root,selection_directory,pilot_receipt,output=map(Path,(old_root,selection_directory,pilot_receipt,output))
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir();(output/'compact_frames').mkdir()
    source=Path(__file__).with_name('fixed_cost.py');pilot=read(pilot_receipt)
    if pilot['source_sha256']!=sha(source):raise ValueError('Fixed cost source differs from scalar-only pilot gate')
    if pilot['max_outline_scalar_only_difference']>1e-9 or not pilot['all_stored_objective_parity']:
        raise ValueError('Original objective/scalar gate did not pass')
    packets=list(rows(old_root/'e_trace_v2/TYPED_PACKETS.jsonl.gz'))
    selection=read(selection_directory/'SELECTION_REPLAY.json')
    if not selection['all_passed'] or selection['passed_rows']!=238:raise ValueError('238 original selection replay required')
    by_id={p['id']:[] for p in packets}
    for candidate in rows(selection_directory/'RETAINED_CANDIDATES.jsonl.gz'):
        by_id[candidate['id']].append(candidate)
    ids=[p['id'] for p in packets]
    if len(ids)!=238 or len(set(ids))!=238 or sum(map(len,by_id.values()))!=11603:
        raise ValueError('Exact original 238/11603 candidate population required')
    if pilot['ids']!=[p['id'] for p in packets if p['partition']=='CAL'][:6]:
        raise ValueError('Pilot was not the original CAL ordered first six')
    bindings={str(p):sha(p) for p in (source,Path(__file__),pilot_receipt,
        old_root/'e_trace_v2/TYPED_PACKETS.jsonl.gz',selection_directory/'SELECTION_SEAL.json',
        selection_directory/'RETAINED_CANDIDATES.jsonl.gz')}
    write(output/'INPUT_BINDINGS.json',dict(source_sha256=bindings,workers=workers,
        exact_pose_key='(id,candidate_id), original R/t/Q',scalar_only_gate=str(pilot_receipt),
        dense_outline_arrays_all_candidates=True,expected_frames=238,expected_candidate_rows=11603,
        new_pose_solves=0,new_model_forwards=0,new_observation_generations=0))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs={executor.submit(frame_job,str(output),i,p,by_id[p['id']]):i for i,p in enumerate(packets)}
        for future in as_completed(jobs):
            receipt=future.result();results.append(receipt)
            line=json.dumps(dict(completed_frames=len(results),**{k:receipt[k] for k in ('id','index','candidates_written','counts','elapsed_seconds')}),separators=(',',':'))+'\n'
            with (output/'PROGRESS.jsonl').open('a') as stream:stream.write(line)
            if len(results)%10==0 or len(results)==238:
                print(f'fixed-cost frames={len(results)}/238 rows={sum(r["candidates_written"] for r in results)}/11603 unresolved={sum(r["counts"].get("unresolved",0) for r in results)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    results.sort(key=lambda r:r['index'])
    joined=output/'CANDIDATE_COSTS.jsonl.gz'
    # Concatenate gzip members: valid streaming gzip, exact original row bytes.
    with joined.open('xb') as destination:
        for receipt in results:
            with (output/receipt['compact_path']).open('rb') as stream:
                while chunk:=stream.read(4*1024*1024):destination.write(chunk)
    counts=Counter();maxima=Counter();blocked=Counter()
    for receipt in results:
        counts.update(receipt['counts']);blocked.update(receipt['runtime_guard']['forbidden_blocked_attempts'])
        for key,value in receipt['max_absolute_differences'].items():maxima[key]=max(maxima[key],value)
    if any(sha(Path(p))!=s for p,s in bindings.items()):raise ValueError('Read-only input/source binding changed during execution')
    summary=dict(schema='FIXED_ORIGINAL_CANDIDATE_COST_BATCH_V1',frames=len(results),candidate_rows=sum(r['candidates_written'] for r in results),
        counts=dict(counts),all_complete=counts.get('complete',0)==11603 and not counts.get('unresolved',0),
        frame_ids=[r['id'] for r in results],stage_family_not_combined=True,atol=1e-9,rtol=0.,
        max_absolute_differences=dict(maxima),runtime_guard=dict(forbidden_blocked_attempts=dict(blocked),
            forbidden_function_bodies_executed=0,profiled_python_calls=sum(r['runtime_guard']['profiled_python_calls'] for r in results),
            scope='each fixed evaluator/prepare call in the two analysis workers; serialization is IO only'),
        dense_outline_arrays_preserved_for_all_rows=True,dense_bytes=sum(r['dense_bytes'] for r in results),
        compact_bytes=joined.stat().st_size,workers=workers,elapsed_seconds=time.monotonic()-started,
        parent_cpu_seconds=time.process_time()-cpu,worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in results),
        new_learning_updates=0,new_model_forwards=0,new_PnP_calls=0,new_optimizers=0,new_candidate_generation_calls=0,
        new_observation_generations=0,source_sha256=bindings,frame_receipts=results)
    write(output/'COST_SUMMARY.json',summary)
    seal=dict(schema='FIXED_ORIGINAL_CANDIDATE_COST_SEAL_V1',rows=summary['candidate_rows'],frames=238,
        all_complete=summary['all_complete'],summary_sha256=sha(output/'COST_SUMMARY.json'),
        input_bindings_sha256=sha(output/'INPUT_BINDINGS.json'),candidate_costs_sha256=sha(joined),
        dense_frame_journals=[dict(id=r['id'],index=r['index'],path=r['dense_path'],sha256=r['dense_sha256'],rows=r['candidates_written']) for r in results],
        new_pose_calls=0,new_model_forwards=0)
    write(output/'COST_SEAL.json',seal)
    return {k:summary[k] for k in ('frames','candidate_rows','counts','all_complete','elapsed_seconds','worker_cpu_seconds','dense_bytes','compact_bytes','max_absolute_differences','runtime_guard')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--old-root',required=True)
    parser.add_argument('--selection-directory',required=True);parser.add_argument('--pilot-receipt',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();print(json.dumps(run(args.old_root,args.selection_directory,args.pilot_receipt,args.output,workers=args.workers),indent=2))
