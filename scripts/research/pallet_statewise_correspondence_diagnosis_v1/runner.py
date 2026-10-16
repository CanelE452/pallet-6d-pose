"""Fixed four-path inference, immutable per-slot caches, no reference access."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import time
import traceback

from .common import ARMS, budget_elapsed, digest, finite, read, rows, save_rows, sha, write, record_stage
from . import reobs


def source_binding(root, arm):
    names = ('common.py', 'reobs.py' if arm.startswith('REOBS') else 'typed_replay.py')
    return {name:sha(Path(root)/'scripts/research/pallet_statewise_correspondence_diagnosis_v1'/name)
            for name in names if (Path(root)/'scripts/research/pallet_statewise_correspondence_diagnosis_v1'/name).exists()}


def execute(job):
    before_cpu = time.process_time()
    start = time.monotonic()
    out, arm, index, frame, packet, source_sha = job
    target = Path(out)/'slots'/f'{index:03d}_{arm}.json'
    binding = digest(dict(packet=packet,arm=arm,source=source_sha))
    if target.exists():
        cached = read(target)
        if (cached['id'],cached['index'],cached['arm']) != (frame['id'],index,arm):
            raise ValueError('Existing slot identity/arm mismatch: '+str(target))
        # The first REOBS pilot bound the independent typed module too. Keep
        # that original receipt; reuse only if packet and every consumed solver
        # module byte are identical. No input or numerical result is rewritten.
        relevant_match = (cached['packet_sha256'] == digest(packet) and
                          all(cached['source_sha256'].get(k)==v for k,v in source_sha.items()))
        if cached['input_binding_sha256'] != binding and not relevant_match:
            raise ValueError('Existing slot belongs to a different input/source contract: '+str(target))
        return dict(id=frame['id'],arm=arm,reused=True,elapsed_seconds=0.,worker_cpu_seconds=0.,
                    original_binding_sha256=cached['input_binding_sha256'],relevant_source_bytes_identical=True)
    try:
        if arm.startswith('REOBS'):
            result = (reobs.solve_multi if arm.endswith('MULTI') else reobs.solve_single)(packet)
        else:
            from . import typed_replay
            result = (typed_replay.run_multi if arm.endswith('MULTI') else typed_replay.run_single)(packet)
        result.update(technical_failure=False,timeout=False)
        status = 'COMPLETE'
    except Exception as exc:
        # A technical failure is retained, not silently converted to a numerical rejection.
        result = dict(technical_failure=True,timeout=False,accepted=None,fallback=None,
                      actual_pose=dict(available=False),error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc(),uses_e_initialization=True)
        status = 'TECHNICAL_FAILURE'
    row = finite(dict(id=frame['id'],index=index,arm=arm,partition=frame['partition'],
        session=frame['session'],cluster_id=frame['cluster_id'],input_binding_sha256=binding,
        packet_sha256=digest(packet),source_sha256=source_sha,result=result,status=status,
        elapsed_seconds=time.monotonic()-start,worker_cpu_seconds=time.process_time()-before_cpu,
        GT_inputs=False))
    write(target,row)
    return {k:row[k] for k in ('id','arm','status','elapsed_seconds','worker_cpu_seconds')}


def jobs_for(source_root, guarded_dir, output, selected_ids, arms):
    guarded_dir, output = Path(guarded_dir),Path(output)
    frames = read(guarded_dir/'e_inputs/INPUTS.json')['frames']
    typed = {r['id']:r for r in rows(output/'e_trace_v2/TYPED_PACKETS.jsonl.gz')} if any(a.startswith('TYPED') for a in arms) else {}
    jobs=[]
    for i,f in enumerate(frames):
        if f['id'] not in selected_ids: continue
        for arm in arms:
            if arm.startswith('REOBS'):
                obs=read(guarded_dir/f'e_run/observations/{i:03d}_E_FROZEN.json')['result']['observation']
                packet=reobs.packet(obs,f['arms']['E_FROZEN']['coarse_pose'],f['K'],f['raw_hw'])
            else:
                packet=deepcopy(typed[f['id']])
            jobs.append((str(output),arm,i,f,packet,source_binding(source_root,arm)))
    return jobs


def run(source_root,guarded_dir,output,*,phase,arms,workers=1):
    out=Path(output)
    protocol=read(out/'REPLAY_PROTOCOL.json')
    if workers not in (1,2,3): raise ValueError('Maximum three workers')
    if not set(arms)<=set(ARMS): raise ValueError('Unregistered arm')
    if phase=='full' and not (out/'PILOT_AUTHORIZATION.json').exists():
        raise ValueError('Full execution requires parity/cost inspection first')
    if budget_elapsed(out)['cpu_solver_analysis'] >= 7200:
        raise ValueError('CPU stage elapsed budget exhausted')
    ids = protocol['pilot_ids'] if phase=='pilot' else protocol['ordered_ids']
    jobs=jobs_for(source_root,guarded_dir,output,set(ids),arms)
    started=time.monotonic()
    answers=[]
    if workers==1:
        for job in jobs:
            ans=execute(job); answers.append(ans); print(json.dumps(ans),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for ans in executor.map(execute,jobs):
                answers.append(ans); print(json.dumps(ans),flush=True)
    ledger=record_stage(out,f'{phase}_'+ '_'.join(arms),started,slots=len(jobs),
        new_slots=sum(not a.get('reused',False) for a in answers),
        worker_cpu_seconds=sum(a['worker_cpu_seconds'] for a in answers),workers=workers)
    # Every attempt remains in the append-only ledger and per-slot evidence.
    return dict(ledger=ledger,results=answers)


def assemble(output,guarded_dir):
    out=Path(output)
    parity=read(out/'REPLAY_PARITY.json')
    if parity['rows']!=238 or not parity['all_passed']:
        raise ValueError('Complete strict original replay parity must precede assembly')
    supplement=out/'supplemental_cal_start_logging_v1/SUPPLEMENT_SEAL.json'
    if not supplement.exists():
        raise ValueError('Missing one-CAL raw-start logging supplement')
    frames=read(Path(guarded_dir)/'e_inputs/INPUTS.json')['frames']
    predictions=[]; candidates=[]
    for i,f in enumerate(frames):
        values={}; generated={}
        for arm in ARMS:
            slot=read(out/f'slots/{i:03d}_{arm}.json')
            if slot['id']!=f['id']: raise ValueError('ID mismatch')
            values[arm]=slot['result']
            if arm=='TYPED_MULTI':
                generated=values[arm].get('generated_candidates',[])
        predictions.append(dict(id=f['id'],partition=f['partition'],session=f['session'],
            cluster_id=f['cluster_id'],predictions=values,GT_inputs=False))
        candidates.append(dict(id=f['id'],index=i,candidates=generated,GT_inputs=False))
    save_rows(out/'REPLAY_PREDICTIONS.jsonl.gz',predictions)
    save_rows(out/'GENERATED_CANDIDATES.jsonl.gz',candidates)
    cand=dict(generated_candidates_sha256=sha(out/'GENERATED_CANDIDATES.jsonl.gz'),
        ordered_ids=[f['id'] for f in frames],rows=238,GT_inputs=False,before_new_scoring=True,
        coverage='ORIGINAL_E_CONSUMER_ENDPOINTS_AND_INCOMING_STARTS_WITH_SUPPLEMENT',
        initial_start_supplement_seal_sha256=sha(supplement),
        original_candidate_manifest_sha256=sha(out/'original_candidate_archive_v1/CANDIDATE_MANIFEST.json'),
        strict_replay_parity_sha256=sha(out/'REPLAY_PARITY.json'),
        earlier_WD_P1_rejected_proposals_not_consumed_by_original_e=True,
        optimizer_internal_iterates_not_claimed_as_output_candidates=True)
    write(out/'CANDIDATE_SEAL.json',cand)
    write(out/'PREDICTION_SEAL.json',dict(predictions_sha256=sha(out/'REPLAY_PREDICTIONS.jsonl.gz'),
        ordered_ids=cand['ordered_ids'],arms=list(ARMS),rows=238,GT_inputs=False,
        before_new_scoring=True,candidate_seal_sha256=sha(out/'CANDIDATE_SEAL.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True); p.add_argument('--guarded-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--phase',choices=['pilot','full','assemble'],required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.phase=='assemble': assemble(a.output,a.guarded_dir)
    else: run(a.source_root,a.guarded_dir,a.output,phase=a.phase,arms=a.arms,workers=a.workers)
