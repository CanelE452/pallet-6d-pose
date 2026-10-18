"""Count actual successful provider receipts, exposing the SAM hook gap.

No inference, mask generation, pose solving, references, or source imports.
The missing CUDA intervals are intentionally not filled by extrapolation.
"""
import argparse,gzip,hashlib,json,time
from collections import Counter
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(p):
    with gzip.open(p,'rt')as f:return json.loads(next(f))


def audit(private,output):
    started,cpu=time.monotonic(),time.process_time();root=Path(private);domains={};failures=[]
    for domain,folder in [('real','d1_real_observations_v1'),('synthetic','d1_synthetic_observations_v1')]:
        out=root/folder;seal=read(out/'OBSERVATION_SEAL.json');recorded=Counter();derived=Counter();phases=Counter();gpu=Counter();peak=0
        if not seal['complete']:raise ValueError('Completed observations required')
        for i,ident in enumerate(seal['ordered_ids']):
            f=out/'frames'/f'{i:04d}';s=read(f/'OBSERVATION_SEAL.json');row=one(f/'OBSERVATION.jsonl.gz')
            if s['id']!=ident or s['observation_sha256']!=sha(f/'OBSERVATION.jsonl.gz'):
                raise ValueError('Original observation binding mismatch')
            calls=s['resource']['actual_calls'];recorded.update({k:v for k,v in calls.items()if k.endswith('_calls')and type(v)is int})
            sam=int(row['tight_mask_provenance'].get('mask_forward_calls',0))+int(row['full_mask_provenance'].get('SAM_mask_forward_calls',0))
            derived['sam_set_image_calls']+=sam;derived['sam_predict_calls']+=sam
            derived['sam_image_encoder_forward_calls']+=sam;derived['sam_mask_decoder_forward_calls']+=sam
            phases['frames_with_SAM_hook_capture_gap']+=calls['sam_image_encoder_forward_calls']!=sam
            phases['actual_original_recovery_stages']+=len(row['recovery_stage_audit'])
            phases['actual_RECOVER_stages']+=sum('CORNER_REACH'not in r['stage']for r in row['recovery_stage_audit'])
            phases['actual_REACH_stages']+=sum('CORNER_REACH'in r['stage']for r in row['recovery_stage_audit'])
            if calls['sam_image_encoder_forward_calls']>sam:failures.append(dict(id=ident,error='hook count exceeds actual successful provider receipts'))
            gpu.update(s['resource']['CUDA_forward_seconds_by_module']);peak=max(peak,s['resource']['peak_allocated_GPU_bytes'])
        domains[domain]=dict(rows=seal['rows'],observation_seal_sha256=sha(out/'OBSERVATION_SEAL.json'),
            direct_hook_counts=dict(recorded),SAM_actual_calls_reconstructed_from_successful_provider_receipts=dict(derived),
            original_recovery_stage_receipts=dict(phases),recorded_partial_CUDA_forward_seconds_by_module=dict(gpu),
            recorded_partial_CUDA_forward_seconds=sum(gpu.values()),GPU_observation_host_wall_upper_bound_seconds=seal['wall_seconds'],
            peak_allocated_GPU_bytes=peak,SAM_CUDA_events_missing_after_first_frame=True,
            provider_receipt_semantics='mask_forward_calls/SAM_mask_forward_calls are written only after actual set_image and predict return; frozen source executes one encoder and one decoder per successful provider call')
    result=dict(schema='D1_ACTUAL_OBSERVATION_COST_AUDIT_V1',status='PASS_WITH_CAPTURE_GAP'if not failures else'FAIL',
        domains=domains,failures=failures,
        actual_GPU_activity_exact_total_seconds=None,
        missing_CUDA_seconds='NOT_MEASURED; no extrapolation from another frame or zero substitution',
        total_observation_GPU_process_wall_upper_bound_seconds=sum(d['GPU_observation_host_wall_upper_bound_seconds']for d in domains.values()),
        extra_model_forwards_for_this_audit=0,original_predictions_or_masks_changed=False,
        limitations='Direct CUDA event/hook values omit cached SAM predictor forwards after the first frame of each domain. Successful provider receipts reconstruct actual call counts separately, while whole observation host wall bounds total GPU-process duration. Cold/warm one-frame profiling is a separate technical artifact.',
        source_sha256=sha(__file__),audit_wall_seconds=time.monotonic()-started,audit_CPU_seconds=time.process_time()-cpu)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.private,a.output);print(json.dumps(dict(status=r['status'],observation_host_wall=r['total_observation_GPU_process_wall_upper_bound_seconds'])))
