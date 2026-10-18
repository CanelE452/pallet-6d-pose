"""Fixed first-CAL cold/warm cost measurement with persistent model hooks.

Technical reexecution only. No generated mask, state, or coordinate replaces
the sealed D1 observation stream, and no pose/accuracy evaluator is called.
"""
from __future__ import annotations
import argparse,time
from pathlib import Path
from . import d1_producer as P
from .d1_batch import NeuralMeter,firewall
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,rows,sha,write,canonical


def run(inputs,old_run,output,GO=False):
    if not GO:raise ValueError('Explicit root technical-profile GO required')
    guard=firewall(old_run);frame=next(f for f in rows(inputs)if f['partition']=='CAL')
    import torch
    started,cpu=time.monotonic(),time.process_time();load_started=time.monotonic()
    bundle=P.load_models('cuda');load_wall=time.monotonic()-load_started
    records=[]
    # One context across both passes keeps the loaded SAM class/encoder/decoder
    # hooks installed. New per-frame contexts caused the prior capture gap.
    with NeuralMeter()as meter:
        for phase in ('cold','warm'):
            meter.meter.reset();meter.events=[]
            if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            tick,tcpu=time.monotonic(),time.process_time()
            observation,arrays=P.observe(frame,bundle)
            resource=meter.receipt();resource.update(phase=phase,host_wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-tcpu)
            counts=resource['actual_calls']
            for key,n in [('tiny_forward_calls',1),('state_backbone_forward_calls',1),
                ('sam_image_encoder_forward_calls',2),('sam_mask_decoder_forward_calls',2),('state_classifier_head_calls',1)]:
                if counts[key]!=n:raise AssertionError('Persistent actual-call measurement mismatch: '+phase+'/'+key)
            records.append(resource)
            del observation,arrays
    result=dict(schema='D1_PERSISTENT_HOOK_COLD_WARM_PROFILE_V1',status='PASS',
        fixed_selection='first original input-order CAL frame before any new accuracy read',id=frame['id'],
        inputs_sha256=sha(inputs),image_sha256=frame['image_sha256'],source_sha256=sha(__file__),
        model_bindings=bundle['bindings'],model_load_host_seconds=load_wall,phases=records,
        total_host_seconds=time.monotonic()-started,total_CPU_seconds=time.process_time()-cpu,
        extra_observation_reexecutions=2,extra_top_level_neural_forwards=12,
        new_optimization_calls=0,labels_or_accuracy_read=False,production_outputs_replaced=False,
        measurement_scope='one fixed CAL image; CUDA-event intervals around all top-level module forwards; not whole-population timing or Jetson timing',
        original_bulk_capture_gap_preserved=True,reference_firewall=guard.receipt())
    write(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--old-run',required=True);p.add_argument('--output',required=True);p.add_argument('--root-GO',action='store_true')
    a=p.parse_args();r=run(a.inputs,a.old_run,a.output,a.root_GO);print(canonical(dict(status=r['status'],phases=[dict(phase=x['phase'],wall_seconds=x['host_wall_seconds'],CUDA_seconds=x['CUDA_forward_seconds'])for x in r['phases']])))
