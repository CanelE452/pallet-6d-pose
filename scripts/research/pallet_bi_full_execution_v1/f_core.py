"""Conditional resource scheduling for unconditional, frozen F core banks.

This controller cannot train H1, select a final system, or read real references.
"""
from pathlib import Path
import argparse
import time
from . import s1_union as U, fullnormal_dispatch as D


def original_process_active(pid,expected_start):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        return fields[0]!='Z'and fields[19]==expected_start
    except FileNotFoundError:return False


def run(campaign,prior_run,*,GO=False,wait=False):
    if GO is not True:raise PermissionError('Root conditional F core GO required')
    campaign,prior_run=map(Path,(campaign,prior_run));go=U.read(campaign/'F_CORE_PRODUCTION_ROOT_GO.json')
    if not go['approved']or go['H1_fits_authorized']!=0 or go['final_configuration_selected']is not False:
        raise ValueError('Only honest unconditional core precomputation is authorized')
    if D.source_bindings()!=go['source_binding']:raise ValueError('Reviewed execution/kernel source changed')
    if U.sha(campaign/'F_CALIBRATION/PACKET_MANIFEST.json')!=go['packet_manifest_sha256']:
        raise ValueError('Own F observation/CAL packets changed')
    output=campaign/'F_CORE_THREE_CPU'
    if (output/'SUMMARY.json').exists()and U.read(output/'SUMMARY.json')['status']=='COMPLETE_NEW':
        raise RuntimeError('Completed F core must be authenticated/reused; do not rerun its gzip aggregator')
    pid=go['conditional_start']['owned_software_renderer_process_exited'];proc=Path('/proc')/str(pid)
    expected_start=None;command=''
    if proc.exists():
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'pallet_bi_full_execution_v1.image_batch'in command:
            text=(proc/'stat').read_text();expected_start=text[text.rfind(')')+2:].split()[19]
    controller=dict(schema='F_CORE_CONDITIONAL_RESOURCE_CONTROLLER_V1',source_sha256=U.sha(__file__),
        GO_sha256=U.sha(campaign/'F_CORE_PRODUCTION_ROOT_GO.json'),observed_owned_renderer_pid=pid,
        observed_owned_renderer_start_ticks=expected_start,observed_owned_renderer_command=command,
        output=str(output),workers=3,head_fit_calls=0,final_F_complete=False)
    U.write(campaign/'F_CORE_CONTROLLER_LOCK.json',controller)
    receipt=campaign/'s3_s4_production/FULL_RECEIPT.json';last_print=0.;start=time.monotonic()
    while True:
        done=receipt.exists()and U.read(receipt).get('all1241_completed')is True
        renderer=expected_start is not None and original_process_active(pid,expected_start)
        if done and not renderer:break
        if not wait:raise RuntimeError('Conditional resource gate not yet satisfied; use --wait')
        if time.monotonic()-last_print>=55:
            print(dict(stage='F_CORE_RESOURCE_WAIT',material_producer_complete=done,owned_renderer_still_active=renderer,wait_seconds=time.monotonic()-start),flush=True);last_print=time.monotonic()
        time.sleep(5)
    U.write(campaign/'F_CORE_RESOURCE_GATE_PASSED.json',dict(complete=True,controller_lock_sha256=U.sha(campaign/'F_CORE_CONTROLLER_LOCK.json'),
        renderer_original_process_exited=True,material_FULL_RECEIPT_sha256=U.sha(receipt),wait_wall_seconds=time.monotonic()-start,
        six_core_pose_workers_plus_one_sequential_postprocessor_max=True))
    result=D.run(prior_run,campaign/'F_CALIBRATION',campaign/'F_CALIBRATION/PACKET_MANIFEST.json',campaign/'ENVIRONMENT_INITIAL.json',output,
        arms=('F_COMMON','F_BI'),workers=3,reuse_root=campaign/'F_CORE_EXACT_REUSE',root_GO=True,
        stop_file=campaign/'F_CORE_CHECKPOINT_STOP')
    U.write(campaign/'F_CORE_CONTROLLER_COMPLETION.json',dict(complete=result['status']=='COMPLETE_NEW',component='unconditional F core banks only',
        final_F_complete=False,source_sha256=U.sha(__file__),producer_summary_sha256=U.sha(output/'SUMMARY.json'),
        no_H1_fit_or_real_scoring_calls=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('campaign','prior-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--GO',action='store_true');p.add_argument('--wait',action='store_true')
    a=p.parse_args();r=run(a.campaign,a.prior_run,GO=a.GO,wait=a.wait)
    print({k:v for k,v in r.items()if k not in ('missing_rows','technical_failures','full_feature_manifests')})
