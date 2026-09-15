"""Complete the first valid balanced timing panel with isolated memory processes."""
import numpy as np
from env import *
def run():
    verify()
    if (DOC/'RUNTIME_PANEL.json').exists():assert read(DOC/'RUNTIME_PANEL.json')['complete'];return
    attempts=sorted((RAW/'runtime').glob('attempt_*.json'))
    eligible=[p for p in attempts if read(p)['status']=='BALANCED_BLOCK_5_COMPLETE' and len(read(p)['records'])==1690]
    if not eligible:
        from runtime_numeric import run as original
        original();return
    path=eligible[0];a=read(path);plan=read(DOC/'RUNTIME_PROTOCOL.json');records=a['records'];names=plan['models']
    expected={(n,k,r) for n in names for k in plan['keys'] for r in range(5)}
    assert len(records)==len(expected)==1690 and {(r['model'],r['key'],r['repeat']) for r in records}==expected
    assert all(np.isfinite(r['end_to_end_ms']) and r['end_to_end_ms']>0 for r in records)
    assert not gpu()['foreign_compute']
    freeze(DOC/'RUNTIME_RECOVERY_LOCK.json',dict(timing_attempt=bound(path),all_attempts=[bound(p) for p in attempts],implementation=bound(HERE/'runtime_recovery.py'),memory_implementation=bound(HERE/'memory_one.py'),reason='All1690 validated latency samples were durable before the model-release memory guard failed. Keep them unchanged; use one new process per model for allocation measurement.',no_fastest_attempt_selection=True,additional_training_updates=0,additional_timing_samples=0))
    isolated={};numerical=[];thread_samples={}
    for n in names:
        subprocess.run([sys.executable,'-B',str(HERE/'memory_one.py'),n],check=True,env=os.environ)
        p=RAW/f'runtime/memory_{n}.json';v=read(p);assert v['complete'];thread_samples[n]=v['threads'];numerical+=v['numerical_replay']
        isolated[n]={k:v[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','scope')};isolated[n]['source']=bound(p)
    stats=lambda x:dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),p90=float(np.quantile(x,.9)))
    base={(r['key'],r['repeat']):r for r in records if r['model']=='R0'};summary={}
    for n in names:
        rr=[r for r in records if r['model']==n]
        summary[n]={f:stats([r[f] for r in rr]) for f in ('image_to_2d_ms','pnp_ms','end_to_end_ms')}
        for label,key in [('2d','image_to_2d_ms'),('e2e','end_to_end_ms')]:summary[n]['paired_added_'+label+'_ms']=stats([r[key]-base[(r['key'],r['repeat'])][key] for r in rr])
    params=a['params']
    for n in params:
        if n!='R0':params[n]['total']=params['R0']['total']+params[n]['refiner_trainable']
    write(DOC/'RUNTIME_PANEL.json',dict(complete=True,status='DESKTOP_ONLY_MEASURED',protocol_sha256=sha(DOC/'RUNTIME_PROTOCOL.json'),raw_attempt=bound(path),recovery=bound(DOC/'RUNTIME_RECOVERY_LOCK.json'),before=a['before'],after=gpu(),threads=dict(torch_intraop=4,torch_interop=None,opencv=None,scope='Torch4 fixed by timed code; inter-op/OpenCV timed-phase samples were not saved before guard failure. Separate observed memory-phase settings below.'),memory_phase_threads=thread_samples,cpu_hardware=json.loads(subprocess.check_output(['lscpu','-J'],text=True)),cpu_hardware_scope='same host queried during immediately following memory recovery',summary=summary,records=records,memory=a['memory'],isolated_memory=isolated,params=params,accuracy_parity=True,accuracy_parity_scope='All1690 timed samples passed unchanged baseline contracts; P/D/L exact and prior corrected corners <=0.0003crop px with zero relative allowance',canonical_MAIN_pose_parity=True,canonical_MAIN_pose_parity_scope='Historical coordinates reproduce canonical metrics; current prior hypothesis and coverage were checked equal, not current floating pose bit-exactness',numeric_amendment=bound(DOC/'RUNTIME_NUMERIC_AMENDMENT.json'),numerical_replay=numerical,numerical_replay_scope='Additional untimed memory phase; timed per-sample numerical bounds passed but individual deltas were not saved',all_models_resident_for_balanced_order=True,memory_scope='one new process per model; excludes external desktop processes',pnp_scope='prediction-only canonical selector plus SQPnP+RefineLM; no GT metric inside timer',limitations='Single desktop GPU, display/RustDesk active; no Jetson; first complete balanced panel retained, initial invalid incomplete attempt preserved; interop/OpenCV thread observations belong to memory phase, not timed phase'))
    print('RUNTIME_PANEL_COMPLETE_REUSED_1690_SAMPLES',flush=True)
