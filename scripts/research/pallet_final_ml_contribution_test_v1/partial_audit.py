"""CPU-only clean-stop checkpoint; never substitute for final scientific audit."""
import numpy as np
import torch
from common import *

def run():
    bindings=read(DOC/'PRESERVED_SOURCES.json')['paths']
    for i,(path,digest) in enumerate(bindings.items(),1):
        assert sha(ROOT/path)==digest,path
        if i%500==0:print('PARTIAL_PRESERVATION_CHECK',i,len(bindings),flush=True)
    assert sha(R0)==R0_SHA
    for name in ('B_IMPLEMENTATION_LOCK.json','LINE_SOURCE_BINDING.json'):
        for p,h in read(B/name)['paths'].items():assert sha(ROOT/p)==h
    selected=read(B/'P_SELECTION.json');completed=read(B/'P_TRAINING_COMPLETE.json')
    assert selected['complete'] and completed['complete'] and completed['fits']==3
    seeds={}
    for seed in (1,2,3):
        path=BRAW/f'runs/seed{seed}/last.pt';c=read(B/f'P_SEED{seed}_COMPLETION.json')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==R0_SHA
        assert sha(path)==c['checkpoint_sha256']==selected['checkpoints'][str(seed)]
        assert all(torch.isfinite(t).all() for t in ck['model_state_dict'].values())
        order=np.load(BRAW/f'order_seed{seed}.npy');h=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        assert h==ck['batch_trace_sha256']==c['batch_order_sha256']==read(B/'INIT_ORDER_PARITY.json')['seeds'][str(seed)]['batch_order_sha256']
        seeds[str(seed)]=dict(complete=True,step=6000,exposures=96000,checkpoint_sha256=sha(path),order_exact=True)
    assert read(B/'SYNTH_HELDOUT_RESULTS.json')['selection_sha256']==sha(B/'P_SELECTION.json')
    assert not (B/'VERDICT.json').exists() and not (B/'FINAL_AUDIT.json').exists()
    assert subprocess.check_output(['git','diff','--name-only'],text=True).strip()==''
    write(B/'TRAINING_AUDIT.json',dict(PASS=True,fits=3,seeds=seeds,total_updates=18000,real_training=0,L_training=0,R0_training=0,
        batch=16,steps_each=6000,optimizer_exact=True,AMP=False,final_checkpoint_only=True,protected_R0_SHA=sha(R0)))
    write(DOC/'PARTIAL_INTEGRITY_AUDIT.json',dict(PASS=True,scope='Completed A/training/synthetic stages only, not a final scientific/performance audit',
        historical_files_verified=len(bindings),R0_SHA=sha(R0),training_source_exact=True,three_fits_complete=True,
        seed_order_exact=True,synthetic_selection_frozen=True,paper_final_untouched=True,old_task_risk_verdict_unchanged=True,
        real_inference_not_started=True,final_verdict_available=False))
    busy=read(DOC/'GPU_BUSY_20260913T102751Z.json')
    write(DOC/'CLEAN_STOP_STATUS.json',dict(status='CLEAN_STOP_FOREIGN_GPU_COMPUTE',complete=False,
        timestamp=datetime.now(timezone.utc).isoformat(),gpu_busy=busy,
        completed=['A nontraining QA/influence/catastrophe/padding-instance audit','35 preregistration contracts;39 pipeline tests',
            'P seeds1/2/3 each6000 updates','synthetic-only T/lambda/cap selection freeze','synthetic heldout comparison'],
        pending=['P actual DEV319+negative2689 inference','canonical real scoring','same-seed paired10000 bootstrap',
            'normal/tangent mechanism','matched runtime','final audit and verdict/reports','commit/push and local-remote SHA verification'],
        scientific_verdict=None,no_foreign_process_killed=True,no_driver_or_system_changes=True,no_CPU_inference_fallback=True,
        no_GPU_poll_loop=True,resume_phase='evaluate_point.py infer',retraining_needed=False,
        current_local_SHA=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        last_fetched_origin_main_SHA=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),
        push_performed_this_task=False,reason_no_push='Final requested workflow is incomplete; preserve work locally without representing it as completed.'))
    print('CLEAN_STOP_SAVED; completed work preserved; no final verdict or push claimed',flush=True)

if __name__=='__main__':run()
