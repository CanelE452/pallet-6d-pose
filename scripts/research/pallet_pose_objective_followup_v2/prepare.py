"""Record non-mutating repository, old-asset and bounded-resource locks."""
from pathlib import Path
import subprocess
import time
from . import common as C

def main():
    if (C.DOC/'INPUT_BINDINGS.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']: C.verify(b)
        print('EXISTING_INPUT_LOCK_VERIFIED'); return
    start = time.time()
    request = Path('/home/minjae/Downloads/pallet_pose_objective_autonomy_goal_plan_cli.txt')
    paths = [C.OLD.DOC / name for name in ('INPUT_BINDINGS.json','REPORT_KO.md','PRIOR_ATTEMPTS.md',
        'REUSE_AND_RETRY_DECISIONS.json','SOURCE_REGISTRY.json','ROOT_CAUSE_MAP.md','ORACLE_POSE_RESULTS.json',
        'ORACLE_COORDINATE_RESULTS.json','TRAIN_TARGET_TRANSFER.json','FINAL_DECISION.json','RESOURCE_LEDGER.json',
        'cycles/C2_REAL_AFFINE_OFF/RESULTS.json','cycles/C3_MANUAL38_CAPABILITY/RESULTS.json')]
    paths += sorted(p for p in (C.ROOT/'_docs/paper/selftraining_submission_v1').rglob('*') if p.is_file())
    old = C.read(C.OLD.DOC/'INPUT_BINDINGS.json')
    def collect(obj):
        if isinstance(obj,dict):
            if 'path' in obj and 'sha256' in obj:
                p=C.ROOT/obj['path']
                if p.is_file(): C.verify(obj); paths.append(p)
            else:
                for v in obj.values(): collect(v)
        elif isinstance(obj,list):
            for v in obj: collect(v)
    collect(old)
    status=subprocess.check_output(['git','status','--short','--branch'],text=True)
    C.save(C.RAW/'GIT_START_STATUS.txt',status,True)
    git={k:subprocess.check_output(v,text=True).strip() for k,v in {
        'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],
        'remote_main':['git','rev-parse','origin/main'],'recent':['git','log','-8','--oneline']}.items()}
    assert git['HEAD']==git['remote_main'] and git['branch']=='main'
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(created_utc=C.now(),git=git,request_sha256=C.sha(request),
        request_lines=760,files=[C.bind(p) for p in sorted(set(paths))],
        existing_untracked_lines=sum(x.startswith('??') for x in status.splitlines()),
        original_assets_preserved=True,old_experiment_budget_not_reused=True),True)
    C.save(C.DOC/'RESOURCE_LEDGER.json',dict(start_unix=start,created_utc=C.now(),
        caps=dict(main_hypothesis_cycles=3,fits=12,optimizer_updates=7680,GPU_seconds=21600,wall_seconds=36000),
        reservations=[dict(role='A_PLASTIC_paired',fits=2,updates=640),
                      dict(role='diagnostic_supported_B_or_C_or_EF_rebudget_before_fit',fits=2,updates=640),
                      dict(role='two_principle_combination_if_justified',fits=2,updates=640),
                      dict(role='WOOD_selected_recipe_applicability',fits=2,updates=640),
                      dict(role='reserved_seed_repeat_recipe_RAW_REF_and_baseline_RAW_REF',fits=4,updates=1280)],
        reserved_fits=12,reserved_updates=3840,
        reservation_note='Allocation is a cap, not a promise to run. Refiner route must first replace reservations with actual teacher cost plus controls.',
        events=[],totals=dict(fits=0,optimizer_updates=0,gpu_seconds=0,elapsed_wall_seconds=0)),True)
    C.state('DIAGNOSIS_AND_PREFLIGHT',['request_read','git_remote_checked','old_asset_bindings'],
            'Freeze T/R metrics; criterion diagnosis; A paired random-occlusion feasibility')
    print('PREPARED',git['HEAD'],len(set(paths)),flush=True)

if __name__=='__main__': main()
