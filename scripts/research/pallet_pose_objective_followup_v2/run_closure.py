"""Sequential GPU owner for the six pre-reserved confirmation fits only."""
import subprocess
import sys
from . import common as C

def main():
    lock=C.read(C.DOC/'FINAL_SELECTION.json')
    assert lock['additional_fits']==6 and not lock['further_method_search']
    logdir=C.RAW/'closure_logs';logdir.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for cycle,material,seed,module in [('BASELINE_REPEAT','PLASTIC',43,'student'),
                                     ('RECIPE_REPEAT','PLASTIC',43,'exposure_student'),
                                     ('WOOD_APPLICABILITY','WOOD',42,'exposure_student')]:
        if material=='WOOD':
            jobs += [('WOOD_A_PREFLIGHT','student',['preflight','--material',material]),
                     ('WOOD_C_PREFLIGHT','exposure_student',['preflight','--material',material])]
        for target in ('RAW','REF'):
            jobs.append((f'{cycle}_{target}',module,['train','--cycle',cycle,'--material',material,'--target',target,'--seed',str(seed)]))
        for phase,mod in [('parity','student'),('infer','eval_student'),('score','eval_student')]:
            jobs.append((f'{cycle}_{phase}',mod,[phase,'--cycle',cycle,'--material',material,'--seed',str(seed)]))
        jobs.append((f'{cycle}_ledger','sync_ledger',[]))
    completed=[]
    for name,module,args in jobs:
        print('CLOSURE_STAGE_START',name,flush=True)
        C.state('CONTROLLED_REPEAT_AND_WOOD',completed,' '.join([module,*args]))
        log=logdir/f'{name}.log'
        # Preserve failed logs, while completed expensive modules verify/reuse artifacts.
        if log.exists():log=logdir/f'{name}_{C.now().replace(":","-")}.log'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','scripts.research.pallet_pose_objective_followup_v2.'+module,*args],
                cwd=C.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            C.save(C.RAW/'CLOSURE_FAILURE.json',dict(stage=name,returncode=result.returncode,log=C.bind(log),completed=completed))
            raise RuntimeError(f'{name} failed; artifacts/logs preserved at {log}')
        completed.append(name); print('CLOSURE_STAGE_DONE',name,flush=True)
    C.state('REPORT_AND_AUDIT',completed,'Generate final reports/figures, run all tests and independent audit, commit/push')
    print('ALL_PREDECLARED_FITS_AND_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
