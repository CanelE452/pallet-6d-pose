"""Sequential resource-checked C fits; no process killing, retries or tuning."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,write
from common.gpu_snapshot import snapshot

def main():
    if (DOC/'EXECUTION_BLOCKER.json').exists():
        authorization=DOC/'C_geometry_preserving_da/RESUME_AUTHORIZATION.json'
        if not authorization.exists():
            raise SystemExit('Budget direction required: consumed C2 seed1 fit has no checkpoint. No automatic refit.')
        approved=json.loads(authorization.read_text())
        assert approved['additional_repair_updates']==900
        assert approved['remaining_fits']==['C2_seed1','C0_seed2','C1_seed2','C2_seed2','C0_seed3','C1_seed3','C2_seed3']
    here=Path(__file__).resolve().parent
    for seed in [1,2,3]:
        for arm in ['C0','C1','C2']:
            name=f'{arm}_seed{seed}'
            out=RAW/'C_geometry_preserving_da'/name
            if (out/'TRAINING_AUDIT.json').exists() and (out/'evaluation_pose/EVALUATOR_BINDINGS.json').exists():continue
            snapshot()
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            if gpu:
                write(DOC/'C_geometry_preserving_da/RESOURCE_UNAVAILABLE.json',dict(status='NOT_RUN',next_fit=name,compute_processes=gpu,action='No wait or process mutation'))
                return 2
            out=RAW/'C_geometry_preserving_da'/name;out.mkdir(parents=True,exist_ok=True)
            for script in ['train.py','evaluate.py']:
                done=out/('TRAINING_AUDIT.json' if script=='train.py' else 'evaluation_pose/EVALUATOR_BINDINGS.json')
                if done.exists():continue
                log_path=out/(script+'.log')
                if log_path.exists():log_path=out/(script+'.authorized_resume.log')
                with log_path.open('x') as log:
                    result=subprocess.run([sys.executable,str(here/script),'--arm',arm,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                print(name,script,'exit',result.returncode,flush=True)
                if result.returncode:return result.returncode
    return 0

if __name__=='__main__':raise SystemExit(main())
