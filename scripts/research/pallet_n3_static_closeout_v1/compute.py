"""Non-lifter evaluation only. No training, selection, old-output writes or network."""
from pathlib import Path
import json, time, hashlib
import numpy as np
from scripts.research.pallet_n3_completion_v3 import common as C, reuse as R, metrics as M

ROOT=C.ROOT
DOC=ROOT/'_docs/experiments/pallet_n3_static_closeout_v1'
RAW=ROOT/'data/pallet/results/pallet_n3_static_closeout_v1'
SOURCE=Path('/home/minjae/Documents/github/pallet-pose')
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bind(p): return {'path':str(p),'sha256':C.sha256(p),'bytes':p.stat().st_size}

def main():
    started=time.time()
    if not (DOC/'CORE_RESULTS.json').exists():
        core,state=R.build_core(SOURCE,include_pose=True)
        old=C.read(C.RAW/'reuse/PER_FRAME_SCORES.json')
        checks={}
        for method,result in state['evaluated'].items():
            for key,oldkey in [('corner_rows','corner_scores'),('pose_rows','pose_scores')]:
                if key in result:
                    checks[method+':'+key]=clean(result[key])==clean(old['core'][method][oldkey])
        assert all(checks.values()),checks
        print('Regression passed',len(checks),flush=True)
        for arm in ['N0_BASE_REPLAY','N1_SYM_ONLY']:
            for seed in (1,2,3):
                name=f'{arm}_seed{seed}'
                state['evaluated'][name]=M.evaluate_method(state['rows'],name,include_pose=True)
                print('Missing pose computed',name,flush=True)
        results=state['evaluated']
        write(RAW/'YOLO_SCORES.json',{k:{'corner_scores':v['corner_rows'],'pose_scores':v['pose_rows']} for k,v in results.items()})
        panels={'R0':R._single_panel(results['R0'])}
        for arm in ('OLD_P',)+R.DCP_ARMS: panels[arm]=R._seed_panel(results,arm)
        write(DOC/'CORE_RESULTS.json',{'methods':panels,'regression':checks,'sources':core['sources'],
            'missing_cause':'reuse.py CORE_ARMS excludes N0_BASE_REPLAY/N1_SYM_ONLY; original preserved',
            'seconds':time.time()-started,'training_runs':0,'optimizer_updates':0})
    # Recompute aggregates directly from already verified comparator and student scores.
    old=C.read(C.RAW/'reuse/PER_FRAME_SCORES.json')
    reused=C.read(C.DOC/'REUSE_RESULTS.json')
    reused_checks={}
    for section,contract in [('comparators','H'),('students_safe128','I')]:
        for name,scores in old[section].items():
            corner=M._corner_summary(scores['corner_scores']); pose=M._pose_summary(scores['pose_scores'])
            if section=='comparators':
                family=name.split('_')[1]; seed=name.rsplit('seed',1)[1]
                expected=reused['contracts'][contract]['methods'][family]['per_seed'][seed]
            else:
                mapping={'R0':'R0','SYN_LR5':'source_only_update','RAW_LR5':'raw_pseudo_student','REF_LR5':'corrected_pseudo_student'}
                panel=reused['contracts']['I']['safe_common_cohort']['methods'][mapping.get(name,'R0_plus_N3')]
                expected=panel['per_seed'][name.rsplit('seed',1)[1]] if name.startswith('N3') else panel['result']
            reused_checks[name]={'corner':clean(corner)==clean(expected['corner']),'pose':clean(pose)==clean(expected['pose'])}
    write(DOC/'REUSED_COMPARATOR_STUDENT_CHECKS.json',reused_checks)
    assert all(all(v.values()) for v in reused_checks.values()),reused_checks
    print('Completed',time.time()-started,flush=True)

if __name__=='__main__':main()
