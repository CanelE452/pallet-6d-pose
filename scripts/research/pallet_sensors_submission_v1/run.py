"""Explicit resumable submission workflow; no historical training entry points."""
import argparse, importlib, traceback
from env import *
STAGES=[('bind','preflight'),('implementation','validate_prior'),('train','train_prior'),('evaluate','evaluate_pipeline'),('confirmation','confirmation'),('manuscript','manuscript'),('audit','finalize'),('publish','publish')]
def status():
    result={}
    for name,module in STAGES:
        p=(RAW if name=='publish' else DOC)/(name.upper()+'_COMPLETE.json')
        r=read(p) if p.exists() else {'complete':False,'status':'NOT_RUN'}
        try:
            if name=='publish':r['hash_verified']='Use git ls-remote to verify current remote; publication receipt records verification time'
            else:r['hash_verified']=complete(name.upper()+'_COMPLETE')
        except AssertionError as e:r=dict(complete=False,status='STALE_RECEIPT',error=str(e))
        result[name]=r
    result['current_training']=[read(p) for p in sorted((RAW/'runs').glob('PRIOR*/PROGRESS.json'))]
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['status','all','resume']+[x[0] for x in STAGES])
    parser.add_argument('--panel',type=Path)
    args=parser.parse_args()
    if args.command=='status': return status()
    chosen=STAGES if args.command in ('all','resume') else [x for x in STAGES if x[0]==args.command]
    failures=[]
    for name,module in chosen:
        try:
            fn=importlib.import_module(module).run
            fn(args.panel) if name=='confirmation' else fn()
        except Exception as e:
            traceback.print_exc();failures.append(name)
            previous=DOC/(name.upper()+'_BLOCKED.json')
            if previous.exists():
                write(RAW/'stage_failures'/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(previous))
            write(DOC/(name.upper()+'_BLOCKED.json'),dict(complete=False,time=now(),error=repr(e),traceback=traceback.format_exc(),command=sys.argv))
            if name=='confirmation':
                write(DOC/'CONFIRMATION_COMPLETE.json',dict(complete=False,status='INVALID_PROVENANCE_OR_EVALUATION_BLOCKED',error=repr(e),checked_at=now(),old_results_not_relabelled=True))
            if args.command not in ('all','resume'): raise
    if failures: print('PARTIAL_TECHNICAL',failures,flush=True)
if __name__=='__main__': main()
