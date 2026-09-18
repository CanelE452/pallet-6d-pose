"""Serial bounded execution. Never touches git or writes historical DCP roots."""
import argparse,subprocess,sys,os,time
import cv_env as E

def main():
    p=argparse.ArgumentParser();p.add_argument('--after-A',action='store_true');p.add_argument('--wait-pid',type=int);args=p.parse_args()
    if args.wait_pid:
        while True:
            try:os.kill(args.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(10)
    stages=[] if args.after_A else ([] if (E.DOC/'PROTOCOL_LOCK.json').exists() else [('prepare.py',[])])+[('inference_manifest.py',[]),('fit.py',['tests']),('fit.py',['smoke']),('fit.py',['A'])]
    stages += [('evaluate.py',['A']),('fit.py',['B']),('evaluate.py',['B']),('evaluate.py',['C']),('code_runtime.py',[]),('code_pose.py',[]),('finish.py',[])]
    root=E.RAW/'logs';root.mkdir(parents=True,exist_ok=True)
    for script,options in stages:
        key=script.removesuffix('.py')+('_'+'_'.join(options) if options else '')
        E.write(E.DOC/'PIPELINE_PROGRESS.json',dict(stage=key,state='RUNNING',time=E.now(),pid=os.getpid()))
        with (root/(key+'.log')).open('a',buffering=1) as log:
            result=subprocess.run([sys.executable,'-B',str(E.HERE/script),*options],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            E.write(E.DOC/'PIPELINE_PROGRESS.json',dict(stage=key,state='FAILED',returncode=result.returncode,time=E.now()));raise SystemExit(result.returncode)
        print('STAGE_COMPLETE',key,flush=True)
    E.write(E.DOC/'PIPELINE_PROGRESS.json',dict(state='COMPLETE',time=E.now()));print('PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
