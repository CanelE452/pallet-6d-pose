from context36 import *
import subprocess

def main():
    stages=[('CACHE_COMPLETE.json','cache36.py',[]),('TRAINING_COMPLETE.json','train36.py',[]),('INFERENCE_COMPLETE.json','eval36.py',['--phase','infer']),('POINT_SUMMARY.json','eval36.py',['--phase','score']),('LINE_SUMMARY.json','lines36.py',[]),('PROBES_COMPLETE.json','probes36.py',[]),('ADAPT_TRAINING_COMPLETE.json','adapt36.py',['--phase','train']),('ADAPT_INFERENCE_COMPLETE.json','adapt36.py',['--phase','eval']),('AUDIT.json','audit36.py',[]),('REPORT_COMPLETE.json','report36.py',[]),('COMPLETION.json','deliver36.py',[])]
    for marker,script,args in stages:
        if not (RUN/marker).exists():subprocess.run([sys.executable,str(KIT/script),*args],cwd=ROOT,check=True)
if __name__=='__main__':main()
