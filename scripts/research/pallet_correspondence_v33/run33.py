"""Idempotent stage driver using the immutable, completed V32 input cache."""
from context33 import *
import subprocess

def main():
    stages=[('TRAINING_COMPLETE.json','train33.py',[]),('INFERENCE_COMPLETE.json','eval33.py',['--phase','infer']),('SUMMARY.json','eval33.py',['--phase','score']),('TRAIN_DIAGNOSIS.json','eval33.py',['--phase','diagnose']),('AUDIT.json','audit33.py',[]),('POSTHOC_DIAGNOSTICS.json','diagnostics33.py',[]),('REPORT_COMPLETE.json','report33.py',[]),('COMPLETION.json','deliver33.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=ROOT,check=True)
    # The already-authorized completion notification follows artifact verification.

if __name__=='__main__':main()
