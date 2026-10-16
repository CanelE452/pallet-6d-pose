from context34 import *
import subprocess

def main():
    stages=[('TRAINING_COMPLETE.json','train34.py',[]),('NUMERIC_REPLAY.json','replay34.py',[]),('DENSE_COMPLETE.json','train_dense34.py',[]),('INFERENCE_COMPLETE.json','eval34.py',['--phase','infer']),('SUMMARY.json','eval34.py',['--phase','score']),('CF_INFERENCE_COMPLETE.json','probe34.py',['--phase','infer']),('COUNTERFACTUAL.json','probe34.py',['--phase','score']),('AUDIT.json','audit34.py',[]),('REPORT_COMPLETE.json','report34.py',[]),('COMPLETION.json','deliver34.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=ROOT,check=True)

if __name__=='__main__':main()
