from context31 import *
import subprocess

def main():
    stages=[('DATA_COMPLETE.json','prepare31.py',[]),('TRAINING_COMPLETE.json','train31.py',[]),('INFERENCE_COMPLETE.json','eval31.py',['--phase','infer']),('SUMMARY.json','eval31.py',['--phase','score']),('TRAIN_DIAGNOSIS.json','eval31.py',['--phase','diagnose']),('AUDIT.json','audit31.py',[]),('REPORT_COMPLETE.json','report31.py',[]),('COMPLETION.json','deliver31.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=ROOT,check=True)
    sys.path.insert(0,str(ROOT));from scripts.research.discord_notify import notify
    notify('요청하신 작업이 모두 끝났습니다.',RUN,evidence=RUN/'COMPLETION.json')

if __name__=='__main__':main()
