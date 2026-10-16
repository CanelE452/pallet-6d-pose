"""Run the authorised fixed experiment through evaluation and visible delivery."""
from context29 import *
import subprocess

def main():
    stages=[('PRECHECK.json','train29.py',['--phase','prepare']),('TRAINING_COMPLETE.json','train29.py',['--phase','train']),('INFERENCE_COMPLETE.json','eval29.py',['--phase','inference']),('SUMMARY.json','eval29.py',['--phase','score']),('RISK_DIAGNOSIS.json','diagnose29.py',[]),('SUPPORT_DIAGNOSIS.json','support29.py',[]),('AUDIT.json','audit29.py',[]),('REPORT_COMPLETE.json','report29.py',[]),('COMPLETION.json','deliver29.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=RUN,check=True)
    sys.path.insert(0,str(ROOT));from scripts.research.discord_notify import notify
    notify('요청하신 작업이 모두 끝났습니다.',RUN,evidence=RUN/'COMPLETION.json')

if __name__=='__main__':main()
