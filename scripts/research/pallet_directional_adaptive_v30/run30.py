"""Resume the authorized fixed experiment through scoring, audit and delivery."""
from context30 import *
import subprocess

def main():
    stages=[('PRECHECK.json','train30.py',['--phase','prepare']),('TRAINING_COMPLETE.json','train30.py',['--phase','train']),('INFERENCE_COMPLETE.json','eval30.py',['--phase','inference']),('SUMMARY.json','eval30.py',['--phase','score']),('GAIN_DIAGNOSIS.json','diagnose30.py',[]),('AUDIT.json','audit30.py',[]),('REPORT_COMPLETE.json','report30.py',[]),('COMPLETION.json','deliver30.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=RUN,check=True)
    sys.path.insert(0,str(ROOT));from scripts.research.discord_notify import notify
    notify('요청하신 작업이 모두 끝났습니다.',RUN,evidence=RUN/'COMPLETION.json')

if __name__=='__main__':main()
