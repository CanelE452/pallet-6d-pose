"""Resume authorised experiment through result delivery without retuning."""
from context27 import *
import subprocess,argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--notify',action='store_true');a=p.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    stages=[('PRECHECK.json','prepare27.py',[]),('TRAINING_COMPLETE.json','train27.py',[]),('INFERENCE_COMPLETE.json','eval27.py',['--phase','inference']),('SUMMARY.json','eval27.py',['--phase','score']),('TRAIN_DIAGNOSIS.json','diagnose27.py',[]),('UTILITY_DIAGNOSIS.json','utility_diagnosis27.py',[]),('AUDIT.json','audit27.py',[]),('REPORT_COMPLETE.json','report27.py',[]),('DELIVERY_CHECK.json','delivery27.py',[]),('COMPLETION.json','complete27.py',[])]
    for marker,script,args in stages:
        if (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script),*args],cwd=RUN,check=True)
    if a.notify:
        sys.path.insert(0,str(ROOT));from scripts.research.discord_notify import notify
        notify('요청하신 작업이 모두 끝났습니다.',RUN,evidence=RUN/'COMPLETION.json')

if __name__=='__main__':main()
