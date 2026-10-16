from context38 import *
import argparse,subprocess
PHASES=dict(train=('experiment38.py','TRAINING_COMPLETE.json'),eval=('evaluate38.py','EVALUATION_COMPLETE.json'),audit=('audit38.py','AUDIT.json'),probe=('probe38.py','PROBES.json'),audit_probe=('audit_probe38.py','PROBE_AUDIT.json'),report=('report38.py','REPORT_COMPLETE.json'),deliver=('deliver38.py','COMPLETION.json'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all',*PHASES],default='all');a=p.parse_args()
    for name,(script,marker) in PHASES.items():
        if a.phase!='all' and a.phase!=name:continue
        if a.phase=='all' and (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script)],cwd=ROOT,check=True)
if __name__=='__main__':main()
