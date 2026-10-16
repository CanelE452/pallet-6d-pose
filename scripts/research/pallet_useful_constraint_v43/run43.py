from context43 import *
import argparse,subprocess
PHASES=dict(train=('experiment43.py','TRAINING_COMPLETE.json'),eval=('evaluate43.py','EVALUATION_COMPLETE.json'),audit=('audit43.py','AUDIT.json'),diagnose=('diagnose43.py','CURVES.json'),audit_diagnostic=('audit_diagnostic43.py','DIAGNOSTIC_AUDIT.json'),report=('report43.py','REPORT_COMPLETE.json'),deliver=('deliver43.py','COMPLETION.json'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all',*PHASES],default='all');args=parser.parse_args();register()
    for phase,(script,marker) in PHASES.items():
        if args.phase not in ('all',phase):continue
        if args.phase=='all' and (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script)],cwd=RUN,check=True)
if __name__=='__main__':main()
