from context37 import *
import argparse,subprocess
PHASES=dict(analysis=('analyze37.py','ANALYSIS_COMPLETE.json'),audit=('audit37.py','AUDIT.json'),residual=('residual37.py','RESIDUAL.json'),objectives=('objectives37.py','OBJECTIVES.json'),audit_extra=('audit_extra37.py','EXTRA_AUDIT.json'),report=('report37.py','REPORT_COMPLETE.json'),deliver=('deliver37.py','COMPLETION.json'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all',*PHASES],default='all');a=parser.parse_args()
    for name,(script,marker) in PHASES.items():
        if a.phase!='all' and a.phase!=name:continue
        if a.phase=='all' and (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script)],cwd=ROOT,check=True)
if __name__=='__main__':main()
