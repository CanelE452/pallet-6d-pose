from context41 import *
import argparse,subprocess
PHASES=dict(train=('experiment41.py','TRAINING_COMPLETE.json'),eval=('evaluate41.py','EVALUATION_COMPLETE.json'),audit=('audit41.py','AUDIT.json'),probe=('probe41.py','PROBE.json'),audit_probe=('audit_probe41.py','PROBE_AUDIT.json'),report=('report41.py','REPORT_COMPLETE.json'),deliver=('deliver41.py','COMPLETION.json'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all',*PHASES],default='all');args=parser.parse_args()
    for phase,(script,marker) in PHASES.items():
        if args.phase not in ('all',phase):continue
        if args.phase=='all' and (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script)],cwd=ROOT,check=True)
if __name__=='__main__':main()
