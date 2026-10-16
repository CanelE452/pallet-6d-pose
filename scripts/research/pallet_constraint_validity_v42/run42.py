from context42 import *
import argparse,subprocess
PHASES=dict(train=('experiment42.py','TRAINING_COMPLETE.json'),eval=('evaluate42.py','EVALUATION_COMPLETE.json'),audit=('audit42.py','AUDIT.json'),probe=('probe42.py','PROBE.json'),audit_probe=('audit_probe42.py','PROBE_AUDIT.json'),report=('report42.py','REPORT_COMPLETE.json'),deliver=('deliver42.py','COMPLETION.json'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all',*PHASES],default='all');args=parser.parse_args()
    for phase,(script,marker) in PHASES.items():
        if args.phase not in ('all',phase):continue
        if args.phase=='all' and (RUN/marker).exists():continue
        subprocess.run([sys.executable,str(KIT/script)],cwd=ROOT,check=True)
if __name__=='__main__':main()
