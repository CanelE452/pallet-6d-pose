from context39 import *
import argparse, subprocess
PHASES = dict(train=('experiment39.py','TRAINING_COMPLETE.json'), eval=('evaluate39.py','EVALUATION_COMPLETE.json'), audit=('audit39.py','AUDIT.json'), phenotypes=('phenotypes39.py','PHENOTYPES.json'), report=('report39.py','REPORT_COMPLETE.json'), deliver=('deliver39.py','COMPLETION.json'))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', *PHASES], default='all')
    args = parser.parse_args()
    for name, (script, marker) in PHASES.items():
        if args.phase not in ('all', name): continue
        if args.phase == 'all' and (RUN/marker).exists(): continue
        subprocess.run([sys.executable, str(KIT/script)], cwd=ROOT, check=True)
if __name__ == '__main__': main()
