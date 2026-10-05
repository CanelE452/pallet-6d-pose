"""Start the real, prediction-blind localhost review screen."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'data/pallet/results'/HERE.name
sys.path.insert(0,str(HERE/'review'))
from serve import main

if __name__ == '__main__':
    args = ['serve','--manifest',str(OUT/'review/MANIFEST.json'),
            '--plan',str(OUT/'LIFTER_EVALUATION_PLAN.json'),
            '--contract',str(OUT/'review/CORNER_CONTRACT.json'),
            '--store',str(OUT/'review/annotations_in_progress.json')]
    args += sys.argv[1:]
    raise SystemExit(main(args))
