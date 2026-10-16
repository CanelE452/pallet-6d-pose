"""V25 observation intervention and direction-capable correction experiments."""
from pathlib import Path
import sys
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT.parent / 'pallet_quality_v23'))
import context23 as PREV
from context23 import *
KIT = Path(__file__).resolve().parent
RUN = ROOT / 'data/pallet/results/pallet_observation_v25'
NOTE = ROOT / '_docs/notes/point_line_fusion/04_failure_analysis/pallet_observation_v25.md'

def record(message):
    p = ROOT / '_docs/history/2026-09-10.md'
    with p.open('a') as f:
        f.write('\n\n### V25 ' + time.strftime('%H:%M:%S') + '\n\n' + message + '\n')

