"""Isolated V31; all completed parents are read-only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_directional_adaptive_v30'))
import context30 as OLD30
from context30 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_separated_gain_v31'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_separated_gain_v31.md'
DATA=RUN/'fresh'
ARMS=('joint_old','joint_new','split_old','split_new','generic_new')
STEPS=2000
BATCH=16

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V31 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')
