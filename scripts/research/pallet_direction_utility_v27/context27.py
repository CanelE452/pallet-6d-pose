"""Isolated V27 run; prior completed experiments stay immutable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_observation_v25'))
import context25 as PREV25
from context25 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_direction_utility_v27'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_direction_utility_v27.md'
ARMS=('point_utility','line_distance','line_utility','line_role_utility')
def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V27 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')
