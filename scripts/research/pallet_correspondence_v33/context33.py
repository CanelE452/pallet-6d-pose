from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_visible_geometry_v32'))
import context32 as V32
from context32 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_correspondence_v33'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_correspondence_v33.md'
ARMS=('pooled','query','no_geometry','line_only','end_only','point_aux')
STEPS=2000

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V33 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')
