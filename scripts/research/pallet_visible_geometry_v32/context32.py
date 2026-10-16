from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_separated_gain_v31'))
import context31 as OLD31
from context31 import *
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_camera_trust_v26'))
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_visible_geometry_v32'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_visible_geometry_v32.md'
ARMS=('point','latent_hough','virtual_hough','visible_hough')
STEPS=2000
BATCH=16

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V32 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')
