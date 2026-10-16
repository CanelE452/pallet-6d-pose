import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_visible_geometry_v32'))
import prepare32
from context44 import RUN
prepare32.RUN=RUN
if __name__=='__main__':prepare32.main()
