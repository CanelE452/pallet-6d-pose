from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_line_validation_v34'))
import context34 as V34
from context34 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_role_offset_v35'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_role_offset_v35.md'
from plpose_local.geometry import EDGES

def register():
    RUN.mkdir(exist_ok=True,parents=True)
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(scope='V32 existing rendered600 observations / 24 appearance groups / 2 shared CAD assets. Train16groups, check8groups; detected-only diagnostic with misses explicit. No real data accessed, no new neural training, no deployable architecture claim.',selection='Per role longest visible CAD-chain support, >=8px total adjacent visible length and >=3samples inside image and ROI. Per point select longest available incident-role support. Selection uses GT line visibility/role, never point error; therefore GT-correspondence diagnostic. Predicted top1 and oracle best-of4 per matched role separate geometry/candidate losses.',offset='For unit normal n and line rho: desired normal correction n.(GT-base)=(rho-n.base)+(n.GT-rho). Physical line bias delta is nonzero for virtual cuboid corners. Fit delta/boxdiag on train only: per(point,role) median and ridge fixed lambda0.001, features normal x/y, quadratic terms, log box aspect; no point coordinates, rho or target-view camera pose as calibration inputs. No threshold selection on check.',controls='Base/strong/V34; GT physical zero/median/ridge offsets; predicted top1 same correspondence zero/median/ridge; oracle top4 geometry matching then ridge; virtual GT line identity control; exact offset single-normal and incident normals SVD observable ceilings. Fully GT-free top-confidence incident-role prediction+train offset also scored separately.',geometry='Canonical normal ny>=0. Single-line correction preserves tangent. Joint exact-offset ceiling uses SVD relative threshold0.1 and preserves unobservable residual. Oracle offsets/virtual lines/GT role selection are diagnostics and never reported as usable inference.',gates='Matched physical ridge must reduce normal residual MAE >=10% vs zero offset on check, paired source-group95%CI upper<0; predicted matched-ridge must improve point Euclidean error >=1% for each3seed and groupCI upper<0 on same supported check points to pass transfer. GT-free predicted score+ridgetransfer evaluated independently, >=1% eachseed and groupCIupper<0. Conditional gains are not overall stability.',limits='Small reused renderer check, same assets, finite depth visibility/rule-defined roles. Training residuals come from models already trained on these render train groups. Counterfactual/oracle checks diagnose available information; no new real generalization, no manual GT validation or edits.'))
    files={}
    for p,h in read(V34.RUN/'COMPLETION.json')['files'].items():files[p]=h
    for p,h in read(V32.RUN/'INPUT_BINDINGS.json')['files'].items():files[p]=h
    files[str((V34.RUN/'COMPLETION.json').relative_to(ROOT))]=sha(V34.RUN/'COMPLETION.json')
    write(RUN/'PRIOR_BINDINGS.json',dict(files=files))
