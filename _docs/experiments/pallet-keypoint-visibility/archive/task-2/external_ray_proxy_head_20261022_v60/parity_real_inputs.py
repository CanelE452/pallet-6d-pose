from common import *
import sys,time
import numpy as np
sys.path.insert(0,str(SRC));sys.path.insert(0,str(V58));from masked_backend import infer_masked
rr=[];start=time.monotonic()
for u in read(ROOT/'PREPARED_INPUTS_REAL.json.gz'):
 p=infer_masked(u['initial_points'],u['K'],u['known_physical_dimensions_xyz_m'],np.ones(9,bool),source=False);old=u['initial_pose'];ok=p['available']==old['available'];gap=None
 if p['available']and old['available']:
  gap=max(float(np.max(np.abs(np.asarray(p[k])-np.asarray(old[k]))))for k in ['R_cf','R_physical','centroid','cf_extents']);ok=ok and gap<=1e-9 and p['selected_hypothesis']==old['selected_hypothesis']
 rr.append(dict(id=u['id'],group=u['group'],PASS=ok,max_pose_component_difference=gap,source_basis=False,finite9=bool(np.isfinite(u['initial_points']).all())))
csvsave('REAL_INPUT_PARITY.csv',rr);save('REAL_INPUT_PARITY.json',dict(UTC=utc(),PASS=all(x['PASS']for x in rr),frames=len(rr),max_component_gap=max(x['max_pose_component_difference']or 0 for x in rr),no_GT_references_read=True,no_real_CAL_loaded=True,no_parameter_fit=True,seconds=time.monotonic()-start));print('REAL_PARITY',all(x['PASS']for x in rr),len(rr),max(x['max_pose_component_difference']or 0 for x in rr),flush=True)
