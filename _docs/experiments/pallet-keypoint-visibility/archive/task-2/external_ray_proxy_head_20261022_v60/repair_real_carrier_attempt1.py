"""Recover original saved N3 coordinates, never synthesize the erased S points."""
from common import *
import sys,copy,time
import numpy as np
sys.path.insert(0,str(SRC));sys.path.insert(0,str(V58))
from scripts.research.pallet_visibility_curriculum_20261013_v1.remainder_native_inputs_v1 import select_prediction_fields
from masked_backend import infer_masked
start=time.monotonic();inp=read(ROOT/'PREPARED_INPUTS_REAL.json.gz');ready=read(SRC/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/run_input_readiness_v1/RUN_INPUTS143.json')['rows'];ready={u['id']:u for u in ready};journal=SRC/'_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz';expected={u['upstream_prediction']['sha256']for u in ready.values()};assert expected=={sha(journal)};wanted={u['id']for u in inp if u['partition']=='REMAINDER'};cached={}
with gzip.open(journal,'rt')as f:
 for line in f:
  h=select_prediction_fields(line,{'id','method'})
  if h.get('id')in wanted and h.get('method')=='N3_SUBPIX':cached[h['id']]=select_prediction_fields(line)
assert set(cached)==wanted
out=[];audit=[]
for old in inp:
 u=copy.deepcopy(old);changed=[]
 if u['id']in cached:
  c=cached[u['id']];assert c['raw_hw']==u['raw_hw']and c['qFinal']==c['native_points']and c['actual_pose']==u['initial_pose'];q=np.array(c['qFinal']);oldq=np.array(u['initial_points']);changed=np.flatnonzero(np.any(q!=oldq,axis=1)).tolist();assert 8 not in changed
  assert all(oldq[k].tolist()==[-1.,-1.]for k in changed);u['initial_points']=c['qFinal'];u['input_repair']=dict(reason='historical REMAINDER trace exported initial-S-erased carrier, not original solver input; restore exact saved qFinal',changed_native_ids=changed,source_prediction_journal_sha256=sha(journal),source_fields=['qFinal','native_points','actual_pose'],new_image_forward=0,new_coordinate_regression=0,no_GT_read=True)
 p=infer_masked(u['initial_points'],u['K'],u['known_physical_dimensions_xyz_m'],np.ones(9,bool),source=False);oldp=u['initial_pose'];ok=p['available']==oldp['available'];gap=max(float(np.max(np.abs(np.array(p[k])-np.array(oldp[k]))))for k in ['R_cf','R_physical','centroid','cf_extents'])if p['available']and oldp['available']else None;ok=ok and gap is not None and gap<=1e-9 and p['selected_hypothesis']==oldp['selected_hypothesis'];audit.append(dict(id=u['id'],group=u['group'],partition=u['partition'],PASS=ok,max_component_gap=gap,restored_native_ids=changed));out.append(u)
save('PREPARED_INPUTS_REAL_VERIFIED.json.gz',out);csvsave('REAL_VERIFIED_PARITY.csv',audit);save('REAL_VERIFIED_PARITY.json',dict(UTC=utc(),PASS=all(a['PASS']for a in audit),frames=len(audit),restored_frames=sum(bool(a['restored_native_ids'])for a in audit),max_component_gap=max(a['max_component_gap']or 0 for a in audit),source_journal_sha256=sha(journal),failed_first_attempt_preserved=True,no_reference_or_real_CAL_read=True,seconds=time.monotonic()-start));print('REAL_REPAIRED_PARITY',len(audit),all(a['PASS']for a in audit),max(a['max_component_gap']or 0 for a in audit),flush=True)
