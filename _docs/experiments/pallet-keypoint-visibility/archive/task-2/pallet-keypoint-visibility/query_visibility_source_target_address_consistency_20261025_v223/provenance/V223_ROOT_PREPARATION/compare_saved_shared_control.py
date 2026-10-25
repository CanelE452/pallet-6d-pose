"""Reproduce the saved-output comparison used for SOURCE_SHARED_CONTROL_REPLAY.

No model/array/image/GT access, no output mutation. This is not a private benchmark.
"""
import json
from pathlib import Path
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
A=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222/paired_source_fit_v1/C67PLUS_EXPECTED_PART_CF_WEAK'
B=P/'query_visibility_source_target_address_consistency_20261025_v223/paired_source_fit_v1/C68SHARED_POINT_PART_LOGIT'
def compare():
 out=[]
 for name,key,pred in [('CF_ALL144_SOURCE_OUTPUTS.json','rows144','prediction_D0_E1_Uminus1'),('WEAK57_SOURCE_OUTPUTS.json','rows57','prediction_D0_E1'),('STRONG_POINT_TRAIN_OUTPUTS.json','rows','prediction_D0_E1')]:
  aa=json.loads((A/name).read_bytes())[key];bb=json.loads((B/name).read_bytes())[key]
  assert len(aa)==len(bb) and all(x['row_id']==y['row_id'] and x['target']==y['target'] for x,y in zip(aa,bb))
  delta=[abs(x['raw_logit']-y['raw_logit']) for x,y in zip(aa,bb)]
  out.append({'family':name,'rows':len(aa),'max_absolute_logit_difference':max(delta),'logits_exact':all(d==0 for d in delta),'prediction_changes':sum(x[pred]!=y[pred] for x,y in zip(aa,bb))})
 return out
if __name__=='__main__':print(json.dumps(compare(),indent=2))
