"""Read-only post-export preservation and TRAIN-normalizer verification."""
from pathlib import Path
import sys,json,gzip,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent;T=R.parent;SRC=Path('${SOURCE_REPO}');sys.path.insert(0,str(SRC))
from scripts.research.pallet_boundary_action_repair_v1 import action_data as A
def rows(p):
 with gzip.open(p,'rt')as h:return [json.loads(s)for s in h]
def rd(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
start=time.monotonic();U=R/'SYNTHETIC_v2';seal=rd(U/'SEAL.json');assert sha(U/'HEAD_FEATURES.jsonl.gz')==seal['sha256'];old=rows(T/'jackknife_bank_20261021_v24/SYNTHETIC_v3/HEAD_FEATURES.jsonl.gz');new=rows(U/'HEAD_FEATURES.jsonl.gz');assert len(new)==len(old)==1003;preserved=0;counts={}
for f,n in zip(old,new):
 assert f['id']==n['id']and f['split']==n['split'];assert [c['candidate_id']for c in n['candidates'][:len(f['candidates'])]]==[c['candidate_id']for c in f['candidates']]
 for a,b in zip(f['candidates'],n['candidates']):
  assert a['actual_pose']==b['actual_pose']and a['eligible']==b['eligible']and a['features'][:26]==b['features'][:26]and a['features'][31:57]==b['features'][31:57];preserved+=1
 A.validate_frame(n);s=counts.setdefault(f['split'],dict(frames=0,old_raw_candidates=0,new_raw_candidates=0,old_eligible_actions=0,new_eligible_actions=0,minimum_actions=None,maximum_actions=0));s['frames']+=1;s['old_raw_candidates']+=len(f['candidates']);s['new_raw_candidates']+=len(n['candidates']);s['old_eligible_actions']+=len(A.actions(f));k=len(A.actions(n));s['new_eligible_actions']+=k;s['minimum_actions']=k if s['minimum_actions']is None else min(s['minimum_actions'],k);s['maximum_actions']=max(s['maximum_actions'],k)
norm=A.fit_normalizer([f for f in new if f['split']=='train']);assert norm==rd(U/'NORMALIZER.json');assert counts['train']['frames']==768 and counts['validation']['frames']==235
out=dict(status='PASS',frames=1003,preserved_raw_candidates=preserved,old_candidate_order_pose_eligibility_local26_and_validity_bitwise_preserved=True,scoregap_and_bank_context_intentionally_changed=True,own_TRAIN768_normalizer_reproduction_exact=True,counts=counts,feature_sha256=sha(U/'HEAD_FEATURES.jsonl.gz'),normalizer_sha256=sha(U/'NORMALIZER.json'),seconds=time.monotonic()-start,new_training_forward_RGB_PnP=0,limitation='preservation and normalization check, not full independently recomputed observation exporter equivalence or independent physicalGT')
with(R/'FEATURE_PRESERVATION_VERIFICATION.json').open('x')as h:json.dump(out,h,indent=2);h.write('\n')
print(json.dumps(out,indent=2))
