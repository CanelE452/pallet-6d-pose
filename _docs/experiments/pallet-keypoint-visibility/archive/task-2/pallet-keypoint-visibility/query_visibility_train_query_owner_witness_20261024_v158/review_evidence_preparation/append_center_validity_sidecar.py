"""Append exact input validity metadata without changing existing panels/manifests."""
from pathlib import Path
import hashlib,json
import numpy as np
O=Path(__file__).resolve().parent;P=O.parent.parent
def guard(p):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(x in s for x in ('v112','current223','/private','mr6d_native1840_features'))
 assert p.suffix.lower() not in ('.pt','.pth','.ckpt','.safetensors');return p
def sha(p):return hashlib.sha256(guard(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(b):p=guard(b['path']);assert sha(p)==b['sha256'];return p
def read(p):return json.loads(guard(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert not (O/'CENTER4_VALIDITY_INPUT_ONLY_SIDECAR.json').exists()
ready=read(O/'REVIEW_EVIDENCE_READY.json');assert ready['completed'] and ready['cases']==390
source=read(verify(ready['source_bindings']));valids={}
for fr in source['selected_TRAIN_frames']:
 assert fr['split']=='SOURCE_DEVELOPMENT'
 with np.load(verify(fr['crop_packet']),allow_pickle=False) as f:v=np.array(f['valid96'],copy=True)
 assert v.shape==(8,96,96);valids[fr['frame_index']]=v
rows=[];original_files=[]
for group in ('A','B'):
 manifest=read(verify(ready['review_group_'+group]));assert manifest['count']==195
 original_files.append(ready['review_group_'+group])
 for r in manifest['cases']:
  mask=valids[r['frame_index']][r['native_slot'],47:49,47:49];count=int(mask.sum())
  assert bool(count==4)==r['center4_valid']
  original_files.extend(r['images'].values())
  rows.append(dict(case_id=r['case_id'],global_index=r['global_index'],query_id=r['query_id'],review_group=group,
   center4_savedRGB96_indices='[47:49,47:49]',valid2x2_bits=mask.tolist(),valid_count=count,all4valid=count==4,
   required_owner_observation_if_any_invalid='UNKNOWN' if count<4 else None))
assert len(rows)==390 and len({r['query_id'] for r in rows})==390
# Validate unchanged existing artifacts; no re-rendering, panel overlay or role join.
assert all(sha(b['path'])==b['sha256'] for b in original_files)
write(O/'CENTER4_VALIDITY_INPUT_ONLY_SIDECAR.json',dict(schema='V158_BLINDED_INPUT_VALIDITY_SIDECAR',completed=True,count=390,
 original_base_ready=bind(O/'REVIEW_EVIDENCE_READY.json'),code=bind(__file__),
 padding_rule='Any invalid/padded center4 sample requires UNKNOWN. Blackpadding is not naturalBACKGROUND. No case dropped.',
 valid_count_histogram={str(k):sum(r['valid_count']==k for r in rows) for k in range(5)},
 existing_panels_raw_images_and_group_manifests_unchanged=True,original_artifact_hashes_verified=len(original_files),
 no_original_roles_GT_reference_or_model_scores=True,cases=rows))
write(O/'REVIEW_EVIDENCE_READY_WITH_VALIDITY_SIDECAR.json',dict(completed=True,ready_for_fresh_blinded_owner_review=True,cases=390,groups={'A':195,'B':195},
 base_ready=bind(O/'REVIEW_EVIDENCE_READY.json'),center4_validity=bind(O/'CENTER4_VALIDITY_INPUT_ONLY_SIDECAR.json'),
 review_group_A=ready['review_group_A'],review_group_B=ready['review_group_B'],
 blind_crossreview_from_A_by_B=ready['blind_crossreview_from_A_by_B'],blind_crossreview_from_B_by_A=ready['blind_crossreview_from_B_by_A'],
 invalid_center_cases_must_be_UNKNOWN=sum(r['valid_count']<4 for r in rows),source_or_labels_or_qL_or_partition_changed=False))
print(json.dumps(dict(completed=True,ready=bind(O/'REVIEW_EVIDENCE_READY_WITH_VALIDITY_SIDECAR.json'),invalid_center_cases=sum(r['valid_count']<4 for r in rows))),flush=True)
