"""Independent source-GT-only quality-label row join. No feature/model bodies."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent/'head_preparation';C=D/'QUALITY_LABEL_CONTRACT_BEFORE_SOURCE_ERROR_AUDIT.json';R=D/'saved_synthetic_point_quality_v1/RESULTS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
c=read(C);r=read(R);assert r['completed']and r['total_rows']==26624 and r['contract']==bind(C)
allcounts=Counter();cross=defaultdict(Counter);proof=[]
for fam,binding,count in zip(('OLD6144','NEW20480'),c['source_GT'],(6144,20480)):
 gt=[json.loads(s)for s in ck(binding).read_text().splitlines()if s];gt=[x for x in gt if x['convention']=='CF_PRIMARY_DECLARED']
 lb=r['source_families'][fam]['labels'];labels=[json.loads(s)for s in ck(lb).read_text().splitlines()if s];assert len(gt)==len(labels)==count
 cnt=Counter()
 for i,(g,l)in enumerate(zip(gt,labels)):
  assert g['query_id']==l['query_id']and g['native_slot']==l['native_slot']and l['global_index']==i
  assert g['query_xy']==l['original_query_xy']and g['roi_side']==l['original_roi_side']and g['reference_xy']==l['reference_xy_exact_synthetic']
  assert g['teacher_state']==l['original_role_teacher_state_UNCHANGED']and g['split']==l['split']and g['variant']==l['variant']
  assert l['role_visibility_training_mask_UNCHANGED']is True and l['coordinates_corrected']is False
  q,ref,err=g['query_xy'],g['reference_xy'],g['error_px']
  finite=q is not None and ref is not None and err is not None and np.isfinite(q).all()and np.isfinite(ref).all()and math.isfinite(err)
  if finite:assert abs(float(np.linalg.norm(np.asarray(q)-np.asarray(ref)))-err)<1e-9
  eligible=g['nearest_inbounds']and finite and not g['correspondence_boundary']and g['teacher_state']in ('D','E')
  expected=1 if eligible and err<=3 else 0 if eligible and err>20 else None
  assert expected==l['point_quality_label']
  group='accurate1'if expected==1 else'inaccurate0'if expected==0 else'unlabeled';cnt[group]+=1;allcounts[group]+=1
  cross[(fam,g['split'],g['teacher_state'])][group]+=1;cross[('BOTH',g['split'],g['teacher_state'])][group]+=1
 assert dict(cnt)==r['source_families'][fam]['counts']['ALL']
 proof.append(dict(family=fam,source_GT=binding,labels=lb,rows=count,all_ids_q_L_reference_roles_unchanged=True,quality_labels_exact_rule=True,counts=dict(cnt)))
assert allcounts==Counter(accurate1=5245,inaccurate0=260,unlabeled=21119)
assert cross[('BOTH','TRAIN','D')]['inaccurate0']==178 and cross[('BOTH','TRAIN','D')]['accurate1']==3929
assert cross[('BOTH','TRAIN','E')]['inaccurate0']==16 and cross[('BOTH','TRAIN','E')]['accurate1']==160
prototype=D/'separated_query_head.py';checks=read(D/'SEPARATED_HEAD_PROTOTYPE_CHECKS.json');assert checks['code']==bind(prototype)and checks['parameters']==224643
report=dict(completed=True,PASS=True,review_code=bind(__file__),quality_contract=bind(C),quality_dataset=bind(R),prototype=bind(prototype),prototype_generated=bind(D/'SEPARATED_HEAD_PROTOTYPE_CHECKS.json'),source_quality_counts=dict(allcounts),row_join_proof=proof,teacher_conditioned_counts=[dict(family=k[0],split=k[1],teacher_state=k[2],counts=dict(v))for k,v in sorted(cross.items())],
 code_review=dict(parent_visibility_hidden_math_and_state_keys_preserved=True,additional_auxiliary_parameters=130,total_parameters=224643,auxiliary_initial_logits_zero=True,independent_per_output_masks_visibility_not_filtered_by_quality=True,unknown_S_U_middle_rows_not_relabelled=True,public_MR6D_pixel_accuracy_gold_not_created=True,GT_coordinates_not_model_inputs=True,prototype_aux0_gradient_parity_report_verified=True),
 limitations=['This review reads source geometry/label ledgers and code only; no prototype forward, fitted checkpoint, feature or image body is read or evaluated.', 'Quality task is coarse exact-synthetic error classification, not coordinate correction or a deployable GT-error filter.', 'Separate output layers share hidden features. They do not prove statistically independent errors or guarantee multitask transfer.', 'Dataset class/fold/slot/family imbalance requires a prospective sampler/loss/metric protocol; it cannot be resolved by relabelling or choosing thresholds from private outcomes.', 'No output fusion/abstention policy or actual training/adoption is established.'],fitted_model_feature_RGB_private_V112_reads_or_calls=0)
o=H/'SEPARATED_HEAD_AND_QUALITY_REVIEW.json'
with o.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(o),counts=dict(allcounts))))
