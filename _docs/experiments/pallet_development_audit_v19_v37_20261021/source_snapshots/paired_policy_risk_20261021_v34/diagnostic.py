"""Label-perfect safety-target diagnostic only. No executable inference policy."""
from pathlib import Path
import csv,json,sys,hashlib
R=Path(__file__).resolve().parent;T=R.parent;sys.path[:0]=[str(T/'minimal_improvement_20261020_v5'),str(T/'geometric_influence_20261021_v19')]
from kernel_probe import stats
from experiment import gate
p=T/'roi_box_joint_evidence_20261021_v33/SYNTHETIC_VAL_s20261018/PAIRED_ROWS.csv';assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((R/'PROTOCOL.json').read_text())['existing_paired_sha256'];rr=list(csv.DictReader(p.open()));out=[];selected=[]
for r in rr:
 a='JOINT_EVIDENCE'if float(r['JOINT_EVIDENCE_translation_cm'])<=float(r['JK_CONTROL_translation_cm'])+1e-7 and float(r['JOINT_EVIDENCE_rotation_deg'])<=float(r['JK_CONTROL_rotation_deg'])+1e-7 else'JK_CONTROL';selected.append(dict(pose_available=True,translation_cm=float(r[a+'_translation_cm']),rotation_deg=float(r[a+'_rotation_deg']),success=r[a+'_success']=='True'));out.append(dict(id=r['id'],diagnostic_label_selected_method=a,uses_reference_DO_NOT_DEPLOY=True))
s=stats(selected);old=json.loads((T/'roi_box_joint_evidence_20261021_v33/SYNTHETIC_VAL_s20261018/EVALUATION.json').read_text());why={a:gate(s,b)for a,b in old['methods'].items()};ok=not any(why.values());result=dict(status='LABEL_TARGET_FEASIBLE_PROCEED_SEALED_TRAIN_OUTPUTS'if ok else'BLOCKED_LABEL_PERFECT_POLICY_DOES_NOT_MEET_FIXED_GATE',diagnostic_only=True,GT_selected_NOT_METHOD=True,not_general_oracle_bound=True,metrics=s,gate_reasons=why,firstcheck_passed=ok,new_fits=0,new_forward_PnP=0,no_gate_relaxation=True,action='conditional paired-risk fit allowed only if PASS; otherwise target rejected without fit')
(R/'LABEL_PERFECT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
with(R/'LABEL_PERFECT_DIAGNOSTIC_ROWS_NOT_INFERENCE.csv').open('x',newline='')as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(result,indent=2))
