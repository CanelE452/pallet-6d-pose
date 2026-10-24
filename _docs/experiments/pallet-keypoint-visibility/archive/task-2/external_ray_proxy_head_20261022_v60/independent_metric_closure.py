"""Use same historical trace metric, independent scalar summation; diagnose SO3 projection difference."""
from common import *
import math,numpy as np
from scipy.spatial.transform import Rotation
from evaluate_pose import rotations
gaps=[];so3=[];records=[]
for scope in ['SYNTH','REAL']:
 inputs=read(ROOT/('PREPARED_INPUTS_SYNTH.json.gz'if scope=='SYNTH'else'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz'));by={u['id']:u for u in inputs};pred=read(ROOT/f'{scope}_POSE_PREDICTIONS_SEALED.json.gz');score={(r['id'],r['arm']):r for r in csv.DictReader(open(ROOT/f'{scope}_POSE_PER_FRAME.csv'))}
 if scope=='SYNTH':gt={g['id']:dict(R=g['R_fixed'],t=g['t'],order=g['symmetry_order'])for g in rows(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')}
 else:
  w=read(B/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];s={g['id']:g['reference']for g in read(ROOT.parent/'ocpr_wood_square_transfer_20261022_v50/EVALUATION_FIXED/PORTABLE_POSE_AUDIT_BUNDLE.json.gz')['frames']if g['group']=='SQUARE_YOLO'};gt={}
  for u in inputs:
   g=w[u['partition']][u['id']]if u['group'].startswith('WOOD')else s[u['id'].split(':')[-1]];gt[u['id']]=dict(R=g['R'],t=g['t'],order=u['symmetry_order'])
 for p in pred:
  g=gt[p['id']];G=np.array(g['R']);gg=rotations(g['order'])
  for arm,key in [('N0','original_pose'),('EXTERNAL_RAY_HEAD','final_pose')]:
   P=np.array(p[key]['R_physical']);angles=[]
   for Q in gg:
    A=G@Q;dot=math.fsum(float(A[i,j])*float(P[i,j])for i in range(3)for j in range(3));angles.append(math.degrees(math.acos(max(-1.,min(1.,(dot-1)/2)))))
   R=min(angles);T=100*math.sqrt(math.fsum((a-b)**2 for a,b in zip(p[key]['centroid'],g['t'])));old=score[(p['id'],arm)];gap=max(abs(T-float(old['T_cm'])),abs(R-float(old['R_deg'])));gaps.append(gap);assert(R<5 and T<5)==(old['success']=='True')
   alt=min((Rotation.from_matrix(G@Q).inv()*Rotation.from_matrix(P)).magnitude()*180/math.pi for Q in gg);so3.append(dict(scope=scope,id=p['id'],arm=arm,historical_trace_R=float(old['R_deg']),same_metric_independent_R=R,project_to_SO3_R=float(alt),different_metric_gap=abs(alt-float(old['R_deg'])),reference_orthogonality_max=float(abs(G.T@G-np.eye(3)).max()),prediction_orthogonality_max=float(abs(P.T@P-np.eye(3)).max())))
assert max(gaps)<1e-7
csvsave('INDEPENDENT_ROTATION_NUMERICS.csv',so3);save('INDEPENDENT_METRIC_CLOSURE.json',dict(UTC=utc(),PASS=True,same_metric_max_gap=max(gaps),rows=len(gaps),all_success_counts_identical=True,SO3_projection_metric_max_gap=max(r['different_metric_gap']for r in so3),worst_SO3_projection_difference=max(so3,key=lambda r:r['different_metric_gap']),note='Initial independent SciPy Rotation check projected input matrices onto SO3, unlike retained historical trace metric. Difference is disclosed, scores and gates are unchanged. Independent scalar same-formula audit passes.'))
# Classifier errors on changed success cases: target is proxy, never DIRECT.
qs={}
for q in csv.DictReader(open(ROOT/'SYNTH_QUERY_OUTPUTS.csv')):qs.setdefault(q['id'],[]).append(q)
for p in csv.DictReader(open(ROOT/'SYNTH_POSE_PAIRED.csv')):
 if p['recovery']!='True'and p['damage']!='True':continue
 a=[q for q in qs[p['id']]if q['pixel_valid']=='True'and float(q['blocked_probability'])>=.5];records.append(dict(**p,excluded_proxy_positive=sum(q['target']=='1'for q in a),excluded_proxy_negative=sum(q['target']=='0'for q in a),physical_DIRECT_truth_available=False))
csvsave('SYNTH_CHANGED_CASE_PROXY_ERRORS.csv',records);print('INDEPENDENT_CLOSURE',max(gaps),'SO3_diagnostic_max',max(r['different_metric_gap']for r in so3),flush=True)
