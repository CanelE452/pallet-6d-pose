"""Projective-reference position within original query crops, never new gold."""
from pathlib import Path
from collections import defaultdict
import csv, datetime, hashlib, json, math

V=Path(__file__).resolve().parent;P=V.parent
Q=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl'
R=P/'query_visibility_mr6d_full_public_source_20261024_v153/native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'
CODE=P/'query_visibility_mr6d_full_public_source_20261024_v153/initializer_feature_preparation/review_full230_saved_features.py'

def bind(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()and'private_evaluation'not in str(p)
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def main():
 q=lines(Q);rr=lines(R);ref={(r['frame_index'],r['gt_instance_index'],r['native_slot']):r for r in rr}
 rows=[];groups=defaultdict(list);occ=defaultdict(list)
 for r in q:
  if not(r['original_input_valid']and r['role_target']>=0):continue
  xy=ref[(r['frame_index'],r['selected_gt_instance_index'],r['native_slot'])]['reference_xy']
  u=r['query_xy'];L=r['L'];dx=xy[0]-u[0];dy=xy[1]-u[1];c=max(abs(dx),abs(dy))/L
  a=dict(global_index=r['global_index'],frame_index=r['frame_index'],split=r['split'],native=r['native_slot'],state=r['role_state'],nominal_ref_dx=dx,nominal_ref_dy=dy,L=L,
   normalized_Chebyshev=c,inside_nominal96_cell_boundaries=c<=.5,inside_central32_cell_boundaries=c<=1/6,
   within_central32_sample_center_extrema=c<=15.5/96,independent_pixel_or_physical_gold=False)
  rows.append(a);groups[r['split']+'::'+r['role_state']].append(a)
  if r['role_state']=='E':occ[r['split']+'::'+r['occluder_or_uncertainty']].append(r)
 summary={k:dict(N=len(a),inside96=sum(r['inside_nominal96_cell_boundaries']for r in a),inside32=sum(r['inside_central32_cell_boundaries']for r in a),inside32_sample_centers=sum(r['within_central32_sample_center_extrema']for r in a),outside96=sum(not r['inside_nominal96_cell_boundaries']for r in a))for k,a in groups.items()}
 out=V/'root_source_label_support'
 c=out/'NOMINAL_REFERENCE_WITHIN_ORIGINAL_CROPS.csv'
 with c.open('x')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 d=dict(completed=True,source_only=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=bind(__file__),inputs=[bind(Q),bind(R),bind(CODE)],
  formula='Producer96 sample centers q + ((i+0.5)/96-0.5)*L. Full96 continuous cell-boundary halfside L/2; central indices32:64 halfside L/6; central sample-center extrema15.5L/96. Not literal visible-surface certification.',
  groups=summary,occluder_coverage={k:dict(rows=len(v),frames=len({r['frame_index']for r in v}),native_counts={str(j):sum(r['native_slot']==j for r in v)for j in range(8)})for k,v in occ.items()},
  rows=bind(c),new_labels_or_filters_or_qL_changes=0,new_model_fit_private_or_V112_calls=0,
  limits_KO=['원래 제공pose/CAD reference의 crop내 위치를 계산한 진단이다. 실제 키포인트 위치 또는3px 품질 gold가 아니다.','기존 D/E 라벨·split·학습마스크는 바꾸지 않았다. 내부ref라고 보임을 보장하거나 외부ref라고 모델 입력을 무효로 만들지 않는다.','public E가 일부native에만 있는 사실은 synthetic native8 범위 부족을 뜻하지 않으며, held E도native4/5이므로 그것만으로 held 실패를 설명하지 않는다.'],selected_model=None)
 f=out/'NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json'
 with f.open('x')as s:json.dump(d,s,ensure_ascii=False,indent=2);s.write('\n')
 print(json.dumps(dict(report=bind(f),groups=summary,occluder_coverage=d['occluder_coverage'])))
if __name__=='__main__':main()
