"""Independent source-only saved label/code/nominal-table review. No model calls."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib,datetime,math
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_source_context_label_generalization_20261024_v157';O=V/'independent_root_source_label_review'
Q=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl'
N=P/'query_visibility_public_native_supervision_control_20261024_v148/source_fit_preparation/PUBLIC_TRAIN43_AND_FIXED_NATIVE35_BEFORE_FIT.json'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
PRE=P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
TRAIN=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
D=V/'root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json';A=V/'root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json'
def allow(p):
 p=Path(p).resolve();assert p.is_relative_to(P) and '_v112' not in str(p).lower() and 'current223' not in str(p).lower() and 'private' not in str(p).lower();return p
def bind(p):p=allow(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(allow(p).read_text())
def check(b):assert bind(b['path'])==b;return allow(b['path'])
def lines(p):return [json.loads(s) for s in allow(p).read_text().splitlines() if s]
d,a=read(D),read(A);rows=lines(Q);spec=read(N);pal=lines(check(spec['public_labels']))
assert len(rows)==1840 and len(pal)==512
for g,r in enumerate(rows):
 assert r['global_index']==g and r['frame_index']==g//8 and r['native_slot']==g%8
 assert r['point_quality_label']==-1 and not r['human_gold'] and not r['exact_real_point_gold']
 assert r['role_target']==({'D':0,'E':1}.get(r['role_state'],-1))
 if r['role_state'] not in ('D','E'):assert r['role_target']==-1
 if r['role_target']>=0:assert r['role_correspondence_supported'] and r['entity_target']==1
c=Counter(r['role_state'] for r in rows);assert c==Counter(D=452,E=101,S=9,OOF=184,U=1094)
mrE=Counter(r['native_slot'] for r in rows if r['split']=='SOURCE_DEVELOPMENT' and r['original_input_valid'] and r['role_target']==1)
hE=Counter(r['native_slot'] for r in rows if r['split']=='SOURCE_VALIDATION' and r['original_input_valid'] and r['role_target']==1)
publicE=mrE.copy();publicD=Counter(r['native_slot'] for r in rows if r['split']=='SOURCE_DEVELOPMENT' and r['original_input_valid'] and r['role_target']==0)
for key,state,count in [('TRAIN_E8','E',publicE),('TRAIN_D35','D',publicD)]:
 for g in spec[key]:
  r=pal[g];assert r['global_index']==g and r['native_slot']==g%8 and r['semantic_state']==state and r['original_input_valid'] and r['split']=='TRAIN_LAB';count[g%8]+=1
full=lambda z:{str(k):z[k] for k in range(8)}
assert full(mrE)==d['MR_available_TRAIN_E_by_native'] and full(hE)==d['MR_available_HELD_E_by_native']
assert full(publicE)==d['all_existing_public_TRAIN_visibility_E_by_native'] and sum(publicE.values())==57
assert full(publicD)==d['all_existing_public_TRAIN_visibility_D_by_native']
assert [j for j in range(8) if not publicE[j]]==d['missing_public_TRAIN_E_native_slots']==[0,1,2,6,7]
for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
 rr=[r for r in rows if r['split']==split];ss=d['MR_counts'][split]
 assert len(rr)==ss['rows'] and sum(r['original_input_valid'] for r in rr)==ss['available']
 for state in ('D','E','S','OOF','U'):assert sum(r['role_state']==state for r in rr)==ss.get('state_'+state,0)
 assert sum(r['original_input_valid'] and r['role_target']>=0 for r in rr)==ss['visibility_loss_rows']
# Independent arithmetic from saved root table; no original nominal-reference body or image read.
with check(a['rows']).open() as f:tbl=list(csv.DictReader(f))
assert len(tbl)==535
seen=set();groups=defaultdict(list)
for t in tbl:
 g=int(t['global_index']);r=rows[g];assert g not in seen;seen.add(g)
 assert r['original_input_valid'] and r['role_target']>=0 and t['state']==r['role_state'] and t['split']==r['split'] and int(t['native'])==r['native_slot'] and float(t['L'])==r['L']
 dx,dy,L=map(float,(t['nominal_ref_dx'],t['nominal_ref_dy'],t['L']));cmax=max(abs(dx),abs(dy))/L
 assert math.isclose(cmax,float(t['normalized_Chebyshev']),rel_tol=0,abs_tol=1e-15)
 # Convert physical displacement to sample indices, then compare index-cell boundaries.
 jx,jy=96*dx/L+47.5,96*dy/L+47.5
 in96=all(-.5<=j<=95.5 for j in (jx,jy));in32=all(31.5<=j<=63.5 for j in (jx,jy));inc=all(32<=j<=63 for j in (jx,jy))
 assert in96==(t['inside_nominal96_cell_boundaries']=='True')
 assert in32==(t['inside_central32_cell_boundaries']=='True')
 assert inc==(t['within_central32_sample_center_extrema']=='True')
 assert t['independent_pixel_or_physical_gold']=='False'
 groups[r['split']+'::'+r['role_state']].append((in96,in32,inc))
recalc={k:dict(N=len(v),inside96=sum(x[0] for x in v),inside32=sum(x[1] for x in v),inside32_sample_centers=sum(x[2] for x in v),outside96=sum(not x[0] for x in v)) for k,v in groups.items()}
assert recalc==a['groups']
# Verify implementation formulas as text only; never import learned or crop producers.
cs=allow(CROP).read_text();ps=allow(PRE).read_text();ts=allow(TRAIN).read_text()
assert 'u=(np.arange(n,dtype=np.float64)+.5)/n-.5' in cs
assert 'q[0]+u*L,q[1]+u*L' in cs
assert 'x[:,:,32:64,32:64]' in ps
assert "vi=np.flatnonzero(rm['available']&rm['training']&(rm['role']>=0))" in ts
assert "primary_visibility_never_filtered_by_entity_quality_or_GTerror=True" in ts
example=[]
for g in (1044,772):
 r=rows[g];t=next(t for t in tbl if int(t['global_index'])==g);xy=r['query_xy'];dx,dy=float(t['nominal_ref_dx']),float(t['nominal_ref_dy'])
 example.append(dict(global_index=g,query_id=r['query_id'],source_id=r['source_id'],split=r['split'],weak_role=r['role_state'],query_xy=xy,nominal_ref_xy=[xy[0]+dx,xy[1]+dy],L=r['L'],inside_nominal96=t['inside_nominal96_cell_boundaries']=='True',inside_nominal32=t['inside_central32_cell_boundaries']=='True',weak_teacher_reason=r['source_role_review_reason'],interpretation='Existing named native-role weak label and actual query crop ownership are different targets. This example is descriptive only; no replacement label or point-quality truth is inferred.'))
review=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_only=True,reviewed=[bind(D),bind(A),d['code'],a['code']],checks=dict(public_TRAIN_E_total=57,public_TRAIN_E_by_native=full(publicE),MR_TRAIN_E49_by_native=full(mrE),MR_HELD_E52_by_native=full(hE),missing_public_TRAIN_E_slots=[0,1,2,6,7],all1840_original_label_identity_order_unchanged=True,original_MR_states=dict(c),S_OOF_U_total=1287,S_OOF_U_role_targets_all_minus1=True,all1840_real_point_quality_unknown=True,held45_original360rows_retained=True,nominal_table_all535_rows_independently_recomputed=True,within32_formula_matches_producer_sample_centers_and_saved32_64=True,inside32_is_continuous_cell_boundary_not_samplecenter_hull=True,all_root_nominal_groups_match=True),nominal_groups_recomputed=recalc,examples=example,findings=['Public-real TRAIN positive E support totals57 across native3/4/5; five slots have no public-real E examples. Synthetic all8 support is a separate source, so this does not imply absent synthetic roles.','All52held E labels are native4/5 already represented in MR TRAIN; missing public-real slots alone cannot explain held failure.','TRAIN and held E occluder types are disjoint in existing weak labels; visual review corroborates different local textures, not a causal explanation.','Nominal-reference positions use cm-uncertain provided pose/native; none is physical/native/pixel gold. Within96/32 diagnoses reference placement, not surface ownership, visibility or sample correctness.','S/OOF/U remain unknown for binary visibility; no held row, split, role, query or availability changed.'],next_source_action_review='TRAIN-only query-local ownership witness audit is a coherent diagnostic next step, with existing native-role D/E retained separately. Any actual-public same-q counterfactual feasibility claim needs explicit support/ownership witnesses. Pixel absence is never sufficient to infer physical external occlusion. No new fit, relabel, threshold or model action follows from this review.',limitations=['Nominal arithmetic verified from bound saved root CSV and producer code; no independent physical ground truth exists.','g1044/g772 demonstrate target distinction, not proof that all native-role labels are wrong.','This review does not authorize moving held45 or fixednative35 into training.'],access=dict(new_image_reads=0,nominal_reference_body_reads=0,model_feature_weight_or_initializer_body_reads=0,new_forwards=0,new_fit=0,private_current_V112_body_reads=0,source_label_mutations=0),inputs=[bind(Q),bind(N),spec['public_labels'],a['rows'],bind(CROP),bind(PRE),bind(TRAIN),bind(__file__)])
(O/'REVIEW.json').write_text(json.dumps(review,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(dict(PASS=True,review=bind(O/'REVIEW.json'),publicE=full(publicE),total_publicE=sum(publicE.values()),groups=recalc,examples=[{k:r[k] for k in ('global_index','query_xy','nominal_ref_xy','L','inside_nominal96','inside_nominal32')} for r in example])))
