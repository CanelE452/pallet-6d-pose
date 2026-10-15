"""Two fixed EB cases, archived-data-only post-hoc diagnosis; no fitting."""
import argparse,gzip,json,math,time,hashlib
from pathlib import Path
import numpy as np
from .independent import projected_cuboid,ray_cost,_cuboid_vertices

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rotdeg(A,B):return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(A).T@np.asarray(B))-1)/2,-1,1))))
def solve_objective(ms,projected):
 values=[]
 for m in ms:
  e=np.linalg.solve(np.linalg.cholesky(m['covariance_px2']),projected[m['native_corner']]-m['xy']);s=float(e@e)
  values.append(18*(math.sqrt(1+s/9)-1))
 return math.fsum(values)
def run(root,output):
 root,output=Path(root),Path(output)
 if output.exists():raise FileExistsError('Preserve every previous diagnostic attempt')
 output.mkdir(parents=True)
 start=time.monotonic();files=[root/'PREDICTION_SEAL.json',root/'score/SCORE_SEAL.json'];before={p.name:sha(p) for p in files}
 frames=read(root/'e_inputs/INPUTS.json')['frames'];scores={r['id']:r for r in read(root/'score/SCORED_ROWS.json')['rows']}
 with gzip.open(root/'GUARD_AUDITS.jsonl.gz','rt') as f:audits={r['id']:r for r in map(json.loads,f) if r['arm']=='EB'}
 cases=[]
 for index,frame in enumerate(frames):
  if frame['id'] not in ['plastic_day_01:010995','plastic_night_01:039093']:continue
  op=root/'e_run/observations'/f'{index:03d}_E_FROZEN.json';sp=root/'e_run/poses'/f'{index:03d}_E_FROZEN.json'
  obs=read(op)['result']['observation'];solved=read(sp)['result'];solution=solved['solution'];E=frame['arms']['E_FROZEN'];coarse=E['coarse_pose'];candidate=solved['actual_pose'];K=frame['K']
  initial=projected_cuboid(obs,coarse,K);final=projected_cuboid(obs,candidate,K);xyz=np.asarray(obs['initial_geometry']['native_xyz'])
  raw=np.asarray(obs['original_raw_points']);measurements=obs['frozen_visible_measurements'];ids=[m['native_corner'] for m in measurements]
  edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(xyz[i]!=xyz[j])==1]
  records=[]
  for m in measurements:
   k=m['native_corner'];xy=np.asarray(m['xy']);L=np.linalg.cholesky(m['covariance_px2']);d0=initial[k]-xy;d1=final[k]-xy;dist=np.linalg.norm(initial-xy,axis=1);j=int(dist.argmin())
   records.append(dict(native_corner=k,source_native_corner=E['native_vertex_mapping']['target_to_source_native_ids'][k],state=m['state'],source=m['source'],raw_xy=raw[k].tolist(),xy=m['xy'],covariance_px2=m['covariance_px2'],movement_px=float(np.linalg.norm(xy-raw[k])),initial_model_xy=initial[k].tolist(),final_model_xy=final[k].tolist(),initial_matched_residual_px=float(np.linalg.norm(d0)),final_matched_residual_px=float(np.linalg.norm(d1)),initial_mahalanobis=float(np.linalg.norm(np.linalg.solve(L,d0))),final_mahalanobis=float(np.linalg.norm(np.linalg.solve(L,d1))),nearest_initial_projected_vertex_id=j,nearest_initial_vertex_residual_px=float(dist[j])))
  rays=[]
  for record in obs['visible_correction']['records']:
   for ray in record.get('rays',[]):
    if not ray.get('supported'):continue
    k,j=ray['native_corner'],ray['native_neighbor'];P=np.asarray(ray['observed_points']);W=np.asarray(ray['observation_confidence']);base=ray_cost(P,W,initial,k,j);end=ray_cost(P,W,final,k,j)
    nearest=sorted([(ray_cost(P,W,initial,*e)[0],e) for e in edges])[:3]
    rays.append(dict(assigned_edge=[k,j],measurement_source=ray['measurement_source'],guide_source=ray['guide_source'],sample_count=len(P),observed_points=P.tolist(),observation_confidence=W.tolist(),observed_points_bbox=[P.min(0).tolist(),P.max(0).tolist()],baseline_cost=base[0],candidate_cost=end[0],nearest_initial_model_edges=[dict(edge=list(e),cost=v) for v,e in nearest],native_semantics_independently_annotated=False))
  feasible=solution['feasible_pose_samples'];feasible_summary=[]
  for x in feasible:
   feasible_summary.append(dict(objective=x['objective'],objective_gap_from_selected=x['objective']-solution['objective'],rotation_difference_deg=rotdeg(x['R'],candidate['R_cf']),translation_difference_m=float(np.linalg.norm(np.asarray(x['t'])-candidate['centroid'])),initialization=x['initialization']))
  cases.append(dict(id=frame['id'],partition=frame['partition'],case='EB_damage' if frame['id'].endswith('010995') else 'EB_recovery',bindings=dict(observation_sha256=sha(op),solve_sha256=sha(sp)),states=obs['states'],BI_ids=ids,S_ids=[k for k,s in enumerate(obs['states']) if s=='S'],E_geometry_BI_roles=obs['initial_geometry']['initial_roles'],source_N3_dimensions=frame['arms']['N3_SUBPIX']['coarse_pose']['cf_extents'],E_dimensions=coarse['cf_extents'],vertex_mapping=E['native_vertex_mapping'],own_BI_measurements=records,all_initial_projection_xy=initial.tolist(),all_final_projection_xy=final.tolist(),observed_rays=rays,guard_decision=audits[frame['id']]['result']['decision_audit'],solver_objective=dict(initial_same_fixed_measurements=solve_objective(measurements,initial),candidate_recomputed=solve_objective(measurements,final),candidate_archived=solution['objective'],objective_definition='radial pseudo-Huber with existing covariance and scale3; no initial pose prior'),solver_rank=solution['rank'],solver_ambiguity=solution['ambiguity'],solver_search=solution['search'],feasible_solution_summary=feasible_summary,final_model_visibility_audit=solution['final_model_visibility_audit'],metrics={a:scores[frame['id']]['metrics'][a] for a in ['R0','N0','N1','E0','E1','EB']},initial_to_candidate_rotation_deg=rotdeg(coarse['R_physical'],candidate['R_physical']),physical_registration_Q_preserved=bool(np.allclose(np.asarray(coarse['R_cf']).T@coarse['R_physical'],np.asarray(candidate['R_cf']).T@candidate['R_physical'],atol=1e-9,rtol=0))))
 out=dict(schema='TWO_FIXED_EB_CASES_POSTHOC_DIAGNOSIS_V1',posthoc_only=True,new_candidates_or_parameters=False,new_model_or_PnP_or_fit_calls=0,original_sealed_artifacts_modified=False,bindings=before,cases=cases,interpretation_limits=['GT metrics identify which pose is accurate; there are no independent per-ray native-edge annotations.','Nearest other projected edge is a posthoc distance diagnostic, never a new correspondence or inference decision.','Vertex bijection is numerically exact conditional on the original N3 physical registration; it cannot certify the physical identity of detector channels.'],wall_seconds=time.monotonic()-start)
 target=output/'DIAGNOSTIC_EB_CASES.json'
 with target.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 assert before=={p.name:sha(p) for p in files}
 lines=['# EB의 회복·손상 두 사례: 선 비용 개선이 정확도를 보증하지 못한 이유','', '봉인된 예측·관측·점수만 읽은 사후 진단이다. 새 후보·임계값·학습·PnP·수치 최적화·정답 수정은 없다.','', '**손상 사례에서는 선을 크게 잘못 이동한 것이 아니다. 기존 N3의 코너 대응을 가져온 5점과 정확했던 E0의 같은 코너 정의가 168~206px 어긋나 있었다. 재PnP는 그 대응을 매우 잘 맞춘 다른 자세를 선택했다.** 선 비용 검사는 같은 잘못 연결된 코너·선 관측을 다시 검사하므로 이를 차단하지 못했다.','', '| 사례 | E0 위치/회전 | EB 위치/회전 | 실제 이동 관측 | 유효 ray | 선 가중 Huber 비용 |','|---|---|---|---|---:|---|']
 for c in cases:
  m=c['metrics'];lc=c['guard_decision']['line_cost'];moves=[f"{x['native_corner']}:{x['movement_px']:.3f}px" for x in c['own_BI_measurements'] if x['movement_px']>1e-8]
  lines.append(f"| {c['id']} | {m['E0']['translation_cm']:.4f}cm/{m['E0']['rotation_deg']:.4f}° | {m['EB']['translation_cm']:.4f}cm/{m['EB']['rotation_deg']:.4f}° | {', '.join(moves)} | {lc['valid_ray_count']} | {lc['baseline_weighted_mean']:.3f} → {lc['candidate_weighted_mean']:.3f} |")
 lines+=['', '## 같은 좌표 변환인데 결과가 반대인 이유','', '두 사례 모두 원 N3는 [1.3,0.11,1.1], E는 [1.1,0.11,1.3]m이며, 원 N3 Q는 약YAW90, E Q는 약YAW180이다. E native→원 N3 native 순열은 동일하게 [1,5,6,2,0,4,7,3]이다. 물리 정점 매칭 오차는 각각2.45e−16,2.51e−16m로 수치 재매핑 오류는 검출되지 않았다. 그러나 이 변환은 원 N3 채널의 물리 코너 정체가 맞다는 조건을 물려받는다. 픽셀의 물리 코너 정체 자체를 인증하지 않는다.','', '손상 사례의 원 N3 N0부터 회전 오차83.15°였고 E0는0.25°였다. 새 B/I 재PnP가 원 N3 대응에 다시 맞추면서84.14°가 되었다. 반대로 회복 사례는 N0가1.87°, E0가84.89°였고 같은 원 N3 대응에 다시 맞추자1.04°로 돌아왔다. **예측끼리 어긋난다는 사실이나 같은 선 비용 감소만으로는 어느 쪽 대응이 맞는지 가를 수 없었다.**','']
 for c in cases:
  lines += [f"## {c['id']}",'',f"상태: {' '.join(f'{i}:{s}' for i,s in enumerate(c['states']))}. B/I 입력은 {c['BI_ids']}이며, S는 {c['S_ids']}다. S는 영상 관측에 넣지 않았다.",'','| E native ID (원 N3 ID) | 관측 | 이동 px | 같은 ID의 초기 재투영 잔차 px | 최종 잔차 px | 가장 가까운 초기 모델 정점 (잔차 px) |','|---|---|---:|---:|---:|---|']
  for z in c['own_BI_measurements']:
   lines.append(f"| {z['native_corner']} ({z['source_native_corner']}) | {z['source']} | {z['movement_px']:.3f} | {z['initial_matched_residual_px']:.3f} | {z['final_matched_residual_px']:.3f} | {z['nearest_initial_projected_vertex_id']} ({z['nearest_initial_vertex_residual_px']:.3f}) |")
  lines+=['','| 관측 ray의 지정 edge | 초기 지정 edge 비용 | 최종 지정 edge 비용 | 초기 모델에서 가장 가까운 다른 edge 비용 |','|---|---:|---:|---|']
  for z in c['observed_rays']:
   near=z['nearest_initial_model_edges'][0]
   lines.append(f"| {z['assigned_edge']} | {z['baseline_cost']:.3f} | {z['candidate_cost']:.3f} | {near['edge']}: {near['cost']:.3f} |")
  obj=c['solver_objective'];a=c['solver_ambiguity'];fs=c['feasible_solution_summary']
  lines+=['',f"PnP rank는6, condition은 {c['solver_rank']['visible']['condition']:.3f}이다. 기존 ambiguity.detected={a['detected']}, 저장 alternatives={len(a['alternatives'])}. 같은 B/I에 대한 목적함수는 초기 {obj['initial_same_fixed_measurements']:.3f} → 최종 {obj['candidate_archived']:.6f}다. 초기 E 자세는 PnP 사전분포로 보호되지 않았으며, 기존 solver가 관측을 맞추는 전역 multistart를 수행했다.",f"저장된 서로 다른 feasible 해는 {len(fs)}개다. 그 목적함수 gap은 {[round(x['objective_gap_from_selected'],9) for x in fs]}, 선택해와 회전 차이는 {[round(x['rotation_difference_deg'],6) for x in fs]}°다. 따라서 이 사례를 '동률인 두 해 중에서 운 나쁘게 골랐다'고 단정할 수 없다. 잘못 연결됐을 수 있는 관측을 조건으로 한 유일·국소적 해도 잘못된 물리 자세가 될 수 있다.",'']
 lines += ['## 확인된 사실과 해석의 한계','', '- 확인: B는 자기 own 관측의 모든 유효 ray 비용 비악화·전체 비용 감소를 만족했고, 독립 계산도 이를 확인했다. 비용 코드 오류로 잘못 채택된 사례는 아니다.','- 확인: 손상 사례의 네 B는 새 선 교점이 아니라 원 검출 좌표 fallback이고, 새 이동은 I0 한 점0.857px뿐이다. 두 유효 ray도 모두 그 I점의 내부 RGB gradient다. 전역 대응을 검증할 별도 경계 선은 B 검사에 남지 않았다.','- 확인: 손상 I0→3 ray는 정확도가 확인된 초기 E0의 지정0→3 선 비용234.80보다 다른1→2 선 비용0.105에 훨씬 가깝다. I0→4 ray도 다른1→5 선 비용2.149에 가깝다. 두 관측이 다른 코너의 실재 선을 잡았다는 해석과 일치한다.','- 해석: 가장 강한 원인은 원 N3의 물리 코너 대응/위상 오류를 E 경로에 다시 주입하고, 그 대응에 묶인 선 증거를 재검사한 것이다. 물리 vertex 순열의 산술 오류와 구분해야 한다.','- 미확정: 각 RGB 선에 독립적인 native edge 정답이 없으므로 픽셀별로 어느 물리 edge인지 확정하지 않는다. 정확한 edge identity는 새 라벨 없이 인증할 수 없다.','- rank6·ambiguity=False는 주어진 대응에 대한 국소 제약 진단이며, 물체 코너의 의미 대응이 맞다는 보증이 아니다.','', '이번 고정 실험 뒤 새 규칙이나 추가 임계값을 만들지 않는다. EB는 PRIMARY에서 회복1·손상1로 성공 수 개선이 없다는 결과를 유지한다.']
 with (output/'DIAGNOSTIC_EB_CASES_KO.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 source_rows=[]
 for c in cases:
  frame=next(f for f in frames if f['id']==c['id']);arm=frame['arms']['N3_SUBPIX'];pose=arm['coarse_pose'];K=np.asarray(frame['K'])
  camera=_cuboid_vertices(pose['cf_extents'])@np.asarray(pose['R_cf']).T+pose['centroid'];h=camera@K.T;q=h[:,:2]/h[:,2,None]
  native=np.asarray(arm['native_points']);mapping=frame['arms']['E_FROZEN']['native_vertex_mapping']['target_to_source_native_ids'];res=[]
  for m in c['own_BI_measurements']:
   i=m['native_corner'];j=mapping[i]
   res.append(dict(E_native_id=i,source_N3_native_id=j,source_N3_own_F_projection_xy=q[j].tolist(),source_N3_raw_xy=native[j].tolist(),
    source_N3_own_F_residual_px=float(np.linalg.norm(q[j]-native[j])),own_E0_same_physical_ID_raw_residual_px=float(np.linalg.norm(np.asarray(c['all_initial_projection_xy'][i])-native[j]))))
  source_rows.append(dict(id=c['id'],sourceQ_rounded=np.rint(c['vertex_mapping']['source_Q']).astype(int).tolist(),targetQ_rounded=np.rint(c['vertex_mapping']['target_Q']).astype(int).tolist(),
   target_to_source_native_ids=mapping,physical_bijection_checked_independently=True,source_N0_metric=c['metrics']['N0'],own_E0_metric=c['metrics']['E0'],matched_source_and_target_residuals=res))
 with (output/'DIAGNOSTIC_EB_SOURCE_CORRESPONDENCE.json').open('x') as f:
  json.dump(dict(schema='EB_ARCHIVED_SOURCE_CORRESPONDENCE_COMPARISON_V1',new_fit_calls=0,new_candidates=0,rows=source_rows),f,indent=2);f.write('\n')
 extra=['','## 원 N3 F와 입력 재번호 구현을 따로 확인','',
  '원 N3 F는 자기 source native 순서에서는 원 좌표를 잘 재투영한다. 손상 사례에서 그 F의 물리 회전은 정답과83.15° 어긋나지만, 새 E 입력에 들어간 원 N3 채널은 자기 F와1.76~4.90px만 어긋난다. 같은 채널을 선언된 물리 Q로 E에 옮기면 정확했던 E0의 같은 물리 ID와168.49~206.66px 어긋난다. 즉 기존 틀린 대응·위상과 내부적으로 일관된 q를 다시 주입하는 현상이다.','',
  '| 사례 | sourceQ / targetQ | 원 N3 F가 원 q를 맞추는 잔차 범위 px | 옮긴 q와 E0 같은 ID 잔차 범위 px |','|---|---|---:|---:|']
 for c in source_rows:
  z=c['matched_source_and_target_residuals'];a=[v['source_N3_own_F_residual_px'] for v in z];b=[v['own_E0_same_physical_ID_raw_residual_px'] for v in z]
  extra.append(f"| {c['id']} | YAW90 / YAW180 | {min(a):.3f}~{max(a):.3f} | {min(b):.3f}~{max(b):.3f} |")
 extra+=['','배열 순열 구현·CF 기저·Q가 계약과 다르다는 증거는 이번 전수 검산에서 없었다. 원 검출/N3가 부여한 코너의 물리적 정체 및 W/D·위상 가정이 맞다는 의미 계약이 실패한 것으로 해석하는 것이 기록과 일치한다. 이는 모든 실패의 원인이 확정됐다는 뜻이 아니다. 원 N3 채널별 물리 코너의 독립 정답이나 ray별 edge 정답은 없으므로 그 부분은 추론으로 남긴다.','',
  '추가 숫자: DIAGNOSTIC_EB_SOURCE_CORRESPONDENCE.json. 여기의 가장 가까운 정점·선 검색은 고정 결과를 설명하기 위한 거리 조회이며 새 대응 선택·새 fit·재추론에 사용하지 않았다.']
 with (output/'DIAGNOSTIC_EB_CASES_KO.md').open('a') as f:f.write('\n'.join(extra)+'\n')
 print(json.dumps(dict(cases=len(cases),wall_seconds=out['wall_seconds'],files=['DIAGNOSTIC_EB_CASES.json','DIAGNOSTIC_EB_CASES_KO.md'],original_seals_unchanged=True),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.folder,a.output)
