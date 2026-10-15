"""Three fixed CAL cached-chain V30 reranks; no new pixels, labels or poses.

Source/selection declaration precedes execution. The original graph and raw
chain archive remain unchanged; output is a separate new-selection-profile
engineering diagnosis, not a validated method or a V29 noise-model transfer.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np

from . import observed_bi_line_association_v4_30 as F
from . import replay_exact_graph_actual_CAL_v4_29 as R
from . import observe_geometry_guided_edges_v4_29b as OBS

C=OBS.C
B=OBS.OUTPUT.parent
OUTPUT=B/'observed_bi_line_association_CAL_cases_v4_30'
PLAN=B/'observed_bi_line_association_plan_v4_30/PREDECLARATION.json'
DIAGNOSIS=B/'CAL_geometry_guided_noise_diagnosis_v4_29'
CASES=('wood_184309:000925','wood_184309:000804','eval_cad:1778653017736058368')


def _sources():
    modules=(F,F.P,F.G,F.P.ORDER,F.B,F.M,R,OBS,OBS.V,C)
    sources={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)for m in modules}
    for path in(Path(__file__),Path(__file__).with_name('test_run_observed_bi_line_association_CAL_v4_30.py'),
            Path(F.__file__).with_name('test_observed_bi_line_association_v4_30.py')):
        sources[str(path.resolve().relative_to(C.ROOT))]=C.sha(path)
    return sources


def fixed_cases(diagnostic_ids):
    if tuple(dict.fromkeys(diagnostic_ids))!=CASES:raise ValueError('Preserve original fixed diagnostic top3 CAL frames')
    if len(set(CASES))!=3:raise ValueError('Three unique fixed CAL cases')
    return CASES


def compare_saved_reference_edge(node,saved_diagnostic):
    """Only the already-sealed named edge projection, never a new GT choice."""
    if sorted(node['native_edge'])!=sorted(saved_diagnostic['native_edge']):return None
    q=np.asarray(saved_diagnostic['projected_CAL_reference_edge']);delta=q[1]-q[0]
    direction=delta/np.linalg.norm(delta);normal=np.array([-direction[1],direction[0]])
    n=np.asarray(node['normal']);n=n/np.linalg.norm(n);offset=float(node['offset'])
    actual=np.asarray(node['actual_inliers']);mean=actual.mean(0);anchor=mean-n*(n@mean+offset)
    return dict(reference='same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term',
        native_edge=saved_diagnostic['native_edge'],signed_normal_residual_px=float(normal@(anchor-q[0])),
        angle_deg=float(np.degrees(np.arcsin(np.clip(abs(float(np.cross(n,normal))),0.,1.)))))


def declare():
    if PLAN.exists():raise FileExistsError('Preserve V30 source/selection declaration')
    diagnostic=C.read(DIAGNOSIS/'DIAGNOSIS_DATA.json');fixed_cases([r['id']for r in diagnostic['records']])
    index={r['id']:r for r in C.read(OBS.OUTPUT/'CAL_ROW_INDEX.json')['entries']}
    protocol=C.read(OBS.OUTPUT/'PROTOCOL.json');cases=[]
    for ident in CASES:
        row=index[ident]
        if row['partition']!='CAL':raise ValueError('No primary-case rerank or selection')
        path=OBS._row_path(OBS.OUTPUT,ident)
        if C.sha(path)!=row['sha256']:raise ValueError('Frozen canonical CAL spool required')
        cases.append(dict(row,spool_relative=str(path.relative_to(C.ROOT))))
    paths=[DIAGNOSIS/'DIAGNOSIS_DATA.json',DIAGNOSIS/'RENDER_SEAL.json',OBS.OUTPUT/'PROTOCOL.json',
        OBS.OUTPUT/'PREPARATION_SEAL.json',OBS.OUTPUT/'CAL_ROW_INDEX.json',OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json']
    paths.extend(C.ROOT/c['spool_relative']for c in cases)
    sources=_sources();inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    for mapping in(sources,inputs,protocol['protected_user_sha256']):R._verify(mapping)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='NEW_V30_ASSOCIATION_FIXED_THREE_CAL_CACHED_RERANK_PLAN',UTC=datetime.now(timezone.utc).isoformat(),
        cases=cases,configuration=protocol['configuration'],selection_profile=F.SELECTION_PROFILE,recipe=F.RECIPE,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protocol['protected_user_sha256'],
        case_selection='same already-declared CAL top3 residual union; no new error/ID routing or candidate selection',
        only_new_admission='all eligible original named B/I endpoints to infinite actual TLS normal distance <= existing cfg normal reach43',
        ranking='same frozen geometry/ordered graph+stable heap+fresh rank memo; all raw alternatives/indices retained in original archive',
        comparison='old stored graph vs NEW graph names/counts/associations; no pose accuracy claim',
        diagnostic_reference='only already-saved fixed top3 named-edge projection; evaluate old/new selected same edge after ranking; other edges reference unset; never guide or candidate selection',
        cache_mask_context='same existing mask loaded after actual GO only; no new RGB/NMS/candidate TLS measurements',
        inherited_rounded_cached_support_TLS_diagnostic_refits_allowed=True,
        old_V29_noise_transfer=False,new_CAL46_noise_and_uniform_evaluation_required=True,
        prepare_completed_observation_rows_or_masks_or_images_decoded=0,actual_rerank_requires_separate_root_GO=True,
        new_RGB_NN_labels_posefit_or_KP_adoption=0,current_V29_inference_and_models_unchanged=True))
    for relative,digest in sources.items():
        target=PLAN.parent/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target)!=digest:raise ValueError('Exact source snapshot required')
    C.write(PLAN.parent/'PLAN_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=sources))
    print(dict(plan_sha256=C.sha(PLAN),selection_profile=F.SELECTION_PROFILE,cases=CASES))


def run(*,root_actual_rerank_GO=False):
    if root_actual_rerank_GO is not True:raise PermissionError('Separate root cached-CAL-rerank GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve previous rerank attempt')
    plan=C.read(PLAN);pseal=C.read(PLAN.parent/'PLAN_SEAL.json')
    if C.sha(PLAN)!=pseal['plan_sha256']or plan['selection_profile']!=F.SELECTION_PROFILE:raise ValueError('Frozen V30 declaration required')
    fixed_cases([c['id']for c in plan['cases']])
    for mapping in(plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256']):R._verify(mapping)
    OUTPUT.mkdir(parents=True);(OUTPUT/'graphs').mkdir();results=[]
    diagnostics={r['id']:r for r in C.read(DIAGNOSIS/'DIAGNOSIS_DATA.json')['records']}
    for i,case in enumerate(plan['cases']):
        path=C.ROOT/case['spool_relative'];row=OBS._read_single(path,case)
        if row['partition']!='CAL':raise ValueError('CAL-only original row required')
        observation=row['observation'];geometry=row['initial_geometry'];guides=R.restore_pose_guides(row)
        binding=dict(sha256=row['image_sha256'],image_hw=geometry['image_hw'])
        regenerated=F.G.prepare_joint_geometry_guides(geometry,row['states'],row['visible_measurements'],
            image_binding=binding,configuration=plan['configuration'],pose_guides=guides)
        if R._canonical_sha(regenerated)!=R._canonical_sha(observation['guide_bank']):raise ValueError('Saved original guide bank must reconstruct exactly')
        mask_path=C.ROOT/row['mask_binding']['path']
        if C.sha(mask_path)!=row['mask_binding']['sha256']:raise ValueError('Original cached mask bytes changed')
        with np.load(mask_path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
        if F.P.VIS.mask_pixel_sha256(mask)!=observation['target_mask_pixel_sha256']:raise ValueError('Exact old mask context required')
        context=F.P.VIS.prepare_edge_visibility(mask,geometry,row['states'],row['visible_measurements'],
            image_binding=binding,occluder_mask=None,occluder_binding=None)
        started=time.monotonic()
        result=F.rank_observed_BI_associated_graph(observation['chain_observation'],geometry,row['states'],row['visible_measurements'],
            image_binding=binding,configuration=plan['configuration'],pose_guides=guides,pallet_mask=mask,visibility_context=context)
        seconds=time.monotonic()-started;graph=result['graph'];cfg,_=F.G._config(plan['configuration'])
        R.decorate_graph(graph,observation['chain_observation'],cfg)
        C.save_rows(OUTPUT/'graphs'/('case'+str(i)+'.jsonl.gz'),[result])
        old=observation['joint_graph'];new_names=graph['selected_line_names'];old_names=old['selected_line_names']
        old_reference=[dict(line_name=node['name'],**comparison)for node in old['line_candidates']
            if node['name']in old_names and (comparison:=compare_saved_reference_edge(node,diagnostics[row['id']]))is not None]
        new_reference=[dict(line_name=node['name'],**comparison)for node in graph['line_candidates']
            if node['name']in new_names and (comparison:=compare_saved_reference_edge(node,diagnostics[row['id']]))is not None]
        if not old_reference or any(abs(r['signed_normal_residual_px']-diagnostics[row['id']]['signed_normal_residual_px'])>1e-7 for r in old_reference):
            raise ValueError('Original fixed CAL diagnostic projection must reproduce saved old error')
        record=dict(id=row['id'],rank_seconds=seconds,old_candidates=old['actual_unique_line_candidates'],
            new_candidates=graph['actual_unique_line_candidates'],checked=result['candidate_nodes_checked'],
            rejected=result['rejected_node_count'],old_selected=old_names,new_selected=new_names,
            previous_selected_rejections=[r for r in result['node_association_ledger']if r['line_name']in old_names and not r['passes']],
            all_selected_q_distance_ledger=[r for r in result['node_association_ledger']if r['line_name']in new_names],
            old_top_named_edge_fixed_CAL_reference=old_reference,new_top_named_edge_fixed_CAL_reference=new_reference,
            other_new_edges_fixed_reference='not available in sealed top3 projection; not invented/reassigned',
            old_joint_alternatives=len(old['joint_alternatives']),new_joint_alternatives=len(graph['joint_alternatives']),
            old_joint_states_evaluated=old['joint_states_evaluated'],new_joint_states_evaluated=graph['joint_states_evaluated'],
            zero_candidate_abstention=graph['actual_unique_line_candidates']==0,zero_selected_edges=len(new_names)==0,
            same_original_spool_sha256=C.sha(path),same_cached_mask_sha256=C.sha(mask_path),
            raw_chains_unchanged=True,original_node_chain_provenance_preserved=True,
            new_selection_profile=F.SELECTION_PROFILE,validated_noise_model_available=False,
            new_RGB_NMS_candidate_TLS_labels_or_posefit_calls=0)
        C.write(OUTPUT/('case'+str(i)+'_SUMMARY.json'),record);results.append(record)
        print(dict(id=row['id'],rejected=record['rejected'],new_selected=new_names,seconds=seconds),flush=True)
        del row,observation,result,graph
    C.write(OUTPUT/'RESULTS.json',dict(selection_profile=F.SELECTION_PROFILE,cases=results,primary_performance_claim=False))
    text='# v30 원 B/I 점과 실제 선 연결: 고정 CAL3 재순위\n\n새 이미지 측정/키포인트/PnP/정답 읽기 없이 원 cached 선을 재순위했다. 모든 raw chain은 원 archive에 보존한다. 기존43px normal reach를 무한 TLS 선과 named-edge의 모든 eligible B/I q 사이에 적용했다. 새 선택 집단이므로 기존 v29 noise를 검증된 모델로 쓰지 않는다. 별도 CAL46/noise/full238 평가 이전에는 새 방법 채택이나 정확도 개선을 주장하지 않는다.\n\n'
    text+='| CAL 사례 | 원/새 후보 수 | gate 제외 | 원/새 선택 선 수 | rank 시간(s) |\n|---|---:|---:|---:|---:|\n'
    for row in results:text+=f'| {row["id"]} | {row["old_candidates"]}/{row["new_candidates"]} | {row["rejected"]} | {len(row["old_selected"])}/{len(row["new_selected"])} | {row["rank_seconds"]:.3f} |\n'
    text+='\n## 기존 큰 잔차 native 변의 고정 CAL 투영 대조\n\n원래 봉인된 top3 진단 projection만 재사용했다. 재순위에는 기준이 들어가지 않는다. 다른 새 선택 변은 기준값을 만들지 않고 미제공으로 남긴다. 같은 변이 선택되지 않으면 정밀 대체 성공이 아닌 미선택이다.\n\n'
    for row in results:
        text+='- '+row['id']+': 원 '+str(row['old_top_named_edge_fixed_CAL_reference'])+' → 새 '+str(row['new_top_named_edge_fixed_CAL_reference'])+'\n'
    text+='\n모든 새 선택선의 원 B/I q 법선 거리·source/양끝 검사는 RESULTS.json에 보존한다. 후보0/선택0 abstention 및 전체 대안 수/계산 수도 숨기지 않는다. gate 통과 자체를 실제 native 대응 정확도 증명으로 표현하지 않는다.\n'
    (OUTPUT/'REPORT_KO.md').write_text(text)
    for mapping in(plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256']):R._verify(mapping)
    C.write(OUTPUT/'RERANK_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
        assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()},
        current_V29_inference_models_or_observations_changed=False,new_RGB_NN_labels_posefit_KP_adoption=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','run'))
    parser.add_argument('--root-actual-rerank-GO',action='store_true');args=parser.parse_args()
    declare()if args.stage=='declare'else run(root_actual_rerank_GO=args.root_actual_rerank_GO)
