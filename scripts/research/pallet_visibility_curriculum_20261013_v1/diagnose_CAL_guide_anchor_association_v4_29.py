"""Fixed top3 CAL groups: TLS distances to existing B/I q, no GT or rerouting."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import diagnose_CAL_geometry_guided_noise_v4_29 as FIRST
from . import image_edge_observations_v4_29 as P

C,D=FIRST.C,FIRST.D
OUTPUT=D.OUTPUT.parent/'CAL_geometry_guided_anchor_association_diagnosis_v4_29'
PLAN=OUTPUT.parent/'CAL_geometry_guided_anchor_association_plan_v4_29/PREDECLARATION.json'


def q_distances(normal,offset,states,hidden,visible):
    result=[]
    for m in visible:
        k=m['native_corner']
        if hidden[k]or states[k]not in('B','I'):continue
        q=np.asarray(m['xy'],float)
        result.append(dict(native_corner=k,state=states[k],source=m['source'],xy=q.tolist(),
            signed_normal_distance_px=float(np.asarray(normal)@q+offset)))
    return result


def declare():
    if PLAN.exists():raise FileExistsError('Preserve no-GT association diagnostic plan')
    paths=[FIRST.OUTPUT/'DIAGNOSIS_DATA.json',FIRST.OUTPUT/'RENDER_SEAL.json',
        FIRST.OBS.OUTPUT/'CAL_ROW_INDEX.json',FIRST.OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json']
    sources={str(p.resolve().relative_to(C.ROOT)):C.sha(p)for p in
        (Path(__file__),Path(__file__).with_name('test_diagnose_CAL_guide_anchor_association_v4_29.py'),Path(P.__file__))}
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='FIXED_TOP3_CAL_TLS_VS_ALL_ORIGINAL_BI_Q_ASSOCIATION_PLAN_V29',UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths},
        frames_groups='same sealed top3 per-policy union; no reselection',
        measurements='absolute/signed actual TLS normal distance to ALL existing eligible initial-nonS B/I q; source anchor/model guide/original initial displacement and configured43 normal reach',
        source_contract='state/S gate before each q coordinate; no S/U observation fields',
        nearest_q_only_descriptive_not_native_reassociation=True,
        reference_or_images_or_new_measurements_fits_parametersearch_runtime_changes=0))
    print(dict(plan_sha256=C.sha(PLAN)))


def render():
    plan=C.read(PLAN)
    for key in('source_sha256','input_sha256'):D._verify(plan[key])
    if OUTPUT.exists():raise FileExistsError('Preserve original association diagnosis')
    data=C.read(FIRST.OUTPUT/'DIAGNOSIS_DATA.json')['records'];indices={r['id']:r for r in C.read(FIRST.OBS.OUTPUT/'CAL_ROW_INDEX.json')['entries']}
    reach=float(P.DEFAULT_CONFIGURATION['line_configuration']['normal_search_radius_px'])
    OUTPUT.mkdir(parents=True);results=[]
    for item in data:
        ident=item['id'];row=FIRST.OBS._read_single(FIRST.OBS._row_path(FIRST.OBS.OUTPUT,ident),indices[ident])
        n=np.asarray(item['measured_line_normal']);offset=item['offset_px'];k,j=item['native_edge']
        distances=q_distances(n,offset,row['states'],row['initial_geometry']['initial_self_hidden'],row['visible_measurements'])
        by_id={r['native_corner']:r for r in distances};ref=item['selected_chain_ref']['guide_reference']
        anchor=np.asarray(ref['actual_search_anchor_xy']);direction=np.asarray(ref['actual_search_direction_xy']);sn=np.asarray(ref['seed_normal_xy'])
        q=np.asarray(by_id[k]['xy']);model=np.asarray(item['selected_generating_edge'][0]);initial=np.asarray(item['initial_edge'][0])
        for r in distances:r['within_configured_normal_reach_from_TLS']=abs(r['signed_normal_distance_px'])<=reach
        results.append(dict(id=ident,native_edge=[k,j],physical_axis=item['physical_axis'],policies=item['policies'],
            configured_normal_reach_px=reach,source_anchor=k,source_anchor_state=row['states'][k],
            source_q=by_id[k],mate_q=by_id.get(j),all_eligible_BI_q_distances=distances,
            selected_guide_id=item['selected_guide_id'],guide_kind=ref['guide_kind'],
            search_anchor_kind=ref.get('search_anchor_kind','FROZEN_BASE_OBSERVED_BI_ANCHOR'),
            actual_search_anchor_xy=anchor.tolist(),actual_search_direction_xy=direction.tolist(),
            TLS_normal_distance_to_search_anchor_px=float(n@anchor+offset),
            search_anchor_displacement_from_q_px=float(np.linalg.norm(anchor-q)),
            search_seed_normal_displacement_from_q_px=float(sn@(anchor-q)),
            TLS_normal_displacement_search_anchor_from_q_px=float(n@(anchor-q)),
            selected_generating_model_endpoint_displacement_from_q_px=float(np.linalg.norm(model-q)),
            initial_model_endpoint_displacement_from_q_px=float(np.linalg.norm(initial-q)),
            original_observed_BI_reference_xy=ref.get('original_observed_BI_reference_xy'),
            graph_anchor_term_scope='frozen G._score_graph checks B/I displacement only for intersections of >=2 incident selected lines; no single-line q normal gate',
            no_q_or_native_edge_association_changed=True))
        del row
    C.write(OUTPUT/'ASSOCIATION_DATA.json',dict(records=results,all_initial_nonS_BI_queries_only=True,new_GT_images_NN_fits_or_runtime_changes=0))
    text='# v29 CAL 상위 선: 원 B/I q와 탐색 anchor 연결 진단\n\n기존 top3/두정책 합집합을 그대로 사용한다. 이 단계는 정답·이미지 재조회 없이 보관된 실제 TLS 선과 모든 기존 B/I q의 법선 거리를 계산했다. S/U는 좌표 접근 전에 제외했다. nearest q를 골라 native 대응을 바꾸지 않는다.\n\n'
    text+='| 사례/edge | search anchor 종류 | 원 source q→TLS 거리(px) | mate q→TLS(px) | 실제 search anchor→TLS(px) | q→search anchor 2D(px) | search seed 법선방향 q이동(px) | reach(px) |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in results:
        mate='미사용/비BI'if r['mate_q']is None else f'{r["mate_q"]["signed_normal_distance_px"]:.3f}'
        text+=f'| {r["id"]} {r["native_edge"]} | {r["search_anchor_kind"]} | {r["source_q"]["signed_normal_distance_px"]:.3f} | {mate} | {r["TLS_normal_distance_to_search_anchor_px"]:.3f} | {r["search_anchor_displacement_from_q_px"]:.3f} | {r["search_seed_normal_displacement_from_q_px"]:.3f} | {reach:.1f} |\n'
    text+='\n## 모든 원 B/I q에 대한 실제 TLS 법선 거리\n\n| 사례/edge | q ID/상태 | 기존 source | signed 거리(px) | 절대거리 43px 이내 |\n|---|---|---|---:|---|\n'
    for r in results:
        for q in r['all_eligible_BI_q_distances']:
            text+=f'| {r["id"]} {r["native_edge"]} | {q["native_corner"]}/{q["state"]} | {q["source"]} | {q["signed_normal_distance_px"]:.3f} | {q["within_configured_normal_reach_from_TLS"]} |\n'
    text+='\n43px는 각 실제 탐색 strip anchor 주변의 공학적 normal reach다. MODEL_GUIDE_IMAGE_ANCHOR가 원 q에서 멀리 이동한 경우 같은43px라도 원 q 주변 제한이라는 뜻이 아니다. 현재 graph의 q displacement 항은 선택된 incident 선 두 개 이상이 교점을 만들 때만 계산되며, 단독 선의 native-point 연결을 자동 증명하지 않는다. 위 수치는 이 계약을 진단할 뿐 새 q gate/threshold/키포인트나 방법을 구현하지 않는다.\n'
    text+='\n[원 실제 RGB/mask 패널](../CAL_geometry_guided_noise_diagnosis_v4_29/REPORT_KO.md) · [전체 선의 coherent C2 비교](../CAL_geometry_guided_C2_identity_diagnosis_v4_29/REPORT_KO.md). 기존 runtime/관측/모델·CAL 기준·primary 평가는 변경하지 않았다.\n'
    (OUTPUT/'REPORT_KO.md').write_text(text)
    for key in('source_sha256','input_sha256'):D._verify(plan[key])
    C.write(OUTPUT/'DIAGNOSIS_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
        assets={p.name:C.sha(p)for p in OUTPUT.iterdir()if p.is_file()},new_reference_images_NN_fits_or_runtime_changes=0))
    print(dict(records=len(results),seal_sha256=C.sha(OUTPUT/'DIAGNOSIS_SEAL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','render'));args=parser.parse_args()
    declare()if args.stage=='declare'else render()
