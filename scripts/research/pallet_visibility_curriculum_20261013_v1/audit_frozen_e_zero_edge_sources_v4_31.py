"""Append-only source audit: frozen e versus zero-new-edge V31 native.

Reads Python source and two pre-existing source audits only. It never imports
an experimental runtime, decodes a prediction/observation/model/score/image or
evaluates an objective. Function locations and all source bytes are bound.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = Path(__file__).resolve().parent
B = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUTPUT = B / 'frozen_e_zero_edge_native_source_audit_v4_31'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every source audit')
    sources = {}
    evidence = {}

    def at(key, filename, symbol, fact):
        path = S / filename
        text = path.read_text()
        tree = ast.parse(text)
        matches = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol]
        if len(matches) != 1:
            raise ValueError((filename, symbol, len(matches)))
        node = matches[0]
        relative = str(path.relative_to(ROOT))
        sources[relative] = sha(path)
        evidence[key] = dict(path=relative, symbol=symbol,
            first_line=node.lineno, last_line=node.end_lineno, fact_ko=fact)
        return key

    at('cached_e_origin', 'evaluate_fixed_wd_archived_tangent_v4_15.py', '_load',
       'CAL46/TEST49 e는 SELECTOR.ARMS[2], REMAINDER143 e는 R.ARMS[SELECTOR.ARMS[2]]에서 task.baseline으로 보관한다.')
    at('e_two_stage', 'select_frozen_wd_partial_refinement_v4_13e.py', 'select_same_family',
       'old G10.selected_family를 먼저 고정하고 D의 같은 family arm을 deepcopy한다. e는 OLD pose가 아니라 D의 같은-family pose이며 새 fit/GT 선택은 없다.')
    at('e_g10', 'evaluate_global_outline_pose_v4_10.py', '_run',
       '같은 V47 frozen q/partial/U와 전체 mask를 사용한다. 각 WD는 이전 WD solution 및 P1 solution/운반 seed에서 G10 연속 refinement하고 joint objective로 family를 고른다.')
    at('e_d', 'evaluate_face_partial_BI_outline_v4_13d.py', '_run',
       'legacy boundary와 typed B/I boundary를 corner ID로 병합한다. typed가 없으면 old G10 전체를 그대로 재사용하고 있으면 G13으로 같은 G10 caller를 실행한다.')
    at('e_d_observe', 'evaluate_face_partial_BI_outline_v4_13d.py', '_observe',
       '이미 보정된 frozen_BI_fit과 최초 gated raw prediction/initial geometry로 typed partial 관측을 만든다. q/cov archives를 변경하지 않는다.')
    at('e_remainder', 'evaluate_remainder_frozen_wd_v4_14.py', '_run',
       '추가143은 fresh P1 seed→WD prior0 search→G10→D→같은-family e 규칙을 직접 실행한다. 원95 pose를 대체 재사용하지 않는다.')
    at('upstream', 'statewise_joint_experiment_v2.py', '_frames',
       '주 코호트 raw_points/initial_pose는 PREDICTION_INPUTS의 points.N3_SUBPIX / initial_poses.N3_SUBPIX를 그대로 받는다. 이것은 YOLO head raw와 동일하다는 뜻이 아니다.')
    at('observe', 'statewise_joint_api_v4.py', 'observe',
       'registered initial geometry/S gate→mask/state→nonS B/I image correction→고정 visible measurement→OT domains 순서이다. 최초 geometry를 보정 후 재분류하지 않는다.')
    at('fixed_cov', 'boundary_partial_observations_v4_4.py', 'fixed_fit_measurements',
       'recovered q와 source/conditional covariance는 correction에서 받고 fit_covariance/fit_reliability는 original calibrated measurement에서 그대로 덮어쓴다.')
    at('recover_v47', 'evaluate_shared_edge_partial_b_v4_7.py', '_run',
       'old correction의 실패 B만 다시 recovery하며 이미 성공한 B와 모든 I는 그대로 둔다. fixed CAL fit scales와 shared legacy B normals/U를 만든다.')
    at('recover_reach', 'visible_bi_corner_reach_v4_7.py', '_working_configuration',
       '기존 normal search radius의 하한을 along_end+boundary_context로 넓힌다. 새 PnP나 보정성공 증명은 아니다.')
    at('raw_bi', 'visible_bi_correction_v2.py', 'correct_visible_bi',
       'B/I만 실제 image line/intersection을 측정하며 성공 source와 VISIBLE_BACKBONE_FALLBACK을 구분한다. S/O/T/U는 per-point correction을 받지 않는다.')
    at('remainder_q', 'observe_remainder_rgb_v4_14.py', 'observe_remainder_native_frame_v4_14',
       'fresh143도 N3_SUBPIX geometry/state→tight/full targetmask→CAL measurement scales→동일 recovery→fixed_fit_measurements→legacy partial/U 순서를 따른다.')
    at('d_loss', 'statewise_partial_BI_u_solver_v4_13.py', '_joint_objective_pair',
       'boundary ID의 full2D q비용을 제거하고 fixed reliability partial normal 비용을 더한다. e 단계에는 archived tangent point cost가 없고 tangent는 hard range뿐이다.')
    at('d_guards', 'statewise_partial_BI_u_solver_v4_13.py', '_joint_constraint_pair',
       '전체 measurements와 OT/U, original boundary tangent bounds를 함께 검사한다. 비용에서 q를 없애도 measurement guard metadata는 보존된다.')
    at('soft_guard', 'statewise_uncertain_solver_v4_bias.py', '_constraints',
       'generic constraint의 visible residual-budget rows를 명시 제거한다. maximum_visible_mahalanobis 설정명만 보고 이 경로에 hard q residual cap이 있다고 주장하면 안 된다.')
    at('old_mask', 'statewise_global_outline_solver_v4_10.py', '_outline_value',
       'projected convex hull↔observed convex-envelope segments 양방향 robust distance를 .5씩 평균한다. mask가 지원되면 lambda1로 연속 최적화하며 states/occluder eligibility 입력은 없다.')
    at('d_solver', 'statewise_partial_BI_outline_solver_v4_13.py', 'refine_global_outline_joint_pose',
       'typed revocation이 없으면 G10으로 exact delegate한다. typed가 있으면 partial normal+동일 wholemask를 optimize한다. prior_weight는0, old full-BI/WD/P1은 seed뿐이다.')
    at('new_native', 'prepare_geometry_guided_native_evidence_v4_30.py', 'native_baseline',
       'current task spec/grazing states/filtered visible/merged boundary/U/domains로 empty edge NativeEdgeProblem(prior0)을 만든 뒤 V26 prepartial proxy를 붙인다.')
    at('new_clone', 'prepare_geometry_guided_native_evidence_v4_31.py', '_parent_function',
       'V31 native_baseline은 위 V30 함수를 G31에만 private binding한 복사이며 새로운 e evaluator가 아니다.')
    at('native_old_parts', 'native_edge_problem_v4_17.py', 'objective',
       'complete BI+remaining old partial normals+native edges+archived tangent T4이다. empty edge에서도 old normal/T4는 남고 wholemask/coarse prior는0이다.')
    at('restore', 'prepartial_RGB_point_preservation_v4_26.py', 'prepare_prepartial_native_problem',
       'partial ID 중 exact VISIBLE_BACKBONE_FALLBACK 이외의 RGB/unknown archive만 exact complete2D로 복구한다. fallback partial은 기존 normal+T4로 남긴다.')
    at('restore_proxy', 'prepartial_RGB_point_preservation_v4_26.py', 'PrepartialRGBPointPreservingNativeProblem',
       '복구한 q는 original q/source/observedSigma/effectiveSigma/reliability를 유지하고 그 owner의 old normal/T4 비용만 제거한다. 원래 boundary guard는 delegate한다.')
    at('grazing', 'evaluate_grazing_native_candidates_v4_18.py', 'adapt_cached_task',
       '최초 model geometry grazing band의 B/I를 U로 바꾸고 visible/partial을 filter하며 U region을 추가한다. task.baseline은 바꾸지 않아 e와 current states/eligible q가 다를 수 있다.')
    at('o_domain', 'o_localization_abstention_v4_19.py', 'abstain_external_o_localization',
       'current O ellipse hard localization을 closed-canvas box+원 extra halfplanes로 바꾸며 O state/T domains는 유지한다. e의 원 hard O ellipse와 같지 않다.')
    at('new_policy', 'geometry_guided_edge_problem_v4_29.py', 'plan_geometry_guided_replacement',
       'active0이면 owners0이고 no-new-owner V26 measurement state를 보존한다. 새 gamma/line noise가 이 경우 비용을 다시 e로 만들지 않는다.')
    at('new_parts', 'geometry_guided_edge_problem_v4_29.py', 'objective',
       'base.objective에 활성 actual edge와 owner actual-line-axis tangent만 더한다. 그룹/owner0이면 이 추가항은0이다.')
    at('new_guards', 'geometry_guided_edge_problem_v4_29.py', 'constraints',
       'owner 없는 original boundary guards와 actual-support guards를 더한다. active0이면 actual-support 추가행이 없으며 source baseline의 OT/U/depth/geometry/canvas/partial guards는 남는다.')
    at('new_g31', 'evaluate_geometry_guided_edges_v4_31.py', '_problem',
       'typed G31 model 또는 explicit unavailable empty evidence를 사용한다. reference_start_problem은 frozen old native start ranking 전용이며 objective 대체가 아니다.')
    at('priors_input', 'evaluate_bounded_geometric_priors_v4_22.py', '_prior_inputs',
       'archived original O domain으로 bounded O regularizer를 준비하고 ORIGINAL first coarse geometry/S로 bounded S model prior를 준비한다. e numeric pose는 target으로 쓰지 않는다.')
    at('os', 'bounded_geometric_priors_problem_v4_22.py', 'BoundedGeometricPriorsProblem',
       'native에 dependent bounded O/S model regularizers를 별도 장부로 더한다. 새 S image point는 없고 hardguard를 새로 추가하지 않는다.')
    at('containment', 'global_foreground_containment_v4_25.py', 'outline_value_and_gradient',
       'C=.5 mean rho(outside distance), observed sample이 full model hull 내부면0/model→observed항0이다. 옛 symmetric wholemask와 다른 함수이다.')
    at('new_run', 'evaluate_geometry_guided_edges_v4_29.py', '_run',
       'baseline e는 deepcopy만 한다. 각 새 policy는 양 WD에서 native+boundedO/S 또는 +C를 refine/재평가하고 현재 총비용으로 WD/pose를 새로 고른다. full/TP/topnormal prior는0이다.')
    at('new_selector', 'evaluate_geometry_guided_edges_v4_29.py', 'select_candidates',
       'fresh-feasible 후보를 전부 인증한 뒤 native+O/S+lambdaC scalar minimum, arithmetic ties는 입력순서. old wholemask/tie score를 쓰지 않는다.')
    at('starts', 'refine_native_edge_pose_v4_17.py', 'choose_native_starts',
       '고정 native start problem에서 fresh feasible objective와 pose diversity로 최대6개+own coarse를 선택한다. 이 유한 탐색은 e의 단계별 full-search seed schedule과 같지 않다.')
    at('legacy_bank', 'evaluate_cached_native_edges_v4_17.py', 'collect_proposal_bank',
       '고정 bank는 seed/G10/D/normal/tangent의 선언된 saved solutions를 전부 수집한다. tangent의 SAME_FROZEN_E arm도 반복문 범위에 들어가므로 historical e-equivalent pose가 bank에 없다고 단정할 수 없다. 현재 e를 새로 주입하는 규칙은 없다.')
    at('wd', 'statewise_two_wd_family_v4_9.py', 'make_wd_families',
       'ORIGINAL/SWAPPED는 같은 unordered physical box를 유지하지만 native channel별対応가 다르다. physical R과 Q를 운반해야 하며 t/rvec만 바꿔 끼우면 같은 관측가설이 아니다.')
    # Upstream source outside this experiment: read Python only.
    upstream = ROOT / 'scripts/research/pallet_n3_subpix_20261008_v1/runtime.py'
    sources[str(upstream.relative_to(ROOT))] = sha(upstream)
    tree = ast.parse(upstream.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pipeline')
    evidence['backbone'] = dict(path=str(upstream.relative_to(ROOT)), symbol='Pipeline',
        first_line=cls.lineno, last_line=cls.end_lineno,
        fact_ko='upstream runtime는 YOLO26 pose checkpoint→공유 neck의 frozen N3_DIM_SYM seed1 step6000→SubPix→BASE q0 기준 image diagonal1% total cap→original prediction-only F를 사용한다. visibility 단계의 N3_SUBPIX는 순수 최초 head q가 아니다.')
    reused_paths = [B/'prepartial_RGB_source_audit_v4_26/AUDIT.json',
                    B/'prepartial_RGB_source_audit_v4_26/PATH_CLOSURE_V1.json']
    reused = {str(p.relative_to(ROOT)): sha(p) for p in reused_paths}
    old = json.loads(reused_paths[0].read_text())
    old2 = json.loads(reused_paths[1].read_text())
    # Reuse source findings only, not old partial_records/counts/metrics.
    reused_findings = old['source_evidence']
    assert old['source_only'] is True and old2['new_observation_or_prediction_journal_reads'] == 0
    columns = [
      ('identity', '수치의 정체', 'e는 v13e가 보존한 D same-family 결과; 현재 arm에서 재계산하지 않음.', '현재 입력/현재 비용/현재 guard로 새로 선택한 native 결과.', ['cached_e_origin','e_two_stage','new_run']),
      ('q', 'q·source·공분산', 'V47 RGB 보정 q + 원 CAL fit covariance/reliability. D가 typed partial로 바꾼 q는 archive만 남고 비용에서 제거.', '남아 있는 nonfallback partial q를 V26이 exact 2D 비용으로 복구; fallback partial은 normal+T4. 저장 q/Σ 자체를 새로 보정하지 않음.', ['fixed_cov','e_d','d_loss','restore','restore_proxy']),
      ('state', '상태·가시점 집합', '원 e의 고정 state/BI 집합.', 'v18 grazing BI→U 후 filtered visible/partial+추가 U region. S는 최초 gate 고정.', ['grazing','new_native']),
      ('partial', 'old partial 비용', 'full q 대신 normal만 soft 비용; tangent는 hard interval, archived tangent point cost0.', 'exact fallback partial에는 normal과 archived tangent variance4 비용이 함께 남음. nonfallback restored에는 둘 다 비용 제거.', ['d_loss','native_old_parts','restore_proxy']),
      ('guards', 'hard guards', '원 OT hard domains/U/visible geometry·canvas/depth/partial tangent range. soft BI 경로는 residual-budget row를 제거.', 'current grazing/U·O box domain과 original partial bounds. active0이면 new support guard는0이지만 모든 legacy guard가 사라지는 것은 아님.', ['d_guards','soft_guard','o_domain','new_guards']),
      ('o', 'O 항', '원 localized O ellipse 등 hard localization. 별도 bounded O soft term 없음.', 'v19 O hard ellipse 제거 후 canvas/extra planes 유지 + bounded O soft regularizer.', ['o_domain','priors_input','os']),
      ('s', 'S 항', 'S local image/point0, 별도 bounded S model regularizer 없음.', 'S local image/point0 유지, first coarse에 대한 bounded S visibility model regularizer는 남음.', ['old_mask','priors_input','os']),
      ('prior', 'coarse/TP/full/top-normal prior', 'G10/D 최종 prior_weight0. 이전 P1 posture는 search seed로만 이용.', 'FULL/TP/top-normal0; bounded O/S는 별도 dependent regularizer. first coarse도 start로 남음.', ['e_g10','d_solver','new_run']),
      ('shape', 'wholemask 대 C1', '지원되는 mask: 양방향 hull↔boundary symmetric loss를 λ1로 연속 최적화.', 'native arm: wholemask/C0. C arm: outside-only observed⊂full-model C1, inside0/forward0. active edge0과 C0은 다른 조건.', ['old_mask','containment','new_run']),
      ('starts', '탐색·초기값', 'WD0/P1/G10 saved winners/samples에서 G10/D 연속 refinement; typed없음은 G10 cache exact alias.', '고정 v17 bank의 모든 feasible cached 후보 유지 + frozen v19 native 최대6 diverse starts와 own coarse. C는 같은 policy native endpoints도 보존.', ['e_g10','e_d','d_solver','starts','new_run','legacy_bank']),
      ('wd', 'WD 선택', 'G10 complete/legacy+wholemask 목적함수로 family 고정 후 D의 그 family 결과를 읽음.', '양 WD의 현재 native+O/S(+C)로 직접 재선택. e-family lock arm은 현재5arm 정의에 없음.', ['e_two_stage','new_run','new_selector','wd']),
      ('fallback', '실패·fallback', 'D same-family fallback과 ambiguity를 그대로 보존; 다른 family나 old pose로 성능 선택하지 않음.', '새 현재-feasible 후보가 없으면 own ORIGINAL initial coarse를 accepted=False로 반환. e의 숫자를 fallback으로 새로 삽입하지 않음.', ['e_two_stage','new_run']),
    ]
    comparison = [dict(key=k, topic_ko=t, frozen_e_ko=e, zero_new_edge_native_ko=n, evidence=refs)
                  for k,t,e,n,refs in columns]
    report = dict(schema='FROZEN_E_VS_ZERO_NEW_EDGE_NATIVE_SOURCE_AUDIT_V31',
        created_utc=datetime.now(timezone.utc).isoformat(), source_only=True,
        conclusion_ko='활성 actual edge가 0이라는 조건은 G31 추가 line/tangent 항이 0이라는 뜻뿐이다. e를 생성했던 관측 상태·partial 비용·hard domains·wholemask·단계별 WD·start schedule과 동일하다는 뜻이 아니다.',
        equations=dict(e_family='f_e = argmin_f min_searched_pose J_G10_f(complete_except_legacy + legacy_normal + symmetric_wholemask); stable ORIGINAL exact ties',
            e_pose='same f_e: D typed partial normal + remaining complete q + symmetric_wholemask, original OT/U/partial guards; no T4 point cost; typed0 delegates G10',
            new_native_zero='J = J_V26_complete_restoration_and_remaining_fallback_normal_T4 + bounded_O + bounded_S',
            new_containment_zero='J = J_new_native_zero + lambda_C*C_outside_only; lambda_C=1 when supported else0',
            new_active_edge_additions='zero active_groups => zero new_owner_ids => no new line factor or new actual-axis tangent; new CAL line sigma/gamma do not by themselves contribute'),
        comparison=comparison, evidence=evidence,
        ancestry_ko=[
            'frozen upstream YOLO→N3_DIM_SYM→SubPix/cap→stored initial PnP/known geometry',
            'initial geometry and S gate→targetmask/state→B/I actual RGB correction→CAL effective covariance/reliability',
            'fallback B recovery v4.3/v4.3b/v4.7; legacy B partial/U→prior1 search seeds→WD prior0 search',
            'G10 wholemask continuous two-WD search/family selection→D typed partial normal refinement→e freezes G10 family and reads D same-family pose',
            'e copied as immutable task.baseline→v15/v17 bank→v18 grazing→v19 O hard-domain abstention→v22 boundedO/S→v25 inputs→G31 native baseline V26 nonfallback q restoration'],
        objective_reuse=dict(feasible=True, implemented=False, numeric_e_pose_insertion_permitted=False,
            closest_reusable_primitives=['statewise_partial_BI_u_solver_v4_13._joint_objective_pair',
                'statewise_partial_BI_u_solver_v4_13._joint_constraint_pair',
                'statewise_partial_BI_outline_solver_v4_13._prepare',
                'statewise_partial_BI_outline_solver_v4_13.outline_value_and_gradient',
                'statewise_global_outline_solver_v4_10._prepare',
                'statewise_global_outline_solver_v4_10.outline_value_and_gradient'],
            explanation_ko='예측숫자 e를 사용하지 않고 원 D loss/guard builder와 wholemask 함수를 새 uniform evaluator에서 재사용할 수 있다. 단 현 V26 q복원·grazing·O box·T4·boundedO/S/C를 유지하면 같은 함수 호출이더라도 e와 같은 objective가 아니다. 원 input profile로 명시 복귀하거나 차이를 별도 대조로 선언해야 한다.',
            staged_WD_caveat_ko='e는 하나의 최종 scalar objective로 두 WD를 동시에 최소화한 방법이 아니다. D objective만 양WD에 적용하면 다른 방법이다. e 절차 자체를 재현하려면 G10 family selection→D within-family의 두 단계를 동일하게 계산해야 한다.',
            search_parity_caveat_ko='목적함수 재사용만으로 e 숫자 자세의 exact 재현/좋은 성능/전역최적이 보장되지 않는다. 동일 q/source/Σ/weights/state/domain/boundary/mask/profile, same Q/native association와 optimizer/start/ties/fallback까지 고정해야 결과 parity를 논할 수 있다.',
            inherited_bank_caveat_ko='현재 선언된 v17 bank는 이미 역사적 e arm solution도 일반 수집 대상으로 포함한다. 이번 감사는 e를 새 prior/start/candidate/fallback으로 삽입하지 않았고, future uniform comparison도 별도의 numeric-e injection을 추가하면 안 된다.'),
        reused_source_audits=reused, reused_source_findings=reused_findings,
        source_sha256=sources, limitations_ko=[
            '소스 경로·수식·입력 정책 감사이며 새 V31 frame의 active0 수/실제 비용/성능을 계산하지 않았다.',
            '개별 frame에서 어떤 차이가 주원인인지 또는 이 함수 복귀가 정확도를 회복할지는 소스만으로 결정할 수 없다.',
            'line 지원0 또는 불확실성은 full q/자세 유일성의 증거가 아니다.',
            'S/U 원좌표·새 RGB·GT/annotation·CAL 숫자·primary rows·prediction/score journal/model numeric decode0; optimizer/evaluate/posefit0.'],
        reads=dict(Python_source_only=True, reused_existing_source_audit_fields_only=True,
            actual_rows=0, score_or_prediction_journals=0, numeric_models=0, labels=0,
            RGB_or_mask_arrays=0, source_edits_to_frozen_modules=0),
        checks=dict(referenced_symbols=len(evidence),source_files=len(sources),
            comparison_rows=len(comparison), reused_reports=len(reused),all_sources_unchanged_after_write=None))
    self_path=Path(__file__).resolve();sources[str(self_path.relative_to(ROOT))]=sha(self_path)
    report['checks']['source_files']=len(sources)
    OUTPUT.mkdir(parents=True)
    def write(name,value):
        (OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def link(ref):
        item=evidence[ref];path=ROOT/item['path'];rel=os.path.relpath(path,OUTPUT)
        return f"[{path.name}:{item['first_line']}]({rel}#L{item['first_line']})"
    md=['# e와 새 활성선 0개 native 경로의 차이','',report['conclusion_ko'],'',
        'e는 초기 backbone/PnP 결과의 이름이 아닙니다. G10으로 WD를 고른 다음 D의 같은 WD 결과를 보존한 후속 단계입니다. 현재 arm의 활성선 수가 0이어도 아래 차이는 남습니다.','',
        '| 항목 | 보존된 e | 새 활성선 0개 경로 | 소스 |','|---|---|---|---|']
    for row in comparison:
        md.append('| '+ ' | '.join([row['topic_ko'],row['frozen_e_ko'],row['zero_new_edge_native_ko'],', '.join(link(k)for k in row['evidence'])])+' |')
    md+=['','## 초기 backbone → B/I 보정 → PnP 출처','']
    md += [f'{i+1}. {text}'for i,text in enumerate(report['ancestry_ko'])]
    md += ['', '초기 `N3_SUBPIX`는 순수 최초 YOLO 점이 아닙니다. 상위 frozen N3와 SubPix 및 전체 이동 제한을 거친 점과 저장 PnP입니다. 이후 visibility 계층이 S를 먼저 제외하고 B/I를 측정합니다. 보정된 q의 관측 공분산과 CAL 유효 fit 공분산은 구분되며, recovery가 q를 바꿔도 원 CAL fit 공분산·신뢰도는 고정합니다.',
        '',', '.join(link(k)for k in ['backbone','upstream','observe','raw_bi','recover_v47','fixed_cov','remainder_q']),
        '', '## e 비용 함수를 uniform branch에 쓸 수 있는가','',report['objective_reuse']['explanation_ko'],'',
        report['objective_reuse']['staged_WD_caveat_ko'],'',report['objective_reuse']['search_parity_caveat_ko'],'',
        report['objective_reuse']['inherited_bank_caveat_ko'],'','## 범위','']
    md += ['- '+s for s in report['limitations_ko']]
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(md)+'\n')
    assert all(sha(ROOT/p)==digest for p,digest in sources.items())
    assert all(sha(ROOT/p)==digest for p,digest in reused.items())
    report['checks']['all_sources_unchanged_after_write']=True
    write('AUDIT_V1.json',report)
    write('AUDIT_SEAL_V1.json',dict(schema='SOURCE_ONLY_E_ZERO_EDGE_AUDIT_SEAL_V31',
        files={n:sha(OUTPUT/n)for n in ('AUDIT_V1.json','REPORT_KO.md')},
        source_sha256=sources,reused_source_audits=reused,actual_data_decode=0,
        frozen_source_changes=0,method_or_default_changed=False))
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)),
        source_files=len(sources),evidence=len(evidence),comparison_rows=len(comparison),
        audit_sha256=sha(OUTPUT/'AUDIT_V1.json'),seal_sha256=sha(OUTPUT/'AUDIT_SEAL_V1.json'))))


if __name__ == '__main__':
    main()
