"""Complete A/B handoff only after actual training, inference, statistics and poses."""
import io,subprocess,unittest
import numpy as np
import env as E
from a_train import verify_binding

def preserve(relative):
    path=E.DOC/relative;dst=E.DOC/'A/history'/('PRE_CLOSEOUT_'+relative.replace('/','__'))
    if path.exists() and not dst.exists():
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())

def avg(s,arm,key):
    return float(np.mean([s[f'{arm}_seed{i}'][key] for i in [1,2,3]]))

def main():
    train=E.read(E.DOC/'A/training_audit.json');assert train['main_fits']==12 and train['main_updates']==24000
    assert train['same_trainable_parameters'] and train['paired_sample_order_exact']
    data=E.read(E.DOC/'A/results_and_intervals.json');pose=E.read(E.DOC/'A/POSE_SECONDARY.json')
    impact=E.read(E.DOC/'A/MATCH_AND_GEOMETRY_DECOMPOSITION.json')['results']
    inference=E.read(E.DOC/'A/INFERENCE_COMPLETE.json')
    assert data['status'] in ['2D_COMPLETE_POSE_PENDING','COMPLETE'] and pose['complete'] and inference['complete']
    assert inference['new_detector_forward_images']==31458
    verify_binding(E.read(E.DOC/'SOURCE_BINDING.json')['files']);verify_binding(E.read(E.DOC/'A/JOINT_TRAINING_LOCK.json')['bindings'])
    verify_binding(E.read(E.DOC/'A/EVALUATION_LOCK.json')['weights']);verify_binding(inference['prediction_files'])
    assert E.sha(E.HERE/'audit_math.py')=='d4526e0babe44bd31c7df02cfa9a697c57b7e78e8356b34849666e4baf44f546'
    assert E.sha(E.HERE/'test_audit_math.py')=='6c8bab79a2ab81503baece2efe1588fe66b34e76566da60d91621b31d2e594e3'
    assert E.read(E.DOC/'B/results_and_intervals.json')['status']=='COMPLETE_FROZEN_B'
    assert E.read(E.DOC/'B/POSE_SECONDARY.json')['complete'] and E.read(E.DOC/'B/runtime.json')['complete']
    orientation=E.read(E.DOC/'A/SOURCE_ORIENTATION_CORRECTION.json')
    assert orientation['status']=='POST_EVALUATION_COORDINATE_CORRECTION'
    assert orientation['centroid_body_IoU_and_primary_2D_unchanged']
    assert orientation['A_nonrotation_and_DEV_invariance_verified_frames']==31458
    assert 'source_orientation_correction' in pose
    assert E.read(E.DOC/'B/POSE_SOURCE_ROTATION_CORRECTION.json')['status']=='CORRECTED_SECONDARY_ONLY'
    out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_*.py'))
    assert result.wasSuccessful()
    for name in ['REGRESSION_TESTS.json','FINAL_AUDIT.json','RUN_MANIFEST.json','FINAL_DECISION_KO.md','A/REPORT_KO.md']:
        preserve(name)
    E.write(E.DOC/'REGRESSION_TESTS.json',dict(cpu_tests=result.testsRun,cpu_PASS=True,log=out.getvalue(),
      A_identity=E.bound(E.DOC/'A/IDENTITY_AND_RLE_AUDIT.json'),A_joint_wiring=E.bound(E.DOC/'A/JOINT_WIRING_AUDIT.json'),
      B_actual_cache=E.bound(E.DOC/'B/ADAPTER_TESTS.json'),A_main_training_completed=True))
    data.update(status='COMPLETE',pose=E.bound(E.DOC/'A/POSE_SECONDARY.json'),definition_integrity='PASS',EQUIV_wiring='PASS')
    E.write(E.DOC/'A/results_and_intervals.json',data)
    labels={'RECT_SYNTH_VAL':'RECT 합성 val','RECT_DEV':'RECT 실사 DEV','SQUARE_DEV':'SQUARE 실사 DEV'}
    lines=['# A 완료 — 대칭 정의와 성능 효과의 분리','',
      '12 fits × 2,000 actual updates = **24,000 main updates** 완료. Smoke는 별도 12 updates이며 R0에서 본학습을 다시 초기화했다.',
      '초기 R0 및 각 cohort의 INDEXED/EQUIV seed1/2/3을 마지막 raw weights로만 평가했다. 새 A 모델에 기존 P 또는 DHT를 붙이지 않았다.','',
      '## 목적함수와 무결성','',
      '사용자가 승인한 **공동 대칭 조합 최소화**를 사용했다. 물체별 독립 min의 원래 제안과 동일하다고 주장하지 않는다. 두 head의 위치·visibility·batch-global RLE clamp와 stock reduction을 보존하고, GT 객체당 하나의 전체 tuple 순열을 두 head가 공유한다.',
      '전체 손실 최적 조합은 선형 하한 인증 또는 exact one-hot MILP로 선택하고, 선택된 target의 실제 loss/gradient는 stock 코드로 계산했다. 원본 라이브러리·학습 source 파일은 바꾸지 않았다.',
      f'CPU 회귀검사 {result.testsRun}/{result.testsRun} PASS. 실제 GPU identity 및 C4/C2·누락점·GT 행 정렬·complete-loss/gradient 검사를 통과했다. `definition_integrity=PASS`는 아래 성능 개선의 증명이 아니다.',
      'R0 초기화·샘플 순서·전체 trainable parameter set·AdamW 1e-4·batch16·FP32·clip10·BN 통계 고정 조건을 검증했다. AMP/EMA/TF32/증강은 없고, 두 head 가중치는 양 군 0.8/0.2로 일정하다.','',
      '## 2D 결과','',
      'R0는 단일 값, INDEXED/EQUIV는 seed별 지표의 3-seed 평균이다. 아래 corner8/full-denominator 지표는 기존 논문의 matched9 표를 대체하지 않는다.','',
      '| 모집단 | 모델 | E_sym ↓ | E_fixed ↓ | sym pooled median px | sym pooled P90 px | 검출 매칭률 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for split,s in data['summary'].items():
        for arm in ['R0','INDEXED','EQUIV']:
            value=lambda k:s['R0'][k] if arm=='R0' else avg(s,arm,k)
            lines.append(f"| {labels[split]} | {arm} | {value('primary_E_sym'):.8f} | {value('E_fixed'):.8f} | {value('pooled_point_median_px'):.4f} | {value('pooled_point_P90_px'):.4f} | {value('detection_coverage'):.4f} |")
    lines+=['','## 주 비교와 판단','']
    for split,c in data['contrasts'].items():
        p=c['EQUIV_minus_INDEXED'];s=data['summary'][split]['R0']
        lines.append(f"- **{labels[split]}: {c['geometry_gain']}**. EQUIV−INDEXED Δ={p['delta']:.9g}, 95% CI [{p['CI95'][0]:.9g}, {p['CI95'][1]:.9g}], 개선 seed {p['improved_seeds']}/3. 예측 {s['total_frames']}장, 주지표 {p['frames']}장, bootstrap {p['level']} 단위 {p['units']}개.")
        lines.append(f"  corner 무주석 비평가 프레임은 {len(s['non_evaluable_frame_ids'])}장이다(해당 ID 목록 보존). 검출 실패는 제거하지 않고 raw diagonal penalty를 유지했다.")
    lines+=['','CI가 0을 포함하는 결과는 동등성/비열등성 증명이 아니다. 합성 frame bootstrap은 실제 시나리오 의존성을 과소평가할 수 있다. 실사는 재사용 DEV이며 square는 같은 세션의 interleave split이다. 세 모집단을 합친 단일 효과나 독립 confirmatory 결과를 주장하지 않는다.',
      'C1/C2/C4별, fixed/sym, PCK5/10/20, true-collapse, center 오차와 branch 분포는 `results_and_intervals.json`에 별도 보존했다. R0 대비 비교는 부 비교이며 primary는 EQUIV−INDEXED다.','',
      '## 검출 변화와 점 위치 변화의 분해','',
      '주지표를 바꾸지 않고, 전체 분모에서 두 모델 모두 매칭/한쪽만 매칭/둘 다 실패의 기여를 분리했다. 두 모델 모두 매칭된 부분집합은 사후 설명용이며 primary를 대체하지 않는다.']
    for split,s in impact.items():
        common=np.mean([s[str(i)]['both_matched']['conditional_delta_frame_mean_px'] for i in [1,2,3]])
        recovered=[s[str(i)]['INDEXED_miss_EQUIV_match']['frames'] for i in [1,2,3]]
        lost=[s[str(i)]['INDEXED_match_EQUIV_miss']['frames'] for i in [1,2,3]]
        lines.append(f'- {labels[split]}: common-matched frame 평균 오차 차이의 seed 평균 {common:.5f}px, EQUIV만 매칭 {recovered}장 / INDEXED만 매칭 {lost}장. 기여 합산 검산은 `MATCH_AND_GEOMETRY_DECOMPOSITION.json`에 있다.')
    lines+=['',
      '## 손해와 6D 보조 결과','',
      '| 모집단 | INDEXED 중심 이동 median cm | EQUIV 중심 이동 median cm | INDEXED 10cm·10deg 성공률 | EQUIV 10cm·10deg 성공률 |',
      '|---|---:|---:|---:|---:|']
    for split,s in pose['summary'].items():
        center=lambda arm:np.mean([s[f'{arm}_seed{i}']['centroid_translation_cm']['median'] for i in [1,2,3]])
        rate=lambda arm:np.mean([s[f'{arm}_seed{i}']['pose_10cm_10deg_full_reference_rate'] for i in [1,2,3]])
        lines.append(f'| {labels[split]} | {center("INDEXED"):.4f} | {center("EQUIV"):.4f} | {rate("INDEXED"):.4f} | {rate("EQUIV"):.4f} |')
    lines+=['']
    for split,damage in data['damage'].items():
        bad=[damage[str(i)]['harmed_frames'] for i in [1,2,3]];points=[damage[str(i)]['good5_to_bad10'] for i in [1,2,3]]
        lines.append(f'- {labels[split]}: frame 평균 오차 악화 {bad}장, INDEXED <5px → EQUIV >10px 점 {points}개(seed1/2/3).')
    lines+=['',
      '6D는 RECT에서 동일 prediction-only 축 선택 + SQPnP/RefineLM이다. SQUARE는 x=z인 동일 W/D 기하 가설을 하나의 대표로 합친 후 같은 solver를 쓴다. 그렇지 않으면 기존 직사각형 selector가 완전 동률을 모호성 실패로 처리하기 때문이다. 이 처리와 C4 회전 fixture는 A pose 성능 열람 전에 고정했다. 예측을 GT bbox match로 gate하지 않는다. median/P90은 유효 pose 쌍에 조건부이고 coverage·성공률은 전체 reference 분모로 보고한다. 물리 축의 전역 C1/C2/C4만 허용하며 unrestricted nearest-neighbour ADD-S로 바꾸지 않았다.',
      '실사 DEV GT와 square reference는 geometry-reconstructed이며 독립 계측이 아니다. Square는 정정된 annotated corners와 등록된 1.1×0.15×1.1m cuboid로 별도 기준 pose를 구했다. 원 annotation의 저장 pose를 번호·원점 확인 없이 재사용하지 않았다. Source C1에서 camera-facing label만으로 물리적 앞면을 복구할 수 없다는 한계도 유지한다.',
      '합성 전체 val의 G38__G__f11070 한 장은 CF index basis의 determinant가 −1이었다. 기존 matched-subset용 assertion이 A pose 추론 전에 이를 검출했다. Source body GT는 저장된 proper renderer R·physical extents로 직접 표현해 같은 박스 부피를 유지했다. 프레임을 빼거나 reflection을 허용 그룹에 넣지 않았고 학습/예측도 재실행하지 않았다. `SOURCE_POSE_FRAME_AUDIT.json`에 정정 경위를 보존한다.',
      'A에서 음성 2,689장 검출을 새로 재검증한 것은 아니므로 false-positive 개선/안전성을 주장하지 않는다.','',
      '합성 회전 보조 지표를 열람한 뒤, renderer의 +Y up과 PnP의 +Y down 사이 고정 proper basis 변환 누락을 발견했다. 모든 합성 예측에 동일하게 R_source = R_previous × diag(1,−1,−1)을 적용해 회전만 재계산했다. 사전 등록 변경으로 가장하지 않으며 GT별 위상 선택이나 학습·추론 재실행은 없다. 정정 전 A 수치와 B 원본 전체는 보존했다. B의 합성 회전 보조 지표만 `../B/POSE_SOURCE_ROTATION_CORRECTION.json`으로 정정한다. 중심·body IoU·2D·DEV 결과는 불변이다. 자세한 증거는 `SOURCE_ORIENTATION_CORRECTION.json`에 있다.','',
      '## 데이터·기록·다음 판단','',
      '합성 train55,980과 square train696은 분리했다. Square 기존 prepared label 2행의 미표시 점 오류만 별도 target view로 정정했고 두 군이 같은 view를 사용했다. 원본 label·영상·기존 A0·B·논문 결과는 보존했다.',
      '실제 새 평가 forward는 **31,458 images**(RECT 합성4020+DEV319, SQUARE155 각각7모델)다. 학습 forward384,000, 고정 train probe768, smoke192, 이전 A0/배선 감사는 별도 기록이다.',
      '수학적으로 올바른 task-equivalent 정의와 실측 gain을 분리해 판단한다. 이번 고정 데이터·초기화·2,000-step 조건 밖으로 일반화하지 않으며, 결과를 이유로 새 seed/계수/학습량 탐색을 자동 추가하지 않는다.','']
    (E.DOC/'A/REPORT_KO.md').write_text('\n'.join(lines))
    run=E.read(E.DOC/'RUN_MANIFEST.json');run.update(status='COMPLETE_A_AND_B')
    run['A']=dict(status='COMPLETE',main_fits=12,main_updates=24000,smoke_updates=12,training_forward_images=384000,probe_forward_images=768,
      smoke_forward_images=192,identity_detector_forward_images=32,joint_wiring_detector_forward_images=2,
      final_evaluation_forward_images=31458,pose_frame_evaluations=pose['expected_pose_frame_evaluations'],
      pose_scoring_frame_evaluations_including_CPU_replay=59598,pose_neural_inference_reruns=0,
      orientation_only_rescored_frames=orientation['A_source_unique_frames']+orientation['B_source_unique_frames'],
      orientation_correction_new_PnP_calls=0,orientation_correction=E.bound(E.DOC/'A/SOURCE_ORIENTATION_CORRECTION.json'),
      pose_execution_corrections=E.bound(E.DOC/'A/POSE_EXECUTION_CORRECTIONS.json'),
      training_audit=E.bound(E.DOC/'A/training_audit.json'),result=E.bound(E.DOC/'A/results_and_intervals.json'))
    E.write(E.DOC/'RUN_MANIFEST.json',run)
    assert subprocess.check_output(['git','diff','--name-only','--','_docs/paper'],text=True).strip()==''
    E.write(E.DOC/'FINAL_AUDIT.json',dict(status='COMPLETE_A_AND_B',completion=dict(A0=True,A_MAIN=True,A_EVALUATION=True,A_POSE=True,B=True),missing=[],
      A_main_fits=12,A_main_updates=24000,A_smoke_updates=12,B_new_updates=0,CPU_tests=result.testsRun,CPU_PASS=True,
      source_bindings_unchanged=True,training_code_lock_unchanged=True,core_SHA_preserved=True,paper_tracked_diff_empty=True,
      original_GT_and_source_cache_modified=False,square_new_target_sidecar_changed_rows=2,
      finalized_A_results=E.bound(E.DOC/'A/results_and_intervals.json'),original_B_results=E.bound(E.DOC/'B/results_and_intervals.json'),
      weights_features_raw_images_not_for_Git=True,user_unrelated_changes_preserved=True,FINAL_access=False,
      independent_confirmatory_claim=False,new_automatic_exploration=False))
    final=['# A/B 실행 완료 — 최종 판단','',
      'A의 12회 × 2,000-step 학습, 초기 R0 포함 평가·통계·6D 보조 분석까지 완료했다. B의 주지표·실사 pose·runtime은 불변이며, 원본 파일을 보존한 채 합성 회전 보조 지표에만 좌표계 정정 sidecar를 추가했다.','',
      '## A — 대칭 지도','']
    for split,c in data['contrasts'].items():
        p=c['EQUIV_minus_INDEXED'];final.append(f"- {labels[split]}: **{c['geometry_gain']}**, ΔE_sym={p['delta']:.8g}, 95% CI [{p['CI95'][0]:.8g}, {p['CI95'][1]:.8g}].")
    final+=['',
      '공동 선택 방식은 사용자 승인 후 실행했으며 기존 RLE를 제거하거나 물체별 clamp로 바꾸지 않았다. `definition_integrity=PASS`와 `geometry_gain`은 별개다. 성능 결론은 fixed checkpoint·data·budget 및 재사용 DEV에 한정한다. 자세한 tail/검출/pose 손해는 `A/REPORT_KO.md`를 따른다.','',
      '## B — 세 incident line 보조','',
      '**LINE_AUXILIARY_NOT_ESTABLISHED** 유지. 합성·실사 primary CI 모두 0을 포함했고, 같은 RTX3080 전체 파이프라인 median은 P12.73ms → B3 39.33ms(약3.09배)였다. 현재 고정 구성에서 최종 추론에 추가할 근거가 부족하다. Hough 원리의 불가능성으로 일반화하지 않는다.','',
      '## 실행량과 보존','',
      '- A: main24,000 + smoke12 optimizer updates; 학습384,000 images + 고정 probe768 + smoke192. 새 평가31,458 detector-image forward 및 동일 수 pose-frame evaluation.',
      '- Pose 실행 중 ID 대응 정정으로 기존 캐시의 CPU 채점만 재수행해 전체 pose 채점 횟수는 59,598이다. 별도 합성 회전 좌표계 정정은 A/B 35,820 frame을 회전만 재평가했으며 새 PnP·neural forward·학습은 없다. 정정 전 수치와 실행 경위를 보존했다.',
      f'- CPU 회귀검사 {result.testsRun}/{result.testsRun} PASS. 가중치·sampler·업데이트별 lr/gradient·BN·마지막 state 무결성 확인.',
      '- 원본 R0/P/DHT·원본 데이터·기존 A0/B 결과·논문·배포는 보존. Square 2행 정정은 새 target sidecar만 변경.',
      '- Git은 코드·표·가벼운 기록만 포함한다. 가중치/상세 raw 예측 및 ignored A/runs의 전체 JSON trace는 로컬 결과 경로에 유지; 24,000-step 요약 CSV와 run별 해시·검증은 Git에 포함한다.',
      '- 재부팅·드라이버 변경·타인 GPU 작업 종료·FINAL 열람·외부 알림·새 탐색 없음.','',
      '추가 학습은 자동으로 시작하지 않는다. 다음 작업은 이 제한된 결과를 근거로 논문에 포함할 범위 결정이며, 원고는 이번 실행에서 자동 수정하지 않았다.','']
    (E.DOC/'FINAL_DECISION_KO.md').write_text('\n'.join(final))
    print('FINAL COMPLETE A_AND_B', {k:v['geometry_gain'] for k,v in data['contrasts'].items()},flush=True)
if __name__=='__main__':main()
