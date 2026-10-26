## V270 — 초기 KP 위치·가림 공동 학습 빠른 확인

**공동 출력은 구현했지만 이번300회 pilot은 가림 판정을 개선하지 못했습니다.** 가림만→공동 학습에서 보임 오판511→471, 외부 가림 누락51→59입니다. 새 모델은 채택하지 않습니다. strict D=0 교사 선택으로 아래쪽 native2/3/6/7의 보임 학습 표본이 모두0개가 됐고 실사223장 전체에서 이 네 슬롯을 가림으로 출력했습니다. 합성 슬롯 사전값만으로도 DEV 균형 recall72.40%, 공동73.09%로 비슷해 영상 단서 학습을 주장할 수 없습니다.

전체 RGB640, frozen backbone/neck 위 기존 KP shared stem와 새 E head를 위치만/가림만/공동 각300회 학습했습니다. 새 학습은 합성768장뿐이고 실사223장은 동결 후 반복 개발 평가만 했습니다. 전체 네트워크 공동 학습의 불가능성을 보인 실험은 아닙니다. 먼저 슬롯별 보임/가림 양쪽 감독과 의미 정의를 바로잡아야 합니다. 새 PnP·실사 fit·threshold/체크포인트 선택0입니다.

[전체 결과·학습·재현](V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/README_KO.md) · [실제19사례](V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/artifacts/V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/visual_v3/INDEX_KO.html) · [슬롯 교사 감사](V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/artifacts/V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/slot_prior_diagnosis_v1/READOUT_KO.html) · [공개본 CPU 재현 확인](V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/PUBLICATION_REPLAY_AND_SCIENTIFIC_QA.json)

![같은 번호에서 보임과 콘 가림을 모두 E로 내는 실패](V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/artifacts/V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/SAME_SLOT_DIAGNOSIS.png)

## V269 — 전체 이미지 DeCLIP의 보임/외부 가림 판정

**초기 좌표는 그대로 두고 보임/가림만 판정했습니다.** 기존 Surgery→DeCLIP은 보임595→655/1147, 외부 가림84→89/90으로 전체 집계가 개선됐습니다. 목재 보임307→393, 정사각형 보임288→262로 **정사각형은 악화했습니다.** 보이는492점을 여전히 가림으로 오판하여 단독 판정기는 완성되지 않았습니다. 가림 recall98.89%는 전체 정확도가 아닙니다(가림 precision15.32%, 전체 정확도60.15%).

원본 EVA 일반 / 같은 가중치의 CSA / 공개 DeCLIP CSA를 동일 합성768학습·235검증·head26721개·각1000회로 비교했습니다. 같은 EVA+CSA→DeCLIP은 보임594→655, 가림89→89여서, 가림 정답 증가를 DeCLIP만의 효과로 부르지 않습니다. 공개 COCO 적응 이력을 밝히고 이번 팔레트용 새 학습은 합성만 사용했습니다. 실사223장은 동결 후 평가만 했으며 반복 개발 평가입니다. 새 PnP·좌표 보정0입니다.

[전체 표·학습·재현](V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/README_KO.md) · [실제123사례](V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/artifacts/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/evaluation_v1/INDEX_KO.html) · [보임/가림 판정 4유형](V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/artifacts/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/FOUR_VISIBILITY_CASES_KO.png) · [합성 원본 M4 대조](V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/artifacts/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/syn_mask_diagnostic_v1/INDEX_KO.html)

![보임/가림 정답과 오판 · 좌표 이동 없음](V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/artifacts/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/FOUR_VISIBILITY_CASES_KO.png)

## V268 — 전체 이미지 경계 보정·V267 실패 시각화·DeCLIP 확인

**Full-image CLIP Surgery는 팔레트 영역을 찾는 앞단으로 활용할 수 있습니다.** 현재 한 지점의 마스크 값만으로 native 보임/외부 가림을 확정하는 방식에 한계가 있습니다. V267만 따로 보이는 모서리 누락·작은 야간 대상·가림 경계 오판·query/native 위치 차이를 실제 영상으로 정리했습니다.

합성만으로 고해상도 보정 A / 경계 손실 추가 B를 각1000회 학습하고 실사223장은 평가만 했습니다. 이전 대비 보임 정답595→619/512, 외부 가림84→78/84로 **두 목표의 동시 개선을 얻지 못해 새 판별기를 채택하지 않았습니다.** 합성 원해상도 IoU65.29→66.01/67.30%의 개선과 실사 native 보임을 구분합니다. 모든 설정은 실사 평가 전에 동결했고 추가 fit·문턱·PnP 변경0입니다.

[전체 표·학습·재현](V268_FULL_IMAGE_BOUNDARY_REFINEMENT/README_KO.md) · [V267만 보는 실패5개](V268_FULL_IMAGE_BOUNDARY_REFINEMENT/artifacts/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/v267_baseline_failures_v1/INDEX_KO.html) · [실사111사례](V268_FULL_IMAGE_BOUNDARY_REFINEMENT/artifacts/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/evaluation_v1/INDEX_KO.html) · [DeCLIP/DCLIP 공식 자료 검토](V268_FULL_IMAGE_BOUNDARY_REFINEMENT/artifacts/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/PRIMARY_SOURCE_DCLIP_REVIEW_KO.md)

![V267 보이는 모서리 누락과 가림 경계 오판](V268_FULL_IMAGE_BOUNDARY_REFINEMENT/artifacts/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/v267_baseline_failures_v1/OVERVIEW_KO.png)

## V267 — 원래 합성 마스크의 전체 이미지 CLIP 분할 비교 완료

**콘을 제외하는 분할은 진전했지만 보이는 모서리를 놓쳐 새 판별기로 채택하지 않았습니다.** 전체 RGB를 처음부터512×512로 입력하고 사용자 원래 RGB/M4 학습768·검증235장만 사용했습니다. KP crop 없이 지도부터 저장한 뒤 초기 q를 읽으며, 실사223장은 평가에만 사용했습니다.

개선형의 외부 가림 recall은 V264의25.56→93.33%(누락67점 중61점 회복·새 누락0), 보임 recall은89.63→51.87%입니다. 보임 precision99.00%는595/601선택점이고 전체 정확도가 아닙니다. 비대상 마스크를 E에 대응한 proxy라 배경과 실제 외부 물체를 직접 구분한 것은 아닙니다. 일반 CLIP과 공식 CLIP Surgery의 **영상 구조 변경**을 같은 head·1000업데이트로 비교했으며 text feature surgery/zero-shot pipeline은 실행하지 않았습니다.

[결과·목재/정사각형 혼동행렬·재현](V267_FULL_IMAGE_CLIP_VISIBLE_MASK/README_KO.md) · [실사104사례](V267_FULL_IMAGE_CLIP_VISIBLE_MASK/artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/evaluation_v2/INDEX_KO.html) · [합성 원본 RGB/M4 대조](V267_FULL_IMAGE_CLIP_VISIBLE_MASK/artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/syn_mask_diagnostic_v1/INDEX_KO.html) · [보임 누락 경계 진단](V267_FULL_IMAGE_CLIP_VISIBLE_MASK/artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/mask_boundary_diagnostic_v1/INDEX_KO.html)

![전체 영상 콘 회복과 보이는 모서리 누락](V267_FULL_IMAGE_CLIP_VISIBLE_MASK/artifacts/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/RESULT_OVERVIEW_KO.png)

## V266 범위 정정 — 전체 이미지 CLIP 분할은 미검증

V266은 KP crop의 DINO 특징을 이용한 실험입니다. 사용자가 뜻한 전체 이미지 기반 분할과 달랐으며, 그 제안이 실패했다는 근거가 아닙니다. 3종 owner GT도 별도 진단용 합성 자료의 것으로, 사용자의 원래 visible-pallet mask와 구분해야 합니다. 원래 팔레트 마스크만으로 전체 이미지의 대상/비대상 이진 분할을 학습할 수 있습니다. 기존 수치 보존, 새 학습0입니다.

[정정 상세·GT 출처·원래 마스크 활용](V266_FULL_IMAGE_SCOPE_CORRECTION/README_KO.md)

## V266 — 영역 분할·대조 학습 비교 완료

**이번 구성은 외부 가림 개선에 실패했습니다.** 현재 방식 / 분할 추가 / 분할+대조 학습의 보임 recall은 89.63 / 92.59 / 92.50%, 외부 가림 recall은 25.56 / 22.22 / 22.22%입니다. 두 새 방식 모두 기존 가림 누락 67점 회복 0점, 새 누락 3점입니다. 새 모델은 채택하지 않았습니다.

합성만으로 두 추가 모듈을 각각 1,000회 학습하고 고정한 뒤, 실사 223장은 평가에만 사용했습니다. 콘의 흰 띠·받침을 팔레트로 분할하는 전이 문제와, 합성에서 분할이 맞아도 최종 가림 판단을 놓치는 문제를 구분했습니다. 학습 조건·체크포인트·전체 표·실제 97사례·합성 정답 대조·CPU 재현 자료를 포함합니다. 실사 223장은 반복 개발 평가 자료입니다.

[전체 결과·재현](V266_SEGMENTATION_CONTRASTIVE/README_KO.md) · [외부 가림 90점 전체](V266_SEGMENTATION_CONTRASTIVE/artifacts/V266_SEGMENTATION_CONTRASTIVE/evaluation_v1/INDEX_KO.html) · [합성 정답 대조](V266_SEGMENTATION_CONTRASTIVE/artifacts/V266_SEGMENTATION_CONTRASTIVE/syn_visual_v1/INDEX_KO.html)

![보임 복구·콘 가림 누락과 새 오판](V266_SEGMENTATION_CONTRASTIVE/artifacts/V266_SEGMENTATION_CONTRASTIVE/RESULT_OVERVIEW_KO.png)

## V265 — 가림 누락67점 진단·보임 조건 위치 보정

누락67점은 참조 crop 밖5, crop 안 참조≤3px10,3–6px15,6px 초과37점입니다. 숨은 E 좌표는 직접 보이는 위치 정답이 아니므로 거리 구간과 실제 crop 해석을 분리했습니다. 콘의 띠·경계·받침 및 주변 구조 반응이 반복됩니다.

**보임이 별도로 확인된 점의 보정 후보 용도가 더 맞습니다.** 정사각형 수동D에서 원시 제안의≤3px가223→352/531점, 중앙값3.68→2.23px입니다. 목재153→136/323점은 개선되지 않았습니다. 사후 humanD 가상 비교이며 새 승인 게이트/배포 성능은 아닙니다. 새 학습·모델 추론·PnP·threshold/라벨 변경0입니다.

[전체 진단·재현](V265_EXTERNAL_MISS_LOCATION_DIAGNOSIS/README_KO.md) · [누락67점 전체 crop](V265_EXTERNAL_MISS_LOCATION_DIAGNOSIS/artifacts/V265_EXTERNAL_MISS_LOCATION_DIAGNOSIS/all67_gallery_v1/INDEX_KO.html) · [대표12개 실제 crop](V265_EXTERNAL_MISS_LOCATION_DIAGNOSIS/artifacts/V265_EXTERNAL_MISS_LOCATION_DIAGNOSIS/diagnosis_v1/INDEX_KO.html)

## V264 — Heatmap 위치 주변 특징의 합성 학습 비교 완료

위치 모델을 고정하고 전체 평균/heatmap 주변 분류기를 동일 조건으로 각1,000회 학습했습니다. 실사223장은 이번 학습·설정 선택에 사용하지 않았습니다. **보임 recall56.2→89.6%로 개선됐지만 외부 가림 recall78.9→25.6%, F121.4→19.8%로 악화해 채택하지 않았습니다.**

[전체 결과·실제 가중 영역·재현](V264_HEATMAP_LOCAL_VISIBILITY/README_KO.md) · [학습 전 조건](V264_HEATMAP_LOCAL_VISIBILITY/artifacts/V264_HEATMAP_LOCAL_VISIBILITY/PROTOCOL_BEFORE_FIT.json) · [고정16셀 실제 crop](V264_HEATMAP_LOCAL_VISIBILITY/artifacts/V264_HEATMAP_LOCAL_VISIBILITY/evaluation_v1/INDEX_KO.html)

![보임 복구와 외부 가림 누락 사례](V264_HEATMAP_LOCAL_VISIBILITY/artifacts/V264_HEATMAP_LOCAL_VISIBILITY/LOCAL_POOL_SUCCESS_FAILURE_KO.png)

## V263 — Heatmap을 가림 판단에 쓰는 방법과 저장 출력 진단

현재 VIS head는 crop 전체 특징의 평균을 읽으며 heatmap이 찾은 위치 주변을 직접 읽지 않습니다. **heatmap으로 위치를 찾고 그 주변 특징과 맥락으로 D/E를 판단하는 방식**은 다음 개선 가설이며 아직 구현·검증하지 않았습니다. 현재 softmax heatmap은 가려져도 봉우리가 생겨 보임 확률로 쓸 수 없습니다.

기존 PRIMARY 주석 D 1,147/E 90점의 저장 출력만 재집계했습니다. 높은 봉우리로 보임을 구분한 AUC는 0.609, 낮은 분포 entropy는 전체 0.735·목재 0.721·정사각형 0.776입니다. AUC는 정확도가 아니며, 반복 개발 평가의 진단입니다. 새 학습·추론·PnP·임계값 탐색은 없습니다.

사용자 시각 검토: G996은 주석 E/모델 D의 불일치로, native 대응과 주석을 재검토해야 합니다. G981·G200은 1.46→3.65px·1.77→4.57px로 3px 기준을 넘었으며 큰 시각적 실패라고 단정하지 않습니다. 기존 주석·지표는 유지합니다.

[분포와 설명 HTML](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/heatmap_visibility_diagnostic_v1/INDEX_KO.html) · [전체 수치·입력 SHA](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/heatmap_visibility_diagnostic_v1/RESULTS.json) · [1,237점 특징 CSV](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/heatmap_visibility_diagnostic_v1/PRIMARY1237_FEATURES.csv) · [재현 코드](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/diagnose_saved_heatmap_visibility.py)

![저장 heatmap의 보임·가림 분포](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/heatmap_visibility_diagnostic_v1/DISTRIBUTIONS_KO.png)

## V263 — 되는 사례와 실패 사례 실사 시각화

저장된 공동 모델 출력으로 위치 복구·보정 손상·가림 정답·오답을 나란히 비교했습니다. 새 학습·추론·문턱 변경은 없습니다. 파랑 원은 초기 추론, 분홍 +는 수동 주석, 녹색 ×는 적용 위치입니다. 그림의 오차는 원본 영상 px입니다.

[확대·성공/실패 필터 HTML](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/visual_success_failure_v1/INDEX_KO.html) · [사례별 수치와 출처](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/visual_success_failure_v1/CASE_SUMMARY.json)

![위치 보정 성공과 손상](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/visual_success_failure_v1/LOCATION_SUCCESS_FAILURE_KO.png)

![가림 판별 정답과 오답](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/visual_success_failure_v1/VISIBILITY_SUCCESS_FAILURE_KO.png)

목재에서는 초기 오차0.73px인 보이는 점을 외부 가림으로 부릅니다. 위치를 복구하는 사례와 정확한 점을 옮겨 손상시키는 사례가 함께 있으며, heatmap peak만으로 지정 native 점의 직접 보임을 확인하지 못합니다. Heatmap 색은 보정된 확률이 아닙니다.

## V263 — 공간 위치 보정과 가림 비교 완료

합성만으로4설정·각1,000업데이트를 실행하고 설정을 동결한 뒤 기존223장을 개발 평가했습니다. **가림 판별은 개선되지 않았고 새 모델은 채택하지 않았습니다. 위치 보정은 정사각형에서 개선됐지만 목재 손상이 남습니다.**

| 동일 평가 | 원래 | 위치 전용 | 공동 모델 |
|---|---:|---:|---:|
| 목재 수동 보임점3px 이내 /323 |153|140|140|
| 정사각형 수동 보임점3px 이내 /531 |223|321|317|
| 목재 고정 후단 참조상대 성공 /105 |57|51|58|
| 정사각형 고정 후단 참조상대 성공 /118 |30|72|50|

공동 모델의 외부가림은35/90점(E precision11.67%, recall38.89%)으로 V261 CONTROL의68/90점(27.09%/75.56%)보다 낮았습니다. 가림 전용은0/90점입니다. 공동 위치 보정은 전체 수동 보임854점에서 복구140·손상59이며, 원오차≤3px 보임376점에서도102점을 E로 오판했습니다. 고정 후단은 기존 solver·C2/C4 참조 계약을 재생한 별도 진단이며 독립6D 정답 평가가 아닙니다.

[전체 표와 실제 crop](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/ROOT_READOUT_KO.html) · [가림·위치 상세 보고](V263_SPATIAL_LOCALIZATION/artifacts/V263_CURRENT_EVALUATION_A/PORTABLE_CURRENT_REPORT_KO.md) · [고정 후단 전체 통계](V263_SPATIAL_LOCALIZATION/artifacts/V263_CURRENT_UTILITY_B/current223_utility_v1/REPORT_KO.md) · [재현 안내](V263_SPATIAL_LOCALIZATION/artifacts/V263_SPATIAL_HEAD_C/RUN_GUIDE_KO.md) · [전체 파일과 대형특징 제외 manifest](V263_SPATIAL_LOCALIZATION/RESULTS_PACKAGE_MANIFEST.json)

![유형별 위치 가림 복구 손상 비교](V263_SPATIAL_LOCALIZATION/artifacts/V263_ROOT_SPATIAL_LOCALIZATION/RESULT_OVERVIEW.png)

223장 외에9월6일20영상의 실사1,505장 ZIP이 있습니다. 현재223장·기존학습851장과 encoded SHA 중복0이나 점별 D/E 정답이 없어 이번 실험에는 편입하지 않았습니다. [추가 실사 조사](V263_SPATIAL_LOCALIZATION/ADDITIONAL_REAL_DATA_INVENTORY.json). 현재223장은 반복 개발 자료이며 이번 제한된4설정 비교를 마쳤습니다.

## V263 — 공간 위치 보정 비교 실행 중 · 추가 실사 자료 확인

사용자가 V262의 최소 후속 계획 실행을 새로 요청했습니다. 합성에서 최대4설정·seed17·각1000업데이트로 제한하고 현재223장은 개발 평가에만 사용합니다. 이 게시 단계에는 새 학습 성능 결과가 없습니다.

| 추가 자료 | 확인 내용 | 독립 가림 검증 준비도 |
|---|---|---|
| 기존 주석 중 현재223장 밖215장 | 목재20·기존 플라스틱194·정사각형1 | 과거 개발 자료 |
| green851와194/290/51장 ZIP | 세 ZIP의 모든 이미지가851장에 중복,851장은 과거 증강 학습 사용 | 미사용 검증 아님 |
| 9월6일 원본 추출1505장·20영상 | encoded SHA 기준 current223/green851 중복0 | 점별 D/E 정답 없음·과거 노출 미확인 |

[정확한 경로·중복·한계](V263_SPATIAL_LOCALIZATION/ADDITIONAL_REAL_DATA_INVENTORY.json) · [확인 표 HTML](V263_SPATIAL_LOCALIZATION/ADDITIONAL_REAL_DATA_READOUT_KO.html) · [고정 실험 조건](V263_SPATIAL_LOCALIZATION/PROSPECTIVE_EXECUTION_POLICY_V2.json) · [읽기 전용 확인 코드](V263_SPATIAL_LOCALIZATION/audit_other_real_data.py)

## V262 — 문헌·공식 GitHub 검토 (새 실험 없음)

OVD와 지정 코너 가시성의 차이, 위치 분포·관측 적합성, 대칭·후보 생성/선택, 합성→실사 전이를 관련 논문·공식 코드와 대조했습니다. [조사와 최소 후속 설계](V262_LITERATURE_REVIEW_ONLY/README_KO.md) — V261 결과·학습 상태는 그대로이며 새 학습/PnP 실행은 없습니다.

## V261 — 마지막 공개 실사 추가 비교 완료

공개 실사 추가 전후 현재 동일 query 1,237점에서 보임 정답 964→959 /1,147, 외부 가림 정답 68→68 /90입니다. 전체 개선을 확인하지 못했으며 추가 학습·문턱 탐색을 종료합니다.

[상세 결과·재현](V261_FINAL_PUBLIC_REAL_COMPARISON/README_KO.md) · [시각화](V261_FINAL_PUBLIC_REAL_COMPARISON/ROOT_READOUT_KO.html) · [고정 실제 crop 12개](V261_FINAL_PUBLIC_REAL_COMPARISON/artifacts/V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/final_current_crop_gallery_v1/GALLERY.html)

> [V259: 공개 depth point teacher pilot](V259_PUBLIC_DEPTH_POINT_TEACHER_PILOT/README_KO.md) — 공개 2프레임의 D/E/U 교사 라벨을 점검했습니다. 모델 정확도 결과가 아니며 학습 0회입니다. 최신 모델 비교는 V256입니다.

> [V257：공개 pallet 보정값·mesh 형식 확인](V257_PUBLIC_GEOMETRY_FORMAT/README_KO.md) — 공식 K·OBJ 원본과 독립 검산을 보존했습니다. 새 학습·정확도 평가 0이며, native 대응·투영·point 가림 라벨은 아직 미확정입니다. 모델 비교는 아래 V256 링크에 있습니다.

> [V256：기존 데이터와 새 합성 가림 대조의 동일 모델 비교](V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md) — 동일 모델·초기값·1,000회 학습으로 새 합성 데이터 추가 효과를 비교했습니다. 합성 가림 recall은 증가했지만 precision·FIT 대조 패턴은 악화했고 공개 YCB 가림은 두 모델 모두 0/57입니다. 모델 채택은 하지 않았습니다.

> [V255：원 query를 유지한 8개 점별 3D 가림 대조](V255_ALL_NATIVE_SOURCE_FIT_INTERVENTIONS/README_KO.md) — 86개 합성 원본·688개 점별 시도·1,892개 실제 장면과 독립 검산 및 원 crop 시각화를 모두 공개합니다. 모델 성능 개선은 아직 입증되지 않았습니다.

> [V254：합성 Source의 고정 좌표 COVER·SHAM 반응](V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE/README_KO.md) — 새 학습 없이 저장된 세 모델의 같은 초기 좌표 BEFORE·COVER·SHAM 반응과 누락을 검산합니다. 현재 평가 데이터와 PnP는 사용하지 않았습니다.

> [V253：합성·공개 실사 저장 오판과 FIT crop 진단](V253_SAVED_FAILURE_SOURCE_FIT_CROP_AUDIT/README_KO.md) — 새 학습 없이 저장 점수의 오판 전이와 참조 분포를 검산하고, 고정 규칙으로 선택한 Source FIT crop을 공개합니다. 현재 평가 데이터와 PnP는 사용하지 않았습니다.

> [V252：일반 참조·대상 참조 direct head의 SOURCE 비교](V252_DIRECT_REFERENCE_HEAD_SOURCE_FIT/README_KO.md) — 합성·공개 실사 FIT으로 direct head 두 개를 각각 1,000회 학습하고, 두 checkpoint를 봉인한 뒤 SOURCE DEV를 평가했습니다. 실제 팔레트 전이는 별도 검증 대상입니다.

> [V251：합성·공개 실사로 학습한 작은 참조 head](V251_FROZEN_CORE_REFERENCE_RESIDUAL_SOURCE_FIT/README_KO.md) — 합성·공개 실사 FIT으로 참조 head와 대조군을 각각 1,000회 학습하고, 두 checkpoint를 봉인한 뒤 SOURCE DEV를 평가했습니다. 실제 팔레트 전이는 별도 검증 대상입니다.

> [V250：입력만으로 RGB 참조를 선택한 SOURCE 진단](V250_INPUT_ONLY_REFERENCE_SELECTOR/README_KO.md) — 정답 라벨을 읽기 전에 1,521행의 참조 선택과 세 특징 점수를 봉인했습니다. SOURCE 진단이며 새 배포 모델·팔레트 정확도가 아닙니다.

<!-- V249 actual SOURCE oracle-reference diagnostic -->
## 2026-10-25 · V249 대상 RGB 참조 진단

[전체 결과·원 점수·독립 검산·공개 CPU 재현](V249_SOURCE_RGB_REFERENCE_ORACLE_DIAGNOSTIC/README_KO.md)

정답 물체 RGB 참조의 중심 특징 점수 차이는 SOURCE AUROC0.9837(오차≤3px 0.9919)이지만 teacher-ID oracle입니다. 기존 V248 head의 같은 SOURCE 순위는 약0.53이며, 배포 참조 선택·팔레트 전이는 미해결입니다. 새 분류기 학습·현재 평가·PnP는 없습니다.

---

<!-- V248 actual SOURCE results -->
## 2026-10-25 · V248 공개 픽셀 보조학습 실제 결과

[결과·코드·혼동행렬·실제 crop·재현 검산](V248_PUBLIC_PIXEL_AUX_ACTUAL_RESULTS/README_KO.md)

AUX 내부 DEV는 보임319/320·가림128/128이지만 기존 YCB 가림은 양군 모두0/57입니다. SOURCE 채택 기준 실패로 모델 채택·새 current/v61 평가·PnP는 없습니다.

---

## V247: 공개 TRAIN 원 q8 prompt의 동결 SAM 진단

[실제 실행·readout·고정 gallery·재검산](V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC/README_KO.md). 원 q/L/native/U와 prompt/runtime 지원을 구분하고, logit>0 exact/5×5 proposal 근거를 별도로 기록합니다. teacher는 모델 입력이 아니며 native 라벨·모델 채택·새 학습 admission을 만들지 않았습니다.

## V246: 공개 TRAIN의 원 query를 바꾸지 않은 조건부 진단

[결과·출처·SOURCE 사례·재검산](V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT/README_KO.md). 입력 메타데이터의 SOURCE VAL 읽기/폐기는 투명하게 기록하며, 인증 배열은 선택 TRAIN22쌍만 사용했습니다. 원 U와 native UNKNOWN을 보존하고 라벨·학습 admission을 만들지 않았습니다.

## V245: 같은 장면 CF 추가의 SOURCE 대조

[실제 결과·제한·고정 gallery·재생 코드](V245_SCENE_CF_SOURCE_TRIAL/README_KO.md). 기존 context baseline은 원본 바이트 재사용·새 업데이트0, treatment는 동일 seed17 초기값에서1,000회 학습했습니다. 사전 SOURCE 기준 미통과이며 현재 실사 개선이나 native gold 인증을 뜻하지 않습니다.

## V242/V244: 실제 장면 합성 검산과 동결 모델 행동 진단

[전체 설명·표·실제 crop·코드](V244_SCENE_DATA_AND_FROZEN_BEHAVIOR/README_KO.md). 실제 생성5,367쌍/미생성39개 보존, 원 BEFORE·q8·문맥·교사 조건의 독립 전량 검산 완료. TREAT 자체 base/residual/total과 V235 학습 COVER/SHAM 기능을 분리 진단했습니다. V243는 코드 준비 상태이며 새 학습 성능 결과가 아닙니다. [공개 검산](V244_PUBLICATION_REVIEW.json).

## V241: 원 장면 일치 가림 대조 데이터 준비 / 실제 생성·학습 전

[상태표·코드·전체 부모5,406행·donor64·검증 명령](V241_SCENE_CONSISTENT_CF_PREPARATION/README_KO.md). 계약/메타데이터 감사와 가상 장면4개 계산 검증까지 공개했습니다. 이전 가상 테스트 실패2개·수정 전 코드도 포함합니다. 실제 scene CF 생성0·새 학습0·현재223/held 새 사용0·PnP0이며 성능 개선 결과가 아닙니다. [공개 파일 검산](V241_PUBLICATION_REVIEW.json) · [가상 계산 재실행](V241_PORTABLE_TOY_REPLAY.json). 최근 실제 모델 결과·crop 시각화는 V240입니다.

## 최신 V239–V240: 공통 특징·두 모델 실제 학습 완료 / SOURCE 기준 미달

[전체 보고서·재현 명령](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md) · [CM/precision/recall/보류율](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V240_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [성공·실패 crop](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [문맥과 8개 query](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html).

7,617개 활성 특징의 byte 일치와 원 좌표·분할 정합, 두 모델 각각 새 1,000회 학습과 동결, 독립 완료 검산·지표 재계산·CPU 예측 전량 일치를 확인했습니다. 공개 실사 SOURCE DEV E는 local0/57→context1/57, 가족 평균 균형 recall gain+0.345657pp로 사전 기준 FAIL입니다. 모델 채택·현재223 새 평가·문턱 조정은 없습니다. 6개 frozen 특징 배열과 두 작은 모델도 제공해 원 이미지·DINO 없이 CPU 검산할 수 있습니다. [공개 검증 기록](V240_PUBLICATION_REVIEW.json).

## 최신 공개 V237–V239: 입력 복원 완료·학습 전 중단 진단·다음 비교 준비

[전체 보고서와 상태표](V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION/README_KO.md) · [원 문맥과 8개 실제 crop](V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION/artifacts/V237_ROOT_SOURCE_CONTEXT_INPUTS/fixed4_original_input_gallery/FIXED4_SOURCE_INPUT_CONTEXT_AND_Q8.png) · [FIT 5,876개 행별 특징 차이](V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION/artifacts/V238_ALL_FIT_DESCRIPTOR_DIAG/ALL5876_DESCRIPTOR_DIFFERENCES.csv).

V237은 원 crop 7,617개의 픽셀·valid SHA 일치와 독립 입력 QA를 완료했습니다. V238은 1e-6 특징 검증 초과로 optimizer 전에 중단됐습니다(66/5,876개, 최대 약 1.55158e-6). V239는 한 번 생성한 공통 특징으로 두 모델을 새로 비교하도록 준비했으며 실제 생성·학습·새 DEV 결과는 아직 없습니다. 완료·실패·준비 상태와 한계를 구분해 보존합니다. [공개 검증 기록](V239_PUBLICATION_REVIEW.json).

## 최신 공개 V234–V236: 실패 원인 진단과 같은 점 가림 대조

[전체 보고서·실행 조건·재검산](V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/README_KO.md) · [CM/precision/recall/보류율](V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [성공·실패 crop](V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/artifacts/V235_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [crop과 공개 원 장면](V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/THREE_SOURCE_CONTEXT_BOARD.png).

실제 SOURCE 대조쌍5,367개와 고정 head1000회 학습·독립 QA를 완료했습니다. 공개 실사 내부 DEV 외부 가림은 0/57이며 SOURCE 기준 미달로 모델을 채택하지 않았습니다. 실패57개 중56개는 crop 밖 다른 원 query에 대상 픽셀이 있음을 확인했고, 다음 입력 검증 경로를 기록했습니다. 새 context 모델은 아직 학습하지 않았습니다. [배포 검증 기록](V236_PUBLICATION_REVIEW.json). 이전 기록은 아래에 유지합니다.

## 최신 공개 V233: 공개 실사·합성 crop 대조

[보고서와 재검산 안내](V233_SOURCE_CROP_ENCODER_ADAPTATION/README_KO.md) · [실제 성공/실패 crop](V233_SOURCE_CROP_ENCODER_ADAPTATION/artifacts/V233_ROOT_CROP_ADAPTATION/fixed8_source_dev_gallery_v2/SOURCE_DEV_FIXED8.png) · [CM/P/R/U 표](V233_SOURCE_CROP_ENCODER_ADAPTATION/artifacts/V233_SOURCE_REPORT_C/results/METRICS.csv). 실제1000회 학습 비교와 독립 QA를 완료했습니다. SOURCE 순개선0pp여서 미채택이며 실사 해결 완료가 아닙니다. 이전 기록은 아래에 유지합니다.

[패키지 배포 검증 기록](V233_PUBLICATION_REVIEW.json)

[V231: 기존 head·라벨·Pallox·특징 cache 경로 점검 원본 추가. 새 학습·성능 개선 결과 없음. V229 실사 비교와 V230 표·실제 crop·검산 코드로 연결.](V231_EXISTING_HEAD_DATA_CACHE_AUDIT/PORTABLE_REPORT_KO.md)

[V230: 공개 YCB8,000점의 q/r 관측·원라벨 재현. 기준점과 crop 중심의 차이·실제6crop·기존 BCE 학습 경로·검산 자료 공개. 새 모델 성능 개선 결과 아님.](V230_YCB_QUERY_OBSERVATION/PORTABLE_REPORT_KO.md)

[V229: 90%대 SOURCE 후보를 실사에 고정 평가. 보임 recall83.26%·외부 가림30.00%로 채택하지 않음. 원1784출력·목재/정사각형 CM/P/R/U·성공/실패8crop·재검산 코드 공개.](V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md)

[V227: 공개 SOURCE에서 고정 읽기 대 학습 attention; attention 추가 효과 미확인. 원 특징·두 head·736출력·실제 crop·학습 없는 검산기 공개, 새 실사 개선 아님.](V227_CONTEXT_READ_SOURCE_PROBE/PORTABLE_REPORT_KO.md)

[V226: 공개 원 query368개 reference 정합; 가림37/44 crop안, 좌표·역할·문맥 혼재 확인. 새학습/실사개선 결과 아님.](V226_REFERENCE_ALIGNMENT/PORTABLE_REPORT_KO.md)

[V225: 원 query crop15장·장면분리 점검; 독립 분리 후보 FIT의 가림 weak8개, 새학습/성능개선 결과 아님](V225_DATA_AND_CROP_REVIEW/PORTABLE_REPORT_KO.md)

[V224: 새 학습 없이 원 초기 query 약라벨 연결·현재 loss 경로 점검; 성능 개선 결과 아님](V224_DATA_TARGET_READINESS/PORTABLE_REPORT_KO.md)

V223: 점·부위 출력 분리 대조와 입력일관성 검사를 완료했습니다. [표·전체crop·실행/QA·범위예외](EXPERIMENT_INDEX_THROUGH_V223_KO.md). 최신실사비교는V220이며 이번 SOURCE 공동개선은확인되지않았습니다.

V222: 예상 부위 COVER/SHAM 감독을 실제 학습했으나 SOURCE 검증 개선은 확인되지 않았습니다. [표·전체 crop·실행/QA 원문](EXPERIMENT_INDEX_THROUGH_V222_KO.md). 최신 실사 비교는V220입니다.

[V221 검토 대조·중복 시도와 남은 과제](EXPERIMENT_INDEX_THROUGH_V221_KO.md)

최신 모델은 보이는 점 보존이 늘었지만 외부 가림 검출은 함께 좋아지지 않았습니다. 첨부 V193 조언과 현재 계약·데이터·실험 이력을 대조한 결과이며 이번 새 학습·실사 추론은 없습니다.

---

# 초기 키포인트 crop의 보임·외부가림 실험 기록

초기 추론 좌표를 중심으로 자른 crop에서 **대상 팔레트가 보이는지, 다른 물체에 가려졌는지** 판단하는 연구입니다. 자기 가림은 초기 추론의 기하 처리와 구분하고, 위치 오차도 따로 분석합니다. WOOD/SQUARE와 native8 전체 목표는 아직 해결되지 않았습니다.

[전체 실험 목차](EXPERIMENT_INDEX_KO.md) · [전체 그림 목차](GALLERY_INDEX_KO.md) · [최신 실사 비교](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md) · [데이터·코드 출처](THIRD_PARTY_ATTRIBUTION_KO.md) · [게시·재검산 방법](PUBLICATION_GUIDE_KO.md)

이 게시본은 허용된 모든 단계의 실행 코드·방법·저장 출력·표·검산·시각화와 실패 기록을 함께 보존합니다. V63–V186의 허용된124개 디렉터리와 초기 v60/v61/v62 관련 자료를 연결합니다. 버전번호가 중복된 V111 두 디렉터리는 따로 남기고, 명시적으로 접근이 금지된 V112는 포함하지 않습니다. 단계별 PASS/완료는 모델 성공을 뜻하지 않습니다.

## 시작 문서와 초기 비교군

[START_HERE_KO](archive/task-2/pallet-keypoint-visibility/START_HERE_KO.md) · [HANDOFF_KO](archive/task-2/pallet-keypoint-visibility/HANDOFF_KO.md) · [PATH_INDEX.csv](archive/task-2/pallet-keypoint-visibility/PATH_INDEX.csv)

[v60 외부가림 proxy](archive/task-2/external_ray_proxy_head_20261022_v60/REPORT_KO.md) · [v61 target presence](archive/task-2/target_presence_classifier_20261022_v61/REPORT_KO.md) · [v62 기존 모델 저장 출력 감사](archive/task-2/presence_previous_models_audit_20261022_v62/AUDIT_COMPLETE.json)

## 현재 실제 분류 결과

V184의 같은 초기 query 비교에서 사람의 D(직접 보임)1,147점·E(외부가림)90점, 총1,237점입니다. 아래 recall 분모에는 보류 U가 포함됩니다. 자기 가림·영상 밖·미주석·대응 불명은 이 이진 표와 분리해 원문에 보존했습니다.

| 모델 | D precision | D recall | E precision | E recall | 보류율 | 외부가림을 보임으로 오판 |
|---|---|---|---|---|---|---|
| v61 | 99.24% | 22.67% | 9.03% | 97.78% | 0.00% | 2 |
| 기존 실사 학습 모델 | 99.54% | 57.11% | 20.65% | 92.22% | 14.31% | 3 |
| C52 RGB | 94.04% | 71.58% | 9.85% | 36.67% | 2.34% | 52 |
| C53 RGB+SAM | 92.54% | 75.68% | 6.83% | 22.22% | 0.49% | 70 |

새 모델은 보이는 점을 더 살렸지만 외부가림을 보임으로 통과시키는 오류가 커졌습니다. **C52/C53 모두 채택하지 않았습니다.** [재질별 혼동행렬·precision/recall·보류율과 실제 crop](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md)을 함께 확인하십시오.

![V184 고정 비교 결과](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/FIXED_PRIMARY_REAL_METRICS.png)

평가에 사용한 자체 촬영223장과 공개 held45장 및 합성 VAL은 반복 노출된 자료이며 독립적인 blind test가 아닙니다. 최근 학습은 합성·공개 SOURCE TRAIN을 사용했고 이 평가 자료는 학습·보정·문턱 선택에 넣지 않았습니다. 기존 실사 학습 모델의 과거 CAL 이력은 원문 그대로 남겼습니다. 공개자료의 weak center-owner 정답과 팔레트 native 부위의 외부가림 정답도 구분합니다.

## 최근 어디까지 확인했나

- [V183](archive/task-2/pallet-keypoint-visibility/query_visibility_query_jitter_paired_source_training_20261024_v183/PORTABLE_REPORT_KO.md): SOURCE TRAIN 좌표 jitter 학습과 두 head의 고정 최종 출력. source 결과와 실제 전이는 분리합니다.
- [V184](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md): 저장된 v61/기존 모델과 새 head를 동일 query로 실제 비교. 보임 회복과 외부가림 손상을 모두 공개했습니다.
- [V185](archive/task-2/pallet-keypoint-visibility/query_visibility_public_part_supervision_readiness_20261024_v185/PORTABLE_REPORT_KO.md): 공개 부위 감독 준비도와 기존 crop·감독 영역 교차를 확인했습니다. 원래 crop 밖의 부위를 감독하는 사례가 있었고, UNKNOWN은 유지했습니다.
- [V186](archive/task-2/pallet-keypoint-visibility/query_visibility_public_source_registration_probe_20261024_v186/PORTABLE_REPORT_KO.md): 고정 공개 SOURCE4프레임·5instance의 원 mesh/K/R/t 렌더와 독립400 ray 검산을 완료했습니다. 제공 mask와의 외곽 정합은 높지만 센서 깊이 차이·결측이 있어 정확한 부위 가림 정답으로 승격하지 않았습니다. 새 모델 개선 결과가 아닙니다.

## 원본과 게시용 문서

원문은 `archive/task-2/`의 원래 형제 구조로 보존하고 각 버전의 `PORTABLE_REPORT_KO.md`를 GitHub에서 읽을 수 있게 별도로 만들었습니다. 원문에 남은 `/home/...` 경로는 당시 실행 환경의 provenance입니다. HTML은 GitHub에서 실행되지 않으므로 Markdown 표·이미지를 먼저 보고, HTML 갤러리는 자료를 내려받아 열 수 있습니다.

[원본 파일·해시·포함/제외 manifest](PUBLICATION_MANIFEST.json)에서 코드, 저장 출력, 소형 자체 학습 head, 검토 그림을 확인할 수 있습니다. 큰 텍스트 원장은 무손실 gzip 분할본과 원문 SHA256으로 보존합니다. 원 데이터 은행·특징 캐시·환경·대형 사전학습 weights 등은 경로와 제외 사유를 남깁니다. 다운로드가 필요한 공식 공개 자료와 라이선스는 [출처 표](THIRD_PARTY_ATTRIBUTION_KO.md)를 따릅니다.

이후 완료하는 실험도 같은 방식으로 원문·실패·그림·검산 결과를 묶어 `main`에 계속 반영합니다. 게시 검산은 숫자와 파일 보존 검산이며 전체 가림 모델의 목표 달성 판정과는 별개입니다.
