# 합성 정답 대입 B 진단

고정 자동 관측에 원 합성 정답을 대입한 1,003장 진단을 완료했다. 새 자세를 풀거나 후보 bank에 정답을 넣지 않았다. 이 결과는 새 선택기의 검증 성능이 아니며, R/F/V 입력이나 학습 설정을 바꾸는 데 사용하지 않았다.

| 집합 | 전체 | B 관측 있음 | B 관측 없음 | ORIGINAL_WD 원 phase 지지 | ORIGINAL_WD 허용 phase 중 지지 | ORIGINAL_WD 모든 허용 phase 거부 | SWAPPED_WD 허용 phase 중 지지 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 507 | 261 | 212 | 316 | 191 | 32 |
| validation | 235 | 166 | 69 | 61 | 61 | 105 | 8 |
| 전체 | 1,003 | 673 | 330 | 273 | 377 | 296 | 40 |

지지 판정의 분모는 B 입력이 있는673장이다. 원 phase 하나만 검사하면 C2의 다른 허용표현에서 성립하는104장을 놓친다. 관측 없는330장을 통과 또는 최고 신뢰로 세지 않았다. SWAPPED_WD의 전체 허용 phase 거부는633/673장이다. 두 family는 같은 물리 직육면체를 서로 다른 native/Q와 코너 대응으로 표현한다. 이 차이는90°를 새로운 정답 대칭으로 추가했다는 뜻이 아니다.

전체1,505개 허용 phase를 두 family에서 평가했다. 고정 profile34,994개 중16,348개는 실제 선분에서 지지됐고7,520개는 앞끝 이전,11,126개는 뒤끝 이후 교점이었다. raw 저장 정답과 원 backend Rodrigues 표현 사이에서 profile 상태 차이는0개였다. 유한 지지는 material owner 인증이 아니다.

| 원 제약 package | ORIGINAL_WD 허용 phase 중 feasible | SWAPPED_WD 허용 phase 중 feasible |
|---|---:|---:|
| fixed | 646/1,003 | 96/1,003 |
| A | 945/1,003 | 945/1,003 |
| B | 647/1,003 | 103/1,003 |
| AB | 945/1,003 | 945/1,003 |
| edge | 647/1,003 | 103/1,003 |

이는 원 package의 제약 결과이며 raw finite 지지 결과와 별개다. A/AB의 owner h는 원 전역 대응 가설이다. 같은 phase에서 finite와 원 feasible가 모두 참이어도 특정 owner h에 새 finite 제약을 붙여 푼 문제의 인증은 아니다. 서로 다른 목적함수의 raw 비용을 비교해 위 표의 우열을 주장하지 않는다.

저장 진단값에서 frame ID를 중복 없이 집계했다. 다음 표는 허용 phase 중 원 제약을 통과하는 GT 표현이 있고, 실패한 실제 E0도 **같은 family/mode의 원 제약**을 통과하는 프레임을 먼저 센 것이다. 그중 E0 비용이 모든 feasible GT 표현 중 최저 비용보다도 낮은 경우를 오른쪽에 기록했다. 표의 각 행은 별개 목적함수이며 서로 raw 비용을 빼지 않았다. finite는 이 원 제약 판정에 새로 추가하지 않았다.

| ORIGINAL_WD 진단 mode | GT와 실패 E0가 모두 원 feasible | E0가 최선 feasible GT보다 저비용 |
|---|---:|---:|
| fixed | 325/1,003 | 203/325 |
| A | 662/1,003 | 449/662 |
| B | 326/1,003 | 122/326 |
| AB | 662/1,003 | 266/662 |
| edge | 326/1,003 | 128/326 |

fixed의203장은 train166/269, validation37/56으로 나뉜다. 이는 원 동일 package에서의 점수 충돌 진단이다. GT를 실제 후보로 추가하거나, 원 E0가 이 다섯 mode 각각으로 선택됐다고 가정하거나, 이 집계로 새로운 선택기의 검증 성능을 만들지 않았다. 비용 자체는 원 evaluator를 재사용했고 이 표는 그 저장값의 집계다.

원 phase의 ORIGINAL_WD에서 full geometry/canvas 관측360/2,813개, partial 관측162/1,216개, O/T 관측1,047/1,957개, U 관측919/2,052개가 원 제약을 거부했다. 이 수치는 관측 수이며 프레임 수가 아니다. full 점의 좌표 잔차는 원 soft likelihood로 별도 보관했고, 큰 잔차를 새로운 hard 예산 실패로 바꾸지 않았다. 관측/상태는 초기 N0에서 고정했으므로 정답 자세에서도 거부되는 현상은 상태·대응·형상·관측 가정의 충돌 가능성을 함께 가진다.

수치 표현 공백도 보존했다. 원 저장 R의 최대 orthogonality 오차는2.3842×10⁻⁷, determinant 오차도2.3842×10⁻⁷이다. 별도 identity-Q/OpenCV 기술 검사에서는1e-10 기준을 넘은 것이298개 프레임이었다. 이 별도 검사는 원 backend의 재현이 아니다. 실제 원 SciPy backend 표현을 허용 phase와 family에서 검사한 결과는330개 phase/family씩 총660개 평가의 FAIL이다. 원1e-10 FAIL을 PASS로 바꾸지 않았다. 원 R을 수정하거나 새로 최적화하지 않고, raw R 투영/finite와 원 backend 회전벡터 비용/제약을 분리했다.

독립 기하 검산은319,090개 검사에서 의미 오류0개였다. 원 renderer native 좌표의 투영 차이는 최대0.0003728px, family/Q/원점의 vertex-set permutation 차이는 최대5.10×10⁻¹⁶m였다. 따라서 이 자료에서의 큰 finite 거부를 단순한 family 좌표 변환 오류로 설명할 근거는 없다. 독립 검산 통과와 원1e-10 표현 FAIL은 서로 다른 판정으로 유지한다.

원 GT cache top-level에는 material/visibility 필드가 없지만, 원 annotation의 nested metadata에는 `v2_labels.material_variant_target/actual`과 `scene_placement_v2`의 front-face/opening visibility 및 mask/f_* 값이 실제 존재한다. 이 정보는 별도 진단용 원장에 원 SHA·키 경로와 연결한다. variant/scene 가시성은 개별 질의의 per-edge/pixel material owner 인증이 아니므로 물리 대응은 `MATERIAL_OWNER_UNVERIFIED`다. cuboid 변을 실제 재료 경계 GT로 만들지 않는다. 실제 E0와의 비용 비교는 같은 G13/family/mode/관측에서의 고정 자세 재평가다. 원 E0가 그 진단 mode로 선택됐다는 뜻은 아니며, admissibility와 허용 phase를 함께 확인해야 점수 충돌을 해석할 수 있다.

현재 근거는 엄격한 finite 검사만 추가하면 관측 계약이 해결된다는 가정을 지지하지 않는다. 다음 관측 설계에서는 해당 B 질의가 실제 어느 경계에 대응하고 경계가 존재하는지를 구분하는 근거부터 확인할 필요가 있다. 이번 실험에서 그런 새 owner/존재 모델을 구현·검증했다고 주장하지 않는다. 최종 방법 채택은 동일 bank의 R/F/V 및 matched control 결과와 함께 판단한다.

원1,003장·모든 실패·원 참조와 source는 private 원장에 보존했다. 구현과 단위/출처 정의는 [고정 특징·B 진단 계약](FEATURE_SUPPORT_AND_B_DIAGNOSIS.md), 실제 전체 수치·채택 판단은 최종 캠페인 보고서를 따른다.
