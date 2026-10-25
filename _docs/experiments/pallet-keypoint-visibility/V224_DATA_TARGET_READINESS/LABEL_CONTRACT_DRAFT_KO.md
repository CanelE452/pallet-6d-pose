# 초기 query 주변 가시성: 최소 라벨 계약 초안

**관측된 기존 계약:** START/HANDOFF는 semantic visibility와 query-pixel presence·초기 좌표 오차를 분리한다. visible 부위라도 query가 배경을 가리킬 수 있다. SELF는 상류 기하 gate와 RGB 판독을 구별한다. 두 문서의 과거 ‘합성만 학습’ 조건은 이후 공개 실사+GT 허용으로 대체됐으며, 실제 평가223의 fit/norm/문턱·모델 선택 금지는 유지된다.

## 새 제안: 두 축과 근거를 별도 보존

| 필드 | 최소 기록 |
|---|---|
| 원 입력·대상 | frame/component, 원 native/부위 ID·순서, 원 q/L/inputU, 원 SOURCE role. 변경하지 않는다. 선택 pallet instance와 intended part의 대응은 `확인/약함/UNKNOWN` 및 근거를 적는다. |
| **semantic native 상태** | `D / E / SELF / UNKNOWN`. D=의도한 부위가 보임, E=다른 물체가 그 부위를 외부 가림, SELF=같은 pallet에 의한 자기 가림. 근거·authority·CROP/FULL_CONTEXT를 함께 기록한다. |
| **query 좌표 관계** | `ON / NEAR / OFF / UNKNOWN`. 정확한 독립 reference가 있을 때만 `error_px`와 reference authority를 추가하고, 없으면 null이다. ON/NEAR 약판독을 GT≤3px 합격으로 바꾸지 않는다. |
| 초기 crop의 직접 관찰 | `intended-part-visible / foreign-cover-at-part / self-cover / background-or-other / UNKNOWN`. 이는 부위·외부가림의 **지역 약관찰**이며 native 상태로 자동 복사하지 않는다. full-context만으로 아는 상태도 crop에 근거가 있는 것처럼 쓰지 않는다. |
| 사용 권한 | `native-semantic / expected-part-weak / associated-surface-AUX`를 구별한다. 원 inputU와 semantic UNKNOWN은 별개다. 이 초안의 `training_admission=false`; 기존 DEV를 FIT로 바꾸지 않는다. |

## 현재 SOURCE가 제공하는 범위

| 자료 | 제공하는 근거 / 제공하지 않는 권한 |
|---|---|
| V223 strong point0 | 합성 own-image의 target-alone/combined ray-state. query 중심 pixel 라벨이 아니고 자연 실사 native 대응·초기 좌표 정답도 아니다. SOURCE reference 차이는 별도 진단이다. |
| natural finite-part / CF weak | 관찰한 finite-part bag의 보임·외부 COVER·SHAM 보존을 약감독한다. native target은−1이다. GT-derived q는 구성 원점이므로 production 초기 query의 위치 정확도 증거가 아니다. ROI는 약한 부위 근거이지 정확한 keypoint 좌표가 아니다. |
| 기존 YCB surface teacher | associated surface의 mask/depth/reference 기반 visibility 근거를 AUX로 제공할 수 있다. 해당 표면이 선택 pallet의 intended native part라는 대응이나 production q 오차를 자동 제공하지 않는다. 이번 작업에서 원 teacher payload는 재검산하지 않았다. |

**UNKNOWN 기준:** instance/부위 대응이 불명확하거나, 초기 crop이 배경·다른 목재만 보여 의도한 부위를 주소화하지 못하거나, 가림 원인을 구별할 문맥·해상도가 없으면 native 상태를 UNKNOWN으로 남긴다. ‘안 보임’만으로 E, query OFF만으로 E, 다른 목재가 있음만으로 D를 주지 않는다. 문맥에서 D가 확인돼도 query는 OFF일 수 있다. 실제 inputU는 그대로 보존하며 학습 가능 입력으로 승격하지 않는다.

**실험 전 빠진 권한은 구체적이다:** ① 이 원 query가 어느 pallet/부위를 묻는가, ② crop에서 그 부위/외부 가림을 판독할 근거가 있는가, ③ 라벨이 native 의미인가 지역 약관찰인가, ④ 좌표 reference가 있는가 아니면 관계 약판독뿐인가. 먼저 이를 검토해야 한다. 완벽한 physical mesh gold가 없어도 지역 weak/AUX는 그 권한으로 사용할 수 있지만, native D/E 정답으로 승격하는 근거는 별도다. 새 모델·데이터 판독·학습·보호 평가를 실행한 문서가 아니다.

근거: START_HERE_KO.md `015ae7af…`, HANDOFF_KO.md `400e3a22…`, 이미 확인한 V223 schema/code 및 V224 메모. 기존 문서의 사실과 위의 **새 제안**을 구분했으며, 과거 점수는 이 제안의 선택 근거로 사용하지 않았다.
