> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# TRAIN 8장 관측–native 기하 양립성 감사 v39

신규 성능 실험이 아니다. 고정 incumbent는 v5 C2 binary seed20261018의 완성 출력이다. 기존 v18의 SYNTHETIC_ANCHOR_IDS.csv에 보존된 v5 TRAIN 선택을 사용했고 checkpoint SHA256도 일치했다. N0로 대체하거나 새 forward를 하지 않았다. 768 TRAIN 중 R>45°이면서 원 bank에 공동5cm5° 성공 후보가 있는 10장, incumbent 성공359장을 확인한 뒤 각 층의 원 ID 오름차순 첫4장을 비용 계산 전에 고정했다. 기존 후보 중 성공 대조 후보는 candidate ID 오름차순 첫 성공 후보이며 GT를 쓰는 진단 선택이다. 실행 정책이 아니다.


## 복원·검산

두 family 각각에서 centered native XYZ와 physical_registration_Q를 합성 GT native XYZ에 하나의 3D bijection으로 연결했다. 영상에서 가까운 GT corner를 관측별로 선택한 것이 아니다. 총16개 family mapping은 1e-6m 이내로 일치했고, 저장 GT 투영과의 반올림 차이 및 직접 family 투영은 사전에 정한0.01px 이내였다. K와 치수도 대조했다. source C2 7장/C1 1장과 v5의 derived C2 평가 목표를 구분했다.

실제 solver의 full point 중 partial로 대체된 것은 제외했다. 864개 관측 residual 행에 원 pixel 잔차, 저장 fit variance/covariance로 정규화한 잔차, reliability, 접선 구간/slack, 출처를 기록했다. partial 점은 법선 잔차이며 선상의 임의 점을 2D corner로 취급하지 않는다. 접선은 실제 solver의 구간 제약으로만 계산했다.

incumbent/동일 원 bank 성공 후보/native GT0/별도 C2 180° 진단을 기존 관측에서 계산했다. 320개의 family×mode 비용과512개의 A/AB 전체 대응 h 행을 보존했다. 각 h의 group energy, replacement, outline, feasible/slack, posterior를 기록하고 실제 marginal 비용을 재구성해 검산했다. AB−A를 추가 경계 항으로 해석하지 않았다. GT 자세를 추론 후보 bank에 넣지 않았다.

## 핵심 결과와 주의할 해석

declared ORIGINAL_WD의 native GT0·h0에서는 큰 잔차가 오류4/4와 성공 대조4/4 모두에 있다(3σ는 설명용 표시이며 새 감점 문턱이 아니다). 그러나 이는 전체 대응과 대칭 표현을 고정한 결과이므로 곧바로 관측 위치 오류율로 사용할 수 없다.

| 비교 범위 | 오류4장 중 최대 잔차>3σ | 성공 대조4장 중 최대 잔차>3σ |
|---|---:|---:|
| 원 GT0, 선언된 h0 | 4/4 | 4/4 |
| source에서 허용한 물리 대칭의 전체 phase만 대조(C1에180° 추가하지 않음) | 4/4 | 3/4 |
| GT0를 유지하고 A의 전체 owner h4개를 진단적으로 대조 | 1/4 | 1/4 |

마지막 행은 네 관측마다 다른 점을 고른 것이 아니라, 물체 전체에 하나의 permutation을 적용한 뒤 최대 정규화 잔차가 가장 작은 h를 본 oracle 진단이다. 90° owner 대안은 직사각 팔레트의 정답 대칭으로 바꾸지 않았다. 성공 대조의 C1 1장도 v5의 derived C2 목표에서는 성공하지만, native source C1에서180°가 물리 정답이라는 주장은 할 수 없다.

따라서 큰 residual을 위치 불일치 하나로 묶는 설명은 부족하다. 일부 사례는 전체 channel/correspondence 가설을 바꾸면 같은 GT 기하와 잘 맞는다. 이것은 대응 모호성의 구체적 진단 근거이지, 어느 물리 edge가 실제 owner인지 독립적으로 확인한 결과는 아니다.

전체 h를 허용해도 남는 불일치도 두 층에 있다. 오류 frame_c264897509def3482244의 best-whole-h=2에서 native B2 부분선은 −52.107px/−9.369σ, 같은 h의 I0는0.531px/0.143σ이다. 성공 대조 frame_3b53ccc522d33c63b309는 h0의 B2 부분선이 −27.006px/−5.402σ이고 다른 full point들은 최대0.952σ다. 두 B2 관측은 모두 expanded_boundary_recovery에서 왔지만 하나의 native 번호나 출처가 문제를 보편적으로 식별한다는 뜻은 아니다. 이8장에 맞춰 B2를 제거하는 규칙을 만들지 않는다.

A의 성공 후보 비용이 incumbent보다 낮은 경우는 오류1/4, 성공 대조0/4이고, AB에서는 오류2/4·대조1/4다. 성공 대조의 진단 성공 후보는 다른 성공 pose일 수도 있으므로 ‘대조1장 손상’이라는 뜻이 아니다. 이 수치는 동일 bank에서 비용이 최종 pose 오류 순서를 일관되게 정렬하지 않는다는 제한된 근거다. GT0는 양쪽 family에서 A/AB의 적어도 한 전체 h에 모두 feasible이었다. 고정 h0/fixed 제약에서는 배제되는 사례가 있으므로 전체 h feasible과 구분해야 한다.

## 결론·다음 제한

단순 관측 제거·새 risk head·가중치 변경으로 넘어갈 근거를 얻지 못했다. 위치 잔차, 전체 대응 표현, source 물리 대칭과 derived 평가 동치를 분리해야 한다. 불일치가 남는 부분선의 실제 재질/mesh edge 출처까지 확정하려면 해당 query 구간의 renderer object/face ID 또는 독립 경계 owner 판독이 필요하다. 이번 검사 캐시의 native/face association은 초기 모델에 조건부이며 독립 물리 정답이 아니다. 관련 자료가 전체 저장소 어디에도 없다고 주장한 것은 아니며, 검사한 E_RUN/bundle에는 독립 query→mesh/face 정답이 없었다.

새 forward/SAM/PnP/학습/GPU/규칙변경/전체 성능평가/공개 게시0. 8장 비용 계산은 마지막 frame까지3.890초, CPU1개로5분 cap 안이었다. 계산 후 provenance JSON에 NumPy array가 있어 저장 예외가 발생했지만 완성된 CSV와 mapping은 보존됐고, raw 캐시를 다시 읽어 export/요약만0.4초 이내 복구했다. 비용을 중복 실행하지 않았다. 첫 select.py 파일명의 표준 라이브러리 충돌도 이름만 바꿔 해결했으며 예전 원자료에는 손대지 않았다.

## 파일과 재현

- SELECTION_PRIVATE_BEFORE_COST.json / SELECTION_COUNTS.json: 사전 표본과 v5 TRAIN 기록 근거
- RESULTS/OBSERVATION_RESIDUALS.csv: 실제 full/partial 잔차
- RESULTS/GLOBAL_HYPOTHESES.csv: 전체 h 비용·constraint·replacement·outline
- RESULTS/COSTS.csv / PER_FRAME_NATIVE_GT_SUMMARY.csv: 후보·기하별 비용
- RESULTS/WHOLE_PHASE_AND_OWNER_DIAGNOSTIC.json: source phase와 전체 owner를 구분한 진단
- RESULTS/NATIVE_MAPPING_VERIFIED_PRIVATE.json: 16개 native mapping 검산
- RESULTS/OBSERVATION_PROVENANCE_V2_PRIVATE.json: 캐시 출처(초기 부분 저장본과 구분)
- RESULTS/COMPLETION_STATUS.json: 실행·보존 상태

기존 pallet-pose Python으로 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1을 설정한다. select_cases_v1.py로 선정하고 audit_geometry.py로 계산한다. 완료 RESULTS는 덮어쓰지 않도록 출력 directory를 새 이름으로 지정해야 한다. 이번 run의 CSV는 완성됐고 JSON export 복구는 repair_exports_and_summarize.py로만 했다. 단독 전달 묶음에는 원자료/checkpoint/private mapping을 넣지 않는다.
