# ARCHITECTURE GATE DECISION — PalletGraph-6D line screen

[관찰]
- P0: ALL yaw 6.025° / corner 0.4516m / success 0.8046 | truncated yaw 6.574°
- P1: ALL yaw 6.734° / corner 0.3326m / success 0.8046 | truncated yaw 6.616°
- P2: ALL yaw 7.320° / corner 0.4108m / success 0.8046 | truncated yaw 6.171°
- P3: ALL yaw 7.437° / corner 0.3781m / success 0.8046 | truncated yaw 6.187°
- P4: ALL yaw 7.257° / corner 0.3902m / success 0.8046 | truncated yaw 6.005°
- P2_f025: ALL yaw 7.129° / corner 0.4009m / success 0.8046 | truncated yaw 6.241°
- P2_f100: ALL yaw 6.542° / corner 0.3986m / success 0.8046 | truncated yaw 6.097°
- P3_f025: ALL yaw 7.213° / corner 0.3835m / success 0.8046 | truncated yaw 6.302°
- P3_f100: ALL yaw 6.895° / corner 0.3779m / success 0.8046 | truncated yaw 6.092°
- P4_f025: ALL yaw 7.209° / corner 0.3669m / success 0.8046 | truncated yaw 6.309°
- P4_f100: ALL yaw 7.324° / corner 0.3987m / success 0.8046 | truncated yaw 5.631°

[DGP point-only parity]
- FAIL — pose success Δ0 frames, yaw Δ+0.709°, reproj Δ-0.938px (common-success N=70)

[Oracle line utility]
- P2 **FAIL** — 통과 subset: 없음, guard_pass=False
- P3 **FAIL** — 통과 subset: 없음, guard_pass=False

[Generic line result]
- P4 **FAIL**

[Learned MSL result]
- 미실행 (oracle gate 결과에 종속)

[지지 증거]
- [확인] DGP 는 oracle line 에서 GT pose 가 에너지 최소가 되도록 동작한다 (unit test: 섭동 시 단조 증가).
- [확인] 모든 비교가 같은 frame paired 이고 yaw 는 modulo-180° 로 계산했다.

[★ 이 실험이 시험하지 못한 것 — 판정 전에 읽을 것]
- [확인] DGP 는 초기 pose 를 필요로 하고, 그 초기값은 현재 point-only PnP 다.  point 가 실패한 **17/87 프레임에서는 초기 pose 가 없어 fallback** 되어 line 이 개입할 기회 자체가 없었다.
- [확인] 그런데 최상위 가설은 바로 그 경우('point 가 사라지거나 틀릴 때 line 이 회복')다.  즉 **가설의 핵심 대상 population 은 이 설계로 검증되지 않았다**.
- [확인] 실제로 비교된 것은 point 가 **이미 성공한** 70 프레임이며, truncated subset 의 common-success 는 6 프레임에 불과하다.
- [추정] 따라서 아래 판정은 'point 가 이미 pose 를 얻은 상황에서 line 을 더해도 개선이 없다' 로 한정된다.  'line 은 원리적으로 쓸모없다' 가 아니다.

[반증 증거]
- [확인] P2(oracle AMODAL) FAIL — 완벽한 line 기하조차 point-only pose 를 개선하지 못한다.

[현재 판정]
- [확인] 현재 파이프라인(point-PnP 초기값 + 6 iteration DGP)에서는 oracle line 을 넣어도 pose 가 개선되지 않는다.  lambda_line 보정값 3종 모두 동일.
- [확인] 그러나 그 원인은 line 정보 부재가 아니라 **최적화가 line basin 에 도달하지 못함 + energy 불연속** 이다 (위 '판정 정정' 참조).
- [판정] Phase F(learned MSL)는 실행하지 않는다.  다만 사유는 'MSL 이 틀렸다' 가 아니라 **'현재 설계로는 MSL 을 시험할 수 없다'** 이다.  learned head 를 붙여도 같은 최적화 한계에 걸린다.

[architecture 결정]
- MSL: **INCONCLUSIVE** — oracle gate 는 FAIL 이지만, 그 FAIL 이 line 정보 부재가 아니라 (a) 초기 pose 가 line basin 밖 (b) visibility 로 인한 불연속 energy (c) point 실패 17 프레임 미검증 때문이므로 REJECT 로 확정하지 않는다.
- DGP: INCONCLUSIVE (parity 미달)
- SAP: DEFERRED (이번 실행에서 학습하지 않음)

[★ 판정 정정 — line 정보는 있다, 도달을 못 했다]
- [확인] oracle line map 을 그린 **GT pose 근처**에서 E_line 은 정상적으로 최소이고 단조 증가한다 (GT 기준 ±10° slice: 0.28 → 3.9/4.5).  즉 line evidence 에 pose 정보가 **없는 것이 아니다**.
- [확인] 그러나 DGP 가 실제로 출발하는 **point-PnP pose 근처**에서는 E_line 의 최소가 GT 방향이 아니다 (에너지 지형 그림 참조).  line energy 의 basin 이 GT 주변에 좁게 있고 초기 pose 가 그 밖이며, 6 iteration × trust 0.05rad(2.9°) 로는 basin 에 진입하지 못한다.
- [확인] 구현 불일치도 있다: P2 는 **amodal** line map 을 쓰는데 energy 는 `visibility_aware=True` 로 계산해 pose 마다 edge 집합이 바뀐다.  mean-over-edges 라서 edge 수가 바뀌면 값이 점프하고 지형이 계단형이 된다.
- [판정] 따라서 이번 FAIL 은 **line 무용의 증거가 아니라 최적화 도달 실패**다.  MSL 을 REJECT 로 확정하지 않고 **INCONCLUSIVE** 로 되돌린다.

[MSL 전제 점검 — mask support]
- [확인] ep57 segmentation 은 real N87 에서 매우 약하다: **31/87 프레임(36%)은 mask 최대 확률조차 0.5 미만**, mask 면적 median 0.40%.
- [판정] 따라서 MSL 의 'Mask-Supported' 전제(배경 억제를 mask 가 해준다)는 real 도메인에서 성립하지 않는다.  Phase F 의 L2(mask support) arm 은 oracle gate 와 무관하게 현재 mask 로는 의미 있게 시험할 수 없다.

[방법론 발견]
- [확인] subset **집계 median** 만 보면 P3(lambda fraction 1.0)가 close-range 에서 corner error 20.4% 감소로 PASS 처럼 보였다.  같은 frame paired 로 보면 개선 6/13 (Δcorner +0.0004m) 로 오히려 나빴다 = **순위 재배열 아티팩트**.  gate 에 paired 보강 조건을 넣어 허위 PASS 를 제거했다.
- [확인] 같은 이유로 P0->P1 parity 도 집계 median 기준으로는 FAIL(yaw +0.71°) 이지만 paired 기준으로는 중립(Δyaw median +0.029°, pose success 동일)이다.

[다음 admissible experiment]
1. **DGP 최적화를 먼저 고친다** — (a) energy 의 visibility 집합을 초기 pose 로 고정하거나 soft weight 로 바꿔 연속화, (b) mean-over-edges 대신 sample 단위 합으로 정규화, (c) iteration/trust 를 늘리거나 multi-start 를 준다.  이걸 고치기 전 line 결론은 확정할 수 없다.
2. **point 실패 17 프레임을 시험 가능하게 만든다** — line + 알려진 W/D/H 만으로 초기 pose 를 세우는 경로가 있어야 최상위 가설을 검증할 수 있다.  현재는 point 가 실패하면 line 이 개입조차 못한다.
3. mask support 는 별개로 선결 — real 에서 31/87 프레임 mask 붕괴.
3. full training / 3-seed / final-test 는 실행하지 않는다.
