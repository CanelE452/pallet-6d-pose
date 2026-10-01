# 새 실사 선택·평가를 실행하지 않은 이유

**이 방법의 source VAL은 45개 조건 중 41개만 통과했으므로 새 실사 routing과 T/R 평가를 실행하지 않았다.** [고정된 source gate](SOURCE_VAL_GATE.json)의 `PASS=false`, `real_routing_authorized=false`를 그대로 적용했다. 학습 수렴 인증을 통과한 것과 T/R 품질 조건을 통과한 것은 다르다.

실패 4개는 모두 `UNION_s3`이다. shared `R0_ONLY` 및 원래 `R0_GEO` 각각에 대해 T 중앙값의 엄격한 개선 조건과 R P90의 5% 이내 조건을 통과하지 못했다. 다른 seed만 선택해 실사로 넘기거나 기준을 변경하지 않았다. 이번 시도의 source 실패를 새 실사 결과로 대신 설명하지 않는다.

[실사 evaluator 코드](../../../scripts/research/pallet_pose_selector_convergence_20261001_v1/evaluate_real.py)는 사용하지 않은 구현으로 공개한다. source 45/45 PASS와 4개 학습 완료·수렴 인증·checkpoint SHA가 확인돼야 다음 단계로 진입한다. 이후 별도의 `REAL_PROTOCOL`에 입력·코드를 고정하고, 기존 173장의 cached feature로 전체 pose의 R과 t를 함께 선택해 lock한 다음 별도 프로세스에서만 실사 참조를 읽도록 구현했다. 단일 RGB·치수·기존 K 계약을 유지하며, image inference·새 PnP·시간 정보는 필요하지 않다.

이 경로를 실제 데이터로 실행해 검증했다는 뜻은 아니다. 실행한 검사는 **인공 배열 self-check**와 AST 검사다. 인공 배열로 전체 pose 선택·동점·공통 fallback, 원래 bootstrap과 5% 경계·실패 보존, 다섯 안정성 gate와 모든 비교 대상, 실사 GT 차단을 확인했다. source 학습·성능 코드나 기존 실사 예측을 수정하지 않았다. 원래 metadata와 cached feature의 shape·ID 정합성을 읽어 재사용 가능성을 확인했지만 새 실사 선택 점수는 계산하지 않았다.

```bash
MPLCONFIGDIR=/tmp/pallet-stability-mpl python -m scripts.research.pallet_pose_selector_convergence_20261001_v1.evaluate_real self_check
```

위 명령의 결과는 `REAL_EVALUATOR_SELF_CHECK_PASS_INVENTED_DATA_ONLY`였다. `freeze`와 `score`는 실행하지 않았다. [상태·SHA 영수증](REAL_EVALUATION_NOT_RUN.json)에 확인 시점의 파일 상태를 남겼다.

| 새 실사 산출물 | 상태 |
|---|---|
| `REAL_PROTOCOL.json` / `REAL_PROTOCOL_SHA.json` | 생성·봉인하지 않음 |
| `REAL_CHOICES.json` / `REAL_ROUTING_LOCK.json` | 생성하지 않음 |
| `REAL_RESULTS.json` / `POSE_METRICS.json` / `REAL_FRAME_RESULTS.csv` | 생성하지 않음 |

기존 phase의 실사 결과는 그대로다. 이 unused evaluator의 코드 완성과 인공 테스트 통과를 **실사 T/R의 개선 증명으로 보고하지 않는다.**
