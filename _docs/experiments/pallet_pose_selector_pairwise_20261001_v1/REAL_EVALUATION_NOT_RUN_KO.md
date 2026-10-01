# 이번 선택기의 새 실사 평가를 실행하지 않음

**Source VAL 45개 조건 중 29개만 통과하고 16개가 실패하여 새 실사 routing·T/R 평가를 실행하지 않았다.** [고정된 source gate](SOURCE_VAL_GATE.json)의 `PASS=false`, `real_routing_authorized=false`를 그대로 적용했다. 세 신규 UNION fit의 수렴 인증과 source T/R 품질 판정을 구분한다.

실패는 `UNION_s1` 5개, `UNION_s2` 3개, `UNION_s3` 8개다. 공통 R0_ONLY와 원래 R0_GEO에 대한 T/R 중앙값 또는 P90 조건에서 발생했다. 통과한 검사나 특정 seed만 골라 실사로 넘기거나 기준을 변경하지 않았다. 재사용한 R0_ONLY wrapper는 원래 수치 파라미터를 유지하며 새 학습으로 세지 않는다.

[실사 evaluator 구현](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/evaluate_real.py)은 준비됐지만 사용하지 않았다. source 45/45 PASS, 신규 3fit+R0 재사용 1개의 영수증·수렴 인증·원본 R0 weight/정규화 bit 일치를 요구한다. 그 뒤 별도 REAL_PROTOCOL을 봉인하고, 기존 173장의 feature로 전체 R,t 선택을 고정한 다음 별도 프로세스에서만 실사 참조를 읽는 구조다. 단일 RGB·치수·기존 보정 K 계약을 유지한다.

실행한 것은 **인공 데이터 self-check와 AST 검사**다. 전체 pose 선택·동점·공통 fallback, bootstrap·5% 경계·실패 보존, 다섯 안정성 gate와 비교 대상, GT 경로 차단을 확인했다. 별도 에이전트가 신규 schema·3fit/1reuse·원본 파라미터·source gate·whole-pose 선택 흐름을 읽고 검토했지만, 실제 실사 경로를 실행해 검증했다고 보고하지 않는다. 기존 성능 코드와 실사 예측은 수정하지 않았다.

```bash
MPLCONFIGDIR=/tmp/pallet-stability-mpl python -m scripts.research.pallet_pose_selector_pairwise_20261001_v1.evaluate_real self_check
```

결과는 `REAL_EVALUATOR_SELF_CHECK_PASS_INVENTED_DATA_ONLY`였다. `freeze`와 `score`는 실행하지 않았다. [상태·SHA 영수증](REAL_EVALUATION_NOT_RUN.json)에 source 실패와 파일 부재를 기록했다.

| 새 실사 artifact | 확인 결과 |
|---|---|
| REAL_PROTOCOL / REAL_PROTOCOL_SHA | 생성·봉인하지 않음 |
| REAL_CHOICES / REAL_ROUTING_LOCK | 생성하지 않음 |
| REAL_RESULTS / POSE_METRICS / REAL_FRAME_RESULTS | 생성하지 않음 |

앞선 natural99·clean29·wood45 결과는 그대로다. **Unused evaluator 구현·인공 테스트·코드 검토 통과는 실사 T/R 개선의 증거가 아니다.**
