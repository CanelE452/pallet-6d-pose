# 진행 상태 — GPU 규칙에 따른 clean stop

> 아래는 이전 GPU 중단 시점의 기록이다. 이후 사용자 재개 요청으로 실사 평가·통계·기전·runtime·최종 감사를 완료했다. 최신 과학적 판정은 [FINAL_DECISION_KO.md](FINAL_DECISION_KO.md), Git 전송 확인은 raw GIT_PUSH_RECEIPT.json 및 최종 CLI를 따른다.

이 작업은 **아직 완료되지 않았다**. P 3개 학습과 synthetic 평가까지 완료했지만, real DEV inference 시작 직전에 외부 GPU compute가 발견되어 요청의 clean-stop 규칙을 적용했다. 최종 architecture verdict/FINAL_AUDIT/PASS는 만들지 않았다.

2026-09-13 19:27:51 KST: RTX3080, 57°C, 전체 사용1742MiB, 외부 PID1545669 `.venv/bin/python` 사용1288MiB. 이 프로세스를 종료하거나 GPU를 반복 polling하지 않았다. CUDA는 정상이며, 드라이버 변경·재부팅·CPU inference fallback도 하지 않았다.

## 완료

- A: 원래 `TASK_RISK_AL_NO_SIGNAL` 유지. 기존 QA flag145/clean0이어서 clean-only 비교는 추정 불가다. 영향도 및 PnP 대안을 비학습 재계산했다. 문제 프레임은 원본640px 오른쪽 밖의 reflect-padding 검출을 최고 score로 선택한 것이 upstream 실패임을 확인했다. 후보 교체나 selector 수정은 하지 않았다.
- B: source/parameter/evidence/geometry/order locks; 학습 전35개 테스트, 추가 pipeline 포함39개 테스트 통과. 이후 초기 C2 테스트의 permutation 이름 오류를 공개하고 실제 정본 yaw180 검사를 추가해 총40개 통과했다. 학습/추론에는 해당 permutation을 적용한 적이 없으며 모델·성능 변화는 없다(`C2_TEST_LABEL_CORRECTION.md`).
- P params18,962 vs L19,810. 같은 batch16,6000updates,3seeds; 총18,000updates, seed당96,000exposures. 기존 L sampler와 순서 exact. R0/L 재학습0.
- 마지막 checkpoint3개 저장·SHA 고정. 실제 학습시간 seed1 353.91s, seed2 362.40s, seed3 364.82s.
- Synthetic만으로 T=[1,1,1], shared lambda1, cap=raw image diagonal1% 선택·고정 후 heldout1985 평가.

| 모델 | Synthetic heldout pooled9 median px | P90 px |
|---|---:|---:|
| R0 | 1.896904 | 6.884871 |
| L1 | 1.741809 | 6.544901 |
| L2 | 1.746558 | 6.519632 |
| L3 | 1.755061 | 6.499972 |
| P1 | 1.735602 | 6.621681 |
| P2 | 1.750109 | 6.654996 |
| P3 | 1.736823 | 6.615881 |

이 표는 synthetic 결과일 뿐, real primary 또는 line-specific claim의 판정이 아니다. `SYNTH_DIAGNOSTIC_FIELD_CLARIFICATION.json`은 R0 row에 담긴 dormant P1 null probability를 R0에는 해당 없음으로 명시한다. 성능·선택·예측은 바뀌지 않았다.

## 남은 작업 / 재개

외부 GPU 작업이 종료된 뒤 **재학습 없이** 다음 순서로 재개한다. 기존 완료 artifact와 설정은 고정한다.

1. `evaluate_point.py infer` — 319+2689 실제 추론, 세 seed.
2. `evaluate_point.py score` — CPU canonical 평가.
3. `statistics_and_mechanism.py` — paired10k 통계·기전.
4. `evaluate_point.py runtime` — 기존26frames/5warmup/3repeats 실제 측정.
5. `audit.py` — 최종 source/학습/출력/통계/성능과 별개의 integrity 감사.
6. `report.py`의 patch로 final report 생성, 새 code/docs만 main commit/push, local/origin/main SHA 일치 확인.

`PARTIAL_INTEGRITY_AUDIT.json`의 PASS는 완료한 단계의 보존 확인일 뿐 final/performance PASS가 아니다. 이 턴에서는 요청된 최종 workflow가 미완료이므로 push하지 않았다. 기존 capacity_screen 미추적 파일은 건드리지 않았다.
