
## 사후분석

| 사건 | corner 수 | band 분포 | corner ID 분포 | session 분포 |
|---|---:|---|---|---|
| preservation_wins | 6 | {'B1': 5, 'B2': 1} | {'4': 1, '5': 2, '3': 2, '2': 1} | {'eval_night09': 1, 'eval_pallet07': 5} |
| preservation_failures | 16 | {'B2': 4, 'B1': 11, 'B0': 1} | {'0': 3, '5': 2, '3': 1, '4': 5, '6': 3, '7': 2} | {'eval_night08': 1, 'eval_night09': 4, 'eval_outside': 1, 'eval_pallet07': 7, 'eval_pallet09': 3} |
| new_preservation_damage | 5 | {'B2': 1, 'B1': 4} | {'5': 1, '4': 3, '6': 1} | {'eval_night09': 1, 'eval_outside': 1, 'eval_pallet07': 2, 'eval_pallet09': 1} |
| lost_FULL_recovery | 1 | {'B3': 1} | {'2': 1} | {'eval_pallet07': 1} |
| retained_FULL_recovery | 5 | {'B3': 5} | {'0': 1, '3': 1, '5': 1, '6': 1, '1': 1} | {'eval_pallet07': 5} |
| new_recovery | 0 | {} | {} | {} |

### PRIMARY의 R0 band별 gain/loss

| band | 코너 | FULL 정답 | FP 정답 | FULL BASE gain/loss | FP BASE gain/loss |
|---|---:|---:|---:|---|---|
| B0 | 116 | 115 | 115 | 0/1 | 0/1 |
| B1 | 192 | 166 | 166 | 6/12 | 5/11 |
| B2 | 155 | 72 | 73 | 24/4 | 25/4 |
| B3 | 99 | 6 | 5 | 6/0 | 5/0 |
| B4 | 97 | 0 | 0 | 0/0 | 0/0 |
| B5_MATCH_FAILURE | 54 | 0 | 0 | 0/0 | 0/0 |

### 보정 이동량 (원래 R0 대비, 정확도 아님)

| method | median px | P90 px | max px |
|---|---:|---:|---:|
| R0 | 0.000 | 0.000 | 0.000 |
| BASE | 2.681 | 6.012 | 21.946 |
| N2 | 2.306 | 6.282 | 8.000 |
| FULL | 4.350 | 10.620 | 25.977 |
| FULL_PRESERVE | 3.964 | 10.301 | 43.480 |

## 한계와 사람 검토

실사253장은 한 recording의 인접 프레임이고 pseudo target 자체가 오답일 수 있다. Replay teacher의 동일 recording 수동학습 노출도 기존과 같다. 평가 recording 제외·기존 center/box/score 유지. 아래 GT는 평가·그림에만 사용했으며 추론·보존 mask·λ에는 미사용. 가림 이미지를 평가했지만 human visibility review 없이는 외부가림 코너 복원이라고 주장하지 않는다. 최종 논문 모델/표는 교체하지 않았다.

[다음 분기 및 질문별 답](NEXT_STAGE_PLAN.md) · [데이터 다양성 감사](DATA_DIVERSITY_INVENTORY.md) · [사후분석](POSTMORTEM_KO.md) · [전체 HTML](GALLERY.html)
