# 위험 감독과 지지 정보 실험

동일 D1 후보 bank에서 18회 새 fit을 실행했다. 모든238행 출력을 고정한 뒤 PRIMARY192/TEST49/REMAINDER143/CAL46을 평가했다. 독립 실사 test0장; 운영 기본값·논문 변경 없음.

대표는 seed20261018이다. 성공 수는 고정 세 seed 순서18/19/20이며 기준 단일 출력은 한 숫자다. 평균은 대표의 전체192장 값이다.

회귀는 H1보다 큰 손상을 줄이면서 회복 일부를 잃었다. F1은 ZERO보다, RGB는 NULL보다 세 seed 모두 성공 수가 낮았다. 운영 기본값은 유지하고, 다음 수정은 B 관측과 실제 변의 대응·지지 계약을 바로잡는 데 둔다. [판단의 수치 근거](DECISION_KO.md)를 함께 읽어야 한다.

| 방법 | 성공 /192 (모든seed) | 평균 위치cm | 평균 회전° | T>50/100 · R>45 |
|---|---|---|---|---|
| N3 + SubPix (N0) | 108 | 11.31 | 10.63 | 8/5/20 |
| Full statewise (E0) | 114 | 14.11 | 13.00 | 15/3/26 |
| E0 regenerated (frozen replay) | 114 | 14.11 | 13.00 | 15/3/26 |
| PoseFix + SubPix / PF anchor | 112 | 9.85 | 10.54 | 10/3/20 |
| PoseFix + SubPix / YOLO anchor | 111 | 9.38 | 10.59 | 9/2/20 |
| Existing full BCE (H1) | 120/117/121 | 12.07 | 11.55 | 9/5/21 |
| Existing two-action MLP | 117/113/114 | 10.39 | 8.89 | 6/5/16 |
| Existing two-action affine | 118/119/118 | 11.05 | 10.47 | 8/5/20 |
| Continuous regression / F0 | 117/119/120 | 10.97 | 10.49 | 8/5/20 |
| Pareto rank / F0 | 111/112/113 | 11.15 | 10.50 | 8/5/20 |
| Rank / F1 zero control | 110/118/114 | 11.13 | 10.57 | 8/5/20 |
| Rank / F1 information | 108/112/111 | 11.25 | 10.59 | 8/5/20 |
| Rank / visual NULL | 107/113/112 | 11.30 | 10.63 | 8/5/20 |
| Rank / actual RGB | 106/112/107 | 11.40 | 11.07 | 8/5/21 |

[전체 보고서](REPORT_KO.md) · [판단과 다음 한 가지](DECISION_KO.md) · [전체 결과 CSV](COMPARISON_ALL_ARMS.csv) · [모든192장 갤러리](gallery/index.html) · [재현](REPRODUCE.md) · [검산](VERIFICATION.json)

![전체 성공과 큰 오류](figures/success_and_large_errors.png)

![회복과 손상](figures/recovery_damage_and_common_failure.png)

GitHub는 HTML을 실행하지 않는다. 재현 안내의 로컬 서버로 상대 자산을 함께 열 수 있다. 원 RGB·private bank·weights는 공개하지 않았다. 결과 패널의 RGB는 사용자 권리가 확인된 PRIMARY 자료의 표시 사본이다. CAL/합성 RGB는 게시하지 않았다.
