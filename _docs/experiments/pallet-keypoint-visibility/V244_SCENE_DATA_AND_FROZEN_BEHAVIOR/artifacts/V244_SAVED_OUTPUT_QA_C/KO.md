# Frozen TREAT own-branch 저장 산술 QA

원 7,617행(SYN 6,096/YCB 1,521, FIT 5,876/DEV 1,741)을 모두 보존해 독립 stdlib 검산했습니다. 원 source/feature/checkpoint/GT/RGB를 열지 않았습니다.

| family/role | D: ownbase→total | E: ownbase→total |
|---|---:|---:|
| 0/FIT | 4565개 중 4529→4528 | 187개 중 187→187 |
| 0/DEV | 1288개 중 1273→1272 | 56개 중 51→51 |
| 1/FIT | 841개 중 839→841 | 283개 중 274→283 |
| 1/DEV | 340개 중 333→332 | 57개 중 0→1 |

YCB DEV E 57개는 ownbase 정답 0개→total 1개입니다. 정답 방향 residual은 1개, 반대 방향은 56개, 0은 0개입니다. 각 branch의 median은 ownbase -2.234952211, residual -6.674595356, total -9.046613693입니다.

동일 TREAT checkpoint의 내부 base와 residual 분해입니다. 별도 학습 CONTROL 비교가 아니며 residual의 sign-only CM은 독립 classifier 성능이 아닙니다. SYN intended-native witness0와 YCB associated-r AUX를 합쳐 하나의 의미로 해석하지 않습니다. SOURCE DEV는 이미 노출·재사용된 진단이고, 이 결과는 시각적 가림의 인과 원인·실사 해결·모델 선택의 증거가 아닙니다. current admission은 false이며 새로운 학습/현재셋 forward/문턱 변경은 없습니다.

[독립 검산 REVIEW](actual_own_branch_review_v1/REVIEW.json) · [검산 코드](verify_saved_branches.py) · [실행 안내](README_KO.md)
