# 첨부 필터 오통과 사례의 원본 Hough 및 점·선 결합

**사례 재검증 완료.** 실제 원본에서 Hough 선을 추출하고 GT 없는 고정 보정
네 가지를 실행했다. 일부 코너의 국소 위치는 개선됐지만, 큰 의미적 번호 오류를
고치거나 이 검출을 자동 거부하는 데 성공한 결과는 아니다.

## 1. 제안

[소비처] 사용자가 제공한 eval_pallet07:1778652166837872128 사례에서 허프 선의 사용 가능성과 점·선 결합의 조건을 판단한다.
[문장] 실제 raw 이미지에서 고전 Hough 선을 추출하면 경계 위치의 추가 근거를 얻을 수 있지만, 코너 번호 혼동과 물리 경계 위치 오류는 별도로 측정해야 한다.

원본 PNG와 첨부 overlay를 만든 prediction/GT/필터를 먼저 일치시킨다. 기존 동일 프레임 분석의 재구성 GT8점 수치를 첨부 M4의 GT지원9점 수치와 혼용하지 않는다. 기존 hough_line_visual 출력은 보존하며 case_recheck_20260908 하위에 새 분석을 둔다.

고정 비교: 원본 Canny(60,160), Gaussian5x5 sigma1.4와 기존 HoughLinesP 설정(rho1,theta0.5도,threshold60,minLength64,maxGap12)을 그대로 재실행한다. 짧은 높이선에 대한 별도 설명용 설정은 threshold20,minLength16,maxGap4다. 두 설정을 모두 보고하며 정답 성능으로 선택하지 않는다. 새 후보 대응/점 보정의 상수는 실행 전에 protocol JSON으로 저장하고 GT를 입력하지 않는다. 같은 프레임의 이미 완료된 학습형 점·선 분기3seed 출력도 출처를 확인해 비교한다.

판단 지표: 첨부 배너 오차 재현, raw/overlay 좌표 일치, 검출 선분 수 및 예측 경계 지지, 점 보정 전후 동일 GT-mask 오차, 의미적 순열을 허용한 사후 위치 오차,12개 무순서 선집합의 순열 불변성. GT를 이용한 순열이나 선 대응은 oracle 진단으로만 표시하고 운영 보정·필터로 주장하지 않는다.

예상 실패: 의자/차/격자 내부선, 짧거나 가려진 경계의 미검출, 선의 연장선에 있는 잘못된 점, 역할 순열을 구분하지 못하는 선집합, 초기 점 주변의 잘못된 선 선택. 한 장에 맞춰 filter threshold를 조정하거나 새 모델/선택규칙을 튜닝하지 않는다. 본 작업은 사례 진단이며 필터 precision/recall 또는 일반화 개선을 입증하는 실험이 아니다.

## 2. 결과

첨부는 [`pseudolabel_bad_accepted_01.png`](../advising/2026-09-professor-consult/figures/pseudolabel_bad_accepted_01.png)이며
`scripts/advising/make_pseudolabel_examples.py`가 M4의 F4 통과 사례 중 GT 최대
오차가 큰 순서로 고른 진단 예시다. 새 학습형 선 분기의 출력이나 실제 학습 pool
이미지를 보여주는 것이 아니다. R0 예측과 현재 선 분기의 `baseline_selected`는
좌표·box·score가 정확히 같다. 원본은
[`1778652166837872128.png`](../../challenge/data/01_real/manual_gt/capturepallet07_manual_gt/1778652166837872128.png)이고,
paper 평가 경로 이미지와 SHA가 같은640×480 PNG다.

배너의 **267.929px**는 GT 지원9점 중 **7개 코너+중심점, 총8점**의 median이다.
Mask는 `[T,T,T,F,T,T,T,T,T]`; GT3은 이미지 아래로 잘리고,5/6/8은 가려진 점이다.
이는 모든8코너의 median272.720px와 분모가 다르다. GT 원본을 바꾸지 않았다.

예측번호와 실제 코너 위치를 비교하면 `P0≈GT4, P1≈GT0, P2≈GT3, P3≈GT7,
P4≈GT5, P5≈GT1, P6≈GT2, P7≈GT6`의 순열이 주된 오류다. **정답을 사용해**
`P[[1,5,6,2,0,4,7,3,8]]`로 번호만 재배열하면 동일8점 median은25.706px,
최대40.874px다. 이 값은 사후 oracle이며 실제 보정 출력이나 운영 규칙이 아니다.
즉 번호 혼동과 수십 픽셀의 위치 오류가 함께 존재한다.

기존 `hough_line_visual.py`의273.700→21.184px는 geometry-reprojected GT의
모든8코너를 사용한 별도 수치다. 그 GT와 현재 manual GT의 차이도 평균3.782px,
최대7.457px였다. 따라서 기존 “번호 오류이므로 위치 오류는 아니다”는 설명은
너무 단정적이다. 비교할 때 GT 출처와 중심점·가시성 분모를 먼저 맞춰야 한다.

### 왜 원래 필터를 통과했는가

실제 적용된 `F4_PROPOSED` 조건은 box confidence≥.85, keypoint confidence≥.5인
유효 코너≥6, `s_remove≤.05`, `s_flip≤.05`다. 이 사례는 confidence .926704,
remove .023261, flip .049934로 통과한다. **표시된 reprojection .016857은 F4의
통과 조건이 아니다.** Flip scalar는 저장값과 소스 알고리즘을 확인했으며,
M4에 flipped 좌표가 없어 새 추론 없이 독립 수치 재계산했다고 주장하지 않는다.

Confidence는 팔레트 검출에 대한 값이며 코너 번호가 정답이라는 보장이 아니다.
Remove/flip은 예측의 일관성을 본다. 이 사례처럼 틀린 번호 배정이 내부적으로
일관되면 통과할 수 있다. 또한 기하 consistency score는 두 W/D 가설 중 작은
값을 사용하고 투영 대각선으로 정규화한다. 작은 값 자체가 정답의 의미적 축·면
배정에 대한 독립 검증은 아니다. 이 한 장을 보고 기존 threshold를 바꾸지 않았다.

### 원본 Hough와 보정 결과

Raw 전체 이미지에 Gaussian5×5/σ1.4, Canny60/160, HoughLinesPρ1px/θ0.5°를
적용했다. Long은 threshold60/minLength64/maxGap12로36개 선분, short는
threshold20/minLength16/maxGap4로146개 선분을 얻었다. 팔레트 외에 의자·차·
시설물·내부 격자 선도 포함한다. 검출 자체에는 GT나 예측 코너를 입력하지 않았다.

그다음 예측 코너가 만드는 선분과 Hough 선분을 GT 없이 연결했다. 각도≤12°,
서로의 midpoint에서 측정한 평균 법선거리≤예측 box 대각선3%, 화면 안 예측
선분에 대한 투영 겹침≥20%를 요구했다. 정규화 cost가 낮은 순서로 일대일
greedy 연결하고, point anchorσ=.025×box대각선·lineσ=.01×box대각선(각각1px
하한), λ1의 WLS를 적용했다. 원본 이미지 대각선1%=8px로 각 점의 이동을 제한하고
중심점·결측은 보존했다. 초기 예측에 조건화된 국소 보정이며 전역 순열 복구 모델이 아니다.

| 고정 방법 | 연결된 역할 | 동일 GT 지원8점 median, px |
|---|---:|---:|
| R0 원본 | — | 267.929 |
| Long + side8, primary graph | 2/8 | 267.922 |
| Long + cuboid12 | 5/12 | 269.403 |
| Short + side8, primary graph | 2/8 | 267.931 |
| Short + cuboid12 | 4/12 | 268.989 |
| 기존 학습형 image_joint seed1 | — | 268.621 |
| 기존 학습형 image_joint seed2 | — | 268.431 |
| 기존 학습형 image_joint seed3 | — | 268.263 |

두 Hough 설정과 두 그래프를 모두 보고했으며 GT로 더 좋은 설정을 선택하지 않았다.
기존 학습형 출력은 이미 완료된 동일 원본 이미지 추론을 읽은 것으로, 이번 사례에
맞춰 재추론·재학습·규칙 조정을 하지 않았다. 어느 방법도 이 사례의 큰 번호 오류를
해결하지 못했다. 기존 학습형 분기 역시 초기 예측 주변 후보에서 결합하므로
여기서의 전역 번호 복구 성능을 입증하지 않는다.

국소적인 가능성은 확인됐다. Short+side8에서 이미지 왼쪽의 실제 상·하 코너
두 개를 GT4/GT7 위치와 사후 비교하면 P0의 거리12.639→6.788px,
P3의 거리9.612→5.246px로 줄었다. **Hough/WLS는 GT 없이 움직였고**, 어느 물리
코너와 비교할지는 oracle 대응으로 설명한 것이다. 번호가 유지되어 전체 indexed
median은267.931px로 남았다. 이 두 점의 개선을 전체 정확도나 필터 성공으로
확대하지 않는다.

## 3. 고전 Hough로 가능한 것과 한계

유한 선분이 실제로 관측되는 곳에서는 점의 법선 방향 위치를 보정할 근거를 얻을
수 있다. 무한 연장선이 점을 지나간다는 것만으로는 부족하므로, 이번 보정은
화면 내 유한 길이와 겹침을 요구했다. 한 선은 접선 위치를 정하지 못한다.
두 선의 대응이 맞는지, 거의 평행하지 않은지, 점 anchor와 충돌하지 않는지를
함께 봐야 한다. Positive anchor와8px cap은 수치적으로 큰 이동을 막지만 잘못된
선 연결 자체를 옳게 만들지는 않는다.

이 사례의 순열은 **12개 무순서 cuboid edge 집합을 정확히 보존**한다. Height는
height끼리 바뀌고 side8의 depth4개는 width4개와 교환된다. 따라서 팔레트 주위에
선이 잘 맞아도 어느 면·축·코너 번호인지 틀릴 수 있다. 이 그래프 불변성은 실제
좌표가 GT와 같다는 뜻도, W/D가 다른 물체에서90°를 물리적 대칭으로 허용한다는
뜻도 아니다.

실제로 전체12선의 방향·거리 조건을 만족하는 Hough support 평균은 다음과 같다.
각 선을101점 샘플링하고3px 이내·방향12° 이내인 실제 유한 선분 비율의 선별
평균이며, 학습하거나 보정한 신뢰확률이 아니다.

| Hough 설정 | 잘못된 R0 선 집합 | Manual GT 구조선 집합 |
|---|---:|---:|
| Long | 18.6% | 9.4% |
| Short | 19.7% | 13.6% |

즉 단순 line support가 높은 쪽을 정답으로 보거나 낮은 점수로 reject하는 규칙은
이 사례에서 검증되지 않았다. 내부 격자·다른 물체의 경계와 겹칠 수 있고, GT도
amodal 구조선이라 가려진 곳에 영상 edge가 반드시 존재하지 않는다. 같은 예측을
oracle로 재인덱싱해도 전체12선 support는 정확히 같다.

별도 GT-assisted matching은 두 설정 모두 **GT 지원 구조선9개 중3개**가 후보에
있다고 기록했다. 이3/9는 숨은 구조점까지 포함하는 GT 지원 조건과 특정 matching
기준의 결과다. **물리적으로 보이는 edge의 recall이나 Hough의 전체 검출률이
아니다.** Visibility>0는 점의 감독 가능성을 뜻하며 선 전체의 실제 가시성 정답이
아니다. 가려진 코너에서 gradient가 없다는 이유만으로 잘못된 점이라고 판정해서는
안 된다.

## 4. 점·선 시너지를 위한 후속 설계

이번 결과가 지지하는 방향은 독립적인 영상 단서와 의미적 역할 판단을 보강하는
것이다. 다음은 후속 가설이며 구현·성능이 검증된 결과로 읽지 않는다.

1. 초기 코너 주변뿐 아니라 원본 영상에서 얻는 선/면 후보를 따로 유지한다.
   각 후보의 팔레트 소속, 높이·폭·깊이 역할, 가림/관측 신뢰도를 추정하고
   의자·트럭·내부 격자와 혼동하는 hard negative를 포함한다.
2. 번호/면/축 가설과 국소 좌표 오차를 분리한다. 여러 역할 배정을 비교하되
   영상 단서가 구별하지 못하면 모호성을 유지하거나 baseline으로 돌아간다.
   알려진 치수와 calibration은 해당 object에 실제로 지원되는 경우에만 추가
   근거로 사용한다. 임의 팔레트에 고정 치수를 강요하는 일반 모델을 가정하지 않는다.
3. 신뢰할 수 있는 유한 선분만 robust point anchor와 결합하고, 가림·충돌·
   평행성·과도한 이동에서는 null/fallback을 사용한다. 선 support 하나를
   semantic correctness의 대용 지표로 쓰지 않는다.
4. 학습·불확실성 보정·gate 선택은 분리된 합성 train/calibration/selection에서
   정하고 고정한 뒤 heldout과 다중세션 실사에서 평가한다. 순열 오류·국소 위치
   오류·가림·쉬운 프레임을 나눠 paired 개선/악화, 미검출·결측 분모, coverage,
   baseline fallback, 최종2D/6D와 시간을 함께 보고한다. 같은 실사 결과로
   threshold나 좋은 seed를 다시 고르지 않는다.

## 5. 근거와 재현

완료된 원본 산출물은 보존한다.
[사례 보고서](../../data/pallet/results/hough_line_visual/case_recheck_20260908/index.html),
[RESULTS](../../data/pallet/results/hough_line_visual/case_recheck_20260908/RESULTS.json),
[GEOMETRY_AUDIT](../../data/pallet/results/hough_line_visual/case_recheck_20260908/GEOMETRY_AUDIT.json),
[출처·필터 검증](../../data/pallet/results/hough_line_visual/case_recheck_20260908/provenance/provenance.json),
[현재 학습형 사례](../../data/pallet/results/hough_line_visual/case_recheck_20260908/provenance/CURRENT_LEARNED_CASE.json),
[실행 전 PROTOCOL](../../data/pallet/results/hough_line_visual/case_recheck_20260908/PROTOCOL.json)을 따른다.
보고서 링크는 같은 결과를 표시하는 HTML이며 숫자 정본은 JSON이다.
프로토콜 SHA는 `f17180efad3a5706f12b6366cfe52d885495a4820f630115bd03880b26485b3e`다.

재현은 기존 RESULTS를 덮어쓰지 않는 새 출력 폴더에서 한다. 아래 명령은
기존 protocol·provenance를 복사하고 같은 raw CPU 실험과 기하 감사를 수행한다.
새 신경망 추론이나 학습은 포함하지 않는다. 폴더가 이미 있으면 다른 이름을
사용한다. 이 문서 갱신 중에는 명령을 실행하지 않았다.

```bash
set -e
HOUGH_CASE_REPO=/home/minjae/Documents/github/pallet-pose
HOUGH_CASE_PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
HOUGH_CASE_SOURCE="$HOUGH_CASE_REPO/data/pallet/results/hough_line_visual/case_recheck_20260908"
HOUGH_CASE_REPLAY="$HOUGH_CASE_REPO/data/pallet/results/hough_line_visual/case_reproduction_001"
mkdir "$HOUGH_CASE_REPLAY"
cp "$HOUGH_CASE_SOURCE/PROTOCOL.json" "$HOUGH_CASE_SOURCE/PURPOSE.md" "$HOUGH_CASE_REPLAY/"
cp -r "$HOUGH_CASE_SOURCE/provenance" "$HOUGH_CASE_REPLAY/provenance"
cd "$HOUGH_CASE_REPLAY"
"$HOUGH_CASE_PY" "$HOUGH_CASE_REPO/scripts/research/hough_case_geometry.py" --self-test
"$HOUGH_CASE_PY" "$HOUGH_CASE_REPO/scripts/research/hough_case_probe.py" --output "$HOUGH_CASE_REPLAY"
"$HOUGH_CASE_PY" "$HOUGH_CASE_REPO/scripts/research/hough_case_geometry.py" --output-dir "$HOUGH_CASE_REPLAY"
```

GT 없는 association/WLS의 null identity, 중심점 보존, 두 직교선의 해석적
anchor 해, 유한·화면 내 후보,8px cap, 결측 보존, 무방향 선분 반전 불변성,
일대일 대응 검사8개가 PASS다. 이 단위 검증은 기하 구현의 증거이며 한 사례를
넘는 보정 성능이나 자동 필터 precision/recall의 증거는 아니다.

실행·시각화 마무리: 원본/Hough/점 보정의 독립 수치 검사와 브라우저 상호작용 QA를 모두 통과했다. HTML 창 표시를 확인했고 Discord 완료 알림도 HTTP204로 전송했다. [완료 기록](../../data/pallet/results/hough_line_visual/case_recheck_20260908/COMPLETION.json)의 PASS는 실험 유효성이며, 이 사례를 해결했다는 뜻은 아니다.
