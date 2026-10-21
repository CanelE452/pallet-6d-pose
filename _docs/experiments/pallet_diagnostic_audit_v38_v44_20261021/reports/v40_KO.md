> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# 실사 CAL12 저장 관측 비교 v40

이번 표는 **원 OCPR(D1+H1 MLP, seed20261018) 대 N0**의 기존 기록이다. TRAIN v39의 v5 incumbent 분석과 섞지 않는다. 새 성능평가·추론·학습이 아니다. CAL46도 여러 결정에 사용된 개발자료이며 독립 확인셋이 아니다. 오차는 기존 기하 재구성 참조에 대한 것으로 독립 물리 GT가 아니다.

## 표본과 시각 검사 순서

원 OCPR의 공동5cm/5° 실패7장과 성공39장 중 각 최대6장을 고정했다. 촬영 묶음 순서 round-robin, 묶음 내 첫 ID 이후에는 이미 고른 숫자 suffix와 거리가 먼 ID 순서(동률 lexID)를 사용했다. 사진의 그림자나 특정 PRIMARY 사례와의 유사성을 보고 고르지 않았다. numeric suffix의 실제 시간 단위를 검증한 것은 아니므로 시간 간격/독립성을 주장하지 않는다.

선정12장은 촬영 묶음4개뿐이다. 실패는2개 묶음의3장씩, 성공은3개 묶음의2장씩이며 한 묶음은 양쪽에 겹친다. 같은 촬영의 여러 프레임을 독립 표본처럼 세지 않는다. 또한 CAL의 유일한 R>45° 사례(84.116°)는 이 사전 고정 표본에 포함되지 않았다. 결과를 본 뒤 교체·추가하지 않았다. 따라서 이12장은 큰90° 분기의 대표 표본이 아니다. 큰 회전 TRAIN4장과 서로 다른 분석 수준이다.

saved detector box에20% 여유를 둔 crop으로 예측선 없는 두 contact sheet를 먼저 직접 열었다. 경계·외부 가림·방향 단서·어두운 영역의 모호성 메모를 저장한 뒤, mask/실제 solver 관측/pose 비교 그림을 만들고12장을 모두 직접 검수했다. 이는 assistant의 보조 시각 점검이며 독립 사람 주석·완전한 blind 평가가 아니다. 물리 W/D 번호는 모든 사례에서 UNKNOWN으로 남겼다. 방향성 있는 판재·리브가 보이는 것과 물리 W/D 대응 정답을 아는 것은 다르다.

## 선정 표본의 공통점과 반례

| 항목 | 실패 표본6장 | 성공 대조6장 | 해석 범위 |
|---|---:|---:|---|
| 원 raw/eligible H1 bank에 성공 후보 존재 | 1/6 | 6/6 | 후보 선택 실패와 후보 부재를 분리해야 함 |
| 좋은 후보 최종 미선택 | 1/6 | 0/6 | 실패 C03만 해당; 전체192 비율 아님 |
| 깨끗한 crop에서 명확한 외부 물체 가림 | 3/6 | 1/6 | 실패3장은 같은 cone 촬영 묶음이라 독립 원인효과 아님 |
| initial_self_hidden에 의한 S gate 존재 | 6/6 | 6/6 | 모두 S image query0; 실제 visible을 잘못 지웠다는 정답은 없음 |
| N0–최종 비용의 A/AB 방향 충돌 | 0/6 | 0/6 | 두 family 각각, 두 후보 모두 feasible인 mode 비교 |
| 광범위한 외부 배경을 포함하는 full mask의 명확한 시각 단서 | 0/6 확정, 1/6 불확실 | 1/6 확정, 1/6 불확실 | C08 성공 대조가 중요한 반례. 이를 특정 그림자 오류로 명명하지 않음 |

A/AB 모델 이름은 관측 역할 B/I와 다르다. A와 AB의 두 후보는 모든12장에서 feasible했다. 같은 pose를 선택한 C05는 비용 tie이고, 나머지 실패5장/성공6장은 A와 AB 모두 최종 OCPR pose의 raw 비용이 N0보다 작았다. 따라서 ‘좋은 출력의 관측 비용을 H1이 무시한다’는 설명이 이 표본 전체에 맞지는 않는다. 비용이 최종 선택을 지지한다는 사실도 물리 owner가 맞다는 증거는 아니다.

## 실제 사례와 사용 관측 수

| 사례 | 결과 | 저장 T/R | full 점 / partial group | bank 성공 포함 | 눈으로 본 제한 |
|---|---|---|---|---|---|
| C01 | 실패 |14.343cm /4.913°|2 /0|아니오|cone 가림, 얕은 시점, 어두운 몸체|
| C02 | 실패 |2.256cm /7.092°|3 /3|아니오|wood 외곽과 그림자는 대체로 분리됨|
| C03 | 실패 |7.844cm /1.713°|2 /1|예|좋은 N0가37위; full/tight 마스크 불일치·외부 픽셀 단서|
| C04 | 실패 |7.699cm /10.447°|4 /3|아니오|명확한 판재 방향이 있어도 물리 대응 정답 미확인|
| C05 | 실패 |13.742cm /2.867°|0 /1|아니오|단일 부분선, N0를 그대로 선택|
| C06 | 실패 |13.121cm /5.309°|6 /1|아니오|mask가 긴 바닥 그림자를 대체로 제외함|
| C07 | 성공 |1.532cm /0.817°|5 /1|예|stool 가림·mask의 외부 물체 포함 단서도 존재|
| C08 | 성공 |1.454cm /0.668°|4 /1|예|full mask는 넓은 배경/캔버스까지, tight는 팔레트에 가까움|
| C09 | 성공 |0.719cm /1.020°|7 /0|예|실패 wood 사례와 같은 촬영 묶음; 강한 바닥 그림자|
| C10 | 성공 |2.253cm /1.759°|4 /1|예|거의 측면 시점·어두운 몸체, W/D 식별 불확실|
| C11 | 성공 |0.790cm /0.665°|6 /0|예|불균일 조명에도 outline은 대체로 보임|
| C12 | 성공 |0.139cm /0.467°|4 /2|예|wood, 큰 배경 그림자를 mask가 대체로 제외함|

full 점은 partial로 대체된 점을 빼고 센 실제 likelihood다. 부분선은2D corner 좌표가 아니고 tangent 구간은 실제 범위 제약이다. gray cross는 저장 raw forecast, 색 원은 사용 full point, 색 선은 실제 관측 inlier 구간, 마지막 열의 상자 꼭짓점은 저장 pose 투영이다. 서로 혼동하지 않는다. native 번호/면 연결은 초기 모델에 조건부이며 독립 물리 owner GT가 아니다.

C03의 실패는 T 오차이며90° 사례가 아니다. 원 N0는 성공, 최종 G10/ORIGINAL_WD/selected는 실패이고 N0 순위37이다. A/AB도 최종을 지지한다. 반면 다른 실패5장은 이 정확한 raw/eligible bank에 성공 후보가 없어서 selector 변경만으로 joint 성공을 복원할 수 없다. 이 정의를 후보 확장 bank나 oracle 상한과 섞지 않는다.

## 결론·입력 한계

‘사람도 헷갈리는 영상’이라는 가능성은 얕은 시점·외부 가림·어두운 접촉 영역에서 실제로 관찰된다. 그러나 실패만의 특징은 아니다. 성공 대조에도 외부 가림, 어두운 몸체, 거의 측면 시점, 비정상 full mask가 있다. wood 실패 중에는 mask가 바닥 그림자를 잘 제외하는 사례도 있다. 따라서 그림자/mask 문제 하나로 모든 실패를 설명하거나 이 신호로 감점·제거하는 규칙을 만들지 않는다.

실사에 독립 물리 경계/owner 정답이 없으므로, 관측선이 실제 팔레트의 어느 edge인지와 cuboid/실제 재질 silhouette 차이를 더 분리하려면 해당 관측 구간만의 독립 판독 또는 검증된 render provenance가 필요하다. 가려진 corner를 사람이 추측해서 찍으라는 요구가 아니다. ‘물체 edge/면, 외부 물체, 바닥, 구별 불가’를 판단하고 애매하면 UNKNOWN으로 남겨야 한다. source mask 자체는 그 정답으로 쓰지 않는다.

계산은 CPU1개4.890초,10분 cap 이내였다. 새 SAM/PnP/model forward/학습/문턱변경/전체 성능평가/공개 게시0. stored full/tight mask, 실제 solver packet과 관측, 기존 R/t·후보 순위만 사용했다. 두 저장 pose의 양쪽 family×5비용 및 첫26특징/validity는 저장값과 정확히 일치했다.

## 파일·재현

- SELECTION_PRIVATE_BEFORE_VISUAL.json / SELECTION_SUMMARY.json: 사전 표본과 촬영 묶음 수
- CLEAN_VISUAL_NOTES_BEFORE_OVERLAYS.json: overlay 전 보조 시각 메모
- OVERLAY_VISUAL_NOTES.json: overlay 후12사례 판독 및 UNKNOWN
- RESULTS/CASE_TABLE_V2.csv: 실제 방법/분모/metadata/점·선 수/초기 S/후보 포함/순위/비용 충돌
- RESULTS/BOTH_FAMILY_COSTS.csv / STORED_CANDIDATE_RANKS.csv: 수치 근거
- CLEAN_CONTACT_01_06.png / CLEAN_CONTACT_07_12.png: 예측선 없는 crop
- RESULTS/PANELS/C01.png … C12.png: 동일 crop의 mask/관측/저장 pose 비교
- RESULTS/OVERLAY_CONTACT_*.png: 전체12사례 검수용 contact sheet

기존 pallet-pose Python과 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1에서 select_and_clean_crops.py → clean crop 검수/메모 동결 → analyze_and_render.py 순서다. 원 결과 directory는 exclusive create로 중복 실행을 거부하므로 삭제·덮어쓰기하지 않는다. 새 재현은 출력 root만 분리한 복사본에서 수행한다.
