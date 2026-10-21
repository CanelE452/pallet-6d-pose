> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# 비교 방법 성공 / 원 OCPR 실패: 저장 출력 감사 v44

**CAL46에서 해당 현상은 중복 없는 1장, PRIMARY192의 사후 진단에서는 19장이다. 좋은 후보 미선택과 후보 부재가 모두 있으며, H1만 고치면 해결된다는 근거는 아니다. 이번에는 방법을 바꾸거나 새 성능을 실행하지 않았다.**

## 비교 계약과 분모

원 D1+H1 `D1_H1_MLP_s20261018`을 고정했다. v5 이후 방법은 섞지 않았다. 동일 ID/hash, recording, K, 등록 치수, 고정 C2(I/Y180) 기하 참조를 사용하는 저장 오류를 join했다. 성공은 T≤5cm AND R≤5°다. 각 비교의 CAL 공통 산출46/전체46, PRIMARY 공통 산출192/전체192이며 미산출0이다. 새 pose 오류 계산·학습·forward·SAM·SubPix·PnP·비용 함수 재실행은0이다.

CAL은 이미 반복 사용한 개발 자료다. PRIMARY는 이미 노출된 저장 결과의 사후 진단이며 독립 검증이 아니다. 참조는 기존 재구성 기하 참조이고 독립 물리 GT라고 하지 않는다. 원 source revision은 `93c24d143fa3901de1fa127b3156e8250d81dba8`, 원 저장소 보존 HEAD는 `7e92fdcefb0a37bdee0aef95d3e86c572e23b967`이다. source 파일 hash와 private 로컬 경로는 별도 보존했다.

| 비교 방법 | CAL 성공/46 | 비교 성공·OCPR 실패 | OCPR 성공·비교 실패 | PRIMARY 성공/192 | 비교 성공·OCPR 실패 | OCPR 성공·비교 실패 |
|---|---:|---:|---:|---:|---:|---:|
| PalletPoseFix |32|1|8|80|10|50|
| SubPix-only |31|1|9|98|12|34|
| PF+SubPix PF-anchor |32|1|8|112|15|23|
| PF+SubPix YOLO-anchor |32|1|8|111|15|24|
| N0 (N3+SubPix) |34|1|6|108|9|21|

원 OCPR은 CAL39/46, PRIMARY120/192 성공이다. CAL의 위1장은 모든 비교 방법에서 **같은 프레임**이다. PRIMARY 방법별10/12/15/15/9를 합산하지 않았다. union은 **19/192**, recording_8509720729881a2bb718의8장/recording_367c657b347d84679fdc의11장이다. 같은 recording의 여러 프레임을 독립 촬영처럼 세지 않는다.

실패 경계로 분류하면 CAL1장은 T만5 초과, PRIMARY19장은 T만5 초과12장·T/R 모두5 초과7장·R만5 초과0장이다. R>45°는 PRIMARY4장이다. 이 R>45가 곧90° W/D swap이라는 뜻은 아니다. PRIMARY19장 중 원 eligible bank에 성공 후보 있음14장/부재5장이다.

‘T만 실패’와 ‘오류의 delta가 T만 악화’는 다르다. CAL의 PFraw/SubPix 대비는 T/R 둘 다 조금 커졌으나 R은5° 이하다. PF+SubPix/N0 대비는 T만 커지고 R은 작아졌다. `METHOD_COUNTS_WITH_DELTA_SIGNS.csv`에 이 두 정의를 분리했다. 다른 seed는 `SEED_STABILITY_AUXILIARY.csv`이며 대표 결과에 합치지 않았다. CAL1장은 세 seed 모두 같은 pose; PRIMARY의 N0성공/OCPR실패 수는 seed18/19/20에서9/12/9다.

## CAL 핵심 사례 A01: 선택 손상, 그러나 비용도 오답 지지

`frame_42d229fb1459ee8c1086`, recording_e6ec6fd7a8cb5c465dea, 플라스틱 W/D/H=110/130/11cm.

- N0: **3.909cm / 1.855°**, 성공.
- PFraw:4.227cm/1.568°, SubPix:4.902cm/1.706°, PF+SubPix 양 anchor:4.191cm/1.725°, 모두 성공.
- 원 OCPR: **7.844cm / 1.713°**, 위치 실패. 세 seed의 출력이 동일하다.
- N0는 원 bank에 정확히 남아 있고 eligible이다. H1 rank37, logit−3.968; 최종 `G10/ORIGINAL_WD/selected` rank1, logit−2.142. 성공 후보를 선택하지 못한 사실은 확정한다.

같은 ORIGINAL_WD 문맥의 저장 signed-log 특징을 역변환하면 N0→선택 후보의 fixed 비용은40.137→28.239, A42.909→31.012, B65.061→20.821, AB67.833→23.593, edge199.585→169.636이다. 원 full3.666→1.650, partial0.786→0.00440도 감소한다. 값은 exported scalar의 근사 역변환이며 exact objective 재실행이 아니다. 서로 다른 mode 또는 family 값을 직접 빼지 않았다. validity는 이 항들에서1이다. **관측/기하 비용도 실패한 후보를 지지하므로 ‘H1이 좋은 관측만 무시했다’는 단순 가설은 이 사례에서 성립하지 않는다.** 비용 우세가 참조 오차 우세를 보장하지 않는 것이다.

그림에서 콘과 물체/바닥 경계가 보이고, D1에는 제한된 full/partial 관측이 남아 있다. 이것만으로 어느 선이 그림자/잘못된 owner인지 확정하지 않는다. native 대응은 초기 기하에 조건부이며 독립 owner 정답이 없다. PF·SubPix 완성 pose는 원 bank의 정확한 pose와 일치하지 않지만, 성공한 N0가 이미 있으므로 이 사례를 후보 부재라고 부르지 않는다.

## 여러 사례의 차이와 대조

선정 규칙은 그림 전에 저장했다. CAL focus는 실제1장뿐이므로 억지로6장을 채우지 않았다. CAL focus1 + 양쪽 실패2 + OCPR 우세2 =5장이다. 촬영 그룹 순환/고정 오류 정렬을 사용했다. 별도 PRIMARY 사후 표본은 recording별 큰 R오류1장씩, 남은 큰 T오류1장으로3장이다. 총8장은 5 recording, 여러 촬영 session에 속한다. 표본 비율을 전체 원인 비율로 확대하지 않는다.

| 사례 | 분리 | N0 T/R | 원 OCPR T/R | PF+SubPix PF-anchor T/R | 관찰 |
|---|---|---|---|---|---|
|A01|CAL 비교 우세|3.91/1.85|7.84/1.71|4.19/1.73|N0 eligible/rank37; 비용도 실패 후보 지지|
|A02|CAL 양쪽 실패|17.77/5.01|14.34/4.91|18.96/5.13|eligible 성공 후보 부재; OCPR은 일부 개선하지만 실패|
|A03|CAL 양쪽 실패·과거 E84|14.79/83.82|14.49/84.12|15.00/84.22|모든 비교 실패; 은행에 성공 후보 없음|
|A04|CAL OCPR 우세|5.51/2.35|2.25/1.76|6.25/2.67|다섯 비교 모두 실패지만 OCPR 성공|
|A05|CAL OCPR 우세|7.92/6.64|1.42/2.82|7.67/5.95|다섯 비교 모두 실패지만 OCPR 성공|
|P01|PRIMARY 사후|5.92/2.08|65.75/164.21|2.89/0.82|비교 성공, 원 eligible bank 성공 후보 없음|
|P02|PRIMARY 사후|164.79/81.67|156.06/83.34|1.59/2.06|은행에 성공 대안 있음; PF 계열 완성 pose는 외부 입력 경로|
|P03|PRIMARY 사후|3.85/2.75|40.76/1.79|1.98/1.62|성공 N0 eligible/rank5를 놓친 큰 위치 손상|

A03은 앞선 v41에서 사후 추가했던 `frame_1d19cea428d6f91e4646`다. 이번 v44 규칙에서는 CAL ‘양쪽 실패’ stratum의 recording_4e5f9461dc35af510c53 최대 회전 오류로 자연스럽게 선택됐다. 과거 후선정 이력을 지우지 않으며, 기존 v40의12장 선정·비율은 바꾸지 않았다. A03의 PFraw15.17/84.20°, SubPix15.82/83.89°도 실패다. 사람에게 불가능한 장면 또는 SAM 오류라고 단정하지 않는다.

P01은 `frame_25d5a619b597bc260252`다. PFraw2.14/2.85°, SubPix1.51/2.45°도 성공한다. 원 bank에 성공 후보 자체가 없으므로 H1만 바꿔 그 bank에서 성공시키는 것은 불가능하다. 정확한 PF/SubPix pose는 raw 생성 bank/HEAD/eligible 모두에 없다. 다만 외부 producer가 다른 점과 경로를 사용했다는 사실만으로 ‘같은 좋은 후보가 어느 필터에서 사라졌다’고 하지 않는다.

P02는 `frame_595b6e2eb1f2b8dd9b8c`다. **PFraw가 이미6.22cm/1.75°**로 방향 오류를 줄이고, PF+SubPix는1.59cm/2.06°로 남은 위치 오류를 줄인다. PF→PF+SubPix supported 점 이동의 중앙1.06px/최대6.01px이고 저장 selected_hypothesis는 둘 다 short-face-front다. 따라서 이 사례에서 SubPix가 처음90° 분기를 전환했다고 말할 수 없다. PFraw부터 다른 N3계열 점/최종 자세가 만들어졌다. 두 pose는 유사한 RGB silhouette로 투영될 수 있어 사진의 outline만으로 실제 방향 정확성을 증명하지 않는다.

P03은 `frame_8d6f6c6bd87e0b4081e0`이다. PFraw7.30cm/1.50°, SubPix5.36cm/1.91°로 각각 위치 경계 밖이지만, 결합은1.98cm/1.62°다. N0도 성공하므로 최종 손상은 선택 문제다. 서로 다른 입력의 PF 결합 이득과 원 H1 선택 손상을 구분했다.

## 입력·후보 계약과 남은 차단

N0는 원 N3+SubPix 경로다. PFraw는 팔레트용 PoseFix 신경망이 YOLO 점/RGB crop을 입력받아 만든 저장 점을 공통 F에 넣은 별도 경로다. SubPix-only는 YOLO 점 경로이며 N0와 같은 초기점이 아니다. 결합은 PF 점 뒤의 native SubPix를 쓰되 PF-anchor/YOLO-anchor cap을 분리한다. 두 anchor는 ‘출발 신경망이 서로 다르다’는 뜻이 아니다. 전체 upstream 모델·학습·정보 예산은 같지 않으며 이를 H1만의 선택기 비교라고 부르지 않는다.

모든8장의 실제 PF raw/native SubPix/두 anchor 최종점, YOLO점, N0 upstream점을 읽었다. `POINT_MOVEMENTS.csv`는 저장 좌표의 실제 dx/dy이며 pose 투영corner가 아니다. 일부 PRIMARY 입력에는 원 N3-before-SubPix 점 필드가 없어 그 이동은 미측정이다. baseline pose row에는 SubPix-only의 정확한 최종 2D점이 없어 재생성하지 않았다. 관측→대응→solver 중 어느 내부 하위 단계가 인과적으로 분기를 바꿨는지 완전히 분리하려면 이 원 입력/중간 기록이 더 필요하다. 없는 점/owner를 추정해 만들지 않았다.

비교 pose의 원 bank membership은 R/t 직접 일치(원소1e−8)와 C2동치 일치를 별도 기록했다. N0 외 비교 pose들은 이8장의 원 bank와 정확히 일치하지 않았다. 성공한 비교 방법이 외부인 경우 그 성공을 원 D1 oracle로 옮기지 않는다. `bank_eligible_good`는 기존 exact-bank coverage 집계를 읽었고 새 oracle/전체 후보 채점은 하지 않았다. 진단 최소오류 후보의 rank는 저장 ID를 추적한 것일 뿐 실행 방법이 아니다.

D1 그림은 저장 solver_input의 full/partial을 구분했고 partial로 대체된 full은 중복 표시하지 않았다. mode별 B profile 소비 native ID를 CSV에 별도 적었다. 마지막 관측 패널은 mode별 profile 대체 전 packet을 보여주며, 이를 모든 cost mode가 동일하게 쓰는 관측이라고 하지 않는다. 서로 다른 family의 context cost를 직접 비교하지 않는다.

목재80/59/14cm, 플라스틱110/130/11cm를 실제 입력 K/등록 geometry와 대조했다. 개별 실물 실측 인증은 미확인이다. 치수 누락, 물리 owner 오류, reference 오류 중 어느 하나를 이번 감사만으로 원인 확정하지 않는다.

## 제안하는 다음 실험 1개 — 아직 실행하지 않음

**TRAIN의 동일 프레임에서 좌표 원천×W/D 대응의 고정 2×2 재생**을 제안한다. 좌표는 저장 N0입력과 저장 PF+SubPix 입력, 대응은 기존 두 W/D 대응을 각각 고정한다. K/치수, 공통 F의 solver/start/예산은 같게 하고 H1은 쓰지 않는다. 합성 TRAIN pose는 진단 정답으로만 사용하며, PRIMARY의 정답으로 조합을 고르지 않는다. 새 방법 평가나 PF를 bank에 그냥 추가하는 실험이 아니라, 실제 방향/깊이 차이가 좌표에서 오는지 대응 선택에서 오는지 분리하는 진단이다. PF union 재학습의 과거 실패를 반복하는 것이 아니다.

먼저 동일 TRAIN 프레임의 두 원천 최종점과 native→camera-facing→physical 대응을 정확히 확보할 수 있는지 확인해야 한다. 자료가 없거나 대응을 복원할 수 없으면 필요한 캐시만 명시하고 중단한다. 허용된다면 TRAIN의 고정 실패/성공 대조에서 작은 pilot을 사전 등록하고, 좌표 또는 대응을 바꿔도 큰 오류가 유지되면 해당 가설을 기각한다. 이 제안은 새 PnP가 필요하므로 **이번 읽기 전용 감사에서는 실행하지 않았다.** 모델/threshold를 이8장에 맞추는 규칙은 제안하지 않는다.

## 전달·재현

`PAIRED_ROWS.csv`:1190행, 모든238×5비교의 저장오류/성공/차이. `CASE_TABLE.csv`:8개 진단, `CANDIDATE_MEMBERSHIP.csv`:40개 membership, `STORED_H1_RANKS.csv`:전체 저장 순위, `POINT_MOVEMENTS.csv`:저장점 이동, `EXPORTED_COST_FEATURES.csv`:동일 문맥 특징 읽기. 숫자/pose hash를 연결하는 figure manifest와 출처 hash를 보존했다.

`FIGURES_ZOOM_V2`:같은 raw crop·K·좌표·투영, uniform display 확대만 적용한8장 비교 PNG와8쪽 PDF. 원사진/관측/pose를 이동·생성·수정하지 않았다. full point와 투영corner는 별도 패널이다. 독립 블라인드 사람 주석이 아닌 보조 시각 점검이다. 모든 사례를 직접 열어 확인했다. v1 그림도 원위치 보존했다.

실행 명령(기존 Python, CPU1):
[Local execution commands omitted from public edition; preserved in private audit.]
출력은 exclusive-create다. 재현 시 새 output 폴더의 스크립트로 실행하고 기존 결과를 덮어쓰지 않는다. 원 캐시와 이전 identity/exact-bank coverage가 필요하며 원자료/체크포인트는 ZIP에 포함하지 않는다. 이 감사의 코드 작성 중 syntax/누락 N3 필드 처리를 수정했으며 저장 pose·수치·선정은 바꾸지 않았다. `INSPECTION_EXECUTION.json`과 `RENDER_EXECUTION_V2.json`에 실제 처리 시간을 기록했다. 새 연구 실행 및 GitHub 게시0.
