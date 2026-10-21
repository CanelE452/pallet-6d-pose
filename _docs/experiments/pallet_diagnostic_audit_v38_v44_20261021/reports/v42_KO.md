> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.

TRAIN render-derived photos are excluded because external background/texture publication rights were not verified. Their scalar diagnostics remain included.

# TRAIN8 renderer mask 의미·partial 출처 감사 v42

**인코딩·면적·좌표 정합은 확인했지만, exact 생성기/RLE와 물리 face·edge owner는 완전히 검증하지 못했다. 새 모델 변경은 보류한다.** 기존 TRAIN8 선정, v5 incumbent, CAL12 집계 및 E84 사후 부록은 바꾸지 않았다.

## 마스크를 어디까지 믿을 수 있는가

같은 TRAIN8의 visible/amodal PNG16개를 실제로 읽었다. 모두 uint8 단일채널이고 값은0/1/2 및254/255뿐이었다. 배경에1·2가 있어서 단순 nonzero 판독은 틀린다. 기존 소비 코드 scripts/stage0/line/edge_mandatory_fast_search.py:313–317의 >127 규칙으로 해석하면 **저장 JSON 면적과16/16 일치**, visible⊆amodal 조건은8/8 성립한다. 이127은 이번 성능에 맞춘 문턱이 아니라 기존 파일 판독 규칙이다.

예를 들어 T05의 visible은7,595px인데 nonzero로 읽으면244,643px다. 따라서 앞선 PNG의 gradient/미세 비영 배경 문제를 무시해선 안 된다. 이번16개에는 넓은0–255 연속 gradient 대신 배경0/1/2와 전경254/255가 분리돼 있다. 왜 미세 비영값이 생겼는지는 생성 코드 없이 확정하지 않았다.

원 renderer label JSON, records.jsonl, PNG shape와 원 RGB·GT·관측의 좌표계를 맞췄다. record는 idx가 아닌 usable_id로 연결했다. 기존 da_final_manifest.py도 이 연결을 명시한다. mask shape가 raw RGB·camera width/height·native GT와 모두 일치하고, 보존 label/RGB hash도 대조했다. padding·추가 resize를 측정 좌표에 넣지 않았다.

records에는 M0/M4 순서, mask_strict_decode_ok, inclusion 검사가 보존돼 있고, label은 public mask profile이다. 이것과 실제 RGB 대조에서 amodal은 target-only 전체 silhouette, visible은 가려진 부분이 빠진 target silhouette라는 **저장 metadata의 해석과 양립**한다. amodal-only 영역이 cargo/배럴/다른 물체에 가려진 위치와 겹치는 것도 직접 보았다.

다만 선택된 JSON8개에는 RLE가 없었다. advanced mask_profile/M0/M4 생성 코드는 검사한 로컬 scripts/Deep_Object_Pose/_archive에서 찾지 못했고, source records의 생성 경로는 다른 Windows FoundationPose 작업본을 가리킨다. 다른 컴퓨터로 우회하지 않았다. 로컬의 legacy render_blender_data.py는 해당 advanced contract 구현을 포함하지 않아 그 코드로 의미를 증명하지 않았다. stored mask_m0_content_sha256의 직렬화 함수도 확인되지 않아 몇 가지 일반 byte hash와의 불일치를 파일 손상으로 해석하지 않았다. **exact producer revision·RLE/pixel 동치의 확인은 미완**이다.

이번 마스크는 선언된 target silhouette와 관측의 형상 양립성 진단에 조건부 사용했다. 전체 물체 mask는 어느 face/edge인지, 내부 재질 경계인지, 바닥 위 그림자의 광학 원인인지까지 알려 주지 않는다. model-conditioned native association을 독립 물리 owner 정답으로 바꾸지 않았다. 이 PNG 판독은 감사용 renderer 참조이며, 기존 OCPR의 SAM NPZ 관측이나 역사적 성능을 수정한 것이 아니다.

## 실제 partial 11개

RGB·GT0 native 참조 투영·실제 solver partial·두 마스크를 동일 crop/좌표에 놓고8장 모두 직접 열었다. G표시는 원 GT0 native index, O표시는 관측이 선언한 native index다. 두 번호가 곧 실제 물리 owner라는 뜻은 아니다. T03/T06/T07에는 사용 partial이 없으므로 새 선을 만들어 넣지 않았다.

| 표본 | partial 존재 frame | 선 개수 | 외곽 지지 | 내부 경계 | 외부 영역 확정 | 판정 불가 |
|---|---:|---:|---:|---:|---:|---:|
| 큰 회전 오류4장 |3/4|7|5|1|0|1|
| v5 성공 대조4장 |2/4|4|2|1|0|1|

이 표는 보조 시각 분류이며 독립 사람 주석·원인 비율·배포용 레이블이 아니다. 각 선의 값과 근거는 PARTIAL_CLASSIFICATIONS.csv에 보존했다.

- **T01 I0 / T05 I0:** visible/amodal 내부에서 전면–상면의 경계와 양립한다. 외곽까지의 중앙 거리는 각각30.87px/12px다. I가 외곽에서 멀다는 이유로 제거하면 안 된다.
- **T01 B2:** amodal 외곽에서 중앙32.20px 떨어져 있지만 visible 경계에서는0px다. 선 위 위치의42.6%는 amodal-only이고 RGB의 배럴 가림 경계와 겹친다. 판정 불가/가림 경계 단서로 기록했다. 팔레트의 정확한 물리 edge 정답으로 명명하지 않았다.
- **T05 B2:** 성공 대조에서도 같은 양상이다. amodal 외곽13.04px, visible 경계1px, amodal-only61.4%이며 앞쪽 회색 물체의 가림 경계와 겹친다. 역시 판정 불가다.
- **T02 B0/B1/B2/B3, T04 B1, T08 B1/B7:** amodal/visible의 외곽에 가까운 지지가 있다. 그렇다고 선언된 native edge가 맞다고 증명한 것은 아니다. 특히 성공 T08은 표본 픽셀의80% 이상이 mask 바깥에 놓여도 외곽까지 중앙1px다. 래스터 경계 근처라는 사실을 background-owner 오류로 바꾸면 안 된다.

실제 저장 observed_points가 있으면 그 좌표를 썼다. 원 inlier_mask도 확인했으며 T05 I0의12 query 중11개가 inlier였다. 그11개만의 amodal 소속도100%여서 분류는 같고, 교차 검산을 별도 보존했다. 없는 B recovery 선은 저장된 fitted inlier 구간에서101개 고정 위치를 평가했고, 이를 새 관측점으로 부르지 않았다. 1px 근접값은 래스터 설명용으로 기록한 값이며 학습·규칙 문턱을 튜닝하지 않았다. 외부 contour 추출에서 hole을 채운 파생 배열은 ‘외부/구멍/내부’ 설명에만 썼고 원 mask·관측·solver를 바꾸지 않았다.

## 다음 개입 판단

가림 경계 단서를 발견했지만 실패와 성공에서 각각 하나씩 나타났다. 외곽 지지가 있어도 큰 방향 오류가 남고, 정상 외곽도 nearest-pixel mask 밖에 놓일 수 있다. 따라서 mask 바깥·visible/amodal 차이·내부 경계라는 이유로 관측을 지우거나 가중치를 줄이는 정책을 정당화하지 못한다. 이번8장에 맞춘 native 번호 예외나 새 head 학습은 하지 않았다.

현재 자료로 말할 수 있는 범위는 ‘선언된 target 실루엣의 어디에 선이 놓였는가’, ‘가림으로 생긴 visible 경계와 양립하는가’까지다. 정확한 physical edge/face owner를 확정하려면 다음 중 최소 자료가 필요하다.

1. 해당 public profile을 생성한 exact 코드/revision과 M0/M4 저장·color/PNG 판독 계약. 원본 RLE가 따로 있다면 같은8프레임의 RLE/PNG 정합.
2. T01 B2와 성공 대조 T05 B2의 해당 query 구간에 연결된 renderer object/face ID 또는 독립 물리 경계 owner 판독. 숨겨진 corner를 추정해 새로 찍을 필요는 없다. 모르면 UNKNOWN이어야 한다.

전체 객체 마스크만으로 face ID를 만들어 냈다고 보고하지 않는다. per-frame front/opening visibility scalar도 per-query edge owner 정답과 다르다. 다음 모델 변경은 보류한다. 가능한 후속 분석도 먼저 계약/출처 검증에 한정해야 한다.

## 비용·파일

CPU1개10.846초. 새 render/forward/SAM/PnP/학습/PRIMARY 평가/GitHub 게시0. 원 자료와 기존 선택·집계 hash는 보존했다.

- RESULTS/ENCODING_AND_METADATA.json:16개 값 분포·면적·subset·RLE 유무
- RESULTS/PARTIAL_MASK_COMPATIBILITY.csv / PARTIAL_CLASSIFICATIONS.csv:11개 선의 측정과 조건부 해석
- RESULTS/CONTACT_T01_T04.png / CONTACT_T05_T08.png:오류·성공 비교 그림
- RESULTS/PANELS/T01.png…T08.png:개별 확대
- RESULTS/FIGURE_MANIFEST.json / EXECUTION.json:좌표·입력 hash·비용
- audit_masks.py / classify_and_report.py:기존 캐시 판독·도식·요약. 원 output은 덮어쓰지 않으므로 새 재현에는 출력 root를 분리한다.

별도 치수 질문의 답과 실제입력 증거는 dimension_input_contract_audit_20261021_v43에 저장했다. CAL 목재80×59×14cm, 플라스틱110×130×11cm를 실제 N3 입력과 PnP/후단/참조에서 확인했고 개별 실물 재실측 여부는 미확인이다. TRAIN8은 각 asset/scale의 canonical 치수다.

최종 전달 그림은 DISPLAY-only nearest-neighbor 확대 V2다. 동일 native crop과 원 관측을 유지하며 작은 물체의 확대 표시만 바꿨다. 기존 V1 그림과 숫자 CSV는 보존했다.
