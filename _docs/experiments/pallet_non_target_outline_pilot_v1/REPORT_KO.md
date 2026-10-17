# 비대상 입력을 이용한 외곽선 처리 기능 시험

**ASSISTED6에서 새 MobileSAM 마스크 3개와 세 경로의 최종 선택 18개를 실제 실행했으며, AUTO238은 실행하지 않았다. 세 경로 모두 성공은 2/6, 기존 e 대비 회복 0·손상 0이었다.**

부모 구간별 처리(`SUPPORT_AWARE_RERANK`)는 후보를 2장, W/D family를 1장에서 바꿨다. 정보량을 맞춘 전체 약화 대조군(`MASS_MATCHED_CONTROL`)은 각각 3장·1장에서 바꿨다. 주경로의 위치 평균은 39.47→29.28cm로 줄었지만 대조군은 25.66cm였다. 한 실패 사례는 두 신규 경로 모두 위치·회전이 악화했다. 따라서 콘 지지 구간을 구분하는 방법이 단순 약화보다 낫다는 근거는 이번 시험에서 얻지 못했다.

이 결과는 사후 선정한 설명 사례 6장의 **입력 보조 + 고정 후보 재선택**이다. 자동 시스템 성능, 독립 검증, 분할 정확도 또는 새 손실로 재최적화한 결과가 아니다. 원 e의 후보 생성기·후보은행·상태·target mask·B/I·제약·가중치·참조·운영 기본값과 논문을 보존했다.

## 실제 결과와 전체 분모

성공은 동일한 최종 자세의 중심 위치≤5cm **AND** physical/C2 회전≤5°다. 직사각형에 90° 동등성을 추가하지 않았다. 아래 `T/R`은 cm/°이며 수치는 표시용으로 반올림했다. 전 정밀도 18행은 [METRICS.csv](METRICS.csv), 짝지은 12행은 [TRANSITIONS.csv](TRANSITIONS.csv)에 있다.

| 원 RGB ID | BASELINE_E T/R | SUPPORT_AWARE T/R | MASS_MATCHED T/R | 입력·주경로 변화 |
|---|---:|---:|---:|---|
| plastic_night_01:038630 | 99.985 / 88.238 | 99.985 / 88.238 | 80.714 / 88.370 | 콘 마스크 사용, 주경로 선택 유지 |
| wood_day_01:005286 | 76.857 / 88.270 | 6.127 / 10.178 | 3.673 / 10.603 | 콘 마스크 사용, 두 신규 경로 모두 family 변경 |
| wood_night_01:030698 | 29.389 / 19.116 | 29.389 / 19.116 | 29.389 / 19.116 | 인정할 콘 입력 없음, 원 e와 동일 |
| plastic_day_01:020955 | 20.662 / 12.681 | 30.310 / 15.639 | 30.310 / 15.639 | 콘 마스크 사용, 두 신규 경로 모두 악화 |
| plastic_day_01:014704 | 4.952 / 0.609 | 4.952 / 0.609 | 4.952 / 0.609 | 인정할 콘 입력 없음, 성공 유지 |
| wood_night_01:030898 | 4.946 / 3.772 | 4.946 / 3.772 | 4.946 / 3.772 | 인정할 콘 입력 없음, 성공 유지 |

| 경로 | 성공/N | 회복 / 손상 | 후보 / family 변경 | T 평균±표본SD cm | T 최대 cm | R 평균±표본SD ° | R 최대 ° | T>50 / >100cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE_E | 2/6 | 0 / 0 | 0 / 0 | 39.465±39.746 | 99.985 | 35.448±41.423 | 88.270 | 2 / 0 |
| SUPPORT_AWARE_RERANK | 2/6 | 0 / 0 | 2 / 1 | 29.285±36.661 | 99.985 | 22.925±32.743 | 88.238 | 1 / 0 |
| MASS_MATCHED_CONTROL | 2/6 | 0 / 0 | 3 / 1 | 25.664±29.691 | 80.714 | 23.018±32.764 | 88.370 | 1 / 0 |

표본SD의 분모는 N−1이다. 이는 선정된 6장을 기술하는 값이며 모집단 추정이나 독립 표본 유의성 검정을 하지 않았다. 회복−손상=성공수 변화=0이며 두 신규 경로의 전이는 각각 성공 유지2·실패 지속4다. **손상0은 이진 성공 판정상의 값**이다. plastic_day_01:020955는 실패 지속 안에서 +9.647cm/+2.958° 악화했다. 성공 반례 2장은 모두 새 입력이 없어서 항등성으로 보존됐다. 이 두 장은 실제 censoring을 적용한 뒤에도 성공을 보존한다는 증거가 아니다.

같은6장의 저장 결과에서 N0는3/6, PoseFix RAW는2/6이다. N0는 **N3+SubPix→기존 공통 F**, PoseFix RAW는 synthetic-only seed1·last6000의 첫 RAW pass를 기존 W/D selector+F로 처리한 결과다. 이번6장은 PoseFix143 저장 보충분에 모두 포함된다. N0가 성공한 사례1·3은 새 지지 처리로 회복되지 않았고, 원 e가 성공한 사례6은 N0에서 실패한다. 동일 참조·SHA·정의·전수 수치는 [CONTROL_COMPARISON.md](CONTROL_COMPARISON.md)와 [CONTROL_COMPARISON.csv](CONTROL_COMPARISON.csv)에 있다. 이 사후6장으로 전체 방법의 성능 순위를 정하지 않으며, 다른 분모의 PF+SubPix95 또는 RAW192와 합치지 않는다.

## 입력은 어떻게 만들었는가

후보·참조·이전 OTHER 범주·비용차 그림을 받지 않는 별도 모델 문맥에서 **원 RGB만** 보고 `visible traffic cone`의 보이는 영역을 감싼 박스와 내부 양성점 하나를 작성했다. 6행 중 3행만 인정 가능한 콘이 있었고, 나머지 3행은 `NO_ADMISSIBLE_NON_TARGET_EVIDENCE`로 남겼다. 영상 선정 자체는 이전 결과를 이용했으므로 독립 블라인드 시험이라고 하지 않는다.

기존 MobileSAM `vit_t`와 출처가 확인된 40,728,226-byte checkpoint를 재사용했다. 박스+양성점, `multimask_output=True`를 각 입력에 한 번 호출했다. 한 RGB의 encoder는 한 번, decoder는 한 번이었다. 출력 9개 대안과 quality를 모두 보존하고 **최대 quality, 정확 동률은 원순서 첫 대안**을 선택했다. 선택 인덱스는 사례1/2/4에서 2/1/1이다. quality는 객체 의미·분할 정확도·깊이 순서를 인증하는 확률이 아니다.

새 선택 비용과 새 참조 오차를 열기 전에 root 모델이 선택 마스크와 RGB를 실제로 한 번 검토했다. 명백한 큰 팔레트 영역 혼합을 발견하지 못한 3개를 `ASSISTED_REVIEWED`로 동결했다. 미세 경계 정확도는 확인되지 않았다. 인간 검수·픽셀 GT·oracle mask가 아니다. 프롬프트 보완, 다른 SAM 대안 재선택, threshold 변경, 좋은 결과만 재실행한 사례는 없다. 야간의 어두운 경계를 그림자로 추측해 마스크에 넣지 않았다.

[PROMPT_MANIFEST.jsonl](PROMPT_MANIFEST.jsonl), [NON_TARGET_INPUT_MANIFEST.jsonl](NON_TARGET_INPUT_MANIFEST.jsonl), [INPUTS_FROZEN.jsonl](INPUTS_FROZEN.jsonl)에 제작자·좌표·모델·원 canvas·파일/픽셀 SHA·대안·선택·사용 상태를 보존했다. 공개 경로를 상대경로/별칭으로 바꾼 파생본과 원 실행 manifest의 관계는 [PUBLIC_DERIVATIONS.json](PUBLIC_DERIVATIONS.json)에 연결한다. 상세 공급기·실패 기록은 [INPUT_SUPPLIER_AUDIT.md](INPUT_SUPPLIER_AUDIT.md)에 있다.

## 새 방법이 실제로 바꾼 것

원 target mask에서 콘을 빼거나 다시 convex hull을 닫지 않았다. 저장된 polygon, 원 hull vertex 쌍, canvas clipping 이후 finite segment와 sample을 그대로 썼다. **clipping 이전 부모의 두 끝점 중 하나라도 새 비대상 binary mask 양성 픽셀에 속하면 그 부모를 결측 처리**했다. 원좌표의 고정 최근접 `floor(x+0.5)`를 사용했고 margin 절단점을 원 끝점으로 대신 쓰지 않았다. 이 규칙은 부모 선분 전체가 콘이라고 선언하지 않는다. 팔레트 지지가 섞인 긴 연결 구간도 함께 잃을 수 있다.

observed→model은 같은 원 sample index의 weighted contribution만 0으로 뒀다. model→observed는 후보마다 저장된 원 model sample과 호 길이를 사용하고, **삭제 전 모든 원 finite segment**에서 원 tie 순서로 찾은 nearest parent의 유효성에 따라 contribution을 0으로 뒀다. 살아남은 선분으로 nearest를 다시 찾거나 숨은 경계를 생성하지 않았다. 원 normalizer, robust loss, noise scale, outline factor와 양방향 1/2를 유지했다. 저장 기여에 이미 포함된 가중치를 다시 곱하지 않았다.

대조군은 같은 후보·같은 입력에서 각 방향의 retained 원가중 mass 비율로 그 방향의 원 outline 비용 전체를 일괄 축소한다. model 방향의 비율은 후보별이다. 별도의 튜닝한 상수가중치가 아니다. 아래 질량과 비용은 **각 사례의 SUPPORT_AWARE 최종 선택 후보 자체**에서 계산한 값으로, 서로 다른 후보 간 잔차를 같은 index로 빼지 않았다.

| 사례 | 유일 원부모 결측/전체 | observed 유지 mass 비율 | model 유지 mass 비율 | 같은 후보의 원 outline → support outline |
|---|---:|---:|---:|---:|
| plastic_night_01:038630 | 5/12 | 0.401985 | 0.383886 | 10.448044 → 3.104163 |
| wood_day_01:005286 | 7/12 | 0.088968 | 0.087214 | 8.030561 → 0.006630 |
| wood_night_01:030698 | 0/35 | 1 | 1 | 7.967611 → 7.967611 |
| plastic_day_01:020955 | 9/18 | 0.129477 | 0.138770 | 7.786037 → 0.216498 |
| plastic_day_01:014704 | 0/22 | 1 | 1 | 0.301080 → 0.301080 |
| wood_night_01:030898 | 0/28 | 1 | 1 | 7.182104 → 7.182104 |

원부모는 총127개이며 각 G10/G13×2 family 문맥을 따로 기록한 [PARENT_SUPPORT.jsonl](PARENT_SUPPORT.jsonl)은 508행이다. 같은 원부모의 문맥별 기록을 독립적인 물리 관측처럼 합산하지 않았다. 이번 6장에서는 네 문맥의 원 외곽 관측이 같은 바이트/좌표임을 확인했다. 원 관측 sample은 장면별 공유하지만 model sample/호 길이/nearest parent는 후보별이다. 전체351 후보의 두 방향 비용·유지 mass와 선택 점수는 [CANDIDATE_COSTS.jsonl.gz](CANDIDATE_COSTS.jsonl.gz), 선택 출력은 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)에 있다.

## 어디에서 회복하지 못했는가

원 retained 후보는 사례 순서대로22/94/42/130/41/22개, 총351개다. 성공 비교 후보 두 개만 점수화하지 않았다. 각 G10 family 전체를 원 comparator로 정렬하고 accepted family 중 원 규칙으로 family를 정한 뒤, 같은 family의 G13 전체를 정렬했다. 정확 동률의 초기화 이름·stable 순서, 자체 fallback, 후보의 자기 Q를 유지했다. **G10과 G13의 목적함수 값은 서로 비교하지 않았다.**

다음 설명의 성공 후보 조회는 예측을 봉인한 **후** 기존 참조를 붙인 진단이다. 참조 성공 표시를 입력·선택·tie-break에 넣거나 이를 별도 새 selector 결과로 보고하지 않았다. 근거는 [G10_FAMILY_DIAGNOSTICS.csv](G10_FAMILY_DIAGNOSTICS.csv), [G13_SELECTION_DIAGNOSTICS.csv](G13_SELECTION_DIAGNOSTICS.csv), [G13_SELECTED_COMPONENTS.csv](G13_SELECTED_COMPONENTS.csv)다.

1. **plastic_night_01:038630 — G10 family 선택이 유지됐다.** 주경로의 G10 원/교환 family winner 점수는 각각4.714219/4.202928로 여전히 SWAPPED_WD를 골랐다. 그 family의 G13 retained에는 참조상 공동성공 후보가0개다. 주경로의 최종 후보도 `SWAPPED_WD/0` 그대로다. 다른 family에 존재하는 성공 후보를 GT로 끌어오는 정책을 추가하지 않았다. 대조군은 `SWAPPED_WD/1`을 골라 위치는 줄였지만 약88° 회전 실패가 남았다.
2. **wood_day_01:005286 — family는 바뀌었지만 family 내부 성공 선택은 못 했다.** 주경로와 대조군 모두 ORIGINAL_WD로 바뀌었다. 주경로는 G13 후보31, 대조군은 후보1을 실제 선택했다. 주경로 family에는 성공 후보가5개 남아 있다. 그중 새 비용이 가장 낮은 성공 후보33의 점수0.007038028은 실제 후보31의0.006629500보다 높아 선택되지 않았다. 두 후보 full_BI는 약10⁻¹²이며 partial/prior는0이다. 실제 후보31의 원 outline8.030561이0.006629500으로 줄어, 약8.9%의 observed mass만 남은 상태에서 위치6.127cm·회전10.178° 후보가 이겼다. 이는 비대상 입력의 기능 연결은 작동했지만 현재 남은 관측 비용이 성공 여부를 구별하지 못한 사례다.
3. **wood_night_01:030698 — 새 범주의 입력이 없다.** 어두운 하부를 비대상 양성으로 추정하지 않았으므로 비용·순위·자세가 원 e와 같다. 같은 family의 성공 후보는2개지만 원 비용에서 실제 후보가 앞선다. 이번 cone 공급기로 이 미확정 경계 문제를 해결했다고 하지 않는다.
4. **plastic_day_01:020955 — 실제 선택이 나빠졌다.** 두 신규 경로 모두 G13 후보1을 골랐다. 주경로 실제 점수0.337350은 partial0.120853+outline0.216498이다. 같은 family의 성공 후보9는0.952511=partial0.127339+outline0.825172여서 여전히 졌다. 두 후보의 partial 차이보다 남은 outline 차이가 훨씬 크다. 비대상 부모를 지운 뒤에도 이 고정 후보은행의 점수는 더 나쁜 자세를 선호했다.
5. **plastic_day_01:014704, wood_night_01:030898 — 성공 반례는 입력 없음 항등성이다.** 두 장 모두 비용·전체 순위·최종 자세가 원 e와 정확히 같다. 이 결과를 실제 결측 처리 후의 안전성·비손상 개선으로 확대하지 않는다.

콘 입력 3개가 모두 부모에 연결됐고 비용도 실제로 변했다. 기술적으로 입력만 추가하고 selector를 건너뛴 시험이 아니다. 반면 **공동성공 회복은0**이며, 사례2의 개선도 대조군에 나타났고 사례4의 악화도 같다. 일부 남은 부모에 팔레트·다른 물체 지지가 섞이는 문제, 원 cuboid 외곽 가정, 적은 유효 관측의 식별력은 해결되지 않았다. 현재 결과만으로 세 요인의 기여를 따로 확정하지 않는다.

## AUTO 공백과 실행 비용

`AUTO_INPUT_SUPPLIER_NOT_AVAILABLE`이다. 보유 COCO80 YOLO의 실제 class metadata와 공식 목록에 traffic cone이 없고, 확인한 로컬 범위에서 cone 지원 의미 검출기의 API·가중치·라이선스 묶음을 확보하지 못했다. 기존 공급기 제안서는 실제 구현·연결 기록이 아니었다. SAM 자동 제안이나 target mask의 보완집합을 cone 검출기로 취급하지 않았다. 따라서238장 생성·평가를 실행하지 않았고 PRIMARY192/CAL46 숫자를 새로 만들거나 ASSISTED를 수동192장으로 확장하지 않았다.

실제 MobileSAM 추가 호출은 encoder3·prompt encoder3·mask decoder3이다. supplier 실제 실행 wall2.551405초는 로딩·세 프레임 추론·출력·마지막 실패 검사를 포함한다. 프레임별 encoder/decoder 시간·model load 시간·GPU 최대 메모리는 저장되지 않아 NA다. 실제 최종 rerank는351후보 기준5.808783초였다. 캐시 rerank와 부분 공급기 시간을 더해 전체 RGB→pose latency 또는 Jetson FPS라고 하지 않는다. 총 실행/실패/검산 비용은 [COST.json](COST.json)에 있다.

실패도 보존했다. supplier_v1은 잘못 넘긴 manifest SHA에서 NN0회로 중단했고 supplier_v2는 optional runtime 임시 경로 접근을 막아 NN 전에 중단했다. supplier_v3는 승인된3+3 추론과 출력 후, 이미 차단한 optional `socket.bind`1회·`subprocess.Popen`4회 때문에 마지막 장부 검사에서 실패했다. 성공한 외부 연결·child process나 금지 자료 읽기는 없었다. 기존 NPZ/PNG에서 manifest를 복구했으며 추론을 반복하지 않았다. 프레임별 시간을 채우기 위한 재실행도 없었다.

첫 actual_rerank_v1을 보존하고, 이후 후보 R/t/중심의 일치 검사를 명시한 최종 소스로 actual_rerank_v2를 실행했다. 현재24개 family published pose에는 `native_t`가 없어 첫 출력의 실제 오류는 없었다. 두 실행의 예측·비용·부모 기록은 **byte exact**로 같았다. 이는 소스 계약 검산이며 프롬프트·가중치·임계값 탐색이 아니다.

## 검산과 해석 한계

최종 [PREDICTION_SEAL.json](PREDICTION_SEAL.json)의 원 실행 SHA256은 `004215e30bd9722e76649de1f62be9f238a83f05a26430b595a8d6c4436c789b`다. 봉인된18개 출력 후 별도 evaluator만 기존 참조를 읽었다. selector는 whitelist neutral bank와 frozen mask만 읽었고, 실제 읽기 경계·model/solver guard 위반은0이었다. supplier의 차단된 optional runtime 시도는 이 값과 따로 보존한다.

독립 검산은 pilot kernel/replay/원 selector를 import하지 않고 전351후보·508부모·18출력·12전이의 산술과 계층 선택을 다시 확인했다. 원 끝점·표본·normalizer·nearest tie·mass 비율·원비용−censored 합을 확인했으며 최대 산술/자세 차이는8.674×10⁻¹⁸, 독립 평가 오차 차이는1.421×10⁻¹⁴였다. 입력 없음의 scalar·전체 순위·pose는 정확히 같다. 실제 API 단위시험7개에는 canvas 절단점 혼동, nearest 재검색, GT 접근과 파일/프로세스 경계 sentinel이 포함된다. 인위적 fixture는 영상 성능으로 집계하지 않았다. [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json)과 [VALIDATION.json](VALIDATION.json)에 범위를 명시했다.

원 RGB6·target mask6·vendor 코드·기존 가중치와 보호 대상 사용자 파일의 SHA, 기존 HEAD/상태·연구 branch를 유지했다. 새 학습·새 pose fit/PnP·새 후보·CAL 재fit·새 데이터·새 정답·외부 모델 다운로드·유료 API·RGB 외부 전송은0이다. 이미 허가된 원자료의 작은 파생 그림만 공개한다.

야간의 미확정 경계는 남아 있다. 전체 원은행에 공동성공 후보가 없던46장은 이번6장 시험 대상이 아니며, 이 방식은 고정은행 밖 성공 후보를 만들 수 없다. 기존 카메라 내부값은 독립 검증되지 않았고 signed axis/물리 참조의 불확실성과 `geometry_rank_ambiguous`를 수정하지 않았다. 기존 참조에 대한 수치 개선을 실제 물리 정확성의 인증으로 바꾸지 않는다.

**다음 판단은 현재 부모 전체 결측 처리의 운영 채택을 보류하는 것 하나다.** 자세한 근거는 [NEXT_DECISION_KO.md](NEXT_DECISION_KO.md), 반복 실행 안내는 [REPRODUCE.md](REPRODUCE.md), 입력과 실제 선택 비교는 [gallery/index.html](gallery/index.html)에 있다. 이번 결과만으로 자동 모델 다운로드를 우선 작업으로 정당화하지 않는다.
