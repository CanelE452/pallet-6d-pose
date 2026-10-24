# V218: 공개 실사의 의도한 부위와 query를 연결한 약한 감독 자료

**이번 진전은 실제 학습 자료 구성입니다. 새 모델 성능 개선을 입증한 결과는 아닙니다.** 공개 SOURCE 12장·15개 GT 후보에서 원 query 96개/U11을 보존하고, 별도의 GT 유래 학습 query crop 360개와 원 crop 기반 가림 대조쌍 9개를 실제 만들었습니다. 두 관찰자가 15개 contact를 각각 실제 열어 120개 부위 질문을 판독했습니다.

첨부 V193 검토 문서는 당시 진단·조언이며 새 실행 지시가 아닙니다. 이후 사용자가 공개 실사 학습을 허용한 계약을 적용합니다. 직접 촬영 평가223장 및 기존 보호 평가 계보는 학습·정규화·문턱·checkpoint·방법 선택에 사용하지 않습니다. 자기 가림은 사용자 계획대로 초기 기하 필터와 구분하며 외부 가림 E에 합치지 않습니다. PnP는 변경하지 않았습니다.

## 무엇이 달라졌나

1. **특정 부위를 가리는 대조 자료를 실제 생성했습니다.** 원 crop에서 두 관찰자의 ROI를 합친 약한 부위 support를 COVER로 가리고, 동일 checker RGB·alpha mass를 옮긴 SHAM에서는 해당 support를 그대로 남겼습니다. 9 triplet=27변형이며 FIT7/DEV2 triplet입니다. 각각 21/6변형이고 서로 독립된27장면이 아닙니다. 실제 픽셀 검산에서 9개 모두 같은 pattern/mass, valid 영역, COVER support 변경, SHAM target 변경0을 확인했습니다. ROI 밖을 background 정답으로 취급하지 않습니다.
2. **의도한 공개 GT 후보 부근의 RGB를 별도 학습 입력으로 만들었습니다.** 15후보×native8×3고정 query 상태=360개입니다. nominal S0와 (+0.25L,0)/(0,+0.25L) 두 shift를 사용했습니다. 입력 query8 자체가 공개 GT에서 유래합니다. 추가 GT pose/ref/ROI/mask를 배포 feature로 넣지 않았다는 의미와, 학습 query가 GT 유래라는 사실을 함께 기록합니다. 원 검출 query를 보정한 결과나 초기 검출기 성능 비교로 해석하지 않습니다.
3. **보임/외부 가림을 실제 RGB에서 판독했습니다.** GT 위치·mask/depth의 부재만으로 E를 만들지 않았습니다. 같은 후보의 기대 DECK/BOTTOM_RUNNER 부위와 외부 foreground 근거가 있어야 E weak입니다. 불일치·빈틈·다른 부위는 UNKNOWN 또는 부위·기하 미해결로 남겼습니다.

## 새 자료의 분모와 상태

| 단위 | 전체 | SOURCE_FIT | SOURCE_DEV |
|---|---:|---:|---:|
| 보존한 원 query | 96 | 64 | 32 |
| 15후보의 nominal 부위 질문 | 120 | 80 | 40 |
| 별도 GT 유래 RGB96 입력 | 360 | 240 | 120 |
| support 확인된 weak D/E 변형 | 57 | 45 | 12 |
| 위57 중 D / E | 48 / 9 | 39 / 6 | 9 / 3 |

원 U는11개입니다. 후보 반복에서의 원 U17개, 360행 반복 provenance U51개, 새 query 가용209개, nominal target in-frame210개는 서로 다른 분모입니다. 전체 crop의 strict-valid181개를 D/E 분모로 강제하지 않고 각 target/evidence support를 따로 검사했습니다. SOURCE_DEV의 모든 파생 입력은 fit·normalizer에서 제외합니다.

### 두 관찰자의 120행 판독

| 상태 | 관찰자 A | 관찰자 C | 동일 상태 합의 |
|---|---:|---:|---:|
| 기대 부위 보임 D weak | 20 | 21 | 19 |
| 기대 부위 외부 가림 E weak | 13 | 8 | 8 |
| 자기 구조에 가림 SELF | 6 | 0 | 0 |
| 부위·기하 미해결 | 26 | 39 | 24 |
| UNKNOWN | 5 | 2 | 0 |
| 화면 밖 | 50 | 50 | 50 |

총 일치는101/120이지만 여기에는 화면 밖50이 포함됩니다. 화면 안에서의 상태 일치는51/70(72.86%)이며 분류기 정확도가 아닙니다. 불일치19개를 UNKNOWN으로 남겨 최종 nominal은 **D19/E8/부위·기하 미해결24/UNKNOWN19/화면 밖50**입니다. SELF 합의0은 실제 자기 가림이 없다는 뜻이 아닙니다. 관찰자 간 SELF와 부위·기하 판단도 불일치했습니다. 원 판독은 수정하지 않았습니다.

Nominal D/E27개는 FIT D15/E5, DEV D4/E3입니다. shift 후 두 evidence ROI가 모두 crop 안에 남고 strict-valid이며 target/query 가용 조건을 통과한 경우만 weak 후보를 유지했습니다. 그 결과 S0 D19/E8, S1 D12/E0, S2 D17/E1로 총57변형입니다. 같은27개 부위의 상관된 변형이지57개 독립 자연 가림 정답이 아닙니다. 잘리거나 근거가 사라지는 D/E는 UNKNOWN으로 바뀌고 입력은 삭제하지 않았습니다.

이것은 두 assistant의 약한 부위 판독입니다. 사람 어노테이션·정확한 물리 native corner gold·독립 blind 검증으로 승격하지 않았습니다. 실제 학습은 아직0이며 `weak_loss_fit_candidate`는 후속 실험의 후보 mask입니다. 완벽한 mesh/물리점 인증을 모든 유효한 약한 부위 감독의 전제조건으로 요구하지 않습니다.

## 남은 문제가 실제로 무엇인가

- **부위 목표 차이가 확인됐습니다.** 과거 weak9 중 f0/native6과 f5/native7은 아래 BOTTOM_RUNNER가 아니라 위 DECK의 아래 윤곽/옆면을 설명했습니다. 세운 팔레트 f6의 native4/5/6 역할 문제까지 총5/9에서 명시적인 대응 문제가 있습니다. 나머지4개도 물리 native gold라는 뜻은 아닙니다. 기존 weak9는 특정 local-part 보임 대조로 보존합니다.
- **GT 부근으로 crop을 만들어도 형상·자기 가림의 모호함이 남습니다.** 명목 cuboid corner가 실제 판재의 빈틈·블록·다른 판의 아래쪽 윤곽과 겹칠 수 있습니다. 외부 물체가 보이는 것과 의도한 부위를 가리는 것은 별도 질문입니다. 19개 관찰 불일치는 모델 크기만 늘려 해결할 문제로 단정할 수 없습니다.
- **외부 가림의 다양성과 아래 부위 감독이 부족합니다.** 합의 E8개는 native4/5에만 있습니다. 합의 D/E는 native0/1/3/4/5에 있고 native2/6/7에는 없습니다. FIT 자연 E는5개 부위/5프레임이며 shift 지원 뒤 E6변형입니다. 360개라는 입력 수로 이 편중을 감추지 않습니다.
- **배포 초기 query와의 간극이 남습니다.** 새 GT 유래 family는 의도한 후보를 구성상 지정하지만 원 초기 검출기의 GT-instance 대응은 UNKNOWN입니다. GT query 학습이 실제 초기 오차에서 바로 작동한다고 보장하지 않습니다. 원 q/L/native/U는 그대로입니다.
- **이미 있던 조건 입력을 신규 해법으로 부르지 않습니다.** C61은 native ID와 다른7점의 상대XY/L 조건을 이미 사용했습니다. 같은32차원 정보를 다시 계산한 것만으로 새 아키텍처나 개선을 주장하지 않습니다. 이번 변화의 핵심은 감독 대상과 RGB·부위·query의 연결입니다.

## 실제 그림과 전체 기록

다음 그림은 GT 유래 family의 nominal 표시용 contact입니다. 학습 RGB96에는 마커가 없습니다. 전체 영상의 원 query와 명목 query를 구분해 표시하며, 표의 D/E 합의는 아래 원 관찰 ledger로 추적할 수 있습니다.

![frame2 candidate0: 실제 판재 끝과 runner/빈틈의 차이](public_source_gt_query_jitter_v1/F02_C000000_S0_CONTACT.png)

![frame6 candidate3: 세운 팔레트의 원 query와 명목 부위, 앞쪽 상자](public_source_gt_query_jitter_v1/F06_C000003_S0_CONTACT.png)

아래 대조는 특정 weak local part만 COVER로 바꾸고, 같은 pattern을 옮긴 SHAM에서는 그 부위를 보존합니다. `COVER`라는 이름만으로 원 native E 정답이 되지 않습니다.

![원 crop의 같은 부위 BEFORE COVER SHAM](weak_part_pair_controls_v1/FRAME_02_NATIVE_0_CARD.png)

- [전체15 GT-query contact 갤러리](public_source_gt_query_jitter_v1/PORTABLE_GALLERY_KO.md)
- [전체9 weak triplet 갤러리](weak_part_pair_controls_v1/PORTABLE_GALLERY_KO.md)
- [120개 질문을 crop·두 관찰·근거와 비교하는 HTML](ROOT_READOUT_KO.html)
- [360행 target·탈락 이유 CSV](joined_weak_intended_part_targets_v1/ALL360_TARGET_CENSUS.csv)
- [120행 원 관찰 보존 join JSONL](joined_weak_intended_part_targets_v1/ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl)
- [360행 입력·지원 검사·후속 weak 후보 JSONL](joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl)
- [표준 condition32 원96 실제 생성](deployable_query_condition_b/actual_conditions_v1/ALL96_DEPLOYABLE_CONDITION32.jsonl)
- [9 triplet 및360crop 전체 독립 픽셀 검산](independent_actual_data_review_b/REVIEW.json)
- [실제13장 root 시각 검수](ROOT_ACTUAL_WEAK9_AND_GT_QUERY_FIXED4_VISUAL_QA.json)
- [숫자·입력 binding·한계](ROOT_RESULTS_KO.json)

## 모델 성능은 어디까지인가

**최신 실제 모델 benchmark는 V213이고 이번에는 재평가하지 않았습니다.** 아래는 기존 공개 저장 집계의 역사적 비교입니다. 같은 D1147/E90 평가 응답을 읽은 것이며 같은 감독·학습 자료의 등조건 비교가 아닙니다. v61 E는 presence 응답 proxy, legacy는 과거 실사 CAL 학습 참조군입니다.

| 모델 | D 정답 / recall | E 정답 / recall | E precision | primary U |
|---|---:|---:|---:|---:|
| v61 역사 응답 | 260 / 22.67% | 88 / 97.78% | 9.03% | 0 |
| legacy | 655 / 57.11% | 83 / 92.22% | 20.65% | 177 |
| C55 (V191) | 831 / 72.45% | 37 / 41.11% | 10.48% | 0 |
| C63 (V213) | 843 / 73.50% | 19 / 21.11% | 5.88% | 0 |

C63은 C55보다 D12개를 더 보존하지만 E18개를 덜 찾습니다. legacy 대비 D188개 증가/E64개 감소입니다. **D 보존과 E 검출이 함께 개선됐다는 증거는 아직 없으며 모델을 채택하지 않았습니다.** U0은 강제 이진 출력 정책이지 불확실성 해결이 아닙니다. 반복 노출 평가223장은 개발 진단이며 fresh blind 일반화 성능이 아닙니다.

## 다음 실제 실험

기존 C61 구조·조건 입력을 재사용하고, 기존 합성 strong 감독에 이번 공개 실사 weak 부위 감독을 명시적으로 구분해 추가하는 작은 source-only 대조를 준비합니다. 원 query local-part 대조9와 새 GT query 자연 D/E27의 label authority를 분리하고, DEV 계보를 모든 학습·보조 손실·정규화에서 제외합니다. GT 유래 query 결과와 원 초기 query에서의 결과를 따로 기록합니다. 독립 장면이 부족한 native2/6/7·아래 러너·자연 E를 공개 SOURCE에서 보완하고, 근거가 불충분한 곳을 E로 채우지 않습니다.

SOURCE에서 보임 보존과 외부 가림 반응을 먼저 확인하고 checkpoint·문턱·보류 정책을 그 범위에서 고정합니다. 현재 보호 평가를 보고 이 후보들 중 하나를 고르지 않습니다. 새 학습 결과가 나오기 전까지 성공 가능성이나 전이 개선을 수치로 확정하지 않습니다. 자기 가림·화면 밖·위치/대응 불가는 외부 가림과 계속 분리합니다.

## 재현·출처·실행 한계

공개 데이터는 **MR6D, 배포자 anas-gouda**, [원 배포 페이지](https://huggingface.co/datasets/anas-gouda/mr6d)의 CC BY4.0 표기를 따른 파생 RGB입니다. 원래 저장된 context/crop에서 새 crop·표시 contact·checker 대조를 생성했습니다. 원 GT JSON, full/visible mask·depth·CAD·NPZ·모델 가중치는 이번 공개에 복사하지 않으며 경로/SHA provenance만 보존합니다. 학습 RGB는 unmarked입니다.

실제 생성은 weak18개 원 파생PNG decode와 GT-family12개 원 파생contextPNG decode로 총30개 source decode였습니다. 새 소유 PNG는 weak54+GT375=429개입니다. 독립 검산은 weak 전체 ROI/alpha/pixel과360개 RGB를 별도 sampler로 전량 재계산해 일치했습니다. 원 출력 파일을 덮어쓰지 않았고 원96/U11과 모든 SOURCE 역할을 보존했습니다. 실제 새 학습·모델 추론·보호 평가 본문 읽기/재평가·PnP는0입니다.

준비 중 GT-family V1은 plan의 tuple/list JSON roundtrip 비교 문제를 실제 생성 전에 발견했습니다. V2는 그 schema 처리를 수정했으며 sampler/render 함수는 동일합니다. 실패한 실제 crop 생성 뒤에 다시 만든 것이 아닙니다. 이전 pure fixture/metadata 오류와 교정 기록도 보존합니다. 재개 시 root의 한 번의 광역 filename 열거가 exact-path 제약을 벗어나 V112 경로를 출력한 metadata deviation은 [별도 기록](ROOT_RESUMPTION_METADATA_DISCOVERY_DEVIATION.json)에 남겼습니다. 해당 본문·hash·복사·모델/label 사용은 없었고 이후 정확 경로만 사용했습니다. 따라서 zero broad path enumeration이라고 주장하지 않습니다.

HTML과 상대 링크는 저장 파일 기준으로 검산합니다. 브라우저 보안 정책이 file: 페이지 접근을 차단하여 실제 브라우저 렌더 확인은 하지 못했습니다. 우회는 시도하지 않았습니다. [해당 한계 기록](ROOT_BROWSER_RENDER_LIMITATION.json).

문제 해결 목표는 계속 active이며 아직 미해결입니다.
