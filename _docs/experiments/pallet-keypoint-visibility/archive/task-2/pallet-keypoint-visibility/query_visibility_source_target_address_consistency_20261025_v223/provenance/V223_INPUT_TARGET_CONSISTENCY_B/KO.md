# V223 SOURCE 입력·목표 일관성 진단

**strong6096과 natural57/CF144 사이에는 같은 필수 runtime condition 입력이 없다. 따라서 이 두 supervision family 사이의 exact-input 반대 라벨 충돌은 배제된다.** 이는 namespace가 다르다는 추정이 아니라, 저장 q8 메타데이터로 재계산한 FP32 native/mask/XY32를 비교한 결과다. 전체 SOURCE의 모든 tensor 충돌을 반증한 결과는 아니며, 목표 혼용이 무해하다는 뜻도 아니다.

| 비교 | 저장 메타데이터에서 확인한 결과 | 해석 범위 |
|---|---|---|
| strong6096 ↔ natural57 / CF BEFORE48 / CF144 | condition32 일치 각각 0쌍; native+q/L 및 전체 q8도 각각 0쌍 | 필수 모델 입력 일부가 다르므로 이 family 간 exact full-input 충돌 없음 |
| strong6096 내부 | 저장 RGB/valid/condition fingerprint 3485그룹; 중복 unordered 6084쌍; 같은 fingerprint의 반대 target 0쌍 | RGB/valid digest는 원본 payload를 다시 열지 않은 저장 증거 |
| natural D48 ↔ CF BEFORE48 | q/L/q8/condition/valid 및 PNG 파일 SHA 동일, target도 모두 D0 | FIT39는 treatment에서 naturalweak와 CF BEFORE로 중복 가중된다. DEV9는 fit에 쓰이지 않는다 |
| 약감독201행의 같은 condition+valid, 반대 target | 348쌍, 모두 PNG 파일 SHA가 다름 | PNG 파일 SHA 차이만으로 decoded RGB 차이를 독립 입증하지 않음 |
| 그중 같은 parent 144쌍 | bound PROOF에 COVER target 변경>0, SHAM target 변경=0/보존 | 저장 pixel-change 증거상 RGB가 다름. 이번 B 진단에서는 픽셀을 재검산하지 않음 |
| 다른 jitter parent 204쌍 | FIT162/DEV42; q/L·전체 q8은 다름, 정규화 condition+valid는 같음 | 공통 decoded-RGB digest가 없어 전역 tensor 동일성 미확인. 이 쌍을 실제 모순으로 세지 않음 |

위 수는 행 쌍이며 독립 장면 수가 아니다. strong은 입력 메타7040 중 실제 point loss known6096(D5853/E243), 나머지944는 해당 known 출력 분모 밖이다. natural57은 FIT45(D39/E6)와 DEV12(D9/E3), CF144는 FIT117/DEV27이다. weak와 CF에서 같은 encoded PNG+valid+condition인데 target이 반대인 확인된 쌍은 0이다.

## 정의 차이와 실제 공유 경로

strong target은 합성 **실제 own-image의 point0**에 대해 target-alone D이고 combined D/E인 ray/point 상태다. query 중심 픽셀이나 generic wood/region owner 라벨이 아니다. natural target은 공개 실사의 GT-derived training query가 주소화한 **expected finite-part**에 대한 assistant weak consensus다. CF는 이 weak part를 덮는 COVER E와 보존하는 BEFORE/SHAM D이며 `native_point_DE_target=-1`을 유지한다. 두 권한을 physical native-point gold로 합치지 않는다.

V222 runner는 strong·natural·CF 모두 같은 `direct_end_external_logit`을 supervised scalar로 사용한다. 입력 조립도 같은 `base.assemble_local_input(..., use_sam=False)`와 native ID/other-mask8/XY16 호출이다. 따라서 **실제 중복 반대 라벨이 없어도, 하나의 scalar에 point와 finite-part의 서로 다른 질문을 학습시키는 의미적 결합은 남는다.** 이 감사는 negative transfer나 실사 실패 원인을 입증하지 않는다.

## 최소 다음 증거와 한 변수 대조

현재 미확인204쌍의 tensor 동일성을 닫으려면 다음 허용된 SOURCE 작업에서 동일 입력 조립 경계의 decoded uint8 RGB96·bool valid96·int64 native·FP32 mask/XY digest만 공통 형식으로 저장하면 된다. 모델 forward나 보호 데이터는 필요 없다. 현재 범위에서는 추가 PNG decode를 하지 않았다.

SOURCE 근거만으로는 encoder·조건·정규화·공통 draw를 유지하고 **point와 finite-part 출력의 gradient routing을 분리하는 대조**를 먼저 하는 편이, condition-before-pooling과 함께 바꾸는 것보다 직접 확인한 scalar 의미 결합을 한 변수로 구분하기 쉽다. shared encoder를 통한 간접 영향은 여전히 남고 성능 이득은 미정이다. pooling/conditioning 위치 변경은 별도의 구조 변수다. 이 문서는 구현·훈련·채택 권한이나 결과가 아니다.

## 범위

정확히 bound된 SOURCE JSON/JSONL과 runner/contract 두 Python 텍스트만 읽었다. 원 NPZ/PT/RGB/GT, 보호 평가, 이미지 뷰, 모델 import/forward/fit, broad scan, symlink follow, 과거 과학폴더 쓰기는 모두 0이다. 저장 예측값은 target 조인에만 사용했고 protected score 또는 모델 선택 근거로 사용하지 않았다. namespace 차이 자체는 충돌 반증으로 사용하지 않았다. 전체 충돌 0, 원 initializer 전이 성공, primary native gold, 목표 해결을 주장하지 않는다.

[전수 검산 결과](REVIEW.json) · [condition 일치와 실제 충돌을 구분한 348쌍](CONDITION_MATCHES_NOT_INPUT_CONTRADICTIONS.csv) · [검산 코드](check_saved_input_target_consistency.py) · [읽기 범위](READ_SCOPE.json)
