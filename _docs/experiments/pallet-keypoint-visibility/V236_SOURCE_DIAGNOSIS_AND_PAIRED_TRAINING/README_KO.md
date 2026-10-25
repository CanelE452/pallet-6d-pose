# V234–V236: 공개 실사 실패 진단과 같은 점 가림 대조 실험

**대조쌍 학습은 완료했으나 실사 가림 판별을 개선하지 못했다. 모델을 채택하지 않았다.** 현재223/held는 새로 읽거나 추론·학습·문턱 선택에 사용하지 않았다. PnP와 원 q/L/native도 변경하지 않았다.

|원 SOURCE 내부 DEV|고정 encoder/기존 head D·E recall|같은 구조 + 새 가림 대조쌍 D·E recall|
|---|---|---|
|SYN main N1344|98.84% · 91.07%|98.68% · 92.86%|
|YCB AUX N397|97.94% · 0/57|96.76% · 0/57|

사전 기준은 두 가족 평균 균형 recall +3pp 이상, 각 D/E recall 하락2pp 이하였다. 관측 gain **+0.113491pp**로 실패했다. SYN witness0와 YCB associated-r surface(native UNKNOWN)는 서로 다른 referent로 따로 평가한다. SOURCE DEV는 이미 노출된 내부 진단 자료이며 blind 평가나 현재 실사 성공이 아니다. 원7,617행 입력은 전부 available로 모델 U율0%; target−1인7,423행을 모델 보류로 세지 않는다.

## 결과와 실제 영상

- [V234 진단 통합 보고서](DIAGNOSIS_V234_KO.md): 코드 오류·좌표·라벨·물체×표면점 구성 차이 확인
- [V235 전체 CM/precision/recall/보류율/paired 변화](artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [전체7617행 비교 CSV](artifacts/V235_SOURCE_REPORT_C/results/PAIRED_ALL7617.csv)
- [같은 점에 가림물을 덮은 경우와 옆에 둔 경우](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/fixed2_prefit_gallery_v2/PREFIT_COUNTERFACTUALS.png)
- [실제 DEV 성공·실패·회복·악화 이미지](artifacts/V235_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [로컬 HTML gallery](artifacts/V235_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html)
- [작은 crop과 원 공개 장면을 나란히 보기](artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/THREE_SOURCE_CONTEXT_BOARD.png) · [로컬 HTML](artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/PORTABLE_CONTEXT_GALLERY_KO.html)
- [crop 밖 다른 query의 대상 픽셀 존재 진단](artifacts/V236_SOURCE_CONTEXT_DIAGNOSTIC/OTHER_QUERY_OWNERSHIP_SUMMARY.json) · [원 q8–실제 RGB 연결 경로](artifacts/V236_CONTEXT_ROUTE_C/KO.md)
- [사전 고정 실험 조건](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/PROTOCOL_BEFORE_DONOR_PIXELS_OR_TREATMENT.json) · [실제 학습 완료](artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/COMPLETE.json)
- [합성 쌍 독립 QA](artifacts/V235_CF_DATA_QA_B/actual_review_v2/REVIEW.json) · [continuous reference 추가 검산](artifacts/V235_CONTINUOUS_REF_QA_B/CONTINUOUS_REFERENCE_4TAP_QA.json) · [학습·저장 출력 독립 QA](artifacts/V235_TRAIN_COMPLETION_QA_B/ACTUAL_QA_COMPLETE.json)

## 이번에 실제로 한 일

V234에서는 YCB DEV E57에 같은 object×aux slot의 FIT E 지원이0/57임을 확인했다. 같은 조합의 FIT D는 있었다. q 자체도56/57은 다른 물체 픽셀 위였고, q–r≤3px인 E30도 모두 그랬다. 좌표 차이만으로 전체 실패를 설명할 수 없다는 근거다. loss·출력 indexing·행 pairing의 명백한 코드 오류는 허용된 코드/저장값 감사에서 발견하지 못했다. 식별 불가능이나 원인 확정의 증거는 아니다.

V235는 모델 구조를 고정한 데이터 대조다. 원 SOURCE FIT D5,406개에 공개 YCB FIT의64개 donor(물체별최대16, scene0..4)만 사용해 COVER와 SHAM을 만들었다. 같은 q/L에서 원 기준점 주변3×3의 모든 raw96 tap을 덮은 COVER는 E, 같은 donor를 옆에 두고 그 support를 보존한 SHAM은 D다. 성공5,367쌍(10,734영상), 실패39개는 masked로 기록하고 원 D를 유지했다. continuous r 자체의 네 tap도 전수 검산했다. donor가 없거나 S/U인 자료를 E로 만들지 않았다. 합성 쌍은 새 자연 관측이나 human gold가 아니다.

기존 CONTROL의 완료1000회 checkpoint/log/CSV를 bytes 그대로 재사용했다(새CONTROL update0). 같은 DINO 고정 encoder와1186→64→2 head(76,098개), 같은 초기 head·원 PCG17 ledger·AdamW·문턱0으로 새 head만1000회 학습했다. 각 가족8D8E 중 D절반은 성공 SHAM, E절반은 별도 PCG235의 성공 COVER로 대체했다. 새 head를 고정한 뒤 원7,617행을 출력했고, 기존 CONTROL의두logit을7,617행 전부 byte까지 재현했다. LoRA는 사용하지 않았다. SOURCE/증강 자료와 초기·최종 조건이 바뀐 부분을 로그와 ledger로 확인할 수 있다.

V236에서는 실패57개 중56개가 다른 원 query에 대상 물체 픽셀이 crop밖에 있음을 저장 SOURCE ownership 진단으로 확인했다. 세 전체장면에는 비슷한 상자가 여러 개 보인다. 이는 crop 밖 정보의 존재 증거이며 새 모델의 성공·인과·대상 정체의 새 human label을 뜻하지 않는다. 다음은 원 q8로 정의한 전체 영역을 실제 source RGB와 연결하는 입력 검증이다. C11 fullcontext는 이미 시도된 방법이므로 첫 context 사용이라고 주장하지 않는다. [다음 입력 검증 범위](artifacts/V236_SOURCE_CONTEXT_DIAGNOSTIC/ROOT_NEXT_INPUT_DECISION_AFTER_SOURCE_FAILURE.json)를 별도로 고정했다. 새 context 모델은 아직 학습하지 않았다.

## 다른 컴퓨터에서 재검산

이 폴더에서 Python3 표준 라이브러리만으로 실행한다.

```bash
python3 -B verify_package.py --package . --manifest MANIFEST.json
python3 -B artifacts/V235_SOURCE_REPORT_C/summarize_pair.py --verify-saved
python3 -B verify_cf_teacher.py --metadata artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_METADATA.jsonl.gz --alpha artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_ALPHA_PACKBITS.npy --parent-diagnostics artifacts/parent_SOURCE/SAMPLE_DIAGNOSTICS.jsonl.gz --ycb-reference artifacts/V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv
```

순서대로 전체 manifest 파일의 bytes, 저장7617행 CM/P/R/U/gate 산술, 저장 packed alpha와 원 q/L/reference의 가림 teacher support를 검증한다. 세 번째 도구는 원 RGB/valid/depth/donor나 모델을 재실행하지 않으며 그 한계를 결과에 표시한다. 절대 경로 pointer를 따라가지 않는다. 신경망 전체 학습을 단일 명령으로 재현하는 패키지는 아니다.

전체 신규 합성 RGB299MB와 기존 raw RGB/valid/prefix cache는 Git에 복제하지 않았다. 원 SHA·크기·조립 코드·전행 metadata·전 packed alpha·작은 모델·훈련로그·행별 예측·실제 사례는 포함한다. [원본과 배포 경로/SHA 대응](PUBLICATION_SOURCE_MAP.json)을 사용한다. 큰 JSONL은 gzip으로 무손실 저장했다. metadata 안의 당시 절대 경로는 실행 출처 기록이다. 원 context용SYN RGB440 buffer는 아직 열지 않은 다음 단계의 경로이며 현재 패키지가 그 픽셀을 검증한 것은 아니다.

공개 사진과 기존 encoder 관련 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록했다. 이전 v61/기존 가림 모델의 현재223 고정 비교는 [V229 기록](../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md)에 남아 있다. 이번 source gate가 실패했으므로 현재223에 새 후보 비교를 추가하지 않았다.
