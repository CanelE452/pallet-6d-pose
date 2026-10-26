# V268 전체 이미지 경계 보정 재현

원래 합성 RGB/M4 학습768·DEV235장만 사용해 두 full-frame refiner를 각1000회 학습했습니다. 두 final1000을 먼저 동결하고 전체 실사223장 지도와1784점 예측을 저장한 뒤 주석을 연결했습니다. 추가 fit·문턱·좌표·PnP 변경은 없습니다.

## 저장소 사본만으로 검산

Python3.10, PyTorch2.1.1+cu118, NumPy1.26.4, Pillow12.2에서 확인했습니다. 아래 출력 폴더는 새 이름이어야 합니다. `artifacts/V268_FULL_IMAGE_BOUNDARY_REFINEMENT` 안에서 실행합니다.

```bash
python -B replay_fullframe_v268.py --out replay_cpu_check
python -B evaluate_fullframe_v268.py --replay-metrics --inputs evaluation_v1/compact_inputs --out replay_metrics_check
python -B replay_syn_pixels_v268.py
```

첫 명령은 고정5개 전체 영상의 RGB256/CLIP특징/기존 coarse32를 이용한 새 refiner forward 두 모델과 전체1784점 map lookup을 CPU에서 재현합니다. CLIP backbone 자체를 다시 실행하지 않습니다. 둘째는6방법×3집단×4평가군=72행 CM/P/R/U, 자기 가림18행, 전이를 재계산합니다. 원본 RGB/주석/CLIP weights는 필요하지 않습니다. 지표6파일은 바이트 동일했습니다.

원 source에서 실행하면 query manifest를 옆 V267에서 읽고, 게시 사본은 `dependencies/V267_FULL_IMAGE_CLIP_VISIBLE_MASK`의 같은 SHA 사본을 읽습니다. compact 지표 검산용 이전 evaluator CODE와 ledger/예측은 `evaluation_v1/compact_inputs`에 있습니다.

세 번째 명령은 NumPy·SciPy·PyTorch로 합성235장의 원본 픽셀 CM/IoU/Dice와 경계1px/3px P/R/F1을 재현합니다. 작은 M4 packbits/이전 coarse32 fixture와 두 새 DEV 지도만 읽으며 원본 이미지·M4·모델은 읽지 않습니다. [정의와 파일 목록](syn_pixel_replay_v1/README_KO.md). 세 모델의 모든 aggregate 카운트와 비율이 봉인 보고서와 일치했습니다.

`ROOT_READOUT_KO.html`과 각 `INDEX_KO.html`은 폴더 구조째 내려받아 브라우저에서 여세요. GitHub에서는 README·PNG·CSV를 직접 볼 수 있습니다.

## 학습과 전체 추론

입력 순서: 원본 RGB전체→PIL bicubic512→2×2평균/255→FP16 RGB256입니다. 기존 동결 CLIP Surgery 영상 구조 특징은 같은 전체512에서 얻은32격자 token입니다. 초기 KP crop·GT crop은 없습니다.

정답은 원본 M4 PNG>127→torch bilinear512 align_corners=False/antialias=True→2×2평균 soft256입니다.32격자 정답을 늘린 것이 아닙니다. 경계 band는 soft256≥.5에서 radius2의 양쪽 경계를 계산하며 손실 가중치에만 씁니다. 출력/정답 마스크 팽창이나 구멍 채우기는 하지 않았습니다.

새26177파라미터 refiner의 출력은 bilinear(coarse32→256)+6*tanh(raw_delta/6)입니다. A는 FIT통계로만 정한 balanced BCE+Dice, B는 동일 손실+0.5경계 양class 균형 BCE입니다. seed17·PCG64배치1000×8·초기값·구조는 같습니다. source `PROTOCOL_BEFORE_FIT.json`의 SHA로 코드·입력을 동결했습니다.

전체 학습 재실행에는 원래1003 RGB/M4 corpus와 V267 frozenCS SYN tokens 약1.05GB, RGB256 입력 약394MB가 필요합니다. 이 대형 입력은 Git에 중복하지 않고 원경로/SHA를 남겼습니다. 선택한6개 합성 사례의 실제 RGB/M4와 예측, 새 두 head,235 DEV 저장 지도,223 real 저장 지도/1784점 판정,compact fixture는 포함했습니다. 저장소만으로 원본1003 전체 재학습을 즉시 실행할 수 있다고 주장하지 않습니다.

실행 순서: `prepare_refine_inputs_v268.py`→`seal_protocol_v268.py`→`train_mask_refine_v268.py`→`infer_fullframe_v268.py`→`evaluate_fullframe_v268.py`. 기존 절대경로 protocol은 고치지 말고 새 환경에는 명시적 재바인딩과 새 출력 폴더를 기록하세요. 원본 소스·범용 CLIP 출처는 V267의 고정 commit/SHA를 이어갑니다. text feature surgery/zero-shot pipeline을 추가 실행한 것은 아닙니다.

## 해석

M4는 pixel target/non-target 정답입니다. 주석 E에는 비대상 출력을 대응한 MASK_PRESENCE_PROXY로 평가하며, 모델이 실제 외부 가림 원인을 직접 학습한 것은 아닙니다. 자기 가림은 별도 집계하고 기존 입력불가45U를 유지했습니다.223장은 반복 개발 평가로 독립 블라인드 검증은 아닙니다. 실사 pixel GT가 없어 real segmentation IoU를 보고하지 않습니다.

256 재보간만으로 과거32 직접 qlookup과2/1739판정이 달라지는 수치 효과를 별도 기록했습니다. 이는 학습/정책 후보가 아니며 사후 문턱을 선택하지 않았습니다. 같은256 grid에서 |coarse logit|>6의 부호 보존과 모든 실제 픽셀의 제한된 residual을 검사했습니다.
