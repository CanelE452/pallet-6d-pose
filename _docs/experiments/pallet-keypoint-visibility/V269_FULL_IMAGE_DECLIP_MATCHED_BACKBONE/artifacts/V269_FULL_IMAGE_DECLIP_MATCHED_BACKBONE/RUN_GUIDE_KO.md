# V269 재현 안내

이번 실험은 전체 영상의 동결된 공식 EVA/DeCLIP 특징 위에 동일한 팔레트 마스크 head를 학습한 비교입니다. 공식 텍스트 prompt 기반 zero-shot 분할 전체를 실행한 실험은 아닙니다.

## 저장 결과 검산

발행 패키지의 `artifacts/V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE` 폴더에서 실행합니다. CPU 검산에는 Python, NumPy, Pillow, PyTorch가 필요합니다. 합성 원본 픽셀 검산에는 SciPy도 필요합니다. 원본223장이나 대형 백본 가중치 없이 실행할 수 있습니다. 아래 출력 이름은 새 폴더 이름으로 바꾸면 재실행할 수 있습니다.

```bash
python -B evaluate_fullframe_v269.py --replay-metrics --inputs evaluation_v1/compact_inputs --out my_metrics_replay
python -B verify_metrics_independent_v269.py --out my_independent_metrics_check.json
python -B replay_fullframe_v269.py --out my_forward_replay
python -B analyze_syn_pixels_v269.py
```

`evaluation_v1/CLASSIFICATION.csv`에 목재/정사각형 및 전체의 CM·precision·recall·보류율이 있습니다. `SELF_RESPONSES.csv`는 자기 가림을 별도로 표시합니다. `EXTERNAL90_TRANSITIONS.csv`는 같은 외부 가림90점의 방식별 판정입니다. `evaluation_v1/INDEX_KO.html`에는 실제 전체 영상, 고정색 확률 지도, 마스크와 표시 전용 확대가 있습니다.

## 원래 실험

1. `fetch_official_assets_v269.py`의 고정 공식 commit/다운로드 URL과 SHA로 공식 소스·두 가중치를 받습니다. 대형 가중치와 타사 소스 전체는 Git에 중복하지 않았습니다.
2. 원래 합성 RGB/M4 1003장(학습768/검증235)을 원본 SHA와 대조합니다. 사용 목록은 source audit와 feature `FRAME_ORDER.jsonl`에 있습니다. 합성 RGB와 M4의 실제 파일은 전체 중복하지 않았습니다.
3. 전체 RGB를512×512로 변환하고 같은 CLIP 정규화를 적용합니다. KP crop, sliding window, GT 재중심 crop은 없습니다. `extract_syn_fullframes_v269.py`는 영상만 읽고 세 특징 배열을 만듭니다.
4. `PROTOCOL_BEFORE_FIT.json`에 코드·입력 SHA, 세 방식, 동일 seed17·배치8·AdamW(lr0.001, wd0.0001)·1000회·문턱0.5를 고정했습니다. `train_fullmask_v269.py`는 동일한26721개 파라미터 head만 학습합니다. 세 final1000을 모두 저장한 뒤 합성 DEV를 점수화했습니다.
5. `NEW_CONFIG_FROZEN_BEFORE_CURRENT.json` 이후 `infer_fullframe_v269.py`로 전체 실사223장의 지도를 저장합니다. 지도가 저장된 후 초기 query에서 raw logit를 bilinear로 읽고 sigmoid를 적용했습니다. 원래 native/초기좌표/입력불가45개를 유지했습니다.
6. `CURRENT_PREDICTIONS_FROZEN.json` 이후에만 기존 사람 주석과 합쳐 평가했습니다. 신규 학습이나 설정 선택에223장을 사용하지 않았으며, 이미 반복 열람한 개발 평가이므로 독립 블라인드 테스트라고 부르지 않습니다.

EVA_STANDARD는 원본 EVA의 공식 vanilla 경로, EVA_CSA는 같은 원본 가중치의 공식 CSA 경로, EVA_DECLIP은 공개 DeCLIP 가중치의 같은 CSA 경로입니다. OpenAI→EVA에는 구조와 사전학습 차이가 함께 있습니다. CSA→DeCLIP은 구조/읽기를 맞춘 적응 가중치 패키지의 차이입니다. 공개 DeCLIP에는 COCO 이미지·DINOv2 기반 dense 적응 이력이 있으며, 이번 팔레트용 추가 학습만 합성 전용입니다.

M4는 보이는 대상 팔레트 픽셀의 정답입니다. 여집합에는 배경·구멍·다른 물체가 함께 있어 외부 가림 원인 라벨과 다릅니다. 초기 query의 픽셀 소속을 특정 native 모서리의 가시성과 동일시할 수 없습니다. 실사 픽셀 마스크 GT가 없으므로 실사 IoU는 계산하지 않습니다. 새 PnP·좌표 보정 학습은 없습니다.
