# V267 재현 안내

원래 사용자 합성 RGB1003/M4로 학습한 전체 이미지 CLIP mask head 두 모델입니다. 전체 영상512를 입력하고, 예측된32×32 지도에서 원래 query를 나중에 읽습니다. 모델 입력에 초기 KP crop은 없습니다. 확대 그림은 표시 전용입니다.

## 공개 패키지만으로 검산

Python3.10, NumPy1.26.4, PyTorch2.1.1, Pillow12.2 환경에서 확인했습니다. 지표 replay는 Pillow도 사용합니다. 아래 새 출력 폴더 이름은 이미 존재하면 안 됩니다.

```bash
python -B replay_fullframe_v267.py --out replay_cpu_check
python -B evaluate_fullframe_v267.py --replay-metrics --inputs evaluation_v2/compact_inputs --out replay_metrics_check
```

첫 명령은 고정5개 전체 영상의 mask-head forward/arm과 전체1784개의 raw 지도 lookup을 재현합니다. frozen CLIP backbone 자체를 다시 실행하는 것은 아닙니다. 둘째 명령은5방법×3집단×4평가군=60행의 CM/P/R/F1/U와 외부 가림 전이를 재계산합니다. 기존 지표6출력은 바이트 일치했습니다.

`ROOT_READOUT_KO.html`은 폴더 구조를 유지해 브라우저에서 여세요. GitHub에서는 README와 PNG를 직접 볼 수 있습니다.

## 전체 학습 재실행

원래 RGB/M4 전체1003장 corpus와 각각약1.05GB의 SYN frozen 특징 두 개,350.8MB 일반 CLIP 가중치는 Git에 중복 복사하지 않았습니다. 시각 대조용으로 선택한6장의 정확한 원본 RGB/M4 쌍은 포함했습니다. 원본 경로·SHA·학습 분할과 공식 소스 commit은 JSON에 있습니다. 저장소만으로 원래 데이터의 전체 학습을 즉시 재실행할 수 있다고 주장하지 않습니다. 위 compact 검산은 원본 데이터와 backbone 가중치 없이 가능합니다.

공식 소스: https://github.com/xmed-lab/CLIP_Surgery/tree/d4696d47f49cfe70f49140afe5eb94f94c5f59bc

`fullframe_clip_v267.py`의 OFFICIAL_CODE와 WEIGHTS_URL/SHA를 확인하세요. official_source/clip/clip_model.py 및 clip_surgery_model.py, generic pretrained/ViT-B-16.pt를 해당 버전으로 준비해야 backbone 추론을 다시 실행할 수 있습니다. 이 실험은 공식 image architecture surgery를 사용했으며 text feature surgery/zero-shot pipeline은 실행하지 않았습니다.

실행 순서: 원본 데이터 감사→M4전체 soft-target→두 frozen encoder 특징→프로토콜 동결→두1000업데이트 fit→두 final 동결→SYNDEV scoring→실사223 전체 mask/1784 lookup 저장·동결→late GT join→평가/시각화. 원 경로에 묶인 protocol은 수정하지 말고 새 환경에서 명시적 재바인딩과 새 output directory로 기록하세요.

## 해석

M4는 target/non-target입니다. non-target를 E로 대응한 점 단위 수치는 human D/E와의 의미 전이 평가이며, 배경과 실제 occluder를 구분하는 별도 E 정답으로 학습하지 않았습니다. 낮은 mask 확률만으로 외부 가림 원인을 확정하면 안 됩니다. 실사 pixel GT는 없으므로 real IoU를 주장하지 않습니다.

같은0.5 기준의 개선형 D precision99.0%는595D/601선택점이며, 보임 recall51.87%와 E precision13.21%도 함께 봐야 합니다. 실사223은 반복 개발 평가이고 블라인드 검증이 아닙니다. 원래45 입력불가U와 자기 가림 별도 집계를 유지했습니다.
