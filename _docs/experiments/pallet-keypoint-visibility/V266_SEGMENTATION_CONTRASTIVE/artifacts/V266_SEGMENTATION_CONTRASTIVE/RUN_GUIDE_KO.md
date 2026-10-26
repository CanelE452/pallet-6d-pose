# V266 실행·재현 안내

현재 V264 / 영역 분할 추가 / 분할+대조 학습을 비교했습니다. 학습은 합성뿐입니다. 결과는 외부 가림 개선 실패이며 새 모델을 채택하지 않았습니다.

## 다른 컴퓨터에서 결과 검산

이 폴더를 복사한 뒤 Python3, NumPy, PyTorch CPU를 사용합니다. 실제 확인 환경은 Python3.10.20, NumPy1.26.4, PyTorch2.1.1+cu118입니다. 지표 검산에는 Pillow도 필요합니다. 다음 명령은 학습·PnP·대형 데이터·원본 실사 접근 없이 동작합니다. 결과 디렉토리 이름은 새 이름이어야 합니다.

```bash
python -B replay_predictions_v266.py --out replay_cpu_check
python -B evaluate_v266.py --replay-metrics --inputs evaluation_v1/compact_inputs --out replay_metrics_check
```

첫 명령은 전체1784 분류를22D 저장 descriptor와 체크포인트로 재현하고, 고정32개 입력의 dense branch 전체 forward도 재현합니다. 전체1784 dense map의 SHA를 검산하지만, 전체 dense forward를 다시 실행하는 것은 아닙니다. 둘째 명령은36개 지표행, 자기 가림 응답, 가림90점 전이 등을 재계산합니다. `METRICS_REPLAY_BYTECHECK.json`에 주요6출력의 바이트 일치를 기록했습니다.

`ROOT_READOUT_KO.html`과 하위 HTML은 폴더 구조를 유지해서 브라우저로 열면 이미지·표를 함께 볼 수 있습니다. GitHub에서는 패키지 README와 PNG를 바로 볼 수 있습니다.

## 학습 전체 재실행에 필요한 외부 입력

`PROTOCOL_BEFORE_FIT.json`, `data_v1/SOURCE_BUFFER_BINDINGS.json`, `DATA_INDEPENDENT_REVIEW.json`의 경로와 SHA가 원본 근거입니다. 대형 합성 DINO 특징1,384,120,448바이트와 실사 특징 약350MB는 Git에 중복 복사하지 않았습니다. 따라서 저장소만으로 원래 특징 추출과 전체 학습을 즉시 재실행할 수 있다고 주장하지 않습니다. 원본 렌더 buffer·동결 모델·특징이 있어야 가능합니다. 공개 패키지만으로 위의 분류/지표 검산은 가능합니다.

원 실행 순서는 `build_dense_labels_v266.py` → `prepare_training_packet_v266.py` → 조건 동결 → `train_syn_v266.py` → 두 final 동결 → `infer_current_v266.py` →1784예측 동결 → `evaluate_v266.py`입니다. 모든 출력은 새 폴더에만 기록됩니다. 학습·추론 스크립트는 원 경로/SHA로 입력을 검증하므로 새 환경에서는 경로를 명시적으로 재바인딩하고 새 실행으로 기록해야 합니다. 기존 봉인 protocol을 수정하지 마세요.

두 새 모델은 각1000업데이트, seed17, 동일초기값/학습행, AdamW lr.001/wd.0001, D16/E16 균형 표본입니다. 손실은 BCE+가중분할CE, 대조 방식만 +.1 pixel contrast(tau.1)입니다. 기본V264는 고정하고 새23,204개 파라미터만 학습합니다. 보류는 기존 입력불가만 유지하며 성능에 맞춘 confidence 문턱이 아닙니다.

## 평가 해석

실사223장은 반복 개발 평가입니다. 새 학습·통계 적합·문턱·체크포인트 선택에는 사용하지 않았지만, 독립 블라인드 검증으로 부르면 안 됩니다. PRIMARY1237은D1147/E90이며, SELF는별도입니다. 전체1784의45U와 PRIMARY의0U를 혼동하지 마세요. 실사 dense분할 정답은 없으므로 실사분할IoU도 없습니다.

초기 q와 위치 후보는 유지했습니다. 가려진 E의 참조좌표는 수동/투영/외삽 출처가 섞이며 직접 보이는 위치 정답이 아닙니다. 사용자 이견G996도 원 주석을 유지한 채 별도 표시합니다.

학습 전 pooled known_mass가1보다최대2.15e-6 큰 float32 누적오차를 확인해 수치검사 허용오차와[0,1]clip을 기록했습니다. 추가 적합이나 데이터 선택 변경이 아닙니다. 데이터 출처 COMPLETE의 basename충돌로 메타데이터 한 항목이 누락된 것은 독립 검토 JSON에 원 SHA로 보완했습니다. 학습 라벨/입력 바이트는 바꾸지 않았습니다.
