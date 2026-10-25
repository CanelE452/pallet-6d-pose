# V251 공개 CPU 저장 추론 검산

이 wrapper는 최종 SOURCE feature·bank·checkpoint를 읽어 BASE_FROZEN_V239 / CONTROL_NO_REFERENCE / TREAT_REFERENCE의 원7617행 저장 CSV를 검산합니다. 학습·optimizer·bank 재적합·encoder/RGB/GT 추론은 하지 않습니다. NumPy와 PyTorch가 설치되어야 합니다.

```bash
python -B /absolute/package/artifacts/V251_PUBLIC_REPLAY_A/verify_saved_fit.py \
  --public-repo /absolute/public/repository \
  --package /absolute/final/package \
  --entry /absolute/late/public-replay-entry.json \
  --output-dir /absolute/fresh/replay-output
```

`ENTRY_FALSE_TEMPLATE.json`의 false는 실행을 거절합니다. Root가 both-final SOURCE 완료 및 QA 후 package 상대 경로를 채운 별도 true entry를 만듭니다. 기본 scientific 입력14개는 `dependency_map.json`의 기존 공개 경로만, pure utility CODE1개는 `implementation_dependency.json`의 공개 경로만 읽습니다. local 원본 절대경로는 provenance 비교만 하고 따라가지 않습니다. 이번 결과의 producer CODE·final CP2·bank·CSV3·fit COMPLETE·final freeze는 late entry의 package 상대경로와 SHA/size만 사용합니다. 어떤 숫자/CP decode보다 모든 입력 bytes 검증과 both-final barrier 검증이 먼저입니다.

동결 B V2 CODE에서 ReferenceResidual/normalize/reference_inputs, V250 CODE에서 unit/select_reference_bank, V239 CODE에서 state_sha/context_args만 AST로 추출합니다. bank 제작·fit 함수와 optimizer는 호출하지 않습니다. exact V239 core와 residual을 weights_only CPU로 읽고 64행 배치로 같은 원 feature·mask·기하를 사용합니다. CONTROL reference9는 0, TREAT는 저장 bank와 input-only selector를 사용합니다. core 계산을 공유해 BASE도 함께 확인합니다.

허용오차는 사전 고정 atol=rtol=1e-5이며 세 모델의 원7617행 예측은 모두 정확히 같아야 합니다. CPU/GPU float 바이트 동일성을 주장하지 않습니다. 실패는 FAILURE.json과 부분 output을 남기고 자동 재시도하지 않습니다. 최종 package에서 단 한 번 실제 replay 후 Root가 receipt를 게시합니다.

SOURCE bank는 FIT-D 감독 통계이며 DEV는 이미 반복 노출된 진단 자료입니다. SYN 원 native 주소 bank를 물리적 part template으로, YCB associated-r를 literal q-owner/native gold로 부르지 않습니다. 현재/held 평가·새 모델/문턱/특징 선택을 하지 않습니다. 이 준비 단계에서는 CODE/metadata만 읽었으며 실제 array/CSV/CP 숫자 읽기·replay 실행은 0회입니다.
