# V253 공개 저장 진단 재현

새 학습이나 모델 추론 없이 저장 Source 점수와 참조 6개 입력에서 9개 표·선택 파일을 다시 만들고 SHA를 비교합니다. 선택 FIT 이미지 파일은 패키지 hash로 검산합니다. 원210MB RGB 전체·교사 생성·모델 학습은 재현 범위에 포함하지 않습니다. 원 라벨에서 유래한 target과 입력 condition32는 읽습니다.

필요 환경: Python3.10, NumPy1.26.4. Root 실제 검산도 이 환경을 사용합니다. Torch와 backbone 가중치는 필요하지 않습니다.

저장소를 clone한 뒤 `runtime_dependencies.json`의 6개 파일을 저장소 안에 그대로 두십시오. 공개 entry template을 복사하고 `approved_public_saved_SOURCE_diagnosis_replay`를 true, `output_directory`를 새 절대 경로로 바꾸십시오. 원 template은 보존하십시오. wrapper_code의 SHA와 나머지 바인딩은 바꾸지 않습니다. wrapper_code.path는 출처 metadata이며 실행 검증은 SHA·size를 사용합니다.

```bash
python -B /ABS/REPO/_docs/experiments/pallet-keypoint-visibility/V253_SAVED_FAILURE_SOURCE_FIT_CROP_AUDIT/artifacts/V253_PUBLIC_REPLAY_ROOT/verify_saved_diagnosis.py --public-repo /ABS/REPO --package /ABS/REPO/_docs/experiments/pallet-keypoint-visibility/V253_SAVED_FAILURE_SOURCE_FIT_CROP_AUDIT --entry /ABS/ENTRY_TRUE.json --output-dir /ABS/FRESH_REPLAY_OUTPUT
```

`COMPLETE.json`의 PASS는 공개 저장 점수의 대응·표·고정 FIT 선택 재현을 뜻합니다. 모델 성능 개선이나 실사 팔레트 전이를 뜻하지 않습니다. 절대 출처 경로를 자동으로 따라가거나 원 로컬 폴더로 fallback하지 않습니다. 모든 입력·출력은 수정하지 않습니다.
