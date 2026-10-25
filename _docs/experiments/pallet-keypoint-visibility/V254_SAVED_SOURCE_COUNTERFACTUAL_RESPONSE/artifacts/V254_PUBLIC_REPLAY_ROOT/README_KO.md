# V254 저장 출력 산술 재현

Python 3.10·NumPy 1.26.4 환경에서 실행했다. GPU·모델 체크포인트는 필요하지 않다. 패키지 루트를 `--package`로, 아직 존재하지 않는 새 디렉터리를 `--output`으로 지정한다.

```bash
python -B artifacts/V254_PUBLIC_REPLAY_ROOT/verify_saved_cf_response.py --package /absolute/path/to/V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE --output /absolute/path/to/new_v254_replay
```

패키지 안의 모든 파일을 manifest와 대조하고, 포함된 정확한 여섯 입력만으로 계산한 여섯 결과 파일을 저장 결과와 바이트 단위로 비교한다. `PUBLIC_REPLAY_QA.json`이 성공을 기록한다. 원 절대경로는 출처 정보이며 입력 탐색이나 fallback에 사용하지 않는다.

이는 저장된 Source 반응의 산술 재현이다. 원 렌더·교사 생성·encoder·학습·실제 팔레트 평가를 다시 실행하는 절차가 아니다. 두 SHAM 층은 BEFORE/COVER를 공유하며, FIT 소속과 개별 행의 실제 학습 draw 노출도 구분한다.
