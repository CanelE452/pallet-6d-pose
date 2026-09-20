# N2 실사 적응: 원본 필터 → 보정 → 재필터

2026-09-19 사용자 요청에 따른 새 예비 실험. 기존 최종 모델은 보존한다.

1. 기존 R0의 미라벨 실사 원본 예측에 confidence 필터를 적용한다.
2. 통과한 이미지에만 frozen N2_DIM_ONLY(seed1)를 원본/반전 입력으로 적용한다.
3. 보정된 좌표에 기존 keypoint-removal/flip consistency 필터를 적용한다.
4. 최종 통과한 pseudo-label과 합성 GT replay로 N2의 **복사본만** 추가 학습한다.

기존 N2 / 합성-only 추가 학습 / 같은 최종 실사 이미지의 원본 pseudo / 보정 pseudo를 비교한다.
원본 pseudo arm은 보정 효과 분리를 위한 대조군이며 사용자가 요청한 주 실험은 REAL_REFINED다.
모든 추가 학습은 각 1,500 step, batch16, seed1, last checkpoint만 사용한다.
R0·teacher·정규화·decode cap·온도·평가 데이터는 변경하지 않는다.

실사 적응 풀은 기존 직사각 팔레트의 주간·야간 촬영이다. 초록 평가 150장은 학습에 넣지 않는다.
촬영 recording alias와 이미지 SHA로 DEV319/green150 중복을 배제한다.
모든 수치 기준은 학습/새 평가 전 PROTOCOL.json에 고정한다.

실행 환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` (실제 CUDA 접근 필요).

```bash
python -m unittest discover -s scripts/research/pallet_real_refiner_twostage_v1 -p 'test_*.py'
python -u scripts/research/pallet_real_refiner_twostage_v1/run.py all
```

진행: STATUS.json. 단계별 수급: STAGE1.json / STAGE2.json.
완료 후 RESULTS.json / RESULTS_KO.md. 기존 논문 표로 자동 승격하지 않는다.

## 실행 전 검사

회귀검사 7개 통과: 1차 탈락 시 정제기 미호출, 통과 시 호출, 2차 두 조건의
AND/비유한값 탈락, centroid를 유효 코너로 세지 않음, 비유한 원본 코너 탈락,
flip 왕복 좌표·confidence 보존, 입력 jitter의 target/centroid/무효점 불변.

첫 검사에서 정수형 bbox fixture의 norm 연산 오류가 발견되어, jitter에서 bbox를
좌표 dtype으로 명시 변환했다. 학습 전에 수정했고 PROTOCOL_PRETEST.json은
해당 수정 전 준비 기록으로만 보존한다. 실제 학습은 수정 후 PROTOCOL.json에
바인딩되며, 수치 설정이나 데이터 선별 기준은 바꾸지 않았다.

## 완료 (2026-09-19 21:21 KST)

1,000 → 272 → 259장(주간120/야간139). 3개 추가 학습 모두 1,500 step 완료.
학습 시간: 합성-only 100.7초, 원본 pseudo 77.9초, 보정 pseudo 77.3초.
DEV319와 초록150 모두 평가 완료. GPU 학습 프로세스 종료 확인.

이번 단일-seed 실험에서는 보정 pseudo가 같은 이미지의 원본 pseudo보다 낫지만,
기존 N2와 합성-only 대조군 대비 E_sym은 두 평가 집단 모두 나빠졌다.
기존 최종 모델을 교체하지 않는다. 자세한 수치는 RESULTS_KO.md 참고.

완료 검증: 3개 checkpoint의 protocol/input SHA·step·유한 가중치 검증 통과.
원본 최종 모델 및 코드 바인딩 유지. 469장×4개 보정기=1,876회 출력 비교에서
검출 수/선택 인덱스/박스/score/confidence/centroid/비선택 객체 보존을 재검증했다.
