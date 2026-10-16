# V39 큰 오류의 보정 방향·크기 식별

V38의 고정 특징에서 큰 오류 학습 비중, 전용 보정 예측기, 실제 큰 오류 선택을 분리한 빠른 진단이다. 새 신경망 아키텍처의 성능 검증이 아니다.

- 2개 선 후보 대조 × 3시드: 회귀 3개, 그룹 OOF 회귀 3개, 분류 2개 = 48개 HGB 모델, 각 60라운드. 별도 감사 재학습 3개.
- 기존 V38 중심 6개를 그대로 재사용한다. 신경망 업데이트 0.
- 기존 합성 12학습/4기준 선택/8확인 그룹, 동일 2CAD. 확인 200관측 중 130검출·1040점. 검출 실패 70개는 조건부 점 오차에서 제외된다.
- 큰 오류는 기존 점 GT 오차 ≥10px. GT는 학습 감독/진단에만 사용한다. 실제 선택 기준은 calibration 4그룹에서 고정하며 check로 조정하지 않는다.
- upstream 모델과 기존 offset은 과거 16그룹 학습 이력이 있다. 새로운 독립 최종 평가나 실사 성능으로 해석하지 않는다.

실행 환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, sklearn 1.4.1.post1, CPU. 기존 V34~V38 캐시/완료 결과가 필요하다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_hard_direction_v39/run39.py --phase all
```

`all`은 완료 마커가 있는 단계를 건너뛴다. 개별 단계는 `train`, `eval`, `audit`, `phenotypes`, `report`, `deliver`다. 완료 산출물의 명시적 단계 재실행은 기존 파일을 덮어쓰므로 후속 실험은 별도 결과 폴더로 분리한다. `deliver`에는 로컬 Chrome/X11 화면 접근이 필요하다. Discord 알림은 명시적 사용자 승인에 따라 완료 증거를 검증한 후 별도로 보낸다.

결과: `data/pallet/results/pallet_hard_direction_v39/`. `PROTOCOL.json`은 일차 실험, `PHENOTYPE_PROTOCOL.json`은 후속 기술 통계다. 모델/GT 분리와 재계산 검사는 `AUDIT.json`, 브라우저 검사는 `DELIVERY.json`, 최종 파일 바인딩은 `COMPLETION.json`을 본다.

큰 오류 AUROC 약 .93이지만 전용 방향 예측은 학습 밖에서 악화됐다. 실제 점 오차 10.463→10.484px로 안정성 기준 실패. 현재 고정 특징에서의 실패이며 점·선 융합의 원천적 불가능성을 증명하지 않는다.
