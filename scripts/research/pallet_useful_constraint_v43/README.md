# V43 개선 가능한 유효 선 학습

V42에서 놓친 유효한 개선 후보를 학습 목표·비중·순위로 회복할 수 있는지 검증한다. 고정 특징을 쓰는 소형 분류 실험이며 새로운 DHT/YOLO 신경망 학습은 아니다.

- V41 actual physical3 후보와 direct fallback을 고정한다.
- valid: V42의 목표점 제약 정확도. useful: valid이고 후보 점 오차가 fallback보다 1px 초과 작음. 실제 추론에 GT를 넣지 않는다.
- valid/useful × 기본/개선 사례 4배 가중의 영상+기하 4대조, useful 4배 가중 기하 대조 1개, 3시드. 15 HGB 분류기 ×60라운드=900라운드, 감사 재학습 2개 별도.
- 모든 새 모델에 후보-minus-fallback 정규화 변위2D·길이 특징을 추가한다. 기존 잔차 양의 이득 순위와 분류 점수 순위를 분리한다. historical V42 비교에는 특징·calibration 변경 효과도 섞이므로 인과 비교는 이번 동일 입력 대조군끼리 한다.
- 그룹별 프레임 가중치를 역할 슬롯으로 확장한 뒤 무효 슬롯을 제외한다. 가중 대조는 useful 양성에 고정4배 적용하고 정규화한다. 학습 후 점수는 보정된 확률이 아니다.
- 같은 2 CAD·12 fit/4 calibration/8 재사용 check. 상류 모델·offset 과거16그룹 노출. 확인200관측 중130검출·1040점, 미검출70은 점 오차 제외. 새로운 렌더/실사평가/GT수정/신경망 업데이트 없음.
- cal4에서 .10~.95 고정격자로 precision≥90%, 실제 변화≥10점·2그룹, 각 조건 악화≤1%를 만족하는 양의 그룹가중 점 이득 최댓값을 선택한다. 없으면1.1기권. check로 기준을 조정하지 않는다.

## 실행

저장소 루트에서:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_useful_constraint_v43/run43.py --phase all
```

`all`은 완료 마커가 있는 단계를 건너뛴다. 개별 단계: train, eval, audit, diagnose, audit_diagnostic, report, deliver. 개별 재실행은 결과를 덮어쓰므로 완료 후 변경 실험은 새 버전에서 한다. Chrome/X11 전달은 접근 권한이 필요하다. Discord 알림은 별도 완료 후 절차이며 사용자 승인된 완료 사실만 전송한다.

PROTOCOL/SUMMARY는 일차 결과, DIAGNOSTIC_PROTOCOL/CURVES는 그 후 무학습 고정 격자 분석이다. 고정 문턱 결과와 이미지를 최종 채택 성능으로 읽지 않는다. 감사는 모델15개, 점45배열, 비학습 영역 교란 재학습2개, GT 분리, 전체격자1620행 독립 계산이다. COMPLETION은 결과/코드/노트/핸드오프 해시를 묶는다.

## 결과

유용성 AP check .224~.341로 유효성만 학습한 matched 대조 .044~.055보다 높다. 하지만 주판정 useful_weighted_visual_probability는 세 시드 모두 기권하여 기존10.410px를 유지했다. 고정0.10 진단에서 유효 개선 기회는45/39/39점으로 늘지만 실제1px초과 피해82/42/55점으로 평균10.498px다. 새 모델 채택 없음. 학습 밖 그룹에서 생성한 후보/복귀점의 효용 감독과 피해·이득 분리, 새 그룹 고정 검증은 미실행 범위다.
