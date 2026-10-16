# 연구 노트 목차

최신 의사 라벨 검사: [Hough 보정 품질·평균 개선과 남은 오류](supervision_and_distillation/hough-pseudolabel-quality-v1.md). 학생 학습 없이 DEV 품질만 검증했다.

최근 아키텍처 실행: [V44 새 장면 두 검증·현재 고정 특징 보정 실험 계열 종료](point_line_fusion/04_failure_analysis/pallet_fresh_decision_v44.md).

> 2026-09-10 정리. 기존51개 노트를 관련 주제로 묶고, 이후 [V25 실행 결과](point_line_fusion/04_failure_analysis/pallet_observation_v25.md) 및 [V26 실제 카메라·신뢰도 진단](point_line_fusion/04_failure_analysis/pallet_camera_trust_v26.md)을 추가했다.

## 지금 점·선 연구를 이어갈 때

**[현재 결론과 시각적 요약](point_line_fusion/STATUS_AND_NEXT_STEPS.md)**부터 읽는다. 주요 성능 그림, 현재 구조도, 실제 실패 이미지, 확인한 원인과 미확정 가설이 있다.

그다음 **[다음 실험 실행 순서](point_line_fusion/NEXT_EXPERIMENTS.md)**에서 이미 해 본 실험과 앞으로 바꿀 요인, 대조군과 중단 기준을 확인한다.

| 주제 | 시작 문서 | 포함 내용 |
|---|---|---|
| 점·선 융합 | [실험 색인](point_line_fusion/README.md) | 초기 Hough/기하, V4~V44, 설계·결과·GT 감사 |
| 자세 기하 | [목차](pose_geometry/README.md) | C4 대칭, DiffPnP, PnP solver |
| 감독·증류 | [목차](supervision_and_distillation/README.md) | 정정 실사 감독, 다중 교사, self-training 과거 대조 |

## 구조

```text
notes/
├── README.md
├── FILE_MAP.json
├── point_line_fusion/
│   ├── README.md
│   ├── STATUS_AND_NEXT_STEPS.md     # 현재 판단과 그림
│   ├── NEXT_EXPERIMENTS.md          # 미실행 제안과 판정 기준
│   ├── 01_hough_and_geometry/
│   ├── 02_local_and_identity/
│   ├── 03_correction_and_stability/
│   └── 04_failure_analysis/
├── pose_geometry/
└── supervision_and_distillation/
```

그림과 출처 정보는 [공유 그림 폴더](../assets/point_line_fusion/README.md)에 둔다. 논문 정본은 별도 [paper](../paper/README.md) 트랙이며, 이 연구 노트의 재사용DEV 결과를 논문 최종 수치로 대체하지 않는다.

## 앞으로 추가하거나 과거 보고서를 재생성할 때

- 새 문서는 해당 주제 폴더에 두고 그 폴더의 README에 링크를 추가한다. 완료한 실험과 아직 실행하지 않은 제안을 구분한다.
- 실험마다 데이터·체크포인트·GT 유효 정책·지표·시드·학습 예산·결과 근거·미해결 범위를 적는다.
- 본문 이미지에는 출처·모집단·선택 방식·표시 의미를 붙인다. 샘플 위치를 attention 중요도로 부르지 않는다.
- 과거 실험의 보고서 생성 코드는 출처 해시 보존을 위해 바꾸지 않았다. 이 코드가 예전 평면 경로를 다시 만들면 아래 정리 도구로 이동한다. 목적지와 내용이 다르면 덮어쓰지 않고 충돌을 알린다.

```bash
python scripts/docs/organize_research_notes.py
python scripts/docs/organize_research_notes.py --apply
```

저장소 루트에서 실행한다. [이전→현재 경로](FILE_MAP.json) · [이동 기록](../assets/point_line_fusion/ORGANIZATION_RECEIPT.json)
