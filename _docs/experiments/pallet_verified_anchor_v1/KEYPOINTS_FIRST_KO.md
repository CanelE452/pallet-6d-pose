# 사용자 변경: 기존 어노테이션에서 점 먼저 입력

선정 18장은 그대로 유지합니다. 기존 `scripts/annotate/annotate.py`를 사용합니다.
새 Tk 상태 입력 도구 대신 **키포인트 입력 → 나중에 D/V 등 상태 검수** 순서입니다.

기존 PnP, 점 선택, 확대, 저장 기능은 그대로입니다. 카메라 내부 파라미터가 달라
15장/3장 두 목록으로 묶습니다. `s` 저장 후 다음 이미지, `n`/`p` 이동,
`[`/`]` 또는 TAB으로 목록 전환합니다.

저장은 `outputs/pallet_verified_anchor_v1/keypoints_first/`에만 합니다.
기존 평가 정답, 원본 사진, 앞서 입력한 Tk 상태 파일은 건드리지 않습니다.
현재 모델 예측이나 기존 정답 좌표를 가져오지 않습니다. 카메라 파라미터만 재사용합니다.

**프로토콜 정정:** PnP를 보면서 입력하므로 이 절차는 엄격한 blind first-pass가 아니라
PnP 도움을 받은 사람의 어노테이션입니다. 직접 클릭과 PnP 보완은 기존 도구의 source 기록으로 구분하고,
후속 D/V 검수 전에는 VERIFIED_LABELS로 승격하거나 모델을 재채점하지 않습니다.

```bash
python -m scripts.research.pallet_verified_anchor_v1.open_existing_annotation
```

## 키포인트 저장 확인 후

18장 중 16장 저장: CLEAN 6, MODERATE 6, SEVERE 4. 남은 SEVERE 2장은 미저장으로 보존합니다.
직접 클릭 75개, PnP 보완 52개, 직선 연장 1개입니다. 직접 클릭 중 3개는 원본 화면 밖이므로
visible-only 상태 검수 대상은 72개입니다. PnP·연장·화면 밖 좌표는 직접 보이는 정답으로 승격하지 않습니다.

`python -m scripts.research.pallet_verified_anchor_v1.review_saved_keypoints`

이 도구는 이미 찍은 점을 노란 원으로 강조합니다. D/V/U 등 상태만 입력하면 다음 점으로 이동합니다.
좌표를 다시 찍지 않습니다. 나머지 미입력점은 미분류/평가 제외로 남기며 가림 상태를 임의로 정하지 않습니다.
이 상태 확인은 18×8 전체 상태 분류를 완료했다는 뜻이 아니며, 아직 모델 재채점은 하지 않았습니다.
