# V241: 원 장면을 일치시킨 가림 대조 데이터 준비

**현재까지 완료한 준비 자료를 공개합니다. V241 실제 데이터 생성·학습·성능 비교는 아직 하지 않았습니다.**
최근 완료한 모델 실험은 [V239–V240 결과](../V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md)입니다. 공개 YCB-V 내부 DEV 외부 가림은 local 0/57 → context 1/57이었고 사전 기준 미달로 미채택입니다. 현재 팔레트 평가 성능의 새 개선 결과가 아닙니다.

| 항목 | 현재 상태 | 확인 자료 |
|---|---|---|
| 이전 COVER/SHAM 생성·교사 정의 감사 | 코드·메타데이터 확인 완료 | [계약 감사](artifacts/V241_SCENE_CF_CONTRACT_A/KO.md), [원본 감사 JSON](artifacts/V241_SCENE_CF_CONTRACT_A/CONTRACT_AUDIT.json) |
| 부모 5,406개·donor64 출처 연결 | 메타데이터 연결·중복 검사 완료 | [설명](artifacts/V241_SCENE_CF_MANIFEST_C/KO.md), [행 수 표](artifacts/V241_SCENE_CF_MANIFEST_C/COUNTS.csv), [분할 검사](artifacts/V241_SCENE_CF_MANIFEST_C/CENSUS_AND_OVERLAP.json) |
| 원 장면 합성 → 전체 crop·문맥 추출 | 순수 계산 코드·가상 장면 4개 테스트 완료 | [코드](artifacts/V241_SCENE_CF_PRODUCER_B/scene_cf_math.py), [테스트 결과](artifacts/V241_SCENE_CF_PRODUCER_B/INVENTED_SCENE_CHECKS.json) |
| 실제 SOURCE 이미지 해독·파일 출력 driver | 미구현 | [준비 상태](artifacts/V241_SCENE_CF_PRODUCER_B/PURE_MATH_READY.json) |
| 실제 합성 데이터·독립 전량 검산·새 특징·학습·성능 | 미실행 | [전체 상태 JSON](STATUS.json) |

## 이번에 바꾸려는 부분

이전 V235는 활성 crop에만 COVER/SHAM을 붙였습니다. 이를 깨끗한 원 장면의 다른 query·문맥과 그대로 결합하면 입력끼리 같은 장면을 나타내지 않습니다. V241 초안은 원 해상도 장면에 가림층을 한 번 합성하고, 그 장면에서 원 q/L 그대로 8개 local crop과 문맥을 다시 추출합니다. 이 차이가 판별 성능을 개선하는지는 아직 확인하지 않았습니다.

부모는 원래 SOURCE FIT의 보임 D 전체 5,406개(합성4,565·YCB841)입니다. 원래 donor64·선택 순서·배치 규칙을 유지합니다. 이전 성공5,367개만 골라 쓰지 않고 실패39개도 목록에 남겼습니다. 부모1,198그룹·실제 원 RGB804개와 donor의 원 DEV group/image 중복은 각각0입니다. 부모와 donor는 같은 FIT 자료 일부를 공유하므로 독립 표본으로 해석하지 않습니다.

가림 교사는 q 중심의 단일 픽셀 정답으로 바꾸지 않습니다. 합성 witness와 공개 YCB associated-reference(r) 정의를 보존합니다. r·물체 ID·ownership은 감독/검증용이며 모델 입력이 아닙니다. 공개 YCB aux slot의 팔레트 물리 native 대응은 UNKNOWN입니다. 연속 r의 양의 보간 가중치 tap과 native half-up 3×3의 가림/SHAM 보존을 다시 확인하는 계산 초안이며, **실제 5,406개에서 새 증명을 통과했다는 결과는 없습니다.** 원래 q/L/native/U/분할을 유지하고, 교사 근거가 부족하면 구성한 쌍은 UNKNOWN으로 남기며 원 보임 D는 보존합니다. 문맥 입력이 원래 지원되지 않는 경우에는 0+flag를 유지하고 라벨 제외 조건으로 새로 쓰지 않습니다.

## 원자료·코드·실패 기록

- [부모 전체5,406행](artifacts/V241_SCENE_CF_MANIFEST_C/PARENTS_ALL5406.jsonl), [원 q8·실제 이미지 연결](artifacts/V241_SCENE_CF_MANIFEST_C/PARENT_GROUPS_INPUT_ONLY.jsonl), [이미지 허용 목록](artifacts/V241_SCENE_CF_MANIFEST_C/PARENT_IMAGE_WHITELIST.json), [donor64 메타데이터](artifacts/V241_SCENE_CF_MANIFEST_C/DONOR64_CLOSED_METADATA.json).
- [메타데이터 연결 코드](artifacts/V241_SCENE_CF_MANIFEST_C/build_closed_metadata.py), [계약·계산 정의](artifacts/V241_SCENE_CF_PRODUCER_B/PURE_DESIGN_AND_REQUIRED_METADATA.json), [가상 테스트 원본](artifacts/V241_SCENE_CF_PRODUCER_B/test_scene_cf_math.py).
- [첫 가상 테스트 실패](artifacts/V241_SCENE_CF_PRODUCER_B/FIRST_TOY_FAILURE.json), [두 번째 실패](artifacts/V241_SCENE_CF_PRODUCER_B/SECOND_TOY_FAILURE.json). 가운데 기준점과 큰 donor에서는 SHAM 위치가 없는 사례였습니다. 실패를 음성 fixture로 유지하고 작은 donor를 쓰는 고정 가상 사례를 양성 fixture로 추가했습니다. 실제 donor·제안 규칙을 변경한 실험은 아닙니다. 수정 전 테스트 두 개도 원본 그대로 포함합니다.
- [복사 원본·SHA 연결](PUBLICATION_SOURCE_MAP.json), [전체 파일 manifest](MANIFEST.json), [제3자 고지](THIRD_PARTY_NOTICES.md). 메타데이터의 절대 경로는 출처 기록이며 로컬에 없을 수 있습니다. 검증기는 그 경로를 따라가지 않습니다.

## 새 학습 없이 검증하기

저장소를 clone한 뒤 아래 폴더에서 실행합니다. 파일 무결성 검사는 Python 표준 라이브러리만 필요합니다. 가상 장면 계산 재실행에는 기존 NumPy 환경이 필요합니다. 원 SOURCE 이미지·평가셋·GPU·DINO 가중치를 읽지 않습니다.

```bash
cd _docs/experiments/pallet-keypoint-visibility/V241_SCENE_CONSISTENT_CF_PREPARATION
python -B verify_package.py --package . --manifest MANIFEST.json
python -B replay_invented_tests.py --package .
```

두 번째 명령은 원본 테스트와 계산 코드를 임시 폴더에 복사해 실행하고 저장된 결과와 비교합니다. 공개 원본을 덮어쓰지 않습니다. 가상 계산 통과는 실제 데이터의 교사 타당성이나 모델 성능 통과를 뜻하지 않습니다. 독립 수학 검토와 실제 생성·전량 검산이 남아 있습니다.

## 이전 결과·시각화 보기

- [V240 혼동행렬·precision/recall·보류율](../V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V240_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md).
- [V240 실제 성공/실패 crop](../V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png), [원 문맥·q8 갤러리](../V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html).
- [V235–V236 기존 가림 대조 학습·결과·코드](../V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/README_KO.md).
- [실험 전체 색인](../EXPERIMENT_INDEX_KO.md).

이번 V241에는 새 실제 crop 시각화가 없습니다. 현재223/held 새 사용·문턱 조정·PnP도 없습니다. 이 공개는 진행 상태를 확인하기 위한 기록이며 문제 해결 완료 보고가 아닙니다.
