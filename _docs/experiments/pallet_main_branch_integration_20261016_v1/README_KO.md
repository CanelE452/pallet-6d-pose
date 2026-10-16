# 2026-10-16 연구 브랜치 main 통합

사용자의 모든 연구 브랜치 main 통합 요청에 따라 최신 origin/main `96360c61418c435602875f5f34ca8ea4c694fbdc`를 기준으로 병합했다. 이미 포함된 브랜치는 중복 병합하지 않았고, 로컬·원격 끝점이 다른 경우 원격 최신 끝점이 로컬 끝점을 포함하는지 확인했다. 다른 저장소를 가리키는 old-origin은 대상에서 제외했다.

최초 전체 이력 병합에서는 30개 refs가 모두 조상이었으나 GitHub가 point-line 원본 커밋의 229,034,780/348,846,341바이트 캐시 두 개 때문에 main push를 GH001로 거부했다. 원본 브랜치와 최초 통합 이력을 보존하고, 이 두 파일만 Git tree에서 Release 자료로 옮긴 동등 커밋 `9b7e2dc91be444a6edd81a37521a9ecebf02cfa5`으로 게시 그래프를 구성했다. 나머지 29개 원 refs는 최종 main의 조상이며, 원본 point-line 커밋 자체는 최종 main의 조상이 아니다. 원본의 다른 코드·문서·결과 blob과 두 캐시의 바이트를 보존했다. 기존 원격 이력의 강제 재작성은 하지 않았다.

| 새로 통합한 계열 | 원래 끝점 |
|---|---|
| 상태별 관측 전달·대응 진단 | `407de865` |
| observation-refiner / robust PnP | `818a2121` |
| point-line / geometry archive | 원본 `d969217e`, 게시 동등본 `9b7e2dc9` |
| SubPix → N3 순서 비교 | `d2ecb51c` |
| visibility curriculum | `fd8f6fbe` |
| joint-action / teacher 동일 이력 | `a22fb14b` |

코드·결과·그림과 기존 커밋의 아카이브를 함께 보존했다. 충돌은 최상위 readme와 2026-10-06 이력 두 문서였으며, 양쪽 실험 기록과 링크를 유지했다. point-line 브랜치가 정리한 18개 노트는 새 경로를 유지하고 이전 main 경로의 원문도 복원해 옛 링크를 보존했다. 과거 실험의 당시 완료·미완료 설명은 새 실행 결과로 바꾸지 않았다.

원 브랜치 변경 파일의 Git blob을 통합 결과와 대조했다. 의도적으로 합친 두 문서 외에 생존 파일의 내용 차이는 없었다. 기존 운영 scripts/src/config 파일 수정은 없고, 원래 작업 폴더의 사용자 수정 6개와 tracked 상태도 보존했다. 브랜치를 삭제하거나 강제 push하지 않았다.

- 최근 상태별 진단 패키지: **72 passed**, failed 0.
- 새로 들어온 Python 원본·스냅샷 2,065개: 구문 검사 실패 0.
- 최초 전체 이력의 30 refs 검사 통과. 실제 게시 이력은 29 refs 직접 포함 + point-line 게시 동등본이며, 예외를 숨기지 않는다.
- 이전 노트 18개 경로: 기존 main 원문과 byte 단위 동일.
- 새 학습·새 추론·성능 재평가: 실행하지 않음.

[실제 통합 감사](MERGE_AUDIT.json), [이전·새 노트 경로](RESTORED_NOTE_PATHS.json), [최신 상태별 결과](../pallet_statewise_correspondence_diagnosis_v1/REPORT_KO.md)를 연결한다. 최종 commit과 원격 SHA 일치는 정상 HEAD:main push 후 별도 게시 영수증 및 최종 응답에 기록한다.

## 대형 캐시 복원

[Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/pallet-point-line-v4-cache-20261016-v1)에 원본 두 파일을 올렸다. 로컬 원 Git blob의 SHA-256과 GitHub가 반환한 업로드 digest·바이트 크기가 두 파일 모두 같다. [정확한 경로·해시·다운로드 URL](LARGE_ASSETS.json), [업로드 검증](RELEASE_VERIFY.json)을 사용한다. 원래 캐시를 사용하는 과거 실험을 재현할 때만 저장소 루트에서 복원한다.

```bash
gh release download pallet-point-line-v4-cache-20261016-v1 --repo CanelE452/pallet-6d-pose --pattern SPATIAL_FEATURE_CACHE.npz --dir challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_e_spatial_fusion
gh release download pallet-point-line-v4-cache-20261016-v1 --repo CanelE452/pallet-6d-pose --pattern SPATIAL_FEATURE_CACHE_60K.npz --dir challenge/yolo_pose_one_model/spatial_concat_scratch
```

기존 파일을 덮어쓰는 옵션은 사용하지 않았다. 다운로드 후 LARGE_ASSETS의 SHA-256과 비교한다. 이번 통합에서는 해당 과거 학습·추론을 재실행하지 않았다.
