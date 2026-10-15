# main 게시 지시 변경과 보존

사용자는2026-10-16(KST)에 새 브랜치를 만들지 말고 기존 결과를 main에 push하도록 명시했다. 이 요청이 앞선 CLI 지시문§15의 연구 브랜치 게시 규칙을 대체한다.

이에 따라 원격 main의 `f1a30071382f6304fe13d6c546b8f8033c6b9f49`에서 기존 두 실험을 포함한 `dea11c25132bfb87a433073853bb632fb5170658`로 일반 fast-forward push를 완료하고 원격 ref를 재확인했다. 추가254경로는 모두 신규 파일이며 기존 운영 코드·논문·사용자 dirty 파일을 변경한 diff는 없다.

- 최소 B/I 비교: [보고서](../pallet_minimal_bi_ablation_v1/RESULTS_KO.md), [갤러리](../pallet_minimal_bi_ablation_v1/gallery/GALLERY_KO.md).
- 제한된 A/B 후속 비교: [보고서](../pallet_guarded_bi_final_v1/RESULTS_KO.md), [미채택 권고](../pallet_guarded_bi_final_v1/FINAL_RECOMMENDATION_KO.md), [갤러리](../pallet_guarded_bi_final_v1/gallery/GALLERY_KO.md).

개별 N3/SubPix 추가 실험도 같은 main의 후속 커밋으로 게시한다. 새 브랜치는 생성하지 않았다. 작업에는 기존 격리 worktree의 detached HEAD를 사용했다. 원래 작업 디렉터리의 미커밋 수정과 다수 미추적 자료를 덮어쓰거나 reset하지 않았다. 기존 연구 브랜치와 실패 기록도 보존했다.

과거 보고서의 'main 미변경' 표시는 그 실험 종료 시점의 기록이다. 이번 명시적 게시 지시 이후의 원격 상태는 이 문서와 최종 커밋 ref로 구분한다. 운영 기본값을 새 방법으로 바꾼 것은 아니다.
