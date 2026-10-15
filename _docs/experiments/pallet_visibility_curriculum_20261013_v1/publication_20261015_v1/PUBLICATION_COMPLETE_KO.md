# main 게시와 원본 자료 공개 완료

2026-10-15 기준, [실험 보고서와 실제 이미지](REPORT_KO.md)를 `main`에 게시했다. 실험 snapshot 커밋은 [`bf8a12cdd0e7ea72e69f11aa63bf60a28972cc85`](https://github.com/CanelE452/pallet-6d-pose/commit/bf8a12cdd0e7ea72e69f11aa63bf60a28972cc85)다.

[전체 자료 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)는 공개 상태다. [다운로드·해제·SHA256 확인 방법](DOWNLOADS.md)에 따라 11개 분할 파일을 받을 수 있다. 원본 실험 파일 26,344개, 25,193,555,820 bytes를 보존했으며 압축 합계는 5,737,224,251 bytes다.

원격 `main`의 전체 Git tree와 6,538개 게시 경로, Release asset 30개의 크기·SHA256을 대조했다. 별도로 인증 없는 공개 주소에서 보고서·이미지·다운로드 접근을 확인했다. 검증 영수증은 [원격 tree와 asset 검증](REMOTE_MAIN_TREE_REVIEW.json), [공개 후 접근 검증](REMOTE_PUBLICATION_COMPLETION.json)이다. 기존 원격 `main`의 17개 후속 커밋과 다른 연구 자료를 보존했다.

정확한 실험 재현에는 전체 압축본을 별도 디렉터리에 복원해야 한다. `main`은 코드·보고서·과정 이미지와 작은 증거 파일을 제공하며, 원시 입력·프레임 후보·마스크·코드 snapshot 전체를 포함한 실행 환경 자체는 아니다. 특히 runtime 입력으로 고정된 MobileSAM 코드 16개와 CAL 코드 snapshot 1개는 압축본에 포함되어 있다.

이 게시물은 관측 시점의 연구 기록이다. v29 전체 자세 평가 완료나 성능 목표 달성을 뜻하지 않는다. snapshot의 CAL 관측은 32/46장이며, 게시 후에는 이 32장을 재측정하지 않고 남은 14장을 같은 관측 함수로 이어간다. 이후 생성되는 결과는 이 snapshot과 구분하여 기록한다.
