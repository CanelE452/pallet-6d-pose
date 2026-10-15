# 전체 자료 다운로드와 무결성 확인

[Markdown 결과 보고서](REPORT_KO.md) · [전체 자료 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)

이 snapshot은 현재 실험 자료와 코드를 보존한 연구 기록이다. v29의 전체 자세 평가가 완료된 모델 릴리스가 아니다. 기본 e는 유지하고 v29 CAL은 32/46장까지의 관측 상태다.

저장소 `main`에는 코드, Markdown/HTML 보고서, 실제 과정 이미지와 작은 판정·모델·봉인 JSON을 둔다. 큰 프레임별 후보·예측·입력·마스크·스냅샷은 아래 원본 바이트 보존 아카이브에 담았다. 큰 원시 파일을 참조하는 과거 보고서는 전체 아카이브를 해제한 디렉터리에서 볼 수 있다.

## 포함 범위

- 실험 파일 **26,344개**, 원본 **25,193,555,820 bytes**.
- 압축 합계 **5,737,224,251 bytes**, **11분할**.
- 가시성 실험 문서·이미지·관측·중간/최종 원장·코드 snapshot 및 관련 연구 코드 3개 디렉터리.
- 최신 고정 runtime protocol이 바인딩한 소스·입력·보호 파일 1,203개 포함.
- `.git`, Python/pytest cache, symlink와 새 publication 디렉터리는 archive에서 제외했다. publication 문서와 검산 원장은 이 `main` 폴더에 별도로 보존한다.
- 원본 실험 파일은 수정·삭제하지 않았다. 이전 main의 다른 연구 결과도 유지한다.

## 분할 다운로드

각 분할을 같은 디렉터리에 저장한다. 파일 이름과 순서를 유지한다.

| 파일 | bytes | SHA256 |
|---|---:|---|
| [pallet-visibility-publication-20261015-v1.tar.zst.part-000](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-000) | 536,870,912 | `6e45898f1a89cc8b69b5e67ab684118a27d75c6864eaba7980116ef620de0653` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-001](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-001) | 536,870,912 | `e24bb73b1dc055fb8d5001c3d13a1c56151345c2d714bf38ac65488ab5112a29` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-002](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-002) | 536,870,912 | `e58b03f670db5755bb4b17aee22307e5ca46bea8b9a7749d4947c26d5db99d4c` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-003](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-003) | 536,870,912 | `47654d679df5ae3488f6feb0f24c7184879a126de81e52e773dbb12c173de0c7` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-004](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-004) | 536,870,912 | `c34fc68ef1b06eb9fbf7641c942937ce4a5c475282ea6f627b285f63a931dea2` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-005](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-005) | 536,870,912 | `ae4c14c62fdd4df6147261c9527ba7a055c0d85a78b6eb49eb2d61dfb04a0ad0` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-006](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-006) | 536,870,912 | `fc5cc2971dd1d4c72983fa63630111845d0f2e91c9fb43c3ca68f237ae3997de` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-007](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-007) | 536,870,912 | `9a47c2d3747de8f384d35dc0ec765a72315f7ba193f0797ebd489463f4cec996` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-008](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-008) | 536,870,912 | `6a1f6b116fe002f65572d9c3007edc438c0754fbb09f2a9c39b96383187ac0eb` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-009](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-009) | 536,870,912 | `a8ef81cad763913d17d1e6ba2532c6874af0d75e13cd7605dc9b17ed840b5036` |
| [pallet-visibility-publication-20261015-v1.tar.zst.part-010](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/pallet-visibility-publication-20261015-v1.tar.zst.part-010) | 368,515,131 | `27f4b4d861246ddf475f0efceffabb96d49b88c833d4962e5063c9ef37879141` |

[분할 SHA256 목록](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/ARCHIVE_SHA256SUMS.txt) · [파일별 manifest](https://github.com/CanelE452/pallet-6d-pose/releases/download/visibility-curriculum-snapshot-20261015-v1/FILE_MANIFEST.jsonl.gz) · [압축 구성](ARCHIVE_FILES.json)

## 다운로드·해제 예시

GitHub CLI와 `zstd`가 설치된 환경에서는 다음처럼 받을 수 있다. 별도 빈 디렉터리에 해제한다.

```bash
gh release download visibility-curriculum-snapshot-20261015-v1 --repo CanelE452/pallet-6d-pose \
  --pattern 'pallet-visibility-publication-20261015-v1.tar.zst.part-*' \
  --pattern 'ARCHIVE_SHA256SUMS.txt'
sha256sum -c ARCHIVE_SHA256SUMS.txt
mkdir -p visibility_snapshot
cat pallet-visibility-publication-20261015-v1.tar.zst.part-* | zstd -d -c | tar -xf - -C visibility_snapshot
```

연결된 압축 스트림의 SHA256:

```text
90284bf0de3c538f53e63192f9a8f1bebbfb7961da9d1b924a6d3079d55e6290
```

파일명은 고정 폭 000~010이므로 위 glob의 사전순 연결이 원래 스트림 순서다.

## 검산 근거

- [파일별 경로·크기·SHA256](FILE_MANIFEST.jsonl.gz)
- [모든 tar 멤버 바이트 재검산](ARCHIVE_STREAM_VERIFICATION.json): 26,344개 모두 일치.
- [최종 소스·runtime 재대조](FINAL_SOURCE_RUNTIME_POSTCHECK.json): 1,470개, 실패 0.
- [최종 archive 봉인](PUBLICATION_FINAL_SEAL.json)
- [기존 v29 검증과 draft 실패 기록](VALIDATION.json) · [draft 수정 후 검증](VALIDATION_V2.json)
- [호스트 재시작과 CAL 완료행 보존](HOST_RESTART_AND_PARTIAL_CAL.json)

수치 성능·테스트 통과·게시 무결성은 각각 다른 검증이다. 위 archive 검산은 v29 정확도 개선이나 목표 달성의 증거가 아니다.

[GitHub의 큰 파일 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)와 [Release asset 안내](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)에 맞춰 큰 원시 증거를 Release 분할 파일로 제공한다.
