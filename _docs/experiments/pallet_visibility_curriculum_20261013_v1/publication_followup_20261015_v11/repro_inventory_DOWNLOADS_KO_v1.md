# 원시 재현 자료: 원본 snapshot 이후 전체 차분

이 문서는 이미지·코드·관측·입력·최종 자세 등 관련 정보를 원래 바이트로 보존하는 아카이브 안내다. 정확도 개선이나 방법 채택의 증거는 아니다. 기본 SAME_FROZEN_E 정책과 각 실험의 실패·보류 기록은 바뀌지 않았다. 현재 결과 설명은 [V11 보고서](REPORT_KO.md)를 따른다.

저장소 main에는 읽을 수 있는 그림·HTML·보고서·코드와 작은 결과를 둔다. 거대한 전체 INPUTS/E_COMPARISON/PRED/OBS 원장과 spool, 중복 source snapshot은 아래 Release 차분에 포함한다. 새 gain10의 작은 입력·관측·예측·마스크는 main에도 명시적으로 일부 포함하며, 중복 프레임 JSON까지 포함한 원형 전체는 아카이브에 보존한다.

## 정확한 범위

- 원본 Release: 26,344개 경로, 25,193,555,820 bytes. [기존 다운로드](../publication_20261015_v1/DOWNLOADS.md).
- 현재 검산 범위: **38,450개 경로, 48,673,271,110 bytes**. 기존 문서/연구 코드 4개 범위와 최신 32·33·gain10 protocol에 바인딩된 runtime 파일을 포함한다.
- 원본 Release와 정확히 같은 바이트: 26,343개. 새 차분: **12,107개, 23,479,722,138 bytes**.
- 차분은 동일 SHA 파일 7,269개를 tar hardlink로 보존한다. 모든 원래 경로가 남으며 독립 payload는 22,140,758,433 bytes다.
- 압축: **19,525,309,856 bytes / 37분할**. 각 분할은 512 MiB 이하로 Release 단일 파일 한도보다 작다.
- 최신 runtime source/input/protected 고유 1,733개 바인딩 일치. 전체 현재 파일은 파일당 한 번의 opaque SHA 계산으로 대조했다. 원장 JSON·영상·마스크·정답 수치는 decode하지 않았다.
- 압축을 다시 스트리밍해 **12,107개 모든 멤버의 원본 바이트 SHA 또는 hardlink 대상 동일성을 확인했다. 실패 0, 파일 추출 0**.
- git/index와 원본·보호 파일은 수정하거나 삭제하지 않았다. Python/pytest cache, .git, symlink, 명백한 기계 설정/credential 이름과 이 publication 폴더 자체는 제외한다. 이번 publication 문서·manifest는 main/Release asset으로 따로 제공한다.

## 다운로드와 복원

[추가 차분 Release 예정 경로](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-followup-20261015-v11) · [분할 SHA256](repro_inventory_archive_SHA256SUMS_v1.txt) · [압축 구성](repro_inventory_archive_files_v1.json)

먼저 원본 snapshot 11분할과 새 차분 37분할을 다운로드한다. ROOT의 Release 게시 완료 후 다음 명령을 사용할 수 있다. **현재 작업 checkout에 덮어쓰지 말고 별도 빈 디렉터리에 원본 → 차분 순서로 해제한다.**

```bash
gh release download visibility-curriculum-snapshot-20261015-v1 --repo CanelE452/pallet-6d-pose \
  --pattern 'pallet-visibility-publication-20261015-v1.tar.zst.part-*' --pattern ARCHIVE_SHA256SUMS.txt
mv ARCHIVE_SHA256SUMS.txt ORIGINAL_ARCHIVE_SHA256SUMS.txt
gh release download visibility-curriculum-followup-20261015-v11 --repo CanelE452/pallet-6d-pose \
  --pattern 'pallet-visibility-followup-20261015-v11.tar.zst.part-*' \
  --pattern 'repro_inventory_archive_SHA256SUMS_v1.txt'
sha256sum -c ORIGINAL_ARCHIVE_SHA256SUMS.txt
sha256sum -c repro_inventory_archive_SHA256SUMS_v1.txt
mkdir restored_visibility_workspace
cat pallet-visibility-publication-20261015-v1.tar.zst.part-* | zstd -d -c | tar -xf - -C restored_visibility_workspace
cat pallet-visibility-followup-20261015-v11.tar.zst.part-* | zstd -d -c | tar -xf - -C restored_visibility_workspace
```

파일명은 고정 폭 000–036이므로 사전순 연결이 압축 스트림 순서다. 연결된 새 스트림 SHA256은 다음과 같다.

```text
ea411bf41cc951ec674f8607e37f83541ff330b97ad57d1b0271c25aff6e6354
```

원래 바이트가 같은 파일은 hardlink로 복원되므로, 이후 개별 소스 편집 실험은 별도 복사본에서 진행하면 된다. main의 기존 보호 파일 6개는 현재 연구 workspace의 고정 바이트와 다르지만 main에서 덮어쓰지 않았다. 재현은 archive의 독립 복원 workspace를 사용한다.

## 검산 원장

- [현재 모든 파일 경로·크기·SHA 및 기존 공개본 대조](repro_inventory_current_files_v2.jsonl.gz)
- [추가 차분 파일·원래 경로·SHA 및 hardlink 대상](repro_inventory_delta_manifest_v1.jsonl.gz)
- [그룹별 수·용량과 100MB 이상 파일](repro_inventory_summary_v2.json)
- [모든 tar 멤버 스트림 검산](repro_inventory_archive_stream_verification_v1.json)
- [원본+차분 복원 커버 확인](repro_inventory_restoration_coverage_v1.json)
- [정확히 같은 내용의 중복 경로 기록](repro_inventory_duplicate_content_v1.json)
- [실제 packaging 소스](repro_inventory_pack_v1.py)

기존 snapshot과 달라진 동일 경로는 저장 CAL 모델 검산 보조 script 한 개이며, 원래 버전은 기존 snapshot에 남고 현재 버전은 이번 차분에 들어 있다. 기존 봉인과 실패 이력은 삭제하지 않았다. 아카이브에는 채택되지 않은 과거 후보·중간 자료도 포함되므로 보존과 유효성/채택을 구분한다.
