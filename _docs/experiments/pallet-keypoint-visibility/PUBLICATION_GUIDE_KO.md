# 게시 자료를 읽고 검산하는 방법

[실험 목차](EXPERIMENT_INDEX_KO.md)에서 각 버전의 `PORTABLE_REPORT_KO.md`를 열면 GitHub에서 결과 표와 그림을 볼 수 있습니다. [전체 그림 목차](GALLERY_INDEX_KO.md)는 원 HTML의 literal image 순서를 보존한 별도 파생 문서입니다. 원 HTML·JSON·코드·CSV·실패 기록은 archive 아래에 함께 있습니다.

## 필요한 폴더만 받기

이 저장소에는 다른 연구의 큰 자료도 있으므로 Git의 sparse checkout을 사용할 수 있습니다.

```bash
git clone --filter=blob:none --sparse https://github.com/CanelE452/pallet-6d-pose.git
cd pallet-6d-pose
git sparse-checkout set _docs/experiments/pallet-keypoint-visibility
cd _docs/experiments/pallet-keypoint-visibility
```

GitHub는 HTML을 실행하지 않습니다. Markdown에서는 표·이미지가 바로 표시되고, HTML 갤러리는 내려받은 자료에서 열 수 있습니다. 원 HTML의 당시 로컬 절대경로는 provenance로 남아 있으므로 다른 파일을 따라갈 때는 portable Markdown 링크나 manifest의 대응 경로를 사용합니다. 이 게시 준비에서 브라우저 렌더를 새로 수행했다고 주장하지 않습니다.

## 파일 보존과 압축

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)은 원 경로, 원문 SHA256, 게시 상대경로, 게시 바이트 SHA256, 포함/제외 사유를 기록합니다. 1MiB 이상 큰 텍스트는 최대32MiB의 원문 구간별 gzip으로 무손실 저장했습니다. 논리 파일명 뒤의 `.parts.json`이 순서·각 조각 해시·원문 해시를 담은 descriptor입니다. portable 링크의 압축 안내는 이 실제 descriptor를 가리킵니다. 조각의 이름을 정렬해 임의로 합치지 말고 descriptor의 `parts` 순서를 따르십시오.

원본 코드는 당시 입력·환경 경로를 그대로 보존합니다. 전체 학습 재실행에는 각 protocol의 공식 데이터와 환경·대형 사전학습 weights가 별도로 필요합니다. 게시 패키지만으로 할 수 있는 검산은 저장 결과 재집계와 파일 무결성 검산입니다. 제외된 원 RGB/GT 마스크 은행·특징/입력 캐시·환경·대형 모델·센서 녹화는 manifest에 사유를 남겼습니다. 검토용 crop·overlay·contact sheet와 소형 자체 학습 head 및 저장 분류 출력은 포함했습니다.

## 모델을 실행하지 않고 숫자 재검산

Python 표준 라이브러리만 필요합니다. 다음 명령은 V184의 저장1784점 원장과 공개 결과표를 다시 집계합니다. 원래 사용 가능한1739점·입력 U45점, 같은 query D1147/E90의1237점, 네 모델과 ALL/WOOD/SQUARE의12개 혼동행렬·precision/recall·보류율이 일치해야 합니다. 보류는 recall 분모에 포함합니다.

```bash
python3 tools/verify_published_results.py --self-test
python3 tools/verify_published_results.py --root . --manifest PUBLICATION_MANIFEST.json --report NEW_RESULT_VERIFICATION.json
```

전체 원문은 다음과 같이 압축 해제 후 원 SHA256까지 검산할 수 있습니다. 출력 경로를 지정하지 않으면 원 파일을 복원하거나 덮어쓰지 않습니다.

```bash
python3 tools/restore_original_artifacts.py --self-test
python3 tools/restore_original_artifacts.py --root . --manifest PUBLICATION_MANIFEST.json --verify-all --report NEW_ORIGINAL_BYTE_VERIFICATION.json
```

게시 시 수행한 독립 검산 결과와 전체 링크·크기·credential 형식 검사는 [publication](publication/)에 남깁니다. 이 검산은 실험 방법·모델의 옳음을 추가로 보증하는 인증이 아니며, 각 단계의 원래 검토·실패·제한도 함께 읽어야 합니다.

## 후속 완료 실험 반영

완료한 범위와 최종 산출물을 고정하고 코드·입력 정의·결과·불리한 결과·UNKNOWN·실제 검토 그림·독립 검산을 남깁니다. 원문과 별도 portable 문서, 경로 대응과 해시를 갱신한 뒤 파일 보존과 지표를 검산하고 `main`에 반영합니다. 큰 자료는512MiB 이하의 게시 파일 묶음으로 나누어 commit/push하며 force push는 사용하지 않습니다. 이전 실험 기록은 덮어쓰지 않고 Git 이력과 snapshot을 보존합니다.

[지속 게시 정책](PUBLICATION_POLICY_KO.json)과 [원문 snapshot 도구 절차](tools/snapshot/PUBLICATION_WORKFLOW_KO.md)는 현재 사용자가 요청한 후속 게시 범위를 기록합니다. 이 기록 자체가 별도의 예약 실행을 생성하지는 않습니다.

현재 평가 자료와 반복 노출된 VAL/public held를 학습·보정·문턱·모델·architecture·데이터 선택에 사용하지 않는 제약을 유지합니다. V112는 이름 단계에서 제외하며 그 내용·해시·크기를 읽지 않습니다. 모든 자료의 게시 완료와 가림 모델의 목표 달성은 별개입니다.
