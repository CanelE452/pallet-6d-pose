# 공개 실험 기록 배치 제안

대상은 저장소의 `_docs/experiments/pallet-keypoint-visibility`입니다. A는 메타데이터·파생 Markdown·링크 검사만 준비했습니다. 원본 보고서와 Git 상태는 변경하지 않았습니다.

원본 P 파일은 `archive/task-2/pallet-keypoint-visibility/<P 상대 경로>`에 byte-exact로 보존하고, v60/v61/v62 형제 기준선은 `archive/task-2/<원 형제 폴더명>/...`에 보존합니다. 파일별 원 source SHA와 published path 대응은 C의 최종 manifest가 담당합니다. 대용량·외부 의존성은 없어진 것처럼 처리하지 않고 별도 파일 목록에 남깁니다.

`metadata_snapshot_final/portable_pages/<실험 폴더>/PORTABLE_REPORT_KO.md`는 최종 archive의 해당 실험 폴더에 추가할 파생 문서입니다. 원본 HTML/JSON을 바꾸지 않습니다. `metadata_snapshot_final/EXPERIMENT_INDEX_KO.md`는 최종 게시 루트에 배치하면 archive 상대 링크가 맞습니다. GitHub가 HTML을 직접 렌더하지 않는 점을 고려해 결과 표·그림과 방법·한계 발췌를 Markdown에도 제공합니다.

최신 단계는 V186의 고정 SOURCE 4프레임/5 GT instance 등록 진단과 독립 400-ray 계산 검산입니다. 최신 실사 분류 비교는 V184이고, V185는 source part-supervision readiness입니다. 등록 진단 완료를 모델 성능 개선으로 쓰지 않습니다. V184의 D recall 상승과 E recall 하락, 미채택, 전체 WOOD/SQUARE·native8 목표 미해결을 함께 유지합니다.

각 실험의 phase PASS와 전체 과제 해결을 분리합니다. source TRAIN apparent fit, 반복 노출된 synthetic VAL/current/public-held의 not-blind 한계, legacy 모델의 historical real-CAL 이력과 새 모델의 평가-only 실사 사용, point-owner와 regional end-support/native visibility의 서로 다른 감독을 유지합니다. 실패 실행·수정 전 code/checks·trace·불리한 결과와 전체 표를 유리한 결과로 덮지 않습니다. 65536 crops를 65536 독립 장면으로 세지 않습니다.

허용된 V63–V186 이름은 124개입니다(V111 이름 두 개 포함). V112는 이름 단계에서 제외했으며 stat/read/hash/search를 하지 않았습니다. 현재 알려진 최상위 ROOT 보고서가 있는 88개와 없는 36개를 구분했습니다. 보고서가 없다는 이유로 완료나 실패를 새로 단정하지 않았습니다. 이전 v60–v62 상태를 담은 README/START_HERE/HANDOFF는 역사적 원문으로 남기고 최신 인덱스로 먼저 진입하도록 합니다.

원문 절대 경로는 provenance로 보존합니다. `PORTABLE_TARGET_CROSSWALK.jsonl`에 원 reference와 project-relative/published-relative target을 별도 기록했습니다. 프로젝트 밖 원RGB 두 경로 등 외부 참조는 자동으로 읽거나 복사하지 않습니다. 원 HTML을 portable HTML로도 제공한다면 별도 파생본에서 href/src만 바꾸고 원본 SHA를 함께 기록합니다.

`DIRECT_REPORT_IMAGE_REFERENCES.jsonl`의 348개 이미지는 원 HTML/MD 또는 새 portable Markdown이 직접 연결한 고정 그림입니다. raw image-bank 전체와 구분해 C의 개별 포함 override에 사용할 수 있습니다. A는 이미지 내용을 새로 열거나 hash하지 않았습니다.

최종 stage 뒤에는 `prepare_portable_index.py --project-root <게시 루트> --audit-list <portable 문서 상대경로 JSON 배열> --audit-output <새 QA.json>`으로 명시된 문서의 로컬 링크 존재 여부를 점검할 수 있습니다. 이 검사는 symlink/V112를 따라가지 않고 대상 파일 내용을 읽지 않습니다. 최종 Git 추적 여부·remote 접근·브라우저 렌더·원문 SHA 동등성은 root/C의 실제 stage/push 검증과 구분합니다.

이후 닫힌 단계가 생길 때 같은 원문 보존·manifest·인덱스/Markdown 갱신·링크 검사·main push 절차를 이어가는 사용자의 지시는 root 게시 흐름에서 유지합니다. A가 별도 자동화나 push를 수행한 것으로 기록하지 않습니다.
