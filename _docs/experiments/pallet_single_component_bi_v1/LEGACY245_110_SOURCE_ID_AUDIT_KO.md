# 이전245·110 표기의 출처와 현재 모집단

기존 최소 B/I 실행의 입력 계약에는 옛245/110장 표와의 차집합이 미확정으로 남아 있었다. 원봉인 파일은 수정하지 않고, 현재 후속 감사에서 확인한 원자료 의미와 전체 ID 차이를 추가한다. [기계 판독 감사](LEGACY245_110_SOURCE_ID_AUDIT.json)에 원자료 SHA와 모든 ID를 남겼다.

| 실제로 확인한 자료 | 저장 행 | 고유 원영상 | 현재 TEST49와 차이 | 현재 PRIMARY192와 차이 |
|---|---:|---:|---|---|
| nonsevere/test_reused_v1 | 245 | 49 | 양방향 차집합0 | 이전에 없던 추가143만 |
| region_completion/TEST_N3 | 245 | 49 | 양방향 차집합0 | 이전에 없던 추가143만 |
| region_completion/TEST_PF | 245 | 49 | 양방향 차집합0 | 이전에 없던 추가143만 |

세 예측 gzip은 각각 자기 PREDICTION_SEAL의 SHA와 일치한다. **이245는49장×5방법의 예측 행 수이며245개 고유영상이 아니다.** 동일 원영상이 다섯 방법에 반복되는 것을 새 데이터로 세지 않았다. 현재 입력은 CAL46과 주평가 TEST49+추가143=192이며 총238개 원영상이다.

[기존 V10 보고서](../pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v10/REPORT_KO_FINAL_V1.md)의 `RGB q 보존 · native` 행에서110은192장 중 위치·회전 공동 성공 수다. 원 V32 `clean_fresh_BI_pose_eval_v4_32_r2/PRIMARY_SCORE.json`의192개 ID 순서가 현재 PRIMARY192와 정확히 같고, 같은 저장 지표로 다시 센 공동 성공이110개임을 확인했다. **이110개 성공 부분집합을 별도110장 평가셋으로 취급하지 않았다.** 성공110개 ID도 감사 JSON에 보존했다.

별도로 과거 evidence-connection 문서에는312개 원 annotation 중 다른 촬영245개를 감사했다는 기록도 있다. 이는 위49×5와 다른 의미이며, 해당 문서도 그245개를 원120장에 넣지 않았다고 명시한다. 이번 원문이 어느 다른245/110개 고유영상 표를 가리켰는지는 확정하지 못했다. 따라서 그 부분을 모두 확인했다고 소급 선언하거나 누락 원인을 가림/정사각형/오류로 추측하지 않는다.

검색 범위는 visibility-curriculum, N3/SubPix, static-closeout, final-paper-tables의 원문/manifest, 전체 실험 Markdown의245/110 모집단 표기 및 선택 JSON의 rows/frames/count/TEST 메타데이터, 보존된 archive/paper/evaluation 보고서다. 검색된 숫자를 모두 고유영상 수로 해석하지 않았다.

이번 감사는 저장 예측의 ID·방법 수·SHA와 이미 계산된 점수만 읽었다. 원영상·정답 좌표·모델·PnP·새 학습 호출은0이다. 기존952행 결과, 정답, 가시성 등급과 입력 봉인은 바꾸지 않았다. [기존 입력 계약의 미확정 기록](../pallet_minimal_bi_ablation_v1/contract/INPUT_CONTRACT_V1.json)을 그대로 보존한다.
