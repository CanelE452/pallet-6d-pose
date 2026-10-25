[V225: 원 query crop15장·장면분리 점검; 독립 분리 후보 FIT의 가림 weak8개, 새학습/성능개선 결과 아님](V225_DATA_AND_CROP_REVIEW/PORTABLE_REPORT_KO.md)

[V224: 새 학습 없이 원 초기 query 약라벨 연결·현재 loss 경로 점검; 성능 개선 결과 아님](V224_DATA_TARGET_READINESS/PORTABLE_REPORT_KO.md)

V223: 점·부위 출력 분리 대조와 입력일관성 검사를 완료했습니다. [표·전체crop·실행/QA·범위예외](EXPERIMENT_INDEX_THROUGH_V223_KO.md). 최신실사비교는V220이며 이번 SOURCE 공동개선은확인되지않았습니다.

V222: 예상 부위 COVER/SHAM 감독을 실제 학습했으나 SOURCE 검증 개선은 확인되지 않았습니다. [표·전체 crop·실행/QA 원문](EXPERIMENT_INDEX_THROUGH_V222_KO.md). 최신 실사 비교는V220입니다.

[V221 검토 대조·중복 시도와 남은 과제](EXPERIMENT_INDEX_THROUGH_V221_KO.md)

최신 모델은 보이는 점 보존이 늘었지만 외부 가림 검출은 함께 좋아지지 않았습니다. 첨부 V193 조언과 현재 계약·데이터·실험 이력을 대조한 결과이며 이번 새 학습·실사 추론은 없습니다.

---

# 초기 키포인트 crop의 보임·외부가림 실험 기록

초기 추론 좌표를 중심으로 자른 crop에서 **대상 팔레트가 보이는지, 다른 물체에 가려졌는지** 판단하는 연구입니다. 자기 가림은 초기 추론의 기하 처리와 구분하고, 위치 오차도 따로 분석합니다. WOOD/SQUARE와 native8 전체 목표는 아직 해결되지 않았습니다.

[전체 실험 목차](EXPERIMENT_INDEX_KO.md) · [전체 그림 목차](GALLERY_INDEX_KO.md) · [최신 실사 비교](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md) · [데이터·코드 출처](THIRD_PARTY_ATTRIBUTION_KO.md) · [게시·재검산 방법](PUBLICATION_GUIDE_KO.md)

이 게시본은 허용된 모든 단계의 실행 코드·방법·저장 출력·표·검산·시각화와 실패 기록을 함께 보존합니다. V63–V186의 허용된124개 디렉터리와 초기 v60/v61/v62 관련 자료를 연결합니다. 버전번호가 중복된 V111 두 디렉터리는 따로 남기고, 명시적으로 접근이 금지된 V112는 포함하지 않습니다. 단계별 PASS/완료는 모델 성공을 뜻하지 않습니다.

## 시작 문서와 초기 비교군

[START_HERE_KO](archive/task-2/pallet-keypoint-visibility/START_HERE_KO.md) · [HANDOFF_KO](archive/task-2/pallet-keypoint-visibility/HANDOFF_KO.md) · [PATH_INDEX.csv](archive/task-2/pallet-keypoint-visibility/PATH_INDEX.csv)

[v60 외부가림 proxy](archive/task-2/external_ray_proxy_head_20261022_v60/REPORT_KO.md) · [v61 target presence](archive/task-2/target_presence_classifier_20261022_v61/REPORT_KO.md) · [v62 기존 모델 저장 출력 감사](archive/task-2/presence_previous_models_audit_20261022_v62/AUDIT_COMPLETE.json)

## 현재 실제 분류 결과

V184의 같은 초기 query 비교에서 사람의 D(직접 보임)1,147점·E(외부가림)90점, 총1,237점입니다. 아래 recall 분모에는 보류 U가 포함됩니다. 자기 가림·영상 밖·미주석·대응 불명은 이 이진 표와 분리해 원문에 보존했습니다.

| 모델 | D precision | D recall | E precision | E recall | 보류율 | 외부가림을 보임으로 오판 |
|---|---|---|---|---|---|---|
| v61 | 99.24% | 22.67% | 9.03% | 97.78% | 0.00% | 2 |
| 기존 실사 학습 모델 | 99.54% | 57.11% | 20.65% | 92.22% | 14.31% | 3 |
| C52 RGB | 94.04% | 71.58% | 9.85% | 36.67% | 2.34% | 52 |
| C53 RGB+SAM | 92.54% | 75.68% | 6.83% | 22.22% | 0.49% | 70 |

새 모델은 보이는 점을 더 살렸지만 외부가림을 보임으로 통과시키는 오류가 커졌습니다. **C52/C53 모두 채택하지 않았습니다.** [재질별 혼동행렬·precision/recall·보류율과 실제 crop](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md)을 함께 확인하십시오.

![V184 고정 비교 결과](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/FIXED_PRIMARY_REAL_METRICS.png)

평가에 사용한 자체 촬영223장과 공개 held45장 및 합성 VAL은 반복 노출된 자료이며 독립적인 blind test가 아닙니다. 최근 학습은 합성·공개 SOURCE TRAIN을 사용했고 이 평가 자료는 학습·보정·문턱 선택에 넣지 않았습니다. 기존 실사 학습 모델의 과거 CAL 이력은 원문 그대로 남겼습니다. 공개자료의 weak center-owner 정답과 팔레트 native 부위의 외부가림 정답도 구분합니다.

## 최근 어디까지 확인했나

- [V183](archive/task-2/pallet-keypoint-visibility/query_visibility_query_jitter_paired_source_training_20261024_v183/PORTABLE_REPORT_KO.md): SOURCE TRAIN 좌표 jitter 학습과 두 head의 고정 최종 출력. source 결과와 실제 전이는 분리합니다.
- [V184](archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/PORTABLE_REPORT_KO.md): 저장된 v61/기존 모델과 새 head를 동일 query로 실제 비교. 보임 회복과 외부가림 손상을 모두 공개했습니다.
- [V185](archive/task-2/pallet-keypoint-visibility/query_visibility_public_part_supervision_readiness_20261024_v185/PORTABLE_REPORT_KO.md): 공개 부위 감독 준비도와 기존 crop·감독 영역 교차를 확인했습니다. 원래 crop 밖의 부위를 감독하는 사례가 있었고, UNKNOWN은 유지했습니다.
- [V186](archive/task-2/pallet-keypoint-visibility/query_visibility_public_source_registration_probe_20261024_v186/PORTABLE_REPORT_KO.md): 고정 공개 SOURCE4프레임·5instance의 원 mesh/K/R/t 렌더와 독립400 ray 검산을 완료했습니다. 제공 mask와의 외곽 정합은 높지만 센서 깊이 차이·결측이 있어 정확한 부위 가림 정답으로 승격하지 않았습니다. 새 모델 개선 결과가 아닙니다.

## 원본과 게시용 문서

원문은 `archive/task-2/`의 원래 형제 구조로 보존하고 각 버전의 `PORTABLE_REPORT_KO.md`를 GitHub에서 읽을 수 있게 별도로 만들었습니다. 원문에 남은 `/home/...` 경로는 당시 실행 환경의 provenance입니다. HTML은 GitHub에서 실행되지 않으므로 Markdown 표·이미지를 먼저 보고, HTML 갤러리는 자료를 내려받아 열 수 있습니다.

[원본 파일·해시·포함/제외 manifest](PUBLICATION_MANIFEST.json)에서 코드, 저장 출력, 소형 자체 학습 head, 검토 그림을 확인할 수 있습니다. 큰 텍스트 원장은 무손실 gzip 분할본과 원문 SHA256으로 보존합니다. 원 데이터 은행·특징 캐시·환경·대형 사전학습 weights 등은 경로와 제외 사유를 남깁니다. 다운로드가 필요한 공식 공개 자료와 라이선스는 [출처 표](THIRD_PARTY_ATTRIBUTION_KO.md)를 따릅니다.

이후 완료하는 실험도 같은 방식으로 원문·실패·그림·검산 결과를 묶어 `main`에 계속 반영합니다. 게시 검산은 숫자와 파일 보존 검산이며 전체 가림 모델의 목표 달성 판정과는 별개입니다.
