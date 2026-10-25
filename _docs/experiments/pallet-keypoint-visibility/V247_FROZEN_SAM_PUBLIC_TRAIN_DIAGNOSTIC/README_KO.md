# V247 · 공개 TRAIN 원 query에서 동결 SAM2.1을 실제 확인한 결과

**SAM은 상자 가림을 분리하는 데 유용한 모습을 보였지만, 마스크 안/밖을 그대로 native 키포인트 보임/가림으로 쓰기에는 부족했습니다. 새 모델을 채택하지 않았으며 문제는 아직 해결되지 않았습니다.**

V246에서 공개 실사의 초기 query 대상과 교사 대상이 서로 다를 수 있음을 확인한 뒤, 기존 SAM2.1을 원 query 8개의 min/max box로 실제 실행했습니다. 공개 TRAIN 22쌍·44프레임·352query를 고정했고, prompt 가능한 42프레임에서 set_image/predict를 각각 42회 실행했습니다. 두 프레임은 원 q8 prompt를 만들 수 없어 그대로 보류했습니다. 새 학습·현재 평가셋 추론·PnP는 0입니다. 이것은 기존 모델과 규칙을 새 TRAIN 코호트에 연결한 진단이며, 새로운 SAM 모델/기법의 발명이 아닙니다.

[고정 네 장과 원 crop 갤러리](V247_FIXED4_SAM_VISUALS_A/gallery_v1/PORTABLE_GALLERY_KO.html) · [독립 검산 결과](V247_SAVED_SAM_QA_C/REVIEW.json) · [전체 352행 CSV](V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/QUERY352.csv) · [실제 실행 완료 기록](V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/COMPLETE.json)

## 1. 조건부 근거와 SAM 관측을 나누어 집계

행의 V/E는 V246에서 저장한 **조건부 픽셀 근거**입니다. 근사 pose/mesh, 동일 카메라/장면, RGB-depth 격자 등의 가정이 붙어 있고, native part 대응이나 실제 키포인트 위치 정답을 인증하지 않습니다. 열은 SAM이 선택한 **보이는 영역 제안**에 원 query가 포함되는지입니다. 마스크 밖을 E·SELF·키포인트 오류로 바꾸지 않았습니다.

원 query의 정확한 반올림 픽셀, logit > 0:

| 저장 조건부 근거 | SAM 마스크 지지 | 마스크 밖 | 추론 보류 | 합계 |
|---|---:|---:|---:|---:|
| V: 보임 근거 | 12 | 13 | 0 | 25 |
| E: 외부 가림 근거 | 2 | 7 | 0 | 9 |
| UNKNOWN | 104 | 158 | 56 | 318 |
| 전체 | 118 | 178 | 56 | 352 |

조건부 V 25점 중 13점은 SAM의 정확한 중심 픽셀에서 지지가 없었습니다. 조건부 E 9점 중 2점은 지지가 있었습니다. **현재 팔레트 평가 정확도나 native 보임/외부 가림 confusion matrix로 해석하면 안 됩니다.** UNKNOWN 318점을 전부 가림 또는 좌표 오류로 취급하지 않았습니다.

사전 고정한 반경 2의 clipped 5×5 근거/마스크 관측:

| 저장 조건부 근거 | SAM 마스크 지지 | 마스크 밖 | 추론 보류 | 합계 |
|---|---:|---:|---:|---:|
| V | 17 | 12 | 0 | 29 |
| E | 2 | 5 | 0 | 7 |
| UNKNOWN | 162 | 98 | 56 | 316 |
| 전체 | 181 | 115 | 56 | 352 |

두 번째 표는 **교사 근거를 판정하는 영역도 바뀌므로 분모가 다릅니다.** 12/25 → 17/29를 같은 25점에서 recall이 개선됐다고 읽으면 안 됩니다. 이 반경을 평가 결과에 맞추어 선택하지 않았습니다. 원 query는 이동하지 않았습니다.

원 연속 좌표 입력 U는 55점입니다. 반올림 후 영상 밖이 되는 1점을 추가로 지지할 수 없어 최종 관측 U는 56점입니다. 전체 352행과 원 q/L/native 번호·입력 U·순서·split을 보존했습니다. 실제 원 native 점까지의 좌표 오차는 이 공개 자료에서 인증되지 않았습니다.

## 2. 실제로 확인한 성공과 한계

모델 결과를 보기 전에 고정한 첫 두 TRAIN 쌍의 네 장을 모두 확인했습니다. 각 카드에는 원 RGB와 원 q8 box, SAM overlay, 원래의 clean96 crop 8개, exact/5×5 관측이 있습니다. clean crop에는 표시를 덧그리지 않았고 기존 V246 crop의 바이트를 보존했습니다.

- **1.92m 전후 쌍:** 초기 q8과 SAM은 뒤쪽의 작은 흰 팔레트를 따라갑니다. 공개 교사 대상은 앞쪽 목재 bin입니다. 뒤쪽에도 실제 팔레트가 있으므로 초기 추론이 존재하지 않는 물체를 골랐다고 단정할 수 없습니다. 앞 bin의 교사 라벨을 뒤 팔레트 query에 적용해서는 안 됩니다.
- **0.55m 가리기 전:** SAM은 앞 목재 면과 밑부분을 잡고, 여러 노출 목재 crop에는 지지가 있습니다. 오른쪽/아래 경계의 일부 원 query는 정확한 픽셀에서 마스크 밖입니다. native2·3은 사전 고정한 5×5 관측에서는 지지가 생깁니다.
- **0.55m 상자로 가린 후:** 왼쪽 상자 영역을 대체로 제외하면서 중앙/오른쪽 목재를 유지합니다. 그러나 노출된 오른쪽 목재가 crop 안에 있어도 원 query 중심은 마스크 밖일 수 있습니다. native1은 5×5에서 지지가 생기고 native2는 계속 밖입니다.

위 관찰은 고정된 사례 설명입니다. 전체 자료의 빈도, 과거 모든 실패의 단일 원인, 각 query의 native 정답을 뜻하지 않습니다. **crop에 목재가 보임 / query 픽셀에 보이는 표면이 있음 / 의도한 native 키포인트가 그 위치에 있음**은 구분해야 합니다.

## 3. 무엇을 검증했고 무엇을 검증하지 않았나

기존 SAM2.1 Hiera base-plus 공식 checkpoint를 동결했습니다. 원 q8 전체의 정확한 min/max box만 prompt로 썼고, GT box·depth·pose·reference·라벨·query 중심 보정은 넣지 않았습니다. 세 마스크 중 SAM 자체의 예측 quality argmax를 선택하며 동률은 첫 index, postprocessing/compile은 껐습니다. logit 문턱은 0으로 고정했습니다.

모델 state의 실행 전후 SHA가 동일하고 checkpoint 파일 SHA도 유지됐습니다. 독립 NumPy 검산은 42개의 저장된 chosen full-FP32 mask에서 352행의 원 입력·quality 선택 index·원 q 반올림·exact/5×5 관측·U·CSV/JSON/NPZ 일치를 확인했습니다. Root의 별도 집계와 독립 표가 일치했습니다.

독립 검산은 encoder를 다시 추론하거나 저장하지 않은 나머지 두 후보 full mask를 재구성한 것이 아닙니다. 품질 점수는 정답 confidence가 아닙니다. 네 장의 실제 렌더를 Root가 직접 열어 확인한 [기록](V247_ROOT_PUBLIC_TRAIN_SAM/ROOT_ACTUAL_FIXED4_VISUAL_REVIEW.json)도 포함했습니다.

이번 V247에서는 현재 평가셋·SOURCE_VAL·보호된 평가 영역의 영상/GT/모델 점수를 읽지 않았습니다. V246에서 기존 공개 혼합 **INPUT_ONLY** metadata를 파싱하여 사전 지정 TRAIN을 추출했던 접근은 이미 공개한 별도 이력입니다. 과거 모든 held metadata 접근이 0이었다고 주장하지 않습니다. 이번 실행은 그 파생 TRAIN 전용 파일을 사용했습니다.

## 4. 재검산과 파일 구성

이 폴더를 내려받은 뒤 Python 3.10 이상과 NumPy가 있는 환경에서 실행합니다:

```bash
python -B verify_v247_published_sam.py
```

명령은 MANIFEST의 파일 바이트를 확인하고 동결된 독립 checker를 패키지 안의 파일로만 재생합니다. 원래 절대 경로는 출처 identity이며 원본 폴더로 fallback하지 않습니다. SAM checkpoint나 원본 42장 RGB 없이 저장 chosen mask와 전체 query 관측을 재검산합니다. 모델 자체를 재추론하는 명령은 아닙니다.

- `V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/`: 실제 실행 metadata·352행·chosen mask 42개.
- `V247_PUBLIC_TRAIN_SAM_ADAPTER_B/`, `V247_SAM_ADAPTER_CODE_QA_A/`: 실행 코드·사전 프로토콜·독립 CODE QA.
- `V247_SAVED_SAM_QA_C/`: 독립 저장 출력 검산 코드와 실제 결과.
- `V247_FIXED4_SAM_VISUALS_A/`: 네 장·원 crop 32개·렌더 코드·원 입력/출력 binding.
- `DEPENDENCIES/`: 검산에 필요한 파생 TRAIN metadata와 조건부 근거. 기존 혼합 metadata, 원 depth/pose/mesh, 모델 가중치를 새로 싣지 않았습니다.
- `V247_ROOT_PUBLIC_TRAIN_SAM/METADATA_CODE_PREFLIGHT_EVENTS.json`, `ROOT_TRAIN44_RGB_WHITELIST.json`: 모델 실행 전의 metadata 경로/공식 legacy symlink/임시 L=16 검증 가정 오류와 수리 기록. 원 q/L이나 모델을 바꾸지 않았고 실제 SAM 실행은 한 번입니다.

## 5. 남은 문제와 후속 방향

이번 결과는 실사 평가 성능 개선이나 v61 대체 성공을 입증하지 않습니다. 현재 후보를 채택하거나 현재 평가에서 문턱을 고르지 않았습니다. 이전 V245 SOURCE 학습 실패 기록은 [이전 보고서](../V245_SCENE_CF_SOURCE_TRIAL/README_KO.md)에 있습니다.

SAM의 visible 영역 제안은 보조 신호로 사용할 수 있으나, 밖이라는 이유만으로 외부 가림 학습 라벨을 만들 수 없습니다. 후속 학습에서는 native part를 아는 합성 정답과 공개 SOURCE의 조건부 V/E 보조 손실을 분리하고, 다른 대상·경계·불충분한 근거의 UNKNOWN을 유지해야 합니다. 자기 가림은 사용자가 정한 초기 기하 필터의 역할로 둡니다. 다음 학습/후보는 SOURCE에서 규칙과 문턱을 고정한 뒤에만 현재 평가셋에서 v61과 비교할 수 있습니다. PnP는 변경하지 않습니다.

공개 실사 출처: Yanming Wu·Eric Demeester, [KU Leuven Pallox, DOI:10.48804/DJLDFS](https://doi.org/10.48804/DJLDFS), CC BY 4.0. 네 장과 crop의 원 출처/바이트 binding을 보존했습니다. SAM2.1 공식 코드·checkpoint 출처는 [facebookresearch/sam2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4), Apache 2.0입니다. 가중치는 이 패키지에 포함하지 않았습니다.
