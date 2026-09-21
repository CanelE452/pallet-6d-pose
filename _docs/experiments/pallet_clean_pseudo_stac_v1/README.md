# Clean real pseudo labels → occlusion adaptation: 결과 묶음

## 결론부터

**선택한 깨끗한 CAD 8장에서는 기존 모델의 보정으로 코너 오차를 줄일 수 있었다. 하지만 그 수도레이블로 학습한 학생이 별도 occlusion 데이터에서 R0를 일관되게 넘지는 못했다.**

이 문서는 두 관찰을 구분한다. 'clean 데이터에서 잘한다'는 표현은 **사용자가 선택한 기존 DEV 8장의 관찰 결과**이며, 모든 깨끗한 실사·다른 세션에 일반화됨을 증명한 주장이 아니다. 이8장은 이후 학습으로 사용했으므로, 현재 학생의 평가/독립 테스트로 다시 사용하지 않는다. CAD 세션 전체를 이번 학생 평가에서 제외했다. 기존 전역 split·과거 보고서는 변경하지 않고 실험별 manifest로 역할 변경을 기록했다.

## 1. 깨끗한 8장: frozen R0/N3/Replay와 자기 가림 PnP

원래18장 갤러리의 **2·6·7·8·9·10·11·12번**이다. 기존 조건 태그는 외부 occlusion 없음 / truncation 없음이다. **Self-occlusion은 여전히 존재**할 수 있으므로 '8개 코너 모두 영상에서 보인다'는 뜻은 아니다.

전체64개 평가 코너 기준이며 중심점은 제외했다.

| 방법 | 중앙값 px ↓ | P90 px ↓ | PCK10 ↑ | PCK20 ↑ |
|---|---:|---:|---:|---:|
| R0 | 11.55 | 19.20 | 37.5% | 90.6% |
| R0 + 자기 가림 PnP | 10.31 | 18.25 | 46.9% | 93.8% |
| N3 DIM-SYM seed1 | 8.88 | 13.93 | 65.6% | 100.0% |
| N3 + 자기 가림 PnP | 8.65 | 13.04 | 68.8% | 100.0% |
| Replay PoseFix raw | 8.37 | 13.01 | 73.4% | 96.9% |
| Replay + 자기 가림 PnP | 7.68 | 12.62 | 76.6% | 96.9% |

직육면체 치수와 **예측 자세**로 자기 가림을 근사 판정한다. 가림으로 판정한 점은 제외하고, 화면 내부의 고신뢰·보이는 코너6개 이상으로 SQPnP+LM을 다시 풀어 **가림 점만** 재투영 좌표로 대체했다. 판정이 불안정하거나 점이 부족하면 원래 좌표를 유지했다. GT로 가림/교체 여부를 정하지 않았고, 보이는 점/중심/검출 출력은 유지했다. 직육면체는 팔레트 구멍과 틈을 표현하지 못한다. 기존 숨은 코너 정답 일부가 PnP 유래일 수 있어 독립 물리 정확도 검증은 아니다.

원래CAD18장 전체의 평균적인 순위와 선택8장의 순위는 같지 않다. 원래18장에서는 N3가 큰 오차를 더 안정적으로 줄였다. 기존 R0 flip+LOO → 보정 후 all8 LOO 필터는 **CAD18장 전부 통과**시켰다. 따라서 필터 통과를 수도레이블 정답 보장으로 해석하지 않는다.

- [선택8장 전후 이미지: 다운로드 후 브라우저에서 열기](clean8_portable.html) — RGB 미리보기 내장, 별도 데이터 폴더 불필요.
- [정확한 선택8장 수치](../../../outputs/pallet_cad_refiner_comparison_v1/self_occlusion/SELECTED8_METRICS.json)
- [자기 가림 전체18장 결과](../../../outputs/pallet_cad_refiner_comparison_v1/self_occlusion/METRICS.json)
- [필터 판정](../../../outputs/pallet_cad_refiner_comparison_v1/filter_audit/DECISIONS.json)

## 2. 이 수도레이블로 full-model fine-tuning

CAD8의 고정 N3+PnP / Replay+PnP 수도레이블로 각각 R0에서 출발했다. 실사GT 좌표를 학습 타깃으로 넣지 않았다. 동일 합성512장 + 실사512슬롯(각8장64회),5epochs,320updates,seed42,AdamW1e-4. 최종last.pt만 사용했다. 평가에는 학생만 사용하고 보정기/PnP/필터를 붙이지 않았다.

주 평가: CAD를 전부 제외하고, 기존 occlusion 태그가 있으며 Replay teacher의 실사 학습 세션과도 겹치지 않는 **6세션96장·737코너**. 보조 결과는 occlusion107장 / 전체 non-CAD176장 / clean69장이다. occlusion 여부는 기존 workspace condition manifest에서 가져왔으며 좌표 annotation의 unknown 필드를 임의로 변경하지 않았다.

| 방법 | 매칭 코너 중앙값 px ↓ | 전체 분모 PCK10 ↑ | 전체 분모 PCK20 ↑ | 매칭 |
|---|---:|---:|---:|---:|
| R0 | 11.39 | 41.79% | 62.82% | 88/96 |
| N3+PnP 수도레이블 학습 | 12.36 | 35.69% | 58.21% | 82/96 |
| Replay+PnP 수도레이블 학습 | 12.20 | 34.19% | 59.16% | 83/96 |

**판정: 악화, 기존 모델 교체 안 함.** 검출 매칭이 줄었으며 동일 공통 매칭 이미지에서도 세밀한 코너 정확도는 개선되지 않았다. 큰 오차 P90이 일부 낮아진 것을 전체 개선으로 해석하지 않았다. [상세 보고](../pallet_cad8_selftrain_v1/HANDOFF_KO.md)

## 3. STAC에서 가져온 발상

Sohn et al., **A Simple Semi-Supervised Learning Framework for Object Detection (STAC)**는 신뢰할 만한 수도레이블과 강한 augmentation 아래의 일관성 학습을 결합한다. [원 논문, arXiv:2005.04757](https://arxiv.org/abs/2005.04757), [저자 공개 구현](https://github.com/google-research/ssl_detection).

우리 적용: 깨끗한 실사에서 만든 좌표를 고정한 뒤 **학생 입력 RGB에만 인공 가림**을 주고 원래 좌표를 예측하도록 학습했다. 가린 점도 원래 trusted 좌표의 supervision을 유지했다.

**정식 STAC 재현/벤치마크가 아니다.** 논문 전체 구조·데이터·augmentation·학습 프로토콜을 복제하지 않았다. YOLO pose 모델, 보정된 코너 수도레이블, 작은 CAD8 집합, 포즈 전용 동결 학습, 합성 replay와 단순 textured-color 가림 패치를 사용한 **STAC-inspired bounded adaptation**이다. 이번 결과를 STAC 원 논문의 성능/실패로 일반화하지 않는다. 논문 PDF/타사 코드 자체를 이 묶음에 복사하지 않았다.

## 4. Pose-only CLEAN vs OCCLUDED 통제 실험

- 동일한 **Replay+자기 가림 PnP** 수도레이블과 원래 학습량/학습률을 두 arm에 고정.
- Backbone/neck/검출부 및 모든 BN 통계 동결. 포즈 branch/flow와 해당 affine parameter만 학습.
- 실사 입력 노출의75% 확률로 코너/상단 변 주변에1–2개 textured-color 사각 패치를 추가. 패치별 너비/높이는 bbox의18–35%; trusted 코너4개 이상은 가리지 않음. 합성 replay에는 이 추가 패치를 넣지 않음.
- 적용 위치는 기본 기하 augmentation 이후. 좌표와 supervision mask를 변경하지 않음. 가려진 trusted 점을ignore로 바꾸지 않음.
- 별도numpy RNG902106로 동일 계획 생성. CLEAN은 계획만 기록하고 칠하지 않음. native augmentation RNG와 분리.
- 실제2560회 실사 노출 중1930회에 적용. 두 arm의320개 입력 순서/타깃/패치 계획과 매epoch 가림 전 RGB 해시 일치 검증.

같은 주 평가96장 결과:

| 방법 | 중앙값 px ↓ | P90 px ↓ | PCK10 ↑ | PCK20 ↑ | 매칭 |
|---|---:|---:|---:|---:|---:|
| R0 | 11.39 | 64.14 | 41.79% | 62.82% | 88/96 |
| 포즈만 학습 + CLEAN | 11.69 | 64.19 | 40.43% | 62.96% | 88/96 |
| 포즈만 학습 + OCCLUDED | 11.52 | 64.24 | 40.98% | 63.64% | 88/96 |

**인공 가림은 CLEAN 대조군보다 소폭 도움. 하지만 R0 대비 일관된 개선은 아님.** PCK20은 R0보다6/737코너 순증(+0.81 percentage points; 표시 반올림 차이에 유의), PCK10은6코너 순감. 검출 box/score는 R0와 정확히 동일했다. 단일seed로 통계적 유의성/강한 일반화 주장을 하지 않는다. 기존 최종모델은 유지했다.

- [학습 가림 예시·전체 결과표: 독립 실행 HTML](occlusion_training_portable.html)
- [세부 결과](../pallet_cad8_occlusion_v1/RESULTS_KO.md)
- [고정 파라미터·입력 일치 감사](../pallet_cad8_occlusion_v1/AUDIT.json)
- [최종 판단](../pallet_cad8_occlusion_v1/DECISION_KO.md)

## 파일·재현 범위

코드,프로토콜,checkpoint SHA,수치,프레임별 예측/오차,학습 trace,loss CSV,HTML을 함께 보존한다. 무관한 다른 연구의 미추적 변경은 이 push에 포함하지 않는다. 원본 RGB/합성 데이터와 `.pt` 가중치는 기존 저장소 정책대로 Git에 넣지 않고 로컬에 보존한다. Portable HTML의 RGB는 검토용 JPEG 미리보기이며 학습 데이터의 대체물이 아니다.

원래 `outputs/`의 HTML은 로컬 RGB 또는 생성 PNG를 참조하므로 데이터가 없는 새 checkout에서 바로 이미지가 뜨지 않을 수 있다. 위 **portable HTML 두 개는 이미지가 내장**돼 별도 원본 없이 볼 수 있다. GitHub 소스 화면에서는 실행되지 않으므로 다운로드 후 브라우저로 연다.

실험 실행은 저장소 루트에서 동일한 의존성/로컬 데이터와 R0 checkpoint를 갖춘 환경에서 한다. 고정 artifact는 덮어쓰기를 거부하므로 이미 완료된 run을 재실행해 대체하지 않는다. 상세 경로·해시는 각 PROTOCOL/FIT에 있다.

```bash
python scripts/research/pallet_cad_refiner_comparison_v1/self_occlusion.py
python scripts/research/pallet_cad_refiner_comparison_v1/check_self_occlusion.py
python scripts/research/pallet_cad8_selftrain_v1/run.py prepare
python scripts/research/pallet_cad8_selftrain_v1/run.py all
python scripts/research/pallet_cad8_occlusion_v1/run.py prepare
python scripts/research/pallet_cad8_occlusion_v1/run.py all
python scripts/research/pallet_cad8_occlusion_v1/report.py
python scripts/research/pallet_clean_pseudo_stac_v1/export_review.py
```

참고: Optional Albumentations의ImageCompression API 불일치가 있어 native Ultralytics augmentation으로 학습했다. 패키지를 바꾸지 않았으며, 두 pose-only arm의 실제 native 입력 trace 일치로 비교 조건을 확인했다. 이 연구는 여러 번 열람한 DEV에 대한 탐색이다. 새로운 독립 confirmation set에 대한 논문 결론으로 승격하지 않는다.
