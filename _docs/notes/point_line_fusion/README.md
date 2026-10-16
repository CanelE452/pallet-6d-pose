# 점·선 융합 실험 색인

> 최신 결과: **V44 · 2026-09-11. 새 24그룹600관측·24회귀모델 두 검증 완료. V41 작은 효과는 새 장면 재현. 새 점+선 평균8.674px지만 점 대조8.767px 대비 추가 이득 불확실·쉬운 잘림 피해 지속. 현재 고정 특징 보정 실험 계열 종료.**

## 먼저 읽을 문서

1. [현재 결론·성능 그림·원인별 근거](STATUS_AND_NEXT_STEPS.md)
   · [V44 새 장면 두 검증과 현재 실험 계열 종료](04_failure_analysis/pallet_fresh_decision_v44.md)
   · [V43 개선 가능성 감독·비중·후보 순위와 피해 분리](04_failure_analysis/pallet_useful_constraint_v43.md)
   · [V42 목표점별 물리 제약 유효성·개선 후보 누락](04_failure_analysis/pallet_constraint_validity_v42.md)
   · [V41 물리 제약 계산·잔차 학습과 후보 방향 분리](04_failure_analysis/pallet_constraint_residual_v41.md)
   · [V40 실제 선분·끝점 대응과 물리 제약 전달 검증](04_failure_analysis/pallet_segment_correspondence_v40.md)
   · [V39 큰 오류 식별과 보정 방향·크기 전달 검증](04_failure_analysis/pallet_hard_direction_v39.md)
   · [V38 점 위치의 선 보정 불확실성·선택 실패 검증](04_failure_analysis/pallet_line_uncertainty_v38.md)
   · [V37 선 후보의 점 보정 방향·크기 효용 검증](04_failure_analysis/pallet_line_utility_v37.md)
   · [V36 연속 선 좌표·후보 품질·인코더 적응 학습](04_failure_analysis/pallet_continuous_lines_v36.md)
   · [V35 실제 경계/점 대응·offset·Hough 격자와 후보 정밀도](04_failure_analysis/pallet_role_offset_v35.md)
   · [V34 반복 재현·선 geometry/영상 특징 분리 검증](04_failure_analysis/pallet_line_validation_v34.md)
   · [V33 점별 선 구간/끝점 대응·제거 재학습·실제 attention](04_failure_analysis/pallet_correspondence_v33.md)
   · [V32 실제 가시 경계/끝점 감독·융합·보조 감독 대조](04_failure_analysis/pallet_visible_geometry_v32.md)
   · [V31 점 선택/선 추가 이득 분리와 새 원본 대조](04_failure_analysis/pallet_separated_gain_v31.md)
   · [V30 상태 없는 방향별 적응형 점·선 학습](04_failure_analysis/pallet_directional_adaptive_v30.md)
   · [V29 적응형 점·선 전문가 학습과 분석](04_failure_analysis/pallet_adaptive_experts_v29.md)
   · [V28 선만 적용한 추론](04_failure_analysis/pallet_line_only_v28.md)
   · [V27 새 구조 학습·검증](04_failure_analysis/pallet_direction_utility_v27.md) · [점/Hough 실제 사례](04_failure_analysis/pallet_direction_utility_v27_cases.md)
   · [V26 카메라·신뢰도 진단](04_failure_analysis/pallet_camera_trust_v26.md)
   · [V25 실행 결과·실제 이미지](04_failure_analysis/pallet_observation_v25.md)
2. [다음 실험의 대조군·순서·판정 기준](NEXT_EXPERIMENTS.md)
3. 아래 폴더에서 해당 실험의 설계와 결과 원문 확인

## 단계별 폴더

| 단계 | 문서 수 | 내용 |
|---|---:|---|
| [초기 Hough·기하·GT 감사](01_hough_and_geometry/README.md) | 11 | 초기 점·선 결합, joint/coupling, 후보·배치 및 수동 GT 감사. 초기319장 계열과 최근297장 평가를 섞지 않는다. |
| [국소 특징·점 번호·관측 상태](02_local_and_identity/README.md) | 10 | V4~V12. 후보 역할·번호, 점 주변 선 안내, 학습한 attention, 국소 점 보정과 amodal 복원. |
| [보정량·후보 선택·안정성](03_correction_and_stability/README.md) | 14 | V13~V19. 선 근거·RGB·가시 실루엣, 직접 보정 벡터, 보정 적용률. 큰 복원 이득과 추가 선의 효과를 분리한다. |
| [실패 원인·학습·일반화](04_failure_analysis/README.md) | 31 | V20~V44. 손실·문맥·PCGrad, 신규 합성 관측, 적응형 보정·물리 경계 감독·점별 선/끝점 대응. |

## 연구 흐름과 결론

| 단계 | 주된 질문 | 확인한 것 |
|---|---|
| 초기 joint/coupling | YOLO와 Hough를 학습 경로로 연결할 수 있는가? | 구현·경사 경로는 성립. 실제 종합 향상과 구현 유효성은 별개. |
| V4~V9 | 후보·점 번호·점 근처 선 정보를 더 잘 쓰는가? | 부분 반응은 있으나 기존 점 정확도를 보존하는 안정적인 개선은 미확인. |
| V10~V12 | 국소 보정·보정 억제·별도 복원이 유리한가? | 국소 점 보정 및 어려운 점 복원의 이득. 약한 잘림의 반대 방향 복원 문제가 나타남. |
| V13~V17 | 후보 선택에 선 영상 근거·세부 RGB·가시 마스크가 도움 되는가? | V16 큰 잘림의 조건부 선 이득. V17 RGB/실루엣만으로 안정화되지 않음. |
| V18~V19 | 보정 벡터와 적용률을 직접 학습하면 되는가? | 큰 복원 이득은 직접 보정 구조에서도 발생. 틀린 방향과 보정 보존 판단이 남음. |
| V20~V22 | 손실·비중·문맥·PCGrad·포화가 원인인가? | 실제 개입·부분 효과가 있어도 전체 안정성은 미달. |
| V23~V24 | 실제 정확한 합성 관측이 부족한가, 학습 가능한가? | 정확한 사례 확보, 작은 집합 학습 성공. 약한 잘림 방향 판단의 실사 이전 실패. |
| V25 | 관측을 짝지어 늘리고 방향까지 바꾸면 해결되는가? | 반대쪽 잘림으로도 합성의 필요한 방향 분포는 거의 동일. 어려운 조건 개선·약한 잘림 악화 유지. 선+픽셀의 작은 추가 신호는 있으나 전체 안정화 실패. |

| V26 | 정확도와 신뢰도 선택만으로 충분한가? | 실제 카메라 통제·역할 대응·수직/접선 오차·후보 oracle 진단. 선택 여지는 있으나 선 정확도만으로 보정 유용성을 보장하지 못함. |

| V27 | 방향별 효용·이동량·역할을 학습하면 안정화되는가? | 12모델 실제 학습 완료. 후보 oracle 여지는 있으나 점 후보 우세와 실사 약한 잘림 악화 유지, 시드별 선 효과 불안정. |

| V28 | 사례3에서 선만 사용하면? | 기존 점 보정 후보를 제거하면78.80→12.08px. 전체 가림/큰 잘림 이득은 잃음. 점 힌트 없는 DHT 선 생성도 별도 추론, 새 학습 없음. |

## 문서와 원본 산출물의 관계

V30은 사용자의 핵심인 점별 adaptive 보정을 직접 검증했다. 선 법선 방향만 보정하며 접선은 보존하고 기존 강한 점 부모를 선택 가능한 경로로 유지했다. 점 대조 대비 일부 조건의 추가 이득은 있지만, 선의 이득을 과신하고 잘못된 점 부모를 고르는 문제가 남았다.

V29는 세 전문가를 따로 학습한 후 점별 선택을 비교했다. 사례3은 독립 선6.18px로 개선됐으나 큰 잘림에서 선만 쓰는 고정 배정은 악화됐다. 합성/실사 전문가 순위 차이와 학습집합부터 부족한 선택 능력이 남았다. 상세 표와 실제 이미지는 V29 노트를 참조한다.

- `*_design.md`는 당시 설계, `*_results.md`는 상세 결과다. 초기 단일 노트에는 둘이 함께 있다. 이름의 “최종”은 그 단계의 후보 명칭이며 현재 채택 모델을 뜻하지 않는다.
- 이 폴더는 읽기 위한 정리본이다. 원본 실험 수치·체크포인트·HTML은 `data/pallet/results/각_실험/`, 구현은 `scripts/research/각_실험/`에 있다.
- 이번 문서 이동에서 실험 소스·모델·원본 산출물을 수정하지 않았다. 최신 요약은 현재 상태를 설명하고, 이전 문서의 과거 상태를 소급해서 성공/실패로 바꾸지 않는다.
- 이전 평면 경로는 [이동표](../FILE_MAP.json)에서 찾을 수 있다. 파일명은 유지했다.

[노트 전체 목차](../README.md)
