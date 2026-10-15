# 고정 세 사례의 초기 제약·비용: 읽기용 요약

세 ORIGINAL coarse는 두 정책 모두 현재 제약을 통과했습니다. 별도 참조/점수 숫자를 다시 읽지 않았고, 아래 값은 봉인한 RESULT.json만 정리한 것입니다.

- noapril: SOURCE 초기 합 N+O+S=0.806598보다 선택된 SWAP 합 0.137011이 낮습니다. C를 포함해도 초기 0.900330보다 C 최종 0.455271이 낮습니다. 초기 자세의 guard 탈락으로 설명되지 않습니다.
- 031426: native 최종 SWAP의 N+O+S=0.932913, 가중 C=4.85706, C 합=5.78998입니다. C 최종 ORIGINAL은 1.59810+2.50015=4.09824입니다. C 감소 2.35692가 native/기존 O·S 합 증가 0.665185보다 커 두 저장 자세의 순위를 1.69173 차이로 뒤집습니다. 두 자세는 현재 guard를 통과합니다.
- 031426의 현재 활성 PROJECTED_NATIVE_EDGE는 0개입니다. B2 complete q와 B0/B4의 legacy normal/T4만 남아 있습니다. 원 RGB에 선이 없다는 뜻은 아니며, 이 방법의 비용에 신규 선 요인이 채택되지 않았다는 뜻입니다.
- 031344: SOURCE coarse N=27.4586 중 B4/B5 legacy normal이 12.0847+14.0879입니다. 선택 native의 N=5.76958을 현재 관측식이 선호합니다.

ORIGINAL/SWAPPED coarse는 같은 물리 몸체·중심이지만 이름이 붙은 모서리 대응이 다릅니다. 따라서 N과 상태별 guard가 달라질 수 있으며, 이것만으로 물리 자세 prior의 불공정성을 주장하지 않습니다.

## wood_night_01:031344

### ALL_BI_REPLACE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 27.5504 | 0 | 0.105692 | 1.44303 | 27.6561 | 29.0991 | 0.163977 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 155.553 | 0 | 1.69029 | 1.44303 | 157.244 | 158.687 | -0.0760436 |
| SELECTED_NATIVE | ORIGINAL_WD | True | 6.02157 | 0 | 1.84717 | 4.1577 | 7.86874 | 12.0264 | -2.29816e-14 |
| SELECTED_C | ORIGINAL_WD | True | 6.26941 | 0 | 1.83082 | 3.68391 | 8.10023 | 11.7841 | -5.32394e-13 |

### RGB_NONFALLBACK_PRESERVE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 27.4586 | 0 | 0.105692 | 1.44303 | 27.5643 | 29.0074 | 0.163977 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 140.587 | 0 | 1.69029 | 1.44303 | 142.277 | 143.721 | -0.0760436 |
| SELECTED_NATIVE | ORIGINAL_WD | True | 5.76958 | 0 | 1.84909 | 4.23684 | 7.61867 | 11.8555 | 5.57054e-14 |
| SELECTED_C | ORIGINAL_WD | True | 6.02895 | 0 | 1.8323 | 3.73869 | 7.86125 | 11.5999 | 7.5695e-13 |

## wood_night_01:031426

### ALL_BI_REPLACE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 0.228768 | 0 | 0.996881 | 3.69202 | 1.22565 | 4.91767 | 0.124426 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 37.3677 | 2.50551 | 6.22703 | 3.69202 | 46.1002 | 49.7922 | -18.3881 |
| SELECTED_NATIVE | SWAPPED_WD | True | 0.0952791 | 0 | 0.837634 | 4.85706 | 0.932913 | 5.78998 | 0.196048 |
| SELECTED_C | ORIGINAL_WD | True | 0.664359 | 0 | 0.933739 | 2.50015 | 1.5981 | 4.09824 | 0.165132 |

### RGB_NONFALLBACK_PRESERVE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 0.228768 | 0 | 0.996881 | 3.69202 | 1.22565 | 4.91767 | 0.124426 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 37.3677 | 2.50551 | 6.22703 | 3.69202 | 46.1002 | 49.7922 | -18.3881 |
| SELECTED_NATIVE | SWAPPED_WD | True | 0.0952791 | 0 | 0.837634 | 4.85706 | 0.932913 | 5.78998 | 0.196048 |
| SELECTED_C | ORIGINAL_WD | True | 0.664359 | 0 | 0.933739 | 2.50015 | 1.5981 | 4.09824 | 0.165132 |

## eval_noapril:1775201428535961600

### ALL_BI_REPLACE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 0.456187 | 0 | 0.0025646 | 0.0937326 | 0.458752 | 0.552484 | 0.273643 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 568.83 | 0 | 0.809628 | 0.0937326 | 569.639 | 569.733 | -0.82547 |
| SELECTED_NATIVE | ORIGINAL_WD | True | 0.196019 | 0 | 0.00301139 | 0.445318 | 0.199031 | 0.644348 | 0.265103 |
| SELECTED_C | ORIGINAL_WD | True | 0.305576 | 0 | 0.00270089 | 0.127758 | 0.308277 | 0.436035 | 0.269141 |

### RGB_NONFALLBACK_PRESERVE

| 자세 | WD | native guard | N | O | S | 가중 C | N+O+S | N+O+S+C | 최소 원 guard slack |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| COARSE_ORIGINAL_WD | ORIGINAL_WD | True | 0.804033 | 0 | 0.0025646 | 0.0937326 | 0.806598 | 0.90033 | 0.273643 |
| COARSE_SWAPPED_WD | SWAPPED_WD | False | 734.093 | 0 | 0.809628 | 0.0937326 | 734.903 | 734.997 | -0.82547 |
| SELECTED_NATIVE | SWAPPED_WD | True | 0.124116 | 0 | 0.0128955 | 0.348936 | 0.137011 | 0.485947 | 0.194783 |
| SELECTED_C | SWAPPED_WD | True | 0.144694 | 0 | 0.0124939 | 0.298083 | 0.157188 | 0.455271 | 0.19629 |

선택된 최종 비용 12개를 독립 재계산한 최대 차이는 3.747002708109903e-15입니다. 원래 branch 및 실제 선의 finite/directed 지지 slack, 개별 native 요인 비용은 [RESULT.json](RESULT.json)에 보존했습니다.

이 결과는 세 고정 자세의 현재 비용·제약 비교이며, 전역 최적해나 참조 오차의 유일한 원인·모서리 정체를 증명하지 않습니다. 원 PLAN/RESULT/REPORT/SEAL은 변경하지 않았습니다.
