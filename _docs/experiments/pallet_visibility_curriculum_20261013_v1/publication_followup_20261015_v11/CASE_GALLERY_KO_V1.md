# 양쪽 모두 실제 B/I 처리를 적용한 10장: 전체 그림

같은 사진의 YOLO 원본과 N3+SUBPIX를 각자의 좌표·초기 F 자세로 따로 처리한 저장 결과다. 사후 선택한 gain10 비교이며 전체49/192 성능을 나타내지 않는다. 모든 그림·보고서는 봉인된 원본 그대로다.

청록 원=B 실제 측정, 보라 원=I 실제 측정, 회색 ×=그 팔의 이전 입력이다. 주황 선은 PnP 모델 투영이며 측정점이 아니다. S는 모델 자기 차폐/투영만, O/T/U는 PnP점으로 넣지 않았다. 상태 분류도 정답이 아니다.

GitHub에서도 아래 PNG는 바로 볼 수 있다. HTML은 저장소 clone 뒤 파일 또는 로컬 정적 서버로 열면 된다. GitHub의 HTML 원문 화면은 실행 갤러리가 아니다.

[전체 결과 보고서](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_v2/POSTSCORE_REPORT_KO_V1.md) · [원본 HTML 인덱스](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/index.html) · [점수 봉인](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_v2/SCORE_SEAL.json)

## 1. eval_noapril:1775201414728017664

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case00/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 6.218 / 1.498 | 5.057 / 0.944 | 3.269 / 1.816 | 3.209 / 0.699 |

## 2. eval_noapril:1775201415399297536

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case01/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 9.498 / 1.611 | 4.959 / 1.046 | 2.880 / 1.678 | 2.672 / 0.740 |

## 3. eval_noapril:1775201416944730624

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case02/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 8.126 / 0.841 | 5.239 / 0.183 | 3.420 / 0.344 | 3.569 / 0.340 |

## 4. eval_noapril:1775201418120676608

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case03/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 6.468 / 0.675 | 3.992 / 0.323 | 2.965 / 0.104 | 2.250 / 0.116 |

## 5. eval_noapril:1775201421043401984

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case04/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 6.273 / 0.770 | 0.842 / 0.633 | 4.223 / 1.220 | 0.561 / 0.898 |

## 6. eval_noapril:1775201428535961600

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case05/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 5.660 / 1.081 | 4.406 / 0.284 | 3.449 / 0.675 | 2.705 / 0.641 |

## 7. eval_noapril:1775201430786751744

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case06/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 5.873 / 0.332 | 2.684 / 0.469 | 3.704 / 1.154 | 1.246 / 0.456 |

## 8. eval_pallet07:1778652130452698368

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case07/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 5.818 / 1.287 | 3.622 / 0.955 | 4.514 / 1.507 | 2.613 / 0.956 |

## 9. eval_pallet07:1778652132535256576

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case08/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 5.318 / 1.809 | 3.761 / 1.853 | 3.468 / 1.486 | 1.906 / 1.614 |

## 10. eval_pallet07:1778652138515809024

[개별 HTML](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/index.html) · [원본 사진](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/source_rgb.png) · [저장 PROCESS](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/PROCESS_DATA.json)

| 단계 | YOLO 원본 팔 | N3+SUBPIX 팔 |
|---|---|---|
| ① 입력·자기 초기 F | ![YOLO ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/RAW_YOLO_01_initial.png) | ![N3 ① 입력·자기 초기 F](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/N3_SUBPIX_01_initial.png) |
| ② 각 팔의 새 상태 | ![YOLO ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/RAW_YOLO_02_states.png) | ![N3 ② 각 팔의 새 상태](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/N3_SUBPIX_02_states.png) |
| ③ 실제 B/I 영상 이동 | ![YOLO ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/RAW_YOLO_03_measurements.png) | ![N3 ③ 실제 B/I 영상 이동](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/N3_SUBPIX_03_measurements.png) |
| ④ 같은 B/I-only PnP 결과 | ![YOLO ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/RAW_YOLO_04_final.png) | ![N3 ④ 같은 B/I-only PnP 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/new_gain10_paired_BI_process_v1/case09/N3_SUBPIX_04_final.png) |

| | YOLO 전 | YOLO B/I 후 | N3 전 | N3 B/I 후 |
|---|---:|---:|---:|---:|
| T cm / R ° | 5.362 / 0.355 | 5.362 / 0.355 | 2.189 / 0.455 | 1.100 / 1.134 |

