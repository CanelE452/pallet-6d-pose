# V257 공개 pallet 자료: 보정값·mesh 형식 확인

새 모델을 학습하거나 정확도를 평가하지 않았습니다. 공개 실사에서 point 가림 교사를 만들 수 있는지 확인하기 위해 공식 카메라 보정값과 3D mesh 두 파일을 실제로 내려받아 검사했습니다. RGB·depth·pose·mask·visibility CSV와 현재/보호 평가 사례는 이번 검사에서 읽지 않았습니다.

| 확인한 항목 | 실제 결과 | 이 검사로 알 수 있는 범위 |
| --- | --- | --- |
| 공식 `cam_K.txt` ID356835 | 198B, publisher MD5 일치, 유한한 3×3 행렬 | 수치 형식 정상. 영상·깊이 정합은 아직 미검증 |
| 공식 `pallox.obj` ID357978 | 8,888,206B, publisher MD5 일치 | 원 게시 mesh 바이트와 일치 |
| OBJ 기하 | vertex 27,598개, face 54,744개 | 좌표·면 인덱스 형식 확인 |
| OBJ 축별 좌표 범위 크기 | 1.314317 / 0.828217 / 1.306874 | mesh 좌표값. 물리 단위·축·pose 해석은 미확정 |
| 새 point 가림 정답 | 0개 | native 대응과 실제 투영을 확인한 뒤 별도로 정의해야 함 |

K의 초점 수치는 약 `(605.821, 606.242)`, 주점은 약 `(322.850, 237.404)`입니다. 이는 읽힌 보정값이며, 실제 RGB-D를 투영해 검증한 결과는 아닙니다. OBJ의 material/texture 참조는 따라가지 않았습니다. [실행 범위](assets/ROOT_PROSPECTIVE_SCOPE.json), [원본 다운로드·수치 QA](assets/GEOMETRY_FORMAT_QA.json), [독립 검산](artifacts/V257_PUBLIC_GEOMETRY_QA_A/REVIEW.json)을 보존합니다.

공식 목록의 RGB 322장 중 pose·mask·depth가 같은 stem으로 모두 있는 것은 284개입니다. 파일명 일치는 실제 instance 정합이나 native point 라벨의 정확성을 보장하지 않습니다. SAM mask와 mesh 전체 면적의 visibility score를 곧바로 각 키포인트의 외부 가림 정답으로 바꾸지 않았습니다. [필요 자료와 미확인 항목](artifacts/V257_PUBLIC_PALLET_TEACHER_READINESS_C/KO.md)을 함께 기록했습니다.

Pallox와 MR6D validation은 공식 자료에서 서로 다른 센서·촬영 절차로 설명됩니다. 이는 서로 다른 촬영 계보를 지지하는 근거입니다. 보호된 MR6D 45개와 Pallox 원본 사이의 개별 ID 교차표는 확인하지 못했으며, 정확한 비중복 증명은 미확인으로 남겨두었습니다. 이번 mesh·K 검사로 영상의 학습 사용을 승인하지 않았습니다. [계보 근거와 한계](artifacts/V257_PUBLIC_PALLET_LINEAGE_B/README_KO.md), [상세 receipt](artifacts/V257_PUBLIC_PALLET_LINEAGE_B/LINEAGE_REVIEW.json)에 원 출처를 기록했습니다.

이전 모델 비교는 [V256 보고서](../V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md)에 있습니다. 합성 외부 가림 recall은 52/56에서 55/56으로 올랐지만 precision은 낮아졌고, 공개 YCB 보조 과업의 외부 가림은 두 모델 모두 0/57이었습니다. 현재 pallet 평가에서 90%를 입증한 결과가 아니며 새 모델도 채택하지 않았습니다. 다음에는 공개 RGB-D와 mesh의 투영 및 native point 정의를 맞춰 point 교사의 타당성을 검증해야 합니다. 현재 평가 데이터로 학습·문턱 선택을 하지 않으며 PnP도 변경하지 않습니다.

원 자료는 Yanming Wu·Eric Demeester의 [KU Leuven Pallox 공개 데이터](https://doi.org/10.48804/DJLDFS), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다. 관련 논문은 *6D Pose Estimation of Pallet Bins for Autonomous Logistics Using a Synthetically Trained Object Detector and FoundationPose*, ICPRAM 2026, [DOI:10.5220/0014455600004067](https://doi.org/10.5220/0014455600004067)입니다. 원 K·OBJ는 수정하지 않았습니다. 공식 [README](references/OFFICIAL_README.md)와 [API 목록](references/OFFICIAL_API_RESPONSE.json)을 보존했습니다.
