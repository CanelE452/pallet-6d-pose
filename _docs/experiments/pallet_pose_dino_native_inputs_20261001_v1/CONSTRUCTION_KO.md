# 원본 이미지 영역으로 제한한 DINO TRAIN 입력

기존 token 재풀링은 완료했습니다. 독립 검산은 아직 진행하지 않았습니다. 새 학습·추론·T/R 성능 평가는 0회이며, 학습 준비가 검증되었다는 판정이 아닙니다.

| model | valid candidates | previous support points | native support points | removed padding points | valid zero-support candidates |
|---|---:|---:|---:|---:|---:|
| R0 | 5194 | 41282 | 37664 | 3618 | 0 |
| DIVERSE251_s1 | 5194 | 41512 | 38122 | 3390 | 0 |
| DIVERSE251_s2 | 5194 | 41505 | 38087 | 3418 | 0 |
| DIVERSE251_s3 | 5194 | 41513 | 38118 | 3395 | 0 |

기존 TRAIN 2,598행·원래 실패 1행·전체 유효 후보 20,776개를 보존했습니다. 이전 crop과 DINO token은 그대로입니다. 지원 조건만 기존 메타데이터의 pad에 따른 원본 이미지 영역으로 제한하고 R0 유효 후보 5,194개에서 FP32 mean/std를 다시 계산했습니다. 지원점이 0개라도 원래 후보를 무효화하지 않았습니다.

token 계산에는 반사 패딩을 포함한 이전 crop을 사용했으므로 내부 위치를 표본화해도 문맥을 통한 패딩 영향은 남을 수 있습니다.

[영수증](TRAIN_APPEARANCE_INPUTS.json) · [프로토콜](INPUT_PROTOCOL.json) · [설계](DESIGN_KO.md)
