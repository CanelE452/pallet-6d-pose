# BASELINE_REPEAT — same-seed main control

새 주 가설이 아닌 사전 선택 후 확인이다. 선택: C의 median 부호만 joint이고 차이가 매우 작다. 이미 확인된 matched RAW 및 R0 대비 악화도 보존한다. 이 후속 결과를 보고 recipe/seed/threshold를 다시 고르지 않는다.

PLASTIC, seed43, RAW/REF 각320update, 동일 R0, 기존 real pool/512real+512source, AdamW1e−5, pose+flow-only, last checkpoint만 사용한다. 입력 가림=False; 위치 손실은 기존 그대로다. 가림이 있으면 C의 schedule1 및 나머지 A 파라미터를 그대로 사용하며 teacher나 support를 다시 선택하지 않는다.

학습난수43은 기존 ORDER43(optimizer42+순서별명 변경)과 다르다. 같은 pretrained R0의 augmentation/batch 학습변동 확인일 뿐 독립 pretrained model이나 새 recording TEST가 아니다. BASELINE_REPEAT와 RECIPE_REPEAT는 같은seed/데이터 기본변환을 사용하고 가림만 다르다. RAW/REF 각 쌍은 원래 좌표값만 다르다.

Wood는 기존361real 및45evaluation/38Clean+7Moderate, Severe없음을 유지한다. Plastic에서 고른 가림 규칙을 Wood 결과를 보기 전에 그대로 적용하며 teacher교체·threshold튜닝·평가제거는없다. 주 발견은Plastic99이고Wood는별도적용성이다.

세 가설 cycle은종료했다. 이 단계는예약된4repeatfit+2Woodfit 안에서닫는다. 추가manual0/teacherfit0/추론비용추가0. 모든seed와손익을보고하고모델자동승격하지않는다.
