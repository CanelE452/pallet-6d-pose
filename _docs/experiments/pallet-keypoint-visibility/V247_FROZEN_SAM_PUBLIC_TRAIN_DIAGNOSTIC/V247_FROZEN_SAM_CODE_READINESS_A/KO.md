# 기존 frozen SAM 코드 준비성

V180은 원 initializer q8의 min/max box를 사용하는 SAM1 ViT-B SOURCE TRAIN runner입니다. point/mask prompt나 GT box가 없으며 single-mask full-image/low-resolution logits를 저장합니다. 원 U, prompt 불가, runtime 불가를 따로 보존합니다. 단 현재 구현은 V129의512×512 자료와128/4096행 계약으로 고정되어 Pallox480×640·44-frame 입력에 그대로 실행할 수 없습니다.

V68은 SAM2.1 Hiera base-plus의 box-only runner입니다. 세 native mask 중 predicted quality argmax를 고르고, 원 q에서 exact logit과5×5 max를 별도로 읽습니다. 원 box를 계승하지만 이 범위에서 initializer-q8-only라는 근원은 확인되지 않았고 기존 source query는 GT-guided 이력입니다. runner의 고정 코호트 자체가 SOURCE VAL1091targets/8590queries여서 이번 TRAIN에 옛 manifest를 직접 재사용해서는 안 됩니다.

두 코드는 고정 threshold0·frozen official 모델·state/hash/repeat 검사와 별도 실행 entry를 이미 구현합니다. 재사용할 runtime/config/checkpoint/input-seal의 정확한 inert 경로와 SHA는 JSON에 기록했습니다. 실제 자산 무결성·설치 상태나 과거 actual 완료/성능은 이번 네 파일로 재검증하지 않았습니다.

mask0는 선택된 proposal 밖이라는 뜻이며 E/SELF/다른 물체/배경/마스크 실패를 분리하지 않습니다. 원 q8가 teacher instance와 다른 물체를 가리킬 때 mask의 양성·음성 어느 쪽도 native teacher 정답으로 자동 승격할 수 없습니다. V246 fixed4는 사례 기술만 가능하고 빈도·인과·통계 일반화나 모델 선택 근거로 쓰지 않습니다.

새 SOURCE 재사용에는 원 q/L/native/U와 prompt 정의를 보존한 TRAIN-only frame binding 및 별도 shape/cohort adapter·Root entry가 필요합니다. 이번에는 제안된 실행·학습·새 라벨·모델 선택을 하지 않았습니다. 원 이미지/GT/array/weights와 평가 행·결과·점수, runtime/model/Git 실행은0입니다.
