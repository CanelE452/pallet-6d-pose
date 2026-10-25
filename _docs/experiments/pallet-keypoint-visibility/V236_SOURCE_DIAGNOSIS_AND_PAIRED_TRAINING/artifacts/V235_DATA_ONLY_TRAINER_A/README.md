# V235 data-only fixed-encoder SOURCE trainer

```sh
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B train_data_only_counterfactual.py --entry <ROOT_TRUE_ENTRY.json>
```

Root protocol61a354e4의 data-only1000 대조입니다. 새 모델/문턱/좌표/PnP 선택을 추가하지 않습니다. CONTROL_LOCAL_CROP은 V233 완료 cp/log/CSV를 byte 그대로 복사하고 재학습하지 않습니다. TREAT_CF_DATA_ONLY만 동일한 initial_head_state와 원 PCG17 draw1000, 같은76098 head·frozen officialDINO·feature/loss/AdamW로 학습합니다. LoRA는 없습니다.

원 draw는 family마다8D8E입니다. 첫4D는 원본, 뒤4D는 같은 부모 성공SHAM(실패면 원본); 첫4E는 원본, 뒤4E는 별도 PCG235가 같은family 성공 FIT-D pool에서 뽑은 COVER입니다. Parent 배열 순서는 저장된 매핑을 보존하며 모든 FIT-D 부모가 정확히 한 번 있어야 합니다. 원 label과 family를 synthetic pair 입력 때문에 섞지 않습니다.

Original prefix feature는 V233 features_for 그대로 FIT64행씩만 읽습니다. CF는 성공한 FIT 부모만 32개 augmented image씩 읽고 full32+centre32를 원 preprocess_views AST에 통과시킵니다. 동일 crop_features를 적용하며 원 valid/fractions/central4/condition을 유지합니다. Cache는 메모리의 compact success×2×1186(COVER0/SHAM1)이고, 실패 RGB나 DEV prefix를 freeze 전에 숫자로 읽지 않습니다.

두 teacher referent는 output0 SYN witness0 MAIN/output1 YCB associated-r AUX입니다. Classifier는1186→64ReLU→2, 출력0 초기화, 원 CONTROL 초기 state를 strict reuse합니다. 각family BCE16의 평균을 다시 .5로 평균합니다. head AdamWlr.001/wd.0001,1000업데이트/FP32 deterministic/mathSDPA/TF32off 고정입니다.

Newhead final1000과 freeze를 저장한 뒤에만 원본 all-row DEV forward를 합니다. Original CONTROL을 같은 features_for64-row schedule로 재생하여 **양 logit 전체 FP32 bytes가 oldCSV와 일치해야** 결과를 완료합니다. 다르면 실패 receipt와 finalcheckpoint를 보존하고 중단합니다. 허용오차 완화·CONTROL 재학습·자동 retry는 없습니다. 최종 CSV/CM/P/R/U는 원본 SOURCE 행만입니다.

ENTRY_FALSE_TEMPLATE은 코드·완료metadata의 literal binding에서만 조립했습니다. Root가 newroot_scope, pairQA PASS 및 원 half-up reference 동등성, 공식model load와 actualfit을 별도 승인해야 합니다. 원 rawRGB/GT/depth/oldmixed/보호자료 및 ref/ROI deployment channel은 없습니다. 실제 source body/stat/header/hash/load와 DINO instantiate/forward는 이번 준비에서0입니다.

Toy는 invented metadata/randomfeature로 actualrouting helper와 원 pinned loss를 호출하고 head optimizer1step만 검사했습니다. 원 PCG17 draw 불변, 실패SHAM fallback, PCG235 COVER, balancedquota, compactfeaturegather, false-entry/경로 guard가 PASS입니다. 최초 toy fixture가 실패 부모를 family0에만 두고 양family fallback을 요구한 오류는 원본/실패를 보존하고 fixture만 고쳤습니다. 실제 데이터·모델·학습은0입니다.
