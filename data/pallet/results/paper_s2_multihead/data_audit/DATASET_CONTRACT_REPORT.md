# DATASET CONTRACT REPORT

생성 0 / 학습 0. 라벨의 실제 계약과 실험 기록만으로 역할을 정했다.
디렉터리 이름에서 역할을 추론하지 않았다.

## 계약 표

```
dataset                             N    gate  corner  line  evidence          role
------------------------------------------------------------------------------------------------------------
BROAD_40K                       40000    100%    True  True  SUPPORTED         MAIN_TRAIN
CORNER_LA_Y15_30                 2500    100%    True  True  NOT_ESTABLISHED   ABLATION_ONLY
CORNER_LA_Y30_PLUS               2500    100%    True  True  NOT_ESTABLISHED   ABLATION_ONLY
CORNER_LA_FRONTAL                   0    nan%    None  None  NOT_TESTED        PRESERVE_UNUSED
EDGE_HARD_TRUNC_TRAIN           10000      0%   False  True  NOT_TESTED        MAIN_TRAIN
EDGE_HARD_TRUNC_DEV              1000      0%   False  True  NOT_TESTED        EVAL_ONLY
EDGE_HARD_TRUNC_UNTOUCHED        1000      0%   False  True  NOT_TESTED        EVAL_ONLY
EDGE_HARD_CLEAN_UNTOUCHED        1000    100%    True  True  NOT_TESTED        EVAL_ONLY
NEGATIVE_SYNTH_V1_TRAIN          9000    nan%    None  None  REJECTED          CALIBRATION_ONLY
NEGATIVE_SYNTH_V1_DEV            1000    nan%    None  None  SUPPORTED         EVAL_ONLY
LEGACY_mixed_v8_train            9000    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_v4_split_base             4000    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_paper_4pallet_mask_v1    10000    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_aug_squash_v2             2212    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_aug_trunc_v2              2971    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_aug_scale_v2              1125    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_val                       1500    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_achieve_all              81233    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_pl_achieve               24484    nan%     nan   nan  NOT_TESTED        ARCHIVE
LEGACY_paper_s2_pl_family        1682    nan%     nan   nan  NOT_TESTED        ARCHIVE
QUARANTINE_win_search2k          2429    nan%     nan   nan  NOT_TESTED        ARCHIVE
BROKEN_addon_v1_train_val           0    nan%     nan   nan  NOT_TESTED        ARCHIVE
```

## 이름이 아니라 계약으로 갈린 것

### EDGE_HARD 는 두 계약이 섞여 있다

```
trunc_train / trunc_dev / trunc_untouched   gate 0%   V_vis<=3 100%   corner 감독 불가
clean_untouched                            gate 100%  V_vis>=4      corner 평가 가능
```

같은 `edge_complement_v1` 접두어를 쓰지만 CLEAN 쪽만 point-valid 다. 이름으로 묶으면 틀린다.

### BROAD 에는 V_vis<=3 프레임이 한 장도 없다

G1 게이트가 `V_vis >= 4` 를 요구하므로 설계상 0 이다. EDGE 가 채우는 영역은 BROAD 의 희소 영역이 아니라 **BROAD 가 정의상 배제한 영역**이다.

### 세 카운트는 서로 다르다 (실측)

```
n_inframe      투영이 화면 안 (계산)          최대 8
V_actual       화면 안 + 자기폐색 아님 (라벨)  최대 7  <- 볼록 육면체는 항상 >=1개가 뒤에 가려진다
V_vis_actual   추가로 외부 폐색 아님 (라벨)
n_supervised   corner loss 가 실제로 학습하는 채널 = 9채널의 화면 안 판정
```

`V` 를 visible count 로 읽으면 안 된다는 지시가 데이터로 확인됐다.
