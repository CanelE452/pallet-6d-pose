# V1 supersession note

V1 remains a historical screen.  Its S1/S3/S4 conclusions remain usable.
S2 used mismatched bbox semantics.
S5 was incorrectly marked blocked despite an existing real line-inference path.
V1B corrects only S2/S5 and does not rewrite V1.

## 근거 (파일에서 확인, 추측 아님)

### S2 의 semantics 불일치

`scripts/paper/fast6d_screen_v1/run_translation_arms.py:71-73` 은 투영된 **8 개
전부**의 min/max 로 상자를 만든다.

YOLO 가 학습한 상자는 그것이 아니다.
`scripts/self_training_yolo/real_ft_v1/build_real_ft_dataset.py:70-80` —
화면 밖 코너는 `v = 0`, 상자는 **화면 안 코너만**의 min/max.
`data/pallet/results/paper_real_ft_v1/REAL_FT_V1_METHOD_LOCK.json` 의
`label_conversion.box_rule_evidence` 가 이 규칙이 합성 규약과 400/400 일치함을
기록한다(2e-3 이내).  R0 는 그 합성 규약으로 학습됐다.

따라서 V1 의 S2 는 관측 불가능한 상자를 목표로 삼았다.  C1 이 이를 교정한다.

### S5 의 provenance 오판

V1 lock 의 S5 사유는 "line prediction cache 가 real 319 프레임에 없고 canonical
adapter 가 없다" 였다.  cache 가 없다는 것은 맞지만 **adapter 가 없다는 것은
틀렸다**:

- `scripts/stage0/final_train/ft_f0f3_eval.py` 가 real 이미지에 대해
  `preprocess_squash → SplitLate → DH.decode → FU.solve_arms` 를 이미 수행한다.
- `scripts/stage0/multihead/mh_fusion.py` 가 F2/F3/F4 를 구현한다.
- checkpoint 두 seed 가 실재한다
  (`weights/paper_s2/paper_s2_multihead/screen_A1_CORNER_LINE_FINAL40K_seed{1,2}/step_25000.pth`).

cache 가 없다는 것은 **만들면 되는 것**이지 arm 을 닫을 사유가 아니었다.  frozen
inference 로 319 프레임 line 예측을 새로 뽑는 것은 new training = 0 이다.
