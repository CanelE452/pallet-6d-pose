# ARCHITECTURE DECISION — PAPER_S2 ep57 mechanism diagnosis

[관찰]
strict N87 에서 failure class 분포는 F2_CONFIDENT_WRONG 35, F1_NO_RESPONSE 24, F5_MIXED 23, F3_GEOMETRY_AMPLIFIED 4, F4_SOLVER_SPECIFIC 1 이고, first-break stage 는 REPRESENTATION_LOCALIZATION 33, IMAGE_OR_RESPONSE 24, POSE_REFINEMENT 14, DECODER 10, PNP_GEOMETRY 3, MIXED_OR_UNRESOLVED 3 이다.

[원인 후보]
- B_far_depth_structured_refinement → F2_CONFIDENT_WRONG
- A_visibility_partial_keypoint → F1_NO_RESPONSE
- C_calibrated_uncertainty_robust_pnp → F3_GEOMETRY_AMPLIFIED
- D_direct_pose_gated_residual → F4_SOLVER_SPECIFIC

[지지 증거]
- [확인] GT 2D 치환으로 pose 가 회복되므로 손실은 solver 하류가 아니라 2D correspondence 상류다.
- [확인] B_far_depth_structured_refinement: 지지 프레임 35 (40%), oracle 회복 17 프레임.
- [확인] A_visibility_partial_keypoint: 지지 프레임 24 (28%), oracle 회복 17 프레임.
- [확인] C_calibrated_uncertainty_robust_pnp: 지지 프레임 4 (5%), oracle 회복 0 프레임.
- [확인] D_direct_pose_gated_residual: 지지 프레임 1 (1%), oracle 회복 0 프레임.
- [확인] far group GT 치환(O3_far) 16 프레임, depth-right(O7, kp1/2/5/6) 17 프레임 회복 vs near group(O2) 10, depth-left(O6, kp0/3/4/7) 9 — far/depth 축이 near 축보다 강한 레버.
- [확인] keypoint convention 은 `annotate_pnp.make_pallet_keypoints_3d` 좌표에서 직접 확인했다: LEFT={0,3,4,7}, RIGHT={1,2,5,6}, near={0,1,2,3}, far={4,5,6,7}.

[반증 증거]
- [확인] solver / K / dimensions / centroid 단독 원인은 GT 2D 조건에서 전부 정상이므로 기각.
- [확인] 'F1 은 화면 밖 keypoint 때문'이라는 해석 기각 — 미검출 corner 125개가 GT 화면 **안**이고, 15/24 프레임은 미검출이 전부 화면 안이다.  Candidate A 를 outside-supervision 문제로만 좁히면 F1 의 대부분을 놓친다.
- [확인] centroid 단독(O9_centroid)은 실패 프레임을 0개 회복한다 → centroid 보정 가설 기각.
- [확인] 'late-stage refinement 가 far face 를 망친다'는 가설 기각 (EARLY_WRONG >> LATE_DRIFT).  Candidate B 의 근거는 late drift 가 아니라 '초기 오류를 refinement 가 못 고치고 신뢰도만 올린다' 이다.
- [추정] depth-left/right 비대칭(회복 9 vs 17)은 소표본(실패 17)에서의 관찰이며, 좌우 어느 쪽이 원인인지 단정하지 않는다.
- [확인] decoder 교체 단독 효과는 판정 기준을 넘지 못한다 → decoder 는 보조 원인.

[현재 판정]
- [확인] 최다 실패 class 는 F2_CONFIDENT_WRONG 이며 대응 후보는 B_far_depth_structured_refinement 이다.
- [추정] 단일 평균 성능이 아니라 failure class 별로 architecture 를 고른다는 전제 아래, 아래 우선순위는 micro-training 으로만 확정된다.

[architecture 우선순위]
1. B_far_depth_structured_refinement (F2_CONFIDENT_WRONG)
2. A_visibility_partial_keypoint (F1_NO_RESPONSE)
3. C_calibrated_uncertainty_robust_pnp (F3_GEOMETRY_AMPLIFIED)
4. D_direct_pose_gated_residual (F4_SOLVER_SPECIFIC)

[다음 admissible experiment]
- MICRO_TRAIN_PLAN.md 의 후보 3개를 1-seed, 200~500 frame, 5~10 epoch 으로만 실행한다. final-test 는 열지 않는다.
