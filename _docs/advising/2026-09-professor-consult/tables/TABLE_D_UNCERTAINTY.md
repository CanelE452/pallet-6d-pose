# Table D — is the ranking of the arms stable?

population   319 frames in 13 recording groups
source       POSE_PAIRED_BOOTSTRAP.json, 10,000 resamples, seed 20260903

```text
contrast                      metric               delta              session CI95  excl 0
------------------------------------------------------------------------------------------
R5 제안 - R0                  IoU3D median       -0.0164        [-0.0436, +0.0220]   False
R5 제안 - R0                  ADDsym AUC         -0.0284        [-0.0624, +0.0039]   False
R1 필터없음 - R0              IoU3D median       -0.0135        [-0.0371, +0.0250]   False
R1 필터없음 - R0              ADDsym AUC         -0.0080        [-0.0267, +0.0141]   False
R2 신뢰도 - R0                IoU3D median       -0.0037        [-0.0411, +0.0392]   False
R2 신뢰도 - R0                ADDsym AUC         -0.0127        [-0.0465, +0.0235]   False
R3 +재투영 - R0               IoU3D median       -0.0034        [-0.0247, +0.0302]   False
R3 +재투영 - R0               ADDsym AUC         -0.0136        [-0.0271, +0.0107]   False
R4 +코너제거 - R0             IoU3D median       -0.0035        [-0.0381, +0.0366]   False
R4 +코너제거 - R0             ADDsym AUC         -0.0164        [-0.0441, +0.0143]   False
R0-CONT 추가학습만 - R0       IoU3D median       -0.0091        [-0.0508, +0.0215]   False
R0-CONT 추가학습만 - R0       ADDsym AUC         -0.0201        [-0.0480, +0.0044]   False
```

How to read it in the meeting

- Not one session-cluster interval excludes zero, in either direction.
- Every point estimate is negative, but the spread across sessions is larger
  than the spread across arms, so the ordering of the arms is not stable.
- The honest sentence is 'this data cannot separate them', not 'they differ'.

Paper role: **main table companion** (uncertainty for Table A).
