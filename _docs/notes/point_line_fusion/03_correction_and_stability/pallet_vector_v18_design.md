# V18 보정 벡터 실험 설계

YOLO 관측 → 고정 점/복원/DHT 부모 → 점 주변3×3 특징 + 해당 점에 연결된 상위3개 선의7×3 특징 → 융합 MLP → 연속2차원 보정 → 기준점에 더하기. 최종 보정 반경은 검출박스 대각선15%로 학습 전에 고정했다. 점 대조군은 새 선 토큰만0이며 부모 문맥에는 이전 선 학습의 영향이 남아 있어 완전히 선 없는 YOLO와는 다르다.

{
  "question": "Does direct direction/magnitude supervision bypass candidate-selection failure?",
  "arms": [
    "point_vector",
    "line_vector",
    "point_risk",
    "line_risk"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "batch": 16,
  "radius_fraction": 0.15,
  "architecture": "Frozen V12 base/amodal context and V11 DHT. 3x3 point patches+context ->64 token; top3 associated finite7x3 line strips ->96 token; shared MLP outputs2D raw vector. delta=.15*box_diagonal*raw/sqrt(1+norm(raw)^2), zero initialization. No pre-enumerated candidate selection, line projection, GT filter, mask or line role supervision.",
  "controls": "Point arms zero optional96line token. Same parameter tensors, initial values, batch plan and2000updates. Local and line arms share parent representations; point control means no NEW line branch, not a line-free lineage. vector loss=mean Euclidean point error; risk loss adds2*mean positive error increase vs immutable base. No predicted gate/threshold or stop-grad trick.",
  "training": "Original253 synthetic images,1462 detected observations across6conditions. All original valid corner IDs; no real training. Refinement parents frozen. Fixed15% radius preregistered; feasibility diagnostics only, no radius search.",
  "evaluation": "Same reused synth505 and real297/9sessions/1251manual corners,8conditions. Original IDs/detections/reflect100 parent wrapper. crop10/30 evaluation-only. Artificial stress; no natural occlusion labels or independent final holdout.",
  "acceptance": "All3seeds clean/crop10/20/30outside <=1.01 base, corner20hidden and crop40outside retain>=80% original amodal improvement. Incremental line effect compared with matched point arm requires all3seeds>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.",
  "optimizer": "AdamW3e-4 wd1e-4 float32 clip5; fixed last checkpoint, no real selection",
  "parent_protocol_sha256": "5bdfc929035057c0168b0a9a1832241ab9f6ebb2e78348e565993d3d9724f55d"
}
