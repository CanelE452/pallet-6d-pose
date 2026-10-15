# coarse prior의 W/D 공정성 점검

동일한 물리 회전과 모델 중심을 두 W/D 좌표계에 넣으면 FULL/TRANSLATION prior의 잔차·값·공통 물리 좌표 미분이 같습니다. 이번 소스/수치 점검에서는 Q 재바인딩 버그나 한 family에만 가산되는 prior를 찾지 못했습니다. 기존 소스·모델·계획·예측·점수는 수정하지 않았습니다.

정의는 R_cf=R_phys Qᵀ, R0_cf=R0_phys Qᵀ, t=C−R_cf mean(X)입니다. 따라서 R_cf R0_cfᵀ=R_phys R0_physᵀ이고 모델 중심 C도 고정됩니다. FULL은 같은 CAL Σ6/μ6를 한 번 사용하고 이동-only는 같은 marginal Σtt/μt를 사용합니다. 서로 다른 rotvec chart의 원시 Jacobian은 다를 수 있어, 같은 카메라 좌회전·중심 이동 증분으로 옮겨 비교했습니다.

W/D swap은 동일 unordered 물리 cuboid를 다른 native 좌표계로 쓰는 변환입니다. 하지만 corner 채널 번호의 물리 위치까지 같은 것은 아닙니다. 같은 물리 자세에서도 ordered native 투영이 순열로 달라지므로, 고정 채널 q와 선 native edge 대응의 영상 비용은 달라질 수 있습니다. prior가 공정하다는 사실은 채널 위상이 올바르거나 두 목적함수가 같은 최적 자세를 고른다는 보장이 아닙니다.

별도 한계: FULL old-P1 prior는 물리 C2(180° yaw) 동치에 최소화한 prior가 아닙니다. 동일 unordered 몸체라도 R_phys 자체를 C2로 바꾸면 plain SO(3) log 비용이 달라집니다. 이는 Q 표현 변경과 다른 coarse 방향 대표값 선호입니다. 이동-only는 같은 모델 중심에서 회전/C2와 무관합니다. 영상 대응을 함께 순열화하는 문제와 단순 Q 변환을 혼동해서는 안 됩니다.

기존 전체 감사는 각 선택 자세에서 original target·Q·잔차·비용을 독립 oldBIAS로 검산했습니다. 이번 보충은 같은 저장 물리 자세를 양 family로 직접 옮긴 대조입니다. 새 GT/점수/영상·추론·최적화는 사용하지 않았으며 자세 정확도나 새 제어 채택을 주장하지 않습니다.
