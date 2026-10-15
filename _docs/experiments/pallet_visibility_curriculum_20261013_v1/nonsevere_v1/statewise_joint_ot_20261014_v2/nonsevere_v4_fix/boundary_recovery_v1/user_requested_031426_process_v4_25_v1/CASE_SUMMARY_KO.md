# 031426 사례 보충 요약

이전 176.5351cm → 이동-native 165.9262cm로 10.6089cm 줄었지만 약1.66m의 큰 실패가 남습니다. 세 새 방법과 이전24는 모두 ORIGINAL_WD이며, 저장 모델 중심 깊이는 이전5.654m → 이동-native5.756m / 이동+C5.747m / 6D+C5.733m입니다. 참조 깊이를 읽거나 투영하지 않았습니다.

상태는 B0/U1/B2/U3/B4/O5/S6/S7입니다. 현재 complete B2는 실제 경계선 보정 출처 B_TARGET_BACKGROUND_LINES의 2D 관측을 유지합니다. B0/B4는 VISIBLE_BACKBONE_FALLBACK 출처를 보관한 기존 partial이며, 완전한 2D 관측으로 사용하지 않습니다. O는1개, U는2개 범위이고 S/U 점 표시는 없습니다.

그림의 ‘CAL 접선0’은 **현재 complete SOURCE 계획에서 새로 대체한 접선 수**입니다. 기존 partial에서 물려받은 uncertain archived tangent B0/B4 두 항은 별도로 유지됩니다(분산×4, 표준편차×2). 실제 이동-native 비용 원장은 complete B2, 미대체 partial normal B4, 기존 tangent B0/B4, projected native edge0–1입니다. 활성 physical edge는1개이며, 여러 NMS 대안은 독립적인 새 점이 아닙니다.

관측·범위 패널에서 새 PnP를 수행한 것은 아닙니다. 마지막 네 결과는 저장된 서로 다른 방법의 최종 자세이며 optimizer 중간 궤적이 아닙니다. 영상선/마스크 비용과 CAL prior만으로 W/D·채널 대응·깊이의 정확성이 보장되지 않습니다. 특정 항을 오차 원인으로 확정하지 않습니다. 추가143의 참조 axes/intrinsics는 UNCONFIRMED이며 주석/PnP 유도값으로 독립 계측이 아닙니다.

원래 RENDER_SEAL 및 완료된 PNG/HTML/JSON은 수정하지 않았습니다. 이 요약은 봉인 후 표시 범위 설명만 추가합니다.
