# 다음 primary route: E — ADAPTIVE SUPPORT

이번은 FIXED_EXPANSION_INSUFFICIENT, 최종 병목 판정 INCONCLUSIVE. 1.50은 기존 도달불가29개 중14개를 열지만15개는 남아 사전 gate 미달.

후속은 C2 point-aware union crop 한 후보의 **무학습 설계/geometry 확인**을 우선 계획한다. 평가 GT를 crop 입력이나 margin 튜닝에 쓰지 않고, 잘못된 R0 point로 과대 crop이 생길 위험과 object scale 손실을 명시해야 한다. 구체적 margin/guard는 새 실험 전 동결·승인 필요. C3 geometry-aware는 이번 다음 primary로 선택하지 않음. C2가 낫다는 실증은 없고 아직 계산/구현하지 않았다.

현재 요청에서 추가 실행 없음. fixed1.75 rescue, decoder/LoRA/preserve, 학생 self-training, 최종 모델 교체 금지 유지.
