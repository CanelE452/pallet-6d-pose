# 새 CAL32 잡음 모델의 저장 수치 재현

CAL46/fixed compatible44에서 별도로 적합한 새20모델(두 source 정책 × full+4LOGO × line/tangent)의60개 시작점을 모두 검산했다. 저장된 CAL error 저널만 읽고, 각 시작점의 NLL·gradient·동일 session 가중치·조건부 floor·선택된 시작점·model digest를 독립 식으로 계산했다. **11,021개 검사 / 실패0**, 최대 수치 차이는2.84217e−14다.

[결과](SAVED_CAL_MODEL_REPLAY_V1.json), [실행 봉인](SAVED_CAL_MODEL_REPLAY_SEAL_V1.json), [별도 소스 검토](../CAL_geometry_guided_edge_noise_saved_replay_peer_review_v4_32/SOURCE_REVIEW_V1.json)를 보존했다. 원 독립 계산의 balanced/evaluate/near/digest 함수는 V29와V31의 AST와 같고, 새32 schema·모집단·source/owner/model pin만 별도로 확인했다. 새 실제32 모델과 이전 모델은 섞지 않았다.

이는 저장 모델 수학의 재현이다. CAL 기준 자세를 새로 decode하거나 refit하지 않았으며, primary·새 RGB/NN·PnP·전체 자세 최적화도 없다. 물리 모서리 대응의 정답, 통계적 독립, 전역 최적 또는 새 자세 정확도를 인증하지 않는다.
