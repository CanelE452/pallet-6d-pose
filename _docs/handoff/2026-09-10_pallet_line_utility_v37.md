# V37 선 후보의 점 보정 정보 검증

사용자 승인 후 V36 개선 선 후보를 고정하고 V35 잔차/관측가능성 진단을 반복한다. 합성600관측/436검출, train16/check8그룹, 신경망 업데이트0. 점기준 base/V34 고정,4arm3seed. 정답역할/선oracle/점oracle/GT없는선택 분리. 작은96개 ridge gain식은 train그룹만 학습하고 check로 설정을 바꾸지 않는다. 자세한 사전기준은 results/pallet_line_utility_v37/PROTOCOL.json. 이전결과불변. 완료후MD/시각화/실제Chrome/승인된Discord알림.

## 2026-09-11 완료 결과

일차96+후속48=144ridge fits,신경망0updates. V34합성전체1040점10.463, V36both candidate point-GToracle7.802 vsgrid7.769 추가효용fail. 지원820점 선거리7.571->6.222이나점직접투영14.788->14.828. 법선proxyMAE8.203 vsno-op2.720, signed agreement64.7%. 선거리개선1369pointseeds중428(31.3%)는점법선오차악화. 화면밖지원102점 연장거리중앙119.79px,각도×거리항12.30,offset항.46.

PrimaryGT-free visual_geometry12.043 vsvisual13.900이지만anchor10.463미달. Train도악화하므로별도OBJECTIVE_PROTOCOL로normal/pixel추가48fits. normal15.086, pixel10.711;모든stablegatesfalse. No neuraltrainingorPCGradorrealGTaccess. Primary/RESIDUAL/OBJECTIVES분리보존. Oracle wrongline우연활용가능.

AUDIT384primaryarrays/96selectors, EXTRA_AUDIT48추가selectors/metric/GTperturbation검증. 그림4실제합성사례와비교표. 브라우저/알림최종상태는DELIVERY/COMPLETION/DISCORD_NOTIFICATION참조. 후속target-point residual uncertainty학습은미실행,새장기freehead학습안함.
