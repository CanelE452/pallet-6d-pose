# V41 물리 제약 계산·잔차 학습 검증

사용자 계속 승인. V40관측/GT대조불변,V34고정3시드. 기존24후보target대조에analytic+residual추가,같은물리normal3후보direct/analytic/residual별도. 3source×3seed×3newfits=27HGB60rounds,신경망0. 같은12fit/4cal/8reusedcheck,same2CAD/old16offsetreuse. GT입력/GTsafe/oracle는진단. PROTOCOL사전등록. 기존결과불변,완료후MD/그림/실제Chrome/완료사실만Discord.

## 실행 완료

27HGB×60=1620라운드,감사3재학습,신경망0. 기존24후보의V40direct재현/analytic/새residual;물리법선3후보direct/analytic/residual,3source3seed. 같은12fit/4cal/8reusedcheck,same2CAD/old16offset. 새실사/GT수정/렌더없음. 모든일차마커완료.

정확GT물리선+physical3residual10.463→8.772px,15.61/16.59/16.26%,CI[-3.639,-.254]. GTdirect9.078이고residual-vs-directCI[-.987,+.233]로추가점이득미확정. GTnormalharddirect14.169→analytic1.456/residual1.582px. GT조건은비배포이며crop20/40easy최대8.07/13.87%악화.

실제physical3direct10.410px,0.39/.61/.52%,CI[-.076,-.028]. 모든check층최대악화.0448%로1%보존기준내. 등록1%성능기준미달,V40평균10.329보다나쁨,작은재사용check효과로보존하되독립검증필요. 실제analytic16.799/residual12.564/residualgate10.821px,primary모든gatefalse/acceptedfalse.

후속PROBE: 실제residualgate피해45점×seed중noGTvisiblesupport34/physicalmismatch8/close3. GTqualityoracle로고정선택거절10.423px,0.24/.25/.69%효과,GT선품질/가시성사용진단. primary변경없음. noGTvisiblesupport는캐시가시지원없음이지모든해당영상선분이가짜라는뜻아님.

AUDIT PASS:27모델replay,186점배열조건별재계산,3비학습영역교란재학습,actual입력GT독립,legacydirectV40완전일치,접선보존/15%bound/잔차offset항등식. PROBE_AUDIT PASS:외적거리품질재계산/GTquality출력재현. 기존완료결과불변.

MD/HTML/2통계그림/3실제사례작성및개별시각검사. actual_false_support원래4.99→51.56px,actualdirecthelp12.08→10.93,GTconstrainthelp2.21→.34. 선택은고정seed1조건부중앙값. 실제HTML표/Chrome창은DELIVERY,해시COMPLETION,완료사실만Discord는DISCORD_NOTIFICATION참조. 다음턴에27모델반복금지,새독립장면/제약유효성후속은미실행.
