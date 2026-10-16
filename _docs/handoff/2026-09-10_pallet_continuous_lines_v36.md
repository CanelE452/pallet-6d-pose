# V36 연속 선 좌표·후보 선택 학습

사용자 '그렇게 해줘'에 따라 V35 후속학습을 실행한다. 코드/결과 pallet_continuous_lines_v36. 4arms(grid/refine/rank/both)×3seed×2000=12models24k. 기존V34 trunk/parents/query/lineencoder 고정, 같은diverse8후보, 새좌표·qualityhead와동일기존pointhead만학습. 좌표는one-bin theta/rho잔차. Geometry와rank는실제선GT, pointhead는점GT; point→linehead gradient 차단. Pointhead를새로학습하므로격자대조도같이학습한다. 실제 V34top4는별도기준.

V32동일데이터/배치, V35최장가시physicalsupport 정답. check/realGT학습금지, 기존결과불변,새실사재사용DEV계보명시. 출처/기준은PROTOCOL. 검증항목:연속좌표정확도/선후보rank/point성능/geometry개입/기존점보존3seed. 수치안정 V34dense sampler유지. 학습전geometry identity·pointgradient차단·finite·동일초기값검사. 완료후MD/시각화/실제Chrome열기/승인된Discord완료알림.

## 일차 완료/추가 대조 진행

일차12모델24k완료. PREFLIGHT PASS:subbin이동,physicalauxgradient,point→refiner차단,동일forward/gradient반복. Point cacheV34재현max3.1e-5. LINE_SUMMARY:check820지원점 grid7.571/refine7.249/rank6.037/both6.222px. Both세시드17.98/12.65/22.34%선거리개선,CI[-2.989,-.00687]gate통과. Best8좌표효용both1~2%로10%기준실패. 후보도달가능비율86.91%.

실사crop20outside grid13.485/refine12.720/rank14.306/both14.394px. Refine세시드9.38/4.56/2.75%개선,CI[-1.412,-.263]이나primaryboth실패,모든arm안정성실패. Refine최종좌표보정끄기+0.000412px,geometrydonor+.004759(CI0포함),lineoff+1.482. 단순히추론정밀좌표덕분인개선으로부를수없음. Bothgeometrydonor+.00121 거의무반응. 추가PROBES는posthoc보존. 선headgeometry/rank hidden gradient20배치/seed:cos중앙+.03~.057,음수40~45%,ranknorm/geomnorm중앙5.4~9.5. PCGrad미실행,지속충돌주원인확정아님.

기존line_token/line_relation을고정한설계의한계분리를위해 ADAPT_PROTOCOL등록: grid/refine/both×3seed×2000추가9모델18k. 총21모델42k. 새linepredictor는일차최종값고정,pointhead와line_token/relation은같은V34초기값으로리셋후공동학습. 모든캐시/배치같음,pointloss만. 일차결과/게이트불변,추가결과는ADAPT_SUMMARY. adapt36.py세션65312진행중,로그adapt.log. 기존coretrunk/query와refiner불변감사필요. 완료후감사/MD/그림/브라우저/Discord남음.

## 최종 추가대조·감사 완료

추가9모델18k도완료,총21모델42k. ADAPT:실사crop20 grid13.377/refine13.469/both13.613. Refine한시드8.30%악화,둘다CI악화. 모든보존/추가효과gateFalse. Geometrydonor refine+.00636(CI0포함)/both-.000175(CI0포함). 인코더고정만이원인이라는가설미지지. 선 정확도부분성공과포즈융합실패구분,운영교체없음.

AUDIT PASS:21models42000updates/543평가예측,고정core/init/plan/추가refiner정확불변,9840선오차검사,manualvalid/crop/sha전달검사2376행. 전체21학습각각2회반복은아님. 기존완료파일불변. 보고서 pallet_continuous_lines_v36.md. 시각화/실제브라우저최종확인/Discord는DELIVERY/COMPLETION/DISCORD_NOTIFICATION에기록. 새개선후보에서점잔차oracle효용이늘었는지후속진단은미실행.
