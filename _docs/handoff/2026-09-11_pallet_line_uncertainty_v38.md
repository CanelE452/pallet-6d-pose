# V38 점 위치의 선 보정 불확실성

사용자 승인. V36grid/both고정3seed,V34점기준. 새mean/scale학습12그룹/calibration4/check8,36HGBfits60rounds,신경망0. OOF잔차학습/후보별또는점별max90%경험적범위보정. 평균보정량동일,상수/영상/기하불확실성비교. 기존부모/offset은old16학습이력이있어엄밀conformal보장아님. PROTOCOL사전기준및원결과보존. 실제개선/보정률없으면전부거절도성공아님. 완료후MD/그림/Chrome/승인Discord.

## 완료 결과

36HGBfits2160rounds+3auditrefits,신경망0. OOF18/finalmean6/scale12. Newfit12/cal4/check8분리,기존upstreamold16사용이력명시. PrimarybothgeompointSpearman.525/.542/.568, pointcoverage.871/.850/.845,halfwidth8.849/8.039/8.542px. Ranking/widthgatepassbutcoveragefail,allprimarygatesfail.

V34합성1040점10.463->primary10.487,보정1/3/0. 선택4pointseeds모두구간밖이고3개harm>1px. Appearancepoint평균10.185한seed악화. 큰개선기회297/332/303->positivepredgain27/51/49->positivelowergain0/0/0. Posthocoraclemean9.360/radius10.161/both7.802. Primary고정,quantile50/70/80/90/95calonly곡선은진단이며threshold선택아님.

AUDIT42pointarrays/모든모델출력/3fitcheckcorruption정확재현/정규화gainidentity/이전파일불변PASS. PROBE_AUDIT18oracles60curvesPASS. 3실제합성사례(잘못된방향,놓친이득,좋은점보존)와그림·MD정리. 실제Chrome/Discord완료상태는DELIVERY/COMPLETION/DISCORD_NOTIFICATION. 다음큰오류식별/선택조건부오차학습은미실행.
