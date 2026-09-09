# Global layout pilot — COMPLETE

실험·독립 검증·실제 HTML 검토·브라우저 표시·Discord 전달까지 완료했다. 정확도 개선 기준은 불충족이다. 재학습/재평가/재알림할 잔여 작업은 없다.

- 완료 UTC: 2026-09-08T22:49:34.334104+00:00 (한국 2026-09-09 07:49).
- 실제 Chrome 창: 0x62003d9, title `Global Layout · 8점·12선 공동 선택`, MapState IsViewable. Discord HTTP204.
- HTML SHA: 7f5636ab070e3dc4eec66850e6fc27d8448585a90d8fd562ca54c8944555fac5
- COMPLETION SHA: 9fdae843d74d4e8f0ce3b8216a5eea82465ae98f17da08afe3ac51d1ec38e772
- SOURCE_FREEZE SHA: 5feee6da16a3c8dfe0928ac282354e8a03680859b324405e082f9b4500ce4897
- PROTOCOL SHA: d75d54c33c02213610c28ef07e6fcff8afee22a8c0de6f331faeae549adb73e1
- RESULTS SHA: 1f92105d6804eab178868e44946d6cd076b00da6d39d00d98aa270342bc52e12
- PREDICTIONS SHA: 13f35f769f711b4cb116b70704dcd6d68607a241531f69d5925363de7c948fd2

입력6,471개SHA보존,25생성기하tests PASS. 기존 hough_joint_seed1 EMA 캐시만사용했고 새CNN학습/forward0. 합성256장2022코너로9grid를선택(wpoint4,wgeom1)후동결,합성val512+실사319의전체배치선택완료. officialsameID·309매칭·2738/2818점·80missing·centroid8 그대로다. GT수정0이며전체GT정확성인증아니다.

실사 baseline/independent/global median6.897/13.341/11.444,P9041.487/105.697/79.682px. global−baseframe평균+9.190 CI[5.784,13.372],global−indep−6.558 CI[−10.651,−3.553]. good1711중577개이탈,hard평균92.547→100.197. synthval P907.522→32.994. calibration best9도baseline정규화오차의3.03배로악화. 진행기준false.

문제사진에는sameID GTmedian20.704인explicit yaw90가있지만score16.191로미선택;selectedmedian278.061 score5.465. pointprior10.363뿐아니라roleline도기존오배치를선호하고DLT는C4불변. 고정19frame사후sameIDGTnearestlayout은모두선택보다GT에가까우면서점수가불리했다. 이는현점수의병목이지결합아키텍처전체의불가능성증명아니다. POSTHOC라인지는GT-derived입력임을명시했으며숫자불변.

831프레임×3선택arm의저장점수2493개독립재계산최대1.07e−14;core동결보존. HTML319원본decode/1276overlay/1914hypothesis,JS오류0. root5개최종screenshots실제확인PASS. 첫QA에서렌더소스동시변경을감지해종료했고,core수치변경없이렌더+QA를동결후다시완료했다. 최종driver81186 exit0,STATUS complete. 기존driver49529 exit1은이렌더검증중단기록이며계산실패아니다.

코드 `scripts/research/pallet_dht_global_layout_v1/`, 산출물 `data/pallet/results/pallet_dht_global_layout_v1/`, 과학적해석 `_docs/notes/pallet_dht_global_layout.md`. 기존decoder/GTaudit/joint/coupling산출물과무관한dirtyworkspace변경보존,commit0.
