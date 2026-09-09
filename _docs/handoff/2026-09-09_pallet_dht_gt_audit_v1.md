# 수동 GT 감사 — COMPLETE

2026-09-08T19:56:29.734079Z 실제 완료. Chrome창0x62003d6 viewable/nonhidden,
Discord HTTP204. `COMPLETION.json`존재하며잔여실행/검토/알림없음.
기존GT646입력 및numeric507입력보존검증,최종19artifact/4source SHA재검증.
COMPLETION SHA `5034c5acddcf5d31d897bcac27e0b71276fa994f6861da440dc56c20be1f9660`.
HTML SHA `aa2d20646e8d2e05f3b2cc697dd63c94dcbfe0d35445c520f3b69fbf1c557ebb`.
Finalizer session8794 exit0. 실제GT교정0/신규학습0/forward0이며전GT정확성인증아님.

사용자 요청: 직접 annotation한 GT가 올바른데도41.49px인지 확인.
이전 decoder probe/coupling/joint는 이미 완료, 다시 학습/추론/driver 실행 금지.
이번에 정본 GT와 이전 실험 변경0,새학습/forward0. Git commit없음.

- 새소스 `scripts/research/pallet_dht_gt_audit_v1/`,결과 `data/pallet/results/pallet_dht_gt_audit_v1/`.
- 수치319DEV/13세션/309매칭/2738관측/2818감독/80missing exact재현,
  baseline median6.89698/P9041.48733. 전체GT정확성을 인증한 것은 아님.
- v2표시1562점P9037.108,explicitmanual643점28.812,priorhumanok22장중19matched164점22.847,
  inscreen2697점39.87993. 서로모집단이다르며GT수정효과나새성능결과아님.
- 선결합v2P9039.738/manual31.075로worse, v1는45.830→44.641.
- 원GT646초기inputhash불변검사는finalize가수행. NUMERIC507inputSHA와공식3arms거리/원시review58→50→ok22조인독립감사PASS.
- NUMERIC SHA `4b3c3cabad0298966c0aaf3c36d67ce6e3296357265eac192e60496d674aa7e8`.
- MASK_POLICY SHA `67d5e43020dd8d90cb47edc6e172a44f5d730e489ba336a1bd26ae942b7b4f3e`.
- AUDIT_CONCLUSION SHA `3bab0d37da29ad4e2c6ad96ceecfd188f65075d8722a70778cbb866032e40abe`.
- Root GT-only19detail+037376endpointcrop+3triagesheets 실제확인,17selectedpoint는경계부근/2occluded,
  정확px/IDface선택보증아님. specialistmetadata14+additional2=16장144점독립GT-only검토완료.
- 037376GT7[173319]는fullheightbottom보다위쪽측벽에보임;재검토후보대체xy없음.
  같은imageGT1실제상단부근이지만baseline오차277.643px.
- 032043GT5[650225] width640화면밖인데visible2/manual_click/in_framefalse. raw와HTML실제확인.
- 0121438점/0320439점:manual_kps대실제scoredxy불일치최대32.773px,
  metadata/field차이가어느GT가정확한지는증명안함.현재evaluatorannotations일관사용.
- ε5px가정시같은mask/ID/matchP90범위36.487–46.487;측정된주석정밀도아님.
- GT_REVIEW_QUEUE4frames는질문/검토후보,확정불량denylist아님.
- 디스코드완료알림/실제HTML자동열기사용자기존승인지속.

최종 전달까지 완료: "Historical pose-review OK"표기를실제응답종류에맞춰
"Historical frame/GT-overlay OK"로수정했고원숫자/GT불변을확인했다. 초기5최종PNG와
문구수정후tablePNG를root실제확인,다른4PNG SHA동일성도확인했다.
VISUAL_REVIEW는현HTML/5PNG해시바인딩. Finalizer의source검증→실제Chrome표시→
DiscordHTTP204→COMPLETION까지성공했다. 재실행/재전송하지않는다.
노트/색인/history/handoff도완료상태로동기화했다.

상세해석: [계열 노트](../notes/pallet_dht_gt_audit.md).
