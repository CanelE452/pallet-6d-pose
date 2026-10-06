# final 6D pose target matched 실험 handoff

최신 직접 지시에 따라 새 branch 없이 main에 반영한다. 계산용 작업 공간은 a22fb14beb5e8df08076385000e0d53503c1ae29의 detached checkout으로 유지한다. 원 사용자 파일과 이전 teacher worktree의 264 staged 파일을 보존하며, 논문/TeX/PDF/bib 변경은 이번 commit에 포함하지 않는다.

실제 계산·분석은 완료했다. TRAIN 55,915×201의 11,238,915 F 중 unavailable9개를 +inf로 기록하고 제외행0/NoOp target3135를 확인했다. old161 chunks/2,071,104 완료 F를 반복하지 않고 나머지9,167,811회를 빠른 writer로 완료했다. 원과학 code/bank/GT격리/FP64/index는 동일하다. whole native audit 874chunks/134,922,895 비교와 PRETRAIN PASS 뒤에만 NN 학습을 시작했다. 원 input/bank/feature/model 대량파일은 복사하지 않았다.

정식 fit은 seed1 1개, 6000updates/96000노출/제외0이다. 초기 state/order/model20259params/AdamW/schedule/matched F를 원 receipt와 대조했고 실제 checkpoint optimizer step6000을 확인했다. 최종6000만 SYNTH1985와REAL319에 평가했다. 신규추론444batches/2304examples, 최종 F2304회, 전체실제 F합11,242,425(비용11,238,915+parity1206+평가2304), PnP/LM 각각33,727,266/Generic0이다. bank생성/CNN backbone/PERMfit/REAL학습/hyperparameter탐색은0이다.

고정 SYNTH-only seed1 screen은 STOP이다. NEW−OLD paired 평균 T+1.984673cm/R+0.658447deg/ADDsym+0.018989m이고 RAW good5→bad10 canonical손상63→450이다. seed2/3은 미실행이며 판정 POSE_TARGET_NOT_SUPPORTED다. N3대비 세pose평균도 모두악화해 N3 개선을지지하지않는다. REAL paired ADD+0.010619m frame95%CI[−0.002181,0.026094], pose319/319와2Dmatching311/319를구분한다. 독립 physical reference pair는0/BLOCKED_DATA다.

W/D oracle전환집단은SYNTH68/REAL16, NEW최종hyp회수11/1(OLD4/0)이다. 이부분회수증가가전체pose개선을뜻하지않는다. oracle회수ratio는음수·1초과를clamp하지않으며 SYNTH분모1840/제외145와REAL분모319/제외0을기록했다. 모델의기존pre-update노출적중6245/96000(6.5052%)와static2D/6Dtargetindex일치11874/55915(21.2358%)를구분한다. 최종고유TRAIN전체정확도와underfitting확증은측정하지않았다.

실제내부wall: CPU비용4890.596120초(최초timing포함), parity14.327230초, aux52.138198초, fit386.748604초, eval17.545599초. 전체subprocess aux55.934849/fit391.544992/eval33.467681초다. GPU/CPU/I/O혼합wall이며isolatedCUDA kernel시간은측정하지않았다. 빠른비용단계1469.714952초/기존wall처리량10.2231배, 비계산I/O전환대기421.897403초를분리한다.

계약테스트30/30PASS와I/Ofixture12/12PASS, final metadataSHA연결PASS, 독립최종134,959,817비교/2304평가행/전체원입력710개76,031,296,408bytes종료SHA/보호351파일/원사용자checkout검산PASS다. rootverification57.929923초중원입력종료부분51.854121초다. 추가과학재계산0이다. RESULT/NEXT의마지막rootprose수정은관측모델정확도와staticteacheragreement, 내부/전체wall의범위를명확히한것이며수치/판정규칙변경0이다.

재현의명시적의존성: Python /home/minjae/anaconda3/envs/pallet-yolo26/bin/python, readonlybaseline /home/minjae/Documents/github/pallet-pose-handoff-20261006(a22), 원데이터 /home/minjae/Documents/github/pallet-pose, 원bank /tmp/pallet-joint-action-cache, 새cost/model/WAL/perID/logcache /tmp/pallet-pose-target-6d-cache. baseline이없거나다르면BLOCKED_DATA/INTEGRITY이며새bank로대체하지않는다. 원입력경로의detached실험worktree를삭제하지않는다.

원main39d4219a와원격88ee557e사이기존파일바이트/symlink를감사했고, COMPUTE_AND_REPORT_DONE+endingPASS뒤publication helper로index/ref만통합한다. 원사용자파일과index는cache/main-preservation에백업한다. 원래없던92파일과symlink아래trackedchild1개의skip-worktree93표시는upstreamworkingfile생성없이mainparent를통합하기위한로컬Gitmetadata다. 기존사용자변경은stage하지않는다. main에이전a22논문변경을merge하지않고전용code/docs/notes/handoff와새historysection만옮긴다. 원main에2026-10-06history가없어이번6Dsection만새로작성한다.

저장소STATUS는commit생성전계산·검산snapshot이다. 실제normalcommit/push와remote main SHA==실제결과commit의검증은외부cache/PUSH_VERIFICATION.json에서확인한다. 자기SHA를본문에넣는재귀commit은만들지않는다. 최종code/doc/handoff SHA는저장된게시manifest와비교한다. 후속실험은계획만있으며자동실행하지않았다.
