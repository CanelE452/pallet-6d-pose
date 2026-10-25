V233 게시용 검토 메모

SYN main은 원영상의 저장 intended-native witness0에 연관된 point D/E 감독이다. 원 query 중심 픽셀 owner와 동일한 정의가 아니다. YCB AUX는 저장 associated-r surface D/E이며 native는 UNKNOWN이다. 두 referent의 CM을 하나의 과업으로 합쳐 해석하지 않는다.

두 팔의 DEV 집계는 같다. SYN main 1344행은 D recall98.84%, E recall91.07%, balanced recall94.95%다. YCB AUX397행은 D recall97.94%지만 E recall은 **0/57=0%**이고 balanced recall48.97%다. 각 가족에서 회복1·악화1로 순증0이므로 모든 개별 예측이 같다는 뜻은 아니다. DEV macro balanced-recall gain은 **0pp**로 사전 +3pp gate를 통과하지 못했다. current_admission=false, selected_arm=null이며 새 현재 실사 추론이나 채택은 없다.

원15040행 중 known D/E7617행을 양팔에 모두 보존했고 이 출력의 U는0이다. 나머지7423행은 원 S/U/제외 target−1로 forward 대상이 아니며 모델 U로 세지 않는다. SOURCE DEV는 사전 노출·재사용됐고, SYN donor/background와 YCB DEV의 교차 계보 독립성도 UNKNOWN이다. 높은 합성 SOURCE 수치를 실사 해결·전이 성공으로 확대할 수 없다.

V1은 CONTROL1000 완료 뒤 TREAT 첫 backward에서 deterministic CUDA grid_sample backward 미지원으로 중단됐다(TREAT optimizer0/DEV0). Root는 실패를 보존하고 SOURCEFIT5876의 수리 전후 특징·CONTROL logits이 동일 FP32 bytes/maxabs0임을 확인한 뒤 V2를 실행했다. V2는 완료 CONTROL checkpoint/log를 byte 그대로 재사용하여 newCONTROL0·newTREAT1000이며, 양 final1000 freeze 후 SOURCE 출력을 산출했다. 이는 runtime 오류 수리이지 결과를 보고 모델·seed·예산·문턱을 고른 절차가 아니다. 이 메모와 saved 분석은 Root의 봉인 provenance를 인용하며 모델/forward를 독립 재실행한 검사는 아니다.

근거: [portable 보고서](results/PORTABLE_REPORT_KO.md), [수치](results/SUMMARY.json), [CM/P/R/U 표](results/METRICS.csv), [7617 paired 행](results/PAIRED_ALL7617.csv), [결과 binding](results/COMPLETE.json), [최종 실제 검산 영수증](FINAL_ACTUAL_RECEIPT.json), [검산 코드](summarize_pair.py). 원 결과·코드·영수증은 수정하지 않았다.
