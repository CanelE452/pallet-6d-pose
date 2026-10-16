# V35 실제 경계↔가상 점 역할·offset 검증

사용자의 V34 후속 '확인해줘' 승인에 따라 작은 합성 진단을 실행한다. V32 기존600관측/24그룹, train16/check8, 동일2CAD. 새실사/수동GT/기존완료결과 접근변경 없이 실행. V34 모델3시드 고정 추론, neural optimizer0회. 정답가시 CAD 선→역할별 최장 support선 선택→점 법선 잔차 분해. 학습구간에서만(point,role)상수/방향·aspect ridge offset을 보정하고 분리check에서 검증한다. 정답선/예측top1/GTline최선top4/GT없는confidence선택을 구분한다. 가상선·정답offset은 항등검사/관측가능상한이지 실사용성능이 아니다. 자세한 기준은 `data/pallet/results/pallet_role_offset_v35/PROTOCOL.json`에 고정.

코드 `scripts/research/pallet_role_offset_v35/`, 결과 같은이름results, 최종노트 `_docs/notes/point_line_fusion/04_failure_analysis/pallet_role_offset_v35.md`. 이전V34 등완료파일 hash를 보존한다. 완료 후 MD/실제렌더이미지/HTML작성·실제Chrome표시확인·기존승인Discord한문장 알림.

## 완료 결과

600관측/436검출, check200/130/지원820점,고정3시드,24offset계산,neural0업데이트. 초기gates모두False. 지원점 기준12.373,정답선0offset11.829,상수11.499,ridge11.570;예측상수16.903. Ridge 외삽 폭주41.820을 후속학습범위제한17.556으로 줄였지만 실패. FOLLOWUP에 원결과와 별도로 보존. 여러GT선상수8.296은CI[-10.944,+.999]여서 안정적성공아님. 정답없는선택은1040전체검출점31.752 vs기준13.393.

추가QUANTIZATION: 전체1152bin GTphysical지원최선 거리1.517 vs예측top4GT최선3.955 vs예측top1 7.571px. 상수offset 점오차 연속정답11.499→전체gridGT13.126→예측top4GT14.446→예측top1 16.903. 후보와격자근사 모두문제. 실제역할offset은평균1.129→상수0.681px의부분효과,중앙값거의0. 다음연속θ/ρ 또는support 선맞춤/후보역할개선 학습은미실행.

AUDIT PASS: 2764physical역할선,독립CV투영/선식/normalidentity최대2.28e-13,63방법×seed 수치;checkGT교란해도fit동일. GRID_AUDIT도PASS. 기존완료파일불변,새실사평가/수동GT교정없음. PRIMARY/FOLLOWUP/QUANTIZATION프로토콜분리보존. 실제HTML표시와알림완료는DELIVERY/COMPLETION/DISCORD_NOTIFICATION참조.
