# V229 고정 후보 실사 비교

[결과와 전체 비교표](PORTABLE_REPORT_KO.md) · [성공·실패 8개 crop](figures/PORTABLE_GALLERY_KO.html) · [전체1784 ledger](results/ALL1784_LEDGER.csv)

공개 SOURCE에서 선택한 CONTROL의 실사 D recall83.26% / 외부E recall30.00%로 채택하지 않았습니다. 새 학습·문턱 조정·PnP는 없습니다.

패키지 루트에서 `python -B VERIFY_SAVED_CURRENT.py`를 실행하면 표준 Python으로 저장 표를 검산합니다. private 원영상·DINO 가중치 없이 CM/P/R/U와 원행순서/입력U/logit부호를 확인합니다. 실제 inference 재실행과 원 GT 권위 검증은 아닙니다. 코드·고정 가중치·입력 해시와 freeze 순서를 함께 보존했지만 실사 원영상223장 전체와 입력 cache는 복사하지 않았습니다.
