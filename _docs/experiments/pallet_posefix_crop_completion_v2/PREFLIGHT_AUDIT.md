# Preflight PASS

HEAD `5b4d42da39b3cb9d5594251513f872c186e7679b`. 기존FULL/PRIOR1 checkpoint·contract, protected 입력 해시 확인. 실사253장의 원영상RGB/고정가림RGB/초기점/bbox/pseudo XY 및1.25 tensor exact parity. 원래 semantic mask 복원 후동일clean/OCC 교집합 적용.

source TRAIN/heldout 분리. 원래1.25교란을300step 전부재생성해 기존trace hash exact일치 후 원영상좌표동결. 신규support에 따라RNG소비 변경없음. real/source row순서일치. 평가GT는학습입력에미사용. 제어crop baseline 불변. 신규학습전 필수테스트 필요.
