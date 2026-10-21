> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# 치수는 보정에도 이미 사용했는가

그렇다. 실제 retained 입력을 확인했다.

- N3 신경망: W/D/H(m)를 log/ratio5차원 context로 직접 받는다. 목재 실제cache[0.8,0.59,0.14], 플라스틱[1.1,1.3,0.11].
- PnP: 치수로 만든 cuboid3D 점을 직접 사용한다. 알려진 XYZ=[W,H,D]m와camera-facing 치수·좌표 변환을 구분한다.
- D1 보정/후보 생성: native_xyz의 실제 길이를 K,R/t로 투영해 점·부분선·윤곽 잔차와 제약을 계산한다. Q는 좌표 표현을 연결하는 회전이고 길이 자체는 native_xyz에 있다. source solver._native_pose는R_native=R_physical Q^T, t_native=centroid−R_native mean(native_xyz)를 사용한다.
- H1: W/D/H를 별도 입력scalar로 추가하지 않는다. 그 치수에서 생성된 후보, 잔차/cost, covariance/scale 및bank 문맥 특징을 간접 사용한다. H1이 치수를 새로 추정하는 것이 아니다.

E84의 물리 XYZ=[0.8,0.14,0.59]m, 원 camera-facing extents=[0.59,0.14,0.8]m이다. paired R/Q로 초기물리회전과 registered geometry를 보존하는 다른 표현이다. 그 자체가 치수 오입력의 증거는 아니다. 길이를 알고 있어도 관측과 물리축의 대응 또는 후보 생성/선택이 틀리면90도 가까운 오류는 남을 수 있다.

실제근거: REAL13_N3_INPUT_METADATA.json, REAL13_ACTUAL_INPUTS.json, retained inference.py/refiner.py, D1 pipeline_contract.py, solver.py, statewise_two_wd_family_v4_9.py. 원방법과 최신감사는 치수를 바꾸지 않았다. 개별 실물 재실측 여부는 미확인이다. 새실험은 하지 않았다.
