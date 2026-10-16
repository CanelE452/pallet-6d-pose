# V41 물리 제약 계산 + 잔차 학습

V40의 선분/물리선 입력을 고정하고 제약 전달 방식과 후보 방향을 분리한다. 기존 V30/V35 투영 자체의 재발명이 아니라 같은 입력에서 직접 목표 회귀 대 계산+잔차 회귀의 대조다.

- 입력3개: 실제 LSD선분 예측대응, 같은 선분 GT대응, 정확한 GT물리선. GT입력은 비배포 진단이다.
- 기존24후보: V40직접회귀 재사용, 계산값, 계산값+새잔차. 물리3후보: 각 incident역할의 대응 선 법선으로 이동하며 직접회귀/계산/잔차를 비교한다.
- 3입력×3시드×3새모델=27HGB모델,60라운드씩1620라운드. 별도감사3재학습,신경망업데이트0.
- 같은12fit/4calibration/8재사용check,같은2CAD. 기존V35old16 offset 동일사용. 실제 gate는V39 것을고정한다. 새렌더/실사/수동GT수정없음.
- 체크200관측 중130검출·1040점. 미검출70은조건부점오차에서제외한다. 법선MAE는유효역할에서만계산하며 입력별역할/방향/분모가달라 같은입력·같은후보군 내 비교만해석한다.
- 물리3학습은 fit프레임별 sourcegroup가중 후 유효후보만 남겨 재정규화한다. 따라서 유효후보수가다른그룹의 최종총가중치는 완전히같지않다. 직접/잔차에는동일가중치가적용된다.

환경은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, CPU/sklearn1.4.1.post1. 기존V34~V40 완료캐시 필요.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_constraint_residual_v41/run41.py --phase all
```

`all`은 완료마커 단계를 건너뛴다. 단계는 `train`, `eval`, `audit`, `probe`, `audit_probe`, `report`, `deliver`. 완료된 결과의 개별단계를재실행하면덮어쓰므로 후속변경은새폴더로분리한다. Chrome/X11권한은`deliver`에서필요하다. Discord는사용자요청에따라완료사실만별도전송한다.

결과폴더 `data/pallet/results/pallet_constraint_residual_v41/`. 일차PROTOCOL/SUMMARY, 후속PROBE_PROTOCOL/PROBE, 계산경로GEOMETRY_CHECK,모델AUDIT,후속PROBE_AUDIT,브라우저DELIVERY,최종해시COMPLETION을구분한다.

실제잔차+gate10.821px로기준10.463px보다악화. 정확GT선+물리방향잔차8.772px는진단. 실제물리방향직접회귀10.410px는3시드0.39/.61/.52%작은개선/조건보존이지만등록1%성능기준미달. 독립장면/실사향상이나배포성공으로표현하지않는다.
