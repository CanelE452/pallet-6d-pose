# V40 선분·끝점 대응 정보 진단

기존 V39 고정 특징에 실제 영상 선분 또는 정답 물리 경계 관측을 추가해 보정 방향·크기 정보를 분리한다. 새 신경망 아키텍처나 실사 최종 평가가 아니다.

- 기존 V34 점 / V36 both의24개 보정 후보,3시드. 입력만 추가하고 후보를 늘리지 않는다.
- 실제 LSD 선분+예측 대응 / 같은 선분+GT 대응 / GT 물리 직선 / GT 가시 구간·끝점 / 역할을 바꾼 GT 구간의5대조.
- 전체 점 및 큰 오류(기존 점GT오차≥10px) 전용 회귀:30HGB×60라운드=1800라운드. 감사 재학습3개 별도. 신경망 업데이트0.
- 학습12/기준 선택4/재사용 확인8그룹, 같은2CAD. 확인200관측 중130검출·1040점. 실제 적용에는 V39 큰 오류 선택기를 고정 재사용한다.
- GT 입력 대조/GT 선택은 비배포 진단이다. 가시 구간의 끝점은 실제 모서리 접합점이나 가상 꼭짓점 GT가 아니다. 첫/마지막 가시 샘플 사이에 중간 가림이 있을 수도 있다.

환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, CPU, sklearn1.4.1.post1. 기존 V34~V39 캐시와 합성 원본 이미지가 필요하다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_segment_correspondence_v40/run40.py --phase all
```

완료 마커가 있는 단계는 `all`에서 건너뛴다. 단계: `train`, `eval`, `audit`, `probe`, `audit_probe`, `report`, `deliver`. 개별 단계 재실행은 해당 결과를 덮어쓰므로 완료된 실험을 변경할 후속 작업은 별도 버전으로 분리한다. `deliver`에는 Chrome/X11 화면 접근이 필요하다. 사용자 승인된 Discord에는 완료 사실만 별도로 전송한다.

결과는 `data/pallet/results/pallet_segment_correspondence_v40/`. 일차 `PROTOCOL.json`, `SUMMARY.json`과 후속 기술 통계 `PROBE_PROTOCOL.json`, `PROBE.json`을 구분한다. 모델/수치/GT분리/좌표 불변성은 `AUDIT.json`, 후속 기하 분해/끝점 거리는 `PROBE_AUDIT.json`, 실제 창/HTML표는 `DELIVERY.json`, 완료 파일 해시는 `COMPLETION.json`을 본다.

실제 영상 선분 추가10.463→10.329px는 시드/조건별 안정성 실패. 정확한 물리 직선의 부분 정보는 있으나 가시 끝점 추가의 개선은 확인되지 않았다. 기존 특징 보정10.290px도 넘지 못했다. 단순 선분/끝점 입력 모듈을 채택하지 않는다. 새로운 독립 그룹, 실제 접합점/가림 종단 구분, 역할·방향이 맞는 제약 전달 검증은 미실행 후속 범위다.
