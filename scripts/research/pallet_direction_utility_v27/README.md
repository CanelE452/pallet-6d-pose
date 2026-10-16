# V27 방향별 보정 효용 구조

새 모듈4대조×3시드×2000업데이트. 기존 YOLO/DHT/V25 점 보정기는 고정하고 점 유지·점 방향·선 법선·비평행 선 쌍 후보의 이동량과 효용을 학습한다. 역할 감독은 합성 cuboid 대응이며 물리적 가시 경계 감독과 다르다.

- [설계와 결과](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_direction_utility_v27.md)
- [HTML](../../../data/pallet/results/pallet_direction_utility_v27/index.html)
- [사전 고정 프로토콜](../../../data/pallet/results/pallet_direction_utility_v27/PROTOCOL.json)
- [전체 통계·판정](../../../data/pallet/results/pallet_direction_utility_v27/SUMMARY.json)

저장소 루트에서 다음 드라이버를 실행한다. 완료 파일이 있는 단계는 건너뛴다. 전체 완료된 실행은 새 데이터를 덮어쓰지 않는다. 재실험은 별도 RUN으로 복제해야 한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_direction_utility_v27/run27.py
```

단계는 `prepare27.py → train27.py → eval27.py --phase inference → eval27.py --phase score → diagnose27.py → utility_diagnosis27.py → audit27.py → report27.py → delivery27.py → complete27.py`다. 드라이버의 `--notify`는 이미 요청된 최소 Discord 완료 알림만 보낸다. GPU는 RTX3080에서 실행했고 FP32·AdamW lr3e-4·clip5를 썼다.

`network27.DirectionUtilityNetwork`는 GT 입력 없이 부모 표현부터 새 보정까지 합성한 추론 모듈이다. YOLO 원본 이미지 전처리/특징 추출은 기존 canonical wrapper가 담당한다. 운영 모델을 교체하지 않았다. 실사297장 재사용DEV의 계보를 유지하며 정본140/논문 평가 계보나 새 봉인 실사 평가로 부르지 않는다.
