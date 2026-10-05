# 최신 사람 입력 등급 재집계

**실제 재집계·검산 완료.** 직사각형 153/92/74, 정사각형 3/85/31의 사람 입력 등급을 사용했다. 분류 기준은 미확인으로 유지하며 확정 외부 가림 판정으로 승격하지 않는다.

- 전체 및 과거 결과 회귀검산 100/100 PASS; 원본 438장 이미지·참조·출처 해시 유지.
- YOLO Base/P/N0/N1/N2/N3 및 DOPE/ResNet Base/N3, 각 3seed 통계와 재질×등급을 재집계했다.
- D/L/PoseFix는 같은319장, 학생 대안은 Base/N3까지 같은128장 별도 패널이다.
- 정사각형 두 모드는 각각602/600점이며 모든 방법에 같은 분모를 썼다. 독립6D참조가 없어 T/R는 x, 한 세션 일반화 CI는 NA다.
- 새 학습·optimizer update·모델 추론·PnP 0회. 실행 wall 5.444초.
- 이 작업에서는 원고를 수정하지 않았다. 별도 원고 작업이 결과를 삽입한다.

[319장](STATIC_REAGGREGATION.json) · [정사각형](SQUARE_REAGGREGATION.json) · [비교군](COMPARATOR_REAGGREGATION.json) · [학생128](STUDENT128_REAGGREGATION.json)
[검산](STATIC_INVARIANCE_CHECK.json) · [사람 입력 출처](LABEL_PROVENANCE_AUDIT.json) · [해시 연결](SOURCE_BINDINGS.json)

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_static_registry_review_20261003_v1.reaggregate_native_closeout_20261006
```
