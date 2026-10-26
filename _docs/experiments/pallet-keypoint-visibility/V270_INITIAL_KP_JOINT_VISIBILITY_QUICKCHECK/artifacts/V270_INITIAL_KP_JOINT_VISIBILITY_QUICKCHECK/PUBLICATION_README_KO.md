# V270 초기 KP 위치·가림 공동 출력 빠른 확인

초기 전체 이미지 KP branch가 위치와 가림을 함께 출력하도록 구현해, 위치만 / 가림만 / 공동 학습을 합성768장으로 각각300회 학습했습니다. **공동 출력은 작동하지만 이번 짧은 pilot은 가림 판정을 개선하지 못해 채택하지 않습니다.**

| PRIMARY1237 | 보임 정답 /1147 | 보임→가림 오판 | 가림 정답 /90 | 가림→보임 누락 | 균형 recall |
|---|---:|---:|---:|---:|---:|
| 가림만 |636|511|39|51|49.39%|
| 위치+가림 |676|471|31|59|46.69%|

이번 strict D=fraction0 선택으로 partial 음성 후보를 제외하면서 합성 아래쪽 native2/3/6/7의 보임 D 학습 표본이 **각0개**가 됐고, 모델도 실사223장 모두 이 네 점을 E로 출력했습니다. 같은 native2에서 노출 목재 G2와 콘 뒤 G370이 모두 E입니다. 정적 슬롯 사전값의 SYNDEV 균형 recall72.40%와 공동73.09%가 비슷해, 합성 점수만으로 영상 단서 학습을 주장할 수 없습니다. 바닥이 external label에 섞였는지는 first-hit owner 미저장으로 확정하지 못했습니다.

![같은 번호의 보임·콘 가림과 다른 번호의 콘 누락](SAME_SLOT_DIAGNOSIS.png)

- [전체 보고서·목재/정사각형·precision/recall·혼동행렬](ROOT_READOUT_KO.html)
- [실제19사례](visual_v3/INDEX_KO.html)
- [합성 슬롯 교사 감사](slot_prior_diagnosis_v1/READOUT_KO.html)
- [합성 라벨 원본 대조](LABEL_AUDIT_READOUT_KO.html)
- [전체 평가 CSV](evaluation_v4/METRICS.csv)
- [저장 데이터 CPU 재현](PORTABLE_REPLAY_GUIDE_KO.md)
- [학습 전 프로토콜](PROTOCOL_BEFORE_FIT.json)

이번 실험은 backbone/neck을 동결하고 기존 KP shared stem와 새 E head를 학습한 300-step pilot입니다. 전체 네트워크의 처음부터 공동 학습 일반을 부정하는 실험은 아닙니다. 새 E 교사는 M4 픽셀 대신 기존 external9ray proxy입니다. partial/self/outside는 E 손실에서 제외했고 위치 유효성과 분리했습니다. 실제223장은 동결 후 평가에만 사용했으며 반복 개발 평가입니다. 새 PnP·실사 fit·threshold/체크포인트 선택0, 추가 fit0입니다.

다음에는 먼저 각 슬롯의 D/E 양쪽 표본과 같은 시점의 가림 추가/제거 쌍을 확보하고 의미 정의·first-hit owner를 감사해야 합니다. 이번 요청에서는 추가 학습을 진행하지 않습니다.
