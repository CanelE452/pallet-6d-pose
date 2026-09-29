# Frozen old-GEO의 간접 수동감독 계보

**현재 학생+D9의 교사 예산은9장/38코너이지만, historical old-GEO를 붙인 전체 경로를9장/38코너만으로 설명하면 안 된다.** old-GEO 학습 입력을 생성한 과거 Plastic S0/S1 학생은 별도의 Plastic10장/48수동코너 감독을 거친 teacher 및 그 수동 출처 support에 의존한다. 두 감독 집합의 ID·RGB SHA·코너 ID 중복은0으로, 이 보완 경로의 추적 가능한 합집합은 **19장/86수동코너**다.

## 실제 연결

```text
기존 Plastic 수동10장/48코너
  → Plastic 전용 Replay teacher (synthetic-only PRIOR1에서 적응)
  → Plastic S0/S1 학생 (teacher 좌표, manual-provenance48 support)
  → 고정 S0/S1이 합성 RGB에서 생성한94-feature
  → renderer parity를 감독으로 학습한 frozen old-GEO

현재 혼합 수동9장/38코너
  → 현재 Replay9/38 teacher
  → clean78 RAW/REF 학생 쌍
  → 동일 frozen old-GEO로 기존두W/D후보 중 선택
```

| 위치 | 직접 사용한 실사 수동 좌표/출처 | 이번 작업의 신규 감독 |
|---|---:|---:|
| 현재 Replay teacher | Plastic3장/15코너 + Wood6장/23코너 =9장/38코너 |0 |
| old-GEO feature를 만든 과거 Plastic teacher/학생 | Plastic10장/48코너; 동일점을 teacher감독과 student support에 재사용 |0 |
| old-GEO scorer 자체 fitting | 합성 renderer parity; real GT read0 |새selector fit0 |
| 추적 가능한 전체 계보 합집합 |19 unique image ID/SHA,86 unique corner ID |0 |

S0와 S1은 동일10장/48코너를 공유하므로 두 번 더하지 않는다. manual 출처 support의 재사용도 새로운48좌표로 다시 세지 않는다. 과거 Clean19에 Wood9장/39코너 arm도 존재하지만, **old-GEO의 실제 feature checkpoint는 `PLASTIC_S0`와 `PLASTIC_S1`뿐**이다. 이 scorer의 계보에 historical Wood9/39를 더하는 근거는 없다. 현재 Replay9/38에 포함된 Wood6장/23코너는 이미 위 현재 예산에 포함돼 있다.

## 합성-only의 정확한 뜻

old-GEO의 fitting은 renderer-group TRAIN4096/VAL1024/TEST1024와 S0/S1 pooled feature를 사용했다. 유효 TRAIN 출력 쌍은8190개였다. real GT를 scorer optimizer에 넣지 않은 것은 맞지만, **feature를 생성한 학생이 real manual supervision의 영향을 받지 않았다는 뜻은 아니다.** 과거 Plastic teacher는 synthetic-only PRIOR1에서 시작했고, 현재의 mixed Replay9/38를 초기값으로 재사용하지 않았다.

추가 감독을 숨기지 않기 위해 원래 student+D9 raw-vs-corrected 표는 유지하고, old-GEO 결과는 이 계보를 포함하는 보완 pipeline 표로 구분한다. 같은 old-GEO를 RAW/REF 두 학생 모두에 적용한 비교는 selector 계약이 공통이지만, pipeline 전체의 수동감독 계보가9/38로 되돌아가는 것은 아니다.

## 평가 중복과 한계

- historical10장의 eval128 exact ID/SHA overlap0, recording overlap0을 확인했다.
- 현재 teacher9장도 eval128 exact ID/SHA overlap0이다.
- current9장과 historical10장 사이 exact ID/SHA 및 코너 ID overlap0을 확인했다.
- 위19/86은 추적한 **좌표감독 계보**다. 이미 여러 번 본 DEV 정답·방법 선택에 쓰인 연구자의 판단·generic upstream pretraining의 레이블까지 포함한 총 인적 비용을 측정한 숫자가 아니다.
- 기존 예산에 대한 정확한 공개이며 이번에 새로 레이블을 요구하거나 historical 결과를 고친 작업은 아니다.

checkpoint·학습코드·support·split·예산 출처18개와 ID/SHA 계보는 [SELECTOR_SUPERVISION_PROVENANCE.json](SELECTOR_SUPERVISION_PROVENANCE.json)에 기록했다. 기존 결과 수치는 변경하지 않았다.
