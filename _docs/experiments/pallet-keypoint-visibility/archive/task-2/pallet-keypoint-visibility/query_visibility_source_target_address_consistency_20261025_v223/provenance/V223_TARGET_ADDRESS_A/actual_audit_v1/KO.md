# SOURCE finite-part 타깃 주소 진단

허용된 SOURCE JSON 5개만 읽었으며 새 PNG·모델·GT·보호 데이터·학습 접근은 0이다. ROI는 학습입력이 아니라 weak bag/region 타깃이다.

| 범위 | 부모 | target central4 접촉 | COVER central4 clear | SHAM central4 접촉 | centroid거리 min/median/max(px) |
|---|---:|---:|---:|---:|---|
| 전체 | 48 | 26 | 22 | 15 | 3.536/17.882/33.022 |
| SOURCE_FIT | 39 | 22 | 17 | 12 | 3.536/17.219/33.022 |
| SOURCE_DEV | 9 | 4 | 5 | 3 | 12.349/20.402/24.137 |

중심은 원 RAW96의 (47.5,47.5), central4는 index47/48의 네 pixel이다. CLEAR는 선언된 alpha support가 네 pixel을 모두 비운다는 뜻이며 실제 RGB·query support·native D/E를 새 판정한 것이 아니다.

48부모를 먼저 접은 뒤 동일 condition32에서 다른 ROI mask가 있는 group은 16개이고, valid까지 같은 group은 14개다. 같은 conditioning의 D/E/D triplet은 의도된 RGB 개입이다. 다른 ROI/라벨만으로 전체 입력이 동일하다고 간주하지 않는다.

기록된 PNG SHA+valid+condition을 함께 묶은 알려진 D/E 라벨 충돌 group은 0개다. 이것은 SOURCE JSON의 선언 identity 검사이며 새 PNG byte/decoded duplicate proof가 아니다. 동일 condition 아래 다른 ROI는 witness 주소의 변화를 보여주지만 RGB가 부위를 주소화할 가능성은 남는다.

각 row→GT candidate/native/q8→finite ROI의 metadata 연결과 위치는 정해져 있다. 그러나 ROI는 배경도 포함하는 weak bag이며 centroid는 물리 부위점/정답 q나 좌표오차가 아니다. 원 detector→GT instance는 UNKNOWN이고 부모 query는 GT-derived training family다. 원 초기 query나 배포에서 native exact point가 유일하게 주소화된다는 결론은 이 자료만으로 확정할 수 없다.

COVER가 중심을 비워도 remote finite target을 가렸을 수 있고, SHAM이 중심을 덮어도 target ROI는 보존될 수 있다. 이를 absence=E 또는 query/native label로 복사하지 않는다. 입력 정보부족·잘못된 annotation·모델 원인·실사 개선은 이번 표에서 증명하지 않는다.

[48부모/native/role/centroid/conditioning 표](PARENTS48_TARGET_ADDRESS.csv) · [144variant 중심 표](VARIANTS144_CENTER_AND_CONDITION.csv) · [condition/충돌 원장](COLLISION_AND_CONDITION_GROUPS.json) · [전체 role×native 요약](SUMMARY.json)
