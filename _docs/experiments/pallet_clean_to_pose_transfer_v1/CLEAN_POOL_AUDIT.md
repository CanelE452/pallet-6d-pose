# Current217 clean 감사 — RGB-only 잠금 완료

1000 후보 →249 승인 →217 실제 TRAIN /512 real 슬롯을 복원했다. 기존123장 사람 난도tag와 승인249장 image SHA 교집합은 0이다. 촬영 세션·주야 meta는 clean 근거가 아니므로 assistant가 예측·GT·error·confidence·보정량 없이 원 RGB249장을 모두 확인했다. 이 판정은 사람이 직접 단 tag나 새 좌표 감독으로 부르지 않는다.

사용자 정정: **“마커판이 붙은 영상도 clean에서 제외”**. 부착 마커가 관찰된 accepted158장/used137장을 제외했다. 사람 하단과 팔레트 상단 윤곽의 겹침이 애매한 2장은 native RGB로 다시 확인한 뒤 UNRESOLVED로 두고 제외했다.

| 집합 | 외부 가림 없는 후보 | 마커판 제외 | 사람 겹침 불확실 | 합계 |
|---|---:|---:|---:|---:|
| accepted | 89 | 158 | 2 | 249 |
| 기존 실제 TRAIN | 78 | 137 | 2 | 217 |

새 primary는 기존 USED217 안의78장/4recording만 허용한다. 이전에 사용하지 않은 clean11장은 넣지 않는다. 세션별78장: capturenight03 29 / capturenight04 6 / capturenight10 27 / capturepallet10 16. 두 불확실 영상의 확인을 기다리지 않고 확실한 subset으로 진행할 수 있다.

**78장이 모두 쉬운 영상이라는 뜻은 아니다.** 저앙각이 많고 일부는 잘려 있다. EXTERNAL_CLEAN과 SELF_OCCLUSION/TRUNCATION/HARD_VIEW는 구분한다. 이번 입력선택 기준은 외부 가림·마커 없음이며, 잘림·시점별 전수수치 tag는 새로 만들지 않았다. 외부 가림 없는 RGB라고 기존 pseudo 좌표의 정확도가 검증된 것도 아니다.

RGB contact sheet30장(accepted16+used14)와 CSV는 private outputs에 유지한다. 공개되는 것은 집계/해시이며 원 RGB를 무심코 push하지 않는다. [CLEAN_LOCK.json](CLEAN_LOCK.json)의 정책·membership hash는 새 fit·평가 전에 잠갔다.
