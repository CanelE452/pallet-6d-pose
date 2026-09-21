# 사용자 지정 DAY264 — 보정기 2×2 파일럿

- [확인] 사용자가 낮 플라스틱 보기의 대략1~264번째 구간을 확인하고 “이거 일단 사용해줘”라고 승인했다. `005480.png`~`005743.png`를 고정 후보로 사용한다. 각 프레임이 모두 clean이라고 검증된 주장은 하지 않는다.
- [확인] v1의 자동 후보/조건 메타데이터 조건을 사용자 지정 후보로 대체하는 새 v2 실험이다. v1 완료 산출물은 보존한다. 시작 HEAD는 `73bfe38a259b3c846e49a98fb82c44578f6e2248`, main이다. 자동 commit/push 금지.
- [확인] 이 촬영은 REC_001로 기존 plastic_day_01/wood_day_01 평가와 겹친다. **이번 실험에서는 REC_001 및 모든 alias를 평가에서 제외**하고 원래 전역split/논문표/annotation은 바꾸지 않는다. 이전 Replay의 wood_day TRAIN9 일부와 같은 recording이므로 teacher-independent target adaptation이라고 주장하지 않는다. 실제 동일 TRAIN 이미지 여부는 SHA로 별도 확인한다.
- [확인] R0 → 고정 confidence/flip/LOO → frozen Replay → 기존 자기 가림 PnP → 고정 all8 LOO 순서로 후보를 만든다. GT는 사용하지 않는다. original R0 confidence를 그대로 보존하며 보정 후 confidence로 부르지 않는다.
- [확인] unique 수·시간상 인접 프레임임을 별도로 기록한다. 264장이 독립적264장/다양한264장이라는 주장은 하지 않는다. 통과8장 미만이면 학습 중단한다.
- [확인] 통과하면 synthetic-only PoseFix PRIOR1에서4arm: CLEAN/OCC 입력 × source replay 없음/있음. R0는 frozen. seed1/300update/real8/source8/micro2/TFAdam1e-4/BN통계 고정/last300만 사용한다. 네 arm의 real 순서·노출·target·supervision mask는 동일하다.
- [확인] OCC는 고정 인공 가림 recipe를 적용한 영상에서 실제 R0를 다시 추론한다. clean R0 좌표를 복사하지 않는다. 가림으로 detection이 없으면 clean 좌표로 대체하지 않고 공통 paired training eligibility에서 제외하고 수를 보고한다. 이 조건은 GT와 무관하다.
- [확인] y*는 original-image 좌표로 네 arm에서 동일하다. supervision mask는 clean/OCC predicted crops에서 공통으로 지원되는 코너의 교집합으로 학습 전에 고정한다. crop 좌표 자체는 입력 bbox에 따라 달라지므로 원영상으로 roundtrip하여 타깃 parity를 검증한다.
- [확인] 기존 recipe는 bbox18~35% 패치1~2개,75% 적용확률,4개 이상 trusted corner 유지, 고정902106 RNG다. 각 clean frame당 고정 augmentation 하나를 사용하고, 추가 noise recipe를 탐색하지 않는다. 합성은 기존 Replay source normal/stress 입력 분포 및 고정 ORDERS/heldout을 재사용한다.
- [확인] 기존 E1 frozen baselines를 같은 축소 평가 집합에서 재채점한다. occlusion/clean/GREEN manual/DEV unknown을 합산하지 않는다. inference는 보정기 출력 그대로, detector/score/confidence/center 보존. GT 사후채점, no evaluation filtering.
- [확인] E2 pilot threshold는 원 지시 그대로다. occlusion A11≥R0+3pp 및 기존 N2/N3/Replay 최고 이상; P90≤그 최고PCK comparator의1.05배(동률 이름순 N2 우선); 정상점 손상≤1%; clean PCK10≥N2−1pp; 합성clean PCK10≥초기synthetic PoseFix−1pp, source P90≤1.10배. 주의: GREEN N3 기존출력 부재는 명시한다.
- [확인] 결과가 실패해도 seed/lr/threshold/budget 구제 변경, E3~E6 실행, 최종모델 승격은 하지 않는다.
- [추정] 목적은 clean pseudo target과 occluded R0 input의 분리 효과 및 source replay의 정상점 보존 효과를 확인하는 것이다. 연속264장/teacher 촬영노출/미확정clean 경계/artificial-real gap으로 일반화 주장은 제한된다.
