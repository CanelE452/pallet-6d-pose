# V211 독립 코드·선택 metadata preflight

PASS. 고정 TRAIN110 parent ×4 scene ×8 native =3520 own point metadata와 C61 eligible7040 입력 대응을 독립 재계산했다. own inputU32 / mapped inputU64, SELF NOMINAL3520 + BEFORE_ANCHOR3520, 각 own teacher는 두 입력에 대응한다. 2640 입력은 실제 image scene과 BEFORE query anchor가 다르며, teacher key는 image 원 global ID//8와 native이고 q/L/flags는 원 anchor에서 별도 보존된다. 원 cache index로 point 상태를 붙이지 않는다.

Reader는 saved top `rows`에서 scene_index·variant·base를 유일하게 고르고 `records_CF`와 원 CF→canonical perm을 검사한다. point0의 target/combined 상태와 region, 원 inputU, frame/crop algebraic locality를 분리한다. 누락 point는 UNKNOWN이며 region/nearest/canonical list로 대체하지 않는다. 실제 GT 조회 전 root 승인·정확 whitelist·SHA가 필요하다.

독립 가상12검사 PASS. 실제 GT/source payload의 stat/hash/header/body·RGB/NPZ·가중치·모델·보호 점수 접근은 모두0이다. C는 B의 선택3520/7040 metadata 정합을 확인했으며 원65536 역할 전수집계나 실제 GT 상태를 재검증했다고 주장하지 않는다. 새 label/학습 admission은 없고, 저장 procedural SOURCE point 권위는 실사 native gold가 아니다.
