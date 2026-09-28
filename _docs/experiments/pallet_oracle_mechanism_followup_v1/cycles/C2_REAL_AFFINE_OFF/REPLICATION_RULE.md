# C2 재현 예산 예약 — 첫 DEV 결과를 열기 전

2026-09-28, C2 첫 네fit의scoring결과를 parent가받기전에고정했다. 원래primarymaterial인 Plastic에한해, 새REF PCK10이기존REF보다증가하면 그recipe효과의 작은재현을실행한다. 다른보조지표손상은숨기지않으며그조건으로자동탈락시키지않는다. 개선0/음수이면유망결과재현용예산을채우지않는다. 우연성위험을줄이려는절차이지 성공threshold가아니다.

4fits를예약: optimizer/data randomness seed43에서 ORIGINAL_RAW / ORIGINAL_REF / AFFINE_OFF_RAW / AFFINE_OFF_REF. 모두같은기존R0초기값과기존Plastic217·동일1024slots·기본320updates, 마지막checkpoint만평가한다. 최초42결과를교체하지않고43에서의recipe차이와raw/corrected차이를별도로보고한다. 기존ORDER43은이seed43실험으로대체하지않는다.

초기화는동일pretrainedR0이므로독립랜덤초기화실험이아니다. Wood까지재현했다고쓰지않는다. C3RAW9/MANUAL9 capability2fits와합쳐도총10fits/3200updates로상한12/7680이내다. GPU는C3와동시사용하지않으며handoff이후순차실행한다. 성능이좋은seed/epoch만고르는선택은없다.
