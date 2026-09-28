# C — 가림 시도 빈도 한 변수

A에서 real2,560회 중542회(21.17%)만 실제 가림, REF감독21,823점 중824점(3.78%)만 덮였다. 작은 TRAIN probe에서 가린 점의 location gradient는 살아 있다. 이것만으로 노출 부족이 원인임을 확정할 수는 없지만, 더 많은 유효 가림 입력의 추가 가치를 단일 대조로 확인할 근거다.

변경은 A의 schedule probability0.5→1이다. 같은 seed에서 shape/aspect/fill/최대32위치proposal 및 covered≥1/remaining≥2/기존RGB기하증강을 그대로 둔다. gate용 random draw도 소비해 이전 적용 입력은 bit-exact, 새로 시도하는 occurrence만 달라진다. 무효 placement는 원본 입력 유지. 원본 supervision/teacher좌표/source비율/update/lr는 안 바꾼다. 오류정답/오답1:1, 실제 자연가림 등급 균형이라고 부르지 않는다. 1은 시도 빈도의 상한 한 값이며 DEV threshold 최적값이 아니다.

주 control은 이미 완료된 A RAW/REF이며 두 recipe의 baseline OLD_RAW/OLD_REF도 보존한다. C RAW/REF는 동일RGB/계획이며 타깃값만 다르다. 원래 affine의 RAW/REF 경계support차이는 그대로 공개한다. A+B 결합은 B가 기존REF 대비 두 주축 모두 개선하지 않아 실행하지 않는다. F는 기존자료 feasibility를 확인하되 teacher+학생4fit보다 현 학생 경로의 작은 exposure 대조를 먼저 선택했다. 이는 F 성능 실패가 아니다.

Plastic2fit×320update/seed42, 실제세번째이자마지막 주cycle. 주99 T/R, full128/66, Clean/Moderate/Severe·recording·P90·coverage·source를 같은 기준으로 본다. C−A를 따로 계산해 가림 자체 효과와 시도빈도 효과를 구분한다. 결과가 나빠도 빈도·크기·손실 재조정은 없다. 이후에는 사전 선택 규칙에 따른 한 recipe의 재현/control·Wood적용성과 마무리만 한다.
