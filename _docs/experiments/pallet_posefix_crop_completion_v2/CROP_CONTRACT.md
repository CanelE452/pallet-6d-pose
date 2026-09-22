# 고정 crop 계약

R0 predicted bbox, 중심/종횡비정규화/보간LINEAR/zero padding/384×288 유지. 공통 explicit expansion:1.25 또는1.50만. output96×72 expectation×4의 연속지지영역 [0,284]×[0,380], 학습 mask[0,288)×[0,384)와 다름. inverse affine로 원영상 복원, invalid점·center8·bbox·score·selected candidate pass-through.

확대 시 물체 입력 scale=5/6, 원영상 grid간격=1.2배. crop좌표 loss의 원영상당 크기와 지원감독량도 바뀌므로 C−A는 crop pipeline 총효과이며 순수 support만의 인과효과가 아님. 원영상 RGB에서 다시crop, 기존crop resize 금지.
