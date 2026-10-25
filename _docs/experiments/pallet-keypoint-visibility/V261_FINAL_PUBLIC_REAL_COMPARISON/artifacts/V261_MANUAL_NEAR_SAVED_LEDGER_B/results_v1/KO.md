# 동결된 current 표의 수동 좌표 근접 사례

원영상 오차 ≤3px인 human D·원입력 valid 사례는 461개다. 좌표 권위는 manual_click 376개, pnp_projected 52개, unknown 33개로 나뉜다. 기존 LE3 집계 전체를 수동 클릭 정답으로 부를 수 없다. 수동 클릭 376개는 모두 primary1237에도 포함되어 두 범위의 이번 분모와 응답 수가 같다.

수동 클릭 376개에서 D/E/U 응답은 v61 85/291/0, legacy 205/108/63, CONTROL 316/60/0, TREAT 308/68/0이다. 이 D-only 부분집합의 E는 기존 human D에 대한 false-absent 응답이다. Legacy의 U/미해결에는 기존 out-of-task·abstention 구분이 포함될 수 있어 원입력 불가와 동일하게 해석하지 않는다.

원영상 오차를 변경하지 않고 `오차×96/L`로 RAW96 좌표 거리만 계산했다. 고정 구간 [0,1], (1,3], (3,6], (6,12], (12,∞)의 수동 클릭 사례 수는 35/181/134/23/3이다. 따라서 원영상에서 ≤3px인 376개 중 160개는 RAW96에서 3px보다 멀다. 이것은 분포 설명용 구간이며 분류 문턱이나 새 학습 조건이 아니다.

이미 선정된 사례 0684는 L=17.795578, 원오차=2.334941이므로 RAW96 거리=12.596으로 확대된다. 사례 1680은 L=81.424974, 원오차=1.964749로 RAW96 거리=2.316이다. 같은 원영상 ≤3px 기준이 crop에서 같은 근접도를 뜻하지 않는다는 기하학적 차이다. 두 사례는 모두 v61 D, CONTROL E, TREAT E이며, 이 두 사례만으로 작은 L을 오판 원인으로 확정할 수 없다.

수동 클릭의 visible 기준점이 q에 가깝다는 사실은 변경하지 않은 q 픽셀 자체의 target-presence 정답을 인증하지 않는다. 새 mask·native gold를 만들거나 q를 기준점으로 이동하지 않았다. 높은 공개 conditional AUX 점수도 current native 일반화의 증거로 대체할 수 없다. 반복 노출된 current·기존 CAL 한계는 그대로다.

새 crop·모델 실행·학습·문턱 또는 데이터/recipe 선택은 하지 않았다. 모든 수치의 입력은 동결된 current ledger와 이미 선정된 gallery CASE_FACTS 두 파일뿐이다.

- [수동 근접 응답 수](MANUAL_NEAR_MODEL_COUNTS.csv)
- [원오차/L/RAW96 거리 분포](ERROR_SCALE_DISTRIBUTIONS.csv)
- [고정 거리 구간](ERROR_RAW96_FIXED_BINS.csv)
- [좌표 권위 구분](ANNOTATION_AUTHORITY_LE3_HUMAN_D_COUNTS.csv)
- [이미 선정된 두 사례](GALLERY_ERROR_SCALE_EXAMPLES.csv)
