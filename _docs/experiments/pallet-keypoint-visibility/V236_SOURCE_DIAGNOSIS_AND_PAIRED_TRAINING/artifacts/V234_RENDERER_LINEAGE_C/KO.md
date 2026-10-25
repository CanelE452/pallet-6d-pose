V129 actual entry54aeba8d…가 renderer f6093f95…와 core96713b5c…를 pin한다. 이3파일의 SHA를 확인하고 CODE만 읽었으며 import/실행·config192/policy/result/배열 조회는 하지 않았다.

**검토한 RGB writer 경로는 절차적이다.** core.material_rgb55–59는 고정 색과 seed/좌표 sin으로 wood/polymer 색을 만들고, shade63–71은 seed 기반 배경색·광원·ambient와 geometry normal을 사용한다. foreign family도 같은 절차 재질, tan/sine, bgbase 또는 고정 고대비 색이다. BACKGROUND_LIKE라는 이름도 실제 배경 사진 복사가 아니라 bgbase 상수다. 배경은 화면좌표 sin/기울기 수식이며 마지막에 sRGB 변환·uint8 rounding한다. renderer에는 Image.open이 없고 Image.fromarray(rgb).save만 있다. np.load는 지정된 geometry array key만 읽으며 RGB/texture member를 가져오지 않는다. 따라서 이 두 pinned CODE의 명시적 RGB 입력 경로에는 YCB 사진 donor/background 로딩·paste가 없다.

selected110에서 본 geometry/foreign_geometry/raw_buffers는 writer55/60/70/72/75에 직접 대응한다. raw buffer의 RGB_uint8은 core.shade 결과이고 같은 binding을 GT record raw_buffers에 적는다. base/scene/variant는 실제 생성 영상 identity이며 원 crop q-anchor를 뜻하지 않는다. entry는 writer code binding을 닫고, CODE는 관찰한 저장 GT field와 output경로 schema를 닫는다. 다만 각 selected parent의 실행 영수증·실제 RGB byte를 이번에 다시 검증한 것은 아니다.

여기서 geometry template의 원 제작 계보나 imported raster/teacher/helper 내부까지 검증했다고 확대하지 않는다. 선택되지 않은 corpus/held body나 geometry NPZ를 열 필요를 새로 만들지 않았다. 더 넓은 imported-dependency CODE 감사가 필요하면 REVIEW의 exact V128 helper literal 또는 entry.protocol에 묶인 geometry_code/raster_code pointer부터 Root가 별도 범위를 줄 수 있으며 현재 모두 inert다.

이는 이전 “GT JSON에 asset 계보가 기록되지 않았다”를 넘어, **pinned RGB 생성 CODE에서 외부 사진 입력 경로가 없다는 직접 증거**를 추가한다. 전체 실험의 fresh 독립성·모델 전이 성공·current admission으로 확대하지 않으며 V233 결과/gate/입력/target/모델/분할은 바꾸지 않았다.
