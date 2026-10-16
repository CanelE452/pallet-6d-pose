# 실제 여섯 진단 사례 갤러리 구성

이 파일은 최종 예측·점수 봉인 후 사용할 표시 구조다. 아직 갤러리 실행 완료나 새 결과를 뜻하지 않는다. 최종 렌더는 담당 root의 실행 지시 이후에 수행한다.

지정 ID는 `wood_night_01:029428`, `plastic_day_01:010995`, `plastic_day_01:011640`, `wood_day_01:005249`, `wood_night_01:031426`, `wood_night_01:031344`다. 없으면 비슷한 ID나 유리한 결과로 대체하지 않는다.

각 사례의 첫 이미지에는 원 N3/SubPix+F → 원 e의 활성 full/partial/region/owner → 원 G10 두 family의 저장된 winner → 선택한 family 내부 d/e refinement → e 이후 새 자기 재관측 → 기존 PoseFix RAW/N0/E0 비교를 나란히 표시한다. G10 winner는 실제 중간 저장 후보이며 전체 생성 bank로 부르지 않는다. 두 번째 이미지는 동일한 확대 범위에서 네 고정 진단의 E0 시작 모델과 최종 모델을 보여준다.

full 영상점은 원, predictor fallback은 ×, partial은 자홍색 선·범위, O/U는 원래 영역의 외곽으로 표시한다. S/O/T/U의 영역 중심이나 모델 투영을 정밀 영상점처럼 표시하지 않는다. 모델 코너 이름은 M으로 구분하며 모델 선은 확대 범위에서 clip한다. 모든 패널의 확대 범위는 원 N3 초기점과 F만으로 고정한다.

실측→모델 fitting residual은 px 단위로 따로 표시한다. 전체 자세의 T/R은 봉인된 numerical score의 cm/°만 읽는다. GT 좌표나 참조 파일은 열지 않는다. full xy·관측/fit 공분산·reliability, partial normal·offset·variance·tangent 범위와 owner 연결은 사례의 펼침 표와 PROCESS.json에 연결한다.

실제 RGB는 기존 공개 허용 inventory와 SHA를 검증한 뒤 JPEG 확대 사본으로만 제공한다. 원본 hash와 확대 좌표를 보존하며 새 영상·선·정답을 생성하지 않는다. 패널12개와 원 RGB 확대6개를 포함한 공개 묶음은8MiB 이하를 검사한다. 결과 HTML/MD/PROCESS/GALLERY_SEAL은 최종 실행에서 생성한다.
