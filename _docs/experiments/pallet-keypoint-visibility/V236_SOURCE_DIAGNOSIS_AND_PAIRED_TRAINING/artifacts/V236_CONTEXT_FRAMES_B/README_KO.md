# V236 고정 SOURCE 세 사례의 문맥/crop 표시

Root가 픽셀 열람 전에 정한 세 그룹의 최소 sample을 그대로 표시했습니다: object2/aux2(sample7536), object7/aux3(sample7220), object7/aux7(sample7348). SOURCE의 기존 TRAIN→internalDEV 자료이며, 추가 학습에 쓰지 않았습니다.

각 카드 왼쪽은 원 fullframe에 q8·q box(cyan), 선택 q(white), 원 L 범위(orange), 진단용 reference r(magenta)를 표시합니다. 오른쪽에는 원 RAW96 clean과 표시용 확대본을 분리했습니다. 원 source PNG3개는 byte copy로 보존했고 raw96는 pinned NPY의 세 행만 읽어 그대로 저장했습니다. q/L/색상·crop은 바꾸지 않았고 GT recrop/recenter도 하지 않았습니다.

전체 board를 실제 tool view하여 세 사례와 글자·범례의 판독을 확인했습니다. Fullframe의 물체 배치와 q8 범위는 crop의 흰 표면/부분 인쇄/붉은 경계보다 넓은 문맥을 보여줍니다. 다만 mask 없는 시각만으로 어느 물체가 GT target instance인지, 그 부위가 실제 외부가림인지 확정하지 않았습니다. 비슷한 인쇄 상자 여러 개가 함께 보이는 사례도 있습니다. object_id는 GT-side 메타데이터로만 명시합니다.

q8은 **GT-jitter emulation 입력**이며 pallet native8이나 실제 initializer 분포가 아닙니다. native mapping은 UNKNOWN입니다. 세 번째 사례의 aux4 좌표는 image 밖에 있지만 그대로 보존해 footer/JSON에 표시했고, 새 unavailable/semantic 라벨을 만들지 않았습니다. magenta r는 표시·진단만이며 모델 입력으로 추가하지 않았습니다.

이는 target 정보가 crop 밖에 있을 수 있다는 가설을 살피는 세 사례의 시각자료입니다. 전체 원인 빈도나 global context의 필요·충분성, 모델 성공을 입증하지 않습니다. 새 human/D/E/SELF 라벨·학습 admission·모델 추론·성능 판정은 없습니다. 원 GT/mask/depth·weights·prefix·current/held는 읽지 않았습니다.

- [전체 board](three_source_context_v1/THREE_SOURCE_CONTEXT_BOARD.png)
- [HTML gallery](three_source_context_v1/PORTABLE_CONTEXT_GALLERY_KO.html)
- [원 좌표와 원색 바인딩](three_source_context_v1/CASE_COORDINATES.json)
- [생성 완료](three_source_context_v1/COMPLETE.json)
- [실제 board view 영수증](ACTUAL_BOARD_VISUAL_QA.json)
