# 입력 감사

[확인] HEAD `ceb323452a77ed8cebb152f6ab2626d018bf7e38`, main. 새 학습/추론 없이 기존 frozen prediction만 사용했다. 전체 입력 SHA와 model checkpoint provenance는 [INPUTS](INPUTS.json)에 있다.

[확인] 모든 7개 후보의 동일 93-frame 집합, 동일 713 canonical GT 코너를 join했다. 651개 frame×candidate를 기존 metric과 다시 계산해 대조했다. whole-object 승인 대칭 한 분기만 사용했다. A10/A11 10px 정답 356/350, 차이 6개.

[확인] E2_FRAME_METRICS는 문서 폴더가 아닌 `data/pallet/results/pallet_occlusion_refiner_transfer_v2/E2_FRAME_METRICS.json`에 있다. A10/A11 freeze lock, 원본 model SHA, 현재 annotation SHA를 검사했다.

[확인] 매칭 실패 프레임 8개, 그 유효 코너 54개를 삭제하지 않고 원래 800px 벌점으로 유지했다. 저장 좌표의 기하 거리와 채점 벌점은 CSV에서 별도 필드다.

[확인] 좌표 출처는 `{'unknown': 713}`이다. GT object의 manual 표기를 코너별 manual 좌표 증명으로 사용하지 않았다. 비초록 legacy reference이며 기존 DEV 재사용이다.

[확인] 가시성은 기존 human amendment와 auto queue를 추적했다. annotation reason만으로 external/self를 구분하지 않는다. 사람의 generic occluded는 UNKNOWN subtype으로 남기며 auto self/visible은 proxy로 별도 집계한다. 수동 검토된 가시성과 좌표 provenance는 서로 다른 속성이다.

[확인] 기존 E2 decision/report 및 checkpoint는 수정하지 않았다. source clean/stress 결과도 기존 파일에서 읽었으며 새 평가를 선택해 과거 PARTIAL을 변경하지 않는다.
