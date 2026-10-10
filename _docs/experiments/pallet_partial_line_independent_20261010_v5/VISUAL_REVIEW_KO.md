# 저장 그림의 실제 확인

최종 `figures_repair1/01_all_operational.png`와 `figures_repair1/case_02.png`를 직접 열어 확인했다. 원본 실사에 출력 코너, 실제 채택 경계점, 숨은 점의 재투영, 남은 선의 support와 최종 inlier/outlier를 구분해 그렸다. 그림은 정답을 사용해 선택한 새 성공 사례가 아니라 이전에 고정한 동일 여섯 사례와 전체 분포다.

case_02의 고정 N3는 3.2691cm·1.8156°, 점 경로는 2.7472cm·1.0679°, 부분 선 경로는 2.8622cm·1.1061°다. 이 사례에서 두 새 경로는 N3보다 좋지만 부분 선 경로는 점 경로보다 나쁘다. 개별 그림이 전체 평균 개선의 증거가 되지 않는 것을 확인했다.

원행과 그림의 연결은 `FIGURE_CASE_ROWS_REPAIR1.jsonl.gz`, 여덟 PNG의 SHA는 `FIGURE_BINDINGS_REPAIR1.json`에 있다. 실패한 첫 renderer의 두 분포 PNG는 `figures/`에 보존했고, 최종 그림 여덟 개는 별도 폴더에 있다. 그림 생성 과정에서 새 모델·PnP·optimizer·학습·RGB 생성 호출은 없었다. 이 확인은 새 물리 경계 정답 어노테이션이나 전체 245장의 수동 판정을 만들지 않는다.
