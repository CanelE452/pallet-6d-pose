# 처음 읽을 내용

사용자는 **초기 추론 키포인트 중심 crop에서 대상 팔레트가 보이는지/가려졌는지 판별**하고 싶다. 실사 키포인트 visibility annotation이 이미 있으므로 평가에 활용한다. 현재 목적은 자세 개선이 아니다.

1. `HANDOFF_KO.md`를 읽는다. 지금까지 완료한 학습은 v60 ray-proxy head와 v61 pixel-presence head 각1회·480update다. 두 목표는 서로 다르며 Pro가 처음 설계한4모델 비교를 완료한 것이 아니다.
2. `PATH_INDEX.csv`에서 v61 보고서/체크포인트/실사 저장 출력, 사람 visibility sidecar와 고정 native mapping을 찾는다. 대용량 데이터와 모델은 새 폴더로 복사하지 않았다.
3. 실사223장 모두에 사람 주석 행이 연결되지만, 모든8코너 슬롯이 채워졌거나 native mapping이 모두 확인됐다는 뜻은 아니다. label inventory의 missing/unknown을 유지한다.
4. 다음 작업은 **기존 annotation 정의·순서 확인 → 같은 초기 query/crop 연결 → 동결 모델 classification 비교 → 실제 RGB 오류 분석**이다. 새 학습, 실사 적합, threshold 변경, PnP 최적화, seed 확대는 시작하지 않는다.

학습은 합성만 한다. 범용 frozen 사전학습은 허용한다. 실사 annotation은 평가만 한다. semantic visibility와 query-pixel presence의 차이·초기 좌표 오차를 별도로 집계한다.

현재 모든 학습/추론은 종료됐다. v62 저장출력 비교도 끝났지만 **새 human annotation 기반 confusion matrix는 아직 계산하지 않았다.** 새 프로젝트는 폴더/문서 준비이며 Codex 앱 등록은 미확인·미완료다. Fast 설정 파일이 저장됐던 사실과 현재 실제 Fast routing은 다르며 routing은 미확인이다.

기존 Pro 주대화: [실험 감사와 실패 원인 진단](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a). 접근할 수 없으면 읽었다고 하지 말고 HANDOFF의 전달된 요약을 사용한다.

사용자는 현재 화면의 **Ubuntu Codex GPT-6.1 Sol Ultra**와 대화하려는 것이다. Pro 연결·모델 전환을 진행하지 않는다. 앱에서 이 폴더를 열고 README의 시작 지시문을 붙여 넣으면 된다. 현재 연결에는 native 화면 제어와 공식 로컬 프로젝트 등록 도구가 없어 자동 등록을 완료하지 못했다. 이전 Pro 대화는 설계 참고자료일 뿐이다.
