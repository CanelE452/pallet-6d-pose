# 여기서 실제 확인한 것

- CPU 테스트 **54개 통과**, 실패0·오류0.
- 별도 생성 fixture에서 P/S/H/HA 각4회, 합계16회 실제 AdamW update를 실행했다.
- 네 군의 초기 tensor와 batch plan이 같고, 학습 후 상태는 서로 달랐다.
- checkpoint 재로딩, 후보 score 저장, 합성 margin 선택, 고정 정책 평가 경로를 실행했다.
- GT 파일을 실제로 지운 상태에서도 scoring이 성공하는 통합 테스트가 있다.
- 기본 실제 모델 구조의 등록 파라미터는 군당160,226개다. 비활성 입력 열에 관한 한계는 지시문과 코드에 명시했다.

이 값은 소프트웨어 검증 기록이다. 실제 팔레트 데이터·GPU·실사 평가·전체 네트워크 학습·canonical scorer 재현은 실행하지 않았다. 생성 fixture의 학습 결과를 팔레트 성능으로 해석하면 안 된다.

자세한 환경·현재 core SHA는 `LOCAL_VALIDATION.json`, 테스트 목록은 `pytest_results.xml`을 따른다.
