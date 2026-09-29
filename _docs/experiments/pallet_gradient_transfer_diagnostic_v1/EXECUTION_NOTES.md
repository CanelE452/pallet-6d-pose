# 실행 정정 기록

첫 시도는 모델 로딩 전 데이터 fixture 검사에서 중단했다. SHA 바인딩이 합성 RGB 심볼릭 링크를 실제 경로로 해석하면서 이미지 이름의 `syn__` alias가 사라졌고, 원래 alias를 가진 label 파일과 연결되지 않았다. 예상한 32개 객체 중 합성 16개가 background로 읽힌 것을 assertion으로 발견했다.

원본과 첫 fixture/cache는 보존했다. 선택한 이미지·label의 SHA와 seed를 바꾸지 않고, 새 `fixture_alias_corrected`에서 label 이름 기준으로 이미지 alias를 복원한다. 모델 gradient/좌표 결과를 보기 전의 구현 수정이며, 학습·checkpoint 쓰기·optimizer step은 0회다.

미세한 대칭 parameter perturbation 계산에는 matmul과 cuDNN TF32를 모두 끄고 FP32를 사용한다. 기존 학습 환경 전체를 변경하지 않으며, 이 진단의 수치 정밀도 설정만 다르다.

두 번째 시도는 32개 이미지/label 연결을 통과한 뒤 설치 패키지 코드의 SHA를 기록하는 단계에서 중단했다. 기존 공용 바인딩 함수는 저장소 내부 경로만 허용했으므로 새 namespace의 바인딩 함수에 외부 설치 파일의 절대 경로 기록을 추가했다. 모델 로딩·gradient 진단 전 오류이며, 기존 공용 코드는 수정하지 않았다.
