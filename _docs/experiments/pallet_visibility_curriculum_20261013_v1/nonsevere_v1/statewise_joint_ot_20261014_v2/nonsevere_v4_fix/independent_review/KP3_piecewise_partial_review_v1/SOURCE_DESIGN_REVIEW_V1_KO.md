새 unknown-split 영상 진단과 부분 법선 PnP 소스 검토는 통과했다. 사용자 표시점·raw U3·GT를 분할 목표로 쓰지 않고, 실제 B7/B2 관측으로 고른 열린 아래 윤곽에서 분할 위치를 찾는다. 열린 arc bootstrap이 양끝을 묶던 문제는 실행 전에 수정됐다. 같은 RGB/SAM 표본으로 한 선과 두 선을 비교하고 매 resample마다 분할을 다시 추정하며, 교점 후보 영역과1px 법선 감도 범위를 구분해서 표시한다.

부분 PnP는 U를 그대로 유지하며 완전한2D 관측점을 만들지 않는다. 선의 법선 residual에 soft loss를 적용하고, 양의 factor에서는 유한 선분 지원·U의 조건부 모델 가시성/화면 guard도 더해져 feasible set이 바뀐다는 점을 명시한다. 기존 BI 관측·공분산·seed·prior·개별 O/T constraint·native/Q·S 제외는 유지하며 zero/empty factor는 기존 solver 결과로 직접 돌아간다.

영상 선과 특정 native 모서리의 대응은 호출자가 세운 조건부 가정이다. 두 선이 거의 평행하거나 실제 윤곽이 둥글면 잘 맞는 두 선과 낮은 residual만으로 실제 KP3를 확정할 수 없다. 법선 분산과 서로 공유한 영상 표본의 상관은 별도 CAL 또는 가정 정책이 필요하다. actual image/K/native/Q 바인딩과 실제 출력 검토는 다음 단계이며 이번 소스 검토는 성능 성공 증거가 아니다.
