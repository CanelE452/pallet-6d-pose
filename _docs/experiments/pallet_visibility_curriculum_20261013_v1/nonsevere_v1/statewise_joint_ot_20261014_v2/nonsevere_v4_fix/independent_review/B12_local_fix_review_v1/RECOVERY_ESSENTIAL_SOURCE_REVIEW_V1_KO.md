새 복구 소스의 핵심 정적 검토는 통과했다. 기존에 보정된 B/I를 그대로 복사하고, 원래 fallback인 B만 실제 SAM 외곽선·RGB 전이와 두 경계선의 교점으로 복구한다. S는 점별 영상 탐색 전에 제외되며, count/fraction/span/RMS/방향/교점 조건과 경계 일관성 검사가 유지된다. 실제 실행·CAL 효과 검증은 별도이며 이 문서는 성공 증거가 아니다.

새 alpha1 관측의 bootstrap covariance는 선택된 두 경계선에 조건부인 분산이다. 기존 B:corrected 모델은 이동 제한과 blending을 쓴 관측으로 학습됐으므로, 새 복구 오차가 보정됐다고 주장할 수 없다. 기존 모델을 유지하면 fallback에서 corrected로 바뀌는 것만으로 floor와 reliability도 바뀐다. 같은 후보에 대한 cap/blend 비교와 기존 fitΣ/reliability를 유지한 좌표 개입으로 효과를 분리해야 한다.

pallet09 B1/B2의 2D 복구와 해당 사례의 큰 3D 오차는 따로 평가한다. 이 프레임은 고정된 과거 native/reference 대응에서 불일치 표시가 있지만, 그것만으로 현재 생산 native/Q가 틀렸다고 단정하지 않는다. CAL 검증으로 조건을 결정하고 이미 알려진 TEST는 사후 사례로 명시한다.
