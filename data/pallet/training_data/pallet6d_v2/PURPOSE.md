[소비처] 논문 §architecture — edge-line query → CIGM → direct PnP 로의 방향 전환 근거,
그리고 차기 predictor 의 정확도 목표치(ANGLE_BUDGET / OFFSET_BUDGET).

[문장] belief-space edge fusion 은 capacity 가 있으나(teacher-forced GT edge 로 R4 254→512),
Round1 은 noisy proposal 과의 co-adaptation 으로 그것을 잃었다. 그리고 GT edge →
CIGM → PnP interface 는 유효하므로, 다음 구조의 문제는 edge localization 하나로 좁혀진다 —
그 요구 정확도를 통제된 noise 주입으로 각도·오프셋 예산으로 정량화한다.

주의: F1/F2 와 direct-PnP 는 inference 에 GT edge 를 쓰는 capacity oracle 이며
deployable 후보가 아니다. validation512 는 이미 개봉된 post-validation diagnostic 용도로만
사용하고, untouched·eval56·wood45·final-test 는 열지 않는다.
