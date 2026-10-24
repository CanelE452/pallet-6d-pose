# V188 공개 SOURCE 끝부위 관찰 및 제어 합성

고정 22사례 전체를 원 순서로 보존한 표시용 갤러리입니다. 왼쪽은 기존 224 RGB 확대, 오른쪽은 같은 이미지에 별도로 그린 끝 후보(빨강)와 같은 부재의 주변 재료 witness(노랑)입니다. 주황은 원 crop 밖의 비국소 UNKNOWN 후보입니다. 원 q/L/native와 원본 입력은 변경하지 않았습니다.

관찰은 assistant 약한 RGB 의미 관찰이며 사람 정답·픽셀 마스크·정확한 물리 코너 좌표가 아닙니다. 16건 finite 관찰 / 6건 UNKNOWN을 모두 남겼습니다. 원 66시도 중 실제 5쌍만 합성 증명 통과, 기하 실패 10건, UNKNOWN 51건입니다. 33시도의 미지원은 crop 범위 밖이 아니라 다운샘플링 raw-tap 공백입니다. Q0057는 별도의 실제 비국소 사례입니다.

실제 QA는 아래 22사례를 담은 contact 11장, 합성 CONTROL 5장, 고정된 첫 기하 실패 plan 4장을 tools.view_image로 확인했습니다. 수치 검산은 전체 66/198 상태와 생성된 25 packet(200 native 입력행, U6)을 별도로 확인했습니다. HTML 브라우저 렌더는 주장하지 않습니다.

- [독립 수치 감사](REVIEW.json)
- [실제 시각 QA 및 한계](ACTUAL_VISUAL_QA.json)
- [모든 이미지 경로·SHA 및 표시 manifest](SOURCE22_DISPLAY_MANIFEST.json)
- [원 66시도 전체와 실패 이유](../../real_end_local_pairs_v1/index.html)

Q1088의 witness는 같은 앞 상판의 넓은 윗면 내부로 보입니다. Q1624는 얇은 상판의 면 경계 가까운 흐린 재료이므로 약한 부재 개연성만 인정합니다. Q1608는 plank/block 접합 인접성 우려를 별도로 보존하며 세 시도 모두 이미 기하 실패입니다. 이 주석으로 동결 관찰이나 학습 허용을 바꾸지 않았습니다.

## 고정 22사례: clean / marked

### 01. Q1608

![Q1608 clean and marked source end](source22_annotation/Q1608_SOURCE_ENDPOINT_ANNOTATION.png)

### 02. Q1088

![Q1088 clean and marked source end](source22_annotation/Q1088_SOURCE_ENDPOINT_ANNOTATION.png)

### 03. Q1624

![Q1624 clean and marked source end](source22_annotation/Q1624_SOURCE_ENDPOINT_ANNOTATION.png)

### 04. Q0024

![Q0024 clean and marked source end](source22_annotation/Q0024_SOURCE_ENDPOINT_ANNOTATION.png)

### 05. Q1728

![Q1728 clean and marked source end](source22_annotation/Q1728_SOURCE_ENDPOINT_ANNOTATION.png)

### 06. Q0337

![Q0337 clean and marked source end](source22_annotation/Q0337_SOURCE_ENDPOINT_ANNOTATION.png)

### 07. Q0033

![Q0033 clean and marked source end](source22_annotation/Q0033_SOURCE_ENDPOINT_ANNOTATION.png)

### 08. Q0041

![Q0041 clean and marked source end](source22_annotation/Q0041_SOURCE_ENDPOINT_ANNOTATION.png)

### 09. Q0281

![Q0281 clean and marked source end](source22_annotation/Q0281_SOURCE_ENDPOINT_ANNOTATION.png)

### 10. Q0057

![Q0057 clean and marked source end](source22_annotation/Q0057_SOURCE_ENDPOINT_ANNOTATION.png)

### 11. Q0314

![Q0314 clean and marked source end](source22_annotation/Q0314_SOURCE_ENDPOINT_ANNOTATION.png)

### 12. Q0266

![Q0266 clean and marked source end](source22_annotation/Q0266_SOURCE_ENDPOINT_ANNOTATION.png)

### 13. Q0315

![Q0315 clean and marked source end](source22_annotation/Q0315_SOURCE_ENDPOINT_ANNOTATION.png)

### 14. Q0052

![Q0052 clean and marked source end](source22_annotation/Q0052_SOURCE_ENDPOINT_ANNOTATION.png)

### 15. Q0300

![Q0300 clean and marked source end](source22_annotation/Q0300_SOURCE_ENDPOINT_ANNOTATION.png)

### 16. Q0277

![Q0277 clean and marked source end](source22_annotation/Q0277_SOURCE_ENDPOINT_ANNOTATION.png)

### 17. Q0381

![Q0381 clean and marked source end](source22_annotation/Q0381_SOURCE_ENDPOINT_ANNOTATION.png)

### 18. Q1709

![Q1709 clean and marked source end](source22_annotation/Q1709_SOURCE_ENDPOINT_ANNOTATION.png)

### 19. Q0094

![Q0094 clean and marked source end](source22_annotation/Q0094_SOURCE_ENDPOINT_ANNOTATION.png)

### 20. Q0102

![Q0102 clean and marked source end](source22_annotation/Q0102_SOURCE_ENDPOINT_ANNOTATION.png)

### 21. Q0389

![Q0389 clean and marked source end](source22_annotation/Q0389_SOURCE_ENDPOINT_ANNOTATION.png)

### 22. Q0245

![Q0245 clean and marked source end](source22_annotation/Q0245_SOURCE_ENDPOINT_ANNOTATION.png)

## 실제 합성 증명 통과 5쌍

각 그림은 BEFORE / COVER / ADJACENT_SHAM입니다. 같은 foreign RGB/alpha를 고정 이동한 2D 개입이며 native D/E/S나 quality 정답으로 승격하지 않았습니다. 다른 상판 재료가 COVER에서도 보일 수 있습니다. 인공 색·윤곽 단서는 자연 실사 전이의 증거가 아닙니다.

### Q1088_D2

![Q1088_D2 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1088_D2/CONTACT.png)

### Q1088_D3

![Q1088_D3 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1088_D3/CONTACT.png)

### Q1624_D1

![Q1624_D1 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D1/CONTACT.png)

### Q1624_D2

![Q1624_D2 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D2/CONTACT.png)

### Q1624_D3

![Q1624_D3 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D3/CONTACT.png)

## 기하 실패: 사례별 원 순서 첫 제안

아래는 합성 전 고정 placement 사각형 표시입니다. 실패 뒤 위치·크기·donor를 교체하지 않았습니다.

### Q1608_D1

![Q1608_D1 plan only](geometry_failure_annotation/Q1608_D1_PLAN_ONLY.png)

### Q1088_D1

![Q1088_D1 plan only](geometry_failure_annotation/Q1088_D1_PLAN_ONLY.png)

### Q1728_D1

![Q1728_D1 plan only](geometry_failure_annotation/Q1728_D1_PLAN_ONLY.png)

### Q1709_D1

![Q1709_D1 plan only](geometry_failure_annotation/Q1709_D1_PLAN_ONLY.png)

