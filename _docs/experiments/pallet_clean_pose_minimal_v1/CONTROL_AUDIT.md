# 동일 GEO 무학습 대조 독립 감사

**PASS**

- 8개 기존 모델 × D9/GEO = 16개 설정, 각 128장을 검사했다. 선택 후보 자세와 metric 행 2,048개가 일치했다.
- 같은 frozen GEO의 CPU 점수·이름 tie-break를 1,024프레임에서 다시 계산했다. 정답 오차를 선택에 사용하지 않았다.
- centroid 위치·C2 전체 회전·yaw 2,048개를 참조에서 독립 재계산했다.
- 전체 난도/촬영 기록 summary 192개, paired 비교 336개, recording 제외 비교 196개를 별도 구현으로 재집계했다.
- RAW/REF CLEAR seed43는 없는 것으로 유지했다. 기존 완료 namespace의 변경도 없었다.
- 누적 비용은 학생 6회, 1,920 update, GPU 학습 417.781초, 새 선택기 학습 0회 그대로다.
- 공개 대조 JSON에는 원본 RGB나 원 좌표 배열이 없다. 비공개 결과는 경로/해시만 연결했다.

## 감사 범위의 한계

- 별도 OS 수준 파일접근 감사를 했다는 뜻은 아니다. 저장된 Python audit read-path 기록과 고정 코드/해시/시각을 검증했다.
- IoU3D 및 ADD 개별 프레임 원식은 다시 풀지 않았다. 선택 후보의 기존 전체 metric 행 일치와 AUC/IoU 집계를 검증했다.
- 두 seed OCC만 존재한다. CLEAR seed43와 독립 TEST를 만들어낸 것이 아니다.
- 자연99는 프레임 오차를 직접 pooling했다. 난도 중앙값 평균이 아니다.

[상세 JSON](CONTROL_AUDIT.json)
