# Real Data — Split 정의 및 촬영 프로토콜

## 디렉토리 구조

```
real_data/
├── real_test_seen/      ← AprilTag GT, 학습에 사용한 팔레트 (50~100장, benchmark 고정)
│   ├── seen_001.jpg
│   ├── seen_001.json    ← GT annotation (NDDS 호환)
│   └── ...
├── real_test_unseen/    ← AprilTag GT, 학습에 없는 팔레트 (50~100장, benchmark 고정)
│   ├── unseen_001.jpg
│   ├── unseen_001.json
│   └── ...
├── real_unlabeled/      ← GT 없음, self-training용 (500~1000장)
│   ├── unlabeled_001.jpg
│   └── ...
├── real_dev/            ← 파일럿/디버깅용 (10~20장)
├── qualitative_panel/   ← 모델 비교용 고정 패널 (20~30장, symlink 또는 복사)
├── metadata.csv         ← 전체 메타데이터
└── README.md
```

## Split 규칙

| Split | 목적 | 크기 | GT | 절대 불변 |
|-------|------|------|-----|----------|
| real_test_seen | 정량 평가 (Seen pallet) | 50~100장 | AprilTag | Yes |
| real_test_unseen | 일반화 평가 (Unseen pallet) | 50~100장 | AprilTag | Yes |
| real_unlabeled | Self-training pool | 500~1000장 | 없음 | No |
| real_dev | 파일럿/디버깅 | 10~20장 | AprilTag | No |
| qualitative_panel | 모델 비교 시각화 | 20~30장 | 선택적 | Yes |

**중요**: `real_test_seen`과 `real_test_unseen`은 한 번 고정하면 절대 변경하지 않는다. 모든 실험은 동일한 benchmark에서 비교해야 한다.

## 촬영 프로토콜

### 조건 분포 (Seen/Unseen 각각)

| 변수 | 값 | 최소 비율 |
|------|-----|----------|
| **거리** | near (<2m) / mid (2-5m) / far (>5m) | 각 20% 이상 |
| **각도** | frontal / diagonal / side | 각 20% 이상 |
| **시점** | low (지게차) / mid / high | low 50%+ |
| **가림** | none / light / heavy | none 50%+ |
| **적재** | empty / loaded | 각 30% 이상 |
| **조명** | bright / normal / dark | normal 50%+ |

### Tag-on / Tag-off 프로토콜

1. 팔레트에 AprilTag 부착 (고정 위치, rigid transform 측정 완료)
2. **tag-on** 촬영 → GT pose 생성용
3. AprilTag 제거 (또는 가림)
4. **tag-off** 촬영 → 모델 추론용
5. 두 이미지를 쌍으로 저장 (`seen_001_tag.jpg` / `seen_001.jpg`)

### AprilTag 설정

- Tag family: `tag36h11` (권장)
- Tag size: 150mm (측정 후 `metadata.csv`에 기록)
- 부착 위치: 팔레트 상면 중앙 (tag↔pallet rigid transform 고정)
- 카메라: RealSense D435i (intrinsics: fx=615, fy=615, cx=320, cy=240)

## GT Annotation 포맷

Synthetic annotation과 호환되는 NDDS 형식:

```json
{
  "camera_data": {
    "width": 640, "height": 480,
    "intrinsics": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
  },
  "objects": [{
    "class": "pallet",
    "name": "real_pallet",
    "pose_transform": [[...], [...], [...], [0,0,0,1]],
    "projected_cuboid": [[x,y], ...],
    "projected_cuboid_centroid": [x, y],
    "gt_source": "apriltag",
    "tag_id": "tag36h11_0",
    "tag_reproj_error": 0.5
  }]
}
```

## 평가 메트릭 (Real Test)

| 메트릭 | 설명 | 스크립트 |
|--------|------|---------|
| ADD | Average Distance of Model Points (<0.1d) | `scripts/data_prep/evaluate_real.py` |
| 5cm 5° | Translation <5cm AND Rotation <5° | `scripts/data_prep/evaluate_real.py` |
| Reproj Error | 2D 재투영 오차 (px) | `scripts/data_prep/evaluate_real.py` |
| 3D Volume Ratio | 추정 부피 / GT 부피 | `scripts/data_prep/evaluate_real.py` |
