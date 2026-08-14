# paper_base — 논문용 base 모델 (명세 + 로드맵)

> **상태: 미학습 (다음 작업).** squash 비율 강건성을 포함해 새로 학습할 논문용 base.
> 작성일 2026-06-04. 결정: prefix=`paper_`, base=squash 포함 신규 학습 (사용자).

## 논문용 모델 명명 체계

```
paper_base   합성(camera-facing) + squash 비율강건 + truncation padding, v1/v2 제외
  └ paper_r1   기하 필터 self-training Round 1 PL finetune
      └ paper_r2 ...
```

- prefix `paper_` = 논문용 (일반화). 과제용 `challenge*` / `dope_cropaug_ft*` 와 명확히 구분.
- 자세한 트랙 구분: memory `two-tracks-paper-vs-challenge`, `_docs/` 방향 문서.

## 목적

처음 보는 파렛트(비율·외형 제각각)도 6D pose 를 잘 추론하는 **일반화** 모델.
내 실제 파렛트(v1/v2 = palletobj)는 학습에 쓰지 않는다 → 인터넷 무료 3D 팔레트
모델 기반 합성 데이터로만 학습.

## convention

**camera-facing 0123** (v4). 0~3 앞면, {0,1,4,5}=위 / {2,3,6,7}=아래, 8=centroid.
object-frame v8 은 폐기. memory `camera-facing-0123-convention` 참조.

## 학습 데이터 (경로 확정 2026-06-04)

```
합성 base   data/pallet/training_data/mixed_v8_train        9,000장
            (Isaac+Blender, 인터넷 무료 팔레트 모델 렌더)
            ✅ camera-facing v4 변환 적용 확인됨 (.json = camera-facing,
               .json.orig = object-frame 원본 7,205 백업). json≠orig 검증 완료.
+ truncation challenge/data/03_derived/truncation_crops_dope/pretrain  8,831장
            (mixed_v8 기반 crop+padding, camera-facing) — 잘린 이미지 강건성
            메커니즘: 9 kp 다 보이는 이미지를 crop 해 코너가 화면 밖으로 나간
            truncation 합성 → DOPE 로더가 padding 영역 확보 후 **화면 밖 코너의
            belief map(히트맵)을 padding 영역에 그려 supervise** (8/8 supervised
            검증). 잘려도 9점 회귀 → PnP 6점 안정 충족. (PnP 23→99%, det 13→94%,
            memory `dope-cropaug-truncation-success` / `yolo-padding-truncation-wins`)
+ squash    [미생성] 비율 강건성: 여러 aspect ratio 로 찌부(squash)/stretch 증강.
            ⚠️ 이미지 변형 시 JSON 꼭짓점(projected_cuboid)도 동기 변형 필수.
제외        challenge/data/02_synthetic/training/v1·v2 (내 실제 파렛트 palletobj) — 절대 미사용
```

- 위 base+truncation = `dope_cropaug_pretrain` 이 학습한 데이터 (논문 트랙 부합, squash만 빠짐).
- TODO: squash 증강 데이터 생성 스크립트 (변형+JSON 동기) → 3d-expert 검증.
- camera-facing v4 변환 logic: `challenge/scripts/convert_to_camera_facing_v4.py` (`compute_perm_v4`).

## 중간 산출물 (참고)

`weights/dope_cropaug_pretrain` (2026-06-02, scratch 60ep): mixed_v8(camera-facing)
+ truncation crop 8,831 학습. **truncation padding 은 적용됐으나 squash 비율강건은 없음.**
→ paper_base 의 전신/중간 산출물. squash 추가 후 재학습한 것이 정식 paper_base.

## 평가 (계획)

- self-training PL 필터링용 신뢰도: 2D 기하 필터 (PnP 불필요).
- 정확도: 치수 known GT(내 파렛트)에서 ADD/reproj — 단 논문 핵심은 일반화라
  처음 본 파렛트 정성/keypoint reproj 위주.
- **PnP solver = SQPnP** (`cv2.SOLVEPNP_SQPNP` + RefineLM). EPnP+RANSAC 대비
  reproj median 5.27→3.12px, ADD 96.6→90.7mm 개선(2026-06-02 YOLO 경로 검증).
  팔레트는 얇은 near-planar 직육면체라 globally optimal SQPnP 가 유리.
  현재 `scripts/self_training/pnp_solver.py` 는 EPnP+RANSAC → 평가/거리용 SQPnP 교체 필요.
- PnP 용도 분리: memory `camera-facing-0123-convention` 참조.

## 상태 체크리스트

- [ ] squash 비율강건 증강 데이터 생성 (JSON 꼭짓점 동기) — 3d-expert
- [ ] camera-facing v4 변환 정합성 최종 검증 — 3d-expert
- [ ] paper_base 학습 (camera-facing 합성 + squash + truncation padding)
- [ ] 2D 기하 필터 설계 (공간대각선 교점≈centroid 등)
- [ ] paper_r1 self-training (기하필터 PL)
