"""utils_pvnet.py — PVNet-style dense vector GT for the DOPE 2nd head.

새 dense 벡터 헤드(STEP 1) 전용 GT 생성. 기존 belief/affinity 와 동일한 50x50
격자에서, 객체 마스크 내 각 픽셀이 9개 키포인트(8 코너 + centroid) 를 가리키는
벡터맵 + 마스크를 키포인트로부터 즉석 계산한다 (렌더링 재생성 없음).

CleanVisiiDopeLoader 가 flag(pvnet_vec=True) 일 때만 호출한다. 기본 off 면 import
만 되고 호출되지 않아 기존 학습 경로는 불변.

좌표계: 입력 키포인트는 belief 격자(50px) 좌표여야 한다. 로더가 output_size=50
으로 albumentations Resize 한 직후의 키포인트가 정확히 이 좌표계다.

용어:
- offset 모드 (기본): v_k = kp_k - (x, y). 방향 + 크기 둘 다 학습.
- unit 모드: v_k 를 단위벡터로 정규화(방향만). PVNet 정통. RANSAC voting 과 결합.
"""
import numpy as np


def decode_mask_rle(rle):
    """uncompressed COCO RLE -> (H, W) uint8 {0,1}.

    column-major, 시작값 0 토글 (v3 합성셋 mask_rle 포맷). v3_overfit32 의 검증된
    디코드와 동일 구현. rle = {"size": [H, W], "counts": [...]}.
    """
    sz = rle["size"]                     # [H, W]
    flat = np.zeros(sz[0] * sz[1], np.uint8)
    idx = 0
    val = 0
    for c in rle["counts"]:
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((sz[1], sz[0])).T  # (H, W)


def make_cuboid_mask(kps9, size, faces=None):
    """projected cuboid 8 코너(+centroid 무시) 로 객체 마스크(size x size, uint8 0/1) 생성.

    8 코너의 convex hull 을 채운다. cuboid 6면 투영의 union 도 hull 과 동일 영역을
    덮으므로(코너가 전부 hull 정점) hull 로 충분. off-frame/음수 좌표 코너는 격자
    경계로 clip 후 hull 계산 → truncation 프레임에서도 견고.

    kps9: (9,2) belief 격자(size px) 좌표. 0~7 코너, 8 centroid.
    """
    import cv2  # local import: 마스크 생성 시에만 필요

    pts = np.asarray(kps9, dtype=np.float64)[:8].copy()  # centroid 제외, 8 코너만
    # 격자 경계로 clip (off-frame 코너 포함해도 hull 이 격자 안에서 닫히게)
    pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)

    mask = np.zeros((size, size), dtype=np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def make_vector_field(kps9, size, mask=None, unit=False):
    """객체 마스크 내 각 픽셀 → 9개 키포인트 방향 벡터맵 생성.

    반환:
      vec  : (18, size, size) float32. 채널 (2k, 2k+1) = 키포인트 k 의 (dx, dy).
             마스크 밖 = 0.
      mask : (size, size) uint8. mask 인자가 None 이면 cuboid hull 로 생성.

    offset (unit=False): v_k = kp_k - (x, y)  (방향 + 크기)
    unit   (unit=True) : v_k 를 L2 정규화 (방향만). 0벡터(픽셀==kp)는 0 유지.

    kps9: (9,2) belief 격자(size px) 좌표.
    """
    kps = np.asarray(kps9, dtype=np.float32)
    if mask is None:
        mask = make_cuboid_mask(kps, size)
    mask = mask.astype(bool)

    # 격자 픽셀 좌표 (x, y). meshgrid: xs[r,c]=c, ys[r,c]=r
    ys, xs = np.mgrid[0:size, 0:size]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    vec = np.zeros((18, size, size), dtype=np.float32)
    for k in range(9):
        dx = kps[k, 0] - xs  # (size, size)
        dy = kps[k, 1] - ys
        if unit:
            norm = np.sqrt(dx * dx + dy * dy)
            nz = norm > 1e-6
            dx = np.where(nz, dx / np.where(nz, norm, 1.0), 0.0)
            dy = np.where(nz, dy / np.where(nz, norm, 1.0), 0.0)
        vec[2 * k] = dx
        vec[2 * k + 1] = dy

    # 마스크 밖 0
    m = mask[None, :, :]
    vec = vec * m
    return vec.astype(np.float32), mask.astype(np.uint8)
