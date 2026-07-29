"""utils.py — Belief / Affinity map 생성 + 시각화 (NVIDIA DOPE 원본).

VisualizeAffinityMap : affinity tensor → RGB 시각화 (각도별 색깔)
VisualizeBeliefMap   : belief tensor → grayscale 시각화
GenerateMapAffinity  : keypoint → affinity field tensor 생성
getAfinityCenter     : 단일 affinity pair 생성 헬퍼
CreateBeliefMap      : keypoint → Gaussian belief map (학습 GT)
"""
import colorsys
import numpy as np
import torch
from PIL import Image, ImageDraw

from utils_math import length, normalize, py_ang


def VisualizeAffinityMap(
    tensor, threshold_norm_vector=0.4, points=None, factor=1.0, translation=(0, 0),
):
    """affinity tensor (2N x H x W) → RGB images (N x 3 x H x W).
    각도별로 hsv 색상 매핑. threshold 이하 vector 는 검정."""
    images = torch.zeros(tensor.shape[0] // 2, 3, tensor.shape[1], tensor.shape[2])
    for i_image in range(0, tensor.shape[0], 2):
        indices = (
            torch.abs(tensor[i_image, :, :]) + torch.abs(tensor[i_image + 1, :, :])
            > threshold_norm_vector
        ).nonzero()
        for indice in indices:
            i, j = indice
            angle_vector = np.array([tensor[i_image, i, j], tensor[i_image + 1, i, j]])
            if length(angle_vector) > threshold_norm_vector:
                angle = py_ang(angle_vector)
                c = colorsys.hsv_to_rgb(angle / 360, 1, 1)
            else:
                c = [0, 0, 0]
            for i_c in range(3):
                images[i_image // 2, i_c, i, j] = c[i_c]
        if not points is None:
            point = points[i_image // 2]
            images[
                i_image // 2, :,
                int(point[1] * factor + translation[1]) - 1
                : int(point[1] * factor + translation[1]) + 1,
                int(point[0] * factor + translation[0]) - 1
                : int(point[0] * factor + translation[0]) + 1,
            ] = 1
    return images


def VisualizeBeliefMap(tensor, points=None, factor=1.0, translation=(0, 0)):
    """belief tensor (N x H x W) → grayscale RGB (N x 3 x H x W). 채널별 minmax 정규화."""
    images = torch.zeros(tensor.shape[0], 3, tensor.shape[1], tensor.shape[2])
    for i_image in range(0, tensor.shape[0]):
        belief = tensor[i_image].clone()
        belief -= float(torch.min(belief).item())
        belief /= float(torch.max(belief).item())
        belief = torch.clamp(belief, 0, 1)
        belief = torch.cat(
            [belief.unsqueeze(0), belief.unsqueeze(0), belief.unsqueeze(0)]
        ).unsqueeze(0)
        images[i_image] = belief
    return images


def getAfinityCenter(
    width, height, point, center, radius=7, tensor=None, img_affinity=None,
):
    """단일 affinity pair (point → center 방향 unit vector) 생성."""
    if tensor is None:
        tensor = torch.zeros(2, height, width).float()
    imgAffinity = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(imgAffinity)
    r1 = radius
    p = point
    draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), (255, 255, 255))
    del draw
    array = (np.array(imgAffinity) / 255)[:, :, 0]
    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array * angle_vector[0]], [array * angle_vector[1]]])
    if not img_affinity is None:
        if length(angle_vector) > 0:
            angle = py_ang(angle_vector)
        else:
            angle = 0
        c = np.array(colorsys.hsv_to_rgb(angle / 360, 1, 1)) * 255
        draw = ImageDraw.Draw(img_affinity)
        draw.ellipse(
            (p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1),
            fill=(int(c[0]), int(c[1]), int(c[2])),
        )
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity


def GenerateMapAffinity(
    size, nb_vertex, pointsInterest, objects_centroid, scale, save=False,
):
    """8 corner × 2 (x, y) = 16 채널 affinity field tensor 생성 (학습 GT).
    각 corner → centroid 방향 unit vector."""
    img_affinity = Image.new("RGB", (int(size / scale), int(size / scale)), "black")
    affinities = [torch.zeros(2, int(size / scale), int(size / scale))
                  for _ in range(nb_vertex)]

    for i_pointsImage in range(len(pointsInterest)):
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            affinity_pair, img_affinity = getAfinityCenter(
                int(size / scale), int(size / scale),
                tuple((np.array(pointsImage[i_points]) / scale).tolist()),
                tuple((np.array(center) / scale).tolist()),
                img_affinity=img_affinity, radius=1,
            )
            affinities[i_points] = (affinities[i_points] + affinity_pair) / 2
            v = affinities[i_points].numpy()
            xvec, yvec = v[0], v[1]
            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0
            xvec[nonzero] /= norms[nonzero]
            yvec[nonzero] /= norms[nonzero]
            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec], [yvec]]))
    return torch.cat(affinities, 0)


def CreateBeliefMap(
    size,
    pointsBelief,
    nbpoints,
    sigma=16,
    save=False,
    clip_at_border=False,
):
    """keypoint 좌표 → Gaussian belief map 리스트 (nbpoints, size, size).
    각 채널 = 한 keypoint 의 Gaussian peak. sigma = 1 이하면 gradient vanishing.

    ``clip_at_border=False`` preserves the historical DOPE target exactly: if
    the full 4-sigma-wide support does not fit, the whole channel is empty.
    With ``clip_at_border=True``, a keypoint whose *centre is inside* the map is
    drawn and only the off-map Gaussian tail is clipped.  Far-outside points
    are still ignored; this avoids inventing a border target for an actually
    invisible corner.
    """
    beliefsImg = []
    for numb_point in range(nbpoints):
        array = np.zeros([size, size])
        for point in pointsBelief:
            p = [point[numb_point][1], point[numb_point][0]]
            w = int(sigma * 2)
            full_support_inside = (
                p[0] - w >= 0
                and p[0] + w < size
                and p[1] - w >= 0
                and p[1] + w < size
            )
            centre_inside = 0 <= p[0] < size and 0 <= p[1] < size
            if full_support_inside or (clip_at_border and centre_inside):
                i0 = max(0, int(p[0]) - w)
                i1 = min(size - 1, int(p[0]) + w)
                j0 = max(0, int(p[1]) - w)
                j1 = min(size - 1, int(p[1]) + w)
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        array[i, j] = max(
                            np.exp(-(((i - p[0]) ** 2 + (j - p[1]) ** 2)
                                     / (2 * (sigma ** 2)))),
                            array[i, j],
                        )
        beliefsImg.append(array.copy())
        if save:
            stack = np.stack([array, array, array], axis=0).transpose(2, 1, 0)
            imgBelief = Image.fromarray((stack * 255).astype("uint8"))
            imgBelief.save("debug/{}.png".format(numb_point))
    return beliefsImg


# --- spatial keypoint validity (PAPER_S2 A1 target semantics) ---------------
# The historical loss treats every GT belief channel as valid, so a keypoint
# that is legitimately outside the belief map is supervised as pure background
# negative, and a keypoint whose centre is inside but whose Gaussian tail is
# not gets an all-zero target under the same all-ones mask.  The mechanism
# diagnosis showed both cases coexisting in the training data, which makes the
# "no response" failure impossible to attribute.  This helper computes channel
# validity from the *transformed* keypoint position alone, so it can be shared
# by the GT path and by any pseudo-label path instead of overloading the
# pseudo-label-only mask function.
SENTINEL_COORDINATE = -90.0
MAX_REASONABLE_COORDINATE = 1.0e4


def spatial_keypoint_validity(points, size, strict=True):
    """Per-keypoint belief-channel validity for one object.

    Parameters
    ----------
    points:
        ``(9, 2)`` transformed keypoints in belief-grid coordinates, ordered
        ``(x, y)`` exactly as ``CreateBeliefMap`` consumes them.
    size:
        Belief map side length.
    strict:
        Raise on non-finite / absurd coordinates instead of silently masking
        them.  Silent clamping is what hid this defect originally.

    Returns
    -------
    numpy.ndarray
        ``(9,)`` float32 in {0.0, 1.0}.  1.0 means "this channel carries a real
        target and must contribute to the loss".  A keypoint whose centre lies
        inside the map is valid (its clipped Gaussian is a genuine positive);
        an exact sentinel or a legitimately off-map point is invalid, so it is
        neither drawn nor supervised as background.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (9, 2):
        raise ValueError(f"expected (9,2) keypoints, got {array.shape}")
    finite = np.isfinite(array).all(axis=1)
    sentinel = (array[:, 0] <= SENTINEL_COORDINATE) & (
        array[:, 1] <= SENTINEL_COORDINATE
    )
    reasonable = (np.abs(array) < MAX_REASONABLE_COORDINATE).all(axis=1)
    if strict and not np.all(finite & (reasonable | sentinel)):
        bad = np.where(~(finite & (reasonable | sentinel)))[0].tolist()
        raise ValueError(f"corrupt keypoint coordinates at indices {bad}")
    centre_inside = (
        (array[:, 0] >= 0.0)
        & (array[:, 0] < float(size))
        & (array[:, 1] >= 0.0)
        & (array[:, 1] < float(size))
    )
    valid = finite & reasonable & (~sentinel) & centre_inside
    return valid.astype(np.float32)
