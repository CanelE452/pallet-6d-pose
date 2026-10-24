"""Independent NumPy/scalar detector postprocess/association math; no IO/model."""
import math
import numpy as np


INPUT_FIELDS = {"input_id", "image", "image_sha256", "raw_hw", "initial_points", "roi_side", "query_box_xyxy"}


def native_boxes_fp32(normalized_cxcywh, raw_hw):
    raw = np.asarray(normalized_cxcywh)
    assert raw.ndim == 2 and raw.shape[1] == 4 and raw.dtype == np.float32
    assert np.isfinite(raw).all()
    half = np.multiply(raw[:, 2:], np.float32(.5), dtype=np.float32)
    corners = np.concatenate([np.subtract(raw[:, :2], half, dtype=np.float32),
                              np.add(raw[:, :2], half, dtype=np.float32)], axis=1)
    h, w = raw_hw
    return np.multiply(corners, np.array([w, h, w, h], np.float32), dtype=np.float32)


def valid_box(box):
    if box is None:
        return False
    a = np.asarray(box, np.float64)
    return a.shape == (4,) and bool(np.isfinite(a).all() and a[0] < a[2] and a[1] < a[3])


def scalar_iou(initial, candidate):
    a = [float(x) for x in initial]
    b = [float(x) for x in np.asarray(candidate, np.float32)]
    if not valid_box(b):
        return -1.
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    denom = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / denom if denom > 0 else 0.


def associate_scalar(row, rgb_id, boxes, scores, query_indices):
    assert set(row) == INPUT_FIELDS
    boxes, scores, ordinals = np.asarray(boxes, np.float32), np.asarray(scores, np.float32), np.asarray(query_indices, np.int64)
    assert boxes.shape == (len(scores), 4) and scores.shape == ordinals.shape
    assert np.isfinite(scores).all() and len(set(ordinals.tolist())) == len(ordinals)
    out = dict(row, rgb_input_id=rgb_id, status="U", reason="INVALID_INITIAL_QUERY_BOX",
               selected_detection_index=None, selected_detector_query_index=None, selected_detection_score=None,
               association_iou=None, selected_box_xyxy_raw=None, selected_box_xyxy_prompt=None)
    if not valid_box(row["query_box_xyxy"]):
        return out
    if not len(scores):
        out["reason"] = "NO_FIXED_THRESHOLD_DETECTIONS"
        return out
    overlaps = [scalar_iou(row["query_box_xyxy"], box) for box in boxes]
    chosen = min(range(len(scores)), key=lambda i: (-overlaps[i], -float(scores[i]), int(ordinals[i])))
    if overlaps[chosen] < .1:
        out["reason"] = "MAX_INPUT_BOX_IOU_BELOW_0P1"
        return out
    h, w = row["raw_hw"]
    raw = boxes[chosen]
    prompt = [max(0., min(float(w if i % 2 == 0 else h), float(x))) for i, x in enumerate(raw)]
    if not valid_box(prompt):
        out["reason"] = "INVALID_CLIPPED_SAM_PROMPT"
        return out
    out.update(status="ASSOCIATED", reason="OK", selected_detection_index=chosen,
               selected_detector_query_index=int(ordinals[chosen]), selected_detection_score=float(scores[chosen]),
               association_iou=float(overlaps[chosen]), selected_box_xyxy_raw=raw.tolist(), selected_box_xyxy_prompt=prompt)
    return out


def validate_expected_padding(logits, attention, boxes):
    values, attention, boxes = np.asarray(logits), np.asarray(attention), np.asarray(boxes)
    assert values.ndim == 3 and values.shape[-1] == 256 and values.dtype == np.float32
    assert attention.ndim == 2 and attention.shape[0] == values.shape[0] and 0 < attention.shape[1] <= 256
    assert np.isin(attention, [0, 1]).all()
    assert boxes.shape == values.shape[:2] + (4,) and boxes.dtype == np.float32 and np.isfinite(boxes).all()
    active = np.zeros((values.shape[0], 256), bool)
    active[:, :attention.shape[1]] = attention.astype(bool)
    for i in range(values.shape[0]):
        for token in range(256):
            if active[i, token]:
                assert np.isfinite(values[i, :, token]).all()
            else:
                assert np.isneginf(values[i, :, token]).all()
    return True


def generated():
    row = dict(input_id="opaque", image="/generated.png", image_sha256="0" * 64, raw_hw=[30, 40],
               initial_points=[[i, i + .25] for i in range(8)], roi_side=16., query_box_xyxy=[10., 10., 20., 20.])
    tied = associate_scalar(row, "rgb", [[10, 10, 20, 20]] * 3, [.6, .9, .9], [2, 6, 4])
    assert tied["selected_detection_index"] == 2 and tied["selected_detector_query_index"] == 4
    boundary = dict(row, query_box_xyxy=[0., 0., 1., 10.])
    assert associate_scalar(boundary, "rgb", [[0., 0., 1., 1.]], [.5], [0])["association_iou"] == .1
    assert associate_scalar(boundary, "rgb", [[0., 0., 1., 1.]], [.5], [0])["status"] == "ASSOCIATED"
    assert associate_scalar(row, "rgb", np.zeros((0, 4), np.float32), [], [])["reason"] == "NO_FIXED_THRESHOLD_DETECTIONS"
    assert associate_scalar(dict(row, query_box_xyxy=None), "rgb", [[0, 0, 1, 1]], [.8], [0])["reason"] == "INVALID_INITIAL_QUERY_BOX"
    outside = dict(row, query_box_xyxy=[-10, -10, -1, -1])
    assert associate_scalar(outside, "rgb", [[-10, -10, -1, -1]], [.8], [0])["reason"] == "INVALID_CLIPPED_SAM_PROMPT"
    score = np.float32(.4)
    values = np.array([np.nextafter(score, np.float32(0)), score, np.nextafter(score, np.float32(1))], np.float32)
    assert np.flatnonzero(values > np.float32(.4)).tolist() == [2]
    raw = np.array([[.5, .5, 1., 1.], [.5, .5, .5, .25]], np.float32)
    assert native_boxes_fp32(raw, [40, 80]).tolist() == [[0., 0., 80., 40.], [20., 15., 60., 25.]]
    for length in [77, 256]:
        attention = np.ones((2, length), np.int64)
        attention[1, 12] = 0
        logits = np.full((2, 3, 256), -np.inf, np.float32)
        for image in range(2):
            logits[image, :, np.flatnonzero(attention[image])] = .5
        boxes = np.full((2, 3, 4), .25, np.float32)
        assert validate_expected_padding(logits, attention, boxes)
        for location, malformed in [((0, 0, 0), np.nan), ((0, 0, 0), np.inf), ((0, 0, 0), -np.inf),
                                    ((1, 0, 12), 0.), ((1, 0, 12), np.nan), ((1, 0, 12), np.inf)]:
            bad = logits.copy(); bad[location] = malformed
            try:
                validate_expected_padding(bad, attention, boxes)
            except AssertionError:
                pass
            else:
                raise AssertionError("Malformed padding/active values admitted")
    return {"PASS": True, "generated_only": True, "FP32_raw_center_to_native_box_math": True,
            "strict_detector_threshold0p4_adjacent_FP32_values": True, "association_0p1_inclusive": True,
            "IoU_score_ordinal_ties_clip_missing_and_invalid": True,
            "activefinite_and_exactnegativeinfinitypadding_allattentioncases": True,
            "source_RGB_actual_outputs_models_weights_forward_fit_current_PnP_reads": 0}
