"""Capture the frozen paper YOLO neck once for the trainable line branch.

Standard Ultralytics owns detection and inverse letterboxing. Training images
already have the reflected border; real images receive it exactly once here.
The branch sees FP16-rounded neck features in training and inference alike.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
BASELINE_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


class FrozenYoloFeatures:
    def __init__(self, weights=BASELINE, device='cuda'):
        self.weights = Path(weights).resolve()
        if sha(self.weights) != BASELINE_SHA:
            raise ValueError('Only the frozen paper YOLO26n R0 is supported')
        self.device = device
        self.yolo = YOLO(str(self.weights), task='pose')
        self.yolo.model.requires_grad_(False).eval()
        self.capture = {}
        self.handles = [
            self.yolo.model.model[0].register_forward_pre_hook(self._input_hook),
            self.yolo.model.model[-1].register_forward_pre_hook(self._neck_hook),
        ]

    def _input_hook(self, module, inputs):
        self.capture['input_shape'] = tuple(inputs[0].shape[-2:])

    def _neck_hook(self, module, inputs):
        features = inputs[0]
        if len(features) != 3:
            raise ValueError('Expected three YOLO neck scales')
        self.capture['p3'] = features[0].detach()
        self.capture['p4'] = features[1].detach()

    @torch.no_grad()
    def predict(self, bgr, *, already_padded=False, cpu_features=False):
        self.capture.clear()
        pad = 0 if already_padded else 100
        canvas = bgr if already_padded else cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        result = self.yolo.predict(canvas, conf=.001, imgsz=640, rect=True,
                                   augment=False, half=False, device=self.device,
                                   verbose=False, save=False, stream=False)[0]
        if any(module.training for module in self.yolo.model.modules()):
            raise ValueError('The frozen detector entered training mode')
        if any(parameter.requires_grad for parameter in self.yolo.model.parameters()):
            raise ValueError('The detector must remain frozen')
        shape = self.capture['input_shape']
        if max(shape) != 640 or any(size % 32 for size in shape):
            raise ValueError(f'Unexpected canonical letterbox shape: {shape}')
        features = {}
        for name, stride, side in [('p3', 8, 80), ('p4', 16, 40)]:
            tensor = self.capture[name]
            expected = (shape[0]//stride, shape[1]//stride)
            if tuple(tensor.shape[-2:]) != expected or tensor.shape[0] != 1:
                raise ValueError('Neck feature shape does not match the input and stride')
            tensor = tensor.to(torch.float16)
            tensor = F.pad(tensor, (0, side-expected[1], 0, side-expected[0]))
            features[name] = tensor[0].cpu() if cpu_features else tensor
        candidates = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            points = result.keypoints.xy.detach().cpu().numpy()
            confidences = result.keypoints.conf.detach().cpu().numpy()
            if points.shape[1:] != (9, 2):
                raise ValueError('Expected nine camera-facing keypoints')
            for index in range(len(scores)):
                candidates.append(dict(candidate_index=index, score=float(scores[index]),
                    box_xyxy=(boxes[index]-pad).astype(np.float64),
                    keypoints_xy=(points[index]-pad).astype(np.float64),
                    keypoints_conf=confidences[index].astype(np.float64)))
        return dict(**features, input_shape=shape, canvas_shape=canvas.shape[:2],
                    added_border=pad, candidates=candidates,
                    selected_index=int(np.argmax([r['score'] for r in candidates])) if candidates else None)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.capture.clear()


def canvas_affine(canvas_shape, input_shape):
    """The label affine of canonical LetterBox, with its integer left/top pad.

    This maps already returned canvas-coordinate predictions into the branch's
    input-pixel frame. It does not attempt to undo clipping in YOLO postprocess.
    Final refinement adds the displacement/gain to the unchanged prediction.
    """
    h, w = canvas_shape
    ih, iw = input_shape
    gain = min(640/h, 640/w)
    resized_w, resized_h = round(w*gain), round(h*gain)
    left, top = round((iw-resized_w)/2-.1), round((ih-resized_h)/2-.1)
    if min(left, top) < 0:
        raise ValueError('Input shape is not a valid canonical letterbox canvas')
    return float(gain), np.array([left, top], dtype=np.float64)


def branch_inputs(captured, candidate_index=None):
    """Use confidence-selected predictions only; this function has no GT input."""
    index = captured['selected_index'] if candidate_index is None else candidate_index
    if index is None:
        return None
    candidate = captured['candidates'][index]
    gain, offset = canvas_affine(captured['canvas_shape'], captured['input_shape'])
    pad = captured['added_border']
    points = (candidate['keypoints_xy']+pad)*gain+offset
    box = ((candidate['box_xyxy']+pad).reshape(2, 2)*gain+offset).reshape(4)
    return dict(points=points.astype(np.float32), boxes=box.astype(np.float32),
                point_valid=np.isfinite(points).all(-1), input_shape=captured['input_shape'],
                gain=gain, candidate_index=index)


def restore_refinement(candidate, points_input, refined_input, gain, *, lam):
    original = np.asarray(candidate['keypoints_xy'], dtype=np.float64)
    if float(lam) == 0:
        return original.copy()
    delta = np.asarray(refined_input, np.float64)-np.asarray(points_input, np.float64)
    output = original+delta/gain
    output[8] = original[8]
    return output
