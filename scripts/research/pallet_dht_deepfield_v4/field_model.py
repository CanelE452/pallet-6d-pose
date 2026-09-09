"""Field-only DeepLSD using the unchanged official VGGUNet and exact heads.

The head/forward definitions below are adapted from official
deeplsd/models/deeplsd_inference.py at f7d9d6258c0cd25d4f6eea882853565403d289be.
No pytlsd, line detector, line-refinement extension, or installation is needed.

MIT License
Copyright (c) 2022 Rémi Pautrat
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

EXPECTED_COMMIT = 'f7d9d6258c0cd25d4f6eea882853565403d289be'
EXPECTED_CHECKPOINT_SHA = '56e9bae263977caa289ae49802987a67d4f00f376ae7e542176c8ae0a6cc2083'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def fp32_inference():
    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
                torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


class FieldOnlyDeepLSD(nn.Module):
    """Official MegaDepth checkpoint; grayscale [B,1,H,W] float32 in [0,1]."""
    def __init__(self, repository, checkpoint):
        super().__init__()
        self.repository, self.checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if sha(self.checkpoint) != EXPECTED_CHECKPOINT_SHA:
            raise ValueError('Expected the reviewed official MegaDepth checkpoint')
        backbone_file = self.repository/'deeplsd/models/backbones/vgg_unet.py'
        spec = importlib.util.spec_from_file_location('pallet_official_deeplsd_vgg_unet', backbone_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backbone = module.VGGUNet(tiny=False)
        self.df_head = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Conv2d(64,1,1), nn.ReLU())
        self.angle_head = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Conv2d(64,1,1), nn.Sigmoid())
        saved = torch.load(self.checkpoint, map_location='cpu')
        conf = saved['conf']['model']
        if conf['line_neighborhood'] != 5 or not conf['sharpen']:
            raise ValueError('Official sharpened five-pixel DF configuration differs')
        result = self.load_state_dict(saved['model'], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError('Official state keys differ')
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.provenance = dict(checkpoint=str(self.checkpoint), checkpoint_sha256=sha(self.checkpoint),
            source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(), backbone_file,
                self.repository/'deeplsd/models/deeplsd_inference.py', self.repository/'LICENSE']},
            checkpoint_state_entries=len(saved['model']), strict_state_load=True,
            parameters=sum(p.numel() for p in self.parameters()), pretraining='Official MegaDepth/MiniDepth generic-real-image pretrained model',
            detect_lines=False, multiscale=False, line_neighborhood=5,
            df_semantics='exp(-df_norm)*5 in input-image pixels, bounded above by5; not calibrated physical error or distance beyond this neighborhood',
            angle_semantics='Official line_level in radians modulo pi; direction of the nearest generic line, no pallet semantic role')

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or image.dtype != torch.float32:
            raise ValueError('Expected grayscale float32 [B,1,H,W]')
        if min(image.shape[-2:]) < 8 or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Finite [0,1] grayscale with H,W>=8 required')
        base = self.backbone(image)
        df_norm = self.df_head(base).squeeze(1)
        return {'df_norm':df_norm, 'df':torch.exp(-df_norm)*5,
                'line_level':self.angle_head(base).squeeze(1)*np.pi}

    def predict_canvas(self, image_gray, input_shape_hw):
        """B1 actual rectangle -> square fields; outside df=5 and valid=false."""
        h, w = map(int, input_shape_hw)
        if image_gray.shape != (1,1,640,640) or min(h,w) < 8 or max(h,w) > 640:
            raise ValueError('Expected B1 square gray plus actual rectangle')
        with fp32_inference():
            values = self(image_gray[:, :, :h, :w])
        df = torch.full_like(image_gray[:,0], 5.)
        angle = torch.zeros_like(df)
        df[:, :h, :w] = values['df']
        angle[:, :h, :w] = values['line_level']
        valid = torch.zeros_like(df, dtype=torch.bool)
        valid[:, :h, :w] = True
        return dict(df=df, line_level=angle, input_extent_mask=valid,
                    input_shape_hw=[h,w])
