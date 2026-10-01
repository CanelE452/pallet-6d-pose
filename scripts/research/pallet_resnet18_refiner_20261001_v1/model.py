# Architecture adapted from Microsoft Simple Baselines for Human Pose Estimation
# https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
# Written there by Bin Xiao. Original license:
#
# MIT License
# Copyright (c) Microsoft Corporation. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""RGB-only ResNet18 SimpleBaseline adaptation for nine pallet heatmaps.

Full-image use and nine pallet roles are adaptations of the original top-down
human estimator. This is neither PVNet nor PoseFix. No detector, crop, camera,
dimensions, target, checkpoint file or network download is used by forward.
The caller owns preprocessing, target construction and baseline/refiner training.

Backbone: torchvision's ResNet18 body (BasicBlock [2,2,2,2]), excluding its pool
and classifier. Head: three 256-channel k4/s2/p1 transposed convolutions, each
with BN(momentum=.1)/ReLU, then a biased 1x1 nine-channel heatmap convolution.
The head initialization follows the official pretrained SimpleBaseline branch:
normal std=.001 convolution weights, BN scale1/bias0, final-conv bias0.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torchvision.models import resnet18


ARCHITECTURE = 'SimpleBaseline-derived ResNet18 full-image pallet9'
FEATURE_CHANNELS = (128, 256)
FEATURE_STRIDES = (8, 16)
FEATURE_TAPS = ('layer2', 'layer3')
INPUT_SHAPE_HW = (384, 512)
OUTPUT_SHAPE_HW = (96, 128)
HEATMAP_STRIDE = 4
BODY_PREFIXES = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')


class SimpleBaselineResNet18(nn.Module):
    def __init__(self, num_keypoints=9, pretrained_state=None):
        super().__init__()
        if not isinstance(num_keypoints, int) or isinstance(num_keypoints, bool) or num_keypoints <= 0:
            raise ValueError('num_keypoints must be a positive integer')
        self.num_keypoints = num_keypoints
        # Explicit weights=None: construction NEVER contacts a model hub.
        backbone = resnet18(weights=None)
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(self, name, getattr(backbone, name))
        del backbone
        layers = []
        channels = 512
        for _ in range(3):
            layers.extend([
                nn.ConvTranspose2d(channels, 256, kernel_size=4, stride=2,
                                   padding=1, output_padding=0, bias=False),
                nn.BatchNorm2d(256, momentum=.1),
                nn.ReLU(inplace=True),
            ])
            channels = 256
        self.deconv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0)
        self._initialize_head()
        self.pretrained_loaded = False
        self.pretrained_mapping = None
        if pretrained_state is not None:
            self.load_imagenet_state(pretrained_state)

    def _initialize_head(self):
        for module in self.deconv_layers.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.final_layer.weight, std=.001)
        nn.init.zeros_(self.final_layer.bias)

    def load_imagenet_state(self, state):
        """Load a supplied state mapping, requiring every body tensor exactly.

        Only the optional pair fc.weight/fc.bias may be excluded. The official
        legacy ImageNet file predates all 20 BN num_batches_tracked counters:
        if and only if all are absent, initialize those integer counters to zero
        explicitly, matching torchvision's legacy BN migration. Learned tensors
        and running means/variances may never be missing. No arbitrary prefix
        removal, partial loads or file fallback is accepted. All validation
        precedes mutation of model state.
        """
        if not isinstance(state, Mapping) or any(not isinstance(k, str) for k in state):
            raise TypeError('Expected a plain torchvision ResNet18 state mapping')
        current = self.state_dict()
        body = {k: v for k, v in current.items() if k.startswith(BODY_PREFIXES)}
        incoming = set(state)
        classifier = {'fc.weight', 'fc.bias'}
        counters = {k for k in body if k.endswith('.num_batches_tracked')}
        legacy_counters = counters if not (incoming & counters) else set()
        missing = set(body) - incoming
        extra = incoming - set(body) - classifier
        classifier_keys = incoming & classifier
        if missing != legacy_counters or extra or classifier_keys not in (set(), classifier):
            raise ValueError(f'ImageNet key mismatch: missing={sorted(missing)}, '
                             f'extra={sorted(extra)}, classifier={sorted(classifier_keys)}')
        expected = {k: (v.shape, v.dtype) for k, v in body.items() if k not in legacy_counters}
        if classifier_keys:
            expected.update({'fc.weight': (torch.Size([1000, 512]), torch.float32),
                             'fc.bias': (torch.Size([1000]), torch.float32)})
        for key, (shape, dtype) in expected.items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f'ImageNet tensor shape/dtype mismatch: {key}')
            if value.is_meta or (value.is_floating_point() and not torch.isfinite(value).all()):
                raise ValueError(f'ImageNet tensor missing/nonfinite values: {key}')
        # Strict full-model load keeps the independently initialized head intact.
        combined = dict(current)
        combined.update({k: (torch.zeros_like(body[k]) if k in legacy_counters else state[k])
                         for k in body})
        self.load_state_dict(combined, strict=True)
        self.pretrained_loaded = True
        self.pretrained_mapping = dict(body_tensors=len(body),
            loaded_body_tensors=len(body) - len(legacy_counters),
            legacy_bn_counters_initialized_to_zero=sorted(legacy_counters),
            excluded_classifier_keys=sorted(classifier_keys),
            missing_keys=[], unexpected_keys=[], head_preserved=True)
        return dict(self.pretrained_mapping)

    @staticmethod
    def _validate_input(images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected RGB tensor [B,3,H,W]')
        if images.shape[0] < 1 or any(d < 32 or d % 32 for d in images.shape[-2:]):
            raise ValueError('Positive batch and H/W >=32 divisible by32 are required')
        if not images.is_floating_point():
            raise ValueError('Preprocessing must supply floating-point RGB')

    def _prefix(self, images):
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        x = self.layer1(x)
        feature8 = self.layer2(x)
        feature16 = self.layer3(feature8)
        return feature8, feature16

    def forward(self, images, return_features=False):
        self._validate_input(images)
        feature8, feature16 = self._prefix(images)
        x = self.layer4(feature16)
        heatmaps = self.final_layer(self.deconv_layers(x))
        if return_features:
            # Keep gradients during baseline training. Frozen inference is the
            # caller's no_grad/eval responsibility; no hidden detach in forward.
            return dict(heatmaps=heatmaps, features=(feature8, feature16))
        return heatmaps

    @torch.no_grad()
    def features_only(self, images):
        """Frozen-prefix features for new P/D training; no layer4/heatmap work."""
        if self.training:
            raise RuntimeError('features_only requires eval() to keep BN statistics fixed')
        self._validate_input(images)
        return tuple(value.detach() for value in self._prefix(images))
