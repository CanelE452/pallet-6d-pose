"""Reversible actual-call counters; no extra forward and no numerical changes.

Hooks are installed when each frozen lazy loader actually returns its model.
Both timing paths use the same counters. Counts describe executed calls, not
counts copied from metadata. This context is process-local and non-nestable.
"""
from __future__ import annotations

import builtins
from collections import Counter
from functools import wraps
import io
import os
from pathlib import Path
import sys

from . import runtime as RT
from scripts.research.pallet_visibility_curriculum_20261013_v1 import segmentation as SEG
from scripts.research.pallet_visibility_curriculum_20261013_v1 import state_visibility_v2 as STATE


_ACTIVE = False
_KEYS = ('observe_calls', 'solve_calls', 'joint_solver_calls', 'state_inference_calls',
    'state_classifier_head_calls', 'state_feature_extractor_calls',
    'state_backbone_forward_calls', 'tiny_segment_calls', 'tiny_forward_calls',
    'sam_set_image_calls', 'sam_predict_calls', 'sam_image_encoder_forward_calls',
    'sam_mask_decoder_forward_calls', 'bi_correction_calls',
    'boundary_recovery_calls', 'n3_forward_calls', 'subpix_calls',
    'OT_builder_calls', 'U_display_builder_calls',
    'blocked_asset_accesses', 'blocked_imports')
_FORBIDDEN_MODULES = ('pallet_n3', 'n3_subpix', 'n3_dim_sym',
    'ot_domains_v2', 'uncertain_u_regions_v4_5', 'evaluate_u_regions_v4_5')
_FORBIDDEN_ASSET_TOKENS = ('pallet_n3', 'n3_dim_sym', 'n3_subpix',
    'n3_features', '/n3_', 'subpix')


class CallMeter:
    def __init__(self, components=None, *, block_unused=True, forbidden_asset_roots=()):
        self.components = components
        self.block_unused = block_unused
        self.forbidden_asset_roots = tuple(Path(p).absolute() for p in forbidden_asset_roots)
        self.counts = Counter({key: 0 for key in _KEYS})
        self._patches = []
        self._hooks = []
        self._hooked = set()
        self._sam_classes = set()

    def snapshot(self):
        return dict(schema='MINIMAL_BI_ACTUAL_CALL_COUNTS_V1', **dict(self.counts),
            counting='executed function wrappers and actual neural forward pre-hooks',
            no_additional_forwards=True, block_unused=self.block_unused,
            N3_assets_or_modules_required=False,
            unused_modules_loaded=[name for name in sys.modules
                if any(token in name.lower() for token in _FORBIDDEN_MODULES)])

    def reset(self):
        """Reset counts only; existing lazy-model hooks remain installed."""
        self.counts.clear()
        self.counts.update({key: 0 for key in _KEYS})

    def _patch(self, obj, name, replacement):
        original = getattr(obj, name)
        self._patches.append((obj, name, original))
        setattr(obj, name, replacement)
        return original

    def _count(self, obj, name, key):
        original = getattr(obj, name)
        @wraps(original)
        def counted(*args, **kwargs):
            self.counts[key] += 1
            return original(*args, **kwargs)
        self._patch(obj, name, counted)

    def _forward_hook(self, model, key):
        identity = (id(model), key)
        if identity in self._hooked:
            return
        self._hooked.add(identity)
        def increment(module, args):
            self.counts[key] += 1
        self._hooks.append(model.register_forward_pre_hook(increment))

    def _sam(self, predictor_class):
        if predictor_class in self._sam_classes:
            return
        self._sam_classes.add(predictor_class)
        for method, key in (('set_image', 'sam_set_image_calls'), ('predict', 'sam_predict_calls')):
            original = getattr(predictor_class, method)
            def factory(original, key):
                @wraps(original)
                def counted(predictor, *args, **kwargs):
                    self.counts[key] += 1
                    self._forward_hook(predictor.model.image_encoder, 'sam_image_encoder_forward_calls')
                    self._forward_hook(predictor.model.mask_decoder, 'sam_mask_decoder_forward_calls')
                    return original(predictor, *args, **kwargs)
                return counted
            self._patch(predictor_class, method, factory(original, key))

    def _forbidden_path(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        name = os.fsdecode(value).lower()
        if any(token in name for token in _FORBIDDEN_ASSET_TOKENS):
            return True
        path = Path(os.fsdecode(value)).absolute()
        return any(path == root or path.is_relative_to(root) for root in self.forbidden_asset_roots)

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE:
            raise RuntimeError('Actual-call contexts cannot overlap in one process')
        if self.block_unused:
            bad = self.snapshot()['unused_modules_loaded']
            if bad:
                raise RuntimeError('Unused experiment/N3 modules already imported: ' + ', '.join(bad))
        _ACTIVE = True
        try:
            self._count(RT, 'observe', 'observe_calls')
            self._count(RT, 'solve', 'solve_calls')
            self._count(RT.JOINT, 'solve_joint_ot_pnp', 'joint_solver_calls')
            self._count(RT, 'infer_state_visibility_v4', 'state_inference_calls')
            self._count(STATE, 'predict_actual_visibility', 'state_classifier_head_calls')
            self._count(STATE, 'spatial_features', 'state_feature_extractor_calls')
            self._count(RT, 'correct_visible_bi', 'bi_correction_calls')
            self._count(RT.RECOVERY, 'recover_visible_boundary', 'boundary_recovery_calls')
            self._count(RT.MASK, 'segment', 'tiny_segment_calls')
            network = RT.JointObservationModel.network
            @wraps(network)
            def state_network(model, *args, **kwargs):
                result = network(model, *args, **kwargs)
                if self.components is None or model is self.components.state_model.frozen_backbone:
                    self._forward_hook(result, 'state_backbone_forward_calls')
                return result
            self._patch(RT.JointObservationModel, 'network', state_network)
            loader = SEG.load_segmenter
            @wraps(loader)
            def tiny_loader(*args, **kwargs):
                result = loader(*args, **kwargs)
                self._forward_hook(result[0], 'tiny_forward_calls')
                return result
            self._patch(SEG, 'load_segmenter', tiny_loader)
            # A loaded predictor is hooked immediately; otherwise the real
            # provider's lazy import installs these wrappers before first use.
            provider = getattr(self.components, 'mask_provider', None)
            predictor = getattr(provider, 'predictor', None)
            if predictor is not None:
                self._sam(type(predictor))
            importer = builtins.__import__
            @wraps(importer)
            def import_guard(name, *args, **kwargs):
                if self.block_unused and any(token in name.lower() for token in _FORBIDDEN_MODULES):
                    self.counts['blocked_imports'] += 1
                    raise RuntimeError('Unused N3/OT/U import blocked: ' + name)
                module = importer(name, *args, **kwargs)
                if name == 'mobile_sam' and hasattr(module, 'SamPredictor'):
                    self._sam(module.SamPredictor)
                return module
            self._patch(builtins, '__import__', import_guard)
            for module in (builtins, io):
                opener = module.open
                def factory(opener):
                    @wraps(opener)
                    def guarded(file, *args, **kwargs):
                        if self.block_unused and self._forbidden_path(file):
                            self.counts['blocked_asset_accesses'] += 1
                            raise RuntimeError('Unused N3/feature/reference asset access blocked')
                        return opener(file, *args, **kwargs)
                    return guarded
                self._patch(module, 'open', factory(opener))
            path_open = Path.open
            @wraps(path_open)
            def guarded_path_open(path, *args, **kwargs):
                if self.block_unused and self._forbidden_path(path):
                    self.counts['blocked_asset_accesses'] += 1
                    raise RuntimeError('Unused N3/feature/reference asset access blocked')
                return path_open(path, *args, **kwargs)
            self._patch(Path, 'open', guarded_path_open)
            subpix = RT.cv2.cornerSubPix
            @wraps(subpix)
            def blocked_subpix(*args, **kwargs):
                self.counts['subpix_calls'] += 1
                if self.block_unused:
                    raise RuntimeError('Legacy SubPix is outside the minimal correction block')
                return subpix(*args, **kwargs)
            self._patch(RT.cv2, 'cornerSubPix', blocked_subpix)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        global _ACTIVE
        for hook in reversed(self._hooks):
            hook.remove()
        for obj, name, original in reversed(self._patches):
            setattr(obj, name, original)
        self._hooks.clear()
        self._patches.clear()
        self._hooked.clear()
        _ACTIVE = False
        return False


def instrument(components=None, *, block_unused=True, forbidden_asset_roots=()):
    return CallMeter(components, block_unused=block_unused,
                     forbidden_asset_roots=forbidden_asset_roots)
