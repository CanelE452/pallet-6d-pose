"""Architecture reuse only. The new output is external-foreign-ray likelihood.

This output is not physical cue visibility, pixel membership, or semantic D/E/S.
No reference coordinates, source depth, owner masks or labels enter forward.
"""
import hashlib
import importlib.util
from pathlib import Path

BASE_CODE = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_reference_local_support_20261023_v76/mixed_fit/query_center_support_head.py')
BASE_SHA = '67fdee4a9ad850ba91482e4ce1fc098834c88f207c4edc0ab178b685ae99c940'
assert hashlib.sha256(BASE_CODE.read_bytes()).hexdigest() == BASE_SHA
spec = importlib.util.spec_from_file_location('immutable_v76_architecture', BASE_CODE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

FORWARD_FIELDS = base.FORWARD_FIELDS


class ExternalRayHead(base.NativeContextSupportHead):
    def forward(self, p3_roi, local_rgb96, local_valid96, target_rgb128, target_valid128, context42):
        result = super().forward(p3_roi, local_rgb96, local_valid96, target_rgb128, target_valid128, context42)
        return {'external_logit': result['support_logit'], 'input_available': result['input_available']}


initialize_from_source_pretrain = base.initialize_from_proxy
