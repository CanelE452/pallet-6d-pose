"""V160 factorial head; imports pinned source code only, never data or weights."""
from pathlib import Path
import hashlib
import importlib.util
import torch
from torch.nn import functional as F

P = Path(__file__).resolve().parents[2]
BASE_SOURCE = P / 'query_visibility_independent_auxiliary_source_control_20261024_v155/independent_head_preparation/independent_query_head.py'
BASE_SHA = '3c54dc75d6650aabce8b76a277998fded6f77f22293ba408bdddd998104bb39a'
assert hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() == BASE_SHA
_spec = importlib.util.spec_from_file_location('_v160_pinned_independent_query_head', BASE_SOURCE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
IndependentAuxiliaryQueryHead = _base.IndependentAuxiliaryQueryHead


class NativeSlotOwnerQueryHead(IndependentAuxiliaryQueryHead):
    """Keep original keys/partitions; add 707 separately optimized parameters.

    Flags are fixed by the constructor, never supplied to forward. Owner logits
    remain diagnostic in every arm; the trainer uses owner loss weight 0 or 1
    from owner_auxiliary. Owner never receives the native-hint representation.
    """
    def __init__(self, native_slot_hint=False, owner_auxiliary=False):
        if type(native_slot_hint) is not bool or type(owner_auxiliary) is not bool:
            raise TypeError('Arm flags must be constructor booleans')
        super().__init__()
        self._native_slot_hint = native_slot_hint
        self._owner_auxiliary = owner_auxiliary
        # Always allocate both branches after all original initializations, so
        # every arm has identical original tensors and consumes identical RNG.
        self.native_hint = torch.nn.Linear(8, 64, bias=False)
        self.owner_head = torch.nn.Linear(64, 3)
        torch.nn.init.zeros_(self.native_hint.weight)
        torch.nn.init.zeros_(self.owner_head.weight)
        torch.nn.init.zeros_(self.owner_head.bias)

    @property
    def native_slot_hint(self):
        return self._native_slot_hint

    @property
    def owner_auxiliary(self):
        return self._owner_auxiliary

    def new_parameters(self):
        return list(self.native_hint.parameters()) + list(self.owner_head.parameters())

    def load_visibility_state(self, state):
        """Preserve the original five-key loading API; leave all aux untouched."""
        expected = {k for k in self.state_dict() if k.startswith(('head.', 'geometry.'))}
        if set(state) != expected:
            raise ValueError('Expected original visibility state keys only')
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert set(missing) == set(self.state_dict()) - expected and not unexpected

    def load_original_state(self, state):
        """Load exactly full V155 original state, preserving both new branches."""
        expected = {k for k in self.state_dict()
                    if not k.startswith(('native_hint.', 'owner_head.'))}
        if set(state) != expected:
            raise ValueError('Expected exact original V155 state keys only')
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert set(missing) == set(self.state_dict()) - expected and not unexpected

    def forward(self, features, geometry42, native_slot):
        assert features.ndim == geometry42.ndim == 2 and features.shape[1] == 1928 and geometry42.shape == (len(features), 42)
        assert features.dtype == geometry42.dtype == torch.float32
        if native_slot.ndim != 1 or native_slot.shape[0] != len(features) or native_slot.dtype != torch.int64:
            raise ValueError('native_slot must be input-metadata int64[N]')
        if native_slot.device != features.device or geometry42.device != features.device:
            raise ValueError('All inputs must be on the same device')
        if bool(((native_slot < 0) | (native_slot > 7)).any()):
            raise ValueError('native_slot must be in 0..7')
        T, Q = features[:, :384], features[:, 384:768]
        C, Pv = features[:, 1152:1536], features[:, 1536:1920]
        visual = torch.cat((features, torch.abs(T-C), T*C,
                            torch.abs(Q-Pv), Q*Pv), 1)
        pre_slot_h = self.head[0](visual) + self.geometry(geometry42)
        owner = self.owner_head(self.head[1](pre_slot_h))
        visibility_h = pre_slot_h
        if self.native_slot_hint:
            onehot = F.one_hot(native_slot, num_classes=8).to(dtype=features.dtype)
            visibility_h = pre_slot_h + self.native_hint(onehot)
        visibility = self.head[2](self.head[1](visibility_h)).squeeze(-1)
        return dict(external_occlusion=visibility,
                    target_entity=self.entity_head(T).squeeze(-1),
                    coarse_point_quality=self.quality_head(visual).squeeze(-1),
                    query_visible_owner=owner)
