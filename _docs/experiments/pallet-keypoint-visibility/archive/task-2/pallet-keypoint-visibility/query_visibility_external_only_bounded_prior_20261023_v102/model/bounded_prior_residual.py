"""Two external-only residual arms; the sole difference is fixed prior clipping.

Import reads pinned CODE only. Source checkpoints are loaded only by an explicit
factory call after the caller's root entry; tests use invented C1-format weights.
The model cannot read labels, query geometry, part IDs, variants or source IDs.
"""
from pathlib import Path
import hashlib,importlib.util,sys
sys.dont_write_bytecode=True
import torch
P=Path(__file__).resolve().parents[2]
C5_CODE=P/'query_visibility_local_residual_head_20261023_v101/model/local_rgb_residual.py'
C5_CODE_SHA256='836f7309d8aa2e4db28fe3b7197c2befc3c8643dd986bc1571a07d17906701f4'
ARMS=('C6_EXTERNAL_ONLY_RAW_PRIOR','C7_EXTERNAL_ONLY_CLIPPED_PRIOR2')

def file_sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if file_sha256(C5_CODE)!=C5_CODE_SHA256:raise RuntimeError('Pinned inert C5 model code changed')
_spec=importlib.util.spec_from_file_location('v102_pinned_c5_geometry_only',C5_CODE)
_C5=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_C5)
_BASE=_C5._BASE
BASE_CODE=_C5.BASE_CODE;BASE_CODE_SHA256=_C5.BASE_CODE_SHA256
C1_CHECKPOINT=_C5.C1_CHECKPOINT;C1_CHECKPOINT_SHA256=_C5.C1_CHECKPOINT_SHA256
FORWARD_FIELDS=_C5.FORWARD_FIELDS
sample_query_feature=_C5.sample_query_feature
state_sha256=_C5.state_sha256
known_binary_loss=_C5.known_binary_loss
decode_binary=_C5.decode_binary

def prior_anchor(reference_base_logit,arm):
    if arm==ARMS[0]:return reference_base_logit
    if arm==ARMS[1]:return torch.clamp(reference_base_logit,min=-2.,max=2.)
    raise ValueError('Unknown prospective source arm')

class QueryAnchoredResidualExternal(_C5.LocalRGBResidualC5):
    """Frozen26573 C1 + fresh copied-encoder residual3221; total29794.

    The Python arm configuration is fixed at construction, never forwarded as a
    tensor. Fresh residual initialization is the identical inherited C5 law, not
    any C5-trained branch. Both arms require the same seed before construction.
    """
    def __init__(self,c1_state_dict,arm=ARMS[0]):
        if arm not in ARMS:raise ValueError('Unknown prospective source arm')
        super().__init__(c1_state_dict)
        self.arm=arm

    def forward(self,p3_roi,local_rgb96,local_valid96):
        with torch.no_grad():
            reference=self.base(p3_roi,local_rgb96,local_valid96)
            raw=reference['query_logit']
            anchor=prior_anchor(raw,self.arm)
        available=reference['query_available']
        raw_delta=self._residual_raw(local_rgb96,local_valid96)
        delta=torch.where(available,raw_delta,raw_delta*0)
        return {'query_logit':anchor+delta,'query_available':available,
                'base_logit':anchor,'reference_base_logit':raw,'delta_logit':delta}

def load_frozen_c1(arm=ARMS[0],checkpoint_path=C1_CHECKPOINT,
                   expected_sha256=C1_CHECKPOINT_SHA256,map_location='cpu'):
    """Explicit delayed load; actual calls require the caller's root entry.

    Generated tests pass their own invented-format path and SHA. The original
    C1 source checkpoint is the only approved actual asset; no C5 fit is loaded.
    """
    if arm not in ARMS:raise ValueError('Unknown prospective source arm')
    checkpoint_path=Path(checkpoint_path);actual=file_sha256(checkpoint_path)
    if actual!=expected_sha256:raise RuntimeError('C1 checkpoint SHA mismatch')
    payload=torch.load(checkpoint_path,map_location=map_location,weights_only=True)
    if not isinstance(payload,dict)or'state_dict'not in payload:raise ValueError('C1 payload requires state_dict')
    model=QueryAnchoredResidualExternal(payload['state_dict'],arm)
    return model,{'arm':arm,'checkpoint':{'path':str(checkpoint_path.resolve()),'sha256':actual},
       'base_code':{'path':str(BASE_CODE),'sha256':BASE_CODE_SHA256},
       'copied_branch_geometry_code':{'path':str(C5_CODE),'sha256':C5_CODE_SHA256},
       'loaded_base_state_sha256':model.loaded_base_state_sha256,'checkpoint_update':payload.get('update'),
       'parameter_count':29794,'trainable_parameter_count':3221,'base_frozen':True,'base_eval':not model.base.training,
       'anchor_policy':'original_raw_C1'if arm==ARMS[0]else'clamp_original_raw_C1_minus2_plus2',
       'C5_trained_branch_loaded':False}
