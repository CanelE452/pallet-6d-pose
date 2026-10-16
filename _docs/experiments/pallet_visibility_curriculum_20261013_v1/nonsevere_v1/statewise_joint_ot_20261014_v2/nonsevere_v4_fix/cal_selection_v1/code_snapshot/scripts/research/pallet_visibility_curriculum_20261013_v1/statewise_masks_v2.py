"""Fresh full-frame Tiny features and non-S-prompted MobileSAM outlines."""
from pathlib import Path
import sys

import cv2
import numpy as np

from . import common as C
from .segmentation import segment
from scripts.research.pallet_foundation_mask_20261013_v1.foundation import choose_mask


class StatewiseMaskProvider:
    def __init__(self, *, tiny_weights, tiny_sha256, sam_vendor, sam_checkpoint,
                 sam_sha256, device="cuda"):
        self.tiny_weights=Path(tiny_weights).resolve()
        self.sam_vendor=Path(sam_vendor).resolve()
        self.sam_checkpoint=Path(sam_checkpoint).resolve()
        self.device=device
        self.predictor=None
        self.bindings=dict(tiny_sha256=tiny_sha256,sam_sha256=sam_sha256)
        if C.sha(self.tiny_weights)!=tiny_sha256 or C.sha(self.sam_checkpoint)!=sam_sha256:
            raise ValueError("Frozen segmentation checkpoint binding changed")

    @classmethod
    def from_configs(cls, state_descriptor, boundary_config_path, device="cuda"):
        path=Path(boundary_config_path).resolve()
        config=C.read(path)
        tiny=state_descriptor["pallet_segmenter"]
        return cls(tiny_weights=C.ROOT/tiny["weights_repo_relative"],tiny_sha256=tiny["weights_sha256"],
            sam_vendor=path.parent/config["mobile_sam_vendor"],
            sam_checkpoint=path.parent/config["mobile_sam_checkpoint"],
            sam_sha256=config["model_sha256"]["mobile_sam_checkpoint"],device=device)

    def __call__(self,image_bgr,*,prompt_points,prompt_native_ids,prompt_box_xyxy,
                 initial_projection_xy,K):
        # Caller computes the geometric S gate before passing this small
        # non-S query subset. No raw S coordinate or 9th centre is available.
        image=np.asarray(image_bgr)
        q=np.asarray(prompt_points,float).reshape(-1,2)
        ids=list(prompt_native_ids)
        h,w=image.shape[:2]
        if len(ids)!=len(q) or len(ids)!=len(set(ids)) or any(i not in range(8) for i in ids):
            raise ValueError("Unique already-gated native prompt IDs required")
        if not np.isfinite(q).all() or np.any((q[:,0]<0)|(q[:,0]>=w)|(q[:,1]<0)|(q[:,1]>=h)):
            raise ValueError("Finite in-frame non-S prompt coordinates required")
        tiny=segment(image,self.tiny_weights,self.device)
        provenance=dict(schema="STATEWISE_MASKS_V2",prompt_native_ids=ids,
            prompt_points=q.tolist(),prompt_box_xyxy=None,mask_forward_calls=0,
            state_features_source="FRESH_FULLFRAME_FROZEN_TINY",BI_boundary_source="FRESH_MOBILESAM_BOX_ONLY",
            S_raw_coordinates_available=False,initial_projection_used_for_prompt=False,
            class_agnostic_mask_is_visibility_certificate=False,model_sha256=self.bindings)
        if prompt_box_xyxy is None:
            return dict(state_pallet_mask=tiny,pallet_mask=np.zeros((h,w),np.float32),
                        provenance=dict(provenance,reason="no_nonS_raw_prompt_box"))
        expected=np.r_[q.min(0),q.max(0)]
        box=np.asarray(prompt_box_xyxy,float)
        if box.shape!=(4,) or not np.allclose(box,expected,atol=1e-8,rtol=0):
            raise ValueError("Prompt box must bind non-S raw queries only")
        import torch
        if self.predictor is None:
            if C.sha(self.sam_checkpoint)!=self.bindings["sam_sha256"]:
                raise ValueError("MobileSAM checkpoint changed")
            sys.path.insert(0,str(self.sam_vendor))
            from mobile_sam import SamPredictor,sam_model_registry
            import mobile_sam
            if not Path(mobile_sam.__file__).resolve().is_relative_to(self.sam_vendor):
                raise ValueError("MobileSAM imported outside frozen vendor")
            model=sam_model_registry["vit_t"]()
            model.load_state_dict(torch.load(self.sam_checkpoint,map_location="cpu",weights_only=True))
            self.predictor=SamPredictor(model.to(self.device).eval().requires_grad_(False))
        with torch.inference_mode():
            self.predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            masks,quality,_=self.predictor.predict(box=box,multimask_output=True)
        mask,index=choose_mask(masks,quality)
        if mask.shape!=(h,w):raise ValueError("SAM mask must preserve the current image canvas")
        provenance.update(prompt_box_xyxy=box.tolist(),mask_forward_calls=1,
                          selected_mask_index=index,uncalibrated_mask_quality=np.asarray(quality).tolist())
        return dict(state_pallet_mask=tiny,pallet_mask=mask.astype(np.float32),provenance=provenance)
