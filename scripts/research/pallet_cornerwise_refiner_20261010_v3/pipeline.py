"""Per-corner observed-coordinate choice, then unchanged finite robust PnP."""
from __future__ import annotations
import copy
import numpy as np
from . import common as C
from .selection import select_corners
from ..pallet_boundary_corner_refiner_20261010_v2.deployment import Pipeline as Parent
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import observation_points,assemble,preserve_prediction
from ..pallet_boundary_corner_refiner_20261010_v2 import pose

class Pipeline(Parent):
    def outcomes(self,capture,methods=C.METHODS):
        n3=capture['native_N3_points'];initial=capture['initial_pose'];hidden=capture['hidden'];meta=capture['metadata']
        sparse,hybrid,old_contract=observation_points(n3,capture['observation'],hidden)
        banks={};results=[]
        def bank(points):
            key=np.asarray(points,dtype='<f8').tobytes()
            if key not in banks:banks[key]=pose.PoseBank(points,np.asarray(meta['K']),np.asarray(meta['xyz']),image_size=(meta['raw_hw'][1],meta['raw_hw'][0]))
            return banks[key]
        with self.old.no_truth_reads():
            selected=None;contract=None
            for method in methods:
                C.require(method in C.METHODS,'unknown cornerwise method')
                if method==C.PRIMARY:
                    if selected is None:
                        selected,contract=select_corners(n3,capture['observation'],initial,hidden,
                            np.asarray(meta['K']),np.asarray(meta['xyz']),(meta['raw_hw'][1],meta['raw_hw'][0]),bank=bank(n3))
                        self.counts['LOO_pose_paths']+=contract['cornerwise_selection']['LOO_solve_calls']
                    points=selected;use_contract=contract
                else:
                    points=hybrid if method=='N3_VALIDATED_ROLE' else n3
                    use_contract=old_contract
                H=[] if method=='N3_BASIN_NO_MASK_ROBUST' else hidden
                solved=pose.refine(points,np.asarray(meta['K']),np.asarray(meta['xyz']),initial,H,
                    (meta['raw_hw'][1],meta['raw_hw'][0]),robust=True,bank=bank(points))
                self.counts['final_pose_paths']+=1
                # The inherited assembler expects the original hybrid method name.
                label='N3_VALIDATED_ROLE' if method==C.PRIMARY else method
                result=assemble(n3,points,initial,H,solved,label,use_contract)
                if method==C.PRIMARY:
                    result['cornerwise_selection']=copy.deepcopy(contract['cornerwise_selection'])
                    result['selection_validation_fit_excludes_candidate_corner']=True
                    result['selection_validation_is_fully_statistically_independent']=False
                    result['selection_validation_has_shared_initial_prior']=True
                result.update(mask_diagnostic=capture['mask_diagnostic'],predicted_initial_N3_hidden=list(hidden),
                    observation_raw_logits_sha256=capture['observation'].get('raw_logits_sha256'),
                    selected_corner_ids=use_contract['hybrid_boundary_corner_ids'] if method in (C.PRIMARY,'N3_VALIDATED_ROLE') else [],
                    selected_queries=capture['observation'].get('selected_queries',0),source_lines=[int(l['edge']) for l in capture['observation']['lines']],
                    model_query_anchor='unchanged original Base predictions',initial_pose_source='fresh fixed N3_SUBPIX',
                    partial_lines_not_pose_inputs=True)
                results.append(self.packet(meta,capture,method,result))
        return results,dict(coordinate_banks=[dict(input_hash=b.digest,**dict(b.ledger)) for b in banks.values()])

    def predict(self,image,K,xyz,metadata=None,method=C.PRIMARY,stage_callback=None):
        C.require(method in C.METHODS+('BASE','N3_SUBPIX'),'unknown route')
        base=method=='BASE';role=method in (C.PRIMARY,'N3_VALIDATED_ROLE')
        with C.inference_canary():
            capture=self.capture(image,K,xyz,metadata,need_role=role,need_n3=not base,need_base_pose=False,stage_callback=stage_callback)
            if base:capture['initial_base_pose']=capture['initial_pose']
            result=self.fixed_result(capture,method) if method in ('BASE','N3_SUBPIX') else self.outcomes(capture,(method,))[0][0]
            prediction=copy.deepcopy(capture['prediction']);idx=prediction['selected_index']
            if idx is not None:prediction['candidates'][idx]['keypoints_xy']=result['native_points']
            preserve_prediction(capture['raw'],prediction)
            if stage_callback:stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction,original_base_points=capture['original_base_points'],native_N3_points=capture['native_N3_points'],observation=capture['observation'])
            return result
