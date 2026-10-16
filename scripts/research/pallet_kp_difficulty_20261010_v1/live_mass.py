"""Live fixed IMAGE_ROLE head with the separately locked match-mass decoder."""
import hashlib
import numpy as np
from ..pallet_observation_refiner_20261009_v1 import common as C
from ..pallet_observation_refiner_20261009_v1.inference import hidden_mask
from ..pallet_observation_refiner_20261009_v1 import learned_infer as L
from .match_mass import decode


def benchmark_adapter(models):
    class Adapter:
        def __init__(self):
            self.head=L.load_head('IMAGE_ROLE');self.forwards=0
            self.bindings=[C.binding(L.checkpoint_path('IMAGE_ROLE')),C.binding(__file__),
                           C.binding(L.__file__),C.binding(L.M.__file__)]
            self.contract=dict(arm='IMAGE_ROLE_MATCH_MASS',checkpoint='unchanged last update3000',
                match_existence='sum65 posterior > none posterior; fixed0.5',
                location='one spatial argmax bin, unchanged1px',
                unobserved_inputs='NaN, no automatic initial-coordinate fill',
                feature_role_pose='fresh standard predicted Base pose',
                final_mask_pose='separate fresh historical Base pose',
                fallback='complete initial Base coordinates and pose')

        def predict(self,image,captured,panel_frame,pipeline):
            raw=dict(candidates=pipeline.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
            index=raw['selected_index']
            original=np.full((9,2),np.nan) if index is None else np.asarray(raw['candidates'][index]['keypoints_xy'],float)
            K=np.asarray(panel_frame['camera_intrinsics']);xyz=np.asarray(panel_frame['dimensions_wdh_m'])[[0,2,1]]
            initial=pipeline.pose.infer(original,K,xyz,source=False);hidden,_=hidden_mask(initial)
            observed=np.full((9,2),np.nan);observed[8]=original[8]
            if index is None:
                diagnostic=dict(no_detection=True,corners=[],lines=[],queries=[],feature_pose_computed=False)
            else:
                x,query=L.M.inputs(image,captured,original,K,xyz)
                logits=self.head(x[None],'IMAGE_ROLE')[0].detach().float().cpu().numpy();self.forwards+=1
                raw_choice=logits.argmax(-1)
                encoded=dict(original_base_points=original.tolist(),queries=[])
                for i,(edge,a,b,u,length) in enumerate(query['identity']):
                    encoded['queries'].append(dict(query=i,edge=int(edge),endpoints=[int(a),int(b)],
                        fraction=float(u),center=query['center'][i].tolist(),normal=query['normal'][i].tolist(),
                        chosen_candidate=int(raw_choice[i]),no_match=int(raw_choice[i])==65 or not bool(query['valid'][i])))
                diagnostic=decode(encoded,logits,True)
                diagnostic.update(feature_pose_computed=True,feature_initial_pose=query['initial_pose'],
                    raw_logits_sha256=hashlib.sha256(logits.astype('<f4').tobytes()).hexdigest(),
                    feature_hidden_initial=query['hidden_initial'],predicted_roles=query['role'])
                for corner in diagnostic['corners']:observed[corner['id']]=corner['xy']
            return dict(prediction=raw,initial_pose=initial,original_points=original,observation_points=observed,
                        hidden=hidden,excluded=hidden,diagnostic=diagnostic)
    return Adapter()
