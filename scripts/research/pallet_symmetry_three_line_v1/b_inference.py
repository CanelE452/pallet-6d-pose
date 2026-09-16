"""Actual RGB -> original R0/P -> optional frozen line evidence -> native decode."""
import copy,time,math
import numpy as np
import torch
import env as E
from b_model import FrozenHeads,decode_raw
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.export import sample_roi

class ThreeLineInference:
    def __init__(self,seed=1):
        self.seed=seed;self.heads=FrozenHeads();self.features=E.C.old('features')
        self.extractor=self.features.FrozenYoloFeatures(E.R0,device='cuda')
        self.perms=E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations']
        self.eta=E.read(E.DOC/'B/selection_lock.json')['selected_eta']
    @torch.no_grad()
    def predict(self,bgr,arm='B3_THREE_AMBIG',stages=False):
        times={};last=time.perf_counter()
        def mark(name):
            nonlocal last
            if stages:
                torch.cuda.synchronize();now=time.perf_counter();times[name]=(now-last)*1000;last=now
        captured=self.extractor.predict(bgr);mark('R0')
        candidates=copy.deepcopy(captured['candidates']);index=captured['selected_index']
        if index is None:return dict(candidates=candidates,selected_index=index,stages=times)
        inputs=self.features.branch_inputs(captured);gain,offset=self.features.canvas_affine(captured['canvas_shape'],captured['input_shape'])
        def tensor(x,dtype=torch.float32):return torch.as_tensor(x,device='cuda',dtype=dtype)[None]
        out=self.heads.points[self.seed](captured['p3'],captured['p4'],tensor(inputs['points']),tensor(inputs['boxes']),
          tensor(inputs['point_valid'],torch.bool),tensor(inputs['input_shape']),lam=0);mark('P')
        eta=0 if arm=='B0_P' else self.eta[arm]
        bias=torch.zeros_like(out['logits'][0]);chosen=candidates[index]
        if eta>0:
            box=np.array(chosen['box_xyxy']);p3,c3=sample_roi(captured['p3'][0].cpu().numpy(),box,gain,offset,8,bgr.shape[:2])
            p4,c4=sample_roi(captured['p4'][0].cpu().numpy(),box,gain,offset,16,bgr.shape[:2]);content=c3&c4
            obs=dict(features=(torch.cat([p3,p4],1)*content).cuda(),content=content.cuda());mark('ROI')
            logits,valid=self.heads.observe(obs);mark('DHT_observation')
            evidence=self.heads.score(out,logits[0],valid[0],tensor(box)[0],tensor(chosen['keypoints_xy'])[0],
              tensor(inputs['point_valid'],torch.bool)[0],self.perms,gain,tensor(offset)[0],arms=[arm])
            bias=evidence['biases'][arm];mark('role_alignment_and_full_distribution_score')
        temperature=self.heads.selection['temperatures'][str(self.seed)]['temperature']
        raw,_=decode_raw(out,bias,temperature,eta,self.heads.selection['selected_rule'],gain,bgr.shape[:2],torch.tensor(chosen['keypoints_xy']))
        candidates[index]['keypoints_xy']=raw.numpy();mark('decode_restore')
        return dict(candidates=candidates,selected_index=index,stages=times)
    def close(self):self.extractor.close()
