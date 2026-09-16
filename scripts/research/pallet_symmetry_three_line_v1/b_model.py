"""Frozen full-posterior observation and point-logit-only scoring. No GT API."""
import math
import numpy as np
import torch
import env as E
from audit_math import line_candidate_evidence, edge_channel_permutations, rerank_logits, derive_permutations
from preflight import geometry, rotations
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.model import build_model
from generic_point_refiner import GenericPointRefiner, decode
MODES={'B1_ONE_AMBIG':'one_ambiguity','B2_THREE_EQUAL':'three_equal','B3_THREE_AMBIG':'three_ambiguity'}

def raw_lattice(lattice,box):
    # x_roi=A*x_raw. l_raw=A.T*l_roi; no line intersections or point solving.
    scale=2/(box[2:]-box[:2]);shift=-(box[2:]+box[:2])/(box[2:]-box[:2])
    raw=torch.cat([lattice[:,:2]*scale, (lattice[:,2]+(lattice[:,:2]*shift).sum(-1))[:,None]],-1)
    norm=raw[:,:2].norm(dim=-1)
    sigma=(2*math.sqrt(2)/64/norm).clamp_min(1.)
    return raw/norm[:,None],sigma

@torch.no_grad()
def align(line_logits,line_valid,lines,sigma,base,point_valid,perms):
    maps=edge_channel_permutations(np.array(perms),np.array(EDGES));scores=[]
    positions=base[:8,None].clone();positions[~point_valid[:8]]=float('nan')
    for mapping in maps:
        o=line_candidate_evidence(positions,line_logits[mapping],line_valid[mapping],lines,sigma,np.array(EDGES),'three_equal')
        # 3 * sum corner means = sum clipped endpoint evidence for all 12 edges.
        scores.append(float(o['bias'].sum()*3))
    choice=int(np.argmax(scores));mapping=maps[choice]
    return line_logits[mapping],line_valid[mapping],dict(inference_orbit=choice,scores=scores,mapping=mapping.tolist(),GT_input=False)

class FrozenHeads:
    def __init__(self,device='cuda'):
        self.device=device;self.selection=E.read(E.C.B/'P_SELECTION.json')
        assert E.sha(E.DHT)==E.DHT_SHA
        ck=torch.load(E.DHT,map_location='cpu',weights_only=False)
        self.line=build_model(ck['model_config']).to(device)
        self.line.load_state_dict(ck['state_dict'],strict=True);self.line.requires_grad_(False).eval()
        self.points={}
        for seed in [1,2,3]:
            path=E.C.BRAW/f'runs/seed{seed}/last.pt';assert E.sha(path)==self.selection['checkpoints'][str(seed)]
            ck=torch.load(path,map_location='cpu',weights_only=False)
            p=GenericPointRefiner(**ck['config']).to(device);p.load_state_dict(ck['model_state_dict'],strict=True)
            p.requires_grad_(False).eval();self.points[seed]=p

    @torch.no_grad()
    def observe(self,observation):
        # Crucial: never calls LocalLineFusionV2.forward, WLS, or utility_head.
        features=observation['features'].to(self.device).float()
        content=observation['content'].to(self.device).float()
        return self.line.line_head(self.line.stem(features),content)

    @torch.no_grad()
    def score(self,point_output,line_logits,line_valid,box_raw,base_raw,point_valid,perms,gain,offset,pad=100,arms=None):
        lines,sigma=raw_lattice(self.line.lattice.lines,box_raw)
        logits,valid,alignment=align(line_logits,line_valid,lines,sigma,base_raw,point_valid,perms)
        locations_input=point_output['points_raw'][0,:8,None]+point_output['candidate_displacements'][0,None]
        locations=(locations_input-offset)/gain-pad
        result={}
        for arm,mode in MODES.items():
            if arms is not None and arm not in arms:continue
            o=line_candidate_evidence(locations,logits,valid,lines,sigma,np.array(EDGES),mode)
            result[arm]=o['bias'].cpu()
        return dict(biases=result,alignment=alignment,ambiguity=o['ambiguity'].cpu(),
                    concentration=o['concentration'].cpu(),available=o['line_available'].cpu())

def decode_raw(output,bias,temperature,eta,rule,gain,raw_hw,base_raw):
    revised=dict(output)
    if eta!=0:revised['logits']=rerank_logits(output['logits'],bias[None].to(output['logits']),temperature,eta)
    cap=rule['max_move_image_diagonal_fraction']*math.hypot(*raw_hw)*gain
    moved=decode(revised,temperature,rule['lam'],cap)
    raw=base_raw+(moved[0]-output['points_raw'][0]).cpu().double()/gain
    raw[8]=base_raw[8]
    return raw,moved
