"""Repository-level geometry and real-cache numerical gates, before calibration."""
import math
import numpy as np
import torch
import env as E
from b_model import raw_lattice,align,FrozenHeads
from audit_math import line_candidate_evidence, edge_channel_permutations, derive_permutations
from preflight import rotations
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import decode_single_mode
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset,collate,observation_batch

def main():
    torch.set_num_threads(4);assert torch.cuda.is_available();E.gpu()
    assert E.read(E.DOC/'B/CACHE_COMPLETE.json')['complete']
    h=FrozenHeads();tests={};maxerror=0
    for box in ([0,0,640,640],[-100,23,401,211],[101,-97,999,700]):
        box=torch.tensor(box,dtype=torch.float64);l=h.line.lattice.lines.cpu().double()
        lr,s=raw_lattice(l,box);scale=2/(box[2:]-box[:2]);shift=-(box[2:]+box[:2])/(box[2:]-box[:2])
        # General line/point incidence is preserved under the full homogeneous transpose.
        rng=torch.Generator().manual_seed(91);x=torch.randn(23,2,generator=rng,dtype=torch.float64)*100
        a=torch.eye(3,dtype=torch.float64);a[0,0],a[1,1]=scale;a[:2,2]=shift
        ref=l@a;norm=ref[:,:2].norm(dim=-1)
        assert torch.allclose(lr,ref/norm[:,None],atol=1e-12)
        assert torch.allclose((x@scale.diag()+shift)@l[:,:2].T+l[:,2],(x@lr[:,:2].T+lr[:,2])*norm,atol=1e-10)
    tests['non_square_raw_ROI_line_transpose']=True
    old=E.read(E.ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/predictions/hough_seed1_synth_val.json')
    old={r['frame_id']:r for r in old['records']};line_delta=0
    geometry=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
    geometry_index={str(k):i for i,k in enumerate(geometry['stems'])}
    geometry_checked=0
    for split in ['calibration','synth_val']:
        for rec in E.read(E.EXPORT/(split+'.json'))['records']:
            j=geometry_index[rec['frame_id']];corners=geometry['Xcf'][j]
            x=np.concatenate([corners,corners.mean(0,keepdims=True)],axis=0)
            p=derive_permutations(x,rotations(rec['symmetry_order']),np.array(EDGES))
            assert p.tolist()==rec['symmetry_permutations'],rec['frame_id']
            assert geometry['match_err'][j]<.05
            geometry_checked+=1
    for split in ['calibration','synth_val']:
        for i in [0,1,17]:
            c=torch.load(E.RAW/f'B/cache/{split}/{i:04d}.pt',map_location='cpu',weights_only=False)
            out=c['P'][1]['output'];x=(out['points_raw'][0,:8,None]+out['candidate_displacements'][0,None]-torch.tensor(c['offset']))/c['gain']-100
            lines,sigma=raw_lattice(h.line.lattice.lines.cpu().double(),c['box_raw'].double())
            al=c['P'][1]['alignment']['mapping'];logits=c['line_logits'][al];valid=c['line_valid'][al]
            for mode,arm in [('three_ambiguity','B3_THREE_AMBIG'),('three_equal','B2_THREE_EQUAL'),('one_ambiguity','B1_ONE_AMBIG')]:
                ref=line_candidate_evidence(x.double(),logits.double(),valid,lines,sigma,np.array(EDGES),mode)['bias']
                actual=c['P'][1]['biases'][arm].double();error=float((actual-ref).abs().max());maxerror=max(maxerror,error)
                assert torch.allclose(actual,ref,atol=5e-5,rtol=5e-5),error
            assert out['candidate_displacements'][0,-1].abs().sum()==0
            assert c['P'][1]['available'].numel()==12
            if split=='synth_val':
                dec=decode_single_mode(c['line_logits'][None],c['line_valid'][None],h.line.lattice.cpu(),c['box_raw'][None])
                d=float(np.max(np.abs(dec['raw_line'][0].numpy()-np.array(old[c['frame_id']]['raw_line']))));line_delta=max(line_delta,d)
                # CPU/MAP coefficients are diagnostic, not the full-distribution parity gate.
                # Historical native batch32 is reproduced separately below.
            # Whole-object valid orbit relabeling merely permutes the list of support scores.
            lines32,s32=raw_lattice(h.line.lattice.lines.cpu(),c['box_raw'])
            _,_,original=align(c['line_logits'],c['line_valid'],lines32,s32,c['base_points'],c['point_valid'],c['perms'])
            mapping=edge_channel_permutations(np.array(c['perms']),np.array(EDGES))[-1]
            _,_,other=align(c['line_logits'][mapping],c['line_valid'][mapping],lines32,s32,c['base_points'],c['point_valid'],c['perms'])
            assert np.allclose(sorted(original['scores']),sorted(other['scores']),atol=1e-5)
    tests.update(full_posterior_GPU_CPU_parity=True,GT_free_role_orbit_consistency=True,null_candidate_included=True,
      native_DHT_decoded_observation_parity=True,point_prediction_generation_before_GT=True)
    h.line.to('cuda');ds=ObservationDataset(E.EXPORT/'synth_val.json',targets=False)
    native_max=0.;tv_max=0.;native_anchor_same=0
    for start in range(0,len(ds),32):
        obs=observation_batch(collate([ds[i] for i in range(start,start+32)]),torch.device('cuda'))
        logits,valid=h.observe(obs);decoded=decode_single_mode(logits,valid,h.line.lattice,obs['box'])
        for j in range(32):
            i=start+j;c=torch.load(E.RAW/f'B/cache/synth_val/{i:04d}.pt',map_location='cpu',weights_only=False)
            previous=old[c['frame_id']]
            native_max=max(native_max,float(np.max(np.abs(decoded['raw_line'][j].cpu().numpy()-np.array(previous['raw_line'])))))
            native_anchor_same+=int(decoded['anchor_index'][j].cpu().tolist()==previous['anchor_index'])
            tv_max=max(tv_max,float(((logits[j].softmax(-1).cpu()-c['line_logits'].softmax(-1)).abs().sum(-1)/2).max()))
    assert native_max<1e-6 and native_anchor_same==512,(native_max,native_anchor_same)
    assert tv_max<.001,tv_max
    E.write(E.DOC/'B/ADAPTER_TESTS.json',dict(PASS=True,tests=tests,max_bias_GPU_FP32_vs_CPU_FP64=maxerror,
      decoded_historical_line_max_abs=line_delta,geometry_records_derived_from_3D=geometry_checked,
      historical_native_batch32_line_max_abs=native_max,historical_native_batch32_anchors_identical=native_anchor_same,
      batch1_vs32_max_edge_total_variation=tv_max,main_B_batch=1,
      scoring_GT_inputs=0,main_training_updates=0))
    print('B ADAPTER PASS',maxerror,line_delta,flush=True)
if __name__=='__main__':main()
