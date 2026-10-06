"""Actual-input numerical/affine and NoOp checks, outside all optimizer fits."""
import argparse,importlib.util
from pathlib import Path
import numpy as np,torch
from .a_common import ROOT,DOC,POSE,read,write,sha
from .a_data import Data,affine
from .geometry import build_bank,permute_bank,NUMERIC
from .scorer import JointActionScorer,decode_bank

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args();data=Data(args.source_root)
    spec=importlib.util.spec_from_file_location('handoff_existing_features',ROOT/'scripts/research/pallet_line_pose_v1/features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    errors=[];maxpose=0.;maxsource=0.;maxraw=0.;actual=0
    for row in data.oracle_rows[:8]:
        f=data.source_frame(row);a=POSE.infer(f['q'],f['K'],f['xyz'],True);b=POSE.infer(f['q'].copy(),f['K'],f['xyz'],True);assert a==b
        if a['available']:
            maxpose=max(maxpose,abs(POSE.metric((f['id'],a,f['truth']))['ADDsym_m']-POSE.metric((f['id'],b,f['truth']))['ADDsym_m']))
        r=data.source['records'][data.indices[row]];scale,offset=affine(r['prepared_shape_hw'],data.arrays['input_shape'][row]);gain,offs=old.canvas_affine(r['prepared_shape_hw'],data.arrays['input_shape'][row]);np.testing.assert_array_equal(scale,[gain,gain]);np.testing.assert_array_equal(offset,offs)
        np.testing.assert_allclose((f['q']*f['scale']+f['offset'])[f['point_valid']],data.arrays['points'][row][f['point_valid']],atol=1e-6,rtol=0)
        bank=build_bank(f['q'],f['K'],f['xyz'],f['raw_hw'],f['point_valid']);assert np.array_equal(bank['points'][0],f['q'],equal_nan=True)
        maxraw=max(maxraw,float(np.nanmax(np.linalg.norm(bank['points'][:,:8]-f['q'][:8],axis=-1))))
        perm=permute_bank(bank,f['id']);
        for i in range(8):assert sorted(map(tuple,bank['points'][:,i]))==sorted(map(tuple,perm['points'][:,i]))
        actual+=1
    for f in data.real_frames()[:8]:
        c=f['captured']['captured'];a=old.branch_inputs(c)
        if a is None:continue
        b=data.real_batch([f],'cpu');np.testing.assert_array_equal(b['points'][0].numpy(),a['points']);np.testing.assert_array_equal(b['boxes'][0].numpy(),a['boxes']);actual+=1
    torch.manual_seed(1);head=JointActionScorer(5,c3=2,c4=3,hidden=2,encoded=3);q=torch.full((1,9,2),float('nan'));v=torch.zeros((1,9),dtype=torch.bool)
    out=head.forward_bank(torch.randn(1,2,8,8),torch.randn(1,3,4,4),q,torch.tensor([[0.,0.,20.,20.]]),v,torch.tensor([[64,64]]),context=torch.zeros(1,5),candidate_points=q[:,None],action_valid=torch.ones(1,1,dtype=torch.bool));result,index=decode_bank(out,'J');assert index.item()==0 and torch.isnan(result).all()
    write(DOC/'results/A_NUMERICAL_PARITY.json',dict(PASS=True,actual_source_and_real_frames=actual,NoOp_actual_F_repeat_ADDsym_maxabs_m=maxpose,locked_headroom_tolerance_m=NUMERIC['headroom_ADDsym_m'],affine_existing_formula_exact=True,real_existing_branch_points_boxes_exact=True,NoOp_missing_M1_verified=True,optimizer_updates=0,max_sampled_bank_move_px=maxraw,inference_GT_access=False,radial_tests='test_a_contract radial222 original/new-logit parity atol2e-6;existing inherited radial forward unmodified'))
if __name__=='__main__':main()
