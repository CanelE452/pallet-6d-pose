"""Clean replay of the unchanged decoder arithmetic with corrected ramp rank.

Run after geometry and ROOT's first32 TRAIN calibration capture, in an empty
new output directory. Then decoder_completion.py writes the full TRAIN audit.
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import decoder_audit as D
import geometry as G

def repaired_half_pixel(source,Base):
    sample=G.load_module(Path(source)/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','replay_source_sampler').sample
    feature=torch.arange(8,dtype=torch.float64)[None,None,None,:].expand(1,1,4,8).clone()
    position=torch.tensor([[[[[12.,12.],[16.,12.],[20.,12.]]]]],dtype=torch.float64)
    got=sample(feature,position,torch.tensor([[32.,64.]],dtype=torch.float64),8).reshape(-1);expected=torch.tensor([1.,1.5,2.],dtype=torch.float64)
    native=torch.tensor([[[[1.,1.],[1.5,1.],[2.,1.]]]],dtype=torch.float64);grid=2*(native+.5)/native.new_tensor([8,4])-1
    ngot=F.grid_sample(feature,grid,align_corners=False).reshape(-1)
    return dict(source_feature_center_max_error=float((got-expected).abs().max()),native_integer_center_max_error=float((ngot-expected).abs().max()),source_feature_formula='2*prepared_position/(stride*feature_size)-1; centers=(index+0.5)*stride',native_formula='2*(native_pixel_index+0.5)/tile_size-1',global_half_pixel_coordinate_shift=False,native_tiles_resized=False)

def run(args):
    doc=Path(args.public_dir)
    G.write(doc/'DECODER_REPLAY_LOCK.json',dict(status='LOCKED_BEFORE_REPLAY',wrapper_sha256=G.sha(__file__),unchanged_arithmetic_sha256=G.sha(D.__file__),diagnostic_repair='source sampler position tensor rank only',no_model_configuration_changes=True))
    original_write=G.write
    def write(p,v):
        p=Path(p)
        if p.name=='DECODER_AUDIT.json':p=p.with_name('DECODER_OPERATOR_AUDIT.json')
        return original_write(p,v)
    D.half_pixel=repaired_half_pixel;G.write=write
    try:D.run(args)
    finally:G.write=original_write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
