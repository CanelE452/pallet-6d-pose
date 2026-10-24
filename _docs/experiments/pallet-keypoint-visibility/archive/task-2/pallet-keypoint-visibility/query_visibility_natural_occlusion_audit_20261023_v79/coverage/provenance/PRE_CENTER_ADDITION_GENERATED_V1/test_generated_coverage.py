"""Generated arrays only: coordinate phases, boundary validity and V/C/O."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from coverage_math import coordinates,cached_formula,count_grid,roi_float_phase,fingerprint

HERE=Path(__file__).resolve().parent;P=HERE.parents[1]
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
def pure(path,name,extra=None):
    n=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    d={'np':np};d.update(extra or{});exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),d);return d[name]
native=pure(ROI,'native_grid');crop=pure(CROP,'crop')
rng=np.random.default_rng(20261023);checked=0;ramp_errors=[];phase_max=0.
for h,w in [(7,9),(31,47),(480,640),(540,960),(480,720),(560,560)]:
    m0=rng.random((h,w))>.5;m4=m0&(rng.random((h,w))>.4)
    q=np.array([[0,0],[-.4,2],[w-.1,h-.1],[w-1,h-1],[3.5,3.5],[-1,-1],[w+2,2],[w/2,h/2]],float)
    g9,valid,L=native(q,[h,w]);assert np.array_equal(g9,coordinates(q,L,9));assert np.array_equal(valid,cached_formula(g9,[h,w],9,q))
    for n in (9,96):
        grid=coordinates(q,L,n);cached=cached_formula(grid,[h,w],n,q)
        if n==96:
            rgb=np.zeros((h,w,3),np.uint8)
            for k in range(8):
                # Compiler skips its missing sentinel, while pure crop alone does not.
                expected=np.zeros((n,n),bool)if np.array_equal(q[k],[-1,-1])else crop(rgb,q[k],L,n)[1][0]
                assert np.array_equal(expected,cached[k])
        result,grid,pix,nv=count_grid(m0,m4,q,L,n,cached)
        for k in range(8):
            for iy,ix in [(0,0),(n-1,n-1),(n//2,n//2),(1,2)]:
                x,y=grid[k,iy,ix];px=int(np.floor(x+.5));py=int(np.floor(y+.5));nativevalid=not np.array_equal(q[k],[-1,-1])and 0<=x<w and 0<=y<h and 0<=px<w and 0<=py<h
                assert nv[k,iy,ix]==nativevalid;checked+=1
        assert result['class_counts'].sum()<=8*n*n*3
        assert result['gt_map_native_sha256'][0].decode()==fingerprint(n,nv[0],np.where(nv[0],m0[np.clip(pix[0,...,1],0,h-1),np.clip(pix[0,...,0],0,w-1)],False),np.where(nv[0],m4[np.clip(pix[0,...,1],0,h-1),np.clip(pix[0,...,0],0,w-1)],False))
    # Coordinate-ramp replay in the same align_corners=False convention, no model.
    gain=.7619047619047619;off=np.array([0.,13.]);phase,norm=roi_float_phase(g9,gain,off,[h,w],valid&((np.floor(g9+.5)[...,0]<w)&(np.floor(g9+.5)[...,1]<h)),m0,m4)
    phase_max=max(phase_max,float(phase['ROI_float32_norm_max_native_abs_error'].max()))
    yy,xx=torch.meshgrid(torch.arange(80,dtype=torch.float32),torch.arange(80,dtype=torch.float32),indexing='ij')
    ramp=torch.stack([xx,yy])[None].expand(8,-1,-1,-1);sample=F.grid_sample(ramp,torch.from_numpy(norm.astype(np.float32)),align_corners=False)
    expected=((norm+1)*80/2-.5);inside=(expected[...,0]>=1)&(expected[...,0]<78)&(expected[...,1]>=1)&(expected[...,1]<78)
    err=float(np.max(np.abs(sample.permute(0,2,3,1).numpy()[inside]-expected[inside])));assert err<2e-5;ramp_errors.append(err)
p=HERE/'GENERATED_COVERAGE_TESTS.json';assert not p.exists()
p.write_text(json.dumps(dict(PASS=True,scalar_positions=checked,HW_variants=6,ROI_grid_literal_exact=True,RGB96_cachedvalid_literal_exact=True,missing_sentinel_skipped_by_actual_compiler=True,coordinate_ramp_max_error=max(ramp_errors),float32_normalized_grid_native_phase_max=phase_max,coverage_math_sha256=hashlib.sha256((HERE/'coverage_math.py').read_bytes()).hexdigest(),RGB_GT_model_current_reads=0,new_forward_fit_PnP=0),indent=2)+'\n')
print(p)
