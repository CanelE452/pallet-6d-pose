"""Independent input-only native/crop phase reconstruction; no images/models."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'


def independent_grid(box):
    x0,y0,x1,y1=map(float,box)
    center=np.array([(x0+x1)/2,(y0+y1)/2])
    step=np.array([(x1-x0)*1.25/64,(y1-y0)*1.25/64])
    x=center[0]+(np.arange(64)+.5-32)*step[0]
    y=center[1]+(np.arange(64)+.5-32)*step[1]
    xx,yy=np.meshgrid(x,y)
    xy=np.stack([xx,yy],-1)
    return xy,np.floor(xy+.5).astype(np.int64),center,step


def independent_mapping(query,side,box,hw):
    query=np.asarray(query,float)
    xx,yy=np.meshgrid((np.arange(9)-4)*side/9,(np.arange(9)-4)*side/9)
    xy=query[:,None,None,:]+np.stack([xx,yy],-1)
    pixel=np.floor(xy+.5).astype(np.int64)
    _,_,center,step=independent_grid(box)
    # Inverse map integer native label location to nearest crop-grid center.
    column=np.floor((pixel[...,0]-center[0])/step[0]+32).astype(np.int64)
    row=np.floor((pixel[...,1]-center[1])/step[1]+32).astype(np.int64)
    h,w=hw
    native=((xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&
            (pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h))
    valid=native&(column>=0)&(column<64)&(row>=0)&(row<64)
    i64=np.where(valid,row*64+column,0)
    i8=np.where(valid,(row//8)*8+column//8,0)
    chosen=center+(np.stack([column,row],-1)+.5-32)*step
    sampled_pixel=np.floor(chosen+.5).astype(np.int64)
    return i64,i8,valid,pixel,sampled_pixel,chosen,step


def main():
    spec=importlib.util.spec_from_file_location('audited_input_only_recipe',ROOT/'global_recipe.py')
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    manifest=V63/'sam_source/INPUTS.jsonl'
    inputs={r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    records={}
    for source in ['SYNTH','YCB']:
        for split in ['TRAIN','VALIDATION']:
            with(V64/'source_cache'/f'{source}_{split}_QUERIES.csv').open(newline='')as f:queries=list(csv.DictReader(f))
            delta=[];parity=[];validcount=0;total=0;stepvalues=[]
            for q in queries:
                item=inputs[q['id']]
                grid=independent_grid(item['box']);production=recipe.grid(item['box'])
                for a,b in zip(grid,production):assert np.allclose(a,b,atol=1e-11,rtol=0)
                i64,i8,good,pixel,sampled,chosen,step=independent_mapping([item['query'][int(q['query_index'])]],item['side'],item['box'],item['raw_hw'])
                p64,p8,pgood=recipe.local_mapping([item['query'][int(q['query_index'])]],item['side'],item['box'],item['raw_hw'])
                assert np.array_equal(i64,p64)and np.array_equal(i8,p8)and np.array_equal(good,pgood)
                total+=1;validcount+=int(good[0,4,4]);stepvalues.append(step)
                if good[0,4,4]:
                    difference=sampled[0,4,4]-pixel[0,4,4]
                    delta.append(float(np.linalg.norm(difference)))
                    parity.append(bool((difference==0).all()))
                    # Continuous nearest-bin distance cannot exceed half a step.
                    assert (np.abs(chosen[0,4,4]-pixel[0,4,4])<=step/2+1e-10).all()
            records[source+'_'+split]=dict(queries=total,center_global_mapping_valid=validcount,
                center_mapping_coverage=validcount/total,center_sampled_native_pixel_parity=int(sum(parity)),
                center_native_pixel_parity_rate=float(np.mean(parity)),
                center_nearest_global_native_pixel_L2_quantiles={str(x):float(np.quantile(delta,x))for x in [0,.5,.9,.95,1]},
                global_native_step_xy_min=np.min(stepvalues,0).tolist(),global_native_step_xy_max=np.max(stepvalues,0).tolist())
    output=dict(schema='V65_INDEPENDENT_GLOBAL_INPUT_MAPPING_AUDIT_V1',PASS=True,
                grids_and_all_local81_mappings_match=True,groups=records,
                semantics='Global64 predictions and8bottleneck are coarse context only; local native exact/r2 labels remain unchanged. Nonzero phase offsets are not errors or relabeled supervision.',
                border='Protocol intentionally permits partial bilinear raw RGB at invalid global centers; SAM/prompt zero, GT loss masked; local sampled global features/logits zero outside valid mapping.',
                source_mask_RGB_or_private_reads=0,new_GPU_model_training_PnP_calls=0,
                bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'global_recipe.py',ROOT/'PROTOCOL.json',manifest]})
    (HERE/'GLOBAL_MAPPING_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
