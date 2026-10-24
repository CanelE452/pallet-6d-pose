"""CPU legacy generated-only exact ROI/crop/invalid tests; no source arrays."""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import torch

import build_actual_query_pack as packer

ROOT=Path(__file__).resolve().parent


def functions(path,names):
    nodes=[t for t in ast.parse(path.read_text()).body if isinstance(t,ast.FunctionDef)and t.name in names]
    assert len(nodes)==len(names)
    env={'np':np,'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


def scalar(rgb,q,L,y,x):
    h,w=rgb.shape[:2];xx=q[0]+((x+.5)/96-.5)*L;yy=q[1]+((y+.5)/96-.5)*L
    valid=-.5<=xx<w-.5 and -.5<=yy<h-.5
    if not valid:return np.zeros(3,np.uint8),False
    x0,y0=int(np.floor(xx)),int(np.floor(yy));dx,dy=xx-x0,yy-y0;value=np.zeros(3,np.float64)
    for ax,ay,wt in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        if 0<=x0+ax<w and 0<=y0+ay<h:value+=rgb[y0+ay,x0+ax].astype(np.float64)*wt
    return np.floor(np.clip(value,0,255)+.5).astype(np.uint8),True


def query_rows(q,L):
    rows=[]
    for k in range(8):
        if q is None:xy=None;continuous=nearest=rgbcentre=False;reason='NO_DETECTION_U';side=None
        else:
            xy=q[k].tolist();x,y=xy;continuous=bool(0<=x<640 and 0<=y<480)
            nearest=bool(0<=np.floor(x+.5)<640 and 0<=np.floor(y+.5)<480)
            rgbcentre=bool(-.5<=x<639.5 and -.5<=y<479.5);side=L
            reason='VALID_QUERY_NOT_A_VISIBILITY_LABEL'if continuous else'OOB_OR_INVALID_U'
        rows.append(dict(query_id=f'GENERATED:native{k}',frame_index=0,frame_input_id='GENERATED',pair_id='GENERATED_PAIR',view='before',split='TRAIN_LAB',native_slot=k,query_xy=xy,roi_side=side,original_query_continuous_valid=continuous,query_nearest_pixel_valid=nearest,query_rgb96_centre_domain_valid=rgbcentre,reason=reason,native_semantic_state='UNKNOWN',human_gold=False))
    return rows


def run():
    assert torch.__version__=='2.1.1+cu118'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    math=packer.math_module();legacy=functions(packer.LEGACY,{'native_grid','patches'});crop74=functions(packer.V74,{'crop'})['crop']
    rgb=np.arange(480*640*3,dtype=np.uint8).reshape(480,640,3)
    yy,xx=np.mgrid[:68,:80];p3=np.stack([(xx*.2+yy*.1+c*.03)for c in range(64)]).astype(np.float16)
    affine=math.expected_affine([480,640]);assert affine['input_shape']==[544,640]and affine['p3_actual_hw']==[68,80]and np.array_equal(affine['offset'],[0,13])
    record=dict(raw_hw=[480,640],P3_shape=[64,68,80],gain=affine['gain'],offset=affine['offset'],added_border=100,frame_index=0,frame_input_id='GENERATED',split='TRAIN_LAB',image_sha256='a'*64)
    regimes=[np.array([[200.+k*4,160.+k*2]for k in range(8)]),
             np.array([[0.,0.],[639.8,479.8],[-.1,240.],[320.,-.1],[640.1,240.],[320.,480.1],[639.49,479.49],[300.,200.]]),
             np.array([[-1.,-1.],[0.,0.],[640.,480.],[-50.,230.],[660.,200.],[320.,20.],[320.,460.],[250.,200.]])]
    roi_exact=0;scalar_points=0;finite_invalid_nonzero_crops=0
    for q in regimes:
        points=np.concatenate([q,np.zeros((1,2))]);grid,valid,L=legacy['native_grid'](points,[480,640])
        rows=query_rows(q,L);actual=packer.pack_frame(rgb,p3,rows,record)
        full=np.zeros((64,80,80),np.float16);full[:,:68,:80]=p3
        old,oldany,oldL=legacy['patches']({'added_border':100,'p3':torch.from_numpy(full)[None]},points,[480,640],record['gain'],np.array(record['offset']))
        assert oldL==L and np.array_equal(old.numpy(),actual['p3_roi'])and np.array_equal(oldany,actual['ROI_anyvalid'])
        assert np.array_equal(actual['p3_roi'][:,64].astype(bool),valid)
        assert np.array_equal(actual['ROI_center_valid'],valid[:,4,4]);roi_exact+=8
        for k in range(8):
            ref,rv=crop74(rgb,q[k],L,96)
            assert np.array_equal(ref,actual['rgb96'][k])and np.array_equal(rv,actual['crop_valid96'][k])
            assert np.array_equal(actual['query_xy_raw'][k],q[k])and actual['L_raw'][k]==L
            for y,x in [(0,0),(95,95),(47,47),(48,48),(21,72)]:
                value,good=scalar(rgb,q[k],L,y,x)
                assert np.array_equal(value,actual['rgb96'][k,:,y,x])and good==bool(actual['crop_valid96'][k,0,y,x]);scalar_points+=1
            if not rows[k]['original_query_continuous_valid']and actual['rgb96'][k].any():finite_invalid_nonzero_crops+=1
    assert finite_invalid_nonzero_crops>0
    missing=packer.pack_frame(rgb,p3,query_rows(None,None),record)
    assert not missing['query_present'].any()and np.isnan(missing['query_xy_raw']).all()and np.isnan(missing['L_raw']).all()
    assert not missing['rgb96'].any()and not missing['crop_valid96'].any()and not missing['p3_roi'].any()
    # Different factual flags are deliberately not aliases.
    assert regimes[1][1,0]>639.5
    edge=packer.pack_frame(rgb,p3,query_rows(regimes[1],legacy['native_grid'](np.concatenate([regimes[1],[[0.,0.]]]),[480,640])[2]),record)
    assert edge['query_continuous_valid'][1]and not edge['query_nearest_valid'][1]and edge['ROI_center_valid'][1]
    assert not edge['RGB_central4_valid'][1]and not edge['operator_centre_support_v94_draft'][1]
    # All entry flags must reject before any source/array reader.
    temporary=ROOT/'GENERATED_REJECTED_ENTRY.json';assert not temporary.exists();temporary.write_text('{"root_approved":false}')
    try:packer.run(temporary)
    except AssertionError:pass
    else:raise AssertionError('Unapproved entry accepted')
    temporary.unlink()
    calls={t.func.attr for t in ast.walk(ast.parse(Path(packer.__file__).read_text()))if isinstance(t,ast.Call)and isinstance(t.func,ast.Attribute)}
    assert not calls.intersection({'predict','forward','load_state_dict'})
    return dict(completed=True,PASS=True,code=packer.bind(Path(packer.__file__)),generated_code=packer.bind(Path(__file__)),math_code=packer.bind(packer.MATH),V60_code=packer.bind(packer.LEGACY),V74_code=packer.bind(packer.V74),runtime={'torch':torch.__version__,'device':'CPU','threads':1},exact_V60_ROIs=roi_exact,RGB96_V74_exact_queries=roi_exact,scalar_RGB_exact_positions=scalar_points,finite_centre_invalid_crops_preserved=finite_invalid_nonzero_crops,missing8_raw_nan_metadata_finite_zero_tensors=True,continuous_nearest_ROIcentre_RGBcentral4_distinguished=True,unapproved_entry_rejected_before_source_reads=True,original_q_L_preserved=True,actual_source_array_pixel_checkpoint_model_current_reads=0,actual_forward_fit_PnP=0)


if __name__=='__main__':
    result=run();dest=ROOT/'GENERATED_PACK_TESTS.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
