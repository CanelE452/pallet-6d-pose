"""Saved-only byte/crop/order QA; never renders or invokes learned models."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,json,collections
import numpy as np
from render_corpus_once import OUT,H,rows,read,bind,check,write,crop_operator
from render_math import STAGES,array_sha

def run(output):
    from PIL import Image
    seal=read(OUT/'RENDER_SEAL.json');assert seal['completed']and seal['PASS_execution']
    for b in seal['files']:check(b)
    manifest=read(check(seal['manifest']));rr=rows(check(manifest['full_RGB']));qq=rows(check(manifest['input_occurrences']));base=rows(check(manifest['query_inputs']))
    assert len(rr)==288 and len(qq)==2304 and len(base)==96
    crop,_=crop_operator();crop_N=0;valid_N=0;split=collections.Counter();partial=0;zero=0;current_chunk=None
    for i,r in enumerate(rr):
        b=i//3;s=i%3
        assert r['image_index']==i and r['base_index']==b and r['variant']==STAGES[s]
        assert r['query_inputs']==manifest['query_inputs']and r['query_input_row']==b and r['split']==base[b]['split']
        rgb=np.array(Image.open(check(r['image'])).convert('RGB'),copy=True)
        assert rgb.shape==(640,640,3)and array_sha(rgb)==r['decoded_RGB_sha256']
        ci=b//8
        if current_chunk!=ci:
            if current_chunk is not None:z.close()
            desc=manifest['chunks'][ci];z=np.load(check(desc['file']),allow_pickle=False);current_chunk=ci
            arrays={k:z[k]for k in z.files}
            assert arrays['rgb96'].shape==(8,3,8,3,96,96)and arrays['crop_valid96'].shape==(8,3,8,1,96,96)
            assert arrays['rgb96'].dtype==np.uint8 and arrays['crop_valid96'].dtype==bool
            assert arrays['stage'].tolist()==list(STAGES)
        for k in range(8):
            j=24*b+8*s+k;q=qq[j];xy=base[b]['query8_xy'][k];L=base[b]['L'][k]
            assert q['occurrence_index']==j and q['image_index']==i and q['aux_corner_index']==k and q['variant']==STAGES[s]
            assert q['query_xy']==xy and q['L']==L and q['chunk_index']==ci and q['index_in_chunk_base']==b%8 and q['stage_index']==s
            assert np.array_equal(arrays['query8_xy'][b%8,k],xy)and arrays['L'][b%8,k]==L
            cc,vv=crop(rgb,np.asarray(xy),L,96);cached=arrays['rgb96'][b%8,s,k];valid=arrays['crop_valid96'][b%8,s,k]
            assert np.array_equal(cc,cached)and np.array_equal(vv,valid)
            assert array_sha(cc)==q['rgb96_sha256']and array_sha(vv)==q['crop_valid96_sha256']
            crop_N+=1;valid_N+=int(vv.sum());partial+=int(vv.any()and not vv.all());zero+=int(not vv.any());split[q['split']]+=1
    z.close()
    result=dict(completed=True,PASS=True,code=bind(__file__),render_seal=bind(OUT/'RENDER_SEAL.json'),manifest=seal['manifest'],full_RGB_decoded_SHA_replays=288,
        all_query_RGB96_and_valid_exact_replays=crop_N,q_L_identity_order_parity=True,query_counts_by_split=dict(split),valid96_cell_count=valid_N,
        partial_validity_crops=partial,zero_validity_crops=zero,new_renders_models_features_fit_private_current_PnP=0,
        scope='Saved bytes/order/photometric crop replay only; not human semantic gold or independent geometry witness validation.')
    output.mkdir();write(output/'INPUT_OUTPUT_QA.json',result)
    write(output/'QA_SEAL.json',dict(completed=True,PASS=True,QA=bind(output/'INPUT_OUTPUT_QA.json'),render_seal=bind(OUT/'RENDER_SEAL.json')))
    print(json.dumps(result))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'saved_output_QA');a=ap.parse_args();run(a.output)
