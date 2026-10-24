"""Generated RGB/logit smoke; no private/source image or annotation is opened."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));import infer_current_frozen as api
recipe=api.import_bound({'path':str(ROOT/'global_recipe.py'),'sha256':api.sha(ROOT/'global_recipe.py')},'smoke_recipe')

def main():
    h,w=91,117;yy,xx=np.mgrid[:h,:w]
    rgb=np.stack((xx%256,yy%256,(xx+2*yy)%256),-1).astype(np.uint8)
    logits=((xx-50)*.17+(yy-40)*.09).astype(np.float32)
    q=np.array([[18.2,16.7],[92.1,16.1],[91.8,74.4],[18.4,74.2],[30.1,25.9],[81.3,25.5],[81.1,64.3],[30.5,64.1]])
    box=api.initial_query_box(q,[h,w]);side=24.7;f=api.input_features(rgb,q,side,box,logits,recipe)
    p=np.floor(q+.5).astype(int)
    assert np.array_equal(f['rgb_native'][:,[12,37,62],4,4],rgb[p[:,1],p[:,0]])
    assert np.array_equal(f['sam'][:,0,4,4],np.clip(logits[p[:,1],p[:,0]]/8,-4,4).astype(np.float16))
    for k,(px,py)in enumerate(p):
        assert f['sam'][k,1,4,4]==np.float16(np.clip(logits[py-2:py+3,px-2:px+3].max()/8,-4,4))
    xy,gp,_,_=recipe.grid(box);gv=recipe.native_valid(xy,gp,[h,w])
    expected=np.where(gv,np.clip(logits[np.clip(gp[...,1],0,h-1),np.clip(gp[...,0],0,w-1)]/8,-4,4),0).astype(np.float16)
    assert np.array_equal(f['global_sam'][0,0],expected)
    assert np.array_equal(f['global_index8'][f['global_query_valid']],(((f['global_index64']//64)//8)*8+(f['global_index64']%64)//8)[f['global_query_valid']])
    rng=np.random.default_rng(20261022);x=rng.normal(size=(1,8,65,9,9)).astype(np.float16);x[:,:,64]=1
    context=api.context64(x);manual=x[:,:,:64].astype(np.float32).sum((1,3,4))/(8*81)
    assert np.array_equal(context,manual)
    badq=q.copy();badq[0]=[-1,-1];badq[1]=[w-.1,4]
    valid,_=api.query_validity(badq,[h,w],[True]*8);assert not valid[0]and not valid[1]
    z=np.random.default_rng(4).normal(size=(100,6))*40;scores=api.probabilities(z)
    assert(scores['external']>=0).all()and(scores['outside']>=0).all()
    assert np.allclose(scores['visible']+scores['external']+scores['outside'],1)
    assert(scores['visible'][:,0]<=scores['visible'][:,1]+1e-15).all()
    with tempfile.TemporaryDirectory()as tmp:
        gate=Path(tmp)/'unapproved.json';api.write(gate,{'approved':False})
        try:api.validate_gate(gate)
        except PermissionError:pass
        else:raise AssertionError('unapproved gate accepted')
    # The exact selected source checkpoint runs only on generated features here.
    source_seal=json.loads((ROOT/'SOURCE_SELECTION_SEAL.json').read_text())
    selected=next(r for r in source_seal['candidates']if r['candidate']==api.SELECTED)
    freeze=json.loads(Path(selected['freeze']['path']).read_text())
    modelpath=Path(selected['freeze']['path']).parent/'code/query_global_amodal_head.py'
    head=api.import_bound({'path':str(modelpath),'sha256':freeze['code_sha256'][modelpath.name]},'smoke_head')
    model,_=head.load_model(selected['checkpoint']['path'],'cpu');model.eval().requires_grad_(False)
    z=api.forward_frame(model,x[0],context[0],f,'cpu');repeat=api.forward_frame(model,x[0],context[0],f,'cpu')
    assert z.shape==(8,6)and np.isfinite(z).all()and np.array_equal(z,repeat)
    result={'PASS':True,'generated_RGB_native_center_exact':True,'native_SAM_exact_r2':True,'global_SAM_phase_exact':True,
        'mapping_64_to_8_per_axis':True,'context64_exact':True,'invalid_nearest_query_rejected':True,'direct_probability_marginals':True,
        'unapproved_gate_rejected_before_input_open':True,'frozen_selected_model_generated_feature_repeat_exact':True,
        'private_RGB_cache_or_GT_reads':0,'source_RGB_or_GT_reads':0,'optimizer_updates':0,'PnP_calls':0,
        'code_sha256':api.sha(HERE/'infer_current_frozen.py')}
    api.write(HERE/'MOCK_INPUT_PIPELINE_SMOKE.json',result);print(json.dumps(result))

if __name__=='__main__':main()
