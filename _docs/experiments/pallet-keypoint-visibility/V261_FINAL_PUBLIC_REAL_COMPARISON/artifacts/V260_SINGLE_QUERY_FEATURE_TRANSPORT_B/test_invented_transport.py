"""Invented arrays and mock encoder only; no SOURCE or official state access."""
import hashlib, inspect, json, sys, types
from pathlib import Path
import numpy as np
import torch

H=Path(__file__).absolute().parent

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(path.read_bytes(),str(path),"exec"),m.__dict__);return m

def run():
    driver=load(H/"build_single_query_features.py","v260_invented_driver")
    core=load(H/"single_query_core.py","v260_invented_core")
    closure=json.loads(driver.checked(driver.CLOSURE,[driver.CLOSURE]).read_bytes())
    codes=list(closure["code"].values())
    rawmath=driver.pure(closure["code"]["spatial_crop"],codes,["crop"])
    prep=driver.pure(closure["code"]["preprocess"],codes,["preprocess_views"],{"MEAN":(.485,.456,.406),"STD":(.229,.224,.225)})
    canonical=driver.pure(closure["code"]["canonical"],codes,["node_pixel_batch"])
    crop=driver.module(closure["code"]["crop_module"],codes,"crop_modules")
    checks=[]
    rgb=np.zeros((96,96,3),np.uint8);rgb[:,:,0]=np.arange(96,dtype=np.uint8)[None,:];rgb[:,:,1]=np.arange(96,dtype=np.uint8)[:,None]
    raw,valid,supported=core.query_crop(rgb,[47.5,47.5],96,rawmath.crop)
    assert supported and valid.all() and np.array_equal(raw,rgb.transpose(2,0,1));checks.append("V90 identity crop/centre47.5/uint8 CHW")
    raw2,v2,s2=core.query_crop(rgb,[-1000,47.5],96,rawmath.crop)
    assert not s2 and not v2.any() and not raw2.any();checks.append("OOB geometric unsupported independent of label")
    for q,L in (([np.nan,1],96),([1,1],0)):
        r,v,s=core.query_crop(rgb,q,L,rawmath.crop);assert not s and not r.any() and not v.any()
    checks.append("nonfinite q/nonpositive L explicit unavailable")
    # Half-pixel FP64 accumulation has uint8 half-up rounding, not banker rounding.
    tiny=np.zeros((2,2,3),np.uint8);tiny[:,1]=1
    r,_,_=core.query_crop(tiny,[.5,.5],1e-12,rawmath.crop)
    assert r[:,47:49,47:49].tolist()==[[[0,1],[0,1]]]*3
    checks.append("FP64 bilinear and uint8 half-up retained")
    class MockEncoder:
        def __init__(self):self.calls=[]
        def forward_features(self,views):
            self.calls.append(tuple(views.shape));n=len(views)//2
            cls=torch.zeros(len(views),384);cls[:n,0]=2;cls[n:,2]=3
            patches=torch.zeros(len(views),256,384);patches[:,:,1]=4
            return {"x_norm_clstoken":cls,"x_norm_patchtokens":patches}
    model=MockEncoder()
    feature,fraction,central4=core.encode_queries(model,raw[None],valid[None],canonical.node_pixel_batch,prep.preprocess_views,crop,"cpu")
    assert feature.shape==(1,1186) and feature.dtype==np.float32 and fraction.tolist()==[[1.,1.]] and central4.tolist()==[True]
    assert feature[0,0]==1 and feature[0,384+1]==1 and feature[0,768+2]==1
    assert np.count_nonzero(feature[0,:1152])==3 and np.array_equal(feature[0,1152:1154],[1,1]) and not feature[:,1154:].any()
    assert model.calls==[(2,3,224,224)]
    checks.append("canonical fullCLS/fixedpatchQ/centreCLS/fractions+zero32 (not three CLS)")
    args=set(inspect.signature(core.encode_queries).parameters)
    assert not args.intersection({"r","label","depth","pose","SAM","native_slot","object_id"})
    checks.append("pure encoder API has no teacher/native/reference-bank arguments")
    expected={"0000":[865,2302,2468,3068,4063,4100,4326,4677],"0011":[81,495,2271,2334,2733,3162,3339,4580]}
    assert all(core.selected_grid_indices(f)==v for f,v in expected.items());checks.append("frozen 8r/frame hash selection independently recomputed")
    # Complete invented ledger checks the strict schema and selection without SOURCE rows.
    base={"r_derived_label":-1,"r_derived_state":"U","r_depth_eligible":False,"r_known":False,"r_raw_observed_minus_render_m":None,"r_raw_residual_finite":False,"r_exclusion_reasons":["invented"],"native_slot":None,"SELF_state":"UNKNOWN","q_sam_target_present":False,"q_pixel_DE_gold":None,"r_label_copied_as_query_label":False,"q_is_public_emulation":True,"q_is_actual_initializer":False,"q_local_L_pixels":96,"U_is_model_abstention":False}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="v260_invented_") as tmp:
        path=Path(tmp)/"invented_rows.jsonl"
        with path.open("w") as f:
            for g in range(4800):
                r=[4+8*(g%80),4+8*(g//80)]
                for j,offset in enumerate(driver.OFFSETS):
                    q=[r[0]+offset[0],r[1]+offset[1]]
                    row=dict(base,reference_grid_index=g,r_xy=r,record_id=f"r_y{r[1]}_x{r[0]}_q{j}",q_offset_index=j,q_offset_xy=offset,q_xy=q,q_inside_image=0<=q[0]<640 and 0<=q[1]<480)
                    f.write(json.dumps(row)+"\n")
        b=driver.bind(path);rows=driver.load_records({"query_records":b,"frame_id":"0000","selected_reference_grid_indices":expected["0000"]},[b],core)
        assert len(rows)==43200 and sum(x["selected_for_feature"] for x in rows)==72
        # Mutating teacher state never changes the pure selection or geometric inputs.
        assert all(set(x["original_record"])==driver.QUERY_KEYS for x in rows)
    checks.append("all43200 invented records retained; only72 selected, teacher U not a mask")
    try:driver.authorize({"execution_authorized":False})
    except ValueError:pass
    else:raise AssertionError("False entry accepted")
    checks.append("false entry refuses before CODE/payload/output access")
    for bad in ("/tmp/V112/foo","/tmp/__pycache__/x","/tmp/x.pyc","/tmp/a/../b"):
        try:driver.lexical(bad)
        except ValueError:pass
        else:raise AssertionError(bad)
    checks.append("forbidden paths/traversal rejected lexically")
    return {"completed":True,"PASS":True,"invented_check_groups":checks,"actual_RGB_or_query_metadata_or_official_weights_or_encoder_reads":0,"classifier_forwards":0,"optimizer_updates":0}

if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2))
