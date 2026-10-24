"""Generated arrays/metadata only. No real cache, image, model, or score opened."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARENT=ROOT.parent
spec=importlib.util.spec_from_file_location('adapter',HERE/'current_input_adapter.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def isolated_function(path,name):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def main():
    out=HERE/'GENERATED_ADAPTER_TESTS.json'
    if out.exists():raise FileExistsError(out)
    sampler_path=PARENT/'query_visibility_reference_jitter_20261023_v71/data/ycb_proposal_grid/build_proposal_grid.py'
    v65=PARENT/'query_visibility_amodal_global_20261022_v65/inference/infer_current_frozen.py'
    sampler=isolated_function(sampler_path,'crop96')
    oldbox=isolated_function(v65,'initial_query_box')
    oldvalid=isolated_function(v65,'query_validity')
    rng=np.random.default_rng(20261023);sampled_pixels=0;cases=0
    for h,w in [(13,19),(32,45),(71,64),(2,3),(96,96),(17,9)]:
        rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        for q in [[0,0],[w-.49,h-.49],[-.3,2.2],[w+2,h+3],[w/2+.13,h/2-.37]]:
            for L in [1.,9.,32.06620300493507,96.]:
                x,v=a.crop96(rgb,q,L);xx,vv=sampler(rgb,q,L)
                assert np.array_equal(x,xx) and np.array_equal(v,vv)
                sampled_pixels+=96*96;cases+=1
    q=np.array([[2.2,2.1],[15.3,3.4],[17.1,18.2],[2.5,17.5],[-1,-1],[np.nan,3.],[31.8,2.],[8.,8.]])
    x=rng.normal(size=(8,65,9,9)).astype(np.float32);x[:,64]=1;x[7,64]=0
    rgb=rng.integers(0,256,(24,32,3),dtype=np.uint8)
    frame=dict(id='GENERATED',initial_points=q.tolist(),raw_hw=[24,32],image_sha256='0'*64)
    meta=dict(id='GENERATED',side=17.125)
    geometry=SimpleNamespace(conditional_self_hidden=lambda *args:np.zeros(8,bool))
    f,rows,call=a.adapt_frame(rgb,frame,meta,x,np.ones(8,bool),geometry)
    assert set(f)==a.FORWARD_KEYS and np.array_equal(f['query_xy'],q.astype(np.float32),equal_nan=True)
    assert np.array_equal(a.initial_query_box(q,[24,32]),oldbox(q,[24,32]))
    valid,pixel=a.query_validity(q,[24,32],np.ones(8,bool));vv,pp=oldvalid(q,[24,32],np.ones(8,bool))
    assert np.array_equal(valid,vv) and np.array_equal(pixel,pp)
    assert [r['valid'] for r in rows]==[True,True,True,True,False,False,False,False]
    assert all(r['p_visible'] is None and r['p_D_given_DE'] is None for r in rows)
    assert rows[7]['status']=='UNKNOWN' and call[7]  # raw network allowed, main decision withheld
    frame2=dict(frame,reference_xy=[0,0],semantic_state='E',alpha='HIDDEN_GT',initial_pose=dict(available=False))
    f2,_,_=a.adapt_frame(rgb,frame2,meta,x,np.ones(8,bool),geometry)
    assert all(np.array_equal(f[k],f2[k],equal_nan=True) for k in f)
    bad=dict(frame,initial_points=np.full((8,2),-1.).tolist())
    bf,br,bc=a.adapt_frame(rgb,bad,meta,x,np.ones(8,bool),geometry)
    assert not bc.any() and all(r['reason']=='MISSING_OR_DEGENERATE_QUERY_BOX' for r in br)
    with tempfile.TemporaryDirectory(prefix='V72_GENERATED_GATE_') as tmp:
        p=Path(tmp)
        def b(name,text='GENERATED'):
            path=p/name;path.write_text(text);return a.bind(path)
        code=b('fake_model.py');checkpoint=b('fake_checkpoint.pt');trainer=b('fake_trainer.py')
        protocol=a.bind(ROOT/'FIT_PROTOCOL.json')
        freeze=dict(completed=True,optimizer_updates=1000,parameter_count=40682,current_used_for_fit_selection=False,current_used_for_threshold=False,PnP_calls=0,
            threshold_probability_D=.5,threshold_logit=0.,tie_is_D=True,checkpoint=checkpoint,model_code=code,trainer_code=trainer,protocol=protocol)
        fb=b('freeze.json',json.dumps(freeze))
        private={k:dict(path=str(p/('DO_NOT_OPEN_'+k)),sha256='0'*64) for k in a.INPUT_KEYS}
        prior=b('prior.json',json.dumps(dict(inputs=private)))
        gate=dict(approved=True,source_selection_complete=True,allow_current_evaluation_inference=True,phase='FROZEN_ONE_SHOT_DIAGNOSTIC',
            current_used_for_model_or_threshold_selection=False,threshold_probability_D=.5,threshold_logit=0.,tie_is_D=True,expected_frames=223,expected_slots=1784,
            code=a.bind(HERE/'current_input_adapter.py'),source_selection_seal=fb,source_model_freeze=fb,checkpoint=checkpoint,model_code=code,source_protocol=protocol,
            sampler_recipe=a.bind(sampler_path),geometry_helper=b('fake_geometry.py'),prior_v63_input_gate=prior,inputs=private,
            runtime=dict(device='cpu',num_threads=1,executable='UNUSED',torch_version='UNUSED'),
            roughbox_rule='V65_INITIAL_Q8_MINMAX_CLIP_UNCHANGED',RGB96_recipe='V71_YCB_NATIVE_PIXEL_CENTER_BILINEAR_CONSTANT0_UINT8_ROUND')
        gp=p/'gate.json';a.write(gp,gate);a.validate_gate(gp)
        rejected=[]
        for field,value in [('approved',False),('source_selection_complete',False),('allow_current_evaluation_inference',False),('threshold_probability_D',.6),('expected_frames',222),('current_used_for_model_or_threshold_selection',True)]:
            badg=dict(gate);badg[field]=value;a.write(gp,badg)
            try:a.validate_gate(gp)
            except PermissionError:rejected.append(field)
            else:raise AssertionError('Gate accepted '+field)
        assert len(rejected)==6 and not any(Path(r['path']).exists() for r in private.values())
    a.write(out,dict(PASS=True,code=a.bind(HERE/'current_input_adapter.py'),test_code=a.bind(__file__),
        generated_crop_cases=cases,generated_crop_pixels=sampled_pixels,V71_sampler_bit_differences=0,V65_box_and_validity_differences=0,
        original_q_L_unchanged=True,head_forward_exact_input_whitelist=True,GT_mutation_changes_head_inputs=False,
        invalid_main_scores_blank=True,empty_ROI_unknown=True,missing_box_unknown=True,
        valid_gate_does_not_open_private_paths=True,rejected_generated_gate_mutations=rejected,
        actual_current_source_RGB_cache_or_weights_read=0,model_forward_fit_PnP=0))
    print(json.dumps(a.bind(out)))


if __name__=='__main__':main()
