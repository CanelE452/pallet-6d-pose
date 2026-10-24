"""Code/generated-only independent review. Never executes private adapter main.

Only .py code is read; all image/row/model/gate examples are generated in RAM.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
ADAPTER=ROOT/'private_evaluation/provenance/PRE_BINDING_AND_SENTINEL_FIX_V1/prepare_and_infer_after_freeze.py'
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
BOX=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
CONTEXT=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
SOURCEPACK=ROOT/'synth1003_input_pack_code/build_synth1003_input_pack.py'

def bind(p):
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def function(path,name,extra=None):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    env={'np':np,'ast':ast,'Path':Path}
    if extra:env.update(extra)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]

def scalar_crop(rgb,q,L,n,iy,ix):
    h,w=rgb.shape[:2]
    x=float(q[0])+((ix+.5)/n-.5)*L;y=float(q[1])+((iy+.5)/n-.5)*L
    valid=(-.5<=x<w-.5 and -.5<=y<h-.5)
    if not valid:return np.zeros(3,np.uint8),False
    x0=int(np.floor(x));y0=int(np.floor(y));acc=np.zeros(3,np.float64)
    # Preserve arithmetic sum order of independent scalar bilinear definition.
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        px=x0+dx;py=y0+dy
        if 0<=px<w and 0<=py<h:
            wx=(x-x0)if dx else 1-(x-x0);wy=(y-y0)if dy else 1-(y-y0)
            acc+=rgb[py,px].astype(float)*wx*wy
    return np.floor(np.clip(acc,0,255)+.5).astype(np.uint8),True

def main():
    out=OUT/'CODE_ONLY_GENERATED_REVIEW_V1.json'
    assert not out.exists()
    extract=function(ADAPTER,'extract')
    crop=extract(CROP,{'crop'})['crop']
    box=extract(BOX,{'initial_query_box'})['initial_query_box']
    context=extract(CONTEXT,{'input_context_features'})['input_context_features']
    source_compile=function(SOURCEPACK,'compile_frame',{'initial_query_box':box})
    rng=np.random.default_rng(20261023)
    pixel_tests=0;crop_cases=0
    for hw in [(9,17),(37,53),(480,640)]:
        rgb=rng.integers(0,256,(*hw,3),np.uint8)
        for q in [[0.,0.],[-.5,-.5],[hw[1]-.5,hw[0]-.5],[hw[1]/2+.137,hw[0]/2+.291],[-30.,7.],[hw[1]+30.,hw[0]+4.]]:
            for L in [16.,63.73]:
                got,valid=crop(rgb,np.asarray(q),L,128)
                assert got.shape==(3,128,128)and valid.shape==(1,128,128)and got.dtype==np.uint8
                for iy in [0,1,31,63,64,95,126,127]:
                    for ix in [0,1,31,63,64,95,126,127]:
                        expected,ev=scalar_crop(rgb,q,L,128,iy,ix)
                        np.testing.assert_array_equal(got[:,iy,ix],expected)
                        assert bool(valid[0,iy,ix])==ev
                        pixel_tests+=1
                crop_cases+=1
    qbase=np.asarray([[20.,20.],[60.,20.],[60.,50.],[20.,50.],[25.,25.],[55.,25.],[55.,45.],[25.,45.]])
    box_cases=[qbase,qbase+[-40.,0.],qbase+[1000.,0.],np.full((8,2),np.nan),np.full((8,2),-1.)]
    for q in box_cases:
        q0=q.copy();g=np.isfinite(q).all(1)&~(q==-1).all(1)
        actual=box(q,[80,100])
        expected=None
        if g.sum()>=2:
            lo=np.maximum(np.min(q[g],axis=0),[0.,0.]);hi=np.minimum(np.max(q[g],axis=0),[99.,79.])
            if np.all(hi-lo>=2):expected=np.concatenate([lo,hi])
        if expected is None:assert actual is None
        else:np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(q,q0)
    # Actual source compiler's context preprocessing versus proposed private path.
    rgb=rng.integers(0,256,(80,100,3),np.uint8);roi=np.zeros((8,65,9,9),np.float16);roi[:,64]=1
    source_context_tests=[]
    for name,q in [('finite',qbase),('finite_outside',qbase+[-40.,0.]),('one_nonfinite',qbase.copy()),('one_missing_sentinel',qbase.copy())]:
        if name=='one_nonfinite':q[2]=np.nan
        if name=='one_missing_sentinel':q[2]=[-1.,-1.]
        q9=np.vstack([q,[[0.,0.]]]);L=16.
        source=source_compile(rgb,q9,L,roi,np.ones(8,bool),context,crop)
        b=box(q,[80,100]);safe=np.zeros(4)if b is None else b
        private=np.stack([context(q,k,safe,[80,100])for k in range(8)])
        diff=float(np.max(np.abs(source['context42']-private)))
        source_context_tests.append({'case':name,'context42_maxabs_difference':diff,'equal':bool(np.array_equal(source['context42'],private))})
    assert all(x['equal']for x in source_context_tests if x['case']!='one_missing_sentinel')
    # Same adapter row export on generated slots, no prior prediction/GT files.
    tree=ast.parse(ADAPTER.read_text());main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    row_loop=next(n for n in main_node.body if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='enumerate'and any(isinstance(a,ast.Name)and a.id=='priorrows'for a in n.iter.args))
    z=np.asarray([-2.,0.,2.,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(0),np.float32(-1)),4.],np.float32)
    av=np.asarray([True,True,True,True,True,False]);oldvalid=[True,True,False,True,True,True]
    prior=[{'id':'generated','native_corner':i,'query_x':i+.125,'query_y':i+.25,'image_sha256':'generated',
            'valid':oldvalid[i],'geometry_self_hidden':i%2==0,'geometry_source':'SAVED_INITIAL_POSE','logit':None,'p_visible':None}for i in range(6)]
    env={'np':np,'z':z,'av':av,'priorrows':prior,'rows':[]}
    exec(compile(ast.Module(body=[row_loop],type_ignores=[]),str(ADAPTER),'exec'),env)
    for i,r in enumerate(env['rows']):
        assert r['valid']==(oldvalid[i]and bool(av[i]))
        for key in ['id','native_corner','query_x','query_y','image_sha256','geometry_self_hidden','geometry_source']:assert r[key]==prior[i][key]
        assert r['status']==('OK'if r['valid']else'UNKNOWN')
        if r['valid']:assert r['present_at_frozen_threshold']==bool(z[i]>=0)
        else:assert r['logit']is None and r['p_visible']is None and r['present_at_frozen_threshold']is None
    # Execute only main's early guards against fake in-memory metadata.
    guard_cases=[]
    for case in ['gate_disapproved','freeze_incomplete','current_used','checkpoint_binding_fail']:
        events=[]
        def B(p):return {'path':str(p),'sha256':'generated'}
        F=Path('/generated/source/MODEL_FREEZE.json');R=Path('/generated/V76');A=Path('/generated/adapter.py')
        gate={'adapter_code':B(A),'approved':case!='gate_disapproved','source_model_freeze':B(F)}
        freeze={'completed':case!='freeze_incomplete','optimizer_updates':2000,'parameter_count':41237,
            'visible_probability_threshold':.5,'current_used_for_fit_selection':case=='current_used','PnP':0,
            **{k:B('/generated/source/'+k)for k in ['checkpoint','model_code','protocol','trainer','entry','source_inputs']}}
        def load(p):
            events.append('load:'+str(p))
            if str(p).endswith('POST_SOURCE_FREEZE_GATE.json'):return gate
            if Path(p)==F:return freeze
            raise RuntimeError('Unexpected private access in a failing generated preflight')
        def check(d):
            events.append('check:'+d['path'])
            if case=='checkpoint_binding_fail'and d==freeze['checkpoint']:raise AssertionError('generated badhash')
        fake_main=function(ADAPTER,'main',{'ROOT':R,'F':F,'OUT':Path('/generated/out'),'OLD':Path('/generated/V72'),
            '__file__':str(A),'load':load,'check':check,'b':B})
        failed=False
        try:fake_main()
        except AssertionError:failed=True
        assert failed and not any('/generated/V72' in e for e in events)
        guard_cases.append({'case':case,'rejects_before_old_private_access':True,'events':events})
    source=ADAPTER.read_text()
    result={'scope':'CODE+GENERATED ONLY; no current/source RGB, inputarrays, GT, oldmetadata, weights or realmodel execution',
        'bindings':{p.name:bind(p)for p in [ADAPTER,CROP,BOX,CONTEXT,SOURCEPACK,Path(__file__)]},
        'generated':{'AST_extract_imports_only_numpy':True,'crop_cases':crop_cases,'scalar_RGB_pixels':pixel_tests,
            'crop_maxdiff':0,'box_cases':len(box_cases),'context_vs_actual_source_compiler':source_context_tests,
            'export_generated_slots':6,'strict_zero_and_subnormal_signs_preserved':True,'savedgeometry_q_identity_preserved':True,
            'early_guard_cases':guard_cases},
        'findings':[{'id':'UNBOUND_HELPER_AND_PRIOR_SEALS','severity':'pre-inference provenance blocker',
            'detail':'Current adapter binds self+MODEL_FREEZE, but hardcoded crop/box/context .py and oldadaptergate/oldpreparationseal have no expected binding check in newgate. Bind/check before private input read.'},
            {'id':'MISSING_SENTINEL_CONTEXT_DIFFERENCE','severity':'conditional input parity blocker',
            'detail':'Generated(-1,-1) source-missing q is converted toNaN before context42 by sourcecompiler; private H(q) treats it as finite and changes scale/relativegeometry. Either use same missing preprocessing for context only while preserving rawq, or assert beforeforward that no exact(-1,-1) is present and disclose rejection.'}],
        'GT_read_path_found':False,'pose_or_geometry_recompute_found':False,'no_PnP':True,
        'runtime_precision':'Code preserves originalfloat64q/L, checks oldfloat32q/L, casts model outputfloat32, stablefloat64sigmoid, logit>=0 tieVISIBLE. No precision conversion bug found.',
        'score_scope':'predicted_D_or_E is evaluator compatibility naming; raw model target remains reference-local support, not semantic E/S.',
        'critical_status':'HOLD pending guard/sentinel resolution; arithmetic generatedchecks otherwisePASS'}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'context_cases':source_context_tests,'scalar_pixels':pixel_tests}))

if __name__=='__main__':main()
