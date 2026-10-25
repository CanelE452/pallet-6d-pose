"""Invented data only; no source/cache/model/GT payloads."""
from pathlib import Path
import hashlib,json,os,tempfile,zipfile,io
import numpy as np

H=Path(__file__).absolute().parent
p=H/'build_gallery.py';raw=p.read_bytes();ns={'__name__':'invented_gallery','__file__':str(p)};exec(compile(raw,str(p),'exec'),ns)
rows=[]
for i,(fam,proof,parent) in enumerate([(0,True,80),(0,True,30),(0,False,7),(1,False,20),(1,False,12)]):
    rows.append({'pair_index':i,'parent_sample_index':parent,'proof_pass':proof,'original_input':{'fit':True,'target':0,'family_code':fam}})
cases=ns['select_cases'](rows)
assert [c['parent_sample_index'] for c in cases]==[30,7,None,12] and cases[2]['missing']
bad=[dict(r) for r in rows];bad[1]['parent_sample_index']=80
try:ns['select_cases'](bad)
except ValueError:pass
else:raise AssertionError('Duplicate parent accepted')
for forbidden in ('/never/V112/x','/never/__pycache__/x','/never/x.pyc','/never/a/../b'):
    try:ns['safe'](forbidden)
    except ValueError:pass
    else:raise AssertionError('Forbidden path accepted')

rng=np.random.Generator(np.random.PCG64(242));base=rng.integers(0,256,(128,160,3),dtype=np.uint8)
with tempfile.TemporaryDirectory(prefix='invented_gallery_',dir=H) as td:
    td=Path(td);os.environ['MPLCONFIGDIR']=str(td/'matplotlib')
    b=io.BytesIO();np.save(b,base,allow_pickle=False)
    zp=td/'invented.npz'
    with zipfile.ZipFile(zp,'w') as z:z.writestr('RGB_uint8.npy',b.getvalue());z.writestr('GT_TRAP.npy',b'NOT_AN_NPY_ARRAY_DO_NOT_READ')
    image={'path':str(zp),'sha256':hashlib.sha256(zp.read_bytes()).hexdigest(),'format':'npz','member':'RGB_uint8'}
    got=ns['decode_scene'](image,[image],base.shape[:2]);assert np.array_equal(got,base)
    chw=rng.integers(0,256,(3,96,96),dtype=np.uint8);proof=ns['save_rgb'](td/'RAW96.png',chw)
    assert proof['original_CHW_sha256']==proof['decoded_CHW_sha256']
    f=np.zeros((3,224,224),np.float32);f[0,0,:4]=[0,.49,.5,254.5]
    assert ns['display_context'](f)[0,:4,0].tolist()==[0,0,1,255]
    false=td/'false.json';false.write_text(json.dumps({'approved_V242_SOURCE_gallery':False}))
    try:ns['build'](false)
    except ValueError:pass
    else:raise AssertionError('False gallery entry accepted')

q8=[[45.+10*j,55.+6*j] for j in range(8)];group={'q8':q8,'L8':[96.]*8,'original_available8':[True]*7+[False],'box_xyxy':[45.,55.,115.,97.]}
node=rng.integers(0,256,(8,3,96,96),dtype=np.uint8);valid=np.ones((8,96,96),bool);valid[7]=False
ctx=rng.integers(0,256,(3,224,224),dtype=np.uint8).astype(np.float32)
packet={'NODE_RGB96':node,'NODE_VALID96':valid,'CONTEXT_RGB224_F32':ctx}
alpha=np.zeros((128,160),bool);alpha[35:75,25:65]=True
constructed_packet={**packet,'NATIVE_LAYER_ALPHA_DIAGNOSTIC':alpha}
row={'active_slot':0,'teacher_reference_xy':[45.25,55.75],'targets_D0_E1_Uminus1':[-1,-1],
     'proof_pass':False,'constructed':True,'status':'CONSTRUCTED_UNKNOWN','changed_pixels':{'COVER':{'native':100,'active_RAW96':50,'context_F32_samples':60},'ADJACENT_SHAM':{'native':100,'active_RAW96':40,'context_F32_samples':55}}}
case={'case_number':1,'family':0,'proof_pass':False,'eligible_population':3,'parent_sample_index':7,'invented_toy_only':True}
ns['render_card'](H/'INVENTED_CONSTRUCTED_UNKNOWN_CARD.png',case,row,group,{'BEFORE':packet,'COVER':constructed_packet,'ADJACENT_SHAM':constructed_packet})
row2={**row,'constructed':False,'status':'PROPOSAL_CONSTRUCTION_FAILED','changed_pixels':None}
ns['render_card'](H/'INVENTED_UNCONSTRUCTED_CARD.png',case,row2,group,{'BEFORE':packet})
result={'completed':True,'PASS':True,'groups':4,'fixed_bucket_min_parent_and_empty_no_substitution':True,'exact_RGB_member_only_toy_trap_not_decoded':True,'PNG_CHW_roundtrip_and_F32_display':True,'false_entry_refused':True,'constructed_UNKNOWN_and_unconstructed_layouts_rendered':True,
        'code_sha256':hashlib.sha256(raw).hexdigest(),'actual_SOURCE_GT_cache_model_or_pixel_reads':0,'actual_gallery_execution':0,'invented_only':True}
ns['write_json'](H/'TOY_RESULT.json',result);print(json.dumps(result))
