"""Invented NumPy scenes only; exact pure CODE pinned, never actual payloads."""
from pathlib import Path
import ast,hashlib,json,types
import numpy as np

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');P=R.parent;H=R/'V242_SCENE_MATH_REVIEW_A'
evidence=[]
def load(rel,sha,names=None):
    p=P/rel
    assert all(t not in str(p).casefold() for t in ('v112','__pycache__','.pyc'))
    for q in [*reversed(p.parents),p]:assert not q.is_symlink()
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
    evidence.append({'path':str(p),'sha256':sha,'size_bytes':len(raw)})
    tree=ast.parse(raw)
    if names is not None:
        tree=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    ns={'__name__':'invented_pure_math','np':np};exec(compile(tree,str(p),'exec'),ns)
    return types.SimpleNamespace(**ns)

m=load('publication_root_preparation/V241_SCENE_CF_PRODUCER_B/scene_cf_math.py','77b88de4311ff38c3baa480feb179b15a6e6ad8d9ea409701b98c6794fbd92c8')
crop=load('query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py','6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b',{'crop'}).crop
context=load('query_visibility_target_context_rgb_20261023_v107/target_context_math.py','aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3')
old=load('publication_root_preparation/V235_CF_COMPOSITOR_A/compositor.py','d940ff8cb901482a77761dfac638ab473a5841d682ef0150f1ece0ab8c5fe4f4')
scene=load('publication_root_preparation/V237_CONTEXT_COMPOSITION_A/context_scene_math.py','f48545c907cf0705a1cd9ee408e54f520f7eef4563fb166f20a1821b3fa9b30f')
checks=[]

def independent_pixels(r,hw):
    nearest=np.floor(r+.5).astype(int)
    expected={}
    groups={'continuous_reference':[r], 'nearest_half_up_3x3':[(nearest[0]+dx,nearest[1]+dy) for dy in (-1,0,1) for dx in (-1,0,1)]}
    for key,points in groups.items():
        out=set()
        for px,py in points:
            xx=[(int(np.floor(px)),1-(px-np.floor(px))),(int(np.floor(px))+1,px-np.floor(px))]
            yy=[(int(np.floor(py)),1-(py-np.floor(py))),(int(np.floor(py))+1,py-np.floor(py))]
            for x,wx in xx:
                for y,wy in yy:
                    if wx*wy>0:out.add((x,y))
        expected[key]=(sorted(out),all(0<=x<hw[1] and 0<=y<hw[0] for x,y in out))
    return expected

for r in (np.array([3.,3.]),np.array([3.25,3.75]),np.array([2.5,2.5]),np.array([-.25,3.])):
    got=m.positive_support_pixels(r.astype(np.float64),(5,5));expected=independent_pixels(r,(5,5))
    for key,(pixels,supported) in expected.items():
        assert got[key]['pixels_xy'].tolist()==[list(x) for x in pixels] and got[key]['supported']==supported
# At (3,3), integer 3x3 includes image-edge coordinate4; zero-weight neighbour5 is NOT a gate.
base5=np.zeros((5,5,3),np.uint8);ones=np.ones((5,5),bool);zero=np.zeros((5,5),bool)
assert m.renewed_teacher_proof(base5,base5,base5,ones,zero,np.array([3.,3.],np.float64))['PASS']
assert not m.renewed_teacher_proof(base5,base5,base5,ones,zero,np.array([4.,4.],np.float64))['PASS']
checks.append('Independent positive-weight supports, exact half-up ties, zero-weight edge ignored, positive OOB rejected')

y,x=np.mgrid[:256,:256];base=np.stack((x%256,y%256,(x+y)%256),2).astype(np.uint8)
q8=np.array([[128,128],[160,128],[-1,-1],[np.nan,2],[120,120],[120,120],[120,120],[120,120]],np.float64)
L8=np.array([96,96,16,96,np.nan,0,-1,96],np.float64);avail=np.array([1,0,0,0,0,0,0,0],bool)
g={'q8':q8,'L8':L8,'original_available8':avail,'box_xyxy':[100.,100.,160.,160.]}
packet=m.resample_all(base,g,crop,context)
for j in range(8):
    if np.isfinite(q8[j]).all() and np.isfinite(L8[j]) and L8[j]>0:
        a,v=crop(base,q8[j],L8[j],96)
    else:a=np.zeros((3,96,96),np.uint8);v=np.zeros((1,96,96),bool)
    assert np.array_equal(a,packet['NODE_RGB96'][j]) and np.array_equal(v[0],packet['NODE_VALID96'][j])
assert packet['NODE_RGB96'][1].any() and packet['NODE_VALID96'][1].all() and not packet['node_available'][1]
assert packet['NODE_RGB96'][2].any() and not packet['node_available'][2]
assert not packet['NODE_RGB96'][3:7].any() and not packet['NODE_VALID96'][3:7].any()
checks.append('All8 V237-compatible finite/L predicate, unavailable finite/sentinel arrays retained; flags and invalid zeros separate')

q8=np.array([[128,128],[160,128],[160,160],[128,160],[100,100],[120,100],[100,120],[120,120]],np.float64)
g={'q8':q8,'L8':np.full(8,96.,np.float64),'original_available8':np.array([1,1,1,1,1,1,1,0],bool),'box_xyxy':[100.,100.,160.,160.]}
before=m.resample_all(base,g,crop,context);donor=np.zeros((32,32,3),np.uint8);donor[:]=[12,221,31];mask=np.ones((32,32),bool)
ref=np.array([128.25,128.25],np.float64)
proposal=old.compose_counterfactual_pair(before['NODE_RGB96'][0],before['NODE_VALID96'][0],donor,mask,q8[0],np.float64(96),ref,'V241_INVENTED_1')
parent={'active_slot':0,'original_input':{'q':q8[0].tolist(),'L':96.,'row_id':'V241_INVENTED_1','fit':True,'target':0,'original_query_available':True,'crop_available':True},'same_V235_donor_id':'TOY','same_V235_donor_plan':m.plain(proposal['donor_plan']),'expected_BEFORE_RGB96_CHW_sha256':m.sha(before['NODE_RGB96'][0]),'expected_BEFORE_valid96_bool_sha256':m.sha(before['NODE_VALID96'][0]),'teacher_reference_xy':ref.tolist(),'teacher_authority':'INVENTED_ONLY'}
r=m.compose_scene_pair(base,g,parent,donor,mask,'TOY',crop,context,old,scene)
assert r['proof']['PASS'] and not r['training_admission'] and not r['new_native_gold'] and not r['old5367_proof_inherited']
for role in ('COVER','ADJACENT_SHAM'):
    native=r['native_scenes'][role];alpha=r['native_alpha'][role]
    assert np.array_equal(native[~alpha],base[~alpha])
    for j in range(8):
        a,v=crop(native,q8[j],np.float64(96),96)
        assert np.array_equal(a,r['inputs'][role]['NODE_RGB96'][j]) and np.array_equal(v[0],r['inputs'][role]['NODE_VALID96'][j])
    a,v=context.sample_target_context(native,g['box_xyxy'])
    assert np.array_equal(a.transpose(2,0,1),r['inputs'][role]['CONTEXT_RGB224_F32']) and np.array_equal(v,r['inputs'][role]['CONTEXT_VALID224'])
    # Independently vary clean behind-layer values. Opaque native pixels still carry identical donor RGB.
    changed_base=base.copy();changed_base[alpha]=255-changed_base[alpha]
    recomposed=changed_base.copy();recomposed[alpha]=native[alpha]
    assert np.array_equal(recomposed,native)
assert not np.array_equal(r['inputs']['COVER']['NODE_RGB96'][0],proposal['COVER'])
checks.append('One native composite -> direct V90 all8 / V107 context; opaque overwrite and no behind-layer clean leakage; old post96 not final')

g_none=dict(g,box_xyxy=None)
rr=m.compose_scene_pair(base,g_none,parent,donor,mask,'TOY',crop,context,old,scene)
assert rr['proof']['PASS'] and rr['proposed_reference_targets']==r['proposed_reference_targets'] and not rr['inputs']['COVER']['context_supported'] and not rr['inputs']['COVER']['CONTEXT_RGB224_F32'].any()
unknown=m.compose_scene_pair(base,g,dict(parent,teacher_reference_xy=None),donor,mask,'TOY',crop,context,old,scene)
assert unknown['status']=='UNKNOWN_REFERENCE' and unknown['proposed_reference_targets']=={'BEFORE':0,'COVER':None,'ADJACENT_SHAM':None}
holes=np.ones((5,5),bool);holes[3,3]=False
assert not m.renewed_teacher_proof(base5,base5,base5,holes,zero,np.array([3.,3.],np.float64))['PASS']
class MustNotReach:
    def compose_counterfactual_pair(self,*args):raise AssertionError('Proposal must not run after parity mismatch')
try:m.compose_scene_pair(base,g,dict(parent,expected_BEFORE_RGB96_CHW_sha256='0'*64),donor,mask,'TOY',crop,context,MustNotReach(),scene)
except ValueError as e:assert 'BEFORE' in str(e)
else:raise AssertionError('BEFORE mismatch not rejected')
checks.append('Unsupported context no label gate; missing/failed proof UNKNOWN; wrong BEFORE SHA stops before proposal')

out={'completed':True,'PASS':True,'groups':len(checks),'checks':checks,'code_bindings':evidence,'actual_SOURCE_pixels_arrays_GT_models_or_weights':0,'new_data_generation_or_training':0,'invented_only':True}
p=H/'INDEPENDENT_MATH_CHECKS.json'
with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'groups':len(checks),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
