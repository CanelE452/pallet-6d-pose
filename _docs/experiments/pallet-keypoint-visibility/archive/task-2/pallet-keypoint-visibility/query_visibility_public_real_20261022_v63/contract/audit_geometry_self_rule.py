"""Read source/AST only, verify pure cube rule against registered_geometry.

No production state module is imported, and no real input, RGB, reference or
checkpoint is read. All numeric comparisons below use generated toy cuboids.
"""
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from geometry_self_rule import conditional_self_hidden,native_cuboid

HERE=Path(__file__).resolve().parent
BASE=Path('/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_visibility_curriculum_20261013_v1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def functions(path,names,env):
    tree=ast.parse(Path(path).read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),env)
    return {n.name:sorted(set(ast.unparse(c.func) for c in ast.walk(n) if isinstance(c,ast.Call))) for n in selected}


def project_native(xyz,R,t,K):
    camera=xyz@R.T+t
    h=camera@K.T
    return h[:,:2]/h[:,2:],camera[:,2]


env={'np':np,'cv2':cv2,'ConvexHull':ConvexHull,'project_native':project_native,'Any':object}
a=functions(BASE/'common.py',{'cuboid','self_hidden'},env)
env['C']=SimpleNamespace(cuboid=env['cuboid'])
b=functions(BASE/'statewise_joint_api_v2.py',{'_array','_rotation','registered_geometry'},env)
assert not any(any(token in c.lower() for token in ['open','read','load','solve','optimiz','pnp','state_model']) for cs in [*a.values(),*b.values()] for c in cs)
rng=np.random.default_rng(20261022)
comparisons=0
for _ in range(2000):
    extents=np.exp(rng.uniform(-3,1,size=3))
    R,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(R)<0:R[:,0]*=-1
    t=rng.normal(size=3);t[2]=2*np.linalg.norm(extents)+abs(t[2])+1
    xyz=native_cuboid(extents)
    g=env['registered_geometry']([480,640],np.array([[600.,0,320],[0,600,240],[0,0,1]]),xyz,dict(cf_extents=extents,R_cf=R,centroid=t))
    expected=env['self_hidden'](xyz,R,t)
    actual=conditional_self_hidden(extents,R,t,native_xyz=xyz)
    assert np.array_equal(expected,g['initial_self_hidden']) and np.array_equal(actual,expected)
    comparisons+=8
# Validate native-order guard and positive-depth guard, with no inference input.
try:conditional_self_hidden([1,.2,.6],np.eye(3),np.array([0.,0.,2.]),native_xyz=native_cuboid([1,.2,.6])[::-1]);raise AssertionError('permutation accepted')
except ValueError:pass
try:conditional_self_hidden([1,.2,.6],np.eye(3),np.array([0.,0.,0.]));raise AssertionError('nonpositive depth accepted')
except ValueError:pass
result=dict(schema='PURE_REGISTERED_CUBOID_SELF_RULE_AUDIT_V1',production_module_imported=False,production_CAL_or_model_file_reads=0,current_eval_reads=0,new_PnP_calls=0,
            source_files=[dict(path=str(p),sha256=sha(p)) for p in [BASE/'common.py',BASE/'statewise_joint_api_v2.py',Path(__file__).with_name('geometry_self_rule.py')]],
            extracted_function_calls={**a,**b},toy_cuboids=2000,native_slot_comparisons=comparisons,all_common_registered_geometry_and_numpy_wrapper_equal=True,
            fixed_margin_degrees=2,source='coarse registered cuboid outward-face visibility',
            recommended_scope='Call pure conditional_self_hidden on saved initial R_cf,centroid,cf_extents only; preserve resulting8bool flags as geometry proxy separate from p_visible. Do not instantiate StatewiseJointV2/refine or production state/mask models.',
            policy='Optional explicitly named pixelhead+geometryS hybrid for semantic self task only. Raw pixelpresence predictions and geometryS flag remain independently saved. Do not overwrite exact-pixel label or probability.',
            limitations=['Conditional on saved coarse pose, dimensions and semantic/native basis; wrong pose/dimensions can produce wrong S.','A convex cube gate is not a mesh self-visibility certificate and not human semantic supervision.','Full statewise API module imports solver and learned state components even though registered_geometry function itself never calls them. Extracted numpy-only rule avoids those dependencies.'])
(HERE/'GEOMETRY_SELF_RULE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(toy_cuboids=2000,slot_comparisons=comparisons,all_equal=True,private_eval_reads=0,new_PnP=0)))
