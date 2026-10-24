"""Thin full-source R0 adapter. Preparation is metadata/NumPy only.
Original frozen factory/first-argmax/raw9/q8/L/masks are unchanged. No GT input.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,difflib,hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json';POLICY_SHA='46b3b077edfc78d113dcd0b38771bb3df96053a3741002ac774bc1c9866e2334'
V152=P/'query_visibility_mr6d_actual_query_bridge_20261024_v152'
OLD_RUN=V152/'initializer_preparation/EFFECTIVE_MR6D10_RUN.py';OLD_RUN_SHA='f761bdab352d913118d0c1a1d9954bdb76b410d216997e429027019889128978'
BASE=P/'query_visibility_public_native_initializer_association_20261024_v141/initializer_preparation/infer_public_fixed16.py';BASE_SHA='5537924777a5398b7ec19bb7ed9f414b27f25a1c66103f415a7c23ebea70a16e'
INTAKE=V/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json';MANIFEST=V/'full_source_v1/RGB230_INPUT_ONLY_MANIFEST.json'
PRIOR_SEAL=V152/'actual_mr6d_initializer_v1/PREDICTION_SEAL.json';PRIOR_SHA='041ba0669c2067dbae43c4885dc3b9c897fcd66c286482dfacc2d493ebb6db08'
PROTOCOL=H/'R0_PROTOCOL_BEFORE_FORWARD.json';GENERATED=H/'R0_GENERATED_CHECKS.json';INPUTS=H/'R0_INPUTS230_RGB_ONLY.jsonl';OUT=V/'actual_full230_initializer_v1'
INPUT_KEYS={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder','canonical_RGB_NPY'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)==BASE_SHA and sha(OLD_RUN)==OLD_RUN_SHA
spec=importlib.util.spec_from_file_location('v153_exact_R0_helpers',BASE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
bind,read,check,write,lines,same=b.bind,b.read,b.check,b.write,b.lines,b.same

def validate_transport(e,p,rows):
 assert e['canonical_transport_seal']==p['canonical_transport_seal'] and Path(e['canonical_transport_seal']['path'])==INTAKE
 seal=read(check(e['canonical_transport_seal']));assert seal['completed']and seal['PASS']and seal['frames']==230
 assert seal['policy']==p['policy'] and seal['RGB_manifest']==p['canonical_manifest'] and seal['canonical_decoder']=='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB'
 manifest=read(check(seal['RGB_manifest']));assert manifest['rows']==rows and len(rows)==230
 assert Counter(r['split']for r in rows)=={'SOURCE_DEVELOPMENT':185,'SOURCE_VALIDATION':45}
 assert all(r['source']=='MR6D'and r['frame_index']==i for i,r in enumerate(rows))
 for r in rows:
  scene=r['source_id'].split('/')[1].split(':')[0]
  assert r['split']==('SOURCE_VALIDATION'if scene=='000001'else'SOURCE_DEVELOPMENT')

def array_to_bgr(rgb,row):
 assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3] and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
 return np.ascontiguousarray(rgb[:,:,::-1])

def audit_fixed10_parity(proposals,arrays,allc):
 """Read only sealed prior public raw outputs, after all new R0 calls finish."""
 assert sha(PRIOR_SEAL)==PRIOR_SHA
 seal=read(PRIOR_SEAL);assert seal['completed']and seal['PASS']and not seal['GT_read']
 files={Path(x['path']).name:check(x)for x in seal['outputs']}
 old=lines(files['FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl']);prior=np.load(files['ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz'],allow_pickle=False)
 byid={r['source_id']:(i,r)for i,r in enumerate(proposals)};assert len(byid)==230
 proof=[]
 for oi,r in enumerate(old):
  ni,n=byid[r['source_id']]
  for k in ('source_id','image_sha256','decoded_RGB_sha256','raw_hw','candidates','selected_index','input_shape','canvas_shape','added_border','affine_gain','affine_offset'):assert n[k]==r[k],(r['source_id'],k)
  for k in ('query_xy','L','finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','native_grid_valid','native_slot','no_detection'):
   a,c=arrays[k][ni*8:ni*8+8],prior[k][oi*8:oi*8+8]
   assert a.dtype==c.dtype and a.shape==c.shape and a.tobytes()==c.tobytes(),(r['source_id'],k)
  proof.append(dict(source_id=r['source_id'],old_frame_index=oi,new_frame_index=ni,raw9_all_candidates_firstargmax_q8_L_validities_bitexact=True))
 assert len(proof)==10
 path=OUT/'FIXED10_ORIGINAL_R0_PARITY.json';write(path,dict(completed=True,PASS=True,prior_prediction_seal=bind(PRIOR_SEAL),new_queries=1840,frames_compared=10,rows_compared=80,rows=proof,no_GT_or_model_choice=True))
 return path

ORIGINAL=OLD_RUN.read_text();tree=ast.parse(ORIGINAL);changes=Counter()
class Adapt(ast.NodeTransformer):
 def visit_keyword(self,node):
  node.arg={'all10_and80_retained':'all230_and1840_retained','all10_frames80_native_slots_preserved':'all230_frames1840_native_slots_preserved'}.get(node.arg,node.arg)
  return self.generic_visit(node)
 def visit_Constant(self,node):
  if type(node.value)is int and node.value in(10,80,12):
   changes[node.value]+=1;return ast.copy_location(ast.Constant({10:230,80:1840,12:232}[node.value]),node)
  if isinstance(node.value,str):node.value=node.value.replace('ALL10','ALL230').replace('QUERY80','QUERY1840').replace('v152_exact_R0_factory','v153_exact_R0_factory')
  return node
 def visit_FunctionDef(self,node):
  node=self.generic_visit(node)
  if node.name=='run':
   for i,n in enumerate(node.body):
    if isinstance(n,ast.Try):
     block=n.body
     for j,x in enumerate(block):
      if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in x.targets):
       block[j:j]=ast.parse("parity = audit_fixed10_parity(proposals, arrays, allc)").body;break
     for j,x in enumerate(block):
      if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='outputs'for t in x.targets):
       block[j+1:j+1]=ast.parse("outputs.append(bind(parity))").body;break
  return node
EFFECTIVE=ast.unparse(ast.fix_missing_locations(Adapt().visit(tree)))+'\n';assert set(changes)=={10,80,12}
ns=dict(vars(b));ns.update(__file__=__file__,H=H,V=V,P=P,POLICY=POLICY,POLICY_SHA=POLICY_SHA,PROTOCOL=PROTOCOL,GENERATED=GENERATED,INPUTS=INPUTS,INPUT_KEYS=INPUT_KEYS,OUT=OUT,validate_transport=validate_transport,array_to_bgr=array_to_bgr,audit_fixed10_parity=audit_fixed10_parity)
exec(compile(EFFECTIVE,str(H/'EFFECTIVE_FULL230_R0_RUN.py'),'exec'),ns);run=ns['run']

def generated():
 base=np.array([[90.+11*k,120.+3*k]for k in range(9)],np.float64)
 cand=lambda q:dict(score=.8,keypoints_xy=q)
 z=lambda q:dict(candidates=[cand(q),cand(q+30)],selected_index=0)
 q=b.query_arrays(z(base),[720,1280]);cent=base.copy();cent[8]=[1e8,-1e8]
 assert all(np.array_equal(q[k],b.query_arrays(z(cent),[720,1280])[k],equal_nan=True)for k in q)
 no=b.query_arrays(dict(candidates=[],selected_index=None),[720,1280]);assert np.isnan(no['query_xy']).all()and np.isnan(no['L']).all()and not no['native_grid_valid'].any()
 for xy in ([-.5,0.],[-.25,10.],[1279.5,10.],[1280.,720.]):
  x=base.copy();x[0]=xy;r=b.query_arrays(z(x),[720,1280]);assert r['query_xy'][0].tolist()==xy and np.array_equal(r['nearest_valid'],r['RGBcentre_valid'])
 write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_only=True,unchanged_native_grid_firstargmax_centroid_noquery_OOB=True,source_RGB_GT_weights_model_reads_or_calls=0))

def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(PRIOR_SEAL)==PRIOR_SHA
 intake=read(INTAKE);assert intake['completed']and intake['PASS']and intake['frames']==230 and intake['policy']==bind(POLICY)
 rows=read(check(intake['RGB_manifest']))['rows'];assert len(rows)==230 and all(set(r)==INPUT_KEYS for r in rows)
 assert Path(intake['RGB_manifest']['path'])==MANIFEST
 assert all(r['frame_index']==i for i,r in enumerate(rows))and len({r['source_id']for r in rows})==230
 assert Counter(r['split']for r in rows)=={'SOURCE_DEVELOPMENT':185,'SOURCE_VALIDATION':45}
 with INPUTS.open('x')as f:
  for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
 previous=read(V152/'initializer_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD.json');generated()
 effective=H/'EFFECTIVE_FULL230_R0_RUN.py';effective.write_text(EFFECTIVE)
 diff=H/'EXACT_V152_TO_FULL230_R0_DIFF.patch';diff.write_text(''.join(difflib.unified_diff(ORIGINAL.splitlines(True),EFFECTIVE.splitlines(True),fromfile='V152_RUN',tofile='V153_RUN')))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=intake['selection'],generated=bind(GENERATED),inputs=bind(INPUTS),
  **{k:previous[k]for k in ('reference_code','reference_protocol','backbone_code','checkpoint','native_grid','runtime')},
  helper_code=bind(BASE),original_run=bind(OLD_RUN),original_protocol=bind(V152/'initializer_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD.json'),canonical_transport_seal=bind(INTAKE),canonical_manifest=intake['RGB_manifest'],
  prior_fixed10_prediction_seal=bind(PRIOR_SEAL),frames=230,native_slots=1840,output=str(OUT),effective_run=bind(effective),run_diff=bind(diff),AST_count_replacements={str(k):v for k,v in changes.items()},
  exact_operator='V152 factory/loop unchanged except inventory/names and post-forward mandatory fixed10 raw9/q/L/mask byte parity before final seal',
  no_pose_K_reference_labels_masks_depth_or_historic_visibility_score_inputs=True,no_R0_or_DINO_execution_during_prepare=True,no_full_P3_saved_or_DINO_need=True,automatic_retry=False)
 write(PROTOCOL,p)
 t=H/'ROOT_R0_ENTRY_TEMPLATE.json';write(t,dict(approved_actual_public_initializer=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],canonical_transport_seal=p['canonical_transport_seal'],**{k:False for k in b.FORBIDDEN}))
 write(H/'R0_PREPARATION_READY.json',dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],effective_run=p['effective_run'],template=bind(t),actual_learned_calls=0))
 print(json.dumps(dict(completed=True,prepared_only=True,protocol=bind(PROTOCOL),code=bind(__file__),template=bind(t))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and not a.entry:prepare()
 elif a.entry and not a.prepare:run(a.entry)
 else:raise SystemExit('Use --prepare after completed intake; actual root --entry separately')
