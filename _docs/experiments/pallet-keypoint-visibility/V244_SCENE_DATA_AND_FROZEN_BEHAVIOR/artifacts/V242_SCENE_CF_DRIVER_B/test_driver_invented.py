"""Three invented-array/temporary-file checks. No source payload access."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,json,runpy
import numpy as np

H=Path(__file__).parent
d=runpy.run_path(str(H/'build_scene_pairs.py'),run_name='invented_driver')

# False entry must stop before even the scope/code bindings are followed.
with TemporaryDirectory() as td:
 p=Path(td)/'false.json';p.write_text('{"execution_authorized":false}')
 try:d['run'](str(p))
 except PermissionError:pass
 else:raise AssertionError('False entry accepted')
 for bad in ('/tmp/V112/x','/tmp/__pycache__/x','/tmp/x.pyc','/tmp/../x'):
  try:d['lexical'](bad)
  except ValueError:pass
  else:raise AssertionError('Forbidden lexical path accepted')

# A toy compressed container contains an unselected non-RGB array. Only the
# literal RGB_uint8 member may be indexed; no .files/member enumeration.
with TemporaryDirectory() as td:
 p=Path(td)/'toy.npz'; rgb=np.arange(9*11*3,dtype=np.uint8).reshape(9,11,3)
 np.savez_compressed(p,RGB_uint8=rgb,DO_NOT_DECODE=np.array([9917]))
 image={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'format':'npz','member':'RGB_uint8'}
 original=np.load; accesses=[]
 class OnlyRGB:
  def __init__(self,z):self.z=z
  def __enter__(self):return self
  def __exit__(self,*args):self.z.close()
  def __getitem__(self,k):
   assert k=='RGB_uint8'; accesses.append(k);return self.z[k]
  @property
  def files(self):raise AssertionError('Member enumeration')
 np.load=lambda *a,**kw:OnlyRGB(original(*a,**kw))
 try:
  decoder=d['Decoder']([image],max_bytes=rgb.nbytes)
  assert np.array_equal(decoder.get(image,raw_hw=None),rgb)
  assert np.array_equal(decoder.get(image,raw_hw=[9,11]),rgb)
  assert accesses==['RGB_uint8'] and decoder.bytes<=rgb.nbytes
  try:decoder.get(dict(image,path=str(Path(td)/'unlisted.npz')))
  except ValueError:pass
  else:raise AssertionError('Unlisted payload accepted')
 finally:np.load=original

# Store actual rendered inputs for constructedUNKNOWN while withholding labels;
# old proposal failure alone has zero input. Carrier is exact only at alpha true.
m=d['load_math'](d['DEPENDENCIES']); math=m['scene_cf_math'];crop=m['crop90'];ctx=m['context107'];old=m['v235_compositor'];lift=m['scene_math']
y,x=np.mgrid[:256,:256];base=np.stack((x%256,y%256,(x+y)%256),2).astype(np.uint8)
q8=np.array([[128,128],[160,128],[160,160],[128,160],[100,100],[120,100],[100,120],[-1,-1]],np.float64)
group={'q8':q8,'L8':np.full(8,96.,np.float64),'original_available8':np.array([1]*7+[0],bool),'box_xyxy':None}
before=math.resample_all(base,group,crop,ctx);donor=np.full((32,32,3),[12,221,31],np.uint8);mask=np.ones((32,32),bool);ref=np.array([128.25,128.25],np.float64)
proposal=old.compose_counterfactual_pair(before['NODE_RGB96'][0],before['NODE_VALID96'][0],donor,mask,q8[0],np.float64(96),ref,'V241_INVENTED_1');assert proposal['success']
parent={'active_slot':0,'original_input':{'q':q8[0].tolist(),'L':96.,'row_id':'V241_INVENTED_1','fit':True,'target':0,'original_query_available':True,'crop_available':True},'same_V235_donor_id':'TOY','same_V235_donor_plan':math.plain(proposal['donor_plan']),'expected_BEFORE_RGB96_CHW_sha256':math.sha(before['NODE_RGB96'][0]),'expected_BEFORE_valid96_bool_sha256':math.sha(before['NODE_VALID96'][0]),'teacher_reference_xy':ref.tolist(),'teacher_authority':'INVENTED'}
result=math.compose_scene_pair(base,group,parent,donor,mask,'TOY',crop,ctx,old,lift)
layers,alphas=d['layer_carriers'](result,parent,donor,mask,old,math)
for k,role in enumerate(('COVER','SHAM')):
 assert np.array_equal(layers[k][:,alphas[k]],proposal[role][:,alphas[k]]) and not layers[k][:,~alphas[k]].any()
 native,alpha=lift.lift_raw96_layer(layers[k].transpose(1,2,0),alphas[k],q8[0],np.float64(96),base.shape[:2])
 reconstructed=base.copy();reconstructed[alpha]=native[alpha]
 assert np.array_equal(reconstructed,result['native_scenes'][d['ROLES'][k]])
arrays={k:np.empty((1,)+shape[1:],dtype=dtype) for k,(dtype,shape) in d['ARRAYS'].items() if not k.startswith('GROUP_')}
assert d['store_children'](arrays,0,result,layers,alphas)==(True,True)
assert arrays['PAIR_TARGETS.npy'][0].tolist()==[1,0]
unknown=dict(result,proof={'PASS':False})
assert d['store_children'](arrays,0,unknown,layers,alphas)==(True,False)
assert arrays['PAIR_TARGETS.npy'][0].tolist()==[-1,-1] and np.array_equal(arrays['CF_NODE_RGB96.npy'][0,0],result['inputs']['COVER']['NODE_RGB96'])
assert np.array_equal(np.unpackbits(arrays['PAIR_ALPHA_PACKBITS.npy'][0],axis=1,bitorder='big').reshape(2,96,96).astype(bool),alphas)
assert d['store_children'](arrays,0,{'constructed':False,'proof':None})==(False,False)
assert not arrays['CF_NODE_RGB96.npy'].any() and not arrays['CF_CONTEXT_RGB224_F32.npy'].any() and arrays['PAIR_TARGETS.npy'][0].tolist()==[-1,-1]
assert not before['original_available8'][7] and before['NODE_VALID96'][7].any() and not before['node_available'][7]
summary={'completed':True,'PASS':True,'invented_groups':3,'false_entry_pre_binding_refusal':True,'NPZ_RGB_uint8_only':True,'carrier_native_reconstruction_exact':True,'constructed_UNKNOWN_input_preserved_targets_unknown':True,'unconstructed_zero':True,'raw_payloads':0,'actual_driver_runs':0}
with (H/'INVENTED_DRIVER_CHECKS.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary))
