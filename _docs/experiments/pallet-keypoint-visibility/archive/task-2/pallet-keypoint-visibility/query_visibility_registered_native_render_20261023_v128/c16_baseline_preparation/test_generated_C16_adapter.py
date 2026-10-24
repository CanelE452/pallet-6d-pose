"""Invented input/layout/static checks only. No source arrays, Torch or model calls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,tempfile
import numpy as np
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def run(output):
 code=H/'run_frozen_C16_registered.py';text=code.read_text();compile(ast.parse(text),str(code),'exec');s=importlib.util.spec_from_file_location('invented_C16_adapter',code);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);checks={}
 assert 'torch'not in sys.modules;checks['inert_import_no_Torch_or_payload']=True
 base=np.array([[80,90],[180,90],[180,170],[80,170],[100,100],[160,100],[160,160],[100,160]],np.float64);rows=[]
 for style in m.STYLES:
  for variant in m.VARIANTS:
   for slot in range(8):
    for jitter,(dy,dx)in enumerate((y,x)for y in(-3,0,3)for x in(-3,0,3)):
     q=base.copy();q[slot]+=[dx,dy];g=q[~(q==[-1,-1]).all(1)];rows.append(dict(occurrence_index=len(rows),id='%s_%s_%d_%d'%(style,variant,slot,jitter),query_group_id='%s_%d_%d'%(style,slot,jitter),rgb96_row=len(rows),style=style,variant=variant,logical_slot=slot,carrier_slot=slot,jitter_index=jitter,offset_xy=[dx,dy],initial_points=q.tolist(),query_xy=q[slot].tolist(),context_box_xyxy=np.r_[g.min(0),g.max(0)].tolist(),L=float(np.clip(.25*np.max(np.ptp(g,axis=0)),16,256)),raw_hw=[256,256],full_RGB=dict(path='INVENTED_'+style+variant,sha256='metadata-only'),full_RGB_format='NPZ_RGB_UINT8_HWC',full_RGB_key='RGB_uint8',decoded_RGB_sha256='fakeRGB',rgb96_sha256='fakeCrop',valid96_sha256='fakeValid'))
 groups=m.validate_rows(rows);assert len(rows)==432 and len(groups)==144 and all(len(v)==3 for v in groups.values());checks['all432_6scene_8slot_9jitter_triplet_input_joins']=True
 assert m.identity(dict(id='a'))==m.identity(dict(occurrence_id='a'))==m.identity(dict(idstr='a'))==m.identity(dict(id='a',occurrence_id='a',idstr='a'));checks['canonical_id_and_consistent_optional_aliases']=True
 broken=copy.deepcopy(rows);broken[72]['query_xy'][0]+=.25
 try:m.validate_rows(broken)
 except AssertionError:pass
 else:raise AssertionError('Broken q8/query identity accepted')
 broken=copy.deepcopy(rows);broken[0]['reference_xy']=[80,90]
 try:m.validate_rows(broken)
 except AssertionError:pass
 else:raise AssertionError('GT reference accepted in inputs')
 checks['qL_variant_drift_and_reference_feature_path_rejected']=True
 rng=np.random.default_rng(123);stored=rng.integers(0,256,size=(2,96,96,3),dtype=np.uint8);valid=np.ones((2,96,96),bool);valid[0,47,48]=False;rgb,v=m.stored_layout(stored,valid,2)
 assert rgb.shape==(2,3,96,96)and v.shape==(2,1,96,96)and np.array_equal(rgb.transpose(0,2,3,1),stored)and not v[0,:,47:49,47:49].all()and v[1,:,47:49,47:49].all();checks['stored_HWC_HW_hash_layout_to_canonical_CHW_1HW_exact']=True
 assert m.ash(stored[0])!=m.ash(rgb[0]);checks['stored_layout_hash_is_not_silently_CHW_hash']=True
 available=m.pure(m.PARENT,('factual_availability',))['factual_availability']
 for pattern in range(16):
  vv=np.ones((1,96,96),bool);vv[0,47:49,47:49]=np.array([(pattern>>k)&1 for k in range(4)],bool).reshape(2,2);fa,ca,aa,c4=available([10,20],[256,256],vv,np.ones((224,224),bool));assert fa==aa==c4==(pattern==15)and ca
 assert available([255.8,20],[256,256],np.ones((1,96,96),bool),np.ones((224,224),bool))[:3]==(False,True,False)
 assert available([10,20],[256,256],np.ones((1,96,96),bool),np.zeros((224,224),bool))[:3]==(True,False,False)
 checks['exact_parent_factual_central4_nearest_context_U_rules']=True
 r=rows[0];changed=dict(r,logical_slot=7,style='UNUSED_METADATA',variant='UNUSED_VARIANT');assert m.unit_key(r)==m.unit_key(changed);checks['unit_dedup_excludes_slot_style_variant_or_teacher_identity']=True
 # Mock lazy NPZ: any access to owner/depth/teacher members would fail this fixture.
 fake=np.arange(12,dtype=np.uint8).reshape(2,2,3);access=[]
 class Archive:
  def __enter__(self):return self
  def __exit__(self,*unused):pass
  def __getitem__(self,key):
   access.append(key);assert key=='RGB_uint8';return fake
 oldload=m.np.load;oldcheck=m.check
 try:
  m.np.load=lambda *args,**kwargs:Archive();m.check=lambda b:Path('unused');out=m.load_native_rgb(dict(path='unused',sha256='unused'),'RGB_uint8',[2,2],m.ash(fake));assert np.array_equal(out,fake)and access==['RGB_uint8']
 finally:m.np.load=oldload;m.check=oldcheck
 checks['native_NPZ_loads_RGB_member_only_no_owner_or_teacher']=True
 z=np.array([-1e-8,-0.,0.,1e-8,100],np.float32);a=np.array([1,1,1,1,0],bool);pred=np.where(a,(z>=0).astype(np.int8),-1);assert pred.tolist()==[0,1,1,1,-1];checks['raw_hidden_sign0_E_tie_and_factualU_priority']=True
 assert "crop(image,np.asarray(r['query_xy']"in text and 'np.array_equal(a,rgb96[ix])'in text and "views.shape==(3*n,3,224,224)"in text and 'producer.assemble('in text
 assert "a=(fa&ca)[um]"in text and 'not x[~qi,384:768].any()'in text;checks['exact_C11_3view_operator_Qoutside_not_availability_gate']=True
 assert 'headmod.SpatialTargetQueryHead()'in text and "payload['update']==4500"in text and 'head.load_state_dict' in text and 'torch.equal(raw,again)'in text and 'read(e[' not in text.split('def run(entry):',1)[1].split('def prepare():',1)[0]
 checks['C16source4500_frozen_two_repeat_head_no_GT_reader']=True
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'unapproved.json';p.write_text(json.dumps(dict(approved_source_C16_baseline=False,fit_allowed=False,current_allowed=False,V112_allowed=False,initializer_allowed=False,PnP_allowed=False,automatic_retry=False)))
  try:m.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved source model call accepted')
 checks['unapproved_entry_fails_before_source_parameter_or_pixel_IO']=True
 assert all(checks.values())and 'torch'not in sys.modules
 result=dict(completed=True,PASS=True,code=bind(code),test_code=bind(Path(__file__)),checks=checks,check_count=len(checks),actual_source_RGB_features_weights_encoder_head_fit_current_V112_reads_or_calls=0)
 with output.open('x')as fp:json.dump(result,fp,indent=2,allow_nan=False);fp.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,code=result['code'],checks=len(checks),output=bind(output))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_BASELINE_TESTS.json');a=ap.parse_args();run(a.output)
