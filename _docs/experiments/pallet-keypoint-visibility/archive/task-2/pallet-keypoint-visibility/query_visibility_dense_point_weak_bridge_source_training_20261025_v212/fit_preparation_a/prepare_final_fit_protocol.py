"""Prepare exact metadata/code closure; never opens any NPZ/PNG/GT or weights."""
import hashlib,json,os
from pathlib import Path
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
def rb(p):
 p=Path(p);assert p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold()and p.suffix in('.json','.jsonl','.py')and p.name!='SCENES_GT_ONLY.json'
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for i,s in enumerate(p.parts[1:]):
   n=os.open(s,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts[1:])-1 else 0),dir_fd=fd);os.close(fd);fd=n
  with os.fdopen(fd,'rb')as f:fd=-1;return f.read()
 finally:
  if fd>=0:os.close(fd)
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def read(b):
 z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def new(n,v):
 z=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 with(H/n).open('xb')as f:f.write(z)
 return dict(path=str(H/n),sha256=hashlib.sha256(z).hexdigest())
def main():
 olddir=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b';oldproto=bind(olddir/'PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json');op=read(oldproto)
 scope=bind(V/'ROOT_PROSPECTIVE_DENSE_POINT_WEAK_BRIDGE_HEAD_PREPARATION_SCOPE.json');amend=bind(V/'ROOT_PROSPECTIVE_SOURCE_POINT_LOSS_ELIGIBILITY_AMENDMENT.json');pa=bind(V/'ROOT_SOURCE_PROCEDURAL_POINT_TEACHER_ADMISSION.json');pad=read(pa);wa=bind(V/'ROOT_SOURCE_WEAK44_BRIDGE_ADMISSION.json');wad=read(wa)
 assert scope['sha256']=='4e73179d43a7153c76c55691397f0cce59cf67eea7a1b8e4ccadd49077fd3ea9'and amend['sha256']=='6cfeb4add539f43326e33b1b49398c22577379ee69f06f815be5a68d146ed47f'and pa['sha256']=='f84c754b4965cd989c7e30ad9eebde37116627b1402a9ffddbd6dd278633094a'and wa['sha256']=='0726e277aa06c36efc5e5bdeb7e56b1cce9f5b10c0f2595398d139fd61b104ab'
 weak=bind(H/'WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json');wp=read(weak);assert wad['weak_metadata']==weak and wad['counts']==dict(rows=44,D=26,E=18)and wad['weak_loss_weight']==.25
 source={'head':op['source_bindings']['existing_head'],'candidate':op['source_bindings']['candidate'],'spatial_head':op['spatial_head'],'context_adapter':op['context_adapter'],'mask_adapter':op['mask_adapter'],'common':op['source_bindings']['common_code'],'old_sampler_loss':op['source_bindings']['original_trainer']}
 for b in source.values():assert bind(Path(b['path']))==b
 payload=[]
 aug=read(op['old_cache_bindings']['augmentation']);public=read(op['old_cache_bindings']['public_CF'])
 for domain,r in [('SYNTH_AUG65536',aug)]+[(d,public['domains'][d])for d in('PUBLIC_TRAIN1480','CF3816')]:
  for k in('inputs','targets'):payload.append(dict(r[k],kind='OLD5_'+k.upper(),domain=domain))
 payload+=wp['payload_whitelist'];assert len(payload)==len({r['path']for r in payload})==58
 whitelist=new('FUTURE_ACTUAL58_PAYLOAD_WHITELIST_METADATA_ONLY.json',dict(completed=True,metadata_declarations_only=True,old5_input_target_NPZ=6,weak_RGB_valid_PNG=52,oldDIRECT_NPZ=0,GT_weights_foundation_or_protected_assets=0,files=payload,actual_payload_stat_hash_header_decode=0))
 checks=bind(H/'INVENTED_CPU_CHECKS.json');q=read(checks);trainer=bind(H/'train_dense_point_weak_bridge.py');contract=bind(H/'source_target_contract.py');assert q['completed']and q['PASS']and q['trainer_code']==trainer and q['target_contract']==contract
 p=dict(completed=True,prepared_only=True,trainer_code=trainer,target_contract=contract,inherited_trainer=op['trainer_code'],inherited_protocol=oldproto,source_bindings=source,scope=scope,eligibility_amendment=amend,point_admission=pa,weak_admission=wa,teacher_complete=pad['source_teacher_complete'],teacher_QA=pad['independent_actual_QA'],weak_metadata=weak,checks=checks,test_code=bind(H/'test_fit_contract_invented.py'),weak_metadata_prepare_code=bind(H/'prepare_weak44_metadata.py'),protocol_prepare_code=bind(__file__),payload_whitelist=whitelist,
 old_cache_bindings=op['old_cache_bindings'],split_complete=op['split_complete'],partition_QA=op['partition_QA'],context_input_bindings=op['context_input_bindings'],arms=[dict(id='C62_DENSE_POINT_RAW_Q8',weak_bridge=False,weak_loss_weight=0.),dict(id='C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',weak_bridge=True,weak_loss_weight=.25)],primary_predeclared='C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',parameters_each=118359,spatial_q8=True,use_sam=False,from_scratch=True,
 training=dict(steps_each=4000,same_init_seed=1810,old_rng=1810,point_rng=1811,weak_rng=1812,old_draws=192,point_D_E_draws=[16,16],weak_D_E_draws=[8,8],optimizer='AdamW',lr=.0003,weight_decay=.0001,gradient_clip=5.,FP32=True,TF32=False,no_earlystop=True),
 losses='unchanged old-five mean + balanced admitted own-image point BCE + C63 only 0.25*balanced weak-feature BCE; old finite-end DIRECT pool wholly replaced',
 masks='Original component TRAIN intersect nominal for all old5; all7040 source image mapping retained, targetaloneD+combinedDE+originalavailable finiteqL+finite ref within frame and actual crop canvas only gives6096; S/U/missing/outside loss UNKNOWN. Original roles never relabelled. sample-center flag diagnostic only.',
 model_input='exact C61 RAW8 + original native/mask/FP64->FP32 relativeq8; zeroSAM. GT reference/point/region/role/alpha/weak label never features. originalq/L/native/U unchanged, no new crop/initializer/PnP.',
 weak_authority='8 original factualD once/case including2failed cases +18 SHAM preservedD+18COVER weakE,3originalPUBLIC TRAINframes,q8 active metadata exact,44distinct RGBvalid hashes; assistant local-feature weak, not physical native/human gold. Original V207 admission remains false; new bridge admission separate.',
 outputs='After BOTH final4000 barrier only:3old source FULL+originalTRAIN/HOLD, POINT7040 exactslice of SYNfull and weak44 perarm. No oldDIRECT3144 input,target,oroutput. Raw API/order field names inherited C61; SOURCE point/weak target authority explicitly in checkpoint/protocol.',
 runtime='Exact root actual approved entry plus independent preflight/point+weak admissions and58payload whitelist required before bytes. Source full containers can physically unpack HOLD rows for immutable authority, all fitting/normalization strictly TRAIN. No sourceH/private/COMMON/current/publicheld/VAL model/data/threshold selection.',
 actual_fit_or_SOURCE_payload_header_or_bytes_reads=0,automatic_retry=False,selected_model=None,adopted=False,fully_solved=False)
 protocol=new('PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json',p)
 template=dict(approved_dense_point_weak_bridge_source_fit=False,source_only=True,automatic_retry=False,code=trainer,protocol=protocol,checks=checks,root_scope=dict(path=None,sha256=None),independent_preflight=dict(path=None,sha256=None),weak_admission=wa,split_complete=op['split_complete'],context_input_bindings=op['context_input_bindings'],payload_whitelist=payload,output_directory=str(V/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000'))
 false=new('ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
 ready=new('FINAL_READY.json',dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=trainer,target_contract=contract,protocol=protocol,checks=checks,false_entry=false,payload_whitelist=whitelist,weak_metadata=weak,point_teacher=pad['source_teacher_complete'],point_QA=pad['independent_actual_QA'],actual_SOURCE_payload_header_or_bytes_reads=0,actual_fit_forward_pretrained_or_optimizer_calls=0,authored_CPU_toys_only=True,command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B '+str(H/'train_dense_point_weak_bridge.py')+' --entry <absolute root actual approved entry>'))
 print(json.dumps(dict(PASS=True,READY=ready,code=trainer,protocol=protocol,checks=checks,payloads=58)))
if __name__=='__main__':main()
