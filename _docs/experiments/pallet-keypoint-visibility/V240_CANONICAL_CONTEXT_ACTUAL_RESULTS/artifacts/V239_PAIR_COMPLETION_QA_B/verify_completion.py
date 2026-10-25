"""Independent saved V239 paired head/optimizer/ledger/CSV QA. No forward."""
import argparse,csv,hashlib,io,json,math,os,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).absolute().parent
ARMS=('CONTROL_CANONICAL_LOCAL','TREAT_CANONICAL_CONTEXT')
CODE_SHA='ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419'
PROTOCOL_SHA='4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021'
INITIAL_BASE_SHA='d1b8da9721a5492dcfb166cf78943eb5db18c5b14b8e8fed328ccab31c12648c'
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
SAMPLES_SHA='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'
BASE_SHAPES={'head.0.weight':(64,1186),'head.0.bias':(64,),'head.2.weight':(2,64),'head.2.bias':(2,)}
EXTRA_SHAPES={'relational.project.weight':(16,384),'relational.geometry_bias.weight':(1,2),'relational.readout.0.weight':(32,116),'relational.readout.0.bias':(32,),'relational.readout.2.weight':(2,32),'relational.readout.2.bias':(2,)}
SAMPLE_KEYS={'full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id'}

def lexical(value):
 text=str(value);low=text.casefold()
 assert not any(k in low for k in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','oldval','v177','v178','source_holdout','v182')),'Forbidden lexical path before filesystem access'
 p=Path(text);assert p.is_absolute() and '..' not in p.parts
 return p

def safe(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):assert not q.is_symlink(),'No symlink path'
 return p

def raw(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def identity(b):return str(lexical(b['path'])),b['sha256']

def state_sha(state):
 h=hashlib.sha256()
 for name in sorted(state):
  a=state[name].detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()

def check_state(state,shapes):
 assert set(state)==set(shapes)
 for key,shape in shapes.items():
  a=state[key].detach().cpu().numpy();assert a.dtype==np.float32 and tuple(a.shape)==shape and np.isfinite(a).all()
 return sum(v.numel() for v in state.values())

def check_optimizer(cp,shapes):
 opt=cp['optimizer_state_dict'];assert len(opt['param_groups'])==1
 g=opt['param_groups'][0];ids=g['params'];assert len(ids)==len(shapes) and set(ids)==set(opt['state'])
 assert g['lr']==.001 and g['weight_decay']==.0001 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
 for pid,shape in zip(ids,shapes.values()):
  s=opt['state'][pid];assert float(s['step'])==1000
  for k in ('exp_avg','exp_avg_sq'):
   a=s[k].detach().cpu().numpy();assert a.dtype==np.float32 and tuple(a.shape)==shape and np.isfinite(a).all()
 return len(ids)

def parse_csv(text):
 rows=[]
 for i,r in enumerate(csv.DictReader(io.StringIO(text))):
  assert int(r['sample_index'])==i and r['available'] in ('True','False')
  f=int(r['family']);y=int(r['target']);role=r['role'];available=r['available']=='True'
  assert f in (0,1) and y in (0,1) and role in ('FIT','DEV')
  z=np.asarray([float(r['logit0']),float(r['logit1'])],np.float32);assert np.isfinite(z).all()
  value=float(r['selected_family_logit']);assert value==float(r['logit'+str(f)])
  prediction='U' if not available else ('E' if value>=0 else 'D');assert r['prediction']==prediction
  rows.append({'sample_index':i,'row_id':r['row_id'],'family':f,'target':y,'role':role,'available':available,'prediction':prediction,'logits':z})
 assert len(rows)==7617 and len({r['row_id'] for r in rows})==7617 and all(r['available'] for r in rows)
 assert sum(r['role']=='FIT' for r in rows)==5876 and sum(r['role']=='DEV' for r in rows)==1741
 return rows

def replay_draws(rows,steps=1000):
 pools=[np.asarray([r['sample_index'] for r in rows if r['role']=='FIT' and r['available'] and r['family']==f and r['target']==y],np.int64) for f in (0,1) for y in (0,1)]
 assert all(len(p) for p in pools)
 rng=np.random.Generator(np.random.PCG64(17));draw=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(steps)])
 return draw,hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest(),[len(p) for p in pools]

def metrics(rows):
 out={}
 for f in (0,1):
  for role in ('FIT','DEV'):
   rr=[r for r in rows if r['family']==f and r['role']==role];cm=[[0,0,0],[0,0,0]]
   for r in rr:cm[r['target']][{'D':0,'E':1,'U':2}[r['prediction']]]+=1
   item={'rows':len(rr),'known_DE':len(rr),'target_UNKNOWN':0,'original_input_U':sum(not r['available'] for r in rr),'CM_D_E_by_D_E_U':cm}
   for name,k in (('D',0),('E',1)):
    truth=sum(cm[k]);pred=cm[0][k]+cm[1][k];tp=cm[k][k]
    item[name]={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
   out[f'family{f}_{role}']=item
 return out

def paired(a,b):
 out={}
 for f in (0,1):
  for role in ('FIT','DEV'):
   rr=[(x,y) for x,y in zip(a,b) if x['family']==f and x['role']==role];recovered=damaged=0
   for x,y in rr:
    target='DE'[x['target']];recovered+=x['prediction']!=target and y['prediction']==target;damaged+=x['prediction']==target and y['prediction']!=target
   out[f'family{f}_{role}']={'rows':len(rr),'recovered':recovered,'damaged':damaged,'net':recovered-damaged}
 return out

def equal(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[equal(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
 elif isinstance(a,float):assert isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)
 else:assert a==b,(a,b)

def invented():
 rows=[]
 for f in (0,1):
  for y in (0,1):
   for j in range(4):rows.append({'sample_index':len(rows),'family':f,'target':y,'role':'FIT' if j<3 else 'DEV','available':True,'prediction':'DE'[y]})
 d,s,p=replay_draws(rows,5);assert p==[3,3,3,3] and all(rows[i]['role']=='FIT' for i in d.flat)
 assert np.array_equal(d,replay_draws(rows,5)[0])
 m=metrics(rows);assert all(x['D']['recall']==x['E']['recall']==1 for x in m.values())
 changed=[dict(r) for r in rows];changed[3]['prediction']='E';assert paired(rows,changed)['family0_DEV']['damaged']==1
 assert gate(m,m)['PASS'] is False # zero gain remains FAIL despite perfect values
 for bad in ('/never/v112/forbidden','/never/__pycache__/x'):
  try:safe(bad)
  except AssertionError:pass
  else:raise AssertionError('Lexical guard did not reject')
 return {'completed':True,'PASS':True,'invented_only':True,'groups':['PCG17 exact and DEV loss exclusion','independent class CM/P/R and paired damage','unchanged fixed gate does not admit zero gain','forbidden lexical reject before FS'],'actual_SOURCE_or_weights_or_model_or_fit':0}

def gate(a,b):
 gains=[];drops=[]
 for f in (0,1):
  old=[a[f'family{f}_DEV'][k]['recall'] for k in ('D','E')];new=[b[f'family{f}_DEV'][k]['recall'] for k in ('D','E')]
  if any(v is None for v in old+new):return {'PASS':False,'reason':'Missing fixed SOURCE class denominator','current_admission':False}
  gains.append((sum(new)-sum(old))/2);drops.extend(x-y for x,y in zip(old,new))
 gain=100*sum(gains)/2;drop=100*max(drops);ycb=b['family1_DEV']['E']['recall'];sd=b['family0_DEV']['D']['recall'];se=b['family0_DEV']['E']['recall']
 return {'PASS':gain>=3 and drop<=2 and ycb>=.5 and min(sd,se)>=.9,'mean_family_DEV_balanced_recall_gain_pp':gain,'max_family_class_recall_drop_pp':drop,'YCB_DEV_E_recall':ycb,'SYN_DEV_D_recall':sd,'SYN_DEV_E_recall':se,'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'fixed_SYN_DEV_D_E_floor':.9,'current_admission':False,'automatic_retry':False,'new_current_inference':0}

def check_heads(control,treat):
 full={**{'base.'+k:v for k,v in BASE_SHAPES.items()},**EXTRA_SHAPES};result={}
 for arm,cp,shapes,count in ((ARMS[0],control,BASE_SHAPES,76098),(ARMS[1],treat,full,86054)):
  assert cp['arm']==arm and cp['updates']==1000 and cp['use_lora'] is False and cp['head_parameters']==count and cp['adapter_parameters']==0
  for key in ('head_state_dict','initial_head_state_dict'):assert check_state(cp[key],shapes)==count
  assert state_sha(cp['initial_head_state_dict'])==cp['initial_head_sha256']
  assert cp['same_initial_base_sha256']==INITIAL_BASE_SHA
  assert state_sha(cp['head_state_dict'])!=state_sha(cp['initial_head_state_dict'])
  result[arm]={'parameters':count,'optimizer_states':check_optimizer(cp,shapes),'updates':1000,'initial_sha256':cp['initial_head_sha256'],'final_sha256':state_sha(cp['head_state_dict'])}
 base=control['initial_head_state_dict'];assert state_sha(base)==INITIAL_BASE_SHA
 for key in BASE_SHAPES:assert np.array_equal(base[key].numpy(),treat['initial_head_state_dict']['base.'+key].numpy())
 assert not base['head.2.weight'].count_nonzero() and not base['head.2.bias'].count_nonzero()
 for key in ('relational.geometry_bias.weight','relational.readout.2.weight','relational.readout.2.bias'):assert not treat['initial_head_state_dict'][key].count_nonzero()
 return result

def run(entry_path):
 ep=safe(entry_path);assert ep.parent==H;e=json.loads(raw(ep))
 assert e['approved_actual_completion_QA'] is True and e['both_final1000'] is True
 assert e['allow_original_nonpixel_samples_metadata'] is True
 assert identity(e['verifier'])==(str(safe(__file__)),hashlib.sha256(raw(__file__)).hexdigest())
 white={identity(b):b for b in e['whitelist']};assert len(white)==len(e['whitelist'])
 out=safe(e['output_directory']);assert out.parent==H;out.mkdir(exist_ok=False);access=[]
 def checked(b):
  assert {'path','sha256'}<=set(b)<= {'path','sha256','size_bytes'} and identity(b) in white
  p=safe(b['path']);allowed=p.suffix.lower() in ('.json','.jsonl','.csv','.pt')
  if p.suffix.lower()=='.npz':allowed=b['sha256']==SAMPLES_SHA
  assert allowed,'Only exact small saved artifacts and original nonpixel samples; no feature/pixel/geometry arrays'
  h=hashlib.sha256();size=0
  with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
   for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z);size+=len(z)
  assert h.hexdigest()==b['sha256']
  for bb in (b,white[identity(b)]):
   if 'size_bytes' in bb:assert isinstance(bb['size_bytes'],int) and bb['size_bytes']>=0 and bb['size_bytes']==size
  access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':size});return p
 def js(b):return json.loads(raw(checked(b)))
 def completed(b):
  o=js(b);assert o['completed'] is True and o['PASS'] is True;return o
 def same(a,b):assert identity(a)==identity(b)
 try:
  c=completed(e['complete']);root=js(e['root_fit_entry']);same(c['entry'],e['root_fit_entry']);assert c['code']['sha256']==CODE_SHA and c['root_protocol']['sha256']==PROTOCOL_SHA
  proto=js(c['root_protocol']);assert proto['paired_training']['initial_base_sha256']==INITIAL_BASE_SHA and proto['paired_training']['draws_sha256']==DRAW_SHA
  assert root['source_only'] is True and root['automatic_retry'] is False and root['steps']==1000 and root['threshold_logit']==0
  assert root['old_CONTROL_or_prefix_payload_used'] is False and root['use_V235_counterfactuals'] is False
  for key in ('code','implementation_protocol','root_protocol','feature_complete','samples_npz','geometry_npz','head_module'):same(c[key],root[key])
  assert c['new_optimizer_updates']==dict.fromkeys(ARMS,1000) and c['head_parameters']==dict(zip(ARMS,(76098,86054))) and c['optimizer_state_tensors']==dict(zip(ARMS,(4,10)))
  assert c['new_DEV_head_output_before_bothfinalfreeze']==c['new_V235_CF']==c['old_CONTROL_prefix_DINO_RGB_GT_or_protected_payload_reads']==0 and c['current_admission'] is False and c['automatic_retry'] is False
  packet=completed(c['feature_complete']);same(packet['root_protocol'],c['root_protocol']);same(c['samples_npz'],packet['samples_npz']);same(c['geometry_npz'],packet['geometry_npz'])
  for name,b in c['features'].items():same(b,packet['features'][name]) # strings only, feature bodies never followed
  assert packet['counts']['known']==7617 and packet['counts']['FIT']==5876 and packet['counts']['DEV']==1741 and packet['counts']['all_original_rows']==15040 and packet['counts']['masked_target_minus1']==7423
  sharing=packet['canonical_sharing_QA'];assert sharing['completed'] is True and sharing['PASS'] is True and sharing['known_rows']==7617
  assert all(sharing[k] is True for k in ('active_first384_byte_exact','original_condition32_byte_exact','known_pixel_fractions_byte_exact'))
  q=completed(c['canonical_feature_QA']);init=completed(c['common_initial_draw_QA']);freeze=completed(c['freeze'])
  assert q['checked']==7617 and q['tolerance_used']==q['features_overwritten']==q['source_group_FIT_DEV_overlap']==q['head_DEV_forward_before_QA']==0
  assert all(q[k] is True for k in ('active_first384_byte_exact','original_condition32_byte_exact','original_fractions_byte_exact','original_group_slot_availability_exact'))
  assert q['active_bytes_sha256']==q['node_bytes_sha256']==sharing['active_first384_bytes_sha256']==sharing['node_selected_bytes_sha256']
  assert init['seed']==17 and init['same_initial_base_sha256']==INITIAL_BASE_SHA and init['draws_sha256']==DRAW_SHA and init['updates_per_arm']==1000 and init['new_CONTROL_training'] is True and init['old_CONTROL_reuse'] is False
  assert freeze['all_final_updates']==dict.fromkeys(ARMS,1000) and freeze['new_DEV_head_output_before_bothfinalfreeze']==0 and freeze['old_CONTROL_or_prefix_used'] is False
  same(freeze['canonical_feature_QA'],c['canonical_feature_QA']);same(freeze['common_initial_draw_QA'],c['common_initial_draw_QA'])
  for arm in ARMS:same(freeze['checkpoints'][arm],c['checkpoints'][arm]);same(freeze['logs'][arm],c['logs'][arm])
  assert c['samples_npz']['sha256']==SAMPLES_SHA
  with np.load(checked(c['samples_npz']),allow_pickle=False) as z:
   assert set(z.files)==SAMPLE_KEYS
   data={k:z[k] for k in ('row_id','family','target','fit','available')}
  assert len(data['row_id'])==7617 and data['row_id'].dtype.kind=='U' and data['family'].dtype==data['target'].dtype==np.int8 and data['fit'].dtype==data['available'].dtype==bool
  assert data['available'].all() and data['fit'].sum()==5876 and (~data['fit']).sum()==1741
  rows={arm:parse_csv(raw(checked(c['outputs'][arm])).decode()) for arm in ARMS}
  for arm in ARMS:
   for i,row in enumerate(rows[arm]):
    assert row['row_id']==str(data['row_id'][i]) and row['family']==int(data['family'][i]) and row['target']==int(data['target'][i]) and row['available']==bool(data['available'][i]) and row['role']==('FIT' if data['fit'][i] else 'DEV')
  draws,dsha,pools=replay_draws(rows[ARMS[0]]);assert pools==[4565,187,841,283] and dsha==DRAW_SHA==c['draws_sha256']==freeze['draws_sha256']
  assert c['same_initial_base_sha256']==freeze['same_initial_base_sha256']==INITIAL_BASE_SHA
  ledger=[json.loads(z) for z in raw(checked(c['draw_ledger'])).decode().splitlines()]
  logs={arm:[json.loads(z) for z in raw(checked(c['logs'][arm])).decode().splitlines()] for arm in ARMS};assert len(ledger)==1000 and all(len(z)==1000 for z in logs.values())
  for step,ids in enumerate(draws,1):
   for row in (ledger[step-1],*(logs[a][step-1] for a in ARMS)):
    assert row['update']==step and row['sample_indices']==ids.tolist() and row['draws_sha256']==dsha and row['family_counts']==[16,16] and row['all_fit_available_known'] is True
   for row in (logs[a][step-1] for a in ARMS):
    x,y,z=(float(row[k]) for k in ('family0_BCE','family1_BCE','total_loss'));assert all(math.isfinite(v) and v>=0 for v in (x,y,z));assert math.isclose(z,.5*(x+y),abs_tol=1e-6,rel_tol=1e-6)
  import torch
  cp={arm:torch.load(checked(c['checkpoints'][arm]),weights_only=True,map_location='cpu') for arm in ARMS};states=check_heads(cp[ARMS[0]],cp[ARMS[1]])
  for arm in ARMS:
   same(cp[arm]['entry'],e['root_fit_entry']);same(cp[arm]['feature_complete'],c['feature_complete']);assert cp[arm]['draws_sha256']==dsha
   assert c['initial_head_sha256'][arm]==states[arm]['initial_sha256'] and c['final_head_sha256'][arm]==states[arm]['final_sha256']
  m={arm:metrics(rows[arm]) for arm in ARMS};changes=paired(rows[ARMS[0]],rows[ARMS[1]]);g=gate(m[ARMS[0]],m[ARMS[1]]);summary=completed(c['summary'])
  equal(m,summary['arms']);equal(changes,summary['paired_changes']);equal(g,summary['SOURCE_gate'])
  assert summary['two_SOURCE_referents_separate'] is True and summary['old_V233_CONTROL_reuse'] is False and summary['current_admission'] is False
  assert summary['all_original_rows']==15040 and summary['known_output_rows']==7617 and summary['masked_target_minus1_retained']==7423
  result={'completed':True,'PASS':True,'complete':e['complete'],'root_fit_entry':e['root_fit_entry'],'states':states,'new_updates_both':1000,'same_original_initial_base_sha256':INITIAL_BASE_SHA,'common_original_PCG17_draw_sha256':dsha,'FIT_class_pool_counts':pools,'rows':7617,'FIT':5876,'DEV':1741,'canonical_feature_QA':c['canonical_feature_QA'],'both_final_freeze':c['freeze'],'original_samples_metadata_identity_exact':True,'metrics':m,'paired_changes':changes,'SOURCE_gate':g,'access':access,
   'limits':['No new neural/encoder/model instantiation or training/gradient replay. CPU owned checkpoint tensors and stored logs/CSV only.','The only NPZ decoded is exact original nonpixel samples metadata b837bbe…; feature/pixel/geometry arrays, prefix, original weights/GT/current/held are not accessed.','Byte-sharing and both-freeze-before-DEV chronology are validated through exact reviewed code/sealed runtime receipts; independent CPU feature/head replay is a separate Root scope.','Masked7423/originalqL and full15040 are preserved through bound producer/map authority, not redecoded here. Separate SYN MAIN/YCB associated-r AUX/nativeUNKNOWN, exposed internalDEV and U0 limits remain; no production success or selection claim.']}
  p=out/'REVIEW.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'completed':True,'PASS':True,'REVIEW':{'path':str(p),'sha256':hashlib.sha256(raw(p)).hexdigest()}}))
 except BaseException:
  (out/'FAILED_REVIEW.json').write_text(json.dumps({'completed':True,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False},indent=2)+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:
  o=invented();q=H/'INVENTED_CHECKS.json';assert not q.exists();q.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))
 else:run(a.entry)
