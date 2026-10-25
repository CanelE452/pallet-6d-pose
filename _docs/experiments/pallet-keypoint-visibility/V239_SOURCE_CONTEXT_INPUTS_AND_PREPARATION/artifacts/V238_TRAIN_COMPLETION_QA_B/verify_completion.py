"""Independent saved V238 head/optimizer/draw/CSV QA. No neural/encoder replay."""
import argparse,csv,hashlib,io,json,math,os,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).absolute().parent
ARMS=('CONTROL_LOCAL_CROP','TREAT_CONTEXT_RESIDUAL')
ROOT_ENTRY_SHA='08bd9fd1dd6afabf51d3b6ce2a443e2dce7d88d1485172865fc94903bf192e8b'
CODE_SHA='92442bb4a4e700eed89a6621a6706be5b1f9d305284d3c173c495591bd7bbd35'
PROTOCOL_SHA='17b3b00b1c8b801dc536460f555d40ff8df37f6fe2355955a5e1ad66ecb29ba3'
BASE_SHAPES={'head.0.weight':(64,1186),'head.0.bias':(64,),'head.2.weight':(2,64),'head.2.bias':(2,)}
EXTRA_SHAPES={'relational.project.weight':(16,384),'relational.geometry_bias.weight':(1,2),'relational.readout.0.weight':(32,116),'relational.readout.0.bias':(32,),'relational.readout.2.weight':(2,32),'relational.readout.2.bias':(2,)}

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

def check_heads(control,treat):
 full={**{'base.'+k:v for k,v in BASE_SHAPES.items()},**EXTRA_SHAPES}
 for cp,shapes,count in ((control,BASE_SHAPES,76098),(treat,full,86054)):
  assert cp['updates']==1000 and cp['use_lora'] is False and cp['head_parameters']==count and cp['adapter_parameters']==0
  for key in ('head_state_dict','initial_head_state_dict'):assert check_state(cp[key],shapes)==count
  assert state_sha(cp['initial_head_state_dict'])==cp['initial_head_sha256']
  check_optimizer(cp,shapes)
 assert treat['base_head_parameters']==76098 and treat['residual_parameters']==9956
 initial=control['initial_head_state_dict'];assert check_state(treat['original_initial_base_state_dict'],BASE_SHAPES)==76098
 initial_sha=state_sha(initial);assert initial_sha==treat['original_initial_base_sha256']==state_sha(treat['original_initial_base_state_dict'])
 for key in BASE_SHAPES:
  assert np.array_equal(initial[key].numpy(),treat['initial_head_state_dict']['base.'+key].numpy())
 assert not initial['head.2.weight'].count_nonzero() and not initial['head.2.bias'].count_nonzero()
 for key in ('relational.readout.2.weight','relational.readout.2.bias','relational.geometry_bias.weight'):assert not treat['initial_head_state_dict'][key].count_nonzero()
 assert state_sha(treat['head_state_dict'])!=state_sha(treat['initial_head_state_dict'])
 assert control['base_state_sha256_before_after']==treat['base_state_sha256_before_after']
 return {'CONTROL_parameters':76098,'TREAT_parameters':86054,'residual_parameters':9956,'optimizer_states':[4,10],'optimizer_steps_each':1000,'original_initial_base_sha256':initial_sha,'TREAT_initial_head_sha256':treat['initial_head_sha256'],'TREAT_final_head_sha256':state_sha(treat['head_state_dict'])}

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

def gate(a,b):
 gains=[];drops=[]
 for f in (0,1):
  old=[a[f'family{f}_DEV'][k]['recall'] for k in ('D','E')];new=[b[f'family{f}_DEV'][k]['recall'] for k in ('D','E')]
  if any(v is None for v in old+new):return {'PASS':False,'reason':'Missing prospective SOURCE class denominator','current_admission':False}
  gains.append((sum(new)-sum(old))/2);drops.extend(x-y for x,y in zip(old,new))
 gain=100*sum(gains)/2;drop=100*max(drops);e=b['family1_DEV']['E']['recall']
 return {'PASS':gain>=3 and drop<=2 and e>=.5,'mean_family_DEV_balanced_recall_gain_pp':gain,'max_family_class_recall_drop_pp':drop,'YCB_DEV_E_recall':e,'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'current_admission':False,'new_current_inference':0,'automatic_retry':False}

def equal(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[equal(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
 elif isinstance(a,float):assert isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)
 else:assert a==b,(a,b)

def run(entry_path):
 ep=safe(entry_path);assert ep.parent==H;e=json.loads(raw(ep))
 assert e['approved_actual_completion_QA'] is True and e['both_final1000'] is True
 assert identity(e['verifier'])==(str(safe(__file__)),hashlib.sha256(raw(__file__)).hexdigest())
 white={identity(b):b for b in e['whitelist']};assert len(white)==len(e['whitelist'])
 out=safe(e['output_directory']);assert out.parent==H;out.mkdir(exist_ok=False);access=[]
 def checked(b):
  assert {'path','sha256'}<=set(b)<= {'path','sha256','size_bytes'} and identity(b) in white
  p=safe(b['path']);assert p.suffix.lower() in ('.json','.jsonl','.csv','.pt'), 'Only small explicit saved completion artifacts; no feature/source arrays'
  h=hashlib.sha256();n=0
  with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk);n+=len(chunk)
  assert h.hexdigest()==b['sha256']
  for bb in (b,white[identity(b)]):
   if 'size_bytes' in bb:assert isinstance(bb['size_bytes'],int) and bb['size_bytes']>=0 and bb['size_bytes']==n
  access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':n});return p
 def js(b):return json.loads(raw(checked(b)))
 def completed(b):
  d=js(b);assert d['completed'] is True and d['PASS'] is True;return d
 def ident_equal(a,b):assert identity(a)==identity(b)
 try:
  c=completed(e['complete']);root=js(e['root_fit_entry']);assert e['root_fit_entry']['sha256']==ROOT_ENTRY_SHA
  ident_equal(c['entry'],e['root_fit_entry']);assert c['code']['sha256']==CODE_SHA
  protocol=js(c['root_protocol']);assert c['root_protocol']['sha256']==PROTOCOL_SHA
  assert root['steps']==1000 and root['threshold_logit']==0 and root['active_alignment_atol']==1e-6 and root['use_V235_counterfactuals'] is False
  assert root['source_only'] is True and root['automatic_retry'] is False
  for k in ('code','implementation_protocol','root_protocol','source_complete'):ident_equal(c[k],root[k])
  assert c['new_CONTROL_optimizer_updates']==0 and c['new_TREAT_optimizer_updates']==1000 and c['new_V235_CF']==0 and c['new_DINO_or_LoRA_parameters']==0
  assert c['new_head_DEV_forward_before_freeze']==c['new_head_DEV_forward_before_global_QA']==0
  assert c['head_parameters']==dict(zip(ARMS,(76098,86054))) and c['current_admission'] is False and c['automatic_retry'] is False
  old=completed(c['source_complete'])
  freeze=completed(c['freeze']);reuse=completed(c['CONTROL_reuse_QA']);replay=completed(c['CONTROL_replay_QA'])
  fitqa=completed(c['FIT_active_descriptor_QA']);devqa=completed(c['DEV_active_descriptor_QA']);globalqa=completed(c['global_QA_barrier'])
  for a in ARMS:ident_equal(freeze['checkpoints'][a],c['checkpoints'][a])
  assert freeze['new_CONTROL_optimizer_updates']==0 and freeze['new_TREAT_optimizer_updates']==1000 and freeze['new_head_DEV_forward_before_freeze']==0 and freeze['new_V235_CF']==0
  ident_equal(freeze['CONTROL_reuse_QA'],c['CONTROL_reuse_QA']);ident_equal(freeze['FIT_active_descriptor_QA'],c['FIT_active_descriptor_QA'])
  assert freeze['original_prefix_numeric_rows_before_freeze_only_FIT'] is True
  for q,n in ((fitqa,5876),(devqa,1741)):
   assert q['checked']==n and q['atol']==1e-6 and q['rtol']==0 and math.isfinite(q['max_abs']) and 0<=q['max_abs']<=1e-6
  assert fitqa['mismatches_ignored_or_overwritten']==0 and devqa['FIT_plus_DEV_unique']==7617 and devqa['new_DEV_logits_before_own_descriptor_check']==0
  assert globalqa['DEV_checked']==1741 and globalqa['CONTROL_replayed']==globalqa['all_original_rows']==7617 and globalqa['new_head_DEV_forward_before_global_QA']==0
  for k,ck in (('freeze','freeze'),('DEV_active_descriptor_QA','DEV_active_descriptor_QA'),('CONTROL_replay_QA','CONTROL_replay_QA')):ident_equal(globalqa[k],c[ck])
  assert reuse['new_CONTROL_optimizer_updates']==0
  for k,rootkey,container in (('checkpoint','control_checkpoint','checkpoints'),('log','control_log','logs'),('CSV','control_csv','outputs')):
   ident_equal(reuse['original'][rootkey],root[rootkey]);ident_equal(reuse['copied'][k],c[container][ARMS[0]])
   ident_equal(root[rootkey],old[container][ARMS[0]])
   assert root[rootkey]['sha256']==c[container][ARMS[0]]['sha256'] # verified copied bytes, inert original binding
  csvrows={a:parse_csv(raw(checked(c['outputs'][a])).decode()) for a in ARMS}
  a,b=(csvrows[k] for k in ARMS)
  for x,y in zip(a,b):assert all(x[k]==y[k] for k in ('sample_index','row_id','family','target','role','available'))
  draw,dsha,pools=replay_draws(a);assert pools==[4565,187,841,283]
  assert dsha==c['original_draws_sha256']==freeze['same_original_draws_sha256']==reuse['same_original_draws_sha256']==old['draws_sha256']
  fitids=[r['sample_index'] for r in a if r['role']=='FIT'];assert freeze['original_FIT_sample_indices']==fitids
  logs={k:[json.loads(x) for x in raw(checked(c['logs'][k])).decode().splitlines()] for k in ARMS}
  ledger=[json.loads(x) for x in raw(checked(c['draw_source_ledger'])).decode().splitlines()];assert len(ledger)==1000 and all(len(v)==1000 for v in logs.values())
  for s,indices in enumerate(draw,1):
   for row in [ledger[s-1],logs[ARMS[0]][s-1],logs[ARMS[1]][s-1]]:
    assert row['update']==s and row['sample_indices']==indices.tolist() and row['draws_sha256']==dsha
    assert row['family_counts']==[16,16] and row['all_fit_available_known'] is True
   for row in [logs[k][s-1] for k in ARMS]:
    x,y,z=(float(row[k]) for k in ('family0_BCE','family1_BCE','total_loss'));assert all(math.isfinite(v) and v>=0 for v in (x,y,z))
    assert math.isclose(z,.5*(x+y),abs_tol=1e-6,rel_tol=1e-6)
  import torch
  cp={k:torch.load(checked(c['checkpoints'][k]),weights_only=True,map_location='cpu') for k in ARMS}
  states=check_heads(cp[ARMS[0]],cp[ARMS[1]])
  assert cp[ARMS[0]]['draws_sha256']==cp[ARMS[1]]['original_draws_sha256']==dsha
  ident_equal(cp[ARMS[1]]['entry'],e['root_fit_entry'])
  for k in ARMS:ident_equal(cp[k]['official_base'],root['model_loader_config']['official_state']) # no pointer follow
  assert states['original_initial_base_sha256']==c['original_initial_base_sha256']==freeze['original_initial_base_sha256']==reuse['same_original_initial_head_sha256']==old['head_initial_sha256']
  assert states['TREAT_initial_head_sha256']==c['initial_head_sha256']
  assert c['base_state_sha256_before_after']==cp[ARMS[0]]['base_state_sha256_before_after']
  m={k:metrics(csvrows[k]) for k in ARMS};summary=completed(c['summary'])
  equal(m,summary['arms']);changes=paired(a,b);equal(changes,summary['paired_changes']);g=gate(m[ARMS[0]],m[ARMS[1]]);equal(g,summary['SOURCE_gate'])
  assert summary['two_referents_not_merged'] is True and summary['original7617_preserved'] is True and summary['masked7423_retained_in_full15040_ledger'] is True and summary['current_admission'] is False
  cb=np.asarray([r['logits'] for r in a],np.float32).tobytes();ch=hashlib.sha256(cb).hexdigest()
  assert replay['rows']==7617 and replay['both_logits_byte_exact'] is True and replay['actual_FP32_bytes_sha256']==replay['original_CSV_FP32_bytes_sha256']==ch
  assert replay['original_features_for_batch64_schedule'] is True
  result={'completed':True,'PASS':True,'complete':e['complete'],'root_fit_entry':e['root_fit_entry'],'states':states,'CONTROL_CP_log_CSV_copy_SHA_exact':True,'new_CONTROL_updates':0,'new_TREAT_updates':1000,'original_PCG17_draws_sha256':dsha,'FIT_class_pool_counts':pools,'SOURCE_rows':7617,'FIT':5876,'DEV':1741,'descriptor_QA':{'FIT':fitqa,'DEV':devqa},'global_QA_barrier':c['global_QA_barrier'],'CONTROL_stored_both_logits_FP32_SHA_matches':ch,'metrics':m,'paired_changes':changes,'SOURCE_gate':g,'access':access,
   'limits':['No neural/encoder/training replay, source arrays/pixels/features, original pretrained/prefix/rawGT/current/held payload read.','Chronology, frozen encoder and active-descriptor equality rely on reviewed exact trainer and sealed runtime records; values are validated but not independently re-forwarded.','Original7617 metadata identity is checked from the byte-reused CONTROL CSV against the treatment CSV. Original q/L/raw pixels remain under earlier input QA authority.','SYN witness0 MAIN and YCB associated-r AUX/nativeUNKNOWN remain separate; SOURCE internalDEV is exposed, not blind generalization or current improvement. U0 here is forced binary, not calibrated confidence.','Stored optimizer/log arithmetic/RNG checks do not reproduce gradients or training. No model/data/threshold/recipe selection or new fit.']}
  p=out/'REVIEW.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'completed':True,'PASS':True,'REVIEW':{'path':str(p),'sha256':hashlib.sha256(raw(p)).hexdigest()}}))
 except BaseException:
  (out/'FAILED_REVIEW.json').write_text(json.dumps({'completed':True,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False},indent=2)+'\n');raise

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
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:
  o=invented();q=H/'INVENTED_CHECKS.json';assert not q.exists();q.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))
 else:run(a.entry)
