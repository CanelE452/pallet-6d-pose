#!/usr/bin/env python3
"""Saved V235 head/ledger/CSV QA. No model construction or neural replay."""
import argparse,csv,hashlib,io,json,math,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).parent
ARMS=('CONTROL_LOCAL_CROP','TREAT_CF_DATA_ONLY')

def replay_draws(data,parents,success,steps=1000):
 eligible=data['fit']&data['available']&(data['target']>=0)
 pools=[np.flatnonzero(eligible&(data['family']==f)&(data['target']==t)) for f,t in ((0,0),(0,1),(1,0),(1,1))]
 rng=np.random.Generator(np.random.PCG64(17));draws=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(steps)])
 expected=np.flatnonzero(eligible&(data['target']==0));assert parents.dtype==np.int64 and np.array_equal(np.sort(parents),expected)
 assert success.dtype==bool and success.shape==parents.shape
 lookup={int(parent):j for j,parent in enumerate(parents) if success[j]}
 cpools=[np.flatnonzero(success&(data['family'][parents]==f)) for f in (0,1)]
 rng=np.random.Generator(np.random.PCG64(235));attempt=np.full(draws.shape,-1,np.int64);variant=np.full(draws.shape,-1,np.int8);actual=draws.copy()
 for step,row in enumerate(draws):
  for f in (0,1):
   off=16*f
   for position in range(off+4,off+8):
    parent=int(row[position])
    if parent in lookup:attempt[step,position]=lookup[parent];variant[step,position]=1
   choice=rng.choice(cpools[f],4,replace=True)
   attempt[step,off+12:off+16]=choice;variant[step,off+12:off+16]=0;actual[step,off+12:off+16]=parents[choice]
 changed=attempt>=0;assert eligible[actual].all() and (data['family'][actual]==data['family'][draws]).all()
 assert success[attempt[changed]].all() and (data['target'][actual[changed]]==0).all()
 original_sha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest();h=hashlib.sha256()
 for key,a in [('pair_indices',attempt.astype('<i8')),('variants',variant.astype('i1')),('actual_parent',actual.astype('<i8'))]:h.update(key.encode());h.update(a.tobytes())
 return eligible,draws,attempt,variant,actual,original_sha,h.hexdigest(),[len(p) for p in pools]

def state_sha(state):
 h=hashlib.sha256()
 for name in sorted(state):
  a=state[name].detach().cpu().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()

def check_head(cp):
 expected={'head.0.weight':(64,1186),'head.0.bias':(64,),'head.2.weight':(2,64),'head.2.bias':(2,)}
 assert cp['updates']==1000 and cp['use_lora'] is False and cp['head_parameters']==76098 and cp['adapter_parameters']==0
 if 'adapter_state_dict' in cp:assert cp['adapter_state_dict']=={}
 for key in ('head_state_dict','initial_head_state_dict'):
  s=cp[key];assert set(s)==set(expected)
  for name,a in s.items():assert tuple(a.shape)==expected[name] and a.cpu().numpy().dtype==np.float32 and np.isfinite(a.cpu().numpy()).all()
 assert sum(v.numel() for v in cp['head_state_dict'].values())==76098
 assert state_sha(cp['initial_head_state_dict'])==cp['initial_head_sha256']
 assert not cp['initial_head_state_dict']['head.2.weight'].count_nonzero() and not cp['initial_head_state_dict']['head.2.bias'].count_nonzero()
 opt=cp['optimizer_state_dict'];assert len(opt['param_groups'])==1 and len(opt['state'])==4
 g=opt['param_groups'][0];assert len(g['params'])==4 and g['lr']==.001 and g['weight_decay']==.0001 and g['eps']==1e-8 and tuple(g['betas'])==(.9,.999)
 for s in opt['state'].values():
  assert float(s['step'])==1000
  for v in s.values():
   if hasattr(v,'numpy'):assert np.isfinite(v.cpu().numpy()).all()
 return {'updates':1000,'head_parameters':76098,'optimizer_states':4,'head_state_sha256':state_sha(cp['head_state_dict']),'initial_head_sha256':cp['initial_head_sha256']}

def invented():
 data={'fit':np.ones(16,bool),'available':np.ones(16,bool),'family':np.repeat(np.array([0,1]),8),'target':np.tile(np.repeat(np.array([0,1]),4),2)}
 parents=np.array([0,1,2,3,8,9,10,11],np.int64);success=np.array([1,1,0,1,1,0,1,1],bool)
 a=replay_draws(data,parents,success,5);b=replay_draws(data,parents,success,5)
 assert all(np.array_equal(x,y) for x,y in zip(a[1:5],b[1:5]))
 _,draw,attempt,variant,actual,*_=a
 assert (attempt[:,0:4]==-1).all() and (attempt[:,8:12]==-1).all() and (attempt[:,16:20]==-1).all()
 assert (variant[:,12:16]==0).all() and (variant[:,28:32]==0).all()
 for step,row in enumerate(draw):
  for pos in list(range(4,8))+list(range(20,24)):
   if int(row[pos]) in (2,9):assert attempt[step,pos]==-1 and actual[step,pos]==row[pos]
 return {'completed':True,'PASS':True,'invented_only':True,'PCG17_original_and_PCG235_augmentation_repeat_exact':True,'first4D_and_first4E_original':True,'failed_parent_SHAM_original_fallback':True,'family8D8E_quotas_preserved':True,'actual_source_reads':0}

def run(entry_path):
 ep=Path(entry_path);assert ep.parent==H
 e=json.loads(ep.read_text());assert e['approved_actual_completion_QA'] is True and e['both_final1000'] is True
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==e['verifier']['sha256']
 white={b['path']:b for b in e['whitelist']};assert len(white)==len(e['whitelist'])
 out=Path(e['output_directory']);assert out.parent==H;out.mkdir(exist_ok=False);access=[]
 def ident(b):return b['path'],b['sha256']
 def checked(b):
  assert set(b)<= {'path','sha256','size_bytes'} and {'path','sha256'}<=set(b)
  p=Path(b['path']);assert p.is_absolute() and '..' not in p.parts and str(p) in white and ident(b)==ident(white[str(p)])
  low=str(p).lower()
  for bad in ('v112','__pycache__','current223','publicheld','oldval','source_holdout','v177','v178','v182','prefix.npy','raw_rgb96','raw_valid96'):assert bad not in low,bad
  for q in [p]+list(p.parents):assert not q.is_symlink()
  h=hashlib.sha256();n=0
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk);n+=len(chunk)
  assert h.hexdigest()==b['sha256']
  if 'size_bytes' in b:assert n==b['size_bytes']
  access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':n});return p
 def js(b):return json.loads(checked(b).read_text())
 def same_entry(record,expected):
  p=Path(record['path'])
  if not p.is_absolute():assert p.parts==(Path(expected['path']).name,);p=Path(expected['path']).parent/p
  assert str(p)==expected['path'] and record['sha256']==expected['sha256']
 try:
  c=js(e['complete']);root=js(e['root_fit_entry']);assert c['completed'] is True and c['PASS'] is True;same_entry(c['entry'],e['root_fit_entry'])
  assert root['steps']==1000 and root['threshold_logit']==0 and root['source_only'] is True
  assert c['new_CONTROL_optimizer_updates']==0 and c['new_TREAT_optimizer_updates']==1000 and c['new_head_DEV_forward_before_freeze']==0
  assert c['no_LoRA_or_new_input_channels'] is True and c['original_source_RGB_or_GT_or_protected_payload_reads']==0
  assert c['original_half_up_reference_equivalence_confirmed'] is True
  freeze=js(c['freeze']);reuse=js(c['CONTROL_reuse_QA']);replay=js(c['CONTROL_replay_QA'])
  assert all(d['completed'] is True and d['PASS'] is True for d in (freeze,reuse,replay))
  assert freeze['new_CONTROL_optimizer_updates']==0 and freeze['new_TREAT_optimizer_updates']==1000 and freeze['new_head_DEV_forward_before_freeze']==0
  for arm in ARMS:assert ident(freeze['checkpoints'][arm])==ident(c['checkpoints'][arm])
  samples_path=checked(c['samples_npz'])
  with np.load(samples_path,allow_pickle=False) as z:data={k:z[k] for k in ('row_id','family','target','fit','available','full_view','central_view')}
  assert len(data['target'])==7617
  pairs=js(c['pairs_complete']);parents=np.load(checked(pairs['outputs']['parent_indices.npy']),allow_pickle=False);success=np.load(checked(pairs['outputs']['success.npy']),allow_pickle=False)
  eligible,draws,attempt,variant,actual,dsha,asha,pools=replay_draws(data,parents,success)
  assert len(parents)==5406 and success.sum()==5367 and pools==[4565,187,841,283]
  assert dsha==c['original_draws_sha256']==freeze['same_original_draws_sha256'] and asha==c['CF_draws_sha256']==freeze['CF_draws_sha256']
  fit=np.flatnonzero(eligible);assert freeze['original_FIT_sample_indices']==fit.tolist()
  views=np.unique(np.concatenate([data['full_view'][fit],data['central_view'][fit]]));assert freeze['original_FIT_view_indices']==views.tolist()
  assert freeze['CF_successful_attempt_indices_only_FIT']==np.flatnonzero(success).tolist()
  assert c['CF_feature_mapping']['compact_successful_attempt_order']==np.flatnonzero(success).tolist() and c['CF_feature_mapping']['failed_masked']==39
  ledger=[json.loads(x) for x in checked(c['draw_source_ledger']).read_text().splitlines()];assert len(ledger)==1000
  logs={arm:[json.loads(x) for x in checked(c['logs'][arm]).read_text().splitlines()] for arm in ARMS};assert all(len(v)==1000 for v in logs.values())
  original_keys={'checkpoint':'control_checkpoint','log':'control_log','CSV':'control_csv'}
  current={ 'checkpoint':c['checkpoints'][ARMS[0]],'log':c['logs'][ARMS[0]],'CSV':c['outputs'][ARMS[0]]}
  for key,rootkey in original_keys.items():
   checked(root[rootkey]);checked(current[key]);assert root[rootkey]['sha256']==current[key]['sha256']
   assert ident(reuse['copied'][key])==ident(current[key])
  for step in range(1000):
   k=step+1;l=ledger[step];t=logs[ARMS[1]][step];o=logs[ARMS[0]][step]
   assert l['update']==t['update']==o['update']==k
   assert np.array_equal(l['original_sample_indices'],draws[step]) and np.array_equal(t['original_sample_indices'],draws[step]) and np.array_equal(o['sample_indices'],draws[step])
   for key,expected in [('actual_parent_indices',actual[step]),('augmentation_attempt_indices',attempt[step])]:assert np.array_equal(l[key],expected) and np.array_equal(t[key],expected)
   assert np.array_equal(l['augmentation_variant_minus1Original_0COVER_1SHAM'],variant[step]) and np.array_equal(t['augmentation_variant'],variant[step])
   assert np.array_equal(l['target'],data['target'][draws[step]]) and np.array_equal(l['family'],data['family'][draws[step]])
   assert l['original_draws_sha256']==t['original_draws_sha256']==dsha and l['CF_draws_sha256']==t['CF_draws_sha256']==asha
   assert t['all_FIT_available_known'] is True and t['family_counts']==[16,16]
   for row in (t,o):
    a=float(row['family0_BCE']);b=float(row['family1_BCE']);loss=float(row['total_loss']);assert all(math.isfinite(x) and x>=0 for x in (a,b,loss))
    assert math.isclose(loss,.5*(a+b),abs_tol=1e-6,rel_tol=1e-6)
  import torch
  checkpoints={arm:torch.load(checked(c['checkpoints'][arm]),map_location='cpu',weights_only=True) for arm in ARMS}
  states={arm:check_head(cp) for arm,cp in checkpoints.items()}
  assert states[ARMS[0]]['initial_head_sha256']==states[ARMS[1]]['initial_head_sha256']==c['initial_head_sha256']==freeze['same_initial_head_sha256']
  assert state_sha(checkpoints[ARMS[1]]['head_state_dict'])!=state_sha(checkpoints[ARMS[1]]['initial_head_state_dict'])
  cp=checkpoints[ARMS[1]];assert cp['original_draws_sha256']==dsha and cp['CF_draws_sha256']==asha
  same_entry(cp['entry'],e['root_fit_entry'])
  assert cp['base_state_sha256_before_after']==checkpoints[ARMS[0]]['base_state_sha256_before_after']==c['base_state_sha256_before_after']==freeze['base_sha256']
  csvrows={};metrics={};summary=js(c['summary'])
  for arm in ARMS:
   rr=list(csv.DictReader(io.StringIO(checked(c['outputs'][arm]).read_text())));assert len(rr)==7617;csvrows[arm]=rr
   groups={f'family{f}_{role}':[] for f in (0,1) for role in ('FIT','DEV')}
   for i,r in enumerate(rr):
    f=int(r['family']);target=int(r['target']);role='FIT' if data['fit'][i] else 'DEV';avail=r['available']=='True'
    assert int(r['sample_index'])==i and r['row_id']==str(data['row_id'][i]) and f==int(data['family'][i]) and target==int(data['target'][i]) and r['role']==role and avail==bool(data['available'][i])
    z=[float(r['logit0']),float(r['logit1'])];assert all(map(math.isfinite,z)) and float(r['selected_family_logit'])==z[f]
    assert r['prediction']==('E' if z[f]>=0 else 'D')
    groups[f'family{f}_{role}'].append((target,r['prediction']))
   metrics[arm]={}
   for key,group in groups.items():
    cm=[[0,0,0],[0,0,0]]
    for target,pred in group:cm[target][{'D':0,'E':1,'U':2}[pred]]+=1
    assert summary['arms'][arm][key]['CM_D_E_by_D_E_U']==cm
    assert summary['arms'][arm][key]['rows']==len(group)
    m={'rows':len(group),'CM_D_E_by_D_E_U':cm}
    for ci,cl in enumerate(('D','E')):
     truth=sum(cm[ci]);pred=sum(row[ci] for row in cm);tp=cm[ci][ci]
     value={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
     for vk,v in value.items():
      sv=summary['arms'][arm][key][cl][vk]
      if v is None:assert sv is None
      elif isinstance(v,float):assert math.isclose(v,sv,rel_tol=0,abs_tol=1e-12)
      else:assert v==sv
     m[cl]=value
    metrics[arm][key]=m
  expected_bytes=np.array([[float(r['logit0']),float(r['logit1'])] for r in csvrows[ARMS[0]]],np.float32).tobytes();lh=hashlib.sha256(expected_bytes).hexdigest()
  assert replay['both_logits_byte_exact'] is True and replay['all_packet_rows']==replay['available_replayed']==7617
  assert replay['actual_FP32_logit_bytes_sha256']==replay['original_CSV_FP32_logit_bytes_sha256']==lh
  assert ident(replay['original_CSV_copied_bytes'])==ident(c['outputs'][ARMS[0]])
  result={'completed':True,'PASS':True,'complete':e['complete'],'root_entry':e['root_fit_entry'],'states':states,'same_original_draw_sha256':dsha,'CF_draw_sha256':asha,'original_class_pools':pools,'fixed_updates_each':1000,'new_CONTROL_updates':0,'new_TREAT_updates':1000,'original_rows':7617,'FIT_original_loss_rows':int(eligible.sum()),'failed_parent_count_preserved':39,'CONTROL_CP_log_CSV_copied_byte_exact':True,'CONTROL_both_logits_stored_replay_SHA_matches_CSV':lh,'metrics':metrics,'access':access,'limits':['No model instantiated, feature/neural forward/gradient recomputation or original pretrained/prefix/RGB/valid pixel access.','Live CONTROL replay and DEV chronology rely on reviewed trainer and sealed replay/barrier records; independent QA recomputes CSV FP32 hash, not features.','Stored log objective arithmetic, final safe-loaded owned head/optimizer tensors and exact RNG ledger are checked; not an independent training reproduction.','SYN witness0 MAIN and YCB associated-r AUX are separate SOURCE authorities; no current/held/private read, model selection, threshold or recipe change.']}
  (out/'REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'completed':True,'PASS':True,'review':str(out/'REVIEW.json')}))
 except BaseException:
  (out/'FAILED_REVIEW.json').write_text(json.dumps({'completed':True,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False},indent=2)+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:
  result=invented();(H/'INVENTED_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 else:run(a.entry)
