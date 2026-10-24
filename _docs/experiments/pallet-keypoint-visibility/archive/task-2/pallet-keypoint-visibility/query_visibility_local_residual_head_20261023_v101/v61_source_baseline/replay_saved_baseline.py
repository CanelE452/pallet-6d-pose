from pathlib import Path
import hashlib,json,numpy as np
H=Path(__file__).resolve().parent;O=H/'outputs'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2);f.write('\n')
s=read(O/'BASELINE_SEAL.json');p=read(ck(s['protocol']));res=read(ck(s['results']));zfile=ck(s['scores']);b=p['bindings'];om=read(ck(b['old_manifest']));nm=read(ck(b['new_manifest']));line=lines(ck(om['input_lineage']))+lines(ck(nm['input_lineage']))
with np.load(zfile,allow_pickle=False)as f:
 a={k:np.array(f[k])for k in f.files}
assert len(a['raw_visible_logit'])==len(line)==33984 and len(a['unit_visible_logit'])==3652
assert np.array_equal(a['raw_visible_logit'],a['unit_visible_logit'][a['input_unit_index']])
assert np.array_equal(a['hidden_logit'],-a['raw_visible_logit'])
expected=np.where(a['available'],np.where(a['raw_visible_logit']>=0,0,1),-1)
assert np.array_equal(expected,a['decoded_aux_prediction'])and np.isfinite(a['raw_visible_logit']).all()
for i,r in enumerate(line):
 assert a['occurrence_index'][i]==i and a['original_split'][i]==r['split']and a['variant'][i]==r['variant']
 ident=r.get('occurrence_id',r.get('image_id','')+':AUX'+str(r.get('aux_corner_index',-1)));assert a['occurrence_id'][i]==ident
 q=np.array([np.nan,np.nan]if r['query_xy']is None else r['query_xy']);L=np.nan if r['L']is None else r['L']
 assert np.array_equal(a['query_xy'][i],q,equal_nan=True)and(a['L'][i]==L or(np.isnan(a['L'][i])and np.isnan(L)))
 assert a['available'][i]==r['input_facts']['operator_query_available']
q=np.load(zfile,allow_pickle=False)
checked=0;pairsummary={}
for dom,ss in res['results'].items():
 for split,v in ss.items():
  mask=(a['domain']==dom)&(a['split']==split);known=a['label']>=0
  cm=[[int((mask&known&(a['label']==y)&(expected==pr)).sum())for pr in(0,1,-1)]for y in(0,1)]
  assert cm==v['all_known']['CM'];checked+=1
  pairs=a['old_pairs']if dom=='V96_REAL'else a['new_pairs'];pairs=pairs[a['split'][pairs[:,0]]==split];hit=(expected[pairs]==np.array([0,1,0])).all(1)
  assert int(hit.sum())==v['paired']['all3_correct']and len(pairs)==v['paired']['N'];pairsummary[dom+'|'+split]=dict(N=len(pairs),correct=int(hit.sum()))
assert res['model_state_before']==res['model_state_after']and sha(ck(p['checkpoint']))==p['checkpoint']['sha256']
result=dict(completed=True,PASS=True,code=bind(Path(__file__)),baseline_seal=bind(O/'BASELINE_SEAL.json'),all33984_unitexpansion_ID_q_L_rawsplit_variant_factualU_exact=True,original_presence_sign0_D_decoder_exact=True,factual_U=int((~a['available']).sum()),teacher_U=int((a['label']<0).sum()),allknown_CM_checked=checked,matched_triplets=pairsummary,no_new_model_forward_backbone_fit_private_current_PnP=0)
write(H/'SAVED_BASELINE_REPLAY.json',result);write(H/'FINAL_BASELINE_RECEIPT.json',dict(completed=True,PASS_execution_and_saved_replay=True,baseline_seal=bind(O/'BASELINE_SEAL.json'),saved_replay=bind(H/'SAVED_BASELINE_REPLAY.json'),protocol=bind(H/'BASELINE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'),execution_log=bind(H/'ACTUAL_V61_SOURCE_EXECUTION.log'),actual_terminal_exit=0,actual_head_forward_batches=res['inference_batches'],actual_repeat_head_batches=1,new_backbone_fit_private_current_PnP=0,ENOSPC_before_directory_model_or_run_preserved_as_prior_message=True,original_v61_target_not_redefined=True))
print(json.dumps(dict(final=bind(H/'FINAL_BASELINE_RECEIPT.json'),result=result)))
