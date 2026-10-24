"""C5 source metadata/tensor adapter. Import does not read source data or weights."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
P=Path(__file__).resolve().parents[2]
OLD=P/'query_visibility_query_anchored_cue_20261023_v96'
NEW=P/'query_visibility_diverse_expected_cue_source_20261023_v99'
OLD_CODE=OLD/'source_fit_preparation/train_source_once.py'
OLD_CODE_SHA='330f817e45fe8cbb3cda2c48bea5f2502b2f9053ce8e2921892fef44920ade03'
NOLD=31680;NNEW=2304;N=NOLD+NNEW
FIELDS=('p3_roi','rgb96','crop_valid96');STAGES=('BEFORE','COVER','SHAM')
VAL_NAMES=('old_LOCO_BEFORE_D','old_PALLOX_BEFORE_D','old_SHAM_D','old_BEFORE_E','old_COVER_E','new_BEFORE_D','new_SHAM_D','new_COVER_E','new_ALL_S')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open()as f:return [json.loads(x)for x in f if x.strip()]
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def old_helper():assert sha(OLD_CODE)==OLD_CODE_SHA;return module(OLD_CODE,'c5_pinned_c3_sparse_helpers')

def join_new(lineage,gt,queries,configs):
 assert len(lineage)==NNEW and len(gt)==len(queries)==len(configs)==96
 labels=[];states=[];pairs=[];appearance=[];aspect=[];az=[];el=[]
 for b,(g,q,c)in enumerate(zip(gt,queries,configs)):
  assert g['base_index']==q['base_index']==c['base_index']==b
  assert g['base_id']==q['base_id']==c['base_id']==f'V99_BASE_{b:03d}'
  assert g['split']==q['split']==c['split']and g['split']in('TRAIN','VALIDATION')
  selected=g['selected_aux_index'];assert isinstance(selected,int)and 0<=selected<8
  assert g['after_force_U']is False
  for s,stage in enumerate(STAGES):
   assert len(g['stages'][stage])==8
   for k,x in enumerate(g['stages'][stage]):
    i=b*24+s*8+k;r=lineage[i]
    assert r['occurrence_index']==r['input_unit_index']==i
    assert r['base_index']==b and r['base_id']==g['base_id']and r['variant']==stage and r['split']==g['split']
    assert r['aux_corner_index']==k and r['image_index']==b*3+s
    assert r['query_xy']==q['query8_xy'][k]and r['L']==q['L'][k]
    assert r['native8_mapping']==x['native8_mapping']=='UNKNOWN'and x['human_gold']is False
    assert r['source_emulated_not_initializer']is True and q['source_emulated_query_not_initializer_output']is True
    state=x['state'];assert state in('D','E','S','U')
    labels.append({'D':0,'E':1,'S':1,'U':-1}[state]);states.append(state)
    appearance.append(c['appearance_family']);aspect.append(c['aspect_family']);az.append(c['azimuth_index']);el.append(c['elevation_index'])
  pair=np.array([b*24+selected,b*24+8+selected,b*24+16+selected],np.int64)
  assert [states[x]for x in pair]==['D','E','D'],'Selected triplet teacher must remain factual D/E/D'
  pairs.append(pair+NOLD)
 return dict(labels=np.array(labels,np.int8),typed_state=np.array(states),source=np.full(NNEW,'SYNTH_EXPECTED_CUE'),split=np.array([r['split']for r in lineage]),original_split=np.array([r['split']for r in lineage]),variant=np.array([r['variant']for r in lineage]),new_pairs=np.array(pairs,np.int64),appearance=np.array(appearance),aspect=np.array(aspect),azimuth=np.array(az,np.int8),elevation=np.array(el,np.int8))

def load_metadata(bindings):
 h=old_helper()
 om=read(check(bindings['old_manifest']));nm=read(check(bindings['new_manifest']))
 os=read(check(bindings['old_pack_seal']));ns=read(check(bindings['new_pack_seal']))
 assert os['completed']and os['PASS']and ns['completed']and ns['PASS']
 assert same(os['manifest'],bindings['old_manifest'])and same(ns['manifest'],bindings['new_manifest'])
 assert read(check(os['QA']))['PASS']and read(check(ns['QA']))['PASS']
 assert om['occurrences']==NOLD and om['unique_input_units']==1348
 assert nm['query_count']==NNEW and nm['unique_input_units']==NNEW and nm['model_input_keys']==list(FIELDS)
 og=rows(check(bindings['old_GT']));ol=rows(check(om['input_lineage']));old=h.join_metadata(ol,og)
 ng=rows(check(bindings['new_GT']));nl=rows(check(nm['input_lineage']));q=rows(check(bindings['new_queries']));c=rows(check(bindings['new_config']));new=join_new(nl,ng,q,c)
 gp=read(check(nm['GT_only_provenance']));assert same(gp['GT_only_targets'],bindings['new_GT'])and gp['GT_body_read_by_producer']is False
 pairs=[]
 for b,k in np.argwhere(old['pair_ready']):pairs.append([b*33,b*33+1+2*k,b*33+2+2*k])
 oldpairs=np.array(pairs,np.int64);assert oldpairs.shape==(194,3)
 meta={k:np.concatenate([old[k],new[k]])for k in('labels','source','split','original_split','variant')}
 meta.update(known=meta['labels']>=0,typed_state=np.concatenate([np.where(old['labels']<0,'U',np.where(old['labels']==0,'D','E')),new['typed_state']]),domain=np.array(['V96_REAL']*NOLD+['V99_SYNTH']*NNEW),old_meta=old,old_pairs=oldpairs,new_pairs=new['new_pairs'],appearance=np.concatenate([np.full(NOLD,'NOT_APPLICABLE'),new['appearance']]),aspect=np.concatenate([np.full(NOLD,'NOT_APPLICABLE'),new['aspect']]),azimuth=np.concatenate([np.full(NOLD,-1,np.int8),new['azimuth']]),elevation=np.concatenate([np.full(NOLD,-1,np.int8),new['elevation']]),lineage=ol+nl,old_original_C1_V95_label=np.array([r['original_C1_V95_label']for r in og],np.int8),old_lineage=ol,new_lineage=nl,old_manifest=om,new_manifest=nm)
 meta['counts']=counts(meta)
 return meta

def old_masks(meta,split):return old_helper().cohort_masks(meta['old_meta'],split)
def val_masks(meta):
 out={f'old_{k}':np.r_[v,np.zeros(NNEW,bool)]for k,v in old_masks(meta,'VALIDATION').items()}
 m=(meta['domain']=='V99_SYNTH')&(meta['split']=='VALIDATION');t=meta['typed_state'];v=meta['variant']
 out.update(new_BEFORE_D=m&(v=='BEFORE')&(t=='D'),new_SHAM_D=m&(v=='SHAM')&(t=='D'),new_COVER_E=m&(v=='COVER')&(t=='E'),new_ALL_S=m&(t=='S'))
 assert tuple(out)==VAL_NAMES
 return out

def sampling_pools(meta):
 old=old_masks(meta,'TRAIN');m=(meta['domain']=='V99_SYNTH')&(meta['split']=='TRAIN');t=meta['typed_state']
 pools={'old_triplets':meta['old_pairs'][meta['split'][meta['old_pairs'][:,0]]=='TRAIN'],'new_triplets':meta['new_pairs'][meta['split'][meta['new_pairs'][:,0]]=='TRAIN']}
 for name in('LOCO_BEFORE_D','PALLOX_BEFORE_D','BEFORE_E'):pools['old_'+name]=np.flatnonzero(old[name])
 for state in('S','D','E'):pools['new_ALL_'+state]=np.flatnonzero(m&(t==state))
 assert all(len(v)>0 for v in pools.values()),'Empty required TRAIN cohort; no fallback'
 assert all(v.any()for v in val_masks(meta).values()),'Empty required VAL cohort; no fallback'
 return pools

def counts(m):
 out={}
 for domain in('V96_REAL','V99_SYNTH'):
  out[domain]={}
  for split in('TRAIN','VALIDATION'):
   q=(m['domain']==domain)&(m['split']==split)
   out[domain][split]={'rows':int(q.sum()),'typed_states':{t:int((q&(m['typed_state']==t)).sum())for t in('D','E','S','U')},'by_variant':{v:{t:int((q&(m['variant']==v)&(m['typed_state']==t)).sum())for t in('D','E','S','U')}for v in np.unique(m['variant'][q])}}
 out['VAL_cohorts']={k:int(v.sum())for k,v in val_masks(m).items()}
 out['old_accepted_pairs']={s:int((m['split'][m['old_pairs'][:,0]]==s).sum())for s in('TRAIN','VALIDATION')}
 out['new_selected_pairs']={s:int((m['split'][m['new_pairs'][:,0]]==s).sum())for s in('TRAIN','VALIDATION')}
 return out

def draw(pools,generator):
 oldsel=torch.randint(len(pools['old_triplets']),(4,),generator=generator).numpy()
 newsel=torch.randint(len(pools['new_triplets']),(4,),generator=generator).numpy()
 triples=np.concatenate([pools['old_triplets'][oldsel],pools['new_triplets'][newsel]])
 x=[triples.reshape(-1)]
 quotas=(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_S',8),('new_ALL_D',8),('new_ALL_E',8))
 for name,n in quotas:x.append(pools[name][torch.randint(len(pools[name]),(n,),generator=generator).numpy()])
 unshuffled=np.concatenate(x);assert len(unshuffled)==64
 perm=torch.randperm(64,generator=generator).numpy();inv=np.argsort(perm)
 return unshuffled[perm],inv[np.arange(24).reshape(8,3)],dict(old_pair_pool_indices=oldsel.tolist(),new_pair_pool_indices=newsel.tolist(),unshuffled_occurrence_indices=unshuffled.tolist(),shuffle_permutation=perm.tolist(),cohort_draw_quotas=dict(old_triplets=4,new_triplets=4,**dict(quotas)))

def selector(z,available,meta):
 safe=np.where(available,z,0).astype(np.float64);loss=np.logaddexp(0.,safe)-np.where(meta['known'],meta['labels'],0)*safe;parts={}
 for k,m in val_masks(meta).items():
  assert m.any();parts[k]={'count':int(m.sum()),'mean_BCE':float(loss[m].mean()),'unavailable':int((m&~available).sum())}
 return float(np.mean([v['mean_BCE']for v in parts.values()])),parts

def load_inputs(bindings,meta):
 """Actual root entry only. Retain all occurrences, deduplicate references only."""
 h=old_helper();base=rows(check(bindings['old_base_lineage']))
 old,ou=h.materialize_unique_inputs(base,meta['old_lineage'],bindings['old_base_lineage']);assert len(old['p3_roi'])==1348
 nm=meta['new_manifest'];new={k:[]for k in FIELDS};position=0
 for ci,c in enumerate(nm['chunks']):
  assert c['chunk_index']==ci and c['start_occurrence']==position and c['end_occurrence']==position+c['count']
  with np.load(check(c['file']),allow_pickle=False)as f:
   assert set(f.files)==set(FIELDS)
   for k in FIELDS:new[k].append(np.array(f[k],copy=True));assert len(new[k][-1])==c['count']
  position+=c['count']
 assert position==NNEW
 new={k:np.concatenate(v)for k,v in new.items()}
 for i,r in enumerate(meta['new_lineage']):
  assert r['input_unit_index']==i
  for k,hk in(('p3_roi','p3_roi_sha256'),('rgb96','rgb96_sha256'),('crop_valid96','crop_valid96_sha256')):assert hashlib.sha256(np.ascontiguousarray(new[k][i]).tobytes()).hexdigest()==r[hk]
 data={k:np.ascontiguousarray(np.concatenate([old[k],new[k]]))for k in FIELDS};u=np.r_[ou,np.arange(NNEW)+1348]
 assert len(data['p3_roi'])==3652 and data['p3_roi'].shape==(3652,65,9,9)and data['p3_roi'].dtype==np.float16 and np.isfinite(data['p3_roi']).all()
 assert data['rgb96'].shape==(3652,3,96,96)and data['rgb96'].dtype==np.uint8
 assert data['crop_valid96'].shape==(3652,1,96,96)and data['crop_valid96'].dtype==bool
 assert np.isin(data['p3_roi'][:,64],[0,1]).all()
 a=(data['p3_roi'][:,64,4,4]>0)&data['crop_valid96'][:,0,47:49,47:49].all((1,2));a=a[u]
 assert np.array_equal(a,[r['input_facts']['operator_query_available']for r in meta['lineage']])
 return data,u,a
