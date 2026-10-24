"""Independent scalar spot replay for every source target's dense/global cache."""
from pathlib import Path
import hashlib,json,time,sys
import cv2,numpy as np
from audit_controlled_independent import qtruth
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'source_audit/INDEPENDENT_PARTIAL_CACHE_REPLAY.json';assert not out.exists();start=time.monotonic()
 records={}
 for typ,p in [('ORIGINAL_YCB',ROOT/'provenance/TARGETS.jsonl'),('CONTROLLED',ROOT/'controlled/COMPOSITES.jsonl')]:
  records[typ]={r['id']:r for r in map(json.loads,p.open())}
 totals=dict(targets=0,dense_queries=0,global_queries=0,invalid_queries=0);outputs=[]
 for typ in records:
  for split in ['TRAIN','VAL']:
   p=ROOT/'label_cache_v2'/f'{typ}_{split}.npz'
   with np.load(p)as f:g={k:f[k]for k in f.files}
   for ti,targetid in enumerate(g['global_id']):
    r=records[typ][str(targetid)]
    if typ=='CONTROLLED':
     t=r['target_original']
     with np.load(r['labels']['path'])as z:paste=z['paste_mask']
    else:t=r;paste=np.zeros((480,640),bool)
    vis=cv2.imread(t['visible']['path'],0)>0;full=cv2.imread(t['full_proxy']['path'],0)>0
    native=np.where(~full,2,np.where(vis,np.where(paste,1,0),-1)).astype(np.int8)
    ix=np.flatnonzero(g['target_index']==ti);assert len(ix)==8
    assert np.array_equal(g['query'][ix],np.asarray(t['query']))
    assert np.array_equal(g['side'][ix],np.repeat(t['side'],8))
    box=np.asarray(t['box']);assert np.array_equal(g['global_box'][ti],box)
    def check(q,prefix,index):
     x,y=np.floor(np.asarray(q)+.5).astype(int)
     valid=0<=q[0]<640 and 0<=q[1]<480 and 0<=x<640 and 0<=y<480
     assert bool(g[prefix+'_valid'][index])==valid
     if not valid:
      assert g[prefix+'_allowed'][index]==63;assert g[prefix+'_exact_state'][index]==-1;assert g[prefix+'_radius2_state'][index]==-1
      totals['invalid_queries']+=1;return
     tr=qtruth(native,None,None,q)
     assert tr['joint_compatible_mask']==g[prefix+'_allowed'][index],(targetid,prefix,index)
     assert tr['exact_state']==g[prefix+'_exact_state'][index]
     assert tr['radius2_state']==g[prefix+'_radius2_state'][index]
    for k,ri in enumerate(ix):
     for yy,xx in [(0,0),(4,4),(8,8),(1,7),(7,1)]:
      q=np.asarray(t['query'][k])+np.array([xx-4,yy-4])*t['side']/9
      check(q,'dense',(ri,yy,xx));totals['dense_queries']+=1
    center=(box[:2]+box[2:])/2;step=(box[2:]-box[:2])*1.25/64
    for yy,xx in [(0,0),(63,63),(0,63),(63,0),(32,32),(16,16),(48,48),(16,48),(48,16)]:
     q=center+(np.array([xx,yy])+.5-32)*step
     check(q,'global',(ti,yy,xx));totals['global_queries']+=1
    totals['targets']+=1
   outputs.append(dict(path=str(p),sha256=sha(p)))
   print(typ,split,totals,flush=True)
 result=dict(schema='V66_INDEPENDENT_ALL_TARGET_PARTIAL_CACHE_SPOT_REPLAY',PASS=True,
  totals=totals,dense_rule='5fixed gridlocations per8queries pertarget',global_rule='9fixed64gridlocations pertarget',
  independent_scalar_truth_code_sha256=sha(ROOT/'source_audit/audit_controlled_independent.py'),auditor_sha256=sha(__file__),label_outputs=outputs,
  vectorized_helper_or_label_builder_imported=False,current223_read=False,new_models_training_PnP=0,seconds=time.monotonic()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print('CACHE_REPLAY_PASS',totals,flush=True)
if __name__=='__main__':main()
