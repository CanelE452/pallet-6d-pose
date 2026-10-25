"""Replay every class SHA rank from saved references; no RGB/model/fit."""
import gzip, hashlib, json, os, stat
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
F=R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
def safe(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts and not any(t in str(p).casefold() for t in ('v112','cache','.pyc'))
 for a in [*reversed(p.parents),p]:assert not stat.S_ISLNK(os.lstat(a).st_mode)
 return p
def bind(p):
 h=hashlib.sha256();n=0
 with safe(p).open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b);n+=len(b)
 return dict(path=str(p),sha256=h.hexdigest(),size_bytes=n)
def run():
 cp=F/'public_aux_frozen_features_v1/COMPLETE.json';c=json.loads(safe(cp).read_bytes());assert c['completed'] and c['PASS']
 b=c['outputs']['ALL_REFERENCE_SELECTION.jsonl.gz'];assert bind(b['path'])=={k:b[k] for k in ('path','sha256','size_bytes')}
 h=hashlib.sha256();n=rows=frames=selected=0;chunk=[]
 def check(block):
  frame=block[0]['publisher_frame_path'];assert len(block)==4800 and all(r['publisher_frame_path']==frame and r['original_reference']['reference_grid_index']==i for i,r in enumerate(block))
  chosen=set()
  for label in (0,1):
   indices=[i for i,r in enumerate(block) if r['original_reference']['r_derived_label']==label]
   ranked=sorted(indices,key=lambda i:(hashlib.sha256(('V261_PUBLIC_POINT_STRATIFIED64_V1|'+frame+'|'+str(label)+'|'+str(i)).encode()).digest(),i))
   for rank,i in enumerate(ranked):assert block[i]['class_SHA_rank']==rank
   chosen.update(ranked[:64])
  assert chosen=={i for i,r in enumerate(block) if r['selected_for_feature']}
  assert all(r['sample_index']==-1 for i,r in enumerate(block) if i not in chosen)
  return len(chosen)
 with gzip.open(safe(b['path']),'rb') as s:
  for line in s:
   h.update(line);n+=len(line);rows+=1;chunk.append(json.loads(line))
   if len(chunk)==4800:selected+=check(chunk);frames+=1;chunk=[]
 assert not chunk and frames==284 and rows==1363200 and h.hexdigest()==b['uncompressed_sha256'] and n==b['uncompressed_size_bytes'] and selected==9133
 j=dict(completed=True,PASS=True,all_frames=frames,all_reference_rows=rows,selected_rows=selected,all_class_ranks_and_cap64_recomputed=True,exact_uncompressed_SHA256_and_size=True,feature_complete=bind(cp),selection_ledger={k:b[k] for k in ('path','sha256','size_bytes')},RGB_or_model_or_optimizer_reads=0,checks_do_not_change_frozen_training_or_selection=True)
 p=F/'ROOT_PUBLIC_FULL_SELECTION_REPLAY.json'
 with p.open('x',encoding='utf-8') as s:json.dump(j,s,ensure_ascii=False,sort_keys=True,indent=2);s.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(p))))
if __name__=='__main__':run()
