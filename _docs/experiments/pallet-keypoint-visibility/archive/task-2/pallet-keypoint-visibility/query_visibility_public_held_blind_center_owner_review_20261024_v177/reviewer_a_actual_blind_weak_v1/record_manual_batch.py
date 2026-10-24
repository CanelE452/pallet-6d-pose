from pathlib import Path
import sys,json,hashlib
V=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
src=[json.loads(s) for s in (V/'blind_review_evidence_v1/ALL360_BLINDED_REVIEW_INPUTS.jsonl').read_text().splitlines()]
by={}
for r in src:by.setdefault(r['frame_index'],{})[r['native_slot']]=r
p=O/'ALL360_OBSERVATIONS.jsonl';old=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
seen={r['original_global_index'] for r in old};batch=json.loads(sys.stdin.read());out=[]
centers={'T':'TARGET_VISIBLE_SUPPORTED','F':'FOREIGN_VISIBLE_SUPPORTED','B':'BACKGROUND_SUPPORTED','U':'UNKNOWN'}
for frame,entries in batch.items():
 frame=int(frame);assert len(entries)==8
 for k,(c,cp,reason,note) in enumerate(entries):
  r=by[frame][k];assert r['original_global_index'] not in seen and c in centers and cp in ('YES','NO','UNKNOWN')
  forced=r['forced_center_unknown_reason']
  if forced:assert c=='U';reason=forced
  if not r['feature_constructed']:assert cp=='UNKNOWN'
  sh=r['frame_sheet'];assert hashlib.sha256(Path(sh['path']).read_bytes()).hexdigest()==sh['sha256']
  row={x:r[x] for x in ['original_global_index','frame_index','native_slot','query_id','frame_id','source_id','query_xy','L','split','query_available','feature_constructed','local_central4_available','center4_valid_bits']}
  row.update(center_owner=centers[c],center_reason=reason,crop_presence=cp,crop_reason=('NO_CONSTRUCTED_CROP' if not r['feature_constructed'] else 'CLEAR_SELECTED_TARGET_MATERIAL' if cp=='YES' else 'NO_SELECTED_PALLET_MATERIAL_VISIBLE' if cp=='NO' else 'SELECTED_INSTANCE_OR_VISUAL_SUPPORT_UNCERTAIN'),observation=note,sheet_path=sh['path'],actually_viewed_sheet=sh,reviewer='A',annotation_authority='assistant weak RGB observation, not human/pixel/native gold',training_admitted=False)
  out.append(row);seen.add(r['original_global_index'])
allrows=sorted(old+out,key=lambda r:r['original_global_index']);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in allrows));print(json.dumps({'frames_recorded':len({r['frame_index'] for r in allrows}),'rows_recorded':len(allrows)}))
