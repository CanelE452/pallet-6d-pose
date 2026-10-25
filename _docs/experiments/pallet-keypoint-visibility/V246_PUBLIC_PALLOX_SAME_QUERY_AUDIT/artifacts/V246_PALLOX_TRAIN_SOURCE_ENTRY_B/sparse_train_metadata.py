"""Prospective selected-range reader, NOT an executable/admitted data extractor.

Needs a separately trusted/pinned TRAIN-only byte-range index. This module does
not discover roles/offsets, hash a mixed file, read NPZ or touch image paths.
No CLI or automatic execution. Root must issue a new exact metadata read scope.
"""
import hashlib
import json
import os
import stat
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V94=P/'query_visibility_matched_query_scalar_20261023_v94'
SOURCES={
 'frames':{'path':str(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'),'historical_sha256':'a7691362fe47824cb9ecd66f41e6e030c94fae9985b8580618b5dc225d185b11'},
 'queries':{'path':str(V94/'pallox_actual_initializer/FIXED100/ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl'),'historical_sha256':'3fabf6a2b653feda0cb4cd8c8df14a157f103bcbe17eb71e37d13ad422978767'}}

def safe(path):
 text=str(path)
 if any(x in text.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden before FS')
 p=Path(text)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute literal path required')
 for ancestor in reversed((p,*p.parents)):
  if stat.S_ISLNK(os.lstat(ancestor).st_mode):raise ValueError('Symlink')
 return p

def read_selected_metadata(config):
 """Return exact selected raw records plus parsed input metadata; write nothing."""
 if config.get('Root_approved_sparse_TRAIN_metadata_only') is not True:raise PermissionError('Not admitted: future Root entry required')
 if config.get('index_authority_independent_of_this_read') is not True:raise ValueError('Reader cannot create its own trusted role index')
 if config['sources']!=SOURCES:raise ValueError('Only exact historical input metadata pointers')
 index=config['trusted_TRAIN_index']
 if set(index)!=set(SOURCES) or len(index['frames'])!=44 or len(index['queries'])!=352:raise ValueError('Declared TRAIN44/352 index required')
 # Role/shape/range validation happens before opening either mixed source file.
 for kind,records in index.items():
  end=-1
  for rec in records:
   if rec['declared_split']!='TRAIN_LAB':raise ValueError('TRAIN pre-read role guard')
   if type(rec['offset']) is not int or type(rec['length']) is not int or rec['offset']<end or not 0<rec['length']<=100000:raise ValueError('Closed nonoverlapping row ranges')
   digest=rec['row_sha256']
   if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Selected-row byte SHA required')
   end=rec['offset']+rec['length']
 result={}
 for kind,records in index.items():
  fd=os.open(safe(SOURCES[kind]['path']),os.O_RDONLY|os.O_NOFOLLOW)
  try:
   selected=[]
   for rec in records:
    raw=os.pread(fd,rec['length'],rec['offset'])
    if len(raw)!=rec['length'] or hashlib.sha256(raw).hexdigest()!=rec['row_sha256']:raise ValueError('Selected range changed')
    row=json.loads(raw.decode('utf-8'))
    if row['split']!='TRAIN_LAB' or row['frame_input_id']!=rec['frame_input_id']:raise ValueError('Indexed role/identity mismatch')
    selected.append((raw,row))
   result[kind]=selected
  finally:os.close(fd)
 frames={row['frame_input_id']:row for _,row in result['frames']}
 if len(frames)!=44:raise ValueError('Unique TRAIN frames required')
 groups={key:[] for key in frames}
 for _,row in result['queries']:
  frame=frames[row['frame_input_id']]
  if any(row[k]!=frame[k] for k in ('pair_id','view','split')):raise ValueError('Original frame/query join changed')
  if row['native_semantic_state']!='UNKNOWN' or row['human_gold'] is not False:raise ValueError('No native authority promotion')
  groups[row['frame_input_id']].append(row)
 for rows in groups.values():
  if [r['native_slot'] for r in rows]!=list(range(8)):raise ValueError('All original eight slots/order required')
  if any(r['query_id']!=r['frame_input_id']+':native'+str(r['native_slot']) for r in rows):raise ValueError('Original query ID changed')
 # Raw q/L/reason/U are returned unchanged, including null and finite OOB q.
 # No image stat/hash, whole-file hash, scans, NPZ decode, label or model work.
 return result
