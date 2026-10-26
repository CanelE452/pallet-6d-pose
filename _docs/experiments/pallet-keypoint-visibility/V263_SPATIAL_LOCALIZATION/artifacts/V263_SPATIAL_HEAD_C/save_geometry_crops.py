"""Requested compact deterministic crop replay; no feature forward or training."""
import collections,hashlib,io,json,sys,types
from pathlib import Path
import numpy as np
H=Path(__file__).absolute().parent

def load(path,name):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def run():
 p=load(H/'probe_runtime.py','v263_crop_replay_io');e=json.loads((H/'ACTUAL_GEOMETRY_FEATURE_ENTRY.json').read_text())
 packet=json.loads(p.raw(e['label_packet_complete']));groups=[json.loads(x) for x in p.raw(packet['outputs']['groups']).splitlines()]
 complete=json.loads((H/'source_spatial_features_geometry_v1/COMPLETE.json').read_text())
 sides=np.load(p.safe(complete['outputs']['CROP_SIDE.npy']['path']),allow_pickle=False)
 closure=json.loads(p.raw(e['official_dependency_closure']));crop=p.module(closure['code']['spatial_crop'],'v263_crop_replay_pinned_V90').crop
 rgb96=np.zeros((7040,3,96,96),np.uint8);valid96=np.zeros((7040,1,96,96),bool)
 lru=collections.OrderedDict();original_available=np.zeros(7040,bool)
 for g in groups:
  image=g['image'];path=image['path']
  if path not in lru:
   raw=p.safe(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==image['sha256']
   with np.load(io.BytesIO(raw),allow_pickle=False) as z:lru[path]=np.array(z['RGB_uint8'],copy=True)
   if len(lru)>16:lru.popitem(last=False)
  rgb=lru[path];lru.move_to_end(path)
  for slot,index in enumerate(g['source_indices8']):
   original_available[index]=g['original_available8'][slot]
   if original_available[index]:rgb96[index],valid96[index]=crop(rgb,g['q8'][slot],sides[index],96)
 support=valid96[:,0].reshape(7040,16,6,16,6).any(axis=(2,4))&original_available[:,None,None]
 central=valid96[:,0,47:49,47:49].all(axis=(1,2))&original_available
 assert np.array_equal(support,np.load(p.safe(complete['outputs']['TOKEN_SUPPORT.npy']['path']),allow_pickle=False))
 assert np.array_equal(central,np.load(p.safe(complete['outputs']['CENTRAL4.npy']['path']),allow_pickle=False))
 out=H/'GEOMETRY_RGB96_VALID96_REPLAY.npz';assert not out.exists();np.savez_compressed(out,RGB96=rgb96,valid96=valid96,crop_side=sides,original_available=original_available)
 receipt={'completed':True,'PASS':True,'output':p.binding(out),'original_feature_complete':p.binding(H/'source_spatial_features_geometry_v1/COMPLETE.json'),
  'rows':7040,'RGB96_array_SHA256':hashlib.sha256(rgb96.tobytes()).hexdigest(),'valid96_array_SHA256':hashlib.sha256(valid96.tobytes()).hexdigest(),
  'same_pinned_V90_same_exact_original_q_side_and_bound_RGB':True,'all_input_support_and_central4_byte_equal':True,
  'retrospective_original_RGB_array_byte_comparison':'Not available: original feature run did not persist RGB crop hash; exact deterministic definition replay plus all saved input geometry/support parity',
  'feature_encoder_or_head_forward':0,'optimizer_updates':0,'current_or_held_reads':0}
 path=H/'GEOMETRY_CROP_REPLAY_RECEIPT.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'completed':True,'PASS':True,'receipt':p.binding(path),'crop':p.binding(out)}),flush=True)

if __name__=='__main__':run()
