from __future__ import annotations
import gzip, hashlib, importlib.util, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_square6d_manualpnp_20261010'
DOC = Path(os.environ.get('PALLET_SQUARE_OUTPUT', ROOT/'_docs/experiments'/NAME))
SOURCE = Path(os.environ['PALLET_SOURCE_ROOT']) if 'PALLET_SOURCE_ROOT' in os.environ else ROOT
BASELINE = Path(os.environ.get('PALLET_BASELINE_ROOT', ROOT))
SNAPSHOT = '_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json'
RAW = 'data/pallet/results/pallet_n3_completion_v3'
OLD_DOC = '_docs/experiments/pallet_n3_completion_v3'
SYMMETRY = '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
BACKBONES = ('yolo', 'dope', 'resnet18')
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
SEEDS = (1, 2, 3)
XYZ = np.array([1.1, .15, 1.1])
F_SOURCE_HASHES = {
 'scripts/research/pallet_dim_conditioned_p_v1/pose.py': '4e8c1e6b4c4e885fb671af233ea2d90c416fd7b232b63c75ce9c3ffdeb45b1d7',
 'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py': '312aef156b6b099f6518a17290b7f86256d94ec22c73b1fc40ff924e4aaf9720',
 'challenge/evaluation_v2/pnp_selector.py': '3e89f318c7c4a3396c6e7c2f9280c1755f32d7bd7b3d61e8a56b9bd5d0bd512e'}

def read(path): return json.loads(Path(path).read_text())
def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
 return h.hexdigest()
def finite(x):
 if isinstance(x, dict): return {str(k):finite(v) for k,v in x.items()}
 if isinstance(x, (list,tuple)): return [finite(v) for v in x]
 if hasattr(x, 'tolist'): return finite(x.tolist())
 if isinstance(x, float) and not math.isfinite(x): return None
 return x
def write(name, x):
 DOC.mkdir(parents=True,exist_ok=True)
 with (DOC/name).open('x') as f: json.dump(finite(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(name, rows):
 with gzip.GzipFile(filename=str(DOC/name),mode='xb',mtime=0) as binary:
  for r in rows: binary.write((json.dumps(finite(r),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
def rows(path):
 with gzip.open(path,'rt') as f: return [json.loads(x) for x in f]
def resolve(rel, expected=None):
 p=Path(rel)
 candidates=[p] if p.is_absolute() else [SOURCE/p,BASELINE/p,ROOT/p]
 for f in candidates:
  if f.is_file() and (expected is None or sha(f)==expected): return f
 raise RuntimeError('Missing or changed frozen input: '+(p.name if p.is_absolute() else str(p)))
def binding(path):
 p=Path(path)
 for root in (SOURCE,BASELINE,ROOT):
  try: rel=str(p.relative_to(root));break
  except ValueError: pass
 else: raise RuntimeError('Public input binding requires a relative repository path')
 return dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
def verify_binding(b): return resolve(b['path'],b['sha256'])
def local_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def pose_module():
 for rel,expected in F_SOURCE_HASHES.items():
  assert sha(SOURCE/rel)==expected and sha(ROOT/rel)==expected
 from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
 E,_=legacy()
 assert sha(E.POSE.__file__)==F_SOURCE_HASHES['scripts/research/pallet_dim_conditioned_p_v1/pose.py']
 return E.POSE
def raw_payloads():
 y=read(resolve(RAW+'/SQUARE_YOLO_PREDICTIONS.json'))
 b={name:read(resolve(RAW+f'/predictions/{name}_GREEN0918_119.json')) for name in ('dope','resnet18')}
 return {'yolo':y,**b}
def fixed_predictions(payload,backbone):
 if backbone=='yolo':
  output={}
  for method in ('R0',*(f'N3_DIM_SYM_seed{s}' for s in SEEDS)):
   for r in payload['predictions'][method]:
    candidate=None if r['selected_index'] is None else r['candidates'][r['selected_index']]
    q=np.full((9,2),np.nan) if candidate is None else np.asarray(candidate['keypoints_xy'],float)
    support=np.isfinite(q).all(1)
    output[(r['id'], 'BASE' if method=='R0' else method)]=dict(points=q,support=support,detected=candidate is not None,
     metadata=dict(selected_index=r['selected_index'],candidates=[{k:v for k,v in c.items() if k!='keypoints_xy'} for c in r['candidates']]))
  return output
 output={}
 for r in payload['frames']:
  for method,p in r['predictions'].items():
   name='BASE' if method=='base' else 'N3_DIM_SYM_seed'+method[-1]
   q=np.asarray(p['points'],float);support=np.asarray(p['valid'],bool)
   assert q.shape==(9,2) and support.shape==(9,)
   output[(r['id'],name)]=dict(points=q,support=support,detected=p['detected'],metadata={k:v for k,v in p.items() if k!='points'})
 return output
