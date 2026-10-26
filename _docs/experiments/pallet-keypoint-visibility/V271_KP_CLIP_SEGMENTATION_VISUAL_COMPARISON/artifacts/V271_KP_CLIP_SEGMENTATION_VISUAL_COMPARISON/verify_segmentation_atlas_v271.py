"""Independent saved-artifact audit; zero learned-model/image-encoder calls."""
from pathlib import Path
import ast,hashlib,json,sys,types
import numpy as np
from PIL import Image
W=Path(__file__).absolute().parent

def safe(p):
 p=Path(p);assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..')),str(p)
 for a in [p,*p.parents]:assert not a.is_symlink(),str(a)
 return p

def binding(p):
 p=safe(p);b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def verify(b):
 a=binding(b['path']);assert a==b,(a,b);return safe(b['path'])

out=safe(W/'segmentation_atlas');complete=json.loads(safe(out/'COMPLETE.json').read_bytes());assert complete['PASS']
for b in complete['outputs'].values():verify(b)
# Only the pinned function AST of the old evaluator executes. It has NumPy and
# builtins, without importing any earlier artifact code or running evaluation.
p=verify(complete['source_transform']);source=safe(p).read_bytes();tree=ast.parse(source,str(p));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='upsample_raw_logits');mod=types.ModuleType('v269_raw_resampler_pinned');mod.np=np;sys.modules[mod.__name__]=mod;exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),mod.__dict__)
p=verify(complete['code']);source=safe(p).read_bytes();own=types.ModuleType('v271_atlas_code_pinned');own.__file__=str(p);sys.modules[own.__name__]=own;exec(compile(source,str(p),'exec'),own.__dict__)
cases=json.loads(safe(out/'CASES.json').read_bytes());rawjson=json.loads(safe(out/'SELECTED_RAW32_LOGITS.json').read_bytes())
with np.load(safe(out/'SELECTED_RAW32_LOGITS.npz'),allow_pickle=False)as z:maps={k:z[k]for k in z.files}
assert list(maps['frame_index'])==rawjson['frame_index']
checks=[]
for k,c in enumerate(cases):
 fi=c['frame_index'];rgb=np.asarray(Image.open(safe(out/c['rgb_png'])).convert('RGB'))
 for arm,v in c['models'].items():
  raw=maps[arm+'_logit'][k];assert np.array_equal(raw,np.array(rawjson['models'][arm][k],np.float32))
  dense=mod.upsample_raw_logits(raw,rgb.shape[:2]);ours=own.upsample_raw_logits(raw,rgb.shape[:2]);assert np.array_equal(dense,ours)
  mask=dense>=0;hard=np.asarray(Image.open(safe(out/v['binary_png'])).convert('RGB'));assert set(np.unique(hard)).issubset({0,255});assert np.array_equal(hard,np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=2))
  cutout=np.asarray(Image.open(safe(out/v['cutout_png'])).convert('RGB'));expected=np.full_like(rgb,48);expected[mask]=rgb[mask];assert np.array_equal(cutout,expected)
  overlay=np.asarray(Image.open(safe(out/v['overlay_png'])).convert('RGB'));expected=rgb.copy();expected[mask]=(expected[mask]*.5+np.array([25,230,80])*.5).astype(np.uint8);assert np.array_equal(overlay,expected)
  prob=1/(1+np.exp(-np.clip(dense,-80,80)));expected=np.stack((20+20*prob,30+220*prob,25+65*prob),-1).clip(0,255).astype(np.uint8);assert np.array_equal(np.asarray(Image.open(safe(out/v['probability_png'])).convert('RGB')),expected)
  assert float(mask.mean())==v['target_fraction'];checks.append(dict(frame_index=fi,model=arm,raw32_JSON_NPZ_exact=True,original_resampler_exact=True,binary_values_exact=True,cutout_exact=True,overlay_exact=True,softpalette_exact=True))
p=safe(W/'SEGMENTATION_ATLAS_PIXEL_QA.json');assert not p.exists();p.write_text(json.dumps(dict(PASS=True,completed=True,code=binding(Path(__file__).absolute()),atlas_complete=binding(out/'COMPLETE.json'),checks=checks,output_bindings_verified=len(complete['outputs']),raw_frame_model_checks=len(checks),original_resampler_transform_max_abs_diff=0,neural_model_or_encoder_calls=0,new_fits=0,real_GT_pixel_reads=0,whole_RGB=True,original_point_crop_reads=0,notes='AST-executed pinned pure NumPy original resampler only, no old evaluator main/model. PNG binary is exactly0/255; gray cutout and overlays verified pixelwise for all50 frame-model pairs.'),ensure_ascii=False,indent=2,sort_keys=True)+'\n');print('PASS independent resampler+50binary/cutout/overlay/softmap pixel audits',binding(p))
