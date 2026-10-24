"""Isolated two-original source preflight, no after extraction or GT reads."""
from pathlib import Path
import importlib.util,json,subprocess,hashlib
import numpy as np
from PIL import Image
import torch
HERE=Path(__file__).resolve().parent;MAIN=HERE.parent
sp=importlib.util.spec_from_file_location('v77_guard_original_input_builder',MAIN/'build_after_input_pack.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
def freeze():
 before=b.Before();guards=[]
 for idx in (0,1003):
  a,r=before.get(idx);guards.append(dict(r,stage='guard',initial_points=a['query8_xy'].tolist(),roi_side=float(a['L'])))
 b.write(HERE/'GUARDS_INPUT_ONLY.json',guards);(HERE/'EMPTY_AFTER_INPUTS.jsonl').write_text('')
 deps=[MAIN/'build_after_input_pack.py',MAIN/'GENERATED_AFTER_PRODUCER_TESTS.json',HERE/'frozen_feature_worker.py',HERE/'GUARDS_INPUT_ONLY.json',HERE/'EMPTY_AFTER_INPUTS.jsonl',
  b.SYNTH/'PACK_MANIFEST.json',b.SYNTH/'PACK_SEAL.json',b.PUBLIC/'PACK_MANIFEST.json',b.PUBLIC/'PACK_SEAL.json',b.ROI,b.FEATURE]
 b.write(HERE/'GUARD_ONLY_PROTOCOL.json',dict(schema='V77_SEPARATE_ORIGINAL_GUARDS_ONLY',code=b.bind(Path(__file__)),worker_code=b.bind(HERE/'frozen_feature_worker.py'),bindings=[b.bind(p)for p in deps],
  after_manifest=b.bind(HERE/'EMPTY_AFTER_INPUTS.jsonl'),guard_inputs=b.bind(HERE/'GUARDS_INPUT_ONLY.json'),weights=b.bind(b.WEIGHTS),ROI_code=b.bind(b.ROI),feature_package_root='/tmp/pallet-bi-full-execution-20261019-v1',legacy_python_resolved=str(Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()),
  source_original_guard_images=2,expected_predict_calls=4,after_images_allowed=0,GT_current_fit_PnP=0))
def run():
 p=HERE/'GUARD_ONLY_PROTOCOL.json';prot=b.load(p);assert prot['code']==b.bind(Path(__file__))
 for dep in prot['bindings']:b.check(dep)
 before=b.Before();roi,*_=b.helper_module().helpers();guards=b.load(HERE/'GUARDS_INPUT_ONLY.json');log=open(HERE/'WORKER_STDERR.log','xb')
 proc=subprocess.Popen([prot['legacy_python_resolved'],'-u',str(HERE/'frozen_feature_worker.py'),'--protocol',str(p),'--protocol-sha',b.sha(p)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
 results=[]
 for g in guards:
  a,_=before.get(g['original_row_index'],with_roi=True);rgb=np.array(Image.open(g['image']).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==g['RGB_pixels_sha256']
  b.request(proc.stdin,g);h,payload=b.receive(proc.stdout);assert hashlib.sha256(payload.tobytes()).hexdigest()==h['payload_sha256']
  patch=payload if h['kind']=='ROI'else roi['patches'](dict(p3=torch.from_numpy(payload),added_border=100),a['query8_xy'],g['raw_hw'],h['gain'],np.asarray(h['offset']))[0].numpy()
  equal=np.array_equal(patch,a['p3_roi']);diff=int(np.sum(patch!=a['p3_roi']));maxdiff=float(np.max(np.abs(patch.astype(float)-a['p3_roi'].astype(float))))
  if g['original_row_index']==0:
   with np.load(b.ROI.parent/'SYNTH_ROI_CACHE.npz',allow_pickle=False)as z:assert np.array_equal(a['p3_roi'],z['x'][0])
  result=dict(original_row_index=g['original_row_index'],id=g['id'],backend=g['backend'],ROI_bit_exact=equal,ROI_different_values=diff,max_abs_ROI_diff=maxdiff,repeat_bit_exact=h['repeat_bit_exact'],affine=h)
  b.write(HERE/f'GUARD_{g["original_row_index"]:04d}_RESULT.json',result)
  assert equal,'Noafterextract;preservefailedsourceguardwithouttolerancerelaxation'
  results.append(result)
 b.request(proc.stdin,dict(stop=True));h,payload=b.receive(proc.stdout);assert proc.wait()==0;log.close();b.check(h['receipt'])
 b.write(HERE/'BEFORE_GUARDS_PREFLIGHT_PASS.json',dict(completed=True,PASS=True,protocol=b.bind(p),guards=results,worker_receipt=h['receipt'],after_images=0,GT_current_fit_PnP=0))
 print(json.dumps(b.bind(HERE/'BEFORE_GUARDS_PREFLIGHT_PASS.json')))
if __name__=='__main__':
 import sys
 {'freeze':freeze,'run':run}[sys.argv[1]]()
