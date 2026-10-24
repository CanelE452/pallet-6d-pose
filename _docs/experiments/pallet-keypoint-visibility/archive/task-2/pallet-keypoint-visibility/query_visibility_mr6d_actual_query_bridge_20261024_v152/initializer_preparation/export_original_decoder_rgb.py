"""V152 authorized RGB-only preparation; no learned model imports or calls.

Canonical decoder follows the actual V94 R0 initializer cv2.imread callsite.
V63 visibility-CLI/C111 PIL decoding is traced and compared, not conflated.
"""
from pathlib import Path
import argparse,hashlib,json,sys,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_POLICY_BEFORE_MODEL_OR_REFERENCE_BRIDGE.json';POLICY_SHA='a941442d93e0a49554be86716fd324a4ad3bed01d629abb2d2ed77212bce7270'
V151=P/'query_visibility_mr6d_public_native_intake_20261024_v151'
SELECTION=V151/'factual_sample_v1/FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json'
SELECTION_SHA='af217858e5d43fe63079d4bdbeccb8adb70cf4501150e9d9960b38075b20bcba'
DOWNLOAD=V151/'factual_sample_v1/TINY_SOURCE_DOWNLOAD_COMPLETE.json'
DOWNLOAD_SHA='5ba791ec8563224af01af1f177a1082431a225f544e86a3e670482dc08f2ad61'
OUT=V/'canonical_RGB_transport_v1';PROTOCOL=H/'ORIGINAL_DECODER_TRANSPORT_PROTOCOL.json';INPUTS=H/'SOURCE10_ORIGINAL_JPEG_INPUT_ONLY.jsonl'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
INPUT_KEYS={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder','canonical_RGB_NPY'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]

def decoder_trace():
 paths=dict(R0_initializer_callsite=P/'query_visibility_matched_query_scalar_20261023_v94/pallox_actual_initializer_preparation/infer_fixed_pallox_queries.py',
  R0_factory=P/'query_visibility_public_real_20261022_v63/model/deployment_assets/frozen_yolo_features.py',
  V63_visibility_CLI=P/'query_visibility_public_real_20261022_v63/model/deploy_query_visibility.py',
  C111_feature_cache=P/'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py',
  C111_target_context=P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/current_adapter.py')
 snippets={k:p.read_text()for k,p in paths.items()}
 assert 'cv2.imread(row[\'image\'],cv2.IMREAD_COLOR)'in snippets['R0_initializer_callsite']
 assert 'def predict(self, bgr,'in snippets['R0_factory'] and 'cv2.BORDER_REFLECT_101'in snippets['R0_factory'] and 'conf=.001'in snippets['R0_factory']
 assert 'Image.open(request["image"])'in snippets['V63_visibility_CLI'] and 'np.ascontiguousarray(image_rgb[..., ::-1])'in snippets['V63_visibility_CLI']
 assert "fmt=='CANONICAL_RGB_UINT8_HWC_NPY'"in snippets['C111_feature_cache'] and "Image.open(path).convert('RGB')"in snippets['C111_feature_cache']
 assert "Image.open(frame['image']).convert('RGB')"in snippets['C111_target_context']
 return dict(code_bindings={k:bind(p)for k,p in paths.items()},
  canonical_path='V94 actual initializer: cv2.imread(IMREAD_COLOR) BGR -> FrozenYoloFeatures.predict; RGB NPY is its exact contiguous BGR channel reversal.',
  factory='The factory accepts an array, has no file decoder, adds reflected border100 once and calls standard Ultralytics at conf.001/imgsz640/rect/FP32.',
  alternative_paths='V63 visibility CLI and C111 JPEG fallbacks use PIL RGB. C111 also accepts explicit canonical RGB NPY. These are distinct historical callsites; PIL is diagnostic in this R0 intake.',
  no_universal_decoder_claim=True)

def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(SELECTION)==SELECTION_SHA and sha(DOWNLOAD)==DOWNLOAD_SHA
 policy=read(POLICY);selection=read(SELECTION);download=read(DOWNLOAD);assert selection['completed']and download['completed']and download['PASS']
 assert policy['fixed10_selection']==bind(SELECTION)and len(selection['selected_frames'])==10
 records={r['relative_path']:r for r in download['files']};rows=[]
 for i,r in enumerate(selection['selected_frames']):
  assert r['split']=='val' and r['scene'] in ('000000','000002','000005','000006')
  original=records[r['source_files']['RGB']];assert original['source_content_ID_verified']and original['content_identity_kind']=='LFS_SHA256'
  assert original['sha256']==original['expected_source_content_ID']
  rows.append(dict(frame_index=i,id=f"V152_MR6D_val_{r['scene']}_{r['frame_filename_stem']}",source='MR6D',
   source_id=f"val/{r['scene']}:{r['frame_key']}",split='SOURCE_DEVELOPMENT',image=original['local_path'],image_sha256=original['sha256']))
 assert len({r['image']for r in rows})==10
 with INPUTS.open('x')as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 trace=H/'ORIGINAL_FILE_TO_MODEL_DECODER_TRACE.json';write(trace,dict(completed=True,**decoder_trace()))
 write(PROTOCOL,dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=bind(SELECTION),source_download_seal=bind(DOWNLOAD),
  source_inputs=bind(INPUTS),decoder_trace=bind(trace),frames=10,output=str(OUT),canonical_decoder='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB',
  runtime=dict(python=PYTHON,cv2='4.9.0',NumPy='1.26.4'),PIL_comparison='Same pinned runtime PIL.convert(RGB); compare unchanged raw JPEG, never choose decoder by model/GT outcome.',
  source_split='Official release val retained in source_id/selection; all ten fixed scenes are predeclared SOURCE_DEVELOPMENT, reserved000001 is never opened.',
  source_RGB_reads_during_prepare=0,model_weight_GT_pose_depth_mask_reads_or_calls=0,lossless_transport_only=True,automatic_retry=False))
 print(json.dumps(dict(prepared_only=True,protocol=bind(PROTOCOL),inputs=bind(INPUTS),trace=bind(trace)),ensure_ascii=False))

def export():
 # Explicit root instruction authorizes this reversible original-JPEG data preparation.
 # Learned R0 execution remains gated by a separate root entry in the other script.
 import cv2,PIL
 from PIL import Image,features
 assert sha(POLICY)==POLICY_SHA and Path(sys.executable).resolve()==Path(PYTHON).resolve()and cv2.__version__=='4.9.0'and np.__version__=='1.26.4'
 p=read(PROTOCOL);assert p['code']==bind(__file__)and p['policy']==bind(POLICY)
 for k in('selection','source_download_seal','source_inputs','decoder_trace'):check(p[k])
 rows=lines(INPUTS);assert len(rows)==10 and not OUT.exists();OUT.mkdir()
 write(OUT/'STARTED.json',dict(completed=True,policy=p['policy'],protocol=bind(PROTOCOL),root_authorized_CPU_decode_only=True,learned_model_calls=0))
 arrays=[];comparisons=[]
 try:
  cv2.setNumThreads(1)
  for i,r in enumerate(rows):
   assert r['frame_index']==i and sha(r['image'])==r['image_sha256']
   bgr=cv2.imread(r['image'],cv2.IMREAD_COLOR);assert bgr is not None and bgr.dtype==np.uint8 and bgr.ndim==3 and bgr.shape[2]==3
   rgb=np.ascontiguousarray(bgr[:,:,::-1]);pil=np.array(Image.open(r['image']).convert('RGB'),np.uint8,copy=True)
   assert pil.shape==rgb.shape;delta=rgb.astype(np.int16)-pil.astype(np.int16);changed=np.any(delta!=0,axis=2)
   path=OUT/f'RGB_{i:02d}.npy';np.save(path,rgb,allow_pickle=False);again=np.load(path,allow_pickle=False)
   assert again.dtype==rgb.dtype and again.shape==rgb.shape and again.tobytes()==rgb.tobytes() and np.ascontiguousarray(again[:,:,::-1]).tobytes()==bgr.tobytes()
   row=dict(r,decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),raw_hw=list(rgb.shape[:2]),decoder=p['canonical_decoder'],canonical_RGB_NPY=bind(path))
   assert set(row)==INPUT_KEYS;arrays.append(row)
   comparisons.append(dict(frame_index=i,id=r['id'],raw_JPEG=dict(path=r['image'],sha256=r['image_sha256']),raw_hw=row['raw_hw'],
    cv2_RGB_SHA256=row['decoded_RGB_sha256'],pinned_PIL_RGB_SHA256=hashlib.sha256(pil.tobytes()).hexdigest(),exact_same_RGB=not bool(changed.any()),
    changed_pixels=int(changed.sum()),changed_channel_values=int(np.count_nonzero(delta)),maximum_absolute_channel_difference=int(np.abs(delta).max()),
    mean_absolute_channel_difference=float(np.abs(delta).mean()),PIL_is_diagnostic_not_model_input=True))
  manifest=OUT/'CANONICAL10_RGB_MANIFEST_INPUT_ONLY.json';write(manifest,dict(completed=True,rows=arrays,source_inputs=p['source_inputs'],all_original_JPEG_files_and_frame_order_unchanged=True))
  modelinputs=OUT/'INPUTS10_RGB_ONLY.jsonl'
  with modelinputs.open('x')as f:
   for row in arrays:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  diagnostic=OUT/'PINNED_CV2_VS_PINNED_PIL_JPEG_DIAGNOSTIC.json';write(diagnostic,dict(completed=True,frames=10,
   exact_equal_frames=sum(r['exact_same_RGB']for r in comparisons),rows=comparisons,
   runtime=dict(python=sys.executable,Python=sys.version,NumPy=np.__version__,cv2=cv2.__version__,Pillow=PIL.__version__,PIL_libjpeg=features.version('jpg'),PIL_libjpeg_turbo=features.version('libjpeg_turbo')),
   decoder_choice_not_based_on_model_or_GT=True,canonical_RGB_is_fixed_original_R0_cv2_path=True,
   C111_future_guard='Use explicit canonical NPY input; a PIL JPEG fallback is a separate decoder path and may change pixels.'))
  assert sum(x.stat().st_size for x in OUT.iterdir())<64*1024**2
  for row in rows:assert sha(row['image'])==row['image_sha256']
  seal=OUT/'TRANSPORT_COMPLETE.json';write(seal,dict(completed=True,PASS=True,code=bind(__file__),policy=p['policy'],protocol=bind(PROTOCOL),selection=p['selection'],
   source_download_seal=p['source_download_seal'],source_inputs=p['source_inputs'],manifest=bind(manifest),model_inputs=bind(modelinputs),diagnostic=bind(diagnostic),decoder_trace=p['decoder_trace'],
   frames=10,lossless=True,all_original_JPEG_SHA_HW_and_order_preserved=True,source_folds_unchanged=True,no_resampling_or_new_selection=True,
   canonical_decoder=p['canonical_decoder'],root_authorized_CPU_decode_only=True,GT_pose_K_mask_depth_labels_checkpoint_or_model_reads_or_calls=0,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,seal=bind(seal),manifest=bind(manifest),model_inputs=bind(modelinputs),diagnostic=bind(diagnostic),PIL_equal_frames=sum(r['exact_same_RGB']for r in comparisons))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--export',action='store_true');z=a.parse_args();assert z.prepare!=z.export
 prepare()if z.prepare else export()
