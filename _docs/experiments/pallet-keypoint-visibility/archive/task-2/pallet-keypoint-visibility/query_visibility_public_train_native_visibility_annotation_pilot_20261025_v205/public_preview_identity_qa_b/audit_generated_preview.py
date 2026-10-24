"""Generated PNG/manifest/scalar identity QA only; never reopen original inputs or labels."""
from pathlib import Path
import argparse,hashlib,io,json,os,stat
import numpy as np
from PIL import Image
O=Path(__file__).parent;V=O.parent;H=V/'preview_preparation_a';OUT=V/'fixed6_public_TRAIN_input_previews_v1'
def need(v,m):
 if not v:raise AssertionError(m)
def rb(p):
 p=Path(p);need(p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold(),'Forbidden name before FS')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for k in p.parts[1:-1]:n=os.open(k,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular no-symlink input')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def get(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Bound generated metadata drift');return json.loads(z)
def ah(z):return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
def image(b,mode):
 p=Path(b['path']);need(OUT in p.parents and p.suffix=='.png','Generated PNG only, no original data source')
 z=rb(p);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Saved PNG encoded SHA')
 with Image.open(io.BytesIO(z))as im:
  need(im.mode==mode,'Exact PNG mode');return np.asarray(im).copy()
def write(n,d):
 with (O/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main(cp):
 need(cp==OUT/'COMPLETE.json','Exact completed preview path only')
 need(bind(cp)['sha256']=='002c59fd20149c93b2212e2b953cc0e8c8e9408824f09efa9480d1e5b9a48bca','Exact root completed preview SHA');ready=json.loads(rb(H/'FINAL_READY_V2_LAYOUT.json'));need(bind(H/'FINAL_READY_V2_LAYOUT.json')['sha256']=='e6356e497071dfa7eacf3ce66ee33310ae1f734a5e065757c3cdbb1d2e98d6fa','Pinned READY')
 proto=get(ready['protocol']);plan=get(ready['selection']);complete=json.loads(rb(cp));need(complete['completed']and complete['PASS']and complete['code']==ready['code']and complete['protocol']==ready['protocol']and complete['selection']==ready['selection'],'Actual source/selection closure')
 need(complete['frames6']and complete['native_rows48']and complete['original_inputU']==8,'Six/48/U8 retained')
 need(complete['source_HOLDOUT_current_publicheld_COMMON_VAL_pixels_read']==0 and complete['oldweaklabels_GT_scores_model_or_initializer_reads']==0 and complete['new_qL_sampling_labels_model_fit_or_admission']is False,'Input-only actual scope')
 mb=complete['manifest'];need(Path(mb['path'])==OUT/'ANNOTATOR_INPUT_MANIFEST.json','Exact generated manifest only');manifest=get(mb);need(manifest['selection']==ready['selection']and manifest['no_oldweak_GT_SAM_or_modeloutputs']is True,'No observations or score fields')
 fs=plan['frames'];mr=manifest['frames'];ss=manifest['slots48'];need(len({f['component_id']for f in fs})==6,'Six fixed distinct graphcomponents');need(len(mr)==6 and len(ss)==48,'Full original retained rows')
 rows={r['row_id']:r for r in ss};need(len(rows)==48,'No duplicate native')
 pngs={};checks=[]
 for f,m in zip(fs,mr):
  need(m['frame_key']==f['frame_key']and m['original_frame_index']==f['original_frame_index']and m['component_id']==f['component_id']and m['original_image']==f['image'],'Frozen source/role metadata equality')
  full=image(m['fulloriginal'],'RGB');need(full.shape==tuple(f['image']['raw_hw'])+(3,)and ah(full)==f['image']['decoded_RGB_sha256'],'Generated fullPNG exact original decoded SHA')
  overlay=image(m['overlay'],'RGB');card=image(m['all8contact'],'RGB');need(overlay.shape==full.shape and card.shape==(1020,1600,3),'Display dimensions')
  for b in (m['fulloriginal'],m['overlay'],m['all8contact']):pngs[b['path']]=b['sha256']
  for r in f['native8']:
   a=rows[r['row_id']];k=r['native_slot']
   for key in ('native_slot','row_id','query_xy','L','original_query_available'):need(a[key]==r[key],'Every selected original qL/native/U exact')
   need(a['frame_key']==f['frame_key']and a['original_frame_index']==f['original_frame_index']and a['crop_constructed']==r['original_flags']['feature_constructed'],'Original frame/constructed flags')
   rgb=image(a['clean96'],'RGB');valid=image(a['valid96'],'L');need(rgb.shape==(96,96,3)and ah(rgb)==r['RGB96_sha256'],'Generated clean96 exact original decoded SHA')
   need(valid.shape==(96,96)and set(np.unique(valid))<={0,255}and ah(valid!=0)==r['valid96_sha256'],'Generated valid96 exact original bool SHA')
   y=86+450*(k//4)+78;x=400*(k%4)+300;need(np.array_equal(card[y:y+96,x:x+96],rgb),'Contact clean96 native panel byteidentity')
   q=r['query_xy'];expected=None
   if q is not None and len(q)==2 and all(z is not None and np.isfinite(z)for z in q)and 0<=q[0]<full.shape[1]and 0<=q[1]<full.shape[0]:expected=[float(q[0]),float(q[1]),bool(r['original_query_available'])]
   need(a['q_marker_on_fullframe']==expected and a['oldweaklabels_GT_or_predictions_shown']is False and a['original_raw_input_hash_verified']is True,'Display marker kept original/no imputation')
   for b in (a['clean96'],a['valid96']):pngs[b['path']]=b['sha256']
  checks.append(dict(frame_key=f['frame_key'],native_slots8=True,fullRGB_decoded_SHA_exact=True,all8_RGB96_valid_SHA_and_contact_ROIs_exact=True))
 need(len(pngs)==114 and sum(not r['original_query_available']for r in ss)==8,'All114 generatedPNG and48slots/U8')
 report=dict(completed=True,PASS=True,source_TRAIN_only=True,code=bind(Path(__file__)),preview_complete=bind(cp),ready=bind(H/'FINAL_READY_V2_LAYOUT.json'),selection=ready['selection'],manifest=mb,frames6=True,distinct_graphcomponents6=True,native_slots48=True,original_inputU8=True,generated_PNGs_verified=114,generated_fullRGB_and_RAW96_valid_decodedhashes_match_bound_originalmetadata=True,contact_clean96_native_ROI_byteparity=True,all_originalq_L_native_U_unchanged=True,frames=checks,original12_sourcefiles_reopened_or_decoded=0,labels_GT_models_weights_scores_read=0,actual_toolviews=0,observations_performed=False,training_or_label_admission=False,limits='Generated PNG+saved scalar identity only. No independent semantic native/instance/location labels or physicalgold certified.')
 write('REVIEW.json',report);print(json.dumps({'completed':True,'PASS':True,'review':bind(O/'REVIEW.json'),'frames':6,'slots':48,'inputU':8,'generated_PNGs':114}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--complete',type=Path,required=True);q=a.parse_args();main(q.complete)
