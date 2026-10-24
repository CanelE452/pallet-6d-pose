"""Root-authorized cached LOCO160 triplet, no backbone/model/GT/current."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V74=P/'query_visibility_loco_native_context_20261023_v74/public_initializer/DECODED_RGB_V2_RUN'
V92=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92'
MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
OUT=V/'loco_before_input_pack'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def lines(p):return [json.loads(s)for s in p.read_text().splitlines()if s]
def mathmod():
 assert sha(MATH)==SHA;s=importlib.util.spec_from_file_location('v95_loco_math',MATH);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare():
 m=mathmod();rgb=np.arange(32*48*3,dtype=np.uint32).reshape(32,48,3).astype(np.uint8)
 q=np.array([48.1,16.]);c,v=m.crop(rgb,q,16.,96);assert c.any()and v.any() # finite outside centre preserved
 a=m.expected_affine([480,640]);x,_,vv=m.sample_p3(np.zeros((64,68,80),np.float16),[[320,240]],[64],[480,640],a['gain'],a['offset'],100)
 assert np.array_equal(x[:,64].astype(bool),vv)
 proof=H/'GENERATED_LOCO_BEFORE160.json';write(proof,dict(PASS=True,code=b(__file__),finite_outside_centre_crop_kept=True,native65_validity_exact=True,source_arrays_pixels_model_current_GT=0))
 paths=dict(frames=V92/'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl',rows=V92/'rgb96_review/ROWS_INPUT_ONLY.jsonl',rgb96=V92/'rgb96_review/INPUT_RGB96_ONLY.npz',p3=V74/'P3_INPUT_FEATURES_ONLY.npz',proposals=V74/'FROZEN_PROPOSALS.jsonl')
 write(H/'LOCO_BEFORE_PROTOCOL_BEFORE_ARRAYS.json',dict(completed=True,root_authorization='Parent explicitly approved compact LOCO160 before triplet using cached fullP3/V90 CPU sampling and every finite originalq canonical V74 crop. No backbone.',scope=b(V/'ROOT_SOURCE_SCOPE_BEFORE_ACTUAL_CONSTRUCTION.json'),code=b(__file__),math=b(MATH),generated=b(proof),bindings={k:b(p)for k,p in paths.items()},output=str(OUT),source_only=True,current_allowed=False,head_backbone_weights_GT_fit_PnP=0))

def run():
 p=json.loads((H/'LOCO_BEFORE_PROTOCOL_BEFORE_ARRAYS.json').read_text());assert p['code']==b(__file__)and p['math']==b(MATH)
 for d in p['bindings'].values():assert sha(d['path'])==d['sha256']
 frames=lines(Path(p['bindings']['frames']['path']));rows=lines(Path(p['bindings']['rows']['path']));props=lines(Path(p['bindings']['proposals']['path']));assert len(frames)==len(props)==20 and len(rows)==160
 m=mathmod()
 with np.load(p['bindings']['p3']['path'],allow_pickle=False)as z:full=z['p3']
 with np.load(p['bindings']['rgb96']['path'],allow_pickle=False)as z:oldrgb=z['rgb96'];oldv=z['crop_valid96'];oldq=z['query_xy'];oldL=z['L']
 assert full.shape==(20,64,80,80)and full.dtype==np.float16
 roi=np.zeros((160,65,9,9),np.float16);rgbs=np.zeros((160,3,96,96),np.uint8);vs=np.zeros((160,1,96,96),bool);qraw=np.full((160,2),np.nan);Ls=np.full(160,np.nan);facts=[];RGB_mismatches=[]
 for i,(fr,pr)in enumerate(zip(frames,props)):
  path=fr['decoded_RGB_input']['path'];assert sha(path)==fr['decoded_RGB_input']['sha256'];rgb=np.load(path,allow_pickle=False)
  assert rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==fr['RGB_pixels_sha256']
  a=m.expected_affine(fr['raw_hw']);assert pr['input_shape']==a['input_shape']and pr['canvas_shape']==a['canvas_shape']and pr['added_border']==100
  h,w=a['p3_actual_hw'];assert not full[i,:,h:,:].any()and not full[i,:,:,w:].any()
  if fr['selected_index']is not None:
   selected=int(np.argmax([c['score']for c in pr['candidates']]));assert selected==fr['selected_index']==pr['selected_index']
  for k in range(8):
   ix=i*8+k;r=rows[ix];assert r['input_id']==fr['input_id']and r['native_slot']==k and r['frame_index']==i
   if r['query_xy']is None:
    assert fr['selected_index']is None and not oldrgb[ix].any()and not oldv[ix].any()
   else:
    q=np.asarray(r['query_xy'],np.float64);L=float(r['L']);qraw[ix]=q;Ls[ix]=L
    assert np.array_equal(q,oldq[ix])and L==oldL[ix]and r['query_xy']==fr['query8_xy'][k]and L==fr['L']
    x,_,_=m.sample_p3(full[i,:,:h,:w],q[None],[L],fr['raw_hw'],a['gain'],a['offset'],100);roi[ix]=x[0]
    rgbs[ix],vs[ix]=m.crop(rgb,q,L,96)
    if not np.array_equal(rgbs[ix],oldrgb[ix])or not np.array_equal(vs[ix],oldv[ix]):RGB_mismatches.append(ix)
   facts.append(dict(slot_index=ix,input_id=r['input_id'],native_slot_logical_only=k,frame_index=i,split=r['split'],query_xy=r['query_xy'],L=r['L'],original_input_valid=r['query_center_valid'],image_sha256=r['image_sha256'],decoded_RGB_sha256=fr['RGB_pixels_sha256'],RGB96_changed_from_legacy=ix in RGB_mismatches))
 OUT.mkdir(exist_ok=False);fp=OUT/'BEFORE_INPUTS160.npz'
 np.savez_compressed(fp,p3_roi=roi,rgb96=rgbs,crop_valid96=vs,query_xy_raw=qraw,L_raw=Ls,query_present=np.isfinite(qraw).all(1),original_input_valid=np.array([r['original_input_valid']for r in facts]),query_id=np.array([f"{r['input_id']}:native{r['native_slot_logical_only']}"for r in facts]))
 lin=OUT/'BEFORE160_INPUT_LINEAGE.jsonl';lin.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in facts))
 write(OUT/'INPUT_PACK_QA.json',dict(completed=True,PASS=True,rows=160,finite_q=144,noquery=16,all_original_q_L_exact=True,cached_P3_only_new_backbone_calls=0,canonical_crop_checks=144,legacy_RGB96_different_row_indices=RGB_mismatches,legacy_artifacts_unchanged=True,finite_invalid_not_zeroed=True,code=b(__file__),protocol=b(H/'LOCO_BEFORE_PROTOCOL_BEFORE_ARRAYS.json'),tensor_file=b(fp),lineage=b(lin),source_only=True,current_model_GT_fit_PnP=0))
 write(OUT/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,inputs=b(fp),lineage=b(lin),QA=b(OUT/'INPUT_PACK_QA.json'),protocol=b(H/'LOCO_BEFORE_PROTOCOL_BEFORE_ARRAYS.json'),rows=160,head_backbone_fit_current_PnP=0))
 print(json.dumps(dict(inputs=b(fp),seal=b(OUT/'INPUT_PACK_SEAL.json'),RGB_corrections=RGB_mismatches)),flush=True)
if __name__=='__main__':prepare();run()
