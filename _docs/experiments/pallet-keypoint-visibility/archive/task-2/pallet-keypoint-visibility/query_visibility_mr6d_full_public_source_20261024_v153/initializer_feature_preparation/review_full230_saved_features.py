"""Independent completed DINO-cache input/pixel/component audit; no model imports."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;O=V/'full230_native_DINO_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def clean(a):
 if isinstance(a,np.ndarray):return clean(a.tolist())
 if isinstance(a,np.generic):return clean(a.item())
 if isinstance(a,float):return a if np.isfinite(a)else None
 if isinstance(a,list):return[clean(x)for x in a]
 return a

def own_crop(rgb,q,L):
 h,w=rgb.shape[:2];u=(np.arange(96,dtype=np.float64)+.5)/96-.5
 xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);dx=xx-ix;dy=yy-iy;acc=np.zeros((96,96,3),np.float64)
 for ax,ay,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  x,y=ix+ax,iy+ay;valid=(x>=0)&(x<w)&(y>=0)&(y<h);acc[valid]+=rgb[y[valid],x[valid]].astype(np.float64)*weight[valid,None]
 valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5);a=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);a[~valid]=0
 return a,valid

def own_target(rgb,box):
 if box is None:return np.zeros((224,224,3),np.float32),np.zeros((224,224),bool)
 h,w=rgb.shape[:2];x0,y0,x1,y1=box
 xs=x0+(np.arange(224)+.5)*(x1-x0)/224;ys=y0+(np.arange(224)+.5)*(y1-y0)/224;x,y=np.meshgrid(xs,ys)
 ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy;out=np.zeros((224,224,3),np.float64);valid=np.ones((224,224),bool)
 for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
  out+=rgb[np.clip(yy,0,h-1),np.clip(xx,0,w-1)].astype(np.float64)*(weight*inside)[...,None];valid&=inside|(weight==0)
 return out.astype(np.float32),valid

def main(expected):
 sfile=O/'FEATURE_COMPLETE.json';assert sha(sfile)==expected;s=read(sfile);assert s['completed']and s['PASS']and s['rows']==1840 and s['frames']==230 and s['features_shape']==[1840,1928]
 for k in ('code','policy','protocol','entry','inputs','input_seal','source_initializer','source_intake','features','inputfacts','crop_manifest','fixed10_parity'):ck(s[k])
 assert s['primary_encoder_calls']==230 and s['fixed10_replay_calls']==10 and s['total_encoder_calls']==240 and s['views_per_call']==17
 assert s['encoder_state_unchanged']and s['frozen_encoder_state_before']==s['frozen_encoder_state_after']and s['weight_and_inputs_unchanged']and s['prior_MR10_canonical_factual96_and_new_features_replay_bitexact']
 assert s['GT_pose_K_mask_depth_label_reference_reads']==s['classifier_R0_P3_fit_private_PnP_calls']==0
 p=read(ck(s['protocol']));inp=read(ck(s['inputs']));frames=inp['frames'];assert len(frames)==230 and inp['initializer']==s['source_initializer']and inp['intake']==s['source_intake']
 e=read(ck(s['entry']));assert e['approved_actual_DINO_feature_extraction']and e['source_only']
 for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','PnP_allowed','semantic_pose_depth_reads_allowed','automatic_retry'):assert e[k]is False
 for k in ('code','policy','protocol','inputs'):assert e[k]==s[k]
 with np.load(ck(s['features']),allow_pickle=False)as f:Z={k:f[k].copy()for k in f.files}
 with np.load(ck(inp['query_npz']),allow_pickle=False)as f:Q={k:f[k].copy()for k in f.files}
 facts=lines(ck(s['inputfacts']));assert len(facts)==1840
 for a,b in (('global_index','global_index'),('frame_index','frame_index'),('native_slot','native_slot'),('query_id','query_id'),('frame_id','frame_id'),('split','split'),('query_xy','query_xy'),('L','L'),('no_detection','no_detection'),('native_continuous_valid','continuous_valid'),('nearest_valid','nearest_valid')):assert exact(Z[a],Q[b]),a
 assert exact(Z['query_available'],Q['continuous_valid'])and np.array_equal(Z['occurrence_unit'],np.arange(1840))
 assert Z['source_id'].tolist()==np.repeat([r['source_id']for r in frames],8).tolist()
 crops=read(ck(s['crop_manifest']));assert crops['completed']and crops['frames']==230 and crops['rows']==1840 and len(crops['sheets'])==230
 X=Z['features'];assert X.dtype==np.float32 and X.shape==(1840,1928)and np.isfinite(X).all()
 fracs=[];c4s=[];ctxs=[];constructs=[];insides=[];boxes=[];proof=[]
 for i,(r,cp)in enumerate(zip(frames,crops['sheets'])):
  assert r['frame_index']==cp['frame_index']==i and r['source_id']==cp['source_id']
  rgb=np.load(ck(r['canonical_RGB_NPY']),allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3]and ash(rgb)==r['decoded_RGB_sha256']
  with np.load(ck(cp['crop_packet']),allow_pickle=False)as f:
   assert set(f.files)=={'rgb96','valid96','global_index','crop_constructed'};C,V,M,ix=f['rgb96'].copy(),f['valid96'].copy(),f['crop_constructed'].copy(),f['global_index'].copy()
  assert C.dtype==np.uint8 and C.shape==(8,96,96,3)and V.dtype==bool and V.shape==(8,96,96)and np.array_equal(ix,np.arange(i*8,i*8+8))
  q=Q['query_xy'][i*8:i*8+8];L=Q['L'][i*8:i*8+8];good=np.isfinite(q).all(1)&~(q==[-1,-1]).all(1);construct=good&np.isfinite(L)&(L>0)
  assert np.array_equal(M,construct)
  box=None
  if good.sum()>=2:
   lo=q[good].min(0);hi=q[good].max(0)
   if(hi>lo).all():box=np.r_[lo,hi]
  target,vt=own_target(rgb,box);targetsha=ash(target);validsha=ash(vt)
  for k in range(8):
   if construct[k]:cr,v=own_crop(rgb,q[k],L[k])
   else:cr=np.zeros((96,96,3),np.uint8);v=np.zeros((96,96),bool)
   assert exact(C[k],cr)and exact(V[k],v),(i,k,'crop')
   frac=np.array([np.float32(vt.mean()),np.float32(v.mean()),np.float32(v[32:64,32:64].mean())],np.float32);c4=bool(v[47:49,47:49].all())
   inside=False
   if box is not None and vt.any()and good[k]:
    u=(q[k]-box[:2])/(box[2:]-box[:2]);inside=bool((u>=0).all()and(u<=1).all())
   f=facts[i*8+k];assert f['global_index']==i*8+k and f['frame_index']==i and f['native_slot']==k and f['frame_id']==r['id']and f['query_id']==Q['query_id'][i*8+k]and f['source_id']==r['source_id']and f['split']==r['split']
   assert f['query_xy']==clean(q[k])and f['L']==clean(L[k])and f['context_box_xyxy']==clean(box)
   assert f['feature_constructed']==bool(construct[k])and f['native_continuous_valid']==bool(Q['continuous_valid'][i*8+k])and f['nearest_valid']==bool(Q['nearest_valid'][i*8+k])
   assert f['Q_inside']==inside and f['local_central4_available']==c4 and f['context_available']==bool(vt.any())
   assert f['RGB96_HWC_sha256']==ash(cr)and f['RGB96_CHW_sha256']==ash(cr.transpose(2,0,1).copy())and f['valid96_sha256']==ash(v)and f['target224_F32_HWC_sha256']==targetsha and f['target_valid_sha256']==validsha
   assert np.array_equal(np.array(f['fractions'],np.float32),frac)
   fracs.append(frac);c4s.append(c4);ctxs.append(bool(vt.any()));constructs.append(bool(construct[k]));insides.append(inside);boxes.append(box if box is not None else[np.nan]*4)
  proof.append(dict(frame_index=i,source_id=r['source_id'],all8_crop_pixels_hashes_q_L_context_and_flags_exact=True))
 frac=np.asarray(fracs,np.float32);c4=np.asarray(c4s,bool);ctx=np.asarray(ctxs,bool);construct=np.asarray(constructs,bool);inside=np.asarray(insides,bool)
 for key,want in (('support_fractions',frac),('local_central4_available',c4),('context_available',ctx),('feature_constructed',construct),('unit_Q_inside',inside)):assert exact(Z[key],want),key
 assert np.array_equal(Z['context_box_xyxy'],np.asarray(boxes,np.float64),equal_nan=True)
 assert exact(X[:,1924:1927].copy(),frac)and np.array_equal(X[:,1927],inside.astype(np.float32))and not X[~construct].any()
 gates=[ctx&construct,inside&construct,(frac[:,1]>0)&construct,(frac[:,2]>0)&construct,c4&construct]
 norms=[]
 for j,g in enumerate(gates):
  a=X[:,j*384:(j+1)*384];assert not a[~g].any()
  err=float(np.abs(np.linalg.norm(a[g],axis=1)-1).max())if g.any()else 0.;assert err<2e-5;norms.append(err)
 T,Qb,F,Cb,Pb=[X[:,j*384:(j+1)*384]for j in range(5)]
 def cos(a,b):return (a*b).sum(1)/(np.maximum(np.linalg.norm(a,axis=1),1e-12)*np.maximum(np.linalg.norm(b,axis=1),1e-12))
 cosine=np.stack((cos(T,Cb),cos(T,F),cos(Qb,Cb),cos(Qb,Pb)),1);maxcos=float(np.abs(cosine-X[:,1920:1924]).max());assert maxcos<2e-6
 parity=read(ck(s['fixed10_parity']));assert parity['completed']and parity['PASS']and parity['frames']==10 and parity['rows']==80 and parity['replay_encoder_calls']==10 and parity['no_prior_MR10_DINO_cache_claim']
 for r in parity['rows_compared']:
  i=r['frame_index'];assert frames[i]['source_id']==r['source_id']and r['features_and_raw_tokens_bitexact']and r['prior_factual96_RGB_valid_bytes_exact']and ash(X[i*8:i*8+8])==r['all8_features1928_SHA256']
 assert sum(Z['split']=='SOURCE_DEVELOPMENT')==1480 and sum(Z['split']=='SOURCE_VALIDATION')==360
 assert int(construct.sum())==s['feature_constructed']==1816 and int(Z['query_available'].sum())==s['available']==1538 and int(Z['nearest_valid'].sum())==1535 and int(Z['no_detection'].sum())==24
 report=dict(completed=True,PASS=True,code=bind(__file__),feature_complete=bind(sfile),entry=s['entry'],protocol=s['protocol'],feature_packet=s['features'],inputfacts=s['inputfacts'],crop_manifest=s['crop_manifest'],rows=1840,frames=230,feature_shape=[1840,1928],continuous_available=1538,nearest_available=1535,feature_constructed=1816,no_detection_rows=24,source_development_rows=1480,source_validation_rows=360,all_ID_q_L_original_flags_exact=True,all1840_factual96_RGB_valid_pixels_and_SHA_exact=True,all_target224_and_component_flags_fractions_exact=True,all230_canonical_RGB_SHA_exact=True,feature_block_unit_norm_max_errors=norms,feature_cosine_max_difference=maxcos,recorded_fixed10_input_feature_rawtoken_repeat_receipt_and_feature_hashes_exact=True,frame_checks=proof,
 model_encoder_R0_checkpoint_GT_private_V112_calls_or_body_reads=0,limitations=['DINO latent values and rawtoken replay are covered by bound producer receipts; this independent audit verifies cached feature structure/normalization/scalar arithmetic and all pixel inputs, without rerunning the encoder.','Availability metadata preserves continuous1538 and nearest1535 separately. No semantic visibility labels or new abstention/filter policy is created.'])
 out=H/'FULL230_SAVED_FEATURE_REVIEW.json'
 with out.open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(out),rows=1840,continuous=1538,nearest=1535,constructed=1816)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--feature-seal-sha',required=True);x=a.parse_args();main(x.feature_seal_sha)
