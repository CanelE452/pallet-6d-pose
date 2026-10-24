"""Exactly five GT-only rows; independent math, no helper/model/RGB import."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;C=R.parent/'source_contract'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def save(name,obj):
 p=R/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');return p
def normals(q,faces):
 n=[];cent=[]
 for ix in faces:
  v=q[ix];c=v.mean(0);a=np.cross(v,np.roll(v,-1,axis=0)).sum(0)
  a*=1 if a@(c-q.mean(0))>=0 else -1;n.append(a/np.linalg.norm(a));cent.append(c)
 return np.array(n),np.array(cent)
pack=C/'BUILD_V2/TYPED_SOURCE_PACK_GT_ONLY.npz';seal=C/'BUILD_V2/TYPED_SOURCE_PACK_SEAL.json';s=json.loads(seal.read_text())
z=np.load(pack,allow_pickle=False);ids=z['sample_id'];src=z['source']
rank=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('V74-independent-five|'+str(ids[i])).encode()).hexdigest())
sel=[next(i for i in rank if src[i]==v) for v in sorted(np.unique(src))];sel += [i for i in rank if i not in sel][:2];assert len(sel)==5
lock=save('INDEPENDENT_TYPED_SOURCE_GT_FIVE_ROW_LOCK.json',{'pack':bind(pack),'seal':bind(seal),'code':bind(Path(__file__)),'selection':'First SHA256(V74-independent-five|sample_id) per source, then two remaining globally lowest; no RGB/state/score selection.','rows':[{'index':i,'id':str(ids[i]),'source':str(src[i])} for i in sel],'model_RGB_current':0})
keys=['cuboid_world_dynamic8','perm_dynamic_to_writer_fixed8','reference_projected8','K','camera_world','source_centered_yup_to_camera','world_to_camera_composed','cuboid_centered_yup_dynamic8','reference_camera_depth8','external_9ray_fraction9','convex_self_visible_proxy8','reference_inframe8','reference_proxy_state8','certified_semantic_state8','target_pixel_support8','target_pixel_support_available8']
d={k:z[k][sel] for k in keys};z.close()
with np.load(C/'TYPED_SOURCE_INPUT_COORDS_ONLY.npz',allow_pickle=False) as z:hw=z['raw_hw'][sel]
binds=[s['GT_pack'],s['input_coords_unchanged'],s['row_map']]+s['bindings'];bad=[b['path'] for b in binds if sha(b['path'])!=b['sha256']];assert not bad
sides=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]];faces=sides+[[0,1,5,4],[3,2,6,7]];A=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
rows=[]
for j,i in enumerate(sel):
 q=d['cuboid_world_dynamic8'][j];p=d['perm_dynamic_to_writer_fixed8'][j];r=d['reference_projected8'][j];T=d['source_centered_yup_to_camera'][j];K=d['K'][j];cam=d['camera_world'][j]
 centered=(q-q.mean(0))@A.T;cp=np.c_[centered,np.ones(8)]@T.T;proj=cp[:,:3]@K.T;proj=proj[:,:2]/proj[:,2,None]
 W=np.eye(4);W[:3,:3]=T[:3,:3]@A;W[:3,3]=T[:3,3]-T[:3,:3]@A@q.mean(0);cp2=np.c_[q,np.ones(8)]@W.T
 fixed=np.empty_like(q);fixed[p]=q;fixedr=np.empty_like(r);fixedr[p]=r
 ns,_=normals(fixed,sides);ray=cam-fixed.mean(0);cos=ns@ray/np.linalg.norm(ray);f=int(cos.argmax());face=sides[f]
 top=sorted(sorted(face,key=lambda k:fixed[k,2],reverse=True)[:2],key=lambda k:fixedr[k,0]);bot=[k for k in face if k not in top]
 b=[min(bot,key=lambda k:np.linalg.norm(fixed[k,:2]-fixed[v,:2])) for v in top];front=[top[0],top[1],b[1],b[0]]
 rear=[min(sides[f^1],key=lambda k:np.linalg.norm(fixed[k]-fixed[v])) for v in front];perm=np.array(front+rear)
 ns,fc=normals(q,faces);facing=np.einsum('ij,ij->i',ns,cam-fc)>0;selfv=np.zeros(8,bool)
 for ff,visible in zip(faces,facing):
  if visible:selfv[ff]=True
 frac=d['external_9ray_fraction9'][j,:8];h,w=hw[j];ev=np.isfinite(r).all(1)&np.isfinite(cp[:,2])&np.isfinite(frac)&(frac>=0)&(frac<=1)
 inside=ev&(cp[:,2]>0)&(r[:,0]>=0)&(r[:,0]<w)&(r[:,1]>=0)&(r[:,1]<h);state=np.full(8,4,np.int8)
 for k in range(8):
  if ev[k]:state[k]=3 if not inside[k] else 2 if not selfv[k] else 1 if frac[k]>=.5 else 0
 item={'index':i,'id':str(ids[i]),'source':str(src[i]),'projection_max_error_px':float(np.linalg.norm(proj-r,axis=1).max()),'centered_Yup_max_error':float(abs(centered-d['cuboid_centered_yup_dynamic8'][j]).max()),'composed_pose_max_error':float(abs(W-d['world_to_camera_composed'][j]).max()),'two_pose_routes_max_error':float(abs(cp-cp2).max()),'camera_depth_max_error':float(abs(cp[:,2]-d['reference_camera_depth8'][j]).max()),'camera_recovery_error_m':float(np.linalg.norm(np.linalg.inv(W)[:3,3]-cam)),
 'positive_depth':bool((cp[:,2]>0).all()),'permutation_exact':bool(np.array_equal(perm,p)),'convex_self_exact':bool(np.array_equal(selfv,d['convex_self_visible_proxy8'][j])),'inframe_exact':bool(np.array_equal(inside,d['reference_inframe8'][j])),'reference_proxy_exact':bool(np.array_equal(state,d['reference_proxy_state8'][j])),'semantic_unknown':bool((d['certified_semantic_state8'][j]==-1).all()),'pixel_support_unknown':bool((d['target_pixel_support8'][j]==-1).all() and not d['target_pixel_support_available8'][j].any()),'proxy8':state.tolist()}
 assert item['projection_max_error_px']<.01 and item['camera_recovery_error_m']<2e-6
 assert all(item[k] for k in ['positive_depth','permutation_exact','convex_self_exact','inframe_exact','reference_proxy_exact','semantic_unknown','pixel_support_unknown'])
 rows.append(item)
out=save('INDEPENDENT_TYPED_SOURCE_GT_FIVE_ROW_AUDIT.json',{'schema':'V74_TYPED_GT_INDEPENDENT_FIVE_ROW_AUDIT','PASS':True,'selection_lock':bind(lock),'pack':bind(pack),'seal':bind(seal),'metadata_bindings_checked':len(binds),'metadata_binding_mismatches':bad,'rows':rows,
 'code_review':'V2 correctly treats source T as acting on A(world-cuboid_center) and composes a separate world-to-camera. Stored GT permutation is not applied to deployable predicted slots. Semantic/pixel truth stays UNKNOWN. V1 incorrect world-direct interpretation is separately rejected and preserved.',
 'independent_method':'No helper import. Row-vector homogeneous projection and separate pose composition; Newell polygon normals for source permutation and convex self proxy; scalar F>S>E>D reference proxy.',
 'scope':'Exactly five SHA-fixed GT rows across three sources, not an all60000 replay. Prior all60000 and8024 results are parent evidence only.',
 'RGB_P3_P4_weights_models_current_forward_fit_PnP':0,'old_files_modified':False,'limits':['Numerical source GT consistency only; not human semantic D/E/S, exact rendered-mesh surface visibility or source model performance.','No new pose fitting, model input or query/reference changes.']})
print(json.dumps({'audit':bind(out),'rows':rows}))
