"""Saved geometry/buffers only; no renderer/model imports or new images."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;OUT=V/'actual_render_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False)as f:return {k:np.array(f[k],copy=True)for k in f.files}
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def first_hits(eye,points,xyz,owners):
 """Independent vectorized F64 ray/triangle intersection, same frozen boundaries."""
 d=points-eye;ref=np.linalg.norm(d,axis=1);d=d/ref[:,None]
 a=xyz[:,0];u=xyz[:,1]-a;v=xyz[:,2]-a
 p=np.cross(d[:,None,:],v[None]);det=np.sum(u[None]*p,axis=2)
 good=np.abs(det)>1e-14;inv=np.divide(1.,det,out=np.zeros_like(det),where=good)
 tvec=eye-a;bu=np.sum(tvec[None]*p,axis=2)*inv;qq=np.cross(tvec,u)
 bv=(d@qq.T)*inv;t=np.sum(v*qq,axis=1)[None]*inv
 good&=(bu>=-1e-12)&(bv>=-1e-12)&(bu+bv<=1+1e-12)&(t>1e-10)
 dist=np.min(np.where(good,t,np.inf),axis=1)
 close=good&(np.abs(t-dist[:,None])<=64*np.finfo(float).eps*np.maximum(1.,np.abs(dist[:,None])))
 owner=np.full(len(points),-1,np.int32);face=owner.copy();uncertain=np.ones(len(points),bool)
 for i in range(len(points)):
  ids=np.flatnonzero(close[i]);os=np.unique(owners[ids])
  if len(os)==1:owner[i]=os[0];face[i]=ids.min();uncertain[i]=False
 return owner,face,dist,ref,uncertain
def classify(owner,dist,ref,U,pid):
 diff=dist-ref;tol=1e-8*np.maximum(1.,ref)
 return np.where(U,'U',np.where((np.abs(diff)<=tol)&(owner==pid),'D',np.where(diff<-tol,np.where(owner==999,'E','S'),'U')))
def region(data,eye,K,R,t,foreign=None):
 xyz=data['vertices'][data['triangles']];own=data['part_owner'];points=data['native_witness_points'].reshape(-1,3)
 base=first_hits(eye,points,xyz,own)
 if foreign is not None:xyz=np.concatenate([xyz,foreign[data['foreign_triangles']]]);own=np.r_[own,np.full(12,999)]
 hit=first_hits(eye,points,xyz,own)
 pid=np.repeat(data['native_primary_part_id'],37);bs=classify(base[0],base[2],base[3],base[4],pid).reshape(8,37);cs=classify(hit[0],hit[2],hit[3],hit[4],pid).reshape(8,37)
 pc=points@R.T+t;xy=(pc[:,:2]/pc[:,2,None])*np.diag(K)[:2]+K[:2,2];xy=xy.reshape(8,37,2)
 states=[];active_all=[]
 signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
 for i in range(8):
  front=signs[i]*(eye-data['native_anchor3D'][i])>1e-12;active=[0]
  for axis in range(3):
   if front[[a for a in range(3)if a!=axis]].any():active+=list(range(1+3*axis,4+3*axis))
   if front[axis]:active+=list(range(10+9*axis,19+9*axis))
  active=np.array(sorted(set(active)));active_all.append(active)
  before='S'if np.all(bs[i]=='S')else('D'if front.sum()>=2 and np.all(bs[i,active]=='D')else'U')
  inframe=((xy[i]>=0)&(xy[i]<=255)).all()
  states.append('U'if not inframe else('S'if before=='S'else('U'if before!='D'else('E'if np.all(cs[i,active]=='E')else('D'if np.all(cs[i,active]=='D')else'U')))))
 return states,bs,cs,xy,hit,active_all
def main():
 complete=read(OUT/'RENDER_COMPLETE.json');assert complete['completed']and complete['PASS']
 for b in complete['outputs']:check(b)
 for k in ('entry','code','protocol','policy'):check(complete[k])
 protocol=read(complete['protocol']['path']);source=read(check(protocol['geometry_result']));design=read(check(protocol['geometry_design']))
 scenes=read(OUT/'SCENES6_AND_WITNESS_DIAGNOSTICS.json')['scenes'];assert len(scenes)==6
 eye=np.array(design['camera_eye']);report=[];cm=np.zeros((4,4),int);cm0=cm.copy();allstates=[];raw_by_key={}
 for item in source['results']:
  data=load(check(item['mesh']));style=item['style'];layer=load(OUT/(style+'_TARGET_LAYERS.npz'));d=layer['part_depth'];f=layer['global_triangle_id'];ldigest=digest(d,f)
  K,R,t=(data[k]for k in ('camera_K','camera_R','camera_t'));F=len(data['triangles']);N=len(item['parts'])
  assert d.shape==(N,256,256)and f.shape==d.shape and d.dtype==np.float64 and f.dtype==np.int32
  finite=np.isfinite(d);best=np.min(np.where(finite,d,np.inf),axis=0);valid=np.isfinite(best);tol=64*np.finfo(float).eps*np.maximum(1.,np.abs(best))
  eligible=finite&(np.abs(d-best[None])<=tol[None]);counts=eligible.sum(0);selected=eligible.argmax(0);yy,xx=np.indices((256,256))
  expected_owner=np.where(valid,selected,-1);expected_face=np.where(valid,f[selected,yy,xx],-1);expected_depth=np.where(valid,best,np.nan)
  before=load(OUT/(style+'_BEFORE_BUFFERS.npz'))
  assert np.array_equal(before['part_id'],expected_owner)and np.array_equal(before['global_triangle_id'],expected_face)and np.array_equal(before['camera_depth'],expected_depth,equal_nan=True)
  assert np.array_equal(before['ambiguity'],counts>1)and np.array_equal(before['tie_count'],counts)
  for variant in ('BEFORE','COVER','SHAM'):
   row=next(r for r in scenes if r['style']==style and r['variant']==variant);b=load(check(row['buffers']));raw_by_key[(style,variant)]=b
   assert row['immutable_target_layer_digest_before']==row['immutable_target_layer_digest_after']==ldigest
   for k in ('camera_K','camera_R','camera_t'):assert np.array_equal(b[k],data[k])
   rgb,z,owner,face,amb,tie=(b[k]for k in ('RGB_uint8','camera_depth','part_id','global_triangle_id','ambiguity','tie_count'))
   assert rgb.shape==(256,256,3)and rgb.dtype==np.uint8 and z.shape==owner.shape==face.shape==amb.shape==tie.shape==(256,256)
   foreground=np.isfinite(z);assert np.array_equal(foreground,owner>=0)and np.array_equal(foreground,face>=0)and np.all(z[foreground]>.001)
   assert np.all(owner[~foreground]==-1)and np.all(face[~foreground]==-1)and np.all(tie[~foreground]==0)and not amb[~foreground].any()
   assert np.array_equal(amb,tie>1)and np.all(tie[foreground]>=1)and np.all(rgb[~foreground]==[225,226,228])
   foreign=None if variant=='BEFORE'else data['foreign_'+variant.lower()+'_vertices'];verts=data['vertices'];tris=data['triangles'];owns=data['part_owner']
   if foreign is not None:verts=np.concatenate([verts,foreign]);tris=np.concatenate([tris,data['foreign_triangles']+len(data['vertices'])]);owns=np.r_[owns,np.full(12,999)]
   assert np.all(face[foreground]<len(tris))and np.array_equal(owns[face[foreground]],owner[foreground])
   # Every saved occupied pixel lies on the stated physical triangle; no re-raster.
   y,x=np.nonzero(foreground);zz=z[foreground];cam=np.stack([(x-K[0,2])*zz/K[0,0],(y-K[1,2])*zz/K[1,1],zz],1);world=(cam-t)@R;tri=verts[tris[face[foreground]]]
   a=tri[:,0];u=tri[:,1]-a;v=tri[:,2]-a;w=world-a;n=np.cross(u,v);plane=np.abs(np.sum(n*w,axis=1))/np.linalg.norm(n,axis=1)
   dot=lambda a,b:np.sum(a*b,axis=1)
   aa,bb,cc,dd,ee=dot(u,u),dot(u,v),dot(v,v),dot(w,u),dot(w,v);den=aa*cc-bb*bb;beta=(cc*dd-bb*ee)/den;gamma=(aa*ee-bb*dd)/den
   assert plane.max()<1e-9 and beta.min()>-1e-9 and gamma.min()>-1e-9 and (beta+gamma).max()<1+1e-9
   kept=foreground&(owner!=999);assert np.array_equal(owner[kept],before['part_id'][kept])and np.array_equal(face[kept],before['global_triangle_id'][kept])and np.array_equal(z[kept],before['camera_depth'][kept])
   assert np.array_equal(rgb[kept],before['RGB_uint8'][kept])and not(np.isfinite(before['camera_depth'])&~foreground).any()
   basez=before['camera_depth'];foreignmask=owner==999;both=foreignmask&np.isfinite(basez);assert np.all(z[both]<=basez[both]+64*np.finfo(float).eps*np.maximum(1,np.abs(basez[both])))
   states,bs,cs,xy,hits,active=region(data,eye,K,R,t,foreign)
   assert states==item['states'][variant]==b['continuous_region_states'].tolist()==[r['state']for r in row['continuous_region_records']]
   for i,rec in enumerate(row['continuous_region_records']):
    assert bs[i].tolist()==rec['target_witness_states']and cs[i].tolist()==rec['combined_witness_states']and active[i].tolist()==rec['active_witness_indices']
   assert np.allclose(xy,b['witness_xy_continuous'],rtol=0,atol=1e-12)
   near=np.floor(xy+.5).astype(np.int32);inside=((near>=0)&(near<256)).all(-1);po=np.full((8,37),-1,np.int32);pd=np.full((8,37),np.nan);pf=po.copy();pa=np.ones((8,37),bool)
   for i,j in np.ndindex(8,37):
    if inside[i,j]:x0,y0=near[i,j];po[i,j]=owner[y0,x0];pd[i,j]=z[y0,x0];pf[i,j]=face[y0,x0];pa[i,j]=amb[y0,x0]
   assert np.array_equal(near,b['witness_nearest_xy'])and np.array_equal(inside,b['witness_nearest_in_image'])and np.array_equal(po,b['witness_nearest_owner'])and np.array_equal(pf,b['witness_nearest_triangle_id'])and np.array_equal(pd,b['witness_nearest_depth'],equal_nan=True)and np.array_equal(pa,b['witness_nearest_ambiguity'])
   ro,rf,dist,ref,U=hits;ro=ro.reshape(8,37);rf=rf.reshape(8,37);U=U.reshape(8,37)
   assert np.array_equal(ro,b['witness_continuous_owner'])and np.array_equal(rf,b['witness_continuous_triangle_id'])and np.array_equal(U,b['witness_continuous_U'])
   pp=data['native_witness_points'].reshape(-1,3);direction=pp-eye;direction/=np.linalg.norm(direction,axis=1)[:,None];location=eye+direction*dist[:,None];rd=(location@R.T+t)[:,2].reshape(8,37);rd[U]=np.nan
   assert np.allclose(rd,b['witness_continuous_depth'],rtol=0,atol=1e-12,equal_nan=True)
   aliases=[]
   for i,s in enumerate(row['nearest_pixel_diagnostic']):
    comparable=inside[i]&~pa[i]&~U[i];finitecmp=comparable&np.isfinite(pd[i])&np.isfinite(rd[i]);ddiff=np.abs(pd[i,finitecmp]-rd[i,finitecmp]);mx=float(ddiff.max())if len(ddiff)else None
    own={'all37_nearest_OOB':int((~inside[i]).sum()),'nearest_ambiguous':int((inside[i]&pa[i]).sum()),'nearest_background':int((inside[i]&(po[i]<0)).sum()),'nearest_nonfinite_depth':int((inside[i]&~np.isfinite(pd[i])).sum()),'continuous_ray_U':int(U[i].sum()),'certain_owner_comparable':int(comparable.sum()),'certain_owner_disagreements':int((comparable&(po[i]!=ro[i])).sum()),'finite_depth_comparable':int(finitecmp.sum()),'active_unique_nearest_pixels':len(set(map(tuple,near[i,active[i]][inside[i,active[i]]].tolist())))}
    assert all(s[k]==v for k,v in own.items())
    saved=s['max_abs_depth_difference_between_different_rays'];assert(saved is None and mx is None)or(saved is not None and mx is not None and abs(saved-mx)<1e-12)
    aliases.append({'native_index':i,'state':states[i],**own,'max_abs_depth_difference_between_different_rays':mx})
   report.append({'style':style,'variant':variant,'buffers':row['buffers'],'occupied_pixels':int(foreground.sum()),'foreign_pixels':int(foreignmask.sum()),'ambiguous_pixels':int(amb.sum()),'same_target_RGB_exact_pixels':int(kept.sum()),'max_point_to_assigned_triangle_plane_m':float(plane.max()),'barycentric_min':float(min(beta.min(),gamma.min(),(1-beta-gamma).min())),'continuous_states':states,'witness_aliasing':aliases})
   allstates+=states
   for i,state in enumerate(states):
    for dy in (-3,0,3):
     for dx in (-3,0,3):
      q=xy[i,0]+[dx,dy];qp=np.floor(q+.5).astype(int);op=owner[qp[1],qp[0]];cat=3 if amb[qp[1],qp[0]]else(2 if op<0 else(1 if op==999 else 0));rr='DESU'.index(state);cm[rr,cat]+=1
      if dx==dy==0:cm0[rr,cat]+=1
 rootdiag=read(V/'ROOT_SOURCE_NEAREST_PIXEL_SEMANTIC_DIAGNOSTIC.json')
 assert cm.tolist()==rootdiag['summary']['ALL']['CM_rows_D_E_S_U_cols_TARGET_FOREIGN_BACKGROUND_U'] and cm0.tolist()==rootdiag['summary']['ERROR_ZERO']['CM_rows_D_E_S_U_cols_TARGET_FOREIGN_BACKGROUND_U']
 result={'completed':True,'PASS':True,'blockers':[],'scope':'Independent saved raw6 numeric audit; no renderer re-execution, new image creation/view, model/feature/parameter/current/V112/PnP access.','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':bind(__file__),'render_complete':bind(OUT/'RENDER_COMPLETE.json'),'protocol':complete['protocol'],'geometry_result':protocol['geometry_result'],'geometry_design':protocol['geometry_design'],'root_pixel_diagnostic':bind(V/'ROOT_SOURCE_NEAREST_PIXEL_SEMANTIC_DIAGNOSTIC.json'),'all_output_file_SHA_replay':len(complete['outputs']),'scenes':6,'teacher_region_counts':{s:allstates.count(s)for s in 'DSEU'},'independent_checks':['All target-layer BEFORE nearest owner/face/depth/tie summaries reconstructed from saved per-part layers.','All six occupied pixels deproject onto assigned physical triangle within predeclared numerical audit tolerance 1e-9m/barycentric; global triangle-owner namespace exact.','Background/finite/near/tie mask invariants exact; COVER/SHAM retain target coverage or replace it by nearer actual foreign geometry.','Every surviving target surface owner/face/depth and RGB matches BEFORE bit-exact; producer layer digests independently replayed.','All1776 continuous witness rays independently replayed, including all region states and target/combined D/S/E/U vectors; saved continuous owners/triangleIDs/depth match.','All1776 rounded witness samples and48 summary records independently re-indexed; owner mismatches/background/NaN-depth remain explicit.','Root432 nearest-query pixel diagnostic and48 zero-offset subset independently reproduced.'],'pixel_semantic_diagnostic':{'rows':'D,E,S,U','columns':'TARGET,FOREIGN,BACKGROUND,ambiguityU','all432_CM':cm.tolist(),'zero_offset48_CM':cm0.tolist()},'scene_results':report,'limitations':['Assigned-triangle consistency and saved layer resolution do not prove exhaustive target-layer raster coverage; no new full-scene raster was run.','Independent ray recomputation uses the frozen boundary/tie/classification conventions, not a newly validated real-image visibility definition.','Witness projections span only a few pixels; rounded samples are different rays from continuous region witnesses and their owner/depth disagreements are observations, not render failure or teacher relabeling.','Numerical source consistency establishes neither RGB cue recognizability/photorealism nor real transfer/legacy native identity/training admission.'],'new_teacher_or_input_labels_created':False,'training_admission_created':False}
 with (H/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json'),'region_counts':result['teacher_region_counts'],'pixel_CM':cm.tolist(),'scene_pixel_counts':[(r['style'],r['variant'],r['occupied_pixels'],r['foreign_pixels'])for r in report]}))
if __name__=='__main__':main()
