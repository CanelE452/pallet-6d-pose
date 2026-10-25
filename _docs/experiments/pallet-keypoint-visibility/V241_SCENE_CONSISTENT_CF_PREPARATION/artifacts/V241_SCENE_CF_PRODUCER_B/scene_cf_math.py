"""Pure V241 native-scene CF construction. No file/model access or label admission.

Caller supplies exact pinned V90 crop, V107 context, V235 proposal compositor,
and V237 native-layer helper modules. Old post96 CF is only a layer carrier.
"""
import hashlib
import numpy as np


def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x


def resample_all(scene,group,crop90,context107):
 """Exact original RAW96 and context APIs; no use of helper local F32 sampler."""
 a=np.asarray(scene);q=np.asarray(group['q8'],np.float64);L=np.asarray(group['L8'],np.float64);avail=np.asarray(group['original_available8'],bool)
 if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or q.shape!=(8,2) or L.shape!=(8,) or avail.shape!=(8,):raise ValueError('Original scene/q8/L8/flags')
 nodes=np.zeros((8,3,96,96),np.uint8);valid=np.zeros((8,96,96),bool)
 for j in range(8):
  if np.isfinite(q[j]).all() and np.isfinite(L[j]) and L[j]>0:
   rgb,v=crop90(a,q[j],np.float64(L[j]),96)
   if rgb.dtype!=np.uint8 or rgb.shape!=(3,96,96) or v.dtype!=bool or v.shape!=(1,96,96):raise ValueError('Pinned V90 return contract')
   nodes[j],valid[j]=rgb,v[0]
 box=group['box_xyxy']
 if context107.valid_box(box):
  rgb,v=context107.sample_target_context(a,box)
  context=rgb.transpose(2,0,1).copy();context_valid=v.copy()
 else:context=np.zeros((3,224,224),np.float32);context_valid=np.zeros((224,224),bool)
 if context.dtype!=np.float32 or context.shape!=(3,224,224) or context_valid.dtype!=bool or context_valid.shape!=(224,224):raise ValueError('Pinned V107 return contract')
 support=avail&valid[:,47:49,47:49].all((1,2))&valid.any((1,2))
 return {'NODE_RGB96':nodes,'NODE_VALID96':valid,'CONTEXT_RGB224_F32':context,'CONTEXT_VALID224':context_valid,'original_q8':q.copy(),'original_L8':L.copy(),'original_available8':avail.copy(),'node_available':support,'node_fraction':valid.mean((1,2)).astype(np.float32),'context_fraction':np.float32(context_valid.mean()),'original_box_xyxy':None if box is None else np.asarray(box,np.float64).copy(),'context_supported':bool(context_valid.any())}


def positive_support_pixels(reference,hw):
 r=np.asarray(reference)
 if r.dtype!=np.float64 or r.shape!=(2,) or not np.isfinite(r).all():raise ValueError('Teacher-only finite F64 r')
 nearest=np.floor(r+.5).astype(np.int64)
 sets={'continuous_reference':r[None], 'nearest_half_up_3x3':np.array([(nearest[0]+dx,nearest[1]+dy) for dy in (-1,0,1) for dx in (-1,0,1)],np.float64)}
 out={}
 for name,points in sets.items():
  pixels=[];inside=True
  for x,y in points:
   ix,iy=int(np.floor(x)),int(np.floor(y));dx,dy=x-ix,y-iy
   for ox,oy,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
    if w==0:continue
    px,py=ix+ox,iy+oy;inside&=0<=px<hw[1] and 0<=py<hw[0];pixels.append((px,py))
  out[name]={'supported':bool(inside),'pixels_xy':np.asarray(sorted(set(pixels)),np.int64)}
 return out


def renewed_teacher_proof(base,cover,sham,alpha_cover,alpha_sham,reference):
 """Renew proof on new native alpha, not old post96 acceptance or q-center."""
 if reference is None:return {'status':'UNKNOWN_REFERENCE','PASS':False,'teacher_reference_only':True,'whole_intended_part_correspondence':'UNKNOWN'}
 support=positive_support_pixels(np.asarray(reference,np.float64),base.shape[:2]);result={};passed=True
 for name,item in support.items():
  pp=item['pixels_xy'];ok=item['supported'];x,y=pp.T if len(pp) else (np.array([],int),np.array([],int))
  c=bool(alpha_cover[y,x].all()) if ok else None;s=bool((~alpha_sham[y,x]).all()) if ok else None
  unchanged=bool(np.array_equal(base[y,x],sham[y,x])) if ok else None
  result[name]={'supported':ok,'positive_native_pixels':len(pp),'COVER_all_opaque':c,'SHAM_all_clear':s,'SHAM_reference_RGB_unchanged':unchanged}
  passed&=bool(ok and c and s and unchanged)
 return {'status':'VERIFIED_SOURCE_REFERENCE_FOOTPRINT' if passed else 'CONSTRUCTED_UNKNOWN','PASS':bool(passed),'checks':result,'teacher_reference_only':True,'nearest_rule':'floor(r+.5); positive-weight taps only','whole_intended_part_correspondence':'UNKNOWN','q_center_visibility_label':'NOT_INFERRED'}


def compose_scene_pair(base,group,parent,donor_rgb,donor_mask,donor_id,crop90,context107,v235,scene_math):
 """One chosen old-FIT donor/proposal -> fresh native COVER+ADJACENT_SHAM.

    Parent schema is C's original_input wrapper; decoded donor is exact closed
    FIT metadata choice. Caller owns SHA/role/donor64/different-object admission.
    This pure function never grants training or native-gold authority.
    """
 base=np.asarray(base);original=parent['original_input'];slot=int(parent['active_slot']);q=np.asarray(original['q'],np.float64);L=np.float64(original['L'])
 if base.dtype!=np.uint8 or base.ndim!=3 or base.shape[2]!=3 or not 0<=slot<8:raise ValueError('Original native scene/active slot')
 if original['fit'] is not True or int(original['target'])!=0 or original['original_query_available'] is not True or original['crop_available'] is not True:raise ValueError('Only original known available FIT D parents')
 if not np.array_equal(q,np.asarray(group['q8'],np.float64)[slot],equal_nan=True) or L!=np.asarray(group['L8'],np.float64)[slot]:raise ValueError('Original active q/L unchanged')
 if donor_id!=parent['same_V235_donor_id']:raise ValueError('No new donor choice')
 before=resample_all(base,group,crop90,context107)
 if sha(before['NODE_RGB96'][slot])!=parent['expected_BEFORE_RGB96_CHW_sha256'] or sha(before['NODE_VALID96'][slot])!=parent['expected_BEFORE_valid96_bool_sha256']:raise ValueError('BEFORE must replay original active RGB/valid SHA; no tolerance')
 result={'inputs':{'BEFORE':before,'COVER':None,'ADJACENT_SHAM':None},'native_scenes':{'BEFORE':base.copy()},'native_alpha':{},'status':'UNCONSTRUCTED','constructed':False,'proposed_reference_targets':{'BEFORE':0,'COVER':None,'ADJACENT_SHAM':None},'training_admission':False,'new_native_gold':False,'old5367_proof_inherited':False,'source_original_D_preserved':True,'original_q_L_native_flags_unchanged':True,'donor_id':donor_id,'reference_sidecar_only':parent.get('teacher_authority'),'proof':None}
 ref=parent.get('teacher_reference_xy')
 if ref is None:result.update(status='UNKNOWN_REFERENCE',failure_reason='No fabricated reference; original D retained');return result
 ref=np.asarray(ref,np.float64)
 proposal=v235.compose_counterfactual_pair(before['NODE_RGB96'][slot],before['NODE_VALID96'][slot],donor_rgb,donor_mask,q,L,ref,original['row_id'])
 if 'donor_plan' in proposal and plain(proposal['donor_plan'])!=plain(parent['same_V235_donor_plan']):raise ValueError('Same V235 proposal plan required')
 result['proposal_success']=bool(proposal['success']);result['proposal_failure_reason']=proposal.get('failure_reason')
 if not proposal['success']:result['status']='PROPOSAL_CONSTRUCTION_FAILED';return result
 for oldrole,newrole in [('COVER','COVER'),('SHAM','ADJACENT_SHAM')]:
  carrier=np.asarray(proposal[oldrole]).transpose(1,2,0);alpha=np.asarray(proposal['alpha_'+oldrole])
  donor_native,native_alpha=scene_math.lift_raw96_layer(carrier,alpha,q,L,base.shape[:2]);scene=base.copy();scene[native_alpha]=donor_native[native_alpha]
  if not np.array_equal(scene[~native_alpha],base[~native_alpha]):raise AssertionError('Only opaque layer may change scene')
  result['native_scenes'][newrole]=scene;result['native_alpha'][newrole]=native_alpha
  inputs=resample_all(scene,group,crop90,context107)
  for k in ('NODE_VALID96','CONTEXT_VALID224','original_available8','node_available','original_q8','original_L8'):
   if not np.array_equal(before[k],inputs[k],equal_nan=True):raise AssertionError('Original input support and geometry immutable')
  result['inputs'][newrole]=inputs
 proof=renewed_teacher_proof(base,result['native_scenes']['COVER'],result['native_scenes']['ADJACENT_SHAM'],result['native_alpha']['COVER'],result['native_alpha']['ADJACENT_SHAM'],ref)
 result.update(constructed=True,status='CONSTRUCTED_SOURCE_REFERENCE_PAIR' if proof['PASS'] else 'CONSTRUCTED_UNKNOWN',proof=proof)
 if proof['PASS']:result['proposed_reference_targets'].update(COVER=1,ADJACENT_SHAM=0)
 result['changed_pixels']={role:{'native':int(np.any(result['native_scenes'][role]!=base,axis=2).sum()),'active_RAW96':int(np.any(result['inputs'][role]['NODE_RGB96'][slot]!=before['NODE_RGB96'][slot],axis=0).sum()),'context_F32_samples':int(np.any(result['inputs'][role]['CONTEXT_RGB224_F32']!=before['CONTEXT_RGB224_F32'],axis=0).sum())} for role in ('COVER','ADJACENT_SHAM')}
 result['native_scene_sha256']={role:sha(scene) for role,scene in result['native_scenes'].items()}
 return result
