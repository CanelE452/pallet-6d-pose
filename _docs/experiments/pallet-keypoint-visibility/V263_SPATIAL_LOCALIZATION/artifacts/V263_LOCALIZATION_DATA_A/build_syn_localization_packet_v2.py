"""Root-authorized SYN-only metadata/label builder; no RGB, renderer or model forward."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import numpy as np
sys.dont_write_bytecode = True
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.float64)
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')

def need(x,m):
    if not bool(x): raise ValueError(m)

def safe(value,fresh=False):
    p=Path(value);need(p.is_absolute()and'..'not in p.parts,'Absolute path without traversal')
    need(not any(x in str(p).casefold()for x in('v112','cache','.pyc','current223','private_evaluation','source_holdout')),'Forbidden before FS')
    for a in[*reversed(p.parents),p]:
        try: mode=os.lstat(a).st_mode
        except FileNotFoundError: continue
        need(not stat.S_ISLNK(mode),'Symlink artifact/ancestor rejected')
    if fresh: need(not p.exists(),'Fresh output; no overwrite/retry')
    return p

def normalized(b):
    p=safe(b['path']);return dict(path=str(p),sha256=b['sha256'],size_bytes=b.get('size_bytes',p.stat().st_size))

def read(b):
    b=normalized(b);p=safe(b['path']);raw=p.read_bytes()
    need(len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'],'Bound artifact SHA/size')
    return raw

def doc(b): return json.loads(read(b))
def records(b): return [json.loads(x)for x in read(b).splitlines()if x]
def bind(p):
    p=safe(p);raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
def put(p,x):
    with safe(p,fresh=True).open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return bind(p)
def lines(p,rows):
    with safe(p,fresh=True).open('x',encoding='utf-8')as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    return bind(p)

def project(x,camera):
    # Literal V93 world_to_camera/project_camera arithmetic. No pose fit/refinement.
    R=np.asarray(camera['R'],np.float64);t=np.asarray(camera['t'],np.float64);K=np.asarray(camera['K'],np.float64)
    v=np.asarray(x,np.float64)@R.T+t
    need(np.isfinite(v).all()and(v[:,2]>0).all(),'Positive finite saved camera depth')
    return(v[:,:2]/v[:,2,None])*np.array([K[0,0],K[1,1]])+K[:2,2]

def declared_rotation(anchors,perm):
    candidates=[]
    for k in range(4):
        A=np.linalg.matrix_power(np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),k)
        rot=anchors@A.T;half=np.abs(anchors[0]);half=half[[2,1,0]]if k%2 else half
        expected=SIGNS*half;pp=[]
        for p in expected:
            match=np.flatnonzero(np.all(rot==p,axis=1));need(len(match)==1,'Declared quarter corner equality');pp.append(int(match[0]))
        if pp==perm:candidates.append(A)
    need(len(candidates)==1,'Exactly one declared quarter for stored CF permutation; no GT-best map')
    return candidates[0]

def scene_proof(scene,geometry,tolerance):
    anchors,witness=geometry;perm=scene['permCF_to_canonical'];rr=scene['records_CF'];cam=scene['camera']
    need(sorted(perm)==list(range(8))and len(rr)==8,'Original8CF permutation/records')
    proof=[];maximum=0.
    for j,record in enumerate(rr):
        p=perm[j];need(record['CF_index']==j and record['canonical_index']==p,'Exact declared native ID')
        need(np.array_equal(witness[p,0],anchors[p]),'Existing physical anchor is witness0')
        uv=project(witness[p],cam);r=np.asarray(record['projected_reference_xy'],np.float64)
        error=float(np.max(np.abs(uv[0]-r)));maximum=max(maximum,error);need(error<=tolerance,'Native material-anchor projection hardcheck')
        state=record['combined_witness_states'];alone=record['target_witness_states'];active=set(record['active_witness_indices'])
        need(len(state)==len(alone)==len(witness[p]),'Saved full witness state arrays')
        axes=[a for a in range(3)if all(i in active and state[i]=='D'for i in range(1+3*a,4+3*a))]
        pairs=[];sines=[]
        for ii,a in enumerate(axes):
            u=uv[3+3*a]-uv[0]
            for b in axes[ii+1:]:
                v=uv[3+3*b]-uv[0];den=float(np.linalg.norm(u)*np.linalg.norm(v));cross=float(u[0]*v[1]-u[1]*v[0])
                if den>0 and abs(cross)>64*np.finfo(np.float64).eps*den:pairs.append([a,b]);sines.append(abs(cross)/den)
        proof.append(dict(native_slot=j,canonical_index=p,r=r.tolist(),point_target_alone_state=alone[0],point_combined_state=state[0],
            identity_known=True,identity_correct=True,direct_cue_known=True,direct_geometric_2edge=(state[0]=='D'and bool(pairs)),
            visible_edge_axes=axes,noncollinear_visible_edge_pairs=pairs,edge_pair_abs_sines=sines,
            photometric_direct_observability='UNKNOWN_NOT_MEASURED',GT_location_authority='PROJECTED_REGISTERED_MATERIAL_ANCHOR_WITNESS0'))
    A=declared_rotation(anchors,perm);xyz=anchors[perm]@A.T;R=np.asarray(cam['R'],np.float64);Rcf=R@A.T
    transformed=dict(cam,R=Rcf.tolist());r8=np.asarray([r['r']for r in proof]);error=float(np.max(np.abs(project(xyz,transformed)-r8)))
    need(error<=tolerance,'Declared CF basis projection hardcheck; no fitting')
    return proof,dict(K=cam['K'],GT_R_canonical=cam['R'],GT_R_CF=Rcf.tolist(),GT_t=cam['t'],canonical_to_CF_rotation=A.tolist(),
        native_xyz_canonical_by_CF=anchors[perm].tolist(),native_xyz_CF_frame=xyz.tolist(),canonical_dimensions_xyz=(2*np.abs(anchors[0])).tolist(),
        CF_dimensions_xyz=(2*np.abs(xyz[0])).tolist(),permCF_to_canonical=perm,raw_hw=[cam['size'],cam['size']],
        projection_maxabs=max(maximum,error),initial_pose=None,initial_pose_reason='NO_SAVED_INITIAL_R_t_IN_ORIGINAL_Q8_PACKET;GT_NOT_SUBSTITUTED',
        physical_metric_basis='B/Root must freeze Q_truth and sourceRxPi once pergroup; never use predicted WD to rotate GT')

def local_row(info,q,L,available,central4,anyvalid,row_id,source_index,group_index,fit,intervention=None):
    finite=isinstance(q,list)and len(q)==2 and all(math.isfinite(float(x))for x in q)and L is not None and math.isfinite(float(L))and L>0
    uv=(47.5+96*(np.asarray(info['r'])-np.asarray(q))/L).tolist()if finite else None
    inside=uv is not None and all(-.5<=x<95.5 for x in uv);incentres=uv is not None and all(0<=x<=95 for x in uv)
    support=bool(available and finite and central4 and anyvalid)
    state=info['point_combined_state'];vlabel=0 if state=='D'else 1 if state=='E'else-1
    vmask=support and info['point_target_alone_state']=='D'and vlabel>=0
    locmask=support and inside and info['direct_geometric_2edge']and info['identity_known']and info['identity_correct']
    reason=[]
    if not available:reason.append('ORIGINAL_INPUT_U')
    if not support:reason.append('INPUT_GEOMETRY_UNSUPPORTED')
    if not inside:reason.append('OUT_OF_SEARCH_RANGE')
    if state!='D':reason.append('POINT_NOT_DIRECT_GEOMETRIC_D')
    if not info['direct_geometric_2edge']:reason.append('NO_TWO_NONCOLLINEAR_VISIBLE_EDGES')
    return dict(row_id=row_id,source_index=source_index,group_index=group_index,fit=fit,q0=q,L=L,
        original_available=bool(available),input_supported=support,reference_xy=info['r'],reference_RAW96_xy=uv,
        inside_search_canvas=inside,inside_sample_centre_extent=incentres,semantic_visibility=vlabel,visibility_loss_mask=vmask,
        LOC_valid=locmask,LOC_valid_reasons=reason,intervention_target_native_slot=intervention,**info)

def run(entry_path):
    entry=doc(bind(entry_path));need(entry.get('approved_SYN_localization_label_build')is True,'Root explicit SYN label scope')
    out=safe(entry['output_directory'],fresh=True);out.mkdir()
    try:
        meta=records(entry['inputs']['syn_metadata']);teachers=records(entry['inputs']['point_mapping'])
        need(len(meta)==len(teachers)==7040,'Original SOURCE7040 only')
        byid={r['row_id']:r for r in teachers};need(len(byid)==7040,'Unique original teacher row IDs')
        # Only META6 and valid96 are decoded from this SYN-only NPZ. RGB96 is never decoded.
        raw=read(entry['inputs']['syn_inputs_geometry_members_only'])
        import io
        with np.load(io.BytesIO(raw),allow_pickle=False)as z:
            need(set(z.files)==set(META)|{'rgb96','valid96'},'Pinned original SYN INPUT8 schema')
            arrays={k:z[k]for k in(*META,'valid96')}
        del raw
        need(arrays['valid96'].dtype==bool and arrays['valid96'].shape==(7040,96,96),'Exact original valid96')
        gt=[];geometry_bindings={}
        for b in entry['GT110_metadata_whitelist']:
            d=doc(b);need(d.get('completed')is True and len(d['rows'])==4,'Exact original4scene GT metadata')
            for s in d['rows']:
                need(s['split']=='TRAIN','Only original SOURCE TRAIN110; fixed internalFIT/DEV separate')
                gt.append(s);gb=normalized(s['geometry']);geometry_bindings[gb['path']]=gb
        need(len(gt)==440 and len({s['scene_index']for s in gt})==440,'All original440 scenes')
        put(out/'PROJECTED_WITNESS_GEOMETRY_READ_WHITELIST.json',{'bindings':list(geometry_bindings.values()),'decoded_members':entry['geometry_member_whitelist'],'render_or_RGB_members':False})
        geoms={}
        for path,b in geometry_bindings.items():
            with np.load(io.BytesIO(read(b)),allow_pickle=False)as z:
                geoms[path]=(z['native_anchor3D'].copy(),z['native_witness_points'].copy())
        scene_by_id={s['scene_index']:s for s in gt};proofs={};basis={};base_scene={}
        for s in gt:
            proof,frame=scene_proof(s,geoms[s['geometry']['path']],entry['projection_absolute_tolerance_pixels'])
            proofs[s['scene_index']]=proof;basis[s['scene_index']]=frame
            if s['variant']=='BEFORE':base_scene[s['base_index']]=s
        logical=defaultdict(list);row_fits={};source_base_fits={}
        for i,m in enumerate(meta):
            t=byid[m['row_id']]
            for k in META:
                need(m[k]==t[k],'Original META6 teacher join')
                got=arrays[k][i].tolist()if isinstance(arrays[k][i],np.ndarray)else arrays[k][i].item()if isinstance(arrays[k][i],np.generic)else arrays[k][i]
                need(np.array_equal(np.asarray(got),np.asarray(m[k]),equal_nan=True)if k in('query_xy','L')else got==m[k],'Original NPZ META6 order')
            need(m['component_role']=='TRAIN'and m['nominal_fit_eligible']is True and m['lineage_bound']is True,'Original nominalTRAIN lineage')
            scene=t['source_teacher_key'][0];j=m['native_slot'];need(t['source_teacher_key']==[m['source_image_original_global_index']//8,j],'Actual image nativeID not anchor/cache')
            p=proofs[scene][j];need(t['saved_projected_reference_xy']==p['r']and t['point0_combined_state']==p['point_combined_state']and t['point0_target_alone_state']==p['point_target_alone_state'],'Saved point teacher replay from same-scene metadata')
            fit=hashlib.sha256(('V233:'+m['component_id']).encode()).digest()[0]%5!=0
            b=scene_by_id[scene]['base_index'];need(b not in source_base_fits or source_base_fits[b]==fit,'Original base split coherent');source_base_fits[b]=fit
            row_fits[i]=fit;logical[(m['component_id'],m['family'],scene,m['anchor_original_global_index']//8)].append((i,m,t))
        groups=[];rows=[]
        for identity,entries in sorted(logical.items(),key=lambda x:min(v[0]for v in x[1])):
            entries=sorted(entries,key=lambda v:v[1]['native_slot']);need([m['native_slot']for _,m,_ in entries]==list(range(8)),'All8 original logical group; aliases retained')
            first=entries[0][1];scene=identity[2];gid=len(groups);g=dict(group_index=gid,group_id='ORIGINAL_%04d'%gid,dataset='ORIGINAL_SOURCE7040',component_id=identity[0],
                base_index=scene_by_id[scene]['base_index'],family=identity[1],variant=scene_by_id[scene]['variant'],source_image_scene_index=scene,anchor_scene_index=identity[3],fit=row_fits[entries[0][0]],
                source_role='FIT'if row_fits[entries[0][0]]else'DEV',row_ids8=[m['row_id']for _,m,_ in entries],source_indices8=[i for i,_,_ in entries],
                q8=first['context_q8'],L8=first['context_L8'],original_available8=first['context_available8'],native_proofs8=proofs[scene],
                RGB96_SOURCE_binding=entry['inputs']['syn_inputs_geometry_members_only'],RGB96_member='rgb96',RGB96_layout='NHWC_UINT8',image=entries[0][2]['actual_image_source_binding_inert'],geometry_binding=scene_by_id[scene]['geometry'],**basis[scene])
            for i,m,t in entries:
                j=m['native_slot'];need(g['q8'][j]==m['query_xy']and g['L8'][j]==m['L']and g['original_available8'][j]==m['original_query_available'],'Original active q8/L8/U')
                v=arrays['valid96'][i];rows.append(local_row(proofs[scene][j],m['query_xy'],m['L'],m['original_query_available'],bool(v[47:49,47:49].all()),bool(v.any()),m['row_id'],i,gid,g['fit']))
            groups.append(g)
        # Existing balanced FIT scene labels are independently checked perquery j, never copied from intervention k.
        new_scenes=records(entry['inputs']['new_scenes'])if entry.get('include_V255',False)else[];metadata_bindings=[normalized(s['metadata'])for s in new_scenes]
        put(out/'V255_SCENE_METADATA_READ_WHITELIST.json',{'bindings':metadata_bindings,'RGB_NPZ_reads':0})
        for s,b in zip(new_scenes,metadata_bindings):
            d=doc(b);need(d['source_role']=='FIT'and source_base_fits[d['base_index']],'Only fixed originalFIT parents')
            old=base_scene[d['base_index']];synthetic_scene=dict(old,records_CF=d['fresh_records_CF'],permCF_to_canonical=d['permCF_to_canonical'])
            proof,frame=scene_proof(synthetic_scene,geoms[old['geometry']['path']],entry['projection_absolute_tolerance_pixels'])
            qmeta=d['original_before_input'];q8=qmeta['query_xy']if'query_xy'in qmeta else qmeta['q8'];L8=qmeta.get('L8',qmeta.get('L_by_native'));av=qmeta.get('original_available8',qmeta.get('original_query_available'))
            need(len(q8)==len(L8)==len(av)==8,'V255 unchanged BEFORE q8/L8/U')
            gid=len(groups);sceneid=s.get('scene_id',s.get('id'));k=d.get('target_native_slot');g=dict(group_index=gid,group_id=sceneid,dataset='V255_BALANCED_FIT',base_index=d['base_index'],
                family='V255_NATIVE_INTERVENTION',variant=d['variant'],intervention_target_native_slot=k,source_image_scene_index=None,fit=True,source_role='FIT',q8=q8,L8=L8,original_available8=av,
                native_proofs8=proof,image=s.get('RGB',s.get('rgb',s.get('RGB_binding',s.get('pixels')))),scene_metadata_binding=b,geometry_binding=old['geometry'],**frame)
            h,w=frame['raw_hw']
            for j in range(8):
                q=q8[j];L=L8[j];finite=q is not None and len(q)==2 and all(math.isfinite(float(x))for x in q)and L is not None and math.isfinite(float(L))and L>0
                c4=finite and all(0<=q[a]-L/192 and q[a]+L/192<=([w,h][a]-1)for a in range(2))
                rid=str(sceneid)+':native'+str(j);rows.append(local_row(proof[j],q,L,bool(av[j]),c4,c4,rid,None,gid,True,k))
            groups.append(g)
        gb=lines(out/'GROUPS_WITH_NATIVE_GEOMETRIC_PROOF.jsonl',groups);rb=lines(out/'ALL_LOCALIZATION_ROWS.jsonl',rows)
        arrays_out={'group_index':np.asarray([r['group_index']for r in rows],np.int64),'native_slot':np.asarray([r['native_slot']for r in rows],np.int8),
            'fit':np.asarray([r['fit']for r in rows],bool),'query_xy':np.asarray([r['q0']for r in rows],np.float64),'L':np.asarray([r['L']for r in rows],np.float64),
            'reference_xy':np.asarray([r['reference_xy']for r in rows],np.float64),'reference_RAW96_xy':np.asarray([r['reference_RAW96_xy']if r['reference_RAW96_xy']is not None else[np.nan,np.nan]for r in rows],np.float64),
            'semantic_visibility':np.asarray([r['semantic_visibility']for r in rows],np.int8),'visibility_loss_mask':np.asarray([r['visibility_loss_mask']for r in rows],bool),
            'LOC_valid':np.asarray([r['LOC_valid']for r in rows],bool),'direct_geometric_2edge':np.asarray([r['direct_geometric_2edge']for r in rows],bool),'row_id':np.asarray([r['row_id']for r in rows])}
        np.savez_compressed(out/'LOCALIZATION_TARGETS.npz',**arrays_out);ab=bind(out/'LOCALIZATION_TARGETS.npz')
        census=[]
        for dataset in('ORIGINAL_SOURCE7040','V255_BALANCED_FIT'):
            for role in('FIT','DEV'):
                for j in range(8):
                    rr=[r for r in rows if groups[r['group_index']]['dataset']==dataset and r['fit']==(role=='FIT')and r['native_slot']==j]
                    census.append(dict(dataset=dataset,role=role,native_slot=j,N=len(rr),states=dict(Counter(r['point_combined_state']for r in rr)),
                        geometric_2edge=sum(r['direct_geometric_2edge']for r in rr),LOC_valid=sum(r['LOC_valid']for r in rr),visibility_loss=sum(r['visibility_loss_mask']for r in rr),
                        inputU=sum(not r['original_available']for r in rr),out_of_search=sum(not r['inside_search_canvas']for r in rr)))
        summary=put(out/'SUMMARY.json',dict(completed=True,PASS=True,original7040=len(meta),original_logical_groups=len(logical),V255_scene_groups=len(new_scenes),all_rows=len(rows),
            original_bases_by_role=dict(Counter('FIT'if x else'DEV'for x in source_base_fits.values())),census=census,
            original_split_unchanged=True,GT_nearest_matching=0,RGB_decodes_render_model_PnP_training=0,photometric_observability='UNKNOWN: geometric 2edge cue only',
            presence_mesh_support_prerequisite=False,V255_other_queries='Eachquery independentlypassesits ownfreshpoint+edgeproof;nevercopy interventionk target',
            initial_pose='Not saved; None. GTcamera is target/metric metadata only; not initializer',metric_basis='A/canonical/CF source frames explicit; Root/B Q_truth/sourceRxPi policy separate',
            new_training_admission=False,allS_U_aliases_out_of_search_preserved=True))
        cb=put(out/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),entry=bind(entry_path),outputs=dict(groups=gb,rows=rb,targets=ab,summary=summary),
            original_metadata_and_label_only=True,SourceGT110_json=len(entry['GT110_metadata_whitelist']),geometry_decoded_members=['native_anchor3D','native_witness_points'],SYN_INPUT_members_decoded=list((*META,'valid96')),RGB_or_real_or_model_or_current_access=0,source_inputs=entry['inputs'],include_V255=entry.get('include_V255',False)))
        print(json.dumps(cb),flush=True)
    except Exception as e:
        put(out/'FAILURE.json',dict(completed=False,PASS=False,error_type=type(e).__name__,error=str(e),partial_preserved=True,automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(safe(ap.parse_args().entry))
