"""Saved geometry diagnostic audit; independent scalar rays, no production imports/render."""
from pathlib import Path
import hashlib, json, math, datetime
import numpy as np

ROOT=Path(__file__).resolve().parent
V93=ROOT.parent
OUTPUT=V93/'raster_prototype/FIXED_PART152_SIX_CAMERAS'
EXPECTED_SEAL='7f44b42583d4a5cc912eb78f6d8e4fae62a20d387b6823310085b207e40d0c3e'
EXPECTED_POLICY='99a707543cf1cf0966c3d3848f410a8fa535cb8f8ccaa21540079bebae091932'
EXPECTED_ENTRY='dcf8078dbf47482895db6eae053c1becedb005c2ec621e0f02664131d1f370b0'
SALT='V93_INDEPENDENT_RAY_SPOTS_V1'
NAMES=['plusX','minusX','plusY','minusY','plusZ','minusZ']
SELECTED=152
FOREIGN=308
EPS=np.finfo(np.float64).eps

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):
    return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def check(d):
    assert sha(d['path'])==d['sha256'],d['path']
    return Path(d['path'])
def write(path,obj):
    assert not path.exists()
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def camera_from_bounds(bounds,axis):
    bounds=np.asarray(bounds,float);axis=np.asarray(axis,float)
    centre=(bounds[0]+bounds[1])/2
    radius=float(np.sqrt(np.sum(((bounds[1]-bounds[0])/2)**2)))
    distance=1.3*radius/math.sin(math.radians(25))
    forward=-axis
    up=np.array([0.,0.,1.]) if abs(axis[1])==1 else np.array([0.,1.,0.])
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    rotation=np.stack([right,np.cross(forward,right),forward])
    eye=centre+axis*distance
    focal=128/math.tan(math.radians(25))
    return rotation,-rotation@eye,np.array([[focal,0,127.5],[0,focal,127.5],[0,0,1.]]),radius*1e-4

def top_left(a,b):
    return (b[:,1]<a[:,1])|((b[:,1]==a[:,1])&(b[:,0]>a[:,0]))
def edges(a,b,x,y):return (b[:,0]-a[:,0])*(y-a[:,1])-(b[:,1]-a[:,1])*(x-a[:,0])

def prepare_triangles(vertices,triangles,pids,faceids,orientation,double_sided,K,near):
    xyz=np.asarray(vertices,float)[triangles]
    normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    plane=np.sum(normal*xyz[:,0],axis=1)
    active=(np.asarray(double_sided)|((plane*np.asarray(orientation))<0))&(np.sum(normal*normal,axis=1)>0)
    assert np.all(xyz[:,:,2]>near)
    xy=xyz[:,:,:2]/xyz[:,:,2:]*np.array([K[0,0],K[1,1]])+K[:2,2]
    area=edges(xy[:,0],xy[:,1],xy[:,2,0],xy[:,2,1])
    reverse=area<0
    xy[reverse]=xy[reverse][:,[0,2,1]]
    active&=np.isfinite(area)&(area!=0)
    return dict(xy=xy,normal=normal,plane=plane,active=active,pids=np.asarray(pids),faces=np.asarray(faceids))

def ray_parts(prepared,x,y,K,part_count):
    xy=prepared['xy'];good=prepared['active'].copy()
    for i,j in [(0,1),(1,2),(2,0)]:
        e=edges(xy[:,i],xy[:,j],x,y)
        good&=(e>0)|((e==0)&top_left(xy[:,i],xy[:,j]))
    ray=np.array([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],1.])
    ids=np.flatnonzero(good)
    if len(ids):
        z=prepared['plane'][ids]/np.sum(prepared['normal'][ids]*ray,axis=1)
        assert np.isfinite(z).all() and np.all(z>0)
    else:z=np.empty(0)
    depth=np.full(part_count,np.inf);face=np.full(part_count,-1,np.int32)
    for i,t in zip(ids,z):
        part=int(prepared['pids'][i]);fid=int(prepared['faces'][i])
        if t<depth[part]:depth[part]=t;face[part]=fid
        elif t==depth[part] and (face[part]<0 or fid<face[part]):face[part]=fid
    return depth,face

def resolve(depth,faces,part_ids):
    best=float(np.min(depth))
    if not np.isfinite(best):return dict(depth=None,part_id=-1,face_id=-1,ambiguity=False,tie_part_count=0)
    tolerance=64*EPS*max(1.,abs(best))
    eligible=[i for i,d in enumerate(depth) if np.isfinite(d) and abs(d-best)<=tolerance]
    chosen=min(eligible,key=lambda i:part_ids[i])
    return dict(depth=best,part_id=int(part_ids[chosen]),face_id=int(faces[chosen]),ambiguity=len(eligible)>1,tie_part_count=len(eligible))

def pick(mask,camera):
    candidates=[]
    for y,x in zip(*np.where(mask)):
        key=f'{SALT}\n{camera}\n{int(y)},{int(x)}'
        candidates.append((hashlib.sha256(key.encode()).hexdigest(),int(y),int(x)))
    return min(candidates) if candidates else None

def main():
    policy=ROOT/'SAVED_BUFFER_SPOT_POLICY_BEFORE_ACTUAL_OUTPUTS.json'
    assert sha(policy)==EXPECTED_POLICY
    seal_path=OUTPUT/'RASTER_OUTPUT_SEAL.json'
    assert sha(seal_path)==EXPECTED_SEAL
    seal=json.loads(seal_path.read_text())
    assert seal['completed'] and seal['PASS'] and seal['cameras']==6
    assert seal['selected_part']==SELECTED and seal['foreign_part']==FOREIGN
    assert seal['entry']['sha256']==EXPECTED_ENTRY
    entry=json.loads(check(seal['entry']).read_text())
    assert entry['approved'] and entry['geometry_only'] and not any(entry[k] for k in ['model','current','fit','PnP'])
    for key in ['code','math_code','protocol','export']:check(seal[key])
    files={Path(d['path']).name:d for d in seal['files']}
    for d in files.values():check(d)
    metadata=json.loads(Path(files['GEOMETRY_DIAGNOSTIC_RESULTS.json']['path']).read_text())
    assert metadata['completed'] and metadata['PASS'] and metadata['all6_retained']
    assert [x['camera'] for x in metadata['results']]==NAMES
    assert all(f'{name}.npz' in files and f'{name}.png' in files for name in NAMES)
    protocol=json.loads(Path(seal['protocol']['path']).read_text())
    inventory=json.loads(check(protocol['inventory']).read_text())
    # All complete/hash/6-camera checks above precede the first actual geometry/buffer load.
    geometry=np.load(check(protocol['safe_NPZ']),allow_pickle=False)
    vertices=geometry['vertices_world_m'];tri=geometry['triangles_global'];pids=geometry['triangle_asset_local_part_id']
    orientation=np.array([1 if p['USD_orientation']=='rightHanded' else -1 for p in inventory['per_mesh']])
    sided=np.array([p['double_sided'] for p in inventory['per_mesh']])
    assert vertices.shape==(172656,3) and tri.shape==(110478,3)
    assert set(np.unique(pids))==set(range(308))
    faceids=np.arange(len(tri),dtype=np.int32)
    source_bounds=np.array(inventory['stage']['bounds_world_m'])
    selected_bounds=np.array(inventory['per_mesh'][SELECTED]['world_bounds_m'])
    all_checks=[];selection=[];spots=[];max_depth_error=0.;max_relative_error=0.;all_discrepancies=[]
    for index,item in enumerate(metadata['results']):
        name=item['camera'];saved=np.load(check(item['NPZ']),allow_pickle=False)
        c=item['camera_matrices'];K=np.asarray(c['K']);R=np.asarray(c['R']);t=np.asarray(c['t'])
        ownR,ownt,ownK,near=camera_from_bounds(source_bounds,c['direction'])
        camera_error=max(float(np.max(np.abs(R-ownR))),float(np.max(np.abs(t-ownt))),float(np.max(np.abs(K-ownK))))
        assert camera_error<1e-12
        assert np.array_equal(saved['camera_K'],K) and np.array_equal(saved['camera_R_world_to_cv'],R) and np.array_equal(saved['camera_t_world_to_cv'],t)
        vc=vertices@R.T+t;assert np.min(vc[:,2])>near
        objects={tag:{k:saved[f'{tag}_{k}'] for k in ['depth','part_id','face_id','ambiguity','tie_part_count']} for tag in ['PA','PS','PV']}
        for tag,obj in objects.items():
            assert all(x.shape==(256,256) for x in obj.values())
            assert obj['depth'].dtype==np.float64 and obj['part_id'].dtype==obj['face_id'].dtype==np.int32
            assert obj['ambiguity'].dtype==bool and obj['tie_part_count'].dtype==np.uint16
            good=obj['part_id']>=0
            assert np.array_equal(np.isfinite(obj['depth']),good)
            assert np.all(np.isnan(obj['depth'][~good])) and np.all(obj['face_id'][~good]==-1)
            assert np.all(obj['depth'][good]>near)
            assert np.array_equal(obj['ambiguity'],obj['tie_part_count']>1)
            assert np.all(obj['tie_part_count'][good]>=1) and np.all(obj['tie_part_count'][~good]==0)
            allowed={SELECTED} if tag=='PA' else set(range(308+(tag=='PV')))
            assert set(np.unique(obj['part_id'][good]))<=allowed
            target=good&(obj['part_id']!=FOREIGN)
            assert np.all((obj['face_id'][target]>=0)&(obj['face_id'][target]<len(tri)))
            assert np.array_equal(pids[obj['face_id'][target]],obj['part_id'][target])
            foreign=good&(obj['part_id']==FOREIGN)
            assert np.all(np.isin(obj['face_id'][foreign],[len(tri),len(tri)+1]))
        pa,ps,pv=[objects[k] for k in ['PA','PS','PV']]
        A=(pa['part_id']==SELECTED)&~pa['ambiguity'];S=(ps['part_id']==SELECTED)&~ps['ambiguity'];V=(pv['part_id']==SELECTED)&~pv['ambiguity']
        assert not np.any(S&~A) and not np.any(V&~S)
        self_hidden=A&(ps['part_id']>=0)&(ps['part_id']!=SELECTED)&~ps['ambiguity']
        foreign_blocked=S&(pv['part_id']==FOREIGN)&~pv['ambiguity'];unknown=A&~(self_hidden|foreign_blocked|V)
        assert np.all((self_hidden.astype(int)+foreign_blocked+V+unknown)[A]==1)
        expected={'A_selected_alone':A,'S_selected_in_target':S,'V_selected_with_foreign':V,'S_raw_selected_certain':S,'V_raw_selected_certain':V,'self_hidden_other_target':self_hidden,'foreign_blocked':foreign_blocked,'unknown':unknown}
        for key,a in expected.items():assert np.array_equal(saved[key],a),key
        states=np.full((256,256),-2,np.int8);states[unknown]=-1;states[V]=0;states[foreign_blocked]=1;states[self_hidden]=2
        assert np.array_equal(states,saved['partition_state'])
        assert saved['raw_S_outside_A'].item()==saved['raw_V_outside_S'].item()==0
        counts=dict(A=int(A.sum()),S=int(S.sum()),V=int(V.sum()),SELF=int(self_hidden.sum()),FOREIGN=int(foreign_blocked.sum()),U=int(unknown.sum()))
        assert counts==item['counts'] and item['raw_monotonicity_violation_counts']==dict(S_outside_A=0,V_outside_S=0)
        assert {k:digest(a) for k,a in ps.items()}==item['PS_summary_digest_before_after']
        assert item['immutable_layer_digest_before']==item['immutable_layer_digest_after']
        # Independently derive foreign rectangle from declared world AABB, not masks.
        corners=np.array([[selected_bounds[x,0],selected_bounds[y,1],selected_bounds[z,2]] for x in [0,1] for y in [0,1] for z in [0,1]])
        cc=corners@R.T+t;uv=cc[:,:2]/cc[:,2:]*np.array([K[0,0],K[1,1]])+K[:2,2]
        lo,hi=uv.min(0),uv.max(0);rect=np.array([[lo[0]-2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,lo[1]-2],[lo[0]+.6*(hi[0]-lo[0])+2,hi[1]+2],[lo[0]-2,hi[1]+2]])
        zforeign=.5*float(((selected_bounds[0]+selected_bounds[1])/2@R.T+t)[2])
        fvertices=np.column_stack(((rect-K[:2,2])*zforeign/np.array([K[0,0],K[1,1]]),np.full(4,zforeign)))
        assert np.array_equal(rect,saved['foreign_rectangle_xy']) and np.array_equal(fvertices,saved['foreign_vertices_camera'])
        padded=np.pad(A,1);interior=np.logical_and.reduce([padded[dy:dy+256,dx:dx+256] for dy in range(3) for dx in range(3)])
        selected_tied=np.zeros_like(A);selected_not_tied=np.zeros_like(A)
        for obj in [ps,pv]:
            tol=64*EPS*np.maximum(1.,np.abs(obj['depth']))
            tied=A&obj['ambiguity']&(np.abs(pa['depth']-obj['depth'])<=tol)
            selected_tied|=tied;selected_not_tied|=A&obj['ambiguity']&~tied
        cohorts={'PA_EXPECTED_ANY':A,'PA_EXPECTED_INTERIOR':interior,'PA_EXPECTED_BOUNDARY':A&~interior,'KNOWN_SELF_BLOCKED':self_hidden,'KNOWN_FOREIGN_BLOCKED':foreign_blocked,'KNOWN_SELECTED_VISIBLE':V,'AMBIGUOUS_SELECTED_TIED':selected_tied,'AMBIGUOUS_SELECTED_NOT_TIED':selected_not_tied}
        prepared=prepare_triangles(vc,tri,pids,faceids,orientation[pids],sided[pids],K,near)
        foreign_prepared=prepare_triangles(fvertices,np.array([[0,1,2],[0,2,3]]),[0,0],[len(tri),len(tri)+1],[1,1],[True,True],K,near)
        computed={}
        for stratum,cohort in cohorts.items():
            chosen=pick(cohort,name)
            select=dict(camera=name,stratum=stratum,cohort_pixels=int(cohort.sum()),selected=None)
            if chosen:
                hashkey,y,x=chosen;select['selected']=dict(x=x,y=y,sha256=hashkey)
                if (y,x) not in computed:
                    dp,fp=ray_parts(prepared,x,y,K,308);df,ff=ray_parts(foreign_prepared,x,y,K,1)
                    predictions={'PA':resolve(dp[SELECTED:SELECTED+1],fp[SELECTED:SELECTED+1],[SELECTED]),'PS':resolve(dp,fp,list(range(308))),'PV':resolve(np.r_[dp,df],np.r_[fp,ff],list(range(309)))}
                    comparisons=[]
                    for tag,pred in predictions.items():
                        obj=objects[tag];errors={}
                        for key in ['part_id','face_id','ambiguity','tie_part_count']:
                            actual=obj[key][y,x].item()
                            if actual!=pred[key]:errors[key]=dict(saved=actual,oracle=pred[key])
                        actual=float(obj['depth'][y,x]);error=None
                        if pred['depth'] is None:
                            if not np.isnan(actual):errors['depth']='oracle empty, saved finite'
                        else:
                            error=abs(actual-pred['depth']);max_depth_error=max(max_depth_error,error);max_relative_error=max(max_relative_error,error/max(1.,abs(pred['depth'])))
                            if error>64*EPS*max(1.,abs(pred['depth'])):errors['depth_numeric_difference']=error
                        if errors:all_discrepancies.append(dict(camera=name,x=x,y=y,stage=tag,errors=errors))
                        comparisons.append(dict(stage=tag,oracle=pred,saved_depth=None if np.isnan(actual) else actual,depth_abs_error=error,discrepancies=errors))
                    computed[y,x]=dict(camera=name,x=x,y=y,comparison=comparisons)
                spots.append(dict(stratum=stratum,**computed[y,x]))
            selection.append(select)
        all_checks.append(dict(camera=name,counts=counts,camera_numeric_max_error=camera_error,all_full_array_dtype_ID_depth_ambiguity_partition_checks=True,raw_subset_violations=dict(S_outside_A=0,V_outside_S=0),PS_summary_digest_matches_saved=True,immutable_layer_before_after_certificate_equal=True,actual_layers_not_stored=True,geometry_only_foreign_rectangle_exact=True,ray_unique_pixels=len(computed)))
    report=dict(completed=True,PASS=not all_discrepancies,code=binding(Path(__file__)),source_seal=binding(seal_path),root_entry=seal['entry'],spot_policy=binding(policy),protocol=seal['protocol'],geometry_binding=protocol['safe_NPZ'],
                cameras=all_checks,selection=selection,spot_rows=spots,selection_rows=len(spots),unique_ray_pixels=sum(x['ray_unique_pixels'] for x in all_checks),empty_strata=sum(x['selected'] is None for x in selection),
                max_abs_depth_error=max_depth_error,max_relative_depth_error_over_max1=max_relative_error,discrepancies=all_discrepancies,
                actual_source_mesh_raster_cameras_by_producer=6,new_audit_full_image_renders=0,producer_peak_RSS_bytes=metadata['peak_RSS_bytes'],producer_output_bytes_before_JSON=metadata['output_bytes_before_JSON'],
                limits=['Whole-array checks are saved ID/depth validity/partition/digest reconciliation; independent ray geometry is limited to the fixedSHA spots.','Per-part layer banks were not saved; producer before/after digest equality is a bound certificate, not an independent replay of the entire bank.','Outcome-conditioned diagnostic spots are not prevalence estimates. Wholepart owner proxies are not nativecorner D/E/S, human gold or certified materials/scale.'],
                current_models_fit_native_labels_PnP=0,original_outputs_changed=0,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(ROOT/'SAVED_OUTPUT_AND_SCALAR_RAY_AUDIT.json',report)
    write(ROOT/'SAVED_OUTPUT_AUDIT_SEAL.json',dict(completed=True,PASS=report['PASS'],report=binding(ROOT/'SAVED_OUTPUT_AND_SCALAR_RAY_AUDIT.json'),source_seal=report['source_seal'],code=report['code'],spot_policy=report['spot_policy']))
    print(json.dumps(dict(PASS=report['PASS'],report=binding(ROOT/'SAVED_OUTPUT_AND_SCALAR_RAY_AUDIT.json'),selected=len(spots),max_abs_depth_error=max_depth_error,discrepancy_count=len(all_discrepancies))))

if __name__=='__main__':main()
