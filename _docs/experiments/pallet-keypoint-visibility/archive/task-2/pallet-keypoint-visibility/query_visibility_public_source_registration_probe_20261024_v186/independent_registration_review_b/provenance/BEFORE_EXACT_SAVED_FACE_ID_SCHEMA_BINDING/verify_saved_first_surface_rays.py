"""V186 independent PLY parser + saved first-surface camera-ray audit.

Preparation: metadata contracts, own pure module, invented mesh bytes only.
Actual: separate root entry permits original mesh/K/R/t and saved5 array keys.
No producer import or RGB/sensor/mask/model data is decompressed by this audit.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import platform
import struct
import traceback
import numpy as np

from independent_ray_triangle import camera_vertices,camera_ray,first_hit,depth_bound,fixed_grid_pixels,fixed_face_indices,invented_checks,require

H=Path(__file__).resolve().parent;V=H.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json'
MATH=V/'source_registration_review_a/MATH_CONTRACT.json'
RULES=H/'RAY_VERIFICATION_RULES_BEFORE_ASSET_BODIES.json'
KERNEL=H/'independent_ray_triangle.py'
C_CODE=V/'source_registration_preparation_c/probe_source_registration.py'
C_PROTOCOL=V/'source_registration_preparation_c/PROTOCOL_BEFORE_SOURCE_REGISTRATION_ASSET_BODIES.json'
COMPLETE=V/'fixed_source_registration_probe_v1/COMPLETE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_MESH_POSE_OR_SAVED_DEPTH_BODIES.json'
CHECKS=H/'GENERATED_MEANINGFUL_CHECKS.json';PREP=H/'PREPARATION_COMPLETE.json'
OUT=H/'actual_ray_verification_v1'
SCALAR_TYPES={'char':'b','int8':'b','uchar':'B','uint8':'B','short':'h','int16':'h','ushort':'H','uint16':'H','int':'i','int32':'i','uint':'I','uint32':'I','float':'f','float32':'f','double':'d','float64':'d'}


def safe(path):
    p=Path(path);require(not any(x in str(p).lower() for x in ('_v112','current223','/private_evaluation/','/val/000001/')),'Forbidden path')
    return p
def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):require(bind(b['path'])==b,'Frozen byte binding');return Path(b['path'])
def read(path):return json.loads(safe(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def parse_original_ply(data):
    """Independent header/scalar reader; preserve every vertex/face and fan order."""
    require(isinstance(data,bytes) and data.startswith(b'ply\n') or isinstance(data,bytes) and data.startswith(b'ply\r\n'),'PLY magic')
    end=data.find(b'end_header');require(end>=0,'PLY header termination')
    finish=data.find(b'\n',end);require(finish>=0,'PLY header newline');header=data[:finish+1].decode('ascii').splitlines();body=data[finish+1:]
    fmt=None;elements=[]
    for line in header[1:]:
        t=line.split()
        if not t or t[0] in ('comment','obj_info','end_header'):continue
        if t[0]=='format':require(t[2]=='1.0','PLY version');fmt=t[1]
        elif t[0]=='element':elements.append(dict(name=t[1],count=int(t[2]),properties=[]))
        elif t[0]=='property':
            require(elements,'PLY property requires element')
            if t[1]=='list':require(t[2] in SCALAR_TYPES and t[3] in SCALAR_TYPES,'PLY scalar list types');elements[-1]['properties'].append((t[4],t[2],t[3]))
            else:require(t[1] in SCALAR_TYPES,'PLY scalar type');elements[-1]['properties'].append((t[2],t[1],None))
        else:raise AssertionError('Unsupported PLY header declaration')
    require(fmt in ('ascii','binary_little_endian','binary_big_endian'),'Explicit PLY encoding')
    tokens=iter(body.decode('ascii').split()) if fmt=='ascii' else None;offset=0;order='<' if fmt=='binary_little_endian' else '>'
    def scalar(kind):
        nonlocal offset
        if fmt=='ascii':
            token=next(tokens);return float(token) if SCALAR_TYPES[kind] in ('f','d') else int(token)
        code=order+SCALAR_TYPES[kind];n=struct.calcsize(code);require(offset+n<=len(body),'PLY truncated body')
        result=struct.unpack_from(code,body,offset)[0];offset+=n;return result
    vertices=[];polygons=[]
    for el in elements:
        for _ in range(el['count']):
            row={}
            for name,kind,item in el['properties']:
                if item is None:row[name]=scalar(kind)
                else:
                    n=scalar(kind);require(isinstance(n,int) and n>=0,'PLY list length');row[name]=[scalar(item) for _ in range(n)]
            if el['name']=='vertex':require(all(k in row for k in ('x','y','z')),'Original xyz properties');vertices.append([row['x'],row['y'],row['z']])
            elif el['name']=='face':
                name='vertex_indices' if 'vertex_indices' in row else 'vertex_index';require(name in row,'Original face vertex indices');polygons.append(row[name])
    require(vertices and polygons,'Nonempty original mesh')
    vertices=np.asarray(vertices,np.float64);triangles=[];faceids=[]
    for faceid,poly in enumerate(polygons):
        require(len(poly)>=3 and all(isinstance(x,int) and 0<=x<len(vertices) for x in poly),'Original face indices; no repair')
        for j in range(1,len(poly)-1):triangles.append([poly[0],poly[j],poly[j+1]]);faceids.append(faceid)
    require(np.isfinite(vertices).all(),'Finite original vertex geometry')
    return vertices,np.asarray(triangles,np.int64),np.asarray(faceids,np.int64),dict(format=fmt,vertices=len(vertices),original_polygons=len(polygons),fan_triangles=len(triangles),geometry_repairs_or_unit_changes=0)


def samples(vertices_cam,triangles,K,hw):
    rows=[dict(kind='IMAGE_GRID9x7',slot=i,u=u,v=v,source_triangle_index=None,original_projected_centroid_xy=None) for i,(u,v) in enumerate(fixed_grid_pixels(hw))]
    for i,tid in enumerate(fixed_face_indices(len(triangles))):
        center=np.mean(vertices_cam[triangles[tid]],axis=0,dtype=np.float64)
        if not np.isfinite(center).all() or center[2]<=0:
            rows.append(dict(kind='TRIANGLE_ORDER17_PROJECTED_CENTROID',slot=i,source_triangle_index=tid,u=None,v=None,original_projected_centroid_xy=None,projection_status='NONPOSITIVE_OR_INVALID_Z'))
            continue
        homogeneous=K@center;xy=homogeneous[:2]/homogeneous[2]
        if not np.isfinite(xy).all():
            rows.append(dict(kind='TRIANGLE_ORDER17_PROJECTED_CENTROID',slot=i,source_triangle_index=tid,u=None,v=None,original_projected_centroid_xy=None,projection_status='NONFINITE_PROJECTION'));continue
        u,v=[int(math.floor(float(x)+.5)) for x in xy]
        rows.append(dict(kind='TRIANGLE_ORDER17_PROJECTED_CENTROID',slot=i,source_triangle_index=tid,u=u,v=v,original_projected_centroid_xy=xy.tolist(),projection_status='PROJECTED_NO_CLIP_OR_REPLACEMENT'))
    require(len(rows)==80,'All63+17slots, duplicate pixels kept')
    return rows


def compare_sample(row,vertices_cam,triangles,faceids,K,arrays):
    record=dict(row);zmap=arrays['model_Z_mm'];h,w=zmap.shape
    if row['u'] is None:
        record.update(comparison_status='NONPROJECTABLE_SLOT_RETAINED',strict_failure=False);return record
    u,v=row['u'],row['v'];ray=camera_ray(K,u,v);result=first_hit(vertices_cam,triangles,ray);record['independent_ray']=result
    if not (0<=u<w and 0<=v<h):
        record.update(comparison_status='PROJECTED_PIXEL_OOB_RETAINED',strict_failure=False);return record
    saved=float(zmap[v,u]);saved_hit=bool(np.isfinite(saved) and saved>0);hit=result['hit'];edge=bool(result['near_boundary_candidates'])
    require(bool(arrays['silhouette'][v,u])==saved_hit,'Saved depth/silhouette shape semantics')
    tri=int(arrays['triangle_id'][v,u]);face=int(arrays['original_face_id'][v,u])
    if saved_hit:require(0<=tri<len(triangles) and face==int(faceids[tri]),'Saved triangle/original face index mapping')
    else:require(np.isnan(saved) and tri==face==-1,'Saved nohit NaN/-1 policy')
    record.update(saved_Z_mm=saved if saved_hit else None,saved_triangle_index=tri,saved_original_face_id=face,boundary_case=edge)
    if saved_hit and hit is not None:
        difference=abs(saved-hit['camera_Z_mm']);bound=depth_bound(zmap.dtype,hit['camera_Z_mm']);agree=difference<=bound
        record.update(absolute_depth_difference_mm=difference,prospective_depth_difference_bound_mm=bound,
            saved_ray_triangle_index_equal=tri==hit['triangle_index'],comparison_status='HIT_DEPTH_AGREEMENT' if agree else 'BOUNDARY_DEPTH_DISAGREEMENT' if edge else 'INTERIOR_DEPTH_DISAGREEMENT',strict_failure=not agree and not edge)
    elif not saved_hit and hit is None:record.update(comparison_status='NO_HIT_AGREEMENT',strict_failure=False)
    else:record.update(comparison_status='BOUNDARY_FILL_DISAGREEMENT' if edge else 'INTERIOR_HIT_DISAGREEMENT',strict_failure=not edge)
    return record


def invented_adapter_checks():
    checks=invented_checks()['checks'][:]
    header='ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
    body='-1 -1 2\n1 -1 2\n1 1 2\n-1 1 2\n4 0 1 2 3\n'
    expected=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],np.float64)
    vv,tri,fid,_=parse_original_ply((header+body).encode());require(np.array_equal(vv,expected) and tri.tolist()==[[0,1,2],[0,2,3]] and fid.tolist()==[0,0],'Independent ASCII polygonfan preservesxyz/faceorder')
    checks.append('Own_ASCII_PLY_parser_original_xyz_polygonfan_and_original_faceIDs')
    for order,fmt in (('<','binary_little_endian'),('>','binary_big_endian')):
        raw=header.replace('ascii',fmt).encode()+b''.join(struct.pack(order+'fff',*point) for point in expected)+struct.pack(order+'Biiii',4,0,1,2,3)
        v,t,f,_=parse_original_ply(raw);require(np.array_equal(v,vv) and np.array_equal(t,tri) and np.array_equal(f,fid),'Independent binaryPLY bytefixture')
    checks.append('Own_binary_little_big_endian_scalar_list_PLY_fixtures')
    K=np.eye(3);slots=samples(vv,tri,K,[3,3]);require(len(slots)==80 and len([x for x in slots if x['kind']=='TRIANGLE_ORDER17_PROJECTED_CENTROID'])==17,'No centroidslot dedup')
    arrays=dict(model_Z_mm=np.full((3,3),np.nan,np.float64),triangle_id=np.full((3,3),-1,np.int32),original_face_id=np.full((3,3),-1,np.int32),silhouette=np.zeros((3,3),bool))
    arrays['model_Z_mm'][0,0]=2.;arrays['triangle_id'][0,0]=0;arrays['original_face_id'][0,0]=0;arrays['silhouette'][0,0]=True
    record=compare_sample(dict(kind='invented',slot=0,u=0,v=0),vv,tri,fid,K,arrays);require(record['comparison_status']=='HIT_DEPTH_AGREEMENT','Actual comparisonhelper mathematicalray')
    checks.append('Actual_saved_depth_comparison_helper_uses_independent_ray_and_fixed_bound')
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,source_mesh_GT_header_or_output_body_hash_reads=0,producer_imports=0)


def prepare():
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned CPU F64runtime')
    require(sha(SCOPE)=='c62e53f4337ced281031937e51c6ad9c562afc373a5f9ee351744749299a6b34'
        and sha(C_CODE)=='3b55be0d0a5c077b13ada5ec3fa8859675deb407b273c082160da576dbb0e9be'
        and sha(C_PROTOCOL)=='3abd46d8c3212d8c64b3fdd1d746b13918cab841fc737326254644cce6ef84c3','Frozen producer/source/math contracts')
    scope=read(SCOPE);c=read(C_PROTOCOL);require(c['scope']==bind(SCOPE) and c['official_math_contract']==bind(MATH),'Exact original math/protocol scope')
    write(CHECKS,invented_adapter_checks())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),own_ray_kernel=bind(KERNEL),rules=bind(RULES),scope=bind(SCOPE),official_math=bind(MATH),checks=bind(CHECKS),
        producer_code=bind(C_CODE),producer_protocol=bind(C_PROTOCOL),mesh_binding_inherited_not_read=scope['allowed_body_bindings']['mesh'],
        GT_camera_bindings_inherited_not_read=scope['allowed_body_bindings']['GT_camera'],fixed_frames=scope['fixed_frames_before_body_reads'],
        expected_complete_path=str(COMPLETE),complete_binding='ROOT_FILL_AFTER_ACTUAL_C_PROBE',
        sample_slots_per_instance=80,total_sample_slots400=400,triangle_order='Original PLY polygon fan expansion in file face order, no repair or nearclip fan expansion;17 equallyspaced original fan-triangle centroids. Originalpolygonface IDs separately verified. Duplicates retained.',
        saved_npz_whitelist=['model_Z_mm','triangle_id','original_face_id','silhouette','K','R_m2c','t_m2c_mm'],
        no_saved_sensor_or_RGB_or_mask_keys_loaded=True,
        pass_scope='All5 instances/all400 slots retained and no interior depth/hit mismatch to prospective quantizationbound. Boundary/fill discrepancies explicitly counted, not silently removed or interpreted as real label/registration quality.',
        output=str(OUT),actual_asset_mesh_header_GT_or_saved_output_body_hash_reads=0,automatic_retry=False,new_labels_or_training_admission=False)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_INDEPENDENT_RAY_REVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_independent_source_ray_review=False,source_TRAIN_only=True,no_model_pixels_labels_or_geometry_changes=True,
        code=bind(__file__),protocol=bind(PROTOCOL),probe_complete=None,output=str(OUT)))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_source_asset_header_body_hash_reads=0))
    print(json.dumps(dict(FINAL_READY=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry):
    e=read(entry);require(e.get('approved_independent_source_ray_review') is True and e.get('source_TRAIN_only') is True
        and e.get('no_model_pixels_labels_or_geometry_changes') is True,'Separate root actualray entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['output']==str(OUT),'Frozen owncode/protocol/output')
    p=read(PROTOCOL)
    for name in ('own_ray_kernel','rules','scope','official_math','checks','producer_code','producer_protocol'):check(p[name])
    require(read(CHECKS)['PASS'] and e['probe_complete']['path']==str(COMPLETE),'Inventedchecks/exactprobeCOMPLETE')
    complete=read(check(e['probe_complete']));require(complete['completed'] and complete['PASS'] and complete['code']==p['producer_code']
        and complete['protocol']==p['producer_protocol'] and complete['scope']==p['scope'] and complete['all_obj1_instances']==5,'Frozen completed all5 probe')
    require(not OUT.exists(),'No overwrite or automaticretry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),probe_complete=e['probe_complete'],source_TRAIN_only=True))
    consumed=[e['probe_complete']]
    try:
        vertices,triangles,faceids,meshmeta=parse_original_ply(check(p['mesh_binding_inherited_not_read']).read_bytes());consumed.append(p['mesh_binding_inherited_not_read'])
        summary=read(check(complete['summary']));consumed.append(complete['summary']);records=summary['instances']
        expected=[(f['frame_index'],ix) for f in p['fixed_frames'] for ix in f['all_obj1_GT_instance_indices']]
        require([(r['frame_index'],r['gt_instance_index']) for r in records]==expected and len(records)==5,'All5 fixedinstance originalorder')
        allslots=[];instance_summaries=[];frames={f['frame_index']:f for f in p['fixed_frames']};source={}
        for scene,b in p['GT_camera_bindings_inherited_not_read'].items():
            source[scene]=(read(check(b['GT'])),read(check(b['camera'])));consumed.extend([b['GT'],b['camera']])
        for record in records:
            frame=frames[record['frame_index']];gt,cam=source[frame['scene']];pose=gt[frame['frame_key']][record['gt_instance_index']]
            require(pose['obj_id']==1,'Original obj1 source index')
            K=np.asarray(cam[frame['frame_key']]['cam_K'],np.float64).reshape(3,3);R=np.asarray(pose['cam_R_m2c'],np.float64).reshape(3,3);t=np.asarray(pose['cam_t_m2c'],np.float64)
            b=record['arrays'];consumed.append(b)
            with np.load(check(b),allow_pickle=False) as z:arrays={k:z[k].copy() for k in p['saved_npz_whitelist']}
            require(all(np.array_equal(arrays[k],value) and arrays[k].dtype==np.float64 for k,value in [('K',K),('R_m2c',R),('t_m2c_mm',t)]),'Saved K/R/t exact original source metadata')
            shape=arrays['model_Z_mm'].shape;require(len(shape)==2 and arrays['model_Z_mm'].dtype in (np.float64,np.float32),'Saved depth floatdtype')
            require(arrays['triangle_id'].dtype==arrays['original_face_id'].dtype==np.int32 and arrays['silhouette'].dtype==bool
                and all(arrays[k].shape==shape for k in ('triangle_id','original_face_id','silhouette')),'Saved raster schemas')
            vc=camera_vertices(vertices,R,t);rows=[]
            for slot in samples(vc,triangles,K,shape):
                row=compare_sample(slot,vc,triangles,faceids,K,arrays);row.update(frame_index=record['frame_index'],gt_instance_index=record['gt_instance_index'],source_id=record['source_id']);rows.append(row)
            require(len(rows)==80,'All80slot records perinstance')
            allslots.extend(rows);counts=Counter(r['comparison_status'] for r in rows)
            differences=[r['absolute_depth_difference_mm'] for r in rows if 'absolute_depth_difference_mm' in r]
            instance_summaries.append(dict(frame_index=record['frame_index'],gt_instance_index=record['gt_instance_index'],sample_slots80=len(rows),status_counts=dict(counts),
                boundary_case_slots=sum(bool(r.get('boundary_case')) for r in rows),strict_failures=sum(r['strict_failure'] for r in rows),
                maximum_compared_absolute_depth_difference_mm=max(differences) if differences else None,saved_depth_dtype=str(arrays['model_Z_mm'].dtype)))
        require(len(allslots)==400,'Full5x80slot ledger retained')
        path=OUT/'ALL400_FIXED_RAY_VERIFICATION_SLOTS.jsonl'
        with path.open('x') as f:
            for r in allslots:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        for b in consumed:check(b)
        strict=sum(r['strict_failure'] for r in allslots)
        report=dict(completed=True,PASS=strict==0,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),probe_complete=e['probe_complete'],own_ray_kernel=p['own_ray_kernel'],
            independent_mesh=meshmeta,instances=instance_summaries,total_slots400=len(allslots),all_five_instances_and_duplicate_slots_retained=True,
            status_counts=dict(Counter(r['comparison_status'] for r in allslots)),strict_interior_failures=strict,
            outputs=dict(all400=bind(path)),all_consumed_source_and_saved_bindings_unchanged=consumed,
            no_RGB_sensor_depth_or_mask_array_keys_decompressed=True,
            limits='Firstsurface mathematical render check only; boundary/fill differences separately retained. Not sensor registration quality, physical native/part/occlusion gold, model result or training admission.',
            new_forward_fit_PnP_qL_pose_K_mesh_or_labels_changes=0,new_training_admission=False,selected_model=None,automatic_retry=False)
        write(OUT/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=report['PASS'],review=bind(OUT/'REVIEW.json'))))
        require(report['PASS'],'Independent interior raydepth verification mismatch preserved; no retry')
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--entry',type=Path)
    a=ap.parse_args();prepare() if a.prepare else run(a.entry)
