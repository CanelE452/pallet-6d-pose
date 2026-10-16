"""Compare saved source virtual intersections with authoritative native IDs.

Only existing source JSON annotations and saved public numerical rows are
read. No pose, line or model fitting, new rays or image generation occurs.
The protocol is locked before the numerical comparison runs once.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
NAME='pallet_corner_mechanism_audit_20261010_v1'
BOUNDARY='pallet_boundary_corner_refiner_20261010_v2'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
INPUT_PATHS=dict(
    source_CAL='_docs/experiments/'+BOUNDARY+'/CALIBRATION_ROWS.jsonl.gz',
    calibration='_docs/experiments/'+BOUNDARY+'/CALIBRATION.json',
    source_virtual_intersections='_docs/experiments/'+BOUNDARY+'/CALIBRATION_GEOMETRY_ROWS.jsonl.gz',
    source_native_metadata='_docs/experiments/pallet_kp_difficulty_20261010_v1/SOURCE_CEILING_ROWS.jsonl.gz',
    source_ready_targets='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz',
    original_source_geometry_code='scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return json.load(f)


def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def distance(a,b):return norm([x-y for x,y in zip(a,b)])


def close(a,b,label,tol=1e-9):
    if isinstance(a,list) and isinstance(b,list):
        require(len(a)==len(b),label+' shape')
        for x,y in zip(a,b):close(x,y,label,tol)
    else:require(isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,label+' drift')


def quantile(values,q):
    v=sorted(values);i=(len(v)-1)*q;a=int(math.floor(i));b=int(math.ceil(i));return v[a]+(v[b]-v[a])*(i-a)


def describe(v):
    require(v and all(math.isfinite(x) for x in v),'empty/nonfinite source contract distribution')
    mean=math.fsum(v)/len(v);var=math.fsum((x-mean)**2 for x in v)/(len(v)-1) if len(v)>1 else None
    return dict(n=len(v),mean=mean,sample_variance=var,sample_std=math.sqrt(var) if var is not None else None,
                median=quantile(v,.5),P90=quantile(v,.9),maximum=max(v),unit='px',ddof=1,quantile='linear')


def binding(root,path):
    path=Path(path);return dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size)


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def freeze(root,source,doc):
    require(source.is_dir(),'source annotation root unavailable')
    cal=rows(root/INPUT_PATHS['source_CAL']);require(len(cal)==len({r['id'] for r in cal})==128 and all(r['partition']=='calibration' for r in cal),'exact existing128 CAL scope')
    ceiling={r['id']:r for r in rows(root/INPUT_PATHS['source_native_metadata'])}
    declarations=[]
    for r in cal:
        metadata=ceiling[r['id']];require(metadata['index']==r['index'] and metadata['partition']==r['partition'],'source cache-native ID/index identity')
        b=metadata['source_annotation'];path=source/b['path'];require(path.resolve().is_relative_to(source) and sha(path)==b['sha256'] and path.stat().st_size==b['bytes'],'source annotation declaration drift')
        declarations.append(dict(id=r['id'],index=r['index'],partition=r['partition'],source_annotation=dict(path=b['path'],sha256=b['sha256'],bytes=b['bytes'],origin='external_readonly_source_annotation')))
    code=Path(__file__).resolve();plan=dict(schema='frozen_source_CAL_native_corner_contract_v1',
        purpose='test whether the saved virtual-wire calibration reference differs from the PnP native projected-cuboid definition',
        report_ID=NAME,inputs={k:binding(root,root/v) for k,v in INPUT_PATHS.items()},code=binding(root,code),
        source_annotations=declarations,frames=128,source_partition='calibration only; no training/source-test/real evaluation',
        native_ID='exact annotation objects[0].projected_cuboid[corner]; no new phase/Rx/permutation chosen',
        geometry_check='replay original perm_v4 width/depth convention and nearest eight-point bijection solely to confirm the annotation-native IDs; no pose fit',
        comparisons=['saved actual-wire TLS virtual reference vs annotation native projected cuboid',
                     'saved predicted intersection vs annotation native projected cuboid',
                     'saved predicted intersection vs saved virtual reference'],
        new_model_or_pose_or_line_fits=0,new_rays=0,new_images=0,new_training=0,
        thresholds='no new acceptance/tuning threshold; old source geometry projection check<0.05px and saved calibration radius only',
        limits=['Source annotation projected cuboid is authoritative for existing native ID semantics, not proof of a physically existing mesh vertex.',
                'Saved virtual reference uses certified actual-wire query points, but physical corner ownership is unproved.',
                'The difference diagnoses a reference-definition contract; it cannot establish real-domain causal error or repair performance.'])
    write_new(doc/'CAL_CORNER_CONTRACT_PROTOCOL.json',plan)
    print(json.dumps(dict(frozen=True,frames=128,protocol_sha256=sha(doc/'CAL_CORNER_CONTRACT_PROTOCOL.json'))))


def run(root,source,doc):
    started=time.monotonic();plan_path=doc/'CAL_CORNER_CONTRACT_PROTOCOL.json';plan=read(plan_path)
    require(plan['code']==binding(root,Path(__file__).resolve()),'frozen source contract code drift')
    for name,b in plan['inputs'].items():require(b==binding(root,root/INPUT_PATHS[name]),'frozen public source contract input drift')
    require(plan['frames']==128 and plan['new_model_or_pose_or_line_fits']==plan['new_rays']==plan['new_images']==plan['new_training']==0,'source contract policy drift')
    for name in ('CAL_CORNER_CONTRACT_ROWS.jsonl.gz','CAL_CORNER_CONTRACT_CHECKS.json'):require(not (doc/name).exists(),'preserve previous source contract execution')
    ready={r['id']:r for r in rows(root/INPUT_PATHS['source_ready_targets'])};native={};witness=[];projection_max=0.
    for declaration in plan['source_annotations']:
        b=declaration['source_annotation'];path=source/b['path'];require(sha(path)==b['sha256'] and path.stat().st_size==b['bytes'],'source annotation drift before read')
        annotation=read(path);o=annotation['objects'][0];camera=annotation['camera_data'];intr=camera['intrinsics'];K=[[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]]
        T=o['pose_transform'];R=[line[:3] for line in T[:3]];t=[line[3] for line in T[:3]];d=o['dimensions_m'];dims=[d['width'],d['height'],d['depth']]
        width_pairs={frozenset(p) for p in ((0,1),(2,3),(4,5),(6,7))}
        if frozenset(o['perm_v4'][:2]) not in width_pairs:dims=[dims[2],dims[1],dims[0]]
        actual=o['projected_cuboid'];require(len(actual)==8 and all(len(p)==2 and all(math.isfinite(x) for x in p) for p in actual),'authoritative native annotation shape')
        a,c,e=[v/2 for v in dims];X=[[-a,-c,-e],[a,-c,-e],[a,c,-e],[-a,c,-e],[-a,-c,e],[a,-c,e],[a,c,e],[-a,c,e]]
        projected=[]
        for point in X:
            xyz=[dot(line,point)+v for line,v in zip(R,t)];require(xyz[2]>0,'source annotation behind camera');uvw=[dot(line,xyz) for line in K];projected.append([uvw[0]/uvw[2],uvw[1]/uvw[2]])
        permutation=[min(range(8),key=lambda j:distance(p,projected[j])) for p in actual]
        require(len(set(permutation))==8,'source annotation-native projection bijection ambiguous')
        err=max(distance(p,projected[j]) for p,j in zip(actual,permutation));require(err<.05,'original source projection contract exceeds .05px');projection_max=max(projection_max,err)
        for edge,(aa,bb) in enumerate(EDGES):
            axes=[j for j in range(3) if abs(X[permutation[aa]][j]-X[permutation[bb]][j])>1e-12]
            require(len(axes)==1 and ((axes[0]==1)==(edge in (1,3,5,7))),'native graph or vertical-edge identity mismatch')
        r=ready[declaration['id']];require(r['index']==declaration['index'] and r['partition']=='calibration' and r['raw_hw']==[camera['height'],camera['width']],'READY source-native frame identity')
        close(r['K'],K,'READY K');close(r['R'],R,'READY source body R');close(r['t'],t,'READY source body t');close(r['dimensions'],dims,'READY original source dimensions')
        native[r['id']]=actual;witness.append(dict(id=r['id'],index=r['index'],native_to_original_cuboid_ids=permutation,projection_max_error_px=err,source_annotation=b))
    geometry=rows(root/INPUT_PATHS['source_virtual_intersections']);cal=read(root/INPUT_PATHS['calibration']);scale=cal['uncertainty']['corner_scale'];out=[]
    for item in geometry:
        fid=item['id'];k=item['corner'];r=ready[fid]
        require(fid in native and r['index']==item['index'] and r['partition']=='calibration' and 0<=k<8,'saved CAL corner/native ID join')
        require(item['reference_from_certified_actual_wire_queries'] and item['physical_corner_ownership_independently_proved'] is False,'saved virtual reference ownership contract')
        for edge,qids in zip(item['edges'],item['supporting_query_ids']):
            require(k in EDGES[edge] and len(qids)>=3 and all(q//7==edge for q in qids),'CAL incident edge/query ID contract')
            for q in qids:
                target=r['queries'][q];require(target['query']==q and target['edge']==edge and target['proposed_target']=='POSITIVE' and target['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE' and target['actual_target_uv'] is not None,'CAL reference support not certified actual-wire POS')
        proxy=native[fid][k];virtual=item['reference_xy'];predicted=item['predicted_xy'];radius=item['sigma_px']*scale
        ev=distance(predicted,virtual);close(ev,item['error_px'],'saved CAL predicted-virtual error',1e-8)
        out.append(dict(id=fid,index=item['index'],corner=k,edges=item['edges'],source_annotation_native_xy=proxy,
                        saved_virtual_wire_reference_xy=virtual,saved_predicted_intersection_xy=predicted,
                        virtual_minus_native_xy=[x-y for x,y in zip(virtual,proxy)],virtual_vs_native_error_px=distance(virtual,proxy),
                        predicted_vs_native_error_px=distance(predicted,proxy),predicted_vs_virtual_error_px=ev,
                        recorded_radius_px=radius,predicted_vs_native_within_radius=distance(predicted,proxy)<=radius,
                        predicted_vs_virtual_within_radius=ev<=radius,reference_physical_corner_ownership_proved=False))
    require(len(out)==348 and len(native)==128,'fixed source CAL348/128 population')
    for name,b in plan['inputs'].items():require(b==binding(root,root/INPUT_PATHS[name]),'public source contract input changed during comparison')
    for r in plan['source_annotations']:
        b=r['source_annotation'];path=source/b['path'];require(sha(path)==b['sha256'] and path.stat().st_size==b['bytes'],'source annotation changed during comparison')
    row_path=doc/'CAL_CORNER_CONTRACT_ROWS.jsonl.gz'
    with row_path.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
        for r in out:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())
    result=dict(schema='actual_existing_source_CAL_native_corner_contract_checks_v1',complete=True,passed=True,report_ID=NAME,
                protocol=binding(root,plan_path),rows=binding(root,row_path),frames=128,calibration_corner_rows=348,
                source_native_ID_mapping='authoritative objects[0].projected_cuboid[corner]; no inferred phase added',
                original_geometry_projection_max_error_px=projection_max,frame_geometry_witnesses=witness,
                virtual_vs_native_error_px=describe([r['virtual_vs_native_error_px'] for r in out]),
                predicted_vs_native_error_px=describe([r['predicted_vs_native_error_px'] for r in out]),
                predicted_vs_virtual_error_px=describe([r['predicted_vs_virtual_error_px'] for r in out]),
                within_recorded_radius=dict(native_projected_cuboid=sum(r['predicted_vs_native_within_radius'] for r in out),virtual_wire_reference=sum(r['predicted_vs_virtual_within_radius'] for r in out),n=348),
                all_public_and_external_source_bindings_unchanged=True,
                actual_calls=dict(source_annotation_JSON_reads=128,source_native_projection_arithmetic_frames=128,stored_corner_comparisons=348,
                                  source_annotation_hash_passes=256,detector=0,head=0,PnP=0,line_fits=0,rays=0,optimizer_updates=0,real_GT_loader=0,benchmark=0),
                limits=plan['limits'],wall_seconds=time.monotonic()-started)
    write_new(doc/'CAL_CORNER_CONTRACT_CHECKS.json',result)
    print(json.dumps({k:result[k] for k in ('complete','frames','calibration_corner_rows','original_geometry_projection_max_error_px','virtual_vs_native_error_px','within_recorded_radius','wall_seconds')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--source-root',type=Path,required=True,help='Read-only external source root; public evidence stores relative annotation paths and hashes only.')
    args=p.parse_args();root=args.root.resolve();source=args.source_root.resolve();doc=root/'_docs/experiments'/NAME
    (freeze if args.stage=='freeze' else run)(root,source,doc)


if __name__=='__main__':main()
