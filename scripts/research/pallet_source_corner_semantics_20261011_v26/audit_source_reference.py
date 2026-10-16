"""Compare saved actual-wire virtual references with existing source model corners.

Read-only geometry audit: forward-project the already supplied synthetic R,t.
No RGB, model, new labels, line/pose fits, training or real pose evaluation.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

READY='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
CEILING='_docs/experiments/pallet_kp_difficulty_20261010_v1/SOURCE_CEILING_ROWS.jsonl.gz'
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write_rows(p,rows):
    with p.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
            for r in rows:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def cuboid(d):
    a,b,c=[x/2 for x in d]
    return [[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]]


def project(x,R,t,K):
    camera=[dot(row,x)+v for row,v in zip(R,t)];pixel=[dot(row,camera) for row in K]
    assert camera[2]>0 and pixel[2]>0
    return [pixel[0]/pixel[2],pixel[1]/pixel[2]]


def stats(values):
    a=sorted(values);n=len(a);assert n
    mu=math.fsum(a)/n;var=math.fsum((v-mu)**2 for v in a)/n
    def q(p):
        v=(n-1)*p;lo=int(v);hi=min(n-1,lo+1)
        return a[lo]+(v-lo)*(a[hi]-a[lo])
    return dict(n=n,mean=mu,variance=var,std=math.sqrt(var),median=q(.5),P90=q(.9),max=a[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--labels-root',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();assert not out.exists();out.mkdir(parents=True)
    src=a.source.resolve();prior=a.prior.resolve();labels=a.labels_root.resolve()
    paths={'ready':src/READY,'ceiling':src/CEILING,'saved_corners':prior/'output/CORNER_ROWS.jsonl.gz',
        'source_geometry_code':src/'scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py',
        'cuboid_project_code':src/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        'virtual_reference_code':src/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py',
        'wire_semantic_code':src/'scripts/research/pallet_kp_supervision_gate_20261010_v1/actual_wire_proposal.py',
        'code':Path(__file__)}
    before={k:dict(path=str(v),bytes=v.stat().st_size,sha256=sha(v)) for k,v in paths.items()}
    ready={r['id']:r for r in read(paths['ready']) if r['partition']=='calibration'}
    ceiling={r['id']:r for r in read(paths['ceiling']) if r['partition']=='calibration'}
    assert len(ready)==len(ceiling)==128 and set(ready)==set(ceiling)
    label_paths={fid:labels/r['source_annotation']['path'] for fid,r in ceiling.items()}
    label_bindings={}
    for fid,path in label_paths.items():
        expected=ceiling[fid]['source_annotation'];observed=dict(bytes=path.stat().st_size,sha256=sha(path))
        assert observed=={k:expected[k] for k in ('bytes','sha256')},fid
        label_bindings[fid]=dict(relative_path=expected['path'],**observed)
    save(out/'PROTOCOL.json',dict(schema='fixed_source_CAL128_virtual_to_canonical_reference_audit_v26',
        inputs=before,labels=label_bindings,selection='all fixed source CAL128 frames and all saved573 certified pairs across3 arms',
        phase='Original source geometry rule: dimensions by perm_v4, eight unique nearest annotation UV assignments, existing max error <.05px assertion.',
        purpose='Separate virtual-wire-reference vs canonical-corner mismatch from head query/line errors; not a real pose accuracy experiment.',
        new_synthetic_known_pose_projection_calls=128,new_projected_model_points=1024,
        new_PnP=0,new_line_fit=0,new_model=0,new_training=0,new_RGB=0,new_real_pose_accuracy=0,
        thresholds_changed=False,physical_real_ownership_certified=False))
    frames=[];lookup={};max_phase_error=0.
    for fid,row in ready.items():
        annotation=json.loads(label_paths[fid].read_text());o=annotation['objects'][0];intr=annotation['camera_data']['intrinsics']
        K=[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]]
        T=o['pose_transform'];R=[v[:3] for v in T[:3]];t=[v[3] for v in T[:3]]
        assert K==row['K'] and R==row['R'] and t==row['t']
        dims=[o['dimensions_m'][key] for key in ('width','height','depth')]
        if frozenset(o['perm_v4'][:2]) not in {frozenset(EDGES[i]) for i in (0,2,4,6)}:dims=[dims[2],dims[1],dims[0]]
        assert dims==row['dimensions']
        xyz=cuboid(dims);uv=[project(x,R,t,K) for x in xyz];annotated=o['projected_cuboid'];assert len(annotated)==8
        ids=[min(range(8),key=lambda k:math.dist(q,uv[k])) for q in annotated]
        assert len(set(ids))==8
        errors=[math.dist(q,uv[k]) for q,k in zip(annotated,ids)];assert max(errors)<.05
        max_phase_error=max(max_phase_error,max(errors))
        frame=dict(id=fid,index=row['index'],source_annotation_binding=label_bindings[fid],dimensions=dims,
            existing_synthetic_R=R,existing_synthetic_t=t,K=K,canonical_unordered_xyz=xyz,
            annotation_corner_xy=annotated,native_corner_phase_indices=ids,
            projected_native_corner_xy=[uv[k] for k in ids],source_phase_error_px=errors,
            source_only=True,physical_corner_ownership='UNKNOWN',native_N3_predicted_H_reconstructed=False)
        frames.append(frame);lookup[fid]=frame
    corners=read(paths['saved_corners']);assert len(corners)==573
    rows=[]
    for r in corners:
        frame=lookup[r['id']];gt=frame['projected_native_corner_xy'][r['corner']]
        annotation=frame['annotation_corner_xy'][r['corner']];source=ready[r['id']]
        supports=[]
        for component in r['line_components']:
            edge=component['edge'];aa,bb=EDGES[edge];A,B=[frame['projected_native_corner_xy'][j] for j in (aa,bb)]
            tangent=[x-y for x,y in zip(B,A)];length=math.sqrt(dot(tangent,tangent));assert length>0
            normal=[-tangent[1]/length,tangent[0]/length];offset=dot(normal,A)
            target=[source['queries'][j]['actual_target_uv'] for j in component['queries']]
            assert all(source['queries'][j]['proposed_target']=='POSITIVE' for j in component['queries'])
            residual=[dot(normal,q)-offset for q in target]
            supports.append(dict(edge=edge,queries=component['queries'],actual_target_uv=target,
                canonical_edge_endpoint_xy=[A,B],normal=normal,offset=offset,
                certified_support_canonical_line_residual_px=residual))
        mismatch=math.dist(r['reference_xy'],gt)
        new_error=math.dist(r['predicted_xy'],gt);assert abs(new_error-r['saved_corner_error_px'])<=mismatch+1e-8
        rows.append(dict(arm=r['arm'],id=r['id'],corner=r['corner'],edges=r['edges'],
            saved_wire_reference_xy=r['reference_xy'],existing_source_canonical_projection_xy=gt,
            existing_annotation_corner_xy=annotation,reference_to_canonical_px=mismatch,
            reference_to_annotation_px=math.dist(r['reference_xy'],annotation),
            predicted_xy=r['predicted_xy'],saved_error_to_wire_reference_px=r['saved_corner_error_px'],
            error_to_existing_source_canonical_px=new_error,
            error_change_abs_px=abs(new_error-r['saved_corner_error_px']),
            decoder_selected=r['decoder_selected'],existing_point_admission_accepted=r['existing_point_admission_accepted'],
            recorded_radius_px=r['recorded_radius_px'],support_witnesses=supports,
            physical_corner_ownership='UNKNOWN',source_native_N3_H_LOO='NOT_EVALUATED',real_pose_accuracy_evaluated=False))
    write_rows(out/'FRAME_ROWS.jsonl.gz',frames);write_rows(out/'CORNER_ROWS.jsonl.gz',rows)
    counts=collections.Counter();groups={}
    for arm in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'):
        rr=[r for r in rows if r['arm']==arm]
        groups[arm]=dict(reference_to_canonical_px=stats([r['reference_to_canonical_px'] for r in rr]),
            reference_to_annotation_px=stats([r['reference_to_annotation_px'] for r in rr]),
            reference_error_change_abs_px=stats([r['error_change_abs_px'] for r in rr]),
            canonical_line_support_abs_residual_px=stats([abs(v) for r in rr for s in r['support_witnesses'] for v in s['certified_support_canonical_line_residual_px']]),
            old_reference_gt8_count=sum(r['saved_error_to_wire_reference_px']>8 for r in rr),
            canonical_reference_gt8_count=sum(r['error_to_existing_source_canonical_px']>8 for r in rr))
    assert before=={k:dict(path=str(v),bytes=v.stat().st_size,sha256=sha(v)) for k,v in paths.items()}
    assert all(sha(p)==label_bindings[fid]['sha256'] for fid,p in label_paths.items())
    four=[r for r in rows if r['arm']=='IMAGE_ROLE' and r['decoder_selected'] and r['saved_error_to_wire_reference_px']>8]
    assert len(four)==4
    save(out/'SUMMARY.json',dict(passed=True,frames=128,corners=573,groups=groups,
        max_existing_annotation_phase_error_px=max_phase_error,four_known_cases=four,
        physical_real_ownership_proved=False,source_predicted_H_or_N3_invented=False,
        new_synthetic_known_pose_projection_calls=128,new_PnP=0,new_line_fit=0,new_real_pose_accuracy=0,new_training=0))
    save(out/'CHECKS.json',dict(passed=True,all128_original_labels_SHA_bytes_verified=True,
        all573_pairs_included=True,source_phase_same_existing_rule=True,all_inputs_unchanged=True,
        source_GT_only=True,real_GT_read=False,new_synthetic_known_pose_projection_calls=128,
        new_PnP=0,new_line_fit=0,new_model=0,new_training=0,new_RGB=0,
        output_hashes={name:sha(out/name) for name in ('FRAME_ROWS.jsonl.gz','CORNER_ROWS.jsonl.gz')}))
    print(json.dumps(dict(passed=True,groups=groups,max_phase_error_px=max_phase_error,
        four=[dict(id=r['id'],corner=r['corner'],reference_gap_px=r['reference_to_canonical_px'],canonical_error_px=r['error_to_existing_source_canonical_px']) for r in four])))


if __name__=='__main__':main()
