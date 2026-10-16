"""Read-only source/bin/actual-mesh contract arithmetic, without any models.

This reproduces the earlier stdout-only arithmetic audit. It verifies the
existing READY labels and cached geometry channels, not real boundary ownership.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_corner_mechanism_audit_20261010_v1'
REPAIR=REPO/'_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
PRIVATE=Path('/dev/shm/pallet-observation-private-20261009')
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def binding(path):
    path=Path(path).resolve();public=path.is_relative_to(REPO)
    return dict(path=str(path.relative_to(REPO)) if public else path.name,
                origin='public_repository' if public else 'private_readonly_dependency',
                sha256=sha(path),bytes=path.stat().st_size)


def read(path):return json.loads(Path(path).read_text())


def edge_definition(path):
    # Read the literal graph; do not import Torch/model/renderer modules.
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id=='EDGES' for t in n.targets)]
    assert len(nodes)==1
    return [tuple(e) for e in ast.literal_eval(nodes[0].value)]


def run(args):
    begin=time.monotonic();output=Path(args.output)
    assert not output.exists() and not output.is_symlink(),'Preserve completed receipt'
    assert args.source_root and args.baseline_root,'Supply explicit source/baseline roots or PALLET environment variables'
    paths=dict(ready_rows=REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz',
        ready_arrays=REPAIR/'READY_PREPARED_TARGETS.npz',features=Path(args.features),
        cache_manifest=Path(args.cache_manifest),actual_mesh=Path(args.mesh),actual_mesh_metadata=Path(args.mesh).with_suffix('.json'),
        source_audit=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py',
        model=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
        observations=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py',
        actual_wire_rule=REPO/'scripts/research/pallet_kp_supervision_gate_20261010_v1/actual_wire_proposal.py',
        source_feature_preparation=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/training.py',
        live_feature_call=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        feature_extractor_source=Path(args.source_root)/'scripts/research/pallet_line_pose_v1/features.py',
        feature_extractor_baseline=Path(args.baseline_root)/'scripts/research/pallet_line_pose_v1/features.py',
        wire_face_witnesses=REPO/'_docs/experiments/pallet_kp_supervision_gate_20261010_v1/WIRE_FACE_WITNESSES.json',code=Path(__file__))
    before={k:binding(p) for k,p in paths.items()}
    assert before['actual_mesh']['sha256']==read(paths['wire_face_witnesses'])['mesh_sha256']
    assert before['feature_extractor_source']['sha256']==before['feature_extractor_baseline']['sha256']
    assert edge_definition(paths['source_audit'])==edge_definition(paths['observations'])==edge_definition(paths['actual_wire_rule'])==EDGES
    with gzip.open(paths['ready_rows'],'rt') as f:rows=[json.loads(x) for x in f]
    with np.load(paths['actual_mesh']) as z:vertices=z['vertices'];triangles=z['triangles']
    with np.load(paths['ready_arrays']) as z:targets={k:z[k] for k in ('lo','hi','weight','valid','source_index','partitions')}
    features=np.load(paths['features'],mmap_mode='r');manifest=read(paths['cache_manifest'])
    assert features.shape==(1024,84,28,65) and features.dtype==np.float16
    assert len(rows)==len(manifest['records'])==1024
    assert np.array_equal(targets['source_index'],np.arange(1024))
    assert Counter(r['partition'] for r in rows)==dict(train=768,calibration=128,source_test=128)
    counts=dict(frames=0,queries=0,positive=0,cache_frame_partition_or_selected_points_mismatch=0,
        edge_query_order_mismatch=0,bin_array_mismatch=0,wire_face_shared_endpoint_missing=0,
        geometry_feature_centerbin_mismatch=0,geometry_feature_offset_bins_mismatch=0)
    maxima=dict(target_uv_vs_bin_px=0.,actual_point_projection_px=0.,wire_point_distance_m=0.,
        wire_direction_relative_residual=0.,triangle_noncoplanarity_min=1.,wire_fixed_axes_gap_m=0.)
    wireids=set();positive_edges=set();examples=[]
    for row in rows:
        i=row['index'];q=np.array(row['frozen_selected_points']);h,w=row['raw_hw']
        R=np.array(row['R']);t=np.array(row['t']);K=np.array(row['K']);dim=np.array(row['dimensions']);m=manifest['records'][i]
        counts['frames']+=1
        if (row['id']!=m['id'] or row['partition']!=m['partition'] or row['partition']!=targets['partitions'][i] or
                not np.array_equal(q,np.array(m['selected_points']))):counts['cache_frame_partition_or_selected_points_mismatch']+=1
        for j,r in enumerate(row['queries']):
            counts['queries']+=1;e=j//7;a,b=EDGES[e];u=(j%7+1)/8
            d=q[b]-q[a];L=np.linalg.norm(d);tangent=d/L;normal=np.array([-tangent[1],tangent[0]])
            center=(1-u)*q[a]+u*q[b]
            if r['query']!=j or r['edge']!=e:counts['edge_query_order_mismatch']+=1
            expected=np.array([0.,center[0]/w,center[1]/h,*tangent,L/np.hypot(w,h)],np.float32).astype(np.float16)
            if not np.array_equal(features[i,j,19:25,32],expected):counts['geometry_feature_centerbin_mismatch']+=1
            if not np.array_equal(features[i,j,19,:],(np.arange(-32,33)/32).astype(np.float16)):counts['geometry_feature_offset_bins_mismatch']+=1
            if r['proposed_target']!='POSITIVE':continue
            counts['positive']+=1;positive_edges.add(e);wireids.add(r['actual_wire_id'])
            if (int(targets['lo'][i,j])!=r['lo'] or int(targets['hi'][i,j])!=r['hi'] or
                targets['weight'][i,j]!=np.float32(r['weight']) or not targets['valid'][i,j]):counts['bin_array_mismatch']+=1
            P=np.array(r['actual_point']);uv=np.array(r['actual_target_uv']);uvbin=center+(r['lo']+r['weight']-32)*normal
            maxima['target_uv_vs_bin_px']=max(maxima['target_uv_vs_bin_px'],float(np.linalg.norm(uv-uvbin)))
            camera=R@P+t;projected=(K@camera)[:2]/camera[2]
            maxima['actual_point_projection_px']=max(maxima['actual_point_projection_px'],float(np.linalg.norm(uv-projected)))
            A,B=vertices[r['actual_endpoint_vertex_ids']]*dim;direction=B-A;fraction=(P-A)@direction/(direction@direction)
            gap=np.linalg.norm(P-(A+fraction*direction));maxima['wire_point_distance_m']=max(maxima['wire_point_distance_m'],float(gap))
            assert -1e-12<=fraction<=1+1e-12
            axis=np.argmax(np.abs(direction));fixed=[k for k in range(3) if k!=axis]
            maxima['wire_direction_relative_residual']=max(maxima['wire_direction_relative_residual'],float(np.linalg.norm(direction[fixed])/np.linalg.norm(direction)))
            maxima['wire_fixed_axes_gap_m']=max(maxima['wire_fixed_axes_gap_m'],float(max(np.linalg.norm(A[fixed]-P[fixed]),np.linalg.norm(B[fixed]-P[fixed]))))
            normals=[]
            for face in r['actual_face_ids']:
                verts=vertices[triangles[face]]*dim
                if not all(np.any(np.all(verts==endpoint,axis=1)) for endpoint in [A,B]):counts['wire_face_shared_endpoint_missing']+=1
                n=np.cross(verts[1]-verts[0],verts[2]-verts[0]);normals.append(n/np.linalg.norm(n))
            cross=float(np.linalg.norm(np.cross(*normals)))
            maxima['triangle_noncoplanarity_min']=min(maxima['triangle_noncoplanarity_min'],cross)
            if len(examples)<2:examples.append(dict(index=i,id=row['id'],query=j,edge=e,wire=r['actual_wire_id'],
                target_uv=uv.tolist(),reconstructed_uv=uvbin.tolist(),actual_face_ids=r['actual_face_ids']))
    assert counts['frames']==1024 and counts['queries']==86016 and counts['positive']==35011
    assert all(v==0 for k,v in counts.items() if k.endswith('mismatch') or k=='wire_face_shared_endpoint_missing')
    assert maxima['target_uv_vs_bin_px']<1e-9 and maxima['actual_point_projection_px']<1e-9
    assert maxima['wire_point_distance_m']<1e-12 and maxima['triangle_noncoplanarity_min']>.999999
    after={k:binding(p) for k,p in paths.items()};assert before==after,'Inputs changed'
    result=dict(schema='existing_source_bin_actual_mesh_contract_audit_v1',passed=True,counts=counts,maxima=maxima,
        positive_edges=sorted(positive_edges),distinct_positive_wires=len(wireids),examples=examples,
        authoritative_inputs=before,inputs_before_after_equal=True,
        private_access='Only existing FP16 feature/cache manifest and normalized actual-mesh cache/metadata; paths omitted from receipt',
        GT_used_only_for_source_audit=True,real_reference_or_pose_scores_read=False,source_annotation_loader_called=False,
        coordinate_definition='native initial Base edge a,b; center=(1-u)qa+u qb; normal=(-ty,tx); bin b means center+(b-32)normal in original RGB pixels',
        ownership_definition='stored actual-mesh wire endpoints must both belong to each of the two recorded incident triangles; noncoplanar normals checked from actual triangles',
        geometry_feature_definition='original FP16 channels19..24 at zero-offset and all65 fixed offset-channel bins; no image-channel or neck regeneration',
        static_RGB_feature_contract=dict(extractor_source_and_baseline_byte_identical=True,
            source_RGB_decode='training.py uses cv2.imread(raw source original); real pipeline.py uses cv2.imread(...,IMREAD_COLOR)',
            shared_function='source training.py and live pipeline.py both call the same original model.inputs with the same Base-coordinate query anchor',
            grayscale_stencil='model.inputs BGR2GRAY,gray/255,Sobel/4; INTER_LINEAR remap with BORDER_REFLECT_101, brightness and normal/tangent gradients',
            canvas='byte-identical FrozenYoloFeatures.predict uses added_border100 reflect,YOLO640rect; shared canvas_affine gain+integerleft/top offset',
            neck_grid='(candidate+added_border)*gain+offset; grid=uv/640*2-1; padded P3/P4 grouping and grid_sample align_corners=False',
            numeric_rounding='both original source cache and live model.inputs round full candidate tensor FP16 then compute head FP32',
            actual_image_and_neck_channels_recomputed=False,
            static_contract_only=True,limit='No fresh19image/neck-channel numerical parity check or same-RGB replay; sixgeometrychannels do not certify those19channels'),
        prior_stdout_only_arithmetic_runs=1,reproducible_script_runs=1,total_actual_readonly_arithmetic_runs=2,
        copied_prior_actual_run=False,
        prior_run_note='The earlier stdout-only audit covered these35011POSITIVEs/1024frames; this receipt is one separate fresh source-contract run with byte/SHA guards and unchanged numerical definitions.',
        limits=['No inference, source label regeneration, ray, PnP, source split change, or performance/configuration sweep occurred',
            'Cache phase check is frame/partition/selected-point identity; native source annotation phase independently checked by the separate solver audit',
            'Wire fixed-axis residual is axis alignment, not proof of a complete physical corner or all real semantic edge ownership',
            'Source mesh/RGB agreement and visibility proof are inherited from earlier immutable audits; RGB/masks not regenerated or reopened here',
            'This shows no coordinate/index/actualwire-face mismatch in inspected existing source records; it does not assign a causal fraction of real pose harm',
            'Actual-wire intersections may be virtual native corners; real ownership/cross-edge correlation remains unproved by these arithmetic checks'],
        new_detector_forwards=0,new_head_forwards=0,new_rays=0,new_PnP=0,new_training_updates=0,new_images=0,
        elapsed_seconds=time.monotonic()-begin)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,counts=counts,maxima=maxima,output=binding(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',default=str(PRIVATE/'learned_cache/features.npy'))
    p.add_argument('--cache-manifest',default=str(PRIVATE/'learned_cache/CACHE_MANIFEST.json'))
    p.add_argument('--mesh',default=str(PRIVATE/'actual_mesh/scene.usd.npz'))
    p.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT'))
    p.add_argument('--baseline-root',default=os.environ.get('PALLET_BASELINE_ROOT'))
    p.add_argument('--output',default=str(DOC/'SOURCE_CONTRACT_CHECKS.json'))
    run(p.parse_args())
