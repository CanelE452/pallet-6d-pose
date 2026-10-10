"""Frozen source-only projection sensitivity audit; no fit, renderer or model.

Consumes existing public source-test numeric rows. Source R/t are the location
at which to evaluate a derivative, never an optimizer start or deployment prior.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
PHYSICAL = {0,2,4,6,8,9,10,11}
AXIS_X = {0,2,4,6}
AXIS_Z = {8,9,10,11}
STEP = 1e-5
RANK_RELATIVE = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)


def exp_rotation(v):
    theta = np.linalg.norm(v)
    if theta == 0:
        return np.eye(3)
    A = skew(v)
    return np.eye(3) + np.sin(theta)/theta*A + (1-np.cos(theta))/theta**2*(A@A)


def project(X, R, t, K):
    camera = X@R.T+t
    homogeneous = camera@K.T
    return homogeneous[:,:2]/homogeneous[:,2,None]


def source_endpoints(row):
    """Recover exact annotated cuboid phase from public numeric query records.

    Stored image fraction interpolates projected endpoints linearly; reciprocal
    camera depth also interpolates linearly. The reconstruction uses source
    position/fraction/Z only, then applies stored K/R/t inverse algebra.
    """
    K,R,t = (np.asarray(row[k],float) for k in ('K','R','t'))
    corners = {}
    disagreement = []
    for edge, (a,b) in enumerate(EDGES):
        queries = [q for q in row['queries'] if q['edge']==edge and q.get('ray_eligible') and q.get('source_X') is not None]
        if len(queries)<2:
            continue
        left,right = queries[0],queries[-1]
        fa,fb = left['fraction'],right['fraction']
        df = fb-fa
        if abs(df)<1e-10:
            continue
        pa,pb = np.asarray(left['position']),np.asarray(right['position'])
        uv0,uv1 = (fb*pa-fa*pb)/df,((1-fa)*pb-(1-fb)*pa)/df
        ia,ib = 1/left['source_camera_Z'],1/right['source_camera_Z']
        iz0,iz1 = (fb*ia-fa*ib)/df,((1-fa)*ib-(1-fb)*ia)/df
        for corner,uv,iz in zip((a,b),(uv0,uv1),(iz0,iz1)):
            camera = np.linalg.solve(K,np.r_[uv,1.])/iz
            point = R.T@(camera-t)
            if corner in corners:
                disagreement.append(float(np.linalg.norm(point-corners[corner])))
            corners[corner] = point
    if set(corners)!=set(range(8)):
        raise ValueError('Stored source queries do not reconstruct all 8 endpoints')
    X = np.asarray([corners[i] for i in range(8)])
    query_errors = []
    for q in row['queries']:
        if not q.get('ray_eligible') or q.get('source_X') is None:
            continue
        a,b = EDGES[q['edge']]
        za,zb = (X[[a,b]]@R.T+t)[:,2]
        f = q['fraction']
        u = (f/zb)/((1-f)/za+f/zb)
        query_errors.append(float(np.linalg.norm((1-u)*X[a]+u*X[b]-q['source_X'])))
    return X, dict(max_repeated_endpoint_disagreement_m=max(disagreement,default=0),
        max_retained_source_X_reconstruction_error_m=max(query_errors,default=0),
        endpoints_reconstructed=8, new_pose_solves=0)


def truth_line(X,R,t,K,edge):
    uv = project(X,R,t,K)[list(EDGES[edge])]
    d = uv[1]-uv[0]
    if np.linalg.norm(d)<1e-12:
        raise ValueError('Degenerate projected source line')
    normal = np.array([-d[1],d[0]])/np.linalg.norm(d)
    return dict(edge=edge,normal=normal,offset=float(normal@uv.mean(0)))


def residual_factory(X,R,t,K,points,lines,translation_scale):
    def residual(z):
        q = project(X,exp_rotation(z[:3])@R,t+z[3:]*translation_scale,K)
        parts = [(q[c['id']]-c['xy']).ravel() for c in points]
        for line in lines:
            normal = np.asarray(line['normal'],float)
            normal = normal/np.linalg.norm(normal)
            parts.append((q[list(EDGES[line['edge']])]@normal-line['offset'])/np.sqrt(2))
        return np.concatenate(parts) if parts else np.empty(0)
    return residual


def analytical_jacobian(X,R,t,K,points,lines,translation_scale):
    camera = X@R.T+t
    homogeneous = camera@K.T
    Js = []
    for i in range(8):
        z = homogeneous[i,2]
        projection = np.array([(K[j]*z-homogeneous[i,j]*K[2])/z**2 for j in (0,1)])
        Js.append(projection@np.column_stack([-skew(R@X[i]),np.eye(3)*translation_scale]))
    rows = [Js[c['id']] for c in points]
    for line in lines:
        normal = np.asarray(line['normal'],float)
        normal = normal/np.linalg.norm(normal)
        rows.append(np.array([normal@Js[i]/np.sqrt(2) for i in EDGES[line['edge']]]))
    return np.vstack(rows) if rows else np.empty((0,6))


def sensitivity(X,R,t,K,points,lines):
    scale = float(np.linalg.norm(np.ptp(X,axis=0)))
    residual = residual_factory(X,R,t,K,points,lines,scale)
    z = np.zeros(6)
    baseline = residual(z)
    J = np.empty((len(baseline),6))
    for j in range(6):
        delta = np.zeros(6);delta[j] = STEP
        J[:,j] = (residual(delta)-residual(-delta))/(2*STEP)
    analytical = analytical_jacobian(X,R,t,K,points,lines,scale)
    relative = float(np.linalg.norm(J-analytical)/max(np.linalg.norm(analytical),1e-300))
    lengths = np.linalg.norm(J,axis=0)
    normalized = J/np.maximum(lengths,1e-300)
    singular = np.linalg.svd(normalized,compute_uv=False)
    rank = int((singular>singular[0]*RANK_RELATIVE).sum()) if len(singular) and singular[0]>0 else 0
    analytical_norm = analytical/np.maximum(np.linalg.norm(analytical,axis=0),1e-300)
    exact_singular = np.linalg.svd(analytical_norm,compute_uv=False)
    exact_rank = int((exact_singular>exact_singular[0]*RANK_RELATIVE).sum()) if len(exact_singular) and exact_singular[0]>0 else 0
    return dict(residual_dimensions=len(baseline), source_residual_vector_px=baseline.tolist(),
        source_residual_rms_px=float(np.sqrt(np.mean(baseline**2))) if len(baseline) else None,
        source_residual_max_abs_px=float(np.abs(baseline).max()) if len(baseline) else None,
        finite_difference_step=STEP, translation_parameter_scale_m=scale,
        finite_difference_column_norms=lengths.tolist(), normalized_singular_values=singular.tolist(),
        normalized_rank=rank, normalized_condition=float(singular[0]/singular[-1]) if rank==6 else None,
        normalized_singular_ratio_min_max=float(singular[-1]/singular[0]) if len(singular) and singular[0]>0 else None,
        analytical_normalized_singular_values=exact_singular.tolist(),analytical_normalized_rank=exact_rank,
        finite_difference_analytical_relative_Frobenius_error=relative,
        local_linearization_only=True, global_uniqueness_certified=False)


def query_quantization(source,row):
    pts = np.asarray(source['frozen_selected_points'],float)
    errors=[]
    for q in row['queries']:
        i = q['query']
        if q['cached_target']!='POSITIVE':
            continue
        a,b = EDGES[q['edge']]
        center = (1-(i%7+1)/8)*pts[a]+((i%7+1)/8)*pts[b]
        selected = center+(source['ideal_choices'][i]-32)*np.asarray(q['normal'])
        errors.append(float(np.linalg.norm(selected-q['position'])))
    return dict(positive_queries=len(errors), max_selected_MAP_distance_from_stored_source_intersection_px=max(errors,default=0),
        count_above_half_pixel_plus_frozen_roundoff=sum(x>.5001 for x in errors),
        quantization_roundoff_tolerance_px=.0001,
        actual_RGB_boundary_ownership_validated=False)


def toy_checks():
    X=np.array([[-.55,-.07,-.65],[.55,-.07,-.65],[.55,.07,-.65],[-.55,.07,-.65],[-.55,-.07,.65],[.55,-.07,.65],[.55,.07,.65],[-.55,.07,.65]])
    R=exp_rotation(np.array([.3,.2,.1]));t=np.array([.1,-.1,3.]);K=np.array([[600.,0,320.],[0,600.,240.],[0,0,1.]])
    lines={e:truth_line(X,R,t,K,e) for e in PHYSICAL};uv=project(X,R,t,K)
    cases={}
    for key,points,edges,expected in [
        ('three_rectangle_edges',[],[0,8,9],6),
        ('same_sources_consumed_as_two_corners',[dict(id=i,xy=uv[i]) for i in (0,1)],[],4),
        ('three_parallel_true_edges',[],[0,2,4],5),
        ('full_true_rectangle',[],[0,4,8,9],6)]:
        result=sensitivity(X,R,t,K,points,[lines[e] for e in edges])
        if result['normalized_rank']!=expected or result['analytical_normalized_rank']!=expected:
            raise AssertionError((key,result,expected))
        cases[key]=dict(expected_rank=expected,finite_difference_rank=result['normalized_rank'],
            analytical_rank=result['analytical_normalized_rank'],normalized_singular_values=result['normalized_singular_values'],
            derivative_relative_error=result['finite_difference_analytical_relative_Frobenius_error'])
    return dict(passed=True,cases=cases,optimizer_calls=0)


def run(protocol_path):
    begin=time.monotonic();protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text())
    output=protocol_path.parent
    if sha(__file__)!=protocol['code']['sha256']:
        raise ValueError('Frozen code mismatch')
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Frozen source numeric input mismatch')
    sources={r['index']:r for r in read_rows(protocol['inputs'][0]['path'])}
    rays=read_rows(protocol['inputs'][1]['path'])
    if len(rays)!=128 or [r['index'] for r in rays]!=list(range(896,1024)):
        raise ValueError('Frozen 128 source-test population changed')
    checks=toy_checks();records=[];totals=Counter()
    for row in rays:
        source=sources[row['index']]
        if source['partition']!='source_test' or source['id']!=row['id']:
            raise ValueError('Source identity mismatch')
        X,reconstruction=source_endpoints(row)
        R,t,K=(np.asarray(row[k],float) for k in ('R','t','K'))
        lines=source['ideal']['lines'];edges={l['edge'] for l in lines}
        if not edges<=PHYSICAL:
            raise ValueError('Unsupported source physical edge present')
        inside=set(source['before_mask_in_frame_corner_ids']);H=set(source['source_annotation_oracle_H'])
        points=[c for c in source['ideal']['corners'] if c['id'] in inside and c['id'] not in H]
        consumed={e for c in points for e in c['edges']};remaining=[l for l in lines if l['edge'] not in consumed]
        exact_lines={l['edge']:truth_line(X,R,t,K,l['edge']) for l in lines}
        exact_points=[dict(id=c['id'],xy=project(X,R,t,K)[c['id']]) for c in points]
        paths=dict(current_corner_and_remaining_line=sensitivity(X,R,t,K,points,remaining),
            all_original_lines_once=sensitivity(X,R,t,K,[],lines),
            current_factor_topology_with_exact_source_lines=sensitivity(X,R,t,K,exact_points,[exact_lines[l['edge']] for l in remaining]),
            all_lines_once_with_exact_source_geometry=sensitivity(X,R,t,K,[],list(exact_lines.values())))
        directions=[axis for axis,es in [('X',AXIS_X),('Z',AXIS_Z)] if edges&es]
        structural=len(edges)>=3 and len(directions)==2
        local=structural and paths['all_lines_once_with_exact_source_geometry']['normalized_rank']==6 and paths['all_lines_once_with_exact_source_geometry']['analytical_normalized_rank']==6
        quantization=query_quantization(source,row)
        positive_line_query_ids=sorted({i for line in lines for i in line['queries']})
        positive_line_queries=[row['queries'][i] for i in positive_line_query_ids]
        definition_valid=all(q['cached_target']=='POSITIVE' and q['cached_label_reproduced'] and q.get('actual_mesh_point_within_original_tolerance') and q.get('in_image') and q.get('supplied_visible_mask_support_3x3') and q.get('ray_source_depth_agreement',[False])[0] for q in positive_line_queries)
        records.append(dict(id=row['id'],index=row['index'],partition='source_test',source_ceiling_row_sha256=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            source_ray_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            source_K=K.tolist(),source_R=R.tolist(),source_t=t.tolist(),reconstructed_native_X=X.tolist(),reconstruction=reconstruction,
            ideal_selected_line_edges=sorted(edges),three_dimensional_edge_direction_classes=directions,
            original_in_frame_corner_ids=sorted(inside),source_annotation_proxy_hidden_ids=sorted(H),
            retained_corner_ids=[c['id'] for c in points],consumed_edges=sorted(consumed),remaining_line_edges=[l['edge'] for l in remaining],
            source_predicted_initial_hidden_ids='NOT_RETAINED; annotation proxy is separately identified',
            residual_definitions='corners: 2D displacement; each line once: two known 3D endpoint-to-observed-line distances/sqrt(2); no line-source point factor in all-lines path',
            factor_paths=paths,quantization=quantization,
            accepted_line_queries_meet_existing_target_definition=definition_valid,
            accepted_query_boundary_ownership_status='UNVALIDATED: mask occupancy + mesh source-depth agreement are not boundary ownership proof',
            gate=dict(structural_at_least3_edges_and_2_directions=structural,
                exact_geometry_local_six_dof_sensitivity=local,
                correctness_gate='UNRESOLVED_BOUNDARY_OWNERSHIP_AND_NONE_LABEL_POLICY',
                supervised_repair_learning_ready=False,pose_success_claim=False,global_unique_pose_claim=False)))
        totals['frames']+=1;totals['in_frame_ge4_corners']+=len(inside)>=4;totals['ge3_edges_two_directions']+=structural
        totals['current_factor_finite_difference_rank6']+=paths['current_corner_and_remaining_line']['normalized_rank']==6
        totals['all_original_lines_finite_difference_rank6']+=paths['all_original_lines_once']['normalized_rank']==6
        totals['all_exact_source_lines_finite_difference_rank6']+=paths['all_lines_once_with_exact_source_geometry']['normalized_rank']==6
        totals['all_exact_source_lines_analytical_rank6']+=paths['all_lines_once_with_exact_source_geometry']['analytical_normalized_rank']==6
        totals['exact_local_sensitivity_and_structural_gate']+=local
        totals['line_exact_rank6_current_rank_lt6']+=local and paths['current_corner_and_remaining_line']['normalized_rank']<6
        totals['all_original_lines_rank6_but_only_one_3D_direction']+=paths['all_original_lines_once']['normalized_rank']==6 and len(directions)<2
        totals['existing_target_definition_valid_frames']+=definition_valid
        totals['quantization_above_half_pixel_plus_roundoff_queries']+=quantization['count_above_half_pixel_plus_frozen_roundoff']
    records_path=output/'SOURCE_OBSERVATION_GATE_ROWS.jsonl.gz'
    if records_path.exists():
        raise FileExistsError(records_path)
    with records_path.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as zipped:
            for row in records:
                zipped.write((json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
    checks_path=output/'SOURCE_OBSERVATION_GATE_MATH_CHECKS.json';write(checks_path,checks)
    summary_path=output/'SOURCE_OBSERVATION_GATE.json'
    summary=dict(schema='source_only_observation_factor_sensitivity_v1',complete=True,counts=dict(totals),
        factor_rank_histograms={key:dict(Counter(r['factor_paths'][key]['normalized_rank'] for r in records)) for key in records[0]['factor_paths']},
        max_finite_difference_analytical_relative_error=max(p['finite_difference_analytical_relative_Frobenius_error'] for r in records for p in r['factor_paths'].values()),
        observation_sufficiency='Local derivative sensitivity only. Rank6 is not global uniqueness, correct observed edge ownership, robust consensus or pose availability.',
        correctness='Existing accepted positive labels reproduce their existing mesh/depth/mask definition. NONE labels and boundary ownership remain unresolved; no relabeling, repaired target or supervised learning follows this audit.',
        source_predicted_mask='not retained; source annotation cuboid H used only as marked source diagnostic, not deployed initial H',
        all_line_path='each original physical edge contributes one line factor, no derived corner factor; endpoints are model geometry, not observed hidden 2D points',
        protocol=binding(protocol_path),code=binding(__file__),rows=binding(records_path),math_checks=binding(checks_path),
        actual_calls=dict(PnP=0,optimizers=0,detectors=0,heads=0,training_updates=0,rays=0,RGB_reads=0,real_GT_reads=0),
        elapsed_seconds=time.monotonic()-begin)
    write(summary_path,summary)
    write(output/'SOURCE_OBSERVATION_GATE_RECEIPT.json',dict(schema='source_only_math_gate_receipt_v1',complete=True,
        inputs_before=protocol['inputs'],inputs_after=[binding(p['path']) for p in protocol['inputs']],
        protocol=binding(protocol_path),code=binding(__file__),summary=binding(summary_path),rows=binding(records_path),
        math_checks=binding(checks_path),code_frozen_before_numeric_work=True,
        derivative_operations_only=True,no_public_files_written=True,
        no_optimization_or_pose_hypothesis_generation=True,elapsed_seconds=time.monotonic()-begin))
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    args=parser.parse_args();run(args.protocol)
