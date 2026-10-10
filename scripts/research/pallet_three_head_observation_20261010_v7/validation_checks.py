"""Supplemental public-only scalar audit of the frozen V7 point-only paths.

Freeze AFTER complete GT-free geometry, then run once.  No production module,
NumPy, Torch, image, GT cache, solver, optimizer, Jacobian or SVD is imported.
All numeric candidates are streamed rather than discarded from the witnesses.
Python >= 3.9.  This is stored-data verification, not an accuracy experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE = Path('/tmp/pallet-three-head-observation-private-20261010-v7')
ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
METHODS = ('IMAGE_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_BOUNDARY_ONLY',
           'IMAGE_NO_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_CORNERWISE_HYBRID',
           'IMAGE_NO_ROLE_CORNERWISE_HYBRID', 'IMAGE_ROLE_CORNERWISE_HYBRID',
           'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7))
CHECKPOINT_SHA = dict(
    GEOMETRY_ONLY='d188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b',
    IMAGE_NO_ROLE='9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835',
    IMAGE_ROLE='882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d')
FALSE_FLAGS = ('prior_used','initial_pose_used','initial_projection_used',
               'initial_dimension_prior_used','known_dimension_constraint_used',
               'excluded_image_coordinates_used_for_scoring',
               'excluded_image_coordinates_used_for_equivalence',
               'reprojected_points_reused_as_observations')
ATOL, RTOL, RADIUS = 1e-7, 1e-12, 8.0
OUTPUTS = ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
MODES = dict(GEOMETRY_ONLY='zero channels0:19; retain geometry19:25 and role25:28',
    IMAGE_NO_ROLE='zero role25:28; retain image/neck0:19 and geometry19:25',
    IMAGE_ROLE='unchanged all28 channels')
ROLE_FILES = ('CALIBRATION_PROTOCOL.json','CALIBRATION_START.json','CALIBRATION_ROWS.jsonl.gz',
    'CALIBRATION_THRESHOLD_SCAN.jsonl.gz','CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz',
    'CALIBRATION_GEOMETRY_ROWS.jsonl.gz','CALIBRATION.json','CALIBRATION_EXECUTION.json','CALIBRATION_CHECKS.json')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    absolute = path.resolve()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
                sha256=digest.hexdigest(), bytes=path.stat().st_size)


def bound(path, expected):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256','bytes')), 'binding differs: '+Path(path).name)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def guard(args, stage):
    folder, output = args.input.absolute(), args.output.absolute()
    for path in (folder, output, args.protocol.absolute(), args.calibration_root.absolute()):
        require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink input/output ancestry')
    require(folder.is_dir() and output.is_dir(), 'input and output directories must already exist')
    output = output.resolve()
    require(output == DOC.resolve() or output.is_relative_to(PRIVATE.resolve()), 'only new V7 DOC/private output')
    names = OUTPUTS if stage == 'freeze' else OUTPUTS[1:]
    for name in names:
        path = output/name
        require(not path.exists() and not path.is_symlink(), 'preserve existing '+name)
    return folder.resolve(), output


def inputs(args):
    folder = args.input.resolve()
    paths = {n:folder/n for n in ('OBSERVATIONS.jsonl.gz','GEOMETRY_SEALED.jsonl.gz',
        'FIXED_GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json')}
    paths.update(accuracy_protocol=args.protocol.resolve(), checker_code=Path(__file__).resolve(),
        checkpoint_metadata=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/CHECKPOINT_METADATA.json',
        independent_source_CAL_checks=DOC/'SOURCE_CALIBRATION_CHECKS.json',
        original_ROLE_calibration_code=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py',
        source_CAL_protocol=args.calibration_root.resolve()/'PROTOCOL.json',
        source_CAL_completion=args.calibration_root.resolve()/'COMPLETION.json')
    for arm in ARMS:
        for name in ('CALIBRATION_PROTOCOL.json','CALIBRATION.json','CALIBRATION_EXECUTION.json'):
            paths[arm+':'+name] = args.calibration_root.resolve()/arm/name
    for name in ROLE_FILES:
        paths['IMAGE_ROLE:'+name] = args.calibration_root.resolve()/'IMAGE_ROLE'/name
        paths['original_ROLE:'+name] = REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'/name
    paths['ROLE_reuse'] = args.calibration_root.resolve()/'IMAGE_ROLE/REUSE_RECEIPT.json'
    return paths


def freeze(args):
    folder, output = guard(args,'freeze')
    seal, receipt, core = [read(folder/n) for n in ('GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json')] + [read(args.protocol)]
    require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames']==245 and
            seal['rows']==1960 and seal['fixed_rows']==490 and seal['methods']==list(METHODS), 'complete pre-GT geometry required')
    require(receipt['complete'] and receipt['cleanup_error'] is None and receipt['actual_complete_frames']==245,
            'successful cleanup and complete inference receipt required')
    bound(args.protocol,seal['protocol'])
    require(core['methods']==list(METHODS) and core['frames']==245, 'fixed accuracy policy population')
    for name,key in (('OBSERVATIONS.jsonl.gz','observations'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
                     ('FIXED_GEOMETRY_SEALED.jsonl.gz','fixed_geometry')):
        bound(folder/name,seal[key])
    write(output/OUTPUTS[0],dict(schema='supplemental_v7_point_scalar_validation_protocol_v1',
        checker=binding(Path(__file__)), inputs={k:binding(p) for k,p in inputs(args).items()},
        frozen_after_complete_accuracy_geometry=True, required_scored_rows=False,
        changes_to_frozen_core=0, new_models_PnP_optimizer_GT_image_ray_training_calls=0,
        thresholds=dict(point_consensus_px=8.0,corner_radius_px=8.0,score_tie_px2=1e-8,
                        equivalence_projection_px=1e-5,equivalence_physical_rotation=1e-5,
                        equivalence_translation_relative_diagonal=1e-5,stored_singular_rank_relative=1e-10),
        scalar_absolute_tolerance=ATOL, scalar_relative_tolerance=RTOL,
        candidate_count_scope='Stored all_candidate_solutions records per solve packet; chosen rechecks are counted separately. Neither is a unique physical-solution count.',
        historical_ROLE_execution_N3_calls='Absent from the byte-exact historical receipt; not represented as a recorded zero. Current V7 source CAL N3 calls are recorded separately.',
        rank_scope='Recount stored singular values only; no new Jacobian/SVD or global uniqueness proof',
        numeric_hash_scope='Finite float64 inputs reconstructed; JSON null does not certify original NaN payload',
        no_automatic_retry=True))
    print('SUPPLEMENTAL_VALIDATION_FROZEN',binding(output/OUTPUTS[0])['sha256'],flush=True)


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    return value


class Audit:
    def __init__(self):
        self.checks=0;self.failures=[];self.failure_count=0;self.max_difference=0.0
    def ok(self,value,label,actual=None,expected=None):
        self.checks+=1
        if not value:
            self.failure_count+=1
            if len(self.failures)<100:self.failures.append(dict(check=label,actual=safe(actual),expected=safe(expected)))
    def eq(self,a,e,label):self.ok(a==e,label,a,e)
    def near(self,a,e,label):
        if isinstance(e,(list,tuple)):
            self.ok(isinstance(a,(list,tuple)) and len(a)==len(e),label+'.shape',a,e)
            if isinstance(a,(list,tuple)) and len(a)==len(e):
                for i,(aa,ee) in enumerate(zip(a,e)):self.near(aa,ee,label+'.'+str(i))
        elif e is None:self.eq(a,None,label)
        else:
            valid=type(a) in (int,float) and math.isfinite(a) and math.isfinite(e)
            difference=abs(a-e) if valid else None
            if valid:self.max_difference=max(self.max_difference,difference)
            self.ok(valid and difference<=ATOL+RTOL*abs(e),label,a,e)


def finite(value,shape):
    if not isinstance(value,list) or len(value)!=shape[0]:return False
    if len(shape)>1:return all(finite(v,shape[1:]) for v in value)
    return all(type(v) in (int,float) and math.isfinite(v) for v in value)


def ids(value):
    return isinstance(value,list) and len(value)==len(set(value)) and all(type(k) is int and 0<=k<8 for k in value)


def eligible(points,hw):
    h,w=hw
    return [k for k,p in enumerate(points[:8]) if finite(p,(2,)) and p!=[-1,-1] and 0<=p[0]<w and 0<=p[1]<h]


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def project(dim,R,t,K):
    require(finite(dim,(3,)) and min(dim)>0 and finite(R,(3,3)) and finite(t,(3,)) and finite(K,(3,3)), 'invalid saved pose/camera')
    a,b,c=(v/2 for v in dim)
    X=((-a,-b,-c),(a,-b,-c),(a,b,-c),(-a,b,-c),(-a,-b,c),(a,-b,c),(a,b,c),(-a,b,c))
    result=[]
    for p in X:
        cam=[dot(r,p)+v for r,v in zip(R,t)]
        q=[dot(r,cam) for r in K]
        require(cam[2]>1e-9 and q[2]>0 and all(math.isfinite(v) for v in q),'invalid stored camera depth')
        result.append([q[0]/q[2],q[1]/q[2]])
    return result


def determinant3(R):
    a,b,c=R
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def native_hash(points,K,xyz,hw):
    if not (finite(points,(9,2)) and finite(K,(3,3)) and finite(xyz,(3,))):return None
    values=[v for p in points for v in p]+[v for r in K for v in r]+xyz
    return hashlib.sha256(b''.join(struct.pack('=d',float(v)) for v in values)+str((hw[1],hw[0])).encode()).hexdigest()


def dimensions(xyz):
    return [xyz] if abs(xyz[0]-xyz[2])<1e-9 else [xyz,[xyz[2],xyz[1],xyz[0]]]


def rotation_physical(R,dim):
    return R if dim==0 else [[-r[2],r[1],r[0]] for r in R]


def equivalent(a,b,xyz):
    return (max(abs(x-y) for r,s in zip(a['projected'],b['projected']) for x,y in zip(r,s))<1e-5 and
        max(abs(x-y) for r,s in zip(a['R_physical'],b['R_physical']) for x,y in zip(r,s))<1e-5 and
        math.dist(a['centroid'],b['centroid'])/math.sqrt(dot(xyz,xyz))<1e-5)


def stored_rank(value,audit,label):
    singular=value['singular_values']
    audit.ok(isinstance(singular,list) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in singular),label+'.finite_singular_values')
    require(bool(singular),'empty singular witness')
    audit.ok(all(a>=b for a,b in zip(singular,singular[1:])),label+'.descending')
    rank=sum(v>singular[0]*1e-10 for v in singular)
    audit.eq(value['numerical_rank'],rank,label+'.stored_rank')
    return rank


def ledger(snapshot,audit,label,eligible_count,dim_count):
    audit.ok(all(type(v) is int and v>=0 for v in snapshot.values()),label+'.nonnegative')
    audit.eq(snapshot['generic_calls'],sum(snapshot[k] for k in ('subset_generic_calls','standard_generic_calls','refit_generic_calls')),label+'.primitive_phase_sum')
    audit.eq(snapshot['generic_cache_misses'],snapshot['generic_calls']+snapshot['geometry_rejected'],label+'.cache_miss_accounting')
    cap=dim_count*math.comb(eligible_count,4) if eligible_count>=4 else 0
    audit.ok(snapshot['subsets_considered'] in (0,cap),label+'.lazy_full_four_subset_once')
    audit.ok(snapshot['subset_generic_calls']<=snapshot['subsets_considered']<=70*dim_count,label+'.at_most70_per_dimension')


def candidate(c,points,K,xyz,U,blocked,audit,counts,label,*,count_scope):
    counts['candidate_scalar_arithmetic_invocations']+=1
    counts[count_scope+'_scalar_arithmetic_invocations']+=1
    d=c['dimension_index'];dims=dimensions(xyz)
    audit.ok(type(d) is int and 0<=d<len(dims),label+'.registered_dimension')
    audit.eq(c['dimensions'],dims[d],label+'.dimension_values')
    audit.eq(c['selected_hypothesis'],('REGISTRY_WD','REGISTRY_DW')[d],label+'.dimension_name')
    audit.eq(c['dimension_constraint_allowed'],True,label+'.no_known_dimension_filter')
    for key in ('generator_ids','actual_fit_input_ids','inlier_ids'):
        audit.ok(ids(c[key]) and set(c[key])<=set(U) and not set(c[key])&blocked,label+'.active_'+key)
    audit.ok(len(c['generator_ids'])>=4 and len(c['actual_fit_input_ids'])>=4,label+'.actual_minimum4')
    q=project(c['dimensions'],c['R_cf'],c['centroid'],K)
    audit.ok(abs(determinant3(c['R_cf'])-1)<=1e-6,label+'.proper_rotation_determinant')
    audit.near(c['projected'],q,label+'.independent_projection')
    audit.near(c['R_physical'],rotation_physical(c['R_cf'],d),label+'.physical_rotation_mapping')
    residual=[math.dist(q[k],points[k]) for k in U]
    audit.near(c['residuals_used_px'],residual,label+'.residuals_U_only')
    # Stored residuals are validated above before exact fixed-cutoff membership;
    # this avoids claiming bit-identical BLAS/libm arithmetic at the 8px border.
    inliers=[k for k,r in zip(U,c['residuals_used_px']) if r<=RADIUS]
    sse=math.fsum(r*r for r in residual);truncated=math.fsum(min(r*r,64.0) for r in residual)
    audit.eq(c['inlier_ids'],inliers,label+'.inliers8px')
    audit.eq(c['inlier_count'],len(inliers),label+'.inlier_count')
    audit.eq(c['score_count'],len(inliers),label+'.robust_score_count')
    audit.near(c['sse_px2'],sse,label+'.sse')
    audit.near(c['truncated_sse_px2'],truncated,label+'.truncated_sse')
    audit.near(c['objective_px2'],truncated,label+'.robust_objective')
    counts['candidate_scalar_arithmetic_completed']+=1
    counts[count_scope+'_scalar_arithmetic_completed']+=1
    counts[count_scope+'_generator_'+c['generator']]+=1
    return (-c['inlier_count'],c['objective_px2'],d,tuple(c['generator_ids']),c['solution_index'],c['generator'])


def pose_packet(s,points,K,xyz,hw,H,T,audit,counts,events,label):
    blocked=set(H)|set(T);E=eligible(points,hw);U=[k for k in E if k not in blocked]
    for flag in FALSE_FLAGS:audit.eq(s[flag],False,label+'.'+flag)
    audit.eq(s['equivalence_uses_all_eight_model_predictions'],True,label+'.modeled_equivalence_scope')
    audit.eq(s['global_uniqueness_proven'],False,label+'.no_global_unique_claim')
    audit.eq(s['eligible'],E,label+'.eligible_native_pixels')
    audit.eq(s['used'],U,label+'.exact_scoring_pool')
    audit.eq(s['hidden'],H,label+'.actual_H')
    audit.eq(s['temporary_excluded'],T,label+'.temporary_heldout')
    audit.eq(s['excluded'],sorted(blocked),label+'.excluded_union')
    audit.eq(s['solver'],'OBSERVATION_ONLY_FINITE_SUBSET_ROBUST',label+'.fixed_robust_backend')
    audit.eq(s['residual_threshold_px'],8.0,label+'.fixed8')
    for key in ('fit_input_ids','final_inliers','input_inliers','inliers'):
        audit.ok(ids(s[key]) and set(s[key])<=set(U) and not set(s[key])&blocked,label+'.selected_'+key)
    digest=native_hash(points,K,xyz,hw)
    if digest is None:counts['nonfinite_numeric_hash_not_reconstructible']+=1
    else:audit.eq(s['input_hash'],digest,label+'.finite_numeric_hash')
    op,after=s['operation_counts'],s['hypothesis_bank_counts']
    audit.eq(set(op),set(after),label+'.operation_keys')
    before={k:after[k]-op[k] for k in after}
    audit.ok(all(type(v) is int and v>=0 for v in op.values()) and all(v>=0 for v in before.values()),label+'.nonnegative_interval')
    if s['input_hash'] in events:audit.eq(before,events[s['input_hash']],label+'.same_bank_sequential_reuse')
    else:audit.ok(all(v==0 for v in before.values()),label+'.first_bank_zero')
    events[s['input_hash']]=after
    ledger(after,audit,label+'.bank',len(E),len(dimensions(xyz)))
    audit.ok(0<=s['refit_count']<=3,label+'.maximum3_refits')
    if len(U)<4:
        audit.eq(s['state'],'INSUFFICIENT_OBSERVATIONS',label+'.less4_is_insufficient')
        audit.eq(s['all_candidate_solutions'],[],label+'.no_candidate_from_less4')
    if s['state']=='NEW_POSE':
        # A successful packet cannot bypass the rank checks by omitting a
        # witness. These are stored SVD witnesses, not independently made SVDs.
        for key,length in (('image',2),('object',3),('jacobian',6)):
            witness=s['geometry'].get(key)
            audit.ok(isinstance(witness,dict),label+'.accepted_required_'+key+'_witness')
            if isinstance(witness,dict):
                audit.eq(len(witness.get('singular_values',[])),length,label+'.accepted_'+key+'_singular_length')
    if U and s['geometry'].get('image'):
        rank=stored_rank(s['geometry']['image'],audit,label+'.image_layout')
        if s['state']=='NEW_POSE':audit.eq(rank,2,label+'.accepted_image_rank2')
    cs=s['all_candidate_solutions'];keys=[]
    counts['stored_candidate_records_in_packets']+=len(cs)
    for i,c in enumerate(cs):
        keys.append(candidate(c,points,K,xyz,U,blocked,audit,counts,label+'.candidate.'+str(i),count_scope='stored_candidate'))
        counts['stored_candidate_records_checked']+=1
    if cs:
        best=s['best_candidate']
        candidate(best,points,K,xyz,U,blocked,audit,counts,label+'.chosen',count_scope='chosen_candidate_recheck')
        audit.eq(best,cs[keys.index(min(keys))],label+'.best_consensus_then_sse_and_tie_order')
        audit.eq(s['fit_input_ids'],best['actual_fit_input_ids'],label+'.selected_actual_fit')
        audit.eq(s['final_inliers'],best['inlier_ids'],label+'.selected_final_inliers')
        audit.eq(s['input_inliers'],s['fit_input_ids'],label+'.input_inlier_alias')
        audit.eq(s['inliers'],s['final_inliers'],label+'.final_inlier_alias')
        audit.near(s['residuals_used_px'],best['residuals_used_px'],label+'.chosen_residuals')
        bydim=[]
        for d in range(len(dimensions(xyz))):
            index=[i for i,c in enumerate(cs) if c['dimension_index']==d]
            if index:bydim.append(cs[min(index,key=lambda i:keys[i])])
        audit.eq(s['per_dimension_best'],bydim,label+'.all_dimensions_best')
        ambiguous=any(c['inlier_count']==best['inlier_count'] and abs(c['objective_px2']-best['objective_px2'])<=1e-8 and
                      not equivalent(c,best,xyz) for c in cs)
        # Ambiguity is checked only after adequate consensus and local rank;
        # earlier unavailable states legitimately do not perform arbitration.
        if s['state'] in ('NEW_POSE','AMBIGUOUS_PNP'):
            audit.eq(s['unresolved_ambiguity'],ambiguous,label+'.all_branch_ambiguity')
            audit.eq(s['state']=='AMBIGUOUS_PNP',ambiguous,label+'.ambiguous_is_unavailable')
            distinct=[c for c in cs if not equivalent(c,best,xyz)]
            audit.eq(s['multiple_solutions'],bool(distinct),label+'.physically_distinct_branch_flag')
            tied=sum(c['inlier_count']==best['inlier_count'] and abs(c['objective_px2']-best['objective_px2'])<=1e-8 for c in cs)-1
            audit.eq(s['numerical_tied_solution_count'],tied,label+'.every_tied_returned_branch')
        if s['geometry'].get('jacobian'):
            rank=stored_rank(s['geometry']['jacobian'],audit,label+'.local_J')
            counts['stored_local_J_rank_'+str(rank)]+=1
            if s['state']=='NEW_POSE':audit.eq(rank,6,label+'.accepted_local_rank6')
            if s['state']=='NUMERICAL_RANK_DEFICIENT':audit.ok(rank<6,label+'.rank_failure_distinct')
        if s['geometry'].get('object'):
            rank=stored_rank(s['geometry']['object'],audit,label+'.object_layout')
            if s['state']=='NEW_POSE':audit.ok(rank>=2,label+'.accepted_object_rank_at_least2')
    new=s['available']
    audit.eq(s['pose_available'],new,label+'.numeric_pose_available')
    audit.eq(s['new_pose_estimated'],new,label+'.numeric_new')
    audit.eq(s['no_pose'],not new,label+'.numeric_no_pose')
    audit.eq(s['fallback_used'],False,label+'.no_baseline_fallback_inside_solver')
    audit.eq(s['state']=='NEW_POSE',new,label+'.new_state')
    if new:audit.ok(bool(cs),label+'.accepted_has_stored_candidates')
    expected=[list(p) for p in points]
    if new:
        audit.ok(len(U)>=4 and len(s['final_inliers'])>=4 and not s['unresolved_ambiguity'],label+'.accepted_consensus_and_not_ambiguous')
        for key in ('R_cf','R_physical','centroid'):audit.near(s[key],s['best_candidate'][key],label+'.chosen_pose_'+key)
        audit.eq(s['cf_extents'],s['best_candidate']['dimensions'],label+'.chosen_dimension')
        q=project(s['cf_extents'],s['R_cf'],s['centroid'],K);audit.near(s['projected'],q,label+'.final_Rt')
        for k in H:expected[k]=q[k]
        audit.eq(s['hidden_reprojected'],bool(H),label+'.H_reprojection_flag')
    audit.near(s['points_final'],expected,label+'.H_only_output_temporary_k_not_projected')
    counts['pose_state_'+s['state']]+=1
    counts['pose_packets_with_'+str(len(U))+'_observations']+=1
    return events[s['input_hash']]


def query_checks(obs,cal,audit,counts,label):
    q=obs['queries'];audit.eq(len(q),84,label+'.84queries')
    flat=[];selected=0
    base=obs['original_base_points'];h,w=obs['raw_hw']
    for i,r in enumerate(q):
        e,j=divmod(i,7);a,b=EDGES[e];fraction=(j+1)/8
        delta=[base[b][v]-base[a][v] for v in (0,1)];length=math.hypot(*delta)
        center=[(1-fraction)*base[a][v]+fraction*base[b][v] for v in (0,1)]
        normal=[-delta[1]/max(length,1e-6),delta[0]/max(length,1e-6)]
        audit.eq((r['query'],r['edge'],r['endpoints']),(i,e,[a,b]),label+'.query_identity.'+str(i))
        audit.near(r['center'],center,label+'.Base_query_center.'+str(i))
        audit.near(r['normal'],normal,label+'.Base_query_normal.'+str(i))
        z=r['candidate_logits'];require(finite(z,(66,)),'nonfinite raw FP32 logits')
        audit.eq([struct.unpack('<f',struct.pack('<f',v))[0] for v in z],z,label+'.raw_FP32_values.'+str(i));flat.extend(z)
        best=max(range(65),key=lambda k:z[k]);margin=max(-700,min(700,z[best]-z[65]));confidence=1/(1+math.exp(-margin))
        weights=[math.exp(v-max(z[:65])) for v in z[:65]];total=math.fsum(weights)
        sigma=max(.5,math.sqrt(math.fsum(p*(k-best)**2 for k,p in enumerate(weights))/total))
        audit.eq(r['chosen_candidate'],best,label+'.MAP_integer_bin.'+str(i))
        audit.near(r['confidence'],confidence,label+'.confidence.'+str(i))
        audit.near(r['sigma_mode_px'],sigma,label+'.conditional_sigma.'+str(i))
        audit.near(r['radius_px'],r['sigma_mode_px']*cal['uncertainty']['query_scale'],label+'.own_query_scale.'+str(i))
        xy=[center[v]+(best-32)*normal[v] for v in (0,1)]
        valid=finite(delta,(2,)) and length>1e-6 and base[a]!=[-1,-1] and base[b]!=[-1,-1]
        inside=finite(xy,(2,)) and 0<=xy[0]<w and 0<=xy[1]<h
        reason=('INVALID_QUERY_GEOMETRY' if not valid else 'MODEL_CALIBRATION_UNSUPPORTED' if e not in cal['supported_edges'] else
            'CANDIDATE_OUTSIDE_IMAGE' if not inside else 'CALIBRATION_PRECISION_NOT_ATTAINED' if not cal['confidence']['enabled'] else
            'CALIBRATED_NO_MATCH_OR_LOW_CONFIDENCE' if r['confidence']<cal['confidence']['threshold'] else 'CALIBRATED_QUERY_OBSERVATION')
        accept=reason=='CALIBRATED_QUERY_OBSERVATION'
        audit.eq(r['reason'],reason,label+'.own_CAL_membership.'+str(i));audit.eq(r['no_match'],not accept,label+'.abstention.'+str(i))
        audit.eq(r['physical_absence_inferred'],False,label+'.no_physical_absence.'+str(i))
        audit.eq(r['calibrated_model_coverage'],e in cal['supported_edges'],label+'.own_model_coverage.'+str(i))
        audit.near(r['selected_xy'],xy if accept else None,label+'.actual_selected_bin.'+str(i));selected+=accept
    sha=hashlib.sha256(b''.join(struct.pack('<f',v) for v in flat)).hexdigest()
    audit.eq(obs['raw_logits_sha256'],sha,label+'.full_raw_logit_SHA')
    audit.eq(obs['selected_queries'],selected,label+'.selected_query_count')
    counts['source_query_logit_records']+=len(q)
    invalid=[]
    for line in obs['lines']:
        e=line['edge'];audit.eq(line['endpoints'],list(EDGES[e]),label+'.line_endpoint_identity')
        errors=[abs(dot(p,line['normal'])-line['offset']) for p in line['support_points']]
        audit.eq(len(errors),len(line['query_radii_px']),label+'.line_radius_length')
        good=all(math.isfinite(v) and v>=0 and r<=v+1e-9 for r,v in zip(errors,line['query_radii_px']))
        if not good:invalid.append(e)
        for k,p,r in zip(line['queries'],line['support_points'],line['query_radii_px']):
            audit.eq(q[k]['edge'],e,label+'.line_query_edge');audit.eq(q[k]['selected_xy'],p,label+'.line_actual_query_coordinate')
            audit.eq(q[k]['radius_px'],r,label+'.line_own_radius')
    admitted=[]
    for c in obs['corners']:
        audit.ok(len(c['edges'])==2 and all(c['id'] in EDGES[e] for e in c['edges']),label+'.actual_incident_corner')
        if not set(c['edges'])&set(invalid) and math.isfinite(c['radius_px']) and 0<=c['radius_px']<=8:
            admitted.append(c['id'])
    counts['raw_decoder_lines']+=len(obs['lines']);counts['raw_decoder_corners']+=len(obs['corners'])
    return dict(admitted=sorted(admitted),invalid=sorted(invalid))


def selection_checks(contract,obs,row,audit,counts,events,label):
    H=row['hidden_initial'];native=obs['native_N3_points'];corners={c['id']:c for c in obs['corners']}
    records=contract['cornerwise_records'];selected=[];previous=contract['selection_bank_before_counts'];calls=0
    for r in records:
        k=r['id'];rlabel=label+'.LOO.'+str(k)
        audit.eq(r['candidate_xy'],corners[k]['xy'],rlabel+'.actual_boundary_not_projection')
        audit.eq(r['native_N3_xy'],native[k],rlabel+'.native_coordinate')
        audit.eq(r['heldout_fit_requested_excluded_ids'],sorted(set(H)|{k}),rlabel+'.H_plus_k_request')
        audit.eq(r['projected_coordinate_used_as_observation'],False,rlabel+'.projection_not_input')
        audit.eq(r['numeric_validation_pose_prior_used'],False,rlabel+'.no_prior')
        if r['heldout_pose_calls']:
            audit.eq(r['heldout_pose_calls'],1,rlabel+'.one_solve');calls+=1;s=r['loo_solver']
            before={key:s['hypothesis_bank_counts'][key]-s['operation_counts'][key] for key in s['operation_counts']}
            audit.eq(before,previous,rlabel+'.sequential_native_bank')
            previous=pose_packet(s,native,row['K'],row['xyz'],row['raw_hw'],H,[k],audit,counts,events,rlabel)
            audit.eq(r['fit_input_ids'],s['fit_input_ids'],rlabel+'.fit_ID_join')
            audit.eq(r['final_inlier_ids'],s['final_inliers'],rlabel+'.inlier_ID_join')
            accepted=False
            if s['available']:
                qr=s['projected'][k];cr=math.dist(r['candidate_xy'],qr)
                nr=math.dist(native[k],qr) if r['native_N3_available'] else None
                gain=nr*nr-cr*cr if nr is not None else None
                audit.near(r['candidate_LOO_residual_px'],cr,rlabel+'.candidate_prediction_error')
                audit.near(r['native_N3_LOO_residual_px'],nr,rlabel+'.native_prediction_error')
                audit.near(r['native_minus_candidate_squared_residual_px2'],gain,rlabel+'.relative_gain')
                accepted=r['candidate_LOO_residual_px']<=8 and (gain is None or r['native_minus_candidate_squared_residual_px2']>1e-8)
            audit.eq(r['accepted'],accepted,rlabel+'.fixed8_relative_LOO_selection')
        else:
            audit.eq(r['accepted'],False,rlabel+'.no_fit_no_adoption')
            audit.ok(k in H or (r['native_N3_available'] and r['native_distance_px']>8),rlabel+'.skip_only_H_or_native_basin')
        if r['accepted']:selected.append(k)
    audit.eq(contract['heldout_pose_calls'],calls,label+'.actual_LOO_count')
    audit.eq(contract['cornerwise_selection']['LOO_solve_calls'],calls,label+'.summary_LOO_count')
    audit.eq(contract['selection_bank_after_counts'],previous,label+'.selection_bank_end')
    audit.eq(contract['operation_counts'],{k:previous[k]-contract['selection_bank_before_counts'][k] for k in previous},label+'.selection_delta')
    audit.eq(contract['hybrid_boundary_corner_ids'],selected,label+'.selected_actual_corner_IDs')
    counts['heldout_native_solve_packets']+=calls
    expected=[list(p) for p in native]
    for k in selected:expected[k]=corners[k]['xy']
    return expected,selected


def provenance(paths,audit):
    completion=read(paths['source_CAL_completion']);protocol=read(paths['source_CAL_protocol'])
    audit.eq(completion['complete'],True,'complete_three_head_source_CAL')
    bound(paths['source_CAL_protocol'],completion['protocol'])
    audit.eq(protocol['head_modes'],MODES,'head_specific_channel_modes')
    audit.eq(completion['input_bindings'],protocol['inputs'],'CAL_complete_frozen_inputs')
    audit.eq(completion['actual_new_head_forward_calls'],16,'CAL_total_new_head16')
    audit.eq(completion['actual_new_head_image_exposures'],256,'CAL_total_new_exposure256')
    audit.eq(completion['ROLE_new_head_forward_calls'],0,'CAL_ROLE_current0')
    bound(paths['original_ROLE_calibration_code'],protocol['inputs']['algorithm'])
    for key in ('source_test_exposures','real_frames','new_training_updates','new_RGB',
                'detector_calls','N3_calls','PnP_calls','rays','feature_recomputations'):
        audit.eq(protocol[key],0,'CAL_scope_'+key)
        audit.eq(completion[key],0,'CAL_actual_scope_'+key)
    metadata={r['arm']:r for r in read(paths['checkpoint_metadata'])['rows']}
    audit.eq(set(metadata),set(ARMS),'three_original_training_arms')
    for key in ('config','initial_state_sha256','batch_order_sha256','protocol_sha256','repaired_target_sha256','parameters'):
        audit.ok(all(metadata[a][key]==metadata[ARMS[0]][key] for a in ARMS),'same_fixed_training_'+key)
    bound(paths['checkpoint_metadata'],protocol['inputs']['checkpoint_metadata'])
    calibrations={arm:read(paths[arm+':CALIBRATION.json']) for arm in ARMS}
    for arm in ARMS:
        audit.eq(metadata[arm]['steps'],3000,'checkpoint_last_step.'+arm)
        audit.eq(metadata[arm]['checkpoint']['sha256'],CHECKPOINT_SHA[arm],'checkpoint_fixed_SHA.'+arm)
        bound(paths[arm+':CALIBRATION.json'],completion['arms'][arm]['calibration'])
        bound(paths[arm+':CALIBRATION_EXECUTION.json'],completion['arms'][arm]['execution'])
        execution=read(paths[arm+':CALIBRATION_EXECUTION.json'])
        audit.ok(execution['complete'] and execution['arm']==arm and execution['partition']=='calibration','completed_CAL_execution.'+arm)
        audit.eq(execution['head_forward_calls'],8,'historical_or_new_head8.'+arm)
        audit.eq(execution['head_image_exposures'],128,'historical_or_new_CAL128.'+arm)
        for key in ('source_test_exposures','real_frames','training_updates','detector_calls','PnP_calls','rays','new_RGB','feature_recomputations'):
            audit.eq(execution[key],0,'head_CAL_scope.'+arm+'.'+key)
        audit.eq(completion['arms'][arm]['checkpoint']['sha256'],CHECKPOINT_SHA[arm],'own_CAL_checkpoint.'+arm)
        if arm!='IMAGE_ROLE':
            audit.eq(execution['N3_calls'],0,'new_head_CAL_scope.'+arm+'.N3_calls')
            bound(paths[arm+':CALIBRATION_PROTOCOL.json'],execution['protocol'])
            own=read(paths[arm+':CALIBRATION_PROTOCOL.json'])
            audit.eq(own['head_mode'],MODES[arm],'own_CAL_mode.'+arm)
            audit.eq(own['indices'],list(range(768,896)),'own_CAL128_indices.'+arm)
            audit.eq(calibrations[arm]['training_arm'],arm,'own_threshold_training_arm.'+arm)
            audit.eq(calibrations[arm]['checkpoint']['sha256'],CHECKPOINT_SHA[arm],'own_coefficients_checkpoint.'+arm)
            audit.eq(calibrations[arm]['protocol_sha256'],execution['protocol']['sha256'],'own_threshold_protocol.'+arm)
            audit.eq(completion['arms'][arm]['new_head_forward_calls'],8,'new_head8.'+arm)
            audit.eq(completion['arms'][arm]['new_head_image_exposures'],128,'new_CAL128.'+arm)
        else:
            # Preserve the original execution receipt's schema and report its
            # missing field honestly. Byte-exact reuse is checked below; the
            # historical algorithm code is bound above. Do not invent zero.
            audit.eq('N3_calls' in execution,False,'ROLE_historical_N3_calls_field_absent')
            audit.eq(completion['arms'][arm]['new_head_forward_calls'],0,'ROLE_current0')
            audit.eq(completion['arms'][arm]['reused_byte_exact'],True,'ROLE_byte_exact')
    reuse=read(paths['ROLE_reuse']);bound(paths['ROLE_reuse'],completion['arms']['IMAGE_ROLE']['reuse'])
    audit.ok(reuse['complete'] and reuse['current_new_head_forward_calls']==0 and reuse['historical_head_forward_calls']==8,'ROLE_reuse_historical_vs_current')
    audit.eq(reuse['checkpoint']['sha256'],CHECKPOINT_SHA['IMAGE_ROLE'],'ROLE_reuse_fixed_checkpoint')
    audit.eq(set(reuse['files']),set(ROLE_FILES),'ROLE_exactly_nine_files')
    for name in ROLE_FILES:
        bound(paths['IMAGE_ROLE:'+name],reuse['files'][name])
        bound(paths['IMAGE_ROLE:'+name],protocol['inputs']['ROLE:'+name])
        audit.eq({k:binding(paths['IMAGE_ROLE:'+name])[k] for k in ('sha256','bytes')},
                 {k:binding(paths['original_ROLE:'+name])[k] for k in ('sha256','bytes')},'ROLE_original_exact.'+name)
    sourcecheck=read(paths['independent_source_CAL_checks'])
    audit.ok(sourcecheck['complete'] and sourcecheck['passed'] and sourcecheck['scalar_checks']==198848,'independent_source_scalar_checks_complete')
    return calibrations


def run(args):
    folder,output=guard(args,'run');protocol_path=output/OUTPUTS[0]
    protocol=read(protocol_path);paths=inputs(args)
    require(protocol['inputs']=={k:binding(p) for k,p in paths.items()},'supplemental frozen input/code drift')
    write(output/OUTPUTS[1],dict(protocol=binding(protocol_path),actual_saved_scalar_invocations=1,
        no_automatic_retry=True, model_GT_PnP_optimizer_image_ray_training_calls=0))
    start=time.perf_counter();audit=Audit();counts=Counter();complete=False
    result=dict(schema='supplemental_v7_all_point_candidate_scalar_checks_v1',passed=False,complete=False,
        protocol=binding(protocol_path),inputs=protocol['inputs'],actual_saved_scalar_invocations=1,
        new_model_GT_PnP_optimizer_image_ray_training_calls=0, no_production_numeric_imports=True,
        candidate_count_scope=protocol['candidate_count_scope'],
        historical_ROLE_execution_N3_calls=protocol['historical_ROLE_execution_N3_calls'],
        numerical_rank='Stored singular-value rank arithmetic, no independent Jacobian or SVD',
        limitations=['Stored GPU/checkpoint/operation receipts are not independent execution authenticity evidence.',
            'All scored numeric branches are checked; a stored bank does not establish global physical uniqueness.',
            'H and proposals retain conditional N3/Base dependence; exclusions are numerical active-ID witnesses.',
            'Source CAL controls coefficient provenance; real physical wire/corner ownership is not certified.',
            'Finite numeric hashes reconstructed; JSON null cannot establish original NaN payload or its byte hash.',
            'Validated stored residual/confidence values determine exact cutoff membership; scalar replay is within fixed tolerances, not bit-exact BLAS/libm.',
            'Only GT-free sealed geometry is read; no pose-accuracy score is recalculated.'])
    try:
        calibrations=provenance(paths,audit);seal=read(paths['GEOMETRY_SEAL.json']);receipt=read(paths['INFERENCE_RECEIPT.json'])
        audit.eq(seal['methods'],list(METHODS),'eight_frozen_methods')
        audit.ok(seal['complete'] and not seal['GT_read_allowed'] and receipt['complete'] and receipt['cleanup_error'] is None,'complete_before_truth_seal')
        fixed={}
        for r in rows(paths['FIXED_GEOMETRY_SEALED.jsonl.gz']):
            key=(r['method'],r['id']);audit.ok(key not in fixed,'unique_fixed');fixed[key]=r
        audit.eq(len(fixed),490,'fixed490')
        cohort={fid for method,fid in fixed if method=='N3_SUBPIX'};audit.eq(len(cohort),245,'245_unique_frames')
        audit.eq({fid for method,fid in fixed if method=='BASE'},cohort,'same_Base_cohort')
        observations={};admissions={}
        for obs in rows(paths['OBSERVATIONS.jsonl.gz']):
            arm,fid=obs['head_arm'],obs['id'];label=arm+'/'+fid;key=(arm,fid)
            audit.ok(arm in ARMS and key not in observations,'unique_head_observation.'+label)
            audit.eq(obs['GT_input'],False,label+'.no_GT_input')
            audit.eq(obs['native_N3_points'],fixed[('N3_SUBPIX',fid)]['native_points'],label+'.same_N3_anchor')
            audit.eq(obs['original_base_points'],fixed[('BASE',fid)]['native_points'],label+'.same_Base_anchor')
            obs['raw_hw']=fixed[('N3_SUBPIX',fid)]['raw_hw']
            admissions[key]=query_checks(obs,calibrations[arm],audit,counts,label)
            # Keep only fields required downstream; original file is unchanged.
            observations[key]={k:v for k,v in obs.items() if k not in ('queries','diagnostics','partial_lines')}
        audit.eq(len(observations),735,'735_head_observations')
        for arm in ARMS:audit.eq({fid for a,fid in observations if a==arm},cohort,'every_head_same245.'+arm)
        seen=set();bank_events={};bank_bindings={}
        for row in rows(paths['GEOMETRY_SEALED.jsonl.gz']):
            method,fid=row['method'],row['id'];label=method+'/'+fid;key=(method,fid)
            audit.ok(method in METHODS and fid in cohort and key not in seen,'unique_known_geometry.'+label);seen.add(key)
            native=fixed[('N3_SUBPIX',fid)]['native_points'];contract=row['observation_contract']
            arm=row['head_arm'];H=[] if method=='N3_INDEPENDENT_ROBUST_NO_MASK' else observations[('IMAGE_ROLE',fid)]['predicted_N3_hidden']
            audit.eq(row['hidden_initial'],H,label+'.same_H_or_explicit_no_mask')
            events=bank_events.setdefault(fid,{})
            if arm is not None:
                obs=observations[(arm,fid)];admission=admissions[(arm,fid)]
                audit.eq(row['observation_raw_logits_sha256'],obs['raw_logits_sha256'],label+'.own_head_raw_logits')
                audit.eq(row['head_checkpoint_sha256'],CHECKPOINT_SHA[arm],label+'.own_checkpoint')
                audit.eq(row['head_mode'],arm,label+'.training_mode')
                audit.eq(row['calibration_binding']['sha256'],binding(paths[arm+':CALIBRATION.json'])['sha256'],label+'.own_CAL_binding')
                audit.eq(contract['validated_boundary_corner_ids'],admission['admitted'],label+'.final_support_and8px_admission')
                audit.eq(contract['invalid_final_line_edges'],admission['invalid'],label+'.final_support_veto')
                audit.eq(row['source_lines'],[l['edge'] for l in obs['lines']],label+'.same_actual_decoded_lines')
                if row['observation_supply']=='CORNERWISE_HYBRID':
                    expected,selected=selection_checks(contract,obs,row,audit,counts,events,label)
                else:
                    audit.eq(contract['heldout_pose_calls'],0,label+'.boundary_only_no_LOO')
                    audit.eq(contract['boundary_only_native_distance_used_for_admission'],False,label+'.boundary_only_no_native_basin')
                    audit.eq(contract['native_N3_used_for_numeric_final_fit'],False,label+'.boundary_only_no_native_fit')
                    expected=[[None,None] for _ in range(8)]+[native[8]]
                    corners={c['id']:c for c in obs['corners']}
                    selected=admission['admitted']
                    for k in selected:expected[k]=corners[k]['xy']
                audit.eq(row['selected_corner_ids'],selected,label+'.actual_selected_corner_IDs')
            else:
                expected=native;selected=[]
                audit.eq(row['head_checkpoint_sha256'],None,label+'.native_control_no_head')
            audit.eq(row['input_points'],expected,label+'.actual_observation_input_no_projection_fill')
            audit.eq(row['input_points'][8],native[8],label+'.center_input_preserved')
            audit.eq(row['partial_lines_not_pose_inputs'],True,label+'.point_only')
            audit.eq(contract['lines_are_not_extra_pose_factors'],True,label+'.no_line_duplicate_factors')
            audit.eq(row['final_numeric_pose_has_initial_prior'],False,label+'.final_no_initial_prior')
            s=row['solver'];after=pose_packet(s,row['input_points'],row['K'],row['xyz'],row['raw_hw'],H,[],audit,counts,events,label+'.final')
            # Keep final snapshots only for the seal's once-per-bank ledger join.
            bank_bindings[(fid,s['input_hash'])]=after
            new=s['available'];fallback=not new and row['initial_pose'].get('available',False)
            audit.eq(row['new_pose_estimated'],new,label+'.driver_new')
            audit.eq(row['fallback_used'],fallback,label+'.baseline_fallback_distinct')
            audit.eq(row['pose_available'],new or fallback,label+'.operational_availability')
            audit.eq(row['actual_pose']['available'],row['pose_available'],label+'.actual_pose_object_available')
            if new:
                for field in ('R_cf','R_physical','centroid','cf_extents'):
                    audit.eq(row['actual_pose'][field],s[field],label+'.actual_pose_equals_final_solver_'+field)
                counts['driver_new_pose_numeric_join_packets']+=1
            else:
                audit.eq(row['actual_pose'],row['initial_pose'],label+'.fallback_or_failure_uses_declared_initial_pose')
                counts['driver_initial_pose_join_packets']+=1
            audit.eq(row['no_pose'],not row['pose_available'],label+'.operational_failure_distinct')
            audit.eq(row['output_status'],'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',label+'.operational_status')
            out=[list(p) for p in native]
            if new:
                for k in range(8):
                    if k not in H and finite(expected[k],(2,)) and expected[k]!=[-1,-1]:out[k]=expected[k]
                for k in H:out[k]=s['projected'][k]
            audit.near(row['native_points'],out,label+'.new_actual_observations_H_projection_or_fullN3_fallback')
            audit.eq(row['reprojected_ids'],H if new else [],label+'.H_replaced_exactly_once')
            audit.eq(row['reprojections_reused_as_observations'],False,label+'.no_output_refit')
            audit.eq(row['local_point_line_refinement'],False,label+'.no_C2_route')
            counts['geometry_method_'+method]+=1;counts['operational_'+row['output_status']]+=1
        audit.eq(len(seen),1960,'geometry1960')
        for m in METHODS:audit.eq({fid for method,fid in seen if method==m},cohort,'every_method_same245.'+m)
        audit.eq({r['id'] for r in seal['banks']},cohort,'seal_bank_ledger_every_frame')
        for frame in seal['banks']:
            fid=frame['id'];registered=frame['coordinate_banks'];events=bank_events[fid]
            audit.eq(frame['coordinate_bank_count'],len(registered),'unique_bank_count.'+fid)
            audit.eq({b['input_hash'] for b in registered},set(events),'all_numeric_banks_once.'+fid)
            for bank in registered:
                audit.eq({k:v for k,v in bank.items() if k not in ('input_hash','prior_backend')},events[bank['input_hash']],'final_shared_bank_ledger.'+fid)
            counts['unique_coordinate_banks']+=len(registered)
        for name,path in paths.items():audit.eq(binding(path),protocol['inputs'][name],'input_preserved.'+name)
        audit.eq(counts['stored_candidate_records_checked'],counts['stored_candidate_records_in_packets'],'all_stored_candidate_records_reached')
        audit.eq(counts['stored_candidate_scalar_arithmetic_invocations'],counts['stored_candidate_records_checked'],'every_stored_candidate_record_checked_once')
        audit.eq(counts['candidate_scalar_arithmetic_completed'],counts['candidate_scalar_arithmetic_invocations'],'every_candidate_arithmetic_invocation_completed')
        audit.eq(counts['candidate_scalar_arithmetic_invocations'],
                 counts['stored_candidate_records_checked']+counts['chosen_candidate_recheck_scalar_arithmetic_invocations'],
                 'candidate_arithmetic_separates_stored_records_and_chosen_rechecks')
        complete=True
    except Exception as error:
        audit.ok(False,'exception',dict(type=type(error).__name__,message=str(error)))
    result.update(complete=complete,passed=complete and audit.failure_count==0,checks=audit.checks,
        failure_count=audit.failure_count,failures=audit.failures,actual_saved_arithmetic_counts=dict(counts),
        maximum_absolute_scalar_difference=audit.max_difference,wall_seconds=time.perf_counter()-start,Python=sys.version)
    write(output/OUTPUTS[2],safe(result))
    print(json.dumps(dict(passed=result['passed'],complete=complete,checks=audit.checks,failures=audit.failure_count,
        output=str(output/OUTPUTS[2])),ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC,help='complete GT-free V7 geometry folder')
    p.add_argument('--output',type=Path,default=DOC,help='existing new DOC or new V7 private output directory')
    p.add_argument('--protocol',type=Path,default=DOC/'PROTOCOL.json',help='frozen accuracy protocol')
    p.add_argument('--calibration-root',type=Path,default=DOC/'source_calibration')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args)
    else:run(args)


if __name__=='__main__':main()
