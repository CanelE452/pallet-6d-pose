"""Own-frozen small geometry audit; no production imports or pose execution."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = OLD = WITNESS = OUT = None
LIMITS = dict(new_detector_head_N3_PnP_optimizer_SVD_GT_image_calls=0,
              model_weights_images_annotations_raw_rows_read=0,
              original_geometry_stream_rows_read=0,
              small_explicit_2D_fixtures_and_saved21_failure_contexts_only=True,
              pose_performance_or_hardware_latency_claim=False,
              historical_V16_564_FAIL_relabelled=False,
              new_observed_frame_objective_not_old_arithmetic_repair=True,
              successful_counterexample_test_does_not_prove_real_pose_improvement=True)


def binding(p):
    return dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def require(v, reason):
    if not v:
        raise ValueError(reason)


def load_geometry():
    # Fresh namespace; no source module package imports or hidden dependencies.
    tree = ast.parse((HERE/'geometry.py').read_text())
    env = dict(__name__='isolated_observed_frame_geometry')
    exec(compile(tree, str(HERE/'geometry.py'), 'exec'), env)
    return env['observed_frame_block']


def old_kernel():
    tree = ast.parse(OLD.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finite_blocks']
    require(len(nodes) == 1, 'one historical finite_blocks function')
    selected = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    env = dict(math=math, require=require)
    exec(compile(selected, str(OLD)+':unchanged_finite_blocks', 'exec'), env)
    return env['finite_blocks'], hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest()


def norm(v):
    return math.sqrt(math.fsum(x*x for x in v))


def closed_distance(p, A, B):
    d = [B[i]-A[i] for i in range(2)]
    length2 = math.fsum(x*x for x in d)
    u = max(0., min(1., math.fsum((p[i]-A[i])*d[i] for i in range(2))/length2))
    return math.hypot(*(p[i]-A[i]-u*d[i] for i in range(2)))


def fixture(A, B, support, name, old, fixed):
    J = [[[0.]*6 for _ in range(2)] for _ in range(2)]
    line = dict(edge=0, normal=[0., 1.], offset=0., support_query_ids=list(range(len(support))),
                source_line=dict(support_points=support, query_radii_px=[1.]*len(support)))
    h = dict(EDGES=[(0, 1)], dot=lambda a,b: math.fsum(x*y for x,y in zip(a,b)))
    r0, _, b0 = old([A, B], J, [], [], [line], h)
    r1, _, b1 = fixed([A, B], J, line['normal'], 0., support)
    actual = [closed_distance(p, A, B) for p in support]
    return dict(name=name, endpoints=[A, B], actual_support_xy=support,
                old_block_residual=r0, old_edge_RMS_px=norm(r0), old_hinge_branches=b0[0]['hinge_branches'],
                repaired_block_residual=r1, repaired_edge_RMS_px=norm(r1), repaired_branches=b1,
                actual_support_closed_segment_distances_px=actual,
                actual_support_RMS_distance_px=math.sqrt(math.fsum(v*v for v in actual)/len(actual)),
                same_edge_vote_count=1, existing_threshold_px=8.,
                old_inlier=norm(r0) <= 8., repaired_inlier=norm(r1) <= 8.)


def saved_contexts(fixed):
    saved = json.loads(WITNESS.read_bytes())
    require(saved['geometry_validation_passed'] is False
            and saved['scope']['total_failed_predicates'] == 564
            and saved['scope']['scalar_checks'] == 198134007
            and len(saved['contexts']) == 21, 'complete original failed geometry contexts')
    outputs = []
    for key, context in sorted(saved['contexts'].items()):
        candidate = context['candidate']
        original = context['point_line_geometry']
        blocks = {b['edge']:b for b in original['edge_blocks']}
        segments = {b['edge']:b for b in context['stored_projected_finite_segments']}
        for line in context['stored_fit_source_lines']:
            edge = line['edge']; block = blocks[edge]; segment = segments[edge]
            endpoints = [candidate['projected'][k] for k in block['endpoints']]
            require(endpoints == segment['projected_endpoint_xy'], 'literal saved endpoint identity')
            support = line['raw']['support_points']
            require(support == block['actual_support_xy'], 'literal same actual support identity')
            old_residual = original['actual_scalar_residuals_px'][block['scalar_start']:block['scalar_end']]
            require(len(old_residual) == 2+len(support), 'one complete historical edge block')
            zero_J = [[[0.]*6 for _ in range(2)] for _ in range(2)]
            repaired, _, detail = fixed(endpoints, zero_J, line['normal'], line['offset'], support)
            distances = [closed_distance(p, *endpoints) for p in support]
            old_rms, repaired_rms = norm(old_residual), norm(repaired)
            outputs.append(dict(context=key,id=context['id'],edge=edge,
                stored_phase=candidate['phase'],stored_candidate_accepted=candidate['accepted'],
                original_row_output_status=context['row_output_status'],
                candidate_is_approved_NEW_output=context['candidate_is_approved_NEW_output'],
                stored_projected_endpoints=endpoints,actual_support_points=support,
                observed_normal=line['normal'],observed_offset=line['offset'],
                projected_segment_length_px=block['projected_segment_length_px'],
                original_edge_RMS_px=old_rms,repaired_edge_RMS_px=repaired_rms,
                actual_support_closed_segment_distances_px=distances,
                actual_support_closed_segment_RMS_px=math.sqrt(math.fsum(v*v for v in distances)/len(distances)),
                original_inlier=old_rms <= 8.,repaired_inlier=repaired_rms <= 8.,
                observed_frame_detail=detail,
                zero_J_for_residual_only_not_saved_candidate_derivative=True,
                saved_R_t_and_outputs_changed=False,physical_semantic_line_truth='UNKNOWN'))
    require(outputs, 'saved line contexts required')
    return outputs


def main():
    global SOURCE, OLD, WITNESS, OUT
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze','run'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a=parser.parse_args(); SOURCE=a.repo.resolve(); OUT=a.output.absolute()
    OLD=SOURCE/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py'
    WITNESS=SOURCE/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16/FAILURE_WITNESSES.json'
    inputs = {n:binding(HERE/n) for n in ('geometry.py','check.py')}
    inputs['historical_scalar_source'] = binding(OLD)
    inputs['saved_failure_witnesses'] = binding(WITNESS)
    require(inputs['saved_failure_witnesses']['sha256'] == 'b66a14eeaba43282e723e6dd6a5c0d02df05533fc6b8da9c9dd9c9b4c19656d6', 'immutable original failure witness bytes')
    if a.phase == 'freeze':
        require(not OUT.exists(), 'fresh absent evidence output required')
        OUT.mkdir()
        write(OUT/'PROTOCOL.json', dict(schema='finite_support_orthogonal_fixture_protocol_v19', inputs=inputs,
              fixture_names=['short_orthogonal','supported_parallel','short_parallel','reverse_endpoints'],
              source_scalar_function='finite_blocks unchanged AST', fixed_objective='observed-frame support containment',
              existing_threshold_px=8., derivative_fixture_step=1e-4, derivative_absolute_tolerance=1e-8,
              analytic_counterexample_comparison_ULPs=4,saved_failure_contexts=21,
              saved_contexts_new_projection_J_or_R_t_fit_calls=0,
              cusp_derivatives_not_asserted=True, limits=LIMITS))
        print('PROTOCOL_FROZEN'); return
    protocol = json.loads((OUT/'PROTOCOL.json').read_bytes())
    require(protocol['inputs'] == inputs, 'frozen code and old scalar bytes unchanged')
    require(not (OUT/'STARTED.json').exists() and not (OUT/'CHECKS.json').exists(), 'single run only')
    write(OUT/'STARTED.json', dict(schema='finite_support_fixture_started_v19', protocol=binding(OUT/'PROTOCOL.json'),inputs=inputs,limits=LIMITS))
    old, old_ast = old_kernel(); fixed = load_geometry(); support=[[-100.,0.],[0.,0.],[100.,0.]]
    cases=[fixture([0.,-.01],[0.,.01],support,'short_orthogonal',old,fixed),
           fixture([-110.,0.],[110.,0.],support,'supported_parallel',old,fixed),
           fixture([-.01,0.],[.01,0.],support,'short_parallel',old,fixed),
           fixture([110.,0.],[-110.,0.],support,'reverse_endpoints',old,fixed)]
    require(cases[0]['old_inlier'] is True and cases[0]['repaired_inlier'] is False,
            'old finite-support false inlier counterexample and fixed rejection')
    require(abs(cases[0]['old_edge_RMS_px']-.01) <= 4*math.ulp(.01) and cases[0]['actual_support_closed_segment_distances_px'] == [100.,0.,100.],
            'explicit finite-segment geometry counterexample')
    require(cases[1]['old_edge_RMS_px'] == cases[1]['repaired_edge_RMS_px'] == 0., 'valid parallel supported edge remains zero')
    require(cases[2]['old_inlier'] is False and cases[2]['repaired_inlier'] is False, 'short parallel edge remains outlier')
    require(cases[3]['repaired_edge_RMS_px'] == cases[1]['repaired_edge_RMS_px'], 'endpoint order invariant residual norm')
    # Independent central differences at smooth strict outside/inside branches.
    endpoints=[[10.,2.],[20.,-3.]]; smooth_support=[[5.,0.],[15.,0.],[25.,0.]]
    J=[[[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.]],[[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.]]]
    residual, analytic, info=fixed(endpoints,J,[0.,1.],0.,smooth_support)
    require(not info['interval_order_cusp'] and not any(info['endpoint_hit_cusps']), 'strict smooth derivative fixture')
    drift=[]
    for j in range(6):
        plus=[[endpoints[k][i]+1e-4*J[k][i][j] for i in range(2)] for k in range(2)]
        minus=[[endpoints[k][i]-1e-4*J[k][i][j] for i in range(2)] for k in range(2)]
        rp=fixed(plus,J,[0.,1.],0.,smooth_support)[0]; rm=fixed(minus,J,[0.,1.],0.,smooth_support)[0]
        drift.extend(abs((rp[i]-rm[i])/2e-4-analytic[i][j]) for i in range(len(residual)))
    require(max(drift) < 1e-8, 'fixed observed-frame independent derivative fixture')
    saved = saved_contexts(fixed)
    write(OUT/'SAVED_LINE_DIAGNOSTIC.json', dict(schema='fixed_saved21_finite_support_comparison_v19',
          contexts=21,line_blocks=len(saved),inputs=inputs,rows=saved,
          new_fits_projections_GT_or_pose_performance_evaluation=0,
          limits=LIMITS))
    require({k:binding(Path(v['path'])) for k,v in inputs.items()} == inputs, 'all source bytes preserved after execution')
    write(OUT/'CHECKS.json', dict(schema='finite_support_orthogonal_fixture_checks_v19',complete=True,passed=True,
          inputs=inputs,protocol=binding(OUT/'PROTOCOL.json'),old_scalar_AST_sha256=old_ast,
          fixture_cases=cases,derivative_scalar_checks=len(drift),derivative_max_absolute_difference=max(drift),
          saved_diagnostic=binding(OUT/'SAVED_LINE_DIAGNOSTIC.json'),saved_contexts=21,saved_line_blocks=len(saved),
          old_finite_support_false_inlier_reproduced=True,repaired_false_inlier_rejected=True,
          old_V16_FAIL_status_unchanged=True,real_performance_experiment_executed=False,limits=LIMITS))
    print(json.dumps(dict(passed=True,fixtures=len(cases),derivative_scalar_checks=len(drift),
                         derivative_max_absolute_difference=max(drift),
                         old_counterexample_RMS_px=cases[0]['old_edge_RMS_px'],
                         repaired_counterexample_RMS_px=cases[0]['repaired_edge_RMS_px'])))


if __name__ == '__main__':
    main()
