"""Independently check stored finite-segment diagnosis using only stdlib.

Own freeze precedes scalar arithmetic. No fit, projection call, model, image,
GT score, Jacobian, SVD, output filter or candidate selection is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_finite_segment_support_20261011_v12'
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
METHODS = ('FILTERED_LOCAL', 'FILTERED_LOCAL_WEAK_SAFE')
SOURCE_EDGES = frozenset((0,2,4,6,8,9,10,11))
FILES = ('SEGMENT_CHECK_PROTOCOL.json', 'SEGMENT_CHECK_STARTED.json', 'SEGMENT_CHECKS.json')
POLICY = dict(stdlib_only=True, all_245_frames_retained=True, candidate_selection=False,
    radius_is_uncertainty_surrogate=True, ownership_truth='UNKNOWN', new_pose_outputs=0,
    new_GT_scores=0, model_fit_projection_J_SVD_RGB_calls=0,
    floating_audit_tolerance=dict(relative=1e-9, absolute_px=1e-8),
    stored_inlier_threshold_px=8.0, source_consensus_tolerance_px=1e-9,
    numeric_tolerances_are_verification_only=True)


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and path.is_relative_to(REPO) and not any(p.is_symlink() for p in (path, *path.parents)), 'public regular input required: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return dict(path=str(path.relative_to(REPO)), bytes=path.stat().st_size, sha256=digest.hexdigest())


def same(left, right):
    return all(left.get(k) == right.get(k) for k in ('path', 'bytes', 'sha256'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def paths(parent):
    result = dict(checker=Path(__file__), checker_contract=DOC/'CHECKER_CONTRACT_KO.md', diagnostic_protocol=parent/'DIAGNOSTIC_PROTOCOL.json',
        diagnostic_started=parent/'DIAGNOSTIC_STARTED.json', diagnostic_receipt=parent/'DIAGNOSTIC_RECEIPT.json',
        diagnostic_rows=parent/'SEGMENT_ROWS.jsonl.gz')
    for key, value in read(result['diagnostic_protocol'])['inputs'].items():
        relative = Path(value['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'public input path required')
        result['parent:' + key] = REPO / relative
    return result


def authority(files, current):
    protocol, started, receipt = [read(files[k]) for k in ('diagnostic_protocol', 'diagnostic_started', 'diagnostic_receipt')]
    require(protocol['schema']=='stored_finite_segment_diagnostic_protocol_v12' and protocol['frames']==245 and protocol['views']==490 and protocol['source_geometry_rows']==1960 and protocol['source_observation_rows']==735 and protocol['own_arithmetic_started'] is False, 'frozen full-population producer protocol required')
    require(receipt['schema']=='stored_finite_segment_diagnostic_receipt_v12' and receipt['complete'] is True and receipt['passed'] is True and receipt['error'] is None and receipt['cleanup_errors']==[], 'complete producer receipt required')
    require(receipt['new_pose_outputs']==receipt['new_GT_scores']==0 and receipt['latency_claim'] is False and receipt['first_failure_preserved'] is True, 'diagnostic-only receipt required')
    require(receipt['inputs']==started['inputs']==protocol['inputs'] and receipt['policy']==protocol['policy'] and same(receipt['protocol'], current['diagnostic_protocol']) and same(started['protocol'], current['diagnostic_protocol']) and same(receipt['raw_rows'], current['diagnostic_rows']) and started['actual_fit_GT_model_SVD_calls']==0, 'producer protocol/STARTED/rows bytes differ')
    require(protocol['policy']['radius_gate_applied'] is False and protocol['policy']['new_thresholds']==0 and protocol['policy']['branch_consensus_px']==8.0 and protocol['policy']['GT_model_fit_SVD_training_RGB_calls']==0, 'unchanged diagnostic semantics required')
    for key, value in protocol['inputs'].items(): require(same(value, current['parent:'+key]), 'changed producer input '+key)
    seal, cleanup, population = [read(files['parent:'+k]) for k in ('seal', 'receipt', 'population')]
    require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True, 'sealed GT-free source population required')
    require(cleanup['complete'] is True and cleanup['cleanup_completed'] is True and cleanup['error'] is None and cleanup['cleanup_error'] is None and population['complete'] is True and population['passed'] is True and population['frames']==245, 'parent cleanup/population proof required')
    for key, alias in (('protocol','parent_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','receipt'),('parent_population_checks','population')): require(same(seal[key], current['parent:'+alias]), 'parent seal bytes differ '+key)
    failed, review, own, journal = [read(files['parent:'+k]) for k in ('original_failure','review','review_protocol','review_started')]
    require(failed['complete'] is False and failed['passed'] is False and failed['checks']==95 and failed['exception']['message']=="'best_candidate'", 'original checker failure retained')
    require(review['complete'] is True and review['passed'] is True and review['inputs']==own['inputs']==journal['inputs'] and same(review['own_protocol'],current['parent:review_protocol']) and same(journal['own_protocol'],current['parent:review_protocol']), 'separate complete review required')
    require(review['review']==dict(original_checker_changed=False,numerical_core_or_saved_rows_changed=False,AST_optional_best_candidate_access_changes=1), 'only inspector optional-access correction allowed')
    for key,value in review['inputs'].items(): require(same(value,current['parent:review_input:'+key]), 'review source bytes differ '+key)
    return read(files['parent:cohort']), receipt


class Audit:
    def __init__(self): self.checks=0; self.failure_count=0; self.failures=[]
    def check(self, value, label):
        self.checks+=1
        if not value:
            self.failure_count+=1
            if len(self.failures)<100: self.failures.append(label)
    def eq(self, left, right, label): self.check(left==right,label)
    def near(self, left, right, label):
        self.check(type(left) in (int,float) and type(right) in (int,float) and math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-8),label)
    def vector(self, left, right, label):
        self.eq(len(left),len(right),label+'.length')
        for index,(a,b) in enumerate(zip(left,right)): self.near(a,b,label+':'+str(index))


def source_pool(a, original, source, solver, witness, label):
    if witness['contract_error'] is not None:
        a.eq(witness['original_actual_source_lines'],[],label+'.typed_invalid_source_pool')
        a.eq(solver['all_candidate_solutions'],[],label+'.typed_invalid_no_candidate')
        return {}
    used=set(solver['used']); corners={c['id']:c for c in original['corners']}; adopted=source['selected_corner_ids']; consumed=set()
    for corner in adopted:
        if corner in used:
            item=corners[corner];a.eq(item['xy'],source['input_points'][corner],label+'.adopted_xy')
            a.check(len(item['edges'])==2 and len(set(item['edges']))==2 and all(corner in EDGES[e] for e in item['edges']),label+'.incident_edges');consumed.update(item['edges'])
    a.eq(sorted(consumed),witness['consumed_edges'],label+'.consumed'); seen={}; normalized=[]
    for raw in original['lines']:
        edge=raw['edge'];a.eq(raw['endpoints'],list(EDGES[edge]),label+'.endpoints')
        if edge in seen: a.eq(raw,seen[edge],label+'.duplicate');continue
        seen[edge]=raw
        if edge not in SOURCE_EDGES: continue
        normal=raw['normal'];norm=math.hypot(*normal);require(math.isfinite(norm) and norm>0,'invalid original line normal')
        n=[v/norm for v in normal];offset=raw['offset']/norm
        good=math.isfinite(offset) and all(abs(xy[0]*n[0]+xy[1]*n[1]-offset)<=radius+1e-9 for xy,radius in zip(raw['support_points'],raw['query_radii_px']))
        if good and edge not in consumed: normalized.append(dict(edge=edge,endpoints=list(EDGES[edge]),normal=n,offset=offset,support_query_ids=raw['queries'],raw=raw))
    normalized.sort(key=lambda value:value['edge']); saved=witness['original_actual_source_lines']
    a.eq([l['edge'] for l in normalized],[l['edge'] for l in saved],label+'.unused_pool')
    for expected,actual in zip(normalized,saved):
        for key in ('edge','endpoints','support_query_ids','raw'):a.eq(actual[key],expected[key],label+'.source_line.'+key)
        a.vector(actual['normal'],expected['normal'],label+'.normalized_normal');a.near(actual['offset'],expected['offset'],label+'.normalized_offset')
    return {l['edge']:l for l in saved}


def inliers(a, candidate, points, lookup, label):
    point_ids=candidate['fit_input_ids'];edge_ids=candidate['fit_line_edges'];projected=candidate['projected']
    pnorm=[math.hypot(projected[k][0]-points[k][0],projected[k][1]-points[k][1]) for k in point_ids]
    lnorm=[]
    for edge in edge_ids:
        line=lookup[edge];signed=[sum(x*y for x,y in zip(projected[k],line['normal']))-line['offset'] for k in EDGES[edge]]
        lnorm.append(math.sqrt(math.fsum(v*v for v in signed)/2))
    a.vector(candidate['residuals_used_px'],pnorm,label+'.point_residuals');a.vector(candidate['line_residual_RMS_px'],lnorm,label+'.line_residuals')
    a.eq(candidate['final_inliers'],[k for k,v in zip(point_ids,candidate['residuals_used_px']) if v<=8.],label+'.point_inliers')
    a.eq(candidate['final_line_inliers'],[e for e,v in zip(edge_ids,candidate['line_residual_RMS_px']) if v<=8.],label+'.line_inliers')
    return candidate['accepted'] is True and candidate['final_inliers']==point_ids and candidate['final_line_inliers']==edge_ids and all(v>0 for v in candidate['model_camera_depths_m']) and all(candidate['point_line_geometry'][k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'))


def segment(a, candidate, original, saved, label):
    edge=original['edge'];ends=EDGES[edge];first,last=[candidate['projected'][k] for k in ends]
    vx,vy=last[0]-first[0],last[1]-first[1];length=math.hypot(vx,vy);raw=original['raw']
    a.eq(saved['edge'],edge,label+'.edge');a.eq(saved['endpoints'],list(ends),label+'.endpoints');a.eq(saved['projected_endpoints'],[first,last],label+'.projected');a.near(saved['projected_segment_length_px'],length,label+'.length');a.eq(saved['degenerate'],length==0,label+'.degenerate');a.eq(saved['physical_ownership_truth'],'UNKNOWN',label+'.ownership')
    a.eq(len(saved['queries']),len(raw['queries']),label+'.query_count');a.check(len(raw['queries'])==len(raw['support_points'])==len(raw['query_radii_px']) and len(set(raw['queries']))>=3,label+'.support_cardinality');within=[]
    for index,(query,xy,radius,result) in enumerate(zip(raw['queries'],raw['support_points'],raw['query_radii_px'],saved['queries'])):
        item=label+'.query:'+str(index);require(type(radius) in (int,float) and math.isfinite(radius) and radius>=0,'invalid original radius')
        along=0. if length==0 else (xy[0]-first[0])*(vx/length)+(xy[1]-first[1])*(vy/length)
        parameter=0. if length==0 else along/length;bounded=min(length,max(0.,along))
        closest=list(first) if length==0 else [first[0]+bounded*vx/length,first[1]+bounded*vy/length]
        distance=math.hypot(xy[0]-closest[0],xy[1]-closest[1]);normal_distance=None if length==0 else abs((xy[0]-first[0])*(-vy/length)+(xy[1]-first[1])*(vx/length))
        a.eq(result['query'],query,item+'.ID');a.eq(result['support_xy'],xy,item+'.source_xy');a.near(result['radius_px'],radius,item+'.radius');a.near(result['segment_parameter'],parameter,item+'.parameter');a.vector(result['closest_projected_segment_xy'],closest,item+'.closest')
        for key,value in (('closed_segment_distance_px',distance),('distance_minus_radius_px',distance-radius),('before_first_endpoint_px',max(0.,-along)),('after_second_endpoint_px',max(0.,along-length)),('source_line_residual_px',abs(sum(x*y for x,y in zip(xy,original['normal']))-original['offset']))):a.near(result[key],value,item+'.'+key)
        if normal_distance is None:a.eq(result['projected_infinite_line_distance_px'],None,item+'.infinite_degenerate')
        else:a.near(result['projected_infinite_line_distance_px'],normal_distance,item+'.infinite_distance')
        # Exact boundary decisions use the producer's independently checked scalar,
        # avoiding a verification libm rounding difference changing a <= predicate.
        expected=result['closed_segment_distance_px']<=radius;a.eq(result['within_saved_radius'],expected,item+'.within_radius');within.append(expected)
    result=all(within);a.eq(saved['all_support_within_saved_radius'],result,label+'.all_within');return result,len(saved['queries'])


def packet(a, result, pair, ledger, source, original):
    label=result['id'];solver=ledger['full_rank_only_solver'];filtering=ledger['observation_filter']
    a.eq([r['method'] for r in pair],list(METHODS),label+'.view_order');a.check(all(r['id']==result['id']==ledger['id']==source['id']==original['id'] and r['session']==result['session']==ledger['session']==source['session']==original['session'] for r in pair),label+'.frame_session')
    for key,value in (('parent_geometry_semantic_sha256',semantic(source)),('parent_observation_semantic_sha256',semantic(original)),('ledger_semantic_sha256',semantic(ledger)),('geometry_row_semantic_sha256',[semantic(r) for r in pair])):a.eq(result[key],value,label+'.lineage.'+key)
    a.eq(filtering['original_observation_semantic_sha256'],semantic(original),label+'.original_flat_observation');a.eq(result['observation_filter'],filtering,label+'.filter_witness');a.eq(result['used'],solver['used'],label+'.used');a.eq(result['fit_line_edges'],solver['line_edges'],label+'.fit_edges');a.eq(result['H'],source['hidden_initial'],label+'.H');a.check(set(solver['fit_input_ids']).isdisjoint(result['H']),label+'.H_not_fit');a.eq(original['GT_input'],False,label+'.GT_free')
    a.eq(result['view_statuses'],{r['method']:r['output_status'] for r in pair},label+'.view_statuses');a.eq(result['proposal_state'],solver['state'],label+'.proposal_state');a.eq(result['no_outputs_changed'],True,label+'.no_outputs_changed')
    a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_fit');a.eq(ledger['reported_output_views'],2,label+'.two_views');a.eq(semantic(solver),ledger['fresh_filtered_solver_semantic_sha256'],label+'.solver_SHA');a.eq(solver,pair[0]['solver'],label+'.rank_solver')
    for r in pair:
        a.eq(r['solver']['all_candidate_solutions'],solver['all_candidate_solutions'],label+'.shared_candidates')
        for key in ('K','xyz','input_points','hidden_initial'):a.eq(r[key],source[key],label+'.source.'+key)
        a.eq(r['original_native_N3_points'],original['native_N3_points'],label+'.native_source');a.eq(r['parent_geometry_row_semantic_sha256'],semantic(source),label+'.parent_geometry');a.eq(r['parent_observation_row_semantic_sha256'],semantic(original),label+'.parent_observation')
    filtered=copy.deepcopy(original);filtered['lines']=[line for line in original['lines'] if line['edge'] not in filtering['rejected_edge_ids']]
    a.eq(semantic(filtered),filtering['filtered_observation_semantic_sha256'],label+'.filtered_observation');a.check(set(filtering['consumed_edges']).isdisjoint(filtering['rejected_edge_ids']),label+'.no_double_count');a.eq(solver['line_edges'],filtering['retained_unused_edge_ids'],label+'.retained')
    lookup=source_pool(a,original,source,solver,filtering,label)
    for line in lookup.values():
        for query,xy in zip(line['raw']['queries'],line['raw']['support_points']):
            item=original['queries'][query];a.eq(item['edge'],query//7,label+'.query_edge');a.eq(query//7,line['edge'],label+'.source_edge');a.eq(item['selected_xy'],xy,label+'.query_actual');a.eq(item['reason'],'CALIBRATED_QUERY_OBSERVATION',label+'.query_reason');a.eq(item['calibrated_model_coverage'],True,label+'.query_coverage')
    candidates=solver['all_candidate_solutions'];a.eq(len(result['candidates']),len(candidates),label+'.candidate_count');a.check(len({c['candidate_index'] for c in candidates})==len(candidates),label+'.candidate_unique')
    full={c['candidate_index']:inliers(a,c,source['input_points'],lookup,label+'.candidate:'+str(c['candidate_index'])) for c in candidates};best=solver.get('best_candidate');alternatives={c['candidate_index']:c for c in solver['alternatives']};counts=Counter();any_ambiguity=False;selected_outside=False
    if best is not None:a.eq(best,next((c for c in candidates if c['candidate_index']==best['candidate_index']),None),label+'.best_candidate_identity')
    for source_candidate,saved in zip(candidates,result['candidates']):
        index=source_candidate['candidate_index'];item=label+'.candidate:'+str(index);a.eq(saved['stored_candidate'],source_candidate,item+'.full_witness')
        for key in ('candidate_index','dimension_index','start_index'):a.eq(saved[key],source_candidate[key],item+'.'+key)
        selected=best is not None and index==best['candidate_index'];relation=alternatives.get(index);ambiguous=bool(best is not None and relation is not None and relation['numerically_equivalent_to_selected'] is False and full[best['candidate_index']] and full[index]);a.eq(saved['is_selected'],selected,item+'.selected');a.eq(saved['alternative_relation'],relation,item+'.alternative');a.eq(saved['all_same_actual_factors_8px_inliers'],full[index],item+'.full8');a.eq(saved['observation_consensus_ambiguity_with_selected'],ambiguous,item+'.ambiguity');any_ambiguity|=ambiguous
        a.eq([l['edge'] for l in saved['finite_segments']],source_candidate['fit_line_edges'],item+'.finite_edges');a.eq(source_candidate['fit_input_ids'],solver['used'],item+'.factor_points');a.eq(source_candidate['fit_line_edges'],solver['line_edges'],item+'.factor_edges');counts['stored_candidates']+=1;counts['non_equivalent_full8px_alternatives']+=int(ambiguous)
        for finite_line in saved['finite_segments']:
            within,n=segment(a,source_candidate,lookup[finite_line['edge']],finite_line,item+'.line:'+str(finite_line['edge']));counts['candidate_lines']+=1;counts['support_queries']+=n;counts['candidate_lines_exceed_saved_radius']+=int(not within);selected_outside|=selected and not within
    a.eq(result['has_observation_consensus_ambiguity'],any_ambiguity,label+'.frame_ambiguity');a.eq(result['selected_candidate_exceeds_saved_radius'],selected_outside,label+'.frame_selected_radius');counts['frames_with_observation_consensus_ambiguity']+=int(any_ambiguity);counts['frames_with_selected_candidate_exceeding_saved_radius']+=int(selected_outside);return counts


def guard(output, freeze):
    output=Path(output).resolve();require(output==DOC.resolve() and output.is_dir(),'new V12 supplemental output only')
    for name in FILES if freeze else FILES[1:]: require(not (output/name).exists(),'preserve '+name)
    return output


def freeze(args):
    output=guard(args.output,True);files=paths(args.input);current={k:binding(v) for k,v in files.items()};authority(files,current)
    write(output/FILES[0],dict(schema='independent_stored_finite_segment_scalar_protocol_v12',inputs=current,policy=POLICY,frames=245,own_scalar_arithmetic_started=False))


def run(args):
    output=guard(args.output,False);files=paths(args.input);current={k:binding(v) for k,v in files.items()};own=read(output/FILES[0]);require(own['inputs']==current and own['policy']==POLICY,'own frozen closure changed');protocol=binding(output/FILES[0]);write(output/FILES[1],dict(protocol=protocol,inputs=current,policy=POLICY,actual_fit_GT_model_projection_J_SVD_calls=0))
    a=Audit();counts=Counter();statuses={m:Counter() for m in METHODS};exception=None;bindings_unchanged=False
    try:
        cohort,receipt=authority(files,current);streams={key:iter(rows(files[key])) for key in ('diagnostic_rows','parent:geometry','parent:ledgers','parent:source_geometry','parent:source_observations')};ids=[]
        for frame in cohort['frames']:
            result=next(streams['diagnostic_rows']);pair=[next(streams['parent:geometry']) for _ in range(2)];ledger=next(streams['parent:ledgers']);geometry=[next(streams['parent:source_geometry']) for _ in range(8)];observations=[next(streams['parent:source_observations']) for _ in range(3)]
            a.check(all(r['id']==frame['id'] for r in [result,ledger]+pair+geometry+observations),frame['id']+'.population_ID');a.eq({r['method'] for r in geometry},{'IMAGE_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_BOUNDARY_ONLY','IMAGE_NO_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_CORNERWISE_HYBRID','IMAGE_NO_ROLE_CORNERWISE_HYBRID','IMAGE_ROLE_CORNERWISE_HYBRID','N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK'},frame['id']+'.parent_methods');a.eq({r['head_arm'] for r in observations},{'GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'},frame['id']+'.parent_heads')
            source=[r for r in geometry if r['method']=='IMAGE_ROLE_BOUNDARY_ONLY'];original=[r for r in observations if r['head_arm']=='IMAGE_ROLE'];require(len(source)==len(original)==1,'one source row and ROLE supply required');counts.update(packet(a,result,pair,ledger,source[0],original[0]));ids.append(frame['id']);counts.update(frames=1,geometry_rows=2,ledger_rows=1,source_geometry_rows=8,source_observation_rows=3)
            for r in pair: statuses[r['method']][r['output_status']]+=1
        a.eq(ids,cohort['ids'],'full245.ids');a.eq(len(set(ids)),245,'full245.unique');a.eq(Counter(f['label'] for f in cohort['frames']),dict(clean=153,moderate=92),'cohort strata')
        for key,stream in streams.items():a.eq(next(stream,None),None,'no_extra_rows.'+key)
        # Counter omits zero entries; comparison normalizes the union of keys.
        a.eq({k:counts[k] for k in set(counts)|set(receipt['actual_counts'])},{k:receipt['actual_counts'].get(k,0) for k in set(counts)|set(receipt['actual_counts'])},'all_producer_counts')
        a.eq({m:dict(v) for m,v in statuses.items()},receipt['view_statuses'],'all_view_statuses');bindings_unchanged={k:binding(v) for k,v in files.items()}==current and binding(output/FILES[0])==protocol;a.eq(bindings_unchanged,True,'all_inputs_and_own_protocol_unchanged')
    except BaseException as error:exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    complete=exception is None and counts['frames']==245;passed=complete and a.failure_count==0
    write(output/FILES[2],dict(schema='independent_stored_finite_segment_scalar_checks_v12',complete=complete,passed=passed,own_protocol=protocol,inputs=current,policy=POLICY,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=exception,independent_counts=dict(counts),view_statuses={m:dict(v) for m,v in statuses.items()},all_saved_outputs_unchanged=bindings_unchanged,new_pose_outputs=0,new_GT_scores=0,actual_fit_model_projection_J_SVD_RGB_calls=0,limitations=['radii are source-CAL uncertainty surrogates, not certified physical truth','same-factor 8px alternatives do not establish which physical branch is correct','stored projections/ranks/depths and source filtering retain the separate parent review authority','finite support outside radius is not an ownership label or pose-accuracy verdict']))
    if not passed:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'));parser.add_argument('--input',type=Path,default=DOC);parser.add_argument('--output',type=Path,default=DOC)
    args=parser.parse_args();(freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
