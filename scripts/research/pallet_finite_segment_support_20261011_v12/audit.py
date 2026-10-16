"""Own-frozen stdlib diagnosis; no fit, model, GT, SVD or output-policy change."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_finite_segment_support_20261011_v12'
PARENT=REPO/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
SOURCE=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
METHODS=('FILTERED_LOCAL','FILTERED_LOCAL_WEAK_SAFE')
FILES=('DIAGNOSTIC_PROTOCOL.json','DIAGNOSTIC_STARTED.json','SEGMENT_ROWS.jsonl.gz','DIAGNOSTIC_RECEIPT.json')
POLICY=dict(diagnostic_only=True,solver_and_output_unchanged=True,GT_model_fit_SVD_training_RGB_calls=0,
    distance='Euclidean distance of actual support query to closed projected registered edge segment',
    radius='unchanged source_line.query_radii_px; diagnostic uncertainty surrogate, not certified physical truth',
    radius_gate_applied=False,new_thresholds=0,branch_consensus_px=8.0,full_population_retained=True,
    ambiguity='non-equivalent stored alternative with all same full actual factors as 8px inliers, positive depth and both rank6',
    ambiguity_is_not_numerical_equality=True,ambiguity_does_not_choose_pose_or_claim_wrong_correspondence=True)

def require(value,message):
    if not value:raise ValueError(message)

def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def semantic(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def binding(path):
    p=Path(path).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    require(p.is_relative_to(REPO),'public immutable evidence only');return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def inputs(args):
    p=args.parent;s=args.source;paths=dict(code=Path(__file__),contract=DOC/'DIAGNOSTIC_CONTRACT_KO.md',cohort=args.cohort,
        geometry=p/'GEOMETRY_SEALED.jsonl.gz',ledgers=p/'CONTROL_LEDGERS.jsonl.gz',parent_protocol=p/'PROTOCOL.json',
        seal=p/'GEOMETRY_SEAL.json',receipt=p/'CONTROL_RECEIPT.json',population=p/'PARENT_POPULATION_CHECKS.json',
        review=p/'REVIEW_VALIDATION_CHECKS.json',review_protocol=p/'REVIEW_VALIDATION_PROTOCOL.json',review_started=p/'REVIEW_VALIDATION_STARTED.json',
        original_failure=p/'VALIDATION_CHECKS.json',original_protocol=p/'VALIDATION_PROTOCOL.json',
        source_geometry=s/'GEOMETRY_SEALED.jsonl.gz',source_observations=s/'OBSERVATIONS.jsonl.gz',source_protocol=s/'PROTOCOL.json',source_seal=s/'GEOMETRY_SEAL.json')
    for k,v in read(paths['review'])['inputs'].items():
        rel=Path(v['path']);require(not rel.is_absolute() and '..' not in rel.parts,'public review source required');paths['review_input:'+k]=REPO/rel
    return paths

def authority(paths,b):
    seal,receipt,review,proto,failed,pop,started=[read(paths[k]) for k in ('seal','receipt','review','review_protocol','original_failure','population','review_started')]
    require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'complete245/490 parent seal required')
    require(receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,'parent cleanup required')
    for key,alias in (('protocol','parent_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'sealed bytes '+key)
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245,'full source population required')
    require(failed['complete'] is False and failed['passed'] is False and failed['checks']==95 and failed['exception']['message']=="'best_candidate'",'original inspector failure retained')
    require(review['complete'] is True and review['passed'] is True and review['inputs']==proto['inputs'] and same(review['own_protocol'],b['review_protocol']) and started['inputs']==proto['inputs'] and same(started['own_protocol'],b['review_protocol']),'actual separate full review PASS required')
    require(review['review']==dict(original_checker_changed=False,numerical_core_or_saved_rows_changed=False,AST_optional_best_candidate_access_changes=1),'one optional inspector access only')
    for key,value in review['inputs'].items():require(same(value,b['review_input:'+key]),'review current bytes '+key)
    return read(paths['cohort'])

def guard(args,stage):
    out=args.output.resolve();require(out==DOC.resolve() and out.is_dir(),'new V12 DOC output only')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (out/name).exists(),'preserve '+name)
    for name in ('PENDING_SEGMENT_ROWS.jsonl.gz','INTERRUPTED_SEGMENT_ROWS.jsonl.gz'):require(not (out/name).exists(),'preserve '+name)
    return out

def freeze(args):
    out=guard(args,'freeze');paths=inputs(args);b={k:binding(p) for k,p in paths.items()};authority(paths,b)
    write(out/FILES[0],dict(schema='stored_finite_segment_diagnostic_protocol_v12',inputs=b,policy=POLICY,frames=245,views=490,source_geometry_rows=1960,source_observation_rows=735,own_arithmetic_started=False))

def support(candidate,line):
    edge=int(line['edge']);a,b=EDGES[edge];p,q=[candidate['projected'][k] for k in (a,b)];dx,dy=q[0]-p[0],q[1]-p[1];square=dx*dx+dy*dy
    require(all(math.isfinite(v) for v in (*p,*q,square)),'nonfinite projected segment');length=math.sqrt(square);values=[]
    raw=line['raw']
    for query,xy,radius in zip(raw['queries'],raw['support_points'],raw['query_radii_px']):
        require(math.isfinite(radius) and radius>=0,'invalid saved radius');t=0. if square==0 else ((xy[0]-p[0])*dx+(xy[1]-p[1])*dy)/square;clamped=min(1.,max(0.,t));closest=[p[0]+clamped*dx,p[1]+clamped*dy]
        distance=math.hypot(xy[0]-closest[0],xy[1]-closest[1]);signed=t*length
        values.append(dict(query=query,support_xy=xy,radius_px=radius,segment_parameter=t,closest_projected_segment_xy=closest,closed_segment_distance_px=distance,distance_minus_radius_px=distance-radius,
            before_first_endpoint_px=max(0.,-signed),after_second_endpoint_px=max(0.,signed-length),within_saved_radius=distance<=radius,
            projected_infinite_line_distance_px=None if not length else abs(dx*(xy[1]-p[1])-dy*(xy[0]-p[0]))/length,
            source_line_residual_px=abs(xy[0]*line['normal'][0]+xy[1]*line['normal'][1]-line['offset'])))
    return dict(edge=edge,endpoints=[a,b],projected_endpoints=[p,q],projected_segment_length_px=length,degenerate=not square,queries=values,all_support_within_saved_radius=all(v['within_saved_radius'] for v in values),physical_ownership_truth='UNKNOWN')

def full_support(c):
    return c['accepted'] is True and c['final_inliers']==c['fit_input_ids'] and c['final_line_inliers']==c['fit_line_edges'] and all(v>0 for v in c['model_camera_depths_m']) and all(c['point_line_geometry'][k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'))

def packet(pair,ledger,source,obs):
    require([r['method'] for r in pair]==list(METHODS) and all(r['id']==ledger['id']==source['id']==obs['id'] and r['session']==ledger['session']==source['session']==obs['session'] for r in pair),'same frame/session')
    require(all(r['parent_geometry_row_semantic_sha256']==semantic(source) and r['parent_observation_row_semantic_sha256']==semantic(obs) for r in pair),'original source lineage')
    original=obs;f=ledger['observation_filter'];s=ledger['full_rank_only_solver'];require(obs['GT_input'] is False and f['original_observation_semantic_sha256']==semantic(original) and semantic(s)==ledger['fresh_filtered_solver_semantic_sha256']==semantic(pair[0]['solver']),'same original flat GT-free observations and single proposal')
    require(all(r['K']==source['K'] and r['xyz']==source['xyz'] and r['input_points']==source['input_points'] and r['hidden_initial']==source['hidden_initial'] and r['original_native_N3_points']==obs['native_N3_points'] for r in pair),'original camera/dimensions/sparse/native/H identity')
    require(pair[0]['solver']['all_candidate_solutions']==pair[1]['solver']['all_candidate_solutions'] and ledger['actual_filtered_local_solve_calls']==1 and ledger['reported_output_views']==2,'one fit/two views')
    require(set(s['fit_input_ids']).isdisjoint(pair[0]['hidden_initial']),'H excluded from final fit');lookup={l['edge']:l for l in f['original_actual_source_lines']};raw={l['edge']:l for l in original['lines']}
    for e,line in lookup.items():
        require(line['raw']==raw[e] and line['endpoints']==list(EDGES[e]),'actual physical/CAL line identity');r=line['raw'];require(len(r['queries'])==len(r['support_points'])==len(r['query_radii_px']) and len(set(r['queries']))>=3,'original support cardinality')
        require(all(original['queries'][i]['edge']==i//7==e and original['queries'][i]['selected_xy']==xy and original['queries'][i]['reason']=='CALIBRATED_QUERY_OBSERVATION' and original['queries'][i]['calibrated_model_coverage'] is True for i,xy in zip(r['queries'],r['support_points'])),'original selected CAL query coordinates and coverage')
    require(set(f['rejected_edge_ids']).isdisjoint(f['consumed_edges']) and s['line_edges']==f['retained_unused_edge_ids'],'consumed/rejected/fit identities')
    candidates=[];best=s.get('best_candidate');alternatives={a['candidate_index']:a for a in s['alternatives']}
    for c in s['all_candidate_solutions']:
        require(c['fit_line_edges']==s['line_edges'] and c['fit_input_ids']==s['used'],'candidate actual factor identities');a=alternatives.get(c['candidate_index']);ambiguous=bool(best is not None and a is not None and a['numerically_equivalent_to_selected'] is False and full_support(best) and full_support(c))
        candidates.append(dict(candidate_index=c['candidate_index'],dimension_index=c['dimension_index'],start_index=c['start_index'],is_selected=best is not None and c['candidate_index']==best['candidate_index'],stored_candidate=c,
            alternative_relation=a,all_same_actual_factors_8px_inliers=full_support(c),observation_consensus_ambiguity_with_selected=ambiguous,
            finite_segments=[support(c,lookup[e]) for e in c['fit_line_edges']]))
    return dict(id=ledger['id'],session=ledger['session'],view_statuses={r['method']:r['output_status'] for r in pair},proposal_state=s['state'],used=s['used'],fit_line_edges=s['line_edges'],H=pair[0]['hidden_initial'],candidates=candidates,
        has_observation_consensus_ambiguity=any(c['observation_consensus_ambiguity_with_selected'] for c in candidates),
        selected_candidate_exceeds_saved_radius=any(c['is_selected'] and any(not l['all_support_within_saved_radius'] for l in c['finite_segments']) for c in candidates),
        observation_filter=f,parent_geometry_semantic_sha256=semantic(source),parent_observation_semantic_sha256=semantic(obs),ledger_semantic_sha256=semantic(ledger),geometry_row_semantic_sha256=[semantic(r) for r in pair],no_outputs_changed=True)

def run(args):
    out=guard(args,'run');paths=inputs(args);b={k:binding(p) for k,p in paths.items()};own=read(out/FILES[0]);require(own['inputs']==b and own['policy']==POLICY,'own frozen inputs/policy');cohort=authority(paths,b);pb=binding(out/FILES[0]);write(out/FILES[1],dict(protocol=pb,inputs=b,actual_fit_GT_model_SVD_calls=0))
    streams={k:iter(rows(paths[k])) for k in ('geometry','ledgers','source_geometry','source_observations')};counts=Counter();statuses={m:Counter() for m in METHODS};error=None;cleanup=[];raw=None;zipped=None;began=time.monotonic()
    try:
        raw=(out/'PENDING_SEGMENT_ROWS.jsonl.gz').open('xb');zipped=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        for frame in cohort['frames']:
            pair=[next(streams['geometry']) for _ in range(2)];ledger=next(streams['ledgers']);g=[next(streams['source_geometry']) for _ in range(8)];o=[next(streams['source_observations']) for _ in range(3)]
            require(all(r['id']==frame['id'] for r in pair+[ledger]+g+o) and {r['head_arm'] for r in o}=={'GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'} and len({r['method'] for r in g})==8,'full original ordered frame population')
            source=[r for r in g if r['method']=='IMAGE_ROLE_BOUNDARY_ONLY'];obs=[r for r in o if r['head_arm']=='IMAGE_ROLE'];require(len(source)==len(obs)==1,'one original primary and ROLE supply');result=packet(pair,ledger,source[0],obs[0]);zipped.write((json.dumps(result,separators=(',',':'),allow_nan=False)+'\n').encode());counts['frames']+=1;counts['geometry_rows']+=2;counts['ledger_rows']+=1;counts['source_geometry_rows']+=8;counts['source_observation_rows']+=3
            for r in pair:statuses[r['method']][r['output_status']]+=1
            counts['frames_with_observation_consensus_ambiguity']+=int(result['has_observation_consensus_ambiguity']);counts['frames_with_selected_candidate_exceeding_saved_radius']+=int(result['selected_candidate_exceeds_saved_radius'])
            for c in result['candidates']:
                counts['stored_candidates']+=1;counts['non_equivalent_full8px_alternatives']+=int(c['observation_consensus_ambiguity_with_selected'])
                for l in c['finite_segments']:counts['candidate_lines']+=1;counts['support_queries']+=len(l['queries']);counts['candidate_lines_exceed_saved_radius']+=int(not l['all_support_within_saved_radius'])
        require(counts['frames']==245 and [f['id'] for f in cohort['frames']]==cohort['ids'] and len(set(cohort['ids']))==245 and Counter(f['label'] for f in cohort['frames'])==dict(clean=153,moderate=92),'full245 cohort')
        for key,it in streams.items():require(next(it,None) is None,'extra saved population '+key)
        require({k:binding(p) for k,p in paths.items()}==b and binding(out/FILES[0])==pb,'all frozen bytes unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for phase,action in [('gzip_close',None if zipped is None else zipped.close),('raw_flush',None if raw is None else raw.flush),('raw_fsync',None if raw is None else lambda:os.fsync(raw.fileno())),('raw_close',None if raw is None else raw.close)]:
            if action is not None:
                try:action()
                except BaseException as exc:cleanup.append(dict(phase=phase,type=type(exc).__name__,message=str(exc)))
    complete=error is None and not cleanup and counts['frames']==245;pending=out/'PENDING_SEGMENT_ROWS.jsonl.gz';destination=out/(FILES[2] if complete else 'INTERRUPTED_SEGMENT_ROWS.jsonl.gz')
    if pending.exists():
        try:require(not destination.exists(),'preserve existing stream destination');os.rename(pending,destination)
        except BaseException as exc:cleanup.append(dict(phase='stream_publish_or_prefix_preservation',type=type(exc).__name__,message=str(exc)));complete=False
    stored=destination if destination.exists() else pending
    write(out/FILES[3],dict(schema='stored_finite_segment_diagnostic_receipt_v12',complete=complete,passed=complete,protocol=pb,inputs=b,policy=POLICY,actual_counts=dict(counts),view_statuses={m:dict(v) for m,v in statuses.items()},error=error,cleanup_errors=cleanup,raw_rows=binding(stored) if stored.exists() else None,wall_seconds=time.monotonic()-began,first_failure_preserved=True,new_pose_outputs=0,new_GT_scores=0,latency_claim=False))
    if not complete:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name,default in (('parent',PARENT),('source',SOURCE),('cohort',COHORT),('output',DOC)):p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
