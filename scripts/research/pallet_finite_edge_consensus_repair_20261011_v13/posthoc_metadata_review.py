"""Separate actual per-row factor facts from preserved generic contract markers.

Only saved metadata counts/IDs are joined. No pose errors, moments, bootstrap,
models, GT, fitting, Jacobians, SVD or operating policy are calculated or changed.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
PRIVATE=Path('/tmp/pallet-finite-edge-consensus-private-20261011-v13')
FILES=('POSTHOC_METADATA_PROTOCOL.json','POSTHOC_METADATA_STARTED.json','POSTHOC_METADATA_CHECKS.json','POSTHOC_METADATA_ROWS.jsonl.gz')
METHOD='FINITE_EDGE_CONSENSUS'


def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:require(bool(line.strip()),'blank saved row');yield json.loads(line)
def binding(path):
    p=Path(path).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input')
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=digest.hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def inputs(p):
    names=('POSTHOC_ROWS.jsonl.gz','GEOMETRY_SEALED.jsonl.gz','METRICS.json','STATISTICS_AUTHORITY_SUPPLEMENT.json',
        'REVIEW_STATISTICS_STARTED.json','STATISTICS_STARTED.json','STATISTICS_FIRST_FAILURE.json','GEOMETRY_SEAL.json')
    paths={name:p/name for name in names};paths.update(code=Path(__file__),statistics_review=CODE/'statistics_review.py')
    return paths

def authority(p,b):
    metrics=read(p/'METRICS.json');seal=read(p/'GEOMETRY_SEAL.json');supplement=read(p/'STATISTICS_AUTHORITY_SUPPLEMENT.json')
    require(metrics['schema']=='fixed_repair_raw_row_metrics_v13' and metrics['complete'] is True
        and metrics['actual_arithmetic']['posthoc_rows']==245 and seal['complete'] is True and seal['rows']==245,
        'complete saved statistics and sealed245 geometry required')
    require(all(seal['geometry'][k]==b['GEOMETRY_SEALED.jsonl.gz'][k] for k in ('sha256','bytes')),
        'actual sealed geometry bytes differ')
    require(all(metrics['posthoc_rows'][k]==b['POSTHOC_ROWS.jsonl.gz'][k] for k in ('sha256','bytes')),
        'actual POSTHOC row bytes differ')
    require(all(supplement['inputs']['wrapper'][k]==b['statistics_review'][k] for k in ('sha256','bytes')),
        'preserved original statistics-review source differs')
    require(read(p/'REVIEW_STATISTICS_STARTED.json')==read(p/'STATISTICS_STARTED.json'),'preserved/reviewed STARTED inputs differ')
    ids=metrics['strata']['combined']['ids'];require(len(ids)==len(set(ids))==245,'same245 IDs required');return ids

def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directories required')
    out=args.output.resolve();require(out==DOC or out.is_relative_to(PRIVATE),'exclusive V13 output required')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)

def freeze(args):
    guard(args,'freeze');b={k:binding(p) for k,p in inputs(args.input).items()};ids=authority(args.input,b)
    write(args.output/FILES[0],dict(schema='saved_finite_factor_metadata_interpretation_protocol_v13',inputs=b,ordered_ids=ids,
        original_contract_markers_and_rows_changed=False,new_model_GT_error_moments_bootstrap_fit_J_SVD_draw_calls=0))

def run(args):
    guard(args,'run');b={k:binding(p) for k,p in inputs(args.input).items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0])
    ids=authority(args.input,b);require(own['inputs']==b and own['ordered_ids']==ids,'own fixed metadata inputs differ')
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0))
    emitted=[];counts=Counter();error=None
    try:
        posthoc=list(rows(args.input/'POSTHOC_ROWS.jsonl.gz'));geometry=rows(args.input/'GEOMETRY_SEALED.jsonl.gz')
        require([r['id'] for r in posthoc]==ids,'same ordered245 POSTHOC required')
        for index,row in enumerate(geometry):
            require(index<245 and row['id']==ids[index] and row['method']==METHOD,'sealed ordered245 method differs')
            d=posthoc[index]['finite_edge_diagnostics'];s=row['solver'];new=bool(row['new_pose_estimated']);g=s.get('point_line_geometry') if new else None
            points=list(s['fit_input_ids']) if new else [];edges=list(s['fit_line_edges']) if new else []
            require(d['numeric_finite_NEW']==new and d['accepted_fit_point_ids']==points and d['accepted_fit_line_edges']==edges,
                'stored actual accepted fit metadata differs')
            scalar=len(g['actual_scalar_residuals_px']) if g else None
            hinges=sum(len(block['support_query_ids']) for block in g['edge_blocks']) if g else 0
            if new:
                require(g['point_ids']==points and g['line_edges']==edges and [block['edge'] for block in g['edge_blocks']]==edges,
                    'actual finite geometry factor identity differs')
                require(scalar==2*(len(points)+len(edges))+hinges and d['actual_fit_scalar_residuals']==scalar,
                    'actual finite scalar rows include all per-edge support hinges')
            differs=(set(s['used'])!=set(points) or set(s['line_edges'])!=set(edges)) if new else None
            record=dict(id=row['id'],session=row['session'],method=METHOD,output_status=row['output_status'],numeric_NEW=new,
                accepted_fit_point_ids=points,accepted_fit_line_edges=edges,actual_fit_scalar_rows=scalar,
                actual_accepted_hinge_rows=hinges,finite_hinge_rows_present_in_actual_accepted_fit=bool(hinges),
                scalar_rows_equal_old_2_times_points_plus_edges=scalar==2*(len(points)+len(edges)) if new else None,
                whole_scoring_pool_differs_from_actual_fit_by_factor_IDs=differs,
                supplied_point_ids=list(s['used']),supplied_line_edges=list(s['line_edges']),
                preserved_generic_contract_markers={k:d[k] for k in ('finite_hinge_residuals_included',
                    'scalar_rows_are_not_2_times_points_plus_edges','whole_pool_scoring_is_not_all_actual_fit_observations')},
                generic_markers_are_not_per_row_positive_observation_claims=True,new_fit_score_GT_or_pose_errors=0)
            emitted.append(record);counts['rows']+=1;counts['NEW' if new else 'not_NEW']+=1
            counts['actual_accepted_hinge_rows_present']+=bool(hinges);counts['actual_accepted_point_only']+=bool(new and not edges)
            counts['whole_pool_differs_by_factor_IDs']+=bool(differs)
        require(len(emitted)==245,'full245 metadata rows required')
        require({k:binding(p) for k,p in inputs(args.input).items()}==b,'saved inputs changed')
        with (args.output/FILES[3]).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            for row in emitted:stream.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and len(emitted)==245
    write(args.output/FILES[2],dict(schema='saved_finite_factor_metadata_interpretation_checks_v13',complete=passed,passed=passed,
        own_protocol=ob,inputs=b,checker=b['code'],counts=dict(counts),error=error,
        rows=binding(args.output/FILES[3]) if (args.output/FILES[3]).is_file() else None,
        original_rows_statistics_contract_markers_modified=False,new_model_GT_error_moments_bootstrap_fit_J_SVD_draw_calls=0))
    if not passed:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name in ('input','output'):p.add_argument('--'+name,type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
