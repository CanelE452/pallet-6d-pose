"""Verify complete preserved figures after the renderer's shadowed-name failure.

No render, image decode, score, model, fit, GT or metric arithmetic is repeated.
The original failure stays false; this independently bound artifact supplement
checks its complete eight PNGs and 24 saved display panels against old inputs.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
PRIVATE=Path('/tmp/pallet-finite-edge-consensus-private-20261011-v13')
RENDER=CODE/'render.py'
RENDER_SHA='ba663cf2db16b339b42b561a3aac77c6c480b936c55c5e710c867d6bac9845e7'
FILES=('FIGURE_COMPLETION_PROTOCOL.json','FIGURE_COMPLETION_STARTED.json','SUPPLEMENTAL_FIGURE_CHECKS.json')
METHODS=('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_LOCAL_POINT_LINE','FINITE_EDGE_CONSENSUS')
SHOWN=(METHODS[0],METHODS[1],METHODS[2],METHODS[4])


def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:require(bool(line.strip()),'blank saved row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a.get(k)==b.get(k) for k in ('sha256','bytes'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def source_witness():
    require(binding(RENDER)['sha256']==RENDER_SHA,'unchanged failed renderer source required')
    tree=ast.parse(RENDER.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    initial=ast.parse('paths,b=inputs(args)').body[0];shadow=ast.parse('a,b=EDGES[edge]').body[0]
    equal=ast.parse('inputs(args)[1]==b',mode='eval').body
    for label,node in (('initial_bindings',initial),('inner_edge_shadow',shadow),('final_bindings_comparison',equal)):
        require(sum(ast.dump(n,include_attributes=False)==ast.dump(node,include_attributes=False) for n in ast.walk(run))==1,
            'exact frozen name-shadow source witness '+label)
    display=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='display_row')
    module=ast.Module(body=[display],type_ignores=[])
    return module,dict(renderer=binding(RENDER),shadowed_name='b',declared_shadow_sites=3,
        unchanged_display_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())

def inputs(args):
    p=args.input;figure=read(p/'FIGURE_PROTOCOL.json');case=read(REPO/figure['inputs']['case_protocol']['path']);paths={}
    for key,value in figure['inputs'].items():
        if key.startswith('image:'):path=args.source_root/case['cases'][int(key.split(':')[1])]['image']['path']
        else:
            path=Path(value['path']);require(not path.is_absolute() and '..' not in path.parts,'public source authority required');path=REPO/path
        require(same(binding(path),value),'unchanged frozen figure input '+key);paths['frozen:'+key]=path
    paths.update(checker=Path(__file__),figure_protocol=p/'FIGURE_PROTOCOL.json',original_failure=p/'FIGURES_FAILURE.json',
        figure_started=p/'FIGURES_STARTED.json',case_rows=p/'FIGURE_CASE_ROWS.jsonl.gz')
    failure=read(paths['original_failure'])
    require(failure['complete'] is False and failure['completed_case_panels']==24 and failure['error']['type']=='RuntimeError'
        and failure['error']['message']=='SAME_OBSERVATION_GUARD: display inputs changed','preserve specific original end-binding failure')
    prefixes=failure['preserved_PNG_prefix'];require(len(prefixes)==8,'original full eight-PNG prefix required')
    for i,value in enumerate(prefixes):
        path=REPO/value['path'];require(same(binding(path),value),'unchanged produced PNG '+str(i));paths['PNG:'+str(i)]=path
    require(set(q.name for q in (p/'figures').glob('*.png'))=={Path(v['path']).name for v in prefixes},'exact complete eight PNGs')
    started=read(paths['figure_started']);require(same(started['protocol'],binding(paths['figure_protocol']))
        and started['configured_PNGs']==8 and started['configured_panels']==24 and started['scored_rows']==1225,
        'original figure STARTED authority differs')
    require(figure['schema']=='prespecified_six_case_display_v13' and figure['methods']==list(METHODS)
        and figure['display_methods']==list(SHOWN) and figure['cases']==6 and figure['panels']==24
        and figure['selection_uses_new_scores'] is False,'original preselected display protocol differs')
    return paths,case

def guard(args,stage):
    for p in (args.input,args.output,args.source_root):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directory required')
    out=args.output.resolve();require(out==DOC or out.is_relative_to(PRIVATE),'exclusive V13 output required')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)

def freeze(args):
    guard(args,'freeze');paths,_=inputs(args);_,witness=source_witness()
    write(args.output/FILES[0],dict(schema='preserved_complete_figure_artifact_review_protocol_v13',inputs={k:binding(p) for k,p in paths.items()},
        source_witness=witness,source_root=str(args.source_root.resolve()),new_render_decode_model_GT_score_fit_metric_draw_calls=0))

def run(args):
    guard(args,'run');paths,case=inputs(args);b={k:binding(p) for k,p in paths.items()};own=read(args.output/FILES[0]);module,witness=source_witness()
    require(own['inputs']==b and own['source_witness']==witness and own['source_root']==str(args.source_root.resolve()),'own artifact-review freeze differs')
    write(args.output/FILES[1],dict(own_protocol=binding(args.output/FILES[0]),inputs=b,new_production_calls=0))
    env={};exec(compile(module,str(RENDER)+':unchanged_display_only','exec'),env);display=env['display_row']
    cohort=read(paths['frozen:cohort']);ids=set(cohort['ids']);cases=case['cases'];wanted={(m,c['id']) for m in SHOWN for c in cases};saved={};seen=set();counts=Counter()
    for key,allowed in (('new_scores',{METHODS[4]}),('old_local',{METHODS[3]}),('point',{METHODS[2]}),('fixed',set(METHODS[:2]))):
        for row in rows(paths['frozen:'+key]):
            token=(row['method'],row['id']);require(row['method'] in allowed and row['id'] in ids and token not in seen,'unique same245 original scored display population')
            seen.add(token);counts[row['method']]+=1
            if token in wanted:saved[token]=display(row)
    require(seen=={(m,i) for m in METHODS for i in ids} and len(seen)==1225 and set(saved)==wanted,'complete1225 original saved score authority')
    panels=list(rows(paths['case_rows']));require(len(panels)==24,'all24 preserved display panels')
    for index,row in enumerate(panels):
        ci=index//4;method=SHOWN[index%4];require(row['case_index']==ci and row['id']==cases[ci]['id'] and row['method']==method
            and row['display']==saved[method,row['id']],'unchanged frozen saved display panel '+str(index))
    require({k:binding(p) for k,p in paths.items()}==b,'artifact input bytes changed')
    write(args.output/FILES[2],dict(schema='preserved_complete_figure_artifact_checks_v13',complete=True,passed=True,own_protocol=binding(args.output/FILES[0]),
        inputs=b,checker=b['checker'],source_witness=witness,PNG_files=8,case_panels=24,saved_score_rows=1225,
        original_renderer_receipt_passed=False,original_failure_preserved=True,separate_completed_artifacts_verified=True,
        actual_visual_readability_not_certified_by_hashes=True,new_render_decode_model_GT_score_fit_metric_draw_calls=0))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    p.add_argument('--source-root',type=Path,default=Path('/home/minjae/Documents/github/pallet-pose'))
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
