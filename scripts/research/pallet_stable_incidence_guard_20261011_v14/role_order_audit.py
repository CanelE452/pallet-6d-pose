"""Read-only camera-facing role audit; no pose selection or fitting changes."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
PARENT=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
SELECTOR=Path('/home/minjae/Documents/github/pallet-pose/challenge/evaluation_v2/pnp_selector.py')
PAIRS=dict(LR_PAIRS=((0,1),(3,2),(4,5),(7,6)),
    TB_PAIRS=((0,3),(1,2),(4,7),(5,6)),FR_PAIRS=((0,4),(1,5),(2,6),(3,7)))
FILES=('ROLE_ORDER_AUDIT_PROTOCOL.json','ROLE_ORDER_AUDIT_STARTED.json','ROLE_ORDER_AUDIT_CHECKS.json','ROLE_ORDER_AUDIT_ROWS.jsonl.gz')

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def binding(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def inputs():
    paths=dict(code=Path(__file__).resolve(),baseline_selector=SELECTOR)
    for name in ('GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json'):
        paths['parent:'+name]=PARENT/name
    for name in ('GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json'):
        paths['guard:'+name]=DOC/name
    return {k:binding(p) for k,p in paths.items()}
def authority():
    values={}
    for node in ast.parse(SELECTOR.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in PAIRS:
            values[node.targets[0].id]=ast.literal_eval(node.value)
    require(values==PAIRS,'exact existing camera-facing role pairs required')
    for folder,name,checks in ((PARENT,'REVIEW_VALIDATION_CHECKS.json',713803),(DOC,'VALIDATION_CHECKS.json',42715)):
        v=read(folder/name);require(v['passed'] is True and v['complete'] is True and v['failure_count']==0 and v['checks']==checks,'independent geometry authority')
        seal=read(folder/'GEOMETRY_SEAL.json');require(seal['complete'] is True and seal['rows']==245,'complete geometry population')
        require(all(seal['geometry'][k]==binding(folder/'GEOMETRY_SEALED.jsonl.gz')[k] for k in ('bytes','sha256')),'sealed geometry bytes')
    return dict(existing_baseline_pairs=PAIRS,comparisons='strict signed role relations of modeled camera-facing points only',
        diagnostic_only=True,role_constraints_not_yet_applied=True,upright_prior_not_used=True,
        hidden_initial_image_coordinates_not_used=True,all_eight_model_predictions_are_not_independent_observations=True,
        new_GT_model_fit_PnP_J_SVD_metric_draw_calls=0,known_parent_DEV_results=True,new_guard_scores_not_used=True)
def inspect(c):
    uv=c['projected'];depth=c['model_camera_depths_m']
    require(len(uv)==len(depth)==8,'complete stored modeled corners required')
    failed={}
    for name,pairs in PAIRS.items():
        axis=0 if name=='LR_PAIRS' else 1
        failed[name]=[[a,b] for a,b in pairs if not((depth[a]<depth[b]) if name=='FR_PAIRS' else (uv[a][axis]<uv[b][axis]))]
    return dict(violations=failed,violation_count=sum(map(len,failed.values())),passes_all_role_orders=not any(failed.values()),
        hidden_initial_image_coordinates_used=False,pose_selection_changed=False)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));args=p.parse_args()
    for name in FILES if args.stage=='freeze' else FILES[1:]:require(not(DOC/name).exists(),'preserve '+name)
    b=inputs();policy=authority()
    if args.stage=='freeze':write(DOC/FILES[0],dict(schema='read_only_camera_facing_role_protocol_v14',inputs=b,policy=policy,frozen_before_own_comparisons=True));return
    own=read(DOC/FILES[0]);require(own['inputs']==b and own['policy']==policy,'own fixed audit differs')
    write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),inputs=b,new_production_calls=0))
    count=Counter();emitted=[];guard=list(rows(DOC/'GEOMETRY_SEALED.jsonl.gz'))
    for i,row in enumerate(rows(PARENT/'GEOMETRY_SEALED.jsonl.gz')):
        require(i<245 and row['id']==guard[i]['id'],'same ordered245 joins')
        c=row['solver'].get('best_candidate');selected=inspect(c) if c else None
        alternatives=[dict(candidate_index=v['candidate_index'],phase=v['phase'],accepted=v['accepted'],**inspect(v)) for v in row['solver']['all_candidate_solutions']]
        emitted.append(dict(id=row['id'],session=row['session'],parent_NEW=row['new_pose_estimated'],guard_NEW=guard[i]['new_pose_estimated'],
            selected=selected,candidates=alternatives))
        count['frames']+=1;count['parent_NEW']+=row['new_pose_estimated'];count['guard_NEW']+=guard[i]['new_pose_estimated']
        if selected:
            count['selected_candidates']+=1
            if row['new_pose_estimated']:count['parent_NEW_role_valid' if selected['passes_all_role_orders'] else 'parent_NEW_role_invalid']+=1
            if guard[i]['new_pose_estimated']:count['guard_NEW_role_valid' if selected['passes_all_role_orders'] else 'guard_NEW_role_invalid']+=1
        for v in alternatives:count['candidate_rows']+=1;count['candidate_role_valid' if v['passes_all_role_orders'] else 'candidate_role_invalid']+=1
    require(len(emitted)==245 and inputs()==b,'complete unchanged245 audit')
    with (DOC/FILES[3]).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
        for v in emitted:stream.write((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode())
    write(DOC/FILES[2],dict(schema='read_only_camera_facing_role_checks_v14',complete=True,passed=True,inputs=b,
        protocol=binding(DOC/FILES[0]),policy=policy,counts=dict(count),rows=binding(DOC/FILES[3]),changes_to_pose_or_operating_policy=0))
    print('CAMERA_FACING_ROLE_AUDIT',dict(count),flush=True)
if __name__=='__main__':main()
