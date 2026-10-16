"""Review only floating branch selection at a saved finite-hinge cusp.

The original completed 74-failure audit is preserved. Its exact stdlib AST is
reused, with independently differentiated stored one-sided branches only when
a different scalar branch is within an ulp-derived endpoint rounding bound.
No saved observation, solver, policy, pose, fit, SVD or performance gate changes.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
SOURCE=CODE/'validation_checks.py'
SOURCE_SHA='91d014543eba01c142cad9a0c69affd2eb16666913ce13a06a56857f854c98ef'
CONTRACT=DOC/'CUSP_REVIEW_CONTRACT_KO.md'
FILES=('REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json')
STATE=dict(context={},events=[])
REVIEW=dict(original_checker_changed=False,numerical_core_or_saved_rows_changed=False,
    performance_thresholds_changed=False,original_complete_failed_audit_preserved=True,
    original_checks=713802,original_failure_count=74,AST_patch_sites=7,
    expected_roundoff_branch_reconciliations=4,one_sided_cusp_rank_is_not_two_sided_observability_proof=True)

def require(v,s):
    if not v:raise ValueError(s)

def binding(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def authority():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'original checker bytes changed')
    failed=json.loads((DOC/'VALIDATION_CHECKS.json').read_text());own=json.loads((DOC/'VALIDATION_PROTOCOL.json').read_text())
    require(failed['complete'] is True and failed['passed'] is False and failed['checks']==713802 and failed['failure_count']==74 and failed['exception'] is None,'preserve complete original74FAIL')
    require(failed['inputs']==own['inputs'] and failed['own_protocol']['sha256']==binding(DOC/'VALIDATION_PROTOCOL.json')['sha256'] and
        failed['checker']['sha256']==SOURCE_SHA and len(failed['failures'])==74,'original full audit authority')
    labels=[f['check'] for f in failed['failures']]
    for phase in ('proposal','refit'):
        prefix='frame.111.'+phase+'.0.'
        expected=[prefix+matrix+'.8.'+str(j) for matrix in ('independent_observed_J','independent_modeled_J') for j in range(6)]
        expected+=[prefix+'finite_geometry_blocks.0.hinge_branches.6']
        expected+=[prefix+matrix+'.independent_column_norms.'+str(j) for matrix in ('observed_normal_joint','modeled_normal_joint','observed_normal_lines_only','modeled_normal_lines_only') for j in range(6)]
        require(set(f for f in labels if f.startswith(prefix))==set(expected) and len([f for f in labels if f.startswith(prefix)])==37,'original failures must be exactly one cusp-derived J/branch/norm pattern')
    require(len(set(labels))==74,'original failure labels unique')
    return own

def reviewed_tree():
    tree=ast.parse(SOURCE.read_text());original=copy.deepcopy(tree)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    fn=functions['finite_blocks'];require([a.arg for a in fn.args.args][-2:]==['modeled','scoring'] and len(fn.args.defaults)==2,'original finite_blocks signature')
    fn.args.args.append(ast.arg(arg='branch_witness'));fn.args.defaults.append(ast.Constant(None))
    branches=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==ast.dump(ast.parse('u<0',mode='eval').body,include_attributes=False)]
    require(len(branches)==1 and isinstance(branches[0].orelse[0],ast.If),'unique original hinge branch')
    branch=branches[0];after=branch.orelse[0]
    require(ast.dump(after.test,include_attributes=False)==ast.dump(ast.parse('u>L',mode='eval').body,include_attributes=False),'unique original after-end branch')
    before_test,after_test=copy.deepcopy(branch.test),copy.deepcopy(after.test)
    branch.test=ast.parse("_review_branch_choice == 'BEFORE_FIRST_ENDPOINT'",mode='eval').body
    after.test=ast.parse("_review_branch_choice == 'AFTER_SECOND_ENDPOINT'",mode='eval').body
    assignment=ast.parse('_review_branch_choice = _review_branch(u,L,xy,A,B,branch_witness,len(along),line,modeled)').body[0]
    parents=[n for n in ast.walk(fn) if hasattr(n,'body') and isinstance(n.body,list) and any(x is branch for x in n.body)]
    require(len(parents)==1,'single branch parent block');container=parents[0].body;position=container.index(branch);container.insert(position,assignment)
    candidate=functions['candidate_check'];calls=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='finite_blocks']
    require(len(calls)==2 and all(not any(k.arg=='branch_witness' for k in call.keywords) for call in calls),'exact observed/modeled geometry calls')
    for call in calls:call.keywords.append(ast.keyword(arg='branch_witness',value=ast.parse("c['point_line_geometry']['edge_blocks']",mode='eval').body))
    context=ast.parse('_review_context(row,c,phase,label)').body[0];candidate.body.insert(0,context)
    info=dict(original=binding(SOURCE),AST_patch_sites=7,patched_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        patch_sites=['optional finite_blocks witness parameter','one branch-selector assignment','before-end predicate','after-end predicate','observed geometry witness keyword','modeled geometry witness keyword','candidate context witness'])
    # Undo exactly the seven declared sites and prove all remaining audit AST unchanged.
    fn.args.args.pop();fn.args.defaults.pop();branch.test=before_test;after.test=after_test;container.remove(assignment)
    for call in calls:call.keywords.pop()
    candidate.body.remove(context)
    require(ast.dump(tree,include_attributes=False)==ast.dump(original,include_attributes=False),'undeclared old-audit AST edits')
    # Reapply the same reviewed sites; no production AST is imported.
    fn.args.args.append(ast.arg(arg='branch_witness'));fn.args.defaults.append(ast.Constant(None));branch.test=ast.parse("_review_branch_choice == 'BEFORE_FIRST_ENDPOINT'",mode='eval').body
    after.test=ast.parse("_review_branch_choice == 'AFTER_SECOND_ENDPOINT'",mode='eval').body;container.insert(position,assignment)
    for call in calls:call.keywords.append(ast.keyword(arg='branch_witness',value=ast.parse("c['point_line_geometry']['edge_blocks']",mode='eval').body))
    candidate.body.insert(0,context);return ast.fix_missing_locations(tree),info

def review_context(row,c,phase,label):
    STATE['context']=dict(id=row['id'],phase=phase,label=label,proposal_index=c.get('proposal_index'),candidate_index=c.get('candidate_index'))

def strict_branch(u,L):return 'BEFORE_FIRST_ENDPOINT' if u<0 else 'AFTER_SECOND_ENDPOINT' if u>L else 'INSIDE_CLOSED_SEGMENT'

def review_branch(u,L,xy,A,B,witness,index,line,modeled):
    calculated=strict_branch(u,L)
    if witness is None:return calculated
    block=next((b for b in witness if b['edge']==line['edge']),None);require(block is not None,'same candidate edge witness')
    saved_u=block['longitudinal_support_px'][index];saved_L=block['projected_segment_length_px'];saved_branch=block['hinge_branches'][index]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (u,L,saved_u,saved_L)) and L>1e-12 and saved_L>1e-12,'finite cusp witness')
    require(abs(saved_u-u)<=1e-7+1e-12*abs(saved_u) and abs(saved_L-L)<=1e-7+1e-12*abs(saved_L),'existing independent u/L near check')
    require(saved_branch==strict_branch(saved_u,saved_L),'saved branch must match strict stored numeric predicate')
    if saved_branch==calculated:return calculated
    scale=max(1.,L,saved_L,*(abs(v) for p in (xy,A,B) for v in p))
    # Propagate coordinate ulps through projection onto a normalized direction.
    bound=64.*math.ulp(scale)*(1.+math.dist(xy,A)/L)
    before={calculated,saved_branch}=={'BEFORE_FIRST_ENDPOINT','INSIDE_CLOSED_SEGMENT'}
    after={calculated,saved_branch}=={'AFTER_SECOND_ENDPOINT','INSIDE_CLOSED_SEGMENT'}
    require((before and abs(u)<=bound and abs(saved_u)<=bound) or
        (after and abs(u-L)<=bound and abs(saved_u-saved_L)<=bound),'branch disagreement exceeds endpoint floating-roundoff bound')
    require(abs(L-saved_L)<=bound,'segment length disagreement exceeds floating-roundoff bound')
    STATE['events'].append(dict(**STATE['context'],edge=line['edge'],query_id=line['support_query_ids'][index],support_index=index,
        modeled_normals=modeled,independent_u_px=u,stored_u_px=saved_u,independent_L_px=L,stored_L_px=saved_L,
        independent_branch=calculated,stored_one_sided_branch=saved_branch,roundoff_bound_px=bound,
        coordinate_ulp_px=math.ulp(scale),coordinate_scale_px=scale,actual_support_xy=xy,independent_projected_A=A,independent_projected_B=B,
        analytic_derivative_recomputed=True,performance_threshold_changed=False))
    return saved_branch

def main():
    original=authority();tree,info=reviewed_tree()
    env=dict(__file__=str(SOURCE),__name__='frozen_v13_scalar_cusp_review',_review_branch=review_branch,_review_context=review_context)
    exec(compile(tree,str(SOURCE)+' [seven declared scalar cusp witness sites]','exec'),env)
    old_paths,old_write=env['input_paths'],env['write'];env['OUTPUTS']=FILES
    def reviewed_paths(args):
        STATE['output']=args.output
        paths=old_paths(args)
        paths.update(review_checker=Path(__file__).resolve(),original_failure=DOC/'VALIDATION_CHECKS.json',original_validation_protocol=DOC/'VALIDATION_PROTOCOL.json',review_contract=CONTRACT)
        current={k:env['binding'](p) for k,p in paths.items()}
        require({k:current[k] for k in original['inputs']}==original['inputs'],'all original audit input bytes preserved')
        return paths
    def reviewed_write(path,value):
        if path.name==FILES[2]:
            value['review']=dict(REVIEW,actual_roundoff_branch_reconciliations=len(STATE['events']),AST_witness=info)
            value['cusp_reconciliations']=STATE['events']
            if value['passed']:
                exact=len(STATE['events'])==4 and all(e['id']=='plastic_night_01:035421' and e['edge']==2 and e['query_id']==20 and e['phase'] in ('proposal','refit') for e in STATE['events'])
                value['checks']+=1
                if not exact:
                    value['passed']=False;value['failure_count']+=1
                    value['failures'].append(dict(check='review.only_original_four_cusp_branches',actual=STATE['events'],expected=4))
        else:value['review']=dict(REVIEW,AST_witness=info)
        old_write(path,value)
    env['input_paths']=reviewed_paths;env['write']=reviewed_write;env['main']()
    result=STATE.get('output',DOC)/FILES[2]
    if result.is_file() and not json.loads(result.read_text())['passed']:raise SystemExit(1)

if __name__=='__main__':main()
