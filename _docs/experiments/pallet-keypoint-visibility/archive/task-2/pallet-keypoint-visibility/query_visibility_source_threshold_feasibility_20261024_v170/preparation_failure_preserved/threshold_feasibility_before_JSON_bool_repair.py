"""Public saved-score interval feasibility; no threshold choice or model import."""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time
import numpy as np

V=Path(__file__).resolve().parent;P=V.parent
V168=P/'query_visibility_frozen_threshold_full_source_20261024_v168'
RULES=V/'RULES_BEFORE_PUBLIC_SOURCE_SCORE_OR_LABEL_BODIES.json'
INPUTS=V/'INPUT_BINDINGS_BEFORE_PUBLIC_BODIES.json'
PROTOCOL=V/'PROTOCOL_BEFORE_PUBLIC_BODIES.json'
GENERATED=V/'GENERATED_INVENTED_CHECKS.json'
OLD=P/'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py'
NEW=P/'query_visibility_continuous_source_coverage_control_20261024_v137/source_fit_preparation/train_matched_coverage.py'
PUBLIC=P/'query_visibility_public_native_supervision_control_20261024_v148/source_fit_preparation/train_public_native_control.py'
MR=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(x in str(p).lower() for x in ('current223','_v112','/private_evaluation/','_v169'))
    assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def required(N,limit):
    assert isinstance(N,int) and N>0 and np.isfinite(limit) and 0<=limit<=1
    return next(c for c in range(N+1) if c/N>=limit)
def constraint(scores,available,mask,state,limit,N=None,minimum=None):
    assert scores.dtype==np.float32 and available.dtype==bool and mask.dtype==bool
    assert scores.shape==available.shape==mask.shape and np.isfinite(scores[available]).all()
    actual=int(mask.sum());N=actual if N is None else N;assert actual==N
    m=required(N,limit) if minimum is None else minimum;assert 0<=m<=N
    valid=mask&available;x=np.sort(scores[valid].astype(np.float64));count=len(x)
    out=dict(state=state,N=N,limit=limit,required_correct=m,valid_support=count,inputU=N-count,
             bound=None,bound_hex=None,bound_type=None,empty=False)
    if m==0:return out
    if count<m:out.update(empty=True,reason='INSUFFICIENT_ORIGINAL_AVAILABLE_SUPPORT');return out
    assert state in ('D','E');bound=float(x[m-1] if state=='D' else x[-m])
    out.update(bound=bound,bound_hex=bound.hex(),bound_type='OPEN_LOWER_t_GREATER_THAN' if state=='D' else 'CLOSED_UPPER_t_LESS_OR_EQUAL')
    return out
def intersect(rows):
    low=[r for r in rows if r['state']=='D' and r['bound'] is not None]
    high=[r for r in rows if r['state']=='E' and r['bound'] is not None]
    lower=max((r['bound'] for r in low),default=None);upper=min((r['bound'] for r in high),default=None)
    unsupported=[r['condition'] for r in rows if r['empty']]
    feasible=not unsupported and (lower is None or upper is None or lower<upper)
    return dict(feasible=feasible,open_lower=lower,closed_upper=upper,
        lower_witnesses=[r['condition'] for r in low if r['bound']==lower],upper_witnesses=[r['condition'] for r in high if r['bound']==upper],
        insufficient_support_conditions=unsupported,interval='EMPTY' if not feasible else '(open_lower, closed_upper] (null endpoint means unbounded)',
        finite_float64_existence='Raw FP32 endpoints embed exactly in float64; strict lower can be exceeded by float64 nextafter. No working point selected.')
def invented():
    c={};arr=lambda x:np.asarray(x,np.float32);allmask=np.ones(3,bool)
    e=constraint(arr([1,2,2]),allmask,allmask,'E',2/3)
    d=constraint(arr([1,2,2]),allmask,allmask,'D',2/3)
    c['E_closed_upper_at_tie']=e['bound']==2 and sum(arr([1,2,2])>=e['bound'])>=e['required_correct']
    c['D_open_lower_at_tie']=d['bound']==2 and sum(arr([1,2,2])<d['bound'])<d['required_correct'] and sum(arr([1,2,2]).astype(np.float64)<np.nextafter(np.float64(d['bound']),np.inf))>=d['required_correct']
    e['condition']='E';d['condition']='D';c['equal_open_closed_endpoints_empty']=not intersect([e,d])['feasible']
    e2=constraint(arr([3,4,5]),allmask,allmask,'E',2/3);e2['condition']='E2'
    c['strictly_separated_endpoints_nonempty']=intersect([d,e2])['feasible']
    u=constraint(arr([1,2,3]),np.array([1,0,0],bool),allmask,'D',2/3)
    c['inputU_preserves_denominator_and_can_empty']=u['empty'] and u['N']==3 and u['inputU']==2 and u['required_correct']==2
    c['integer_requirement_avoids_ceil_product_error']=required(100,.07)==7 and int(np.ceil(.07*100))==8
    c['minimum_first_integer_exact'] = all(required(n,l)==next(x for x in range(n+1) if x/n>=l) for n in (1,3,33,52,93,828,1772) for l in (0.,.07,.8,.9,1.))
    z=constraint(arr([1,2,3]),np.zeros(3,bool),allmask,'E',0.)
    c['zero_requirement_has_no_bound_despiteU']=not z['empty'] and z['bound'] is None
    t=np.float64(1.);b=constraint(arr([1,1,2]),allmask,allmask,'D',1/3)
    c['nextafter_float64_not_downcast']=np.float32(np.nextafter(t,np.inf))==1 and int((arr([1,1,2]).astype(np.float64)<np.nextafter(t,np.inf)).sum())>=b['required_correct']
    c['unknown_or_other_class_mask_excluded']=constraint(arr([-100,1,2]),allmask,np.array([0,1,1],bool),'E',1.,N=2)['bound']==1
    # Independently enumerate invented candidate decision regions; compare interval feasibility.
    for scores in ([0,0,1,2],[3,2,1,0],[1,1,1,1]):
        x=arr(scores);av=np.array([1,1,1,0],bool);y=np.array([0,1,0,1])
        for ld in (0.,.5,1.):
            for le in (0.,.5,1.):
                rs=[]
                for k,s,l in [(0,'D',ld),(1,'E',le)]:
                    r=constraint(x,av,y==k,s,l);r['condition']=s;rs.append(r)
                candidates=np.r_[np.nextafter(np.float64(x.min()),-np.inf),np.unique(x.astype(np.float64)),np.nextafter(np.unique(x.astype(np.float64)),np.inf)]
                brute=any(all(int(((x.astype(np.float64)<t if k==0 else x.astype(np.float64)>=t)&av&(y==k)).sum())>=required(2,l) for k,l in [(0,ld),(1,le)]) for t in candidates)
                assert intersect(rs)['feasible']==brute
    c['toy_bruteforce_boundary_tie_U_matches_intersection']=True
    assert all(c.values());return dict(completed=True,PASS=True,invented_only=True,checks=c,number_checks=len(c),code=bind(__file__),source_bodies_read=0)
def prepare():
    assert read(RULES)['frozen_before_this_phase_source_score_or_label_bodies']
    # Input pointer JSON is metadata; no referenced score/GT/label body opened here.
    meta=read(V168/'readout_preparation/FIXED_OUTPUT_HEAD_SCALAR_AND_LABEL_BINDINGS.json')
    write(GENERATED,invented())
    write(INPUTS,dict(V168_inputs=bind(V168/'readout_preparation/FIXED_OUTPUT_HEAD_SCALAR_AND_LABEL_BINDINGS.json'),
        prior_authorization=bind(V168/'ROOT_ACTUAL_FULL_SOURCE_READOUT_ENTRY.json'),models=meta['models'],parent_report_protocol=meta['parent_report_protocol'],native35=meta['original_native35'],source_labels=meta['source_labels'],
        pure_sources={'old':bind(OLD),'new':bind(NEW),'public':bind(PUBLIC),'MR':meta['pinned_V155_trainer']},source_bodies_opened=False))
    write(PROTOCOL,dict(completed=True,code=bind(__file__),rules=bind(RULES),inputs=bind(INPUTS),invented=bind(GENERATED),runtime={'executable':sys.executable,'numpy':np.__version__},
        masks='Exact pinned pure AST parser/native_groups/new_groups/public_pools/labels_and_masks; no imported trainer/module/model',condition_count=39,body_order='Rules then code then invented checks then actual public bodies',
        new_threshold_selection=False,new_training_or_forward=False,source_only=True))
    print(json.dumps({'prepared':True,'protocol':bind(PROTOCOL),'tests':bind(GENERATED)}),flush=True)
def pure(path,names,extra):
    ns={'np':np,'rows':lines,'NNATIVE':6144,'NEWN':20480,'CONVENTIONS':('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY'),
        'MATERIALS':('PROCEDURAL_WOOD','DARK_POLYMER'),'ASPECTS':('RECTANGULAR','SQUARE'),'GEOMETRIES':('BOARD_BLOCK_RUNNER','RIB_LEG_RUNNER'),
        'FOREIGN':('SAME_TARGET_MATERIAL','CARDBOARD_TAN','BACKGROUND_LIKE','HIGH_CONTRAST'),**extra}
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_METADATA_ONLY','exec'),ns);return ns
def run():
    started=time.monotonic();protocol=read(PROTOCOL)
    assert protocol['code']==bind(__file__) and protocol['rules']==bind(RULES)
    inputs=read(checked(protocol['inputs']));tests=read(checked(protocol['invented']));assert tests['PASS'] and tests['code']==bind(__file__)
    assert sys.executable==protocol['runtime']['executable'] and np.__version__==protocol['runtime']['numpy']
    auth=read(checked(inputs['prior_authorization']));assert auth['source_only'] and auth['approved_actual_saved_full_source_readout']
    bindings=[bind(RULES),bind(PROTOCOL),bind(INPUTS),bind(GENERATED),inputs['prior_authorization'],inputs['parent_report_protocol'],inputs['native35'],inputs['source_labels'],*inputs['pure_sources'].values()]
    parent=read(checked(inputs['parent_report_protocol']));limits=read(checked(inputs['native35']))['native35_limits'];assert len(limits)==35
    ns={k:pure(checked(b),{'old':{'native_groups','read_native_GT'},'new':{'new_groups','parse_new_GT'},'public':{'public_pools'},'MR':{'labels_and_masks'}}[k],{}) for k,b in inputs['pure_sources'].items()}
    for n in parent['native']:bindings.append(n['GT'])
    old=ns['old']['read_native_GT'](checked(parent['native'][0]['GT']))['CF_PRIMARY_DECLARED']
    new=ns['new']['parse_new_GT'](lines(checked(parent['native'][1]['GT'])))['CF_PRIMARY_DECLARED']
    groups=[ns['old']['native_groups'](old),ns['new']['new_groups'](new)];metas=[old,new]
    bindings.append(parent['public_labels']);pub=ns['public']['public_pools'](lines(checked(parent['public_labels'])))
    mr=ns['MR']['labels_and_masks'](lines(checked(inputs['source_labels'])))
    outputs=[]
    for model in inputs['models']:
        packet={}
        for family in ('OLD6144','NEW20480','PALLOX512','MR1840'):
            b=model['outputs'][family];bindings.append(b)
            with np.load(checked(b),allow_pickle=False) as z:packet[family]={'score':z['raw_three_logits'][:,0].copy(),'available':z['query_available'].copy()}
        bindings.append(model['source_results']);baseline=read(checked(model['source_results']));rows=[]
        def add(name,family,mask,state,limit,N=None,minimum=None):
            r=constraint(packet[family]['score'],packet[family]['available'],mask,state,limit,N,minimum);r.update(condition=name,family=family)
            # Recover the original threshold0 reported correctness as a mask/denominator parity guard.
            x=packet[family]['score'];av=packet[family]['available'];r['original0_correct']=int((mask&av&((x<0) if state=='D' else (x>=0))).sum())
            rows.append(r);return r
        for name,base in limits.items():
            isnew=name.startswith('NEW_NATIVE::');prefix='NEW_NATIVE::' if isnew else 'NATIVE::';group,state=name[len(prefix):].rsplit('::',1);idx=int(isnew)
            family=('OLD6144','NEW20480')[idx];mask=groups[idx][group]&(metas[idx]['target']==int(state=='E'))
            original=baseline['native']['gates'][name];assert original['limit']==base['limit']
            r=add(name,family,mask,state,base['limit'],N=original['N'])
            assert r['original0_correct']/r['N']==original['value']
        for state,m,N in [('D',30,33),('E',5,6)]:
            mask=(pub['split']=='VAL_HALL')&pub['available']&(pub['target']==int(state=='E'))
            r=add('PALLOX::VAL_HALL::INPUT_VALID::'+state,'PALLOX512',mask,state,m/N,N=N,minimum=m)
            assert r['original0_correct']==baseline['Pallox']['sanity'][state+'_correct']
        for state,limit,N in [('D',.90,93),('E',.80,52)]:
            mask=(~mr['training'])&mr['available']&(mr['role']==int(state=='E'))
            r=add('MR::PUBLIC_CAL::INPUT_VALID::'+state,'MR1840',mask,state,limit,N=N)
            met=baseline['MR']['SOURCE_VALIDATION']['visibility_inputvalid'];assert r['original0_correct']==met['CM_rows0_1_columns0_1_U'][int(state=='E')][int(state=='E')]
        assert len(rows)==39
        entity=baseline['MR']['SOURCE_VALIDATION']['entity_inputvalid']
        outputs.append(dict(arm=model['arm'],conditions=rows,all39_intersection=intersect(rows),
            native35_only_intersection=intersect(rows[:35]),Pallox_only_intersection=intersect(rows[35:37]),MR_visibility_only_intersection=intersect(rows[37:]),
            entity_quality_fixed0_unchanged=dict(MR_entity_inputvalid=entity,MR_entity_existing_D_and_E_recall_limits=[.80,.80],
                MR_entity_gate_pass=entity['negative_recall']>=.8 and entity['positive_recall']>=.8,synthetic_quality_report_binding=model['source_results'],visibility_threshold_does_not_modify_entity_or_quality=True),
            fixed_existing_V167_scalar_unchanged=model['threshold_float64'],head_binding_inherited_not_opened=model['head_checkpoint_binding_inherited'],selected_threshold=None,selected_model=None))
    for b in bindings:checked(b)
    report=dict(completed=True,PASS=True,phase='V170',purpose='Fixed-score global threshold feasibility, no new working point selection',models=outputs,
        bindings=bindings,code=bind(__file__),protocol=bind(PROTOCOL),rules=bind(RULES),invented=bind(GENERATED),
        condition_count_each=39,original_native35_limits_and_N_unchanged=True,source_GT_public_weak_labels_and_fixed_raw_only=True,
        bounds_are_order_statistics_not_selected_thresholds=True,critical_bound_authority='Public source constraints only; weak public labels are not physical native point gold or fresh independent validation.',
        new_threshold_or_model_selection=False,new_fit_forward_or_encoder=0,current_V169_private_V112_RGB_features_weights_reads=0,selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-started)
    write(V/'FEASIBILITY.json',report)
    print(json.dumps({'completed':True,'report':bind(V/'FEASIBILITY.json'),'models':[{k:r[k] for k in ('arm','all39_intersection','native35_only_intersection')} for r in outputs]},ensure_ascii=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare and not a.run:prepare()
    elif a.run and not a.prepare:run()
    else:raise SystemExit('Use --prepare before --run')
