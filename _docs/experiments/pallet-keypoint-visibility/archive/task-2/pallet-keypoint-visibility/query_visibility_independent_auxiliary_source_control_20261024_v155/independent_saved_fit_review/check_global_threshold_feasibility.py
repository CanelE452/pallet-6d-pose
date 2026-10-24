"""Retrospective feasibility proof only; never export or apply a threshold.

D is strict z<t and E is inclusive z>=t. Unavailable original inputs stay U.
Only order-statistic inequalities are evaluated; model/gates/raw0 stay frozen.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from audit_independent_saved import (ARMS,RUN,PROTOCOL,read,lines,check,packet,
                                     native_meta,group,target,bind)
H=Path(__file__).resolve().parent
def required_count(N,limit):
    if N<=0 or limit is None:return None
    return next((k for k in range(N+1) if k/N>=limit),N+1)
def make_constraint(name,state,limit,z,available,truth,mask):
    positive=1 if state=='E' else 0;known=mask&(truth==positive)
    N=int(known.sum());k=required_count(N,limit);valid=known&available&np.isfinite(z)
    v=z[valid];n=len(v);capacity=k is not None and k<=n
    bound=None
    if capacity and k:
        ordered=np.sort(v)
        bound=float(ordered[k-1] if state=='D' else ordered[-k])
    # Numeric bounds are internal only and never serialized or printed.
    raw_correct=int((valid&((z<0) if state=='D' else(z>=0))).sum())
    visible=dict(name=name,state=state,limit=limit,N=N,original_input_U=int((known&~available).sum()),
        finite_available=n,minimum_correct=k,raw0_correct=raw_correct,
        raw0_PASS=bool(N and raw_correct/N>=limit),capacity_possible=capacity)
    return dict(visible=visible,bound=bound)
def feasible(constraints):
    bad=[c['visible']['name'] for c in constraints if not c['visible']['capacity_possible']]
    ds=[c for c in constraints if c['visible']['state']=='D' and c['bound']is not None]
    es=[c for c in constraints if c['visible']['state']=='E' and c['bound']is not None]
    conflicts=[dict(D_requirement=d['visible']['name'],E_requirement=e['visible']['name'],
                    relation='strict_D_lower_bound_is_not_below_inclusive_E_upper_bound')
               for d in ds for e in es if d['bound']>=e['bound']]
    return dict(any_one_global_threshold_feasible=not bad and not conflicts,
        impossible_due_to_original_U_or_missing_class=bad,
        conflicting_limit_pairs=conflicts,
        constraints=[c['visible'] for c in constraints],
        threshold_value_exported=False,threshold_selected_or_applied=False)
def E_capacity_at_D_requirement(dconstraint,econstraint,z,av,y):
    # Supremum capacity under a strict-D bound: any E tied at/below that D
    # order statistic is excluded as t must be strictly larger than the bound.
    if not dconstraint['visible']['capacity_possible']:return None
    b=dconstraint['bound'];em=(y==1)&av&np.isfinite(z)
    maximum=int((em if b is None else(em&(z>b))).sum())
    return dict(D_minimum_correct=dconstraint['visible']['minimum_correct'],
        D_N=dconstraint['visible']['N'],maximum_E_correct_with_D_requirement=maximum,
        E_required_correct=econstraint['visible']['minimum_correct'],E_N=econstraint['visible']['N'],
        cannot_meet_both=maximum<econstraint['visible']['minimum_correct'])
def main():
    review=read(H/'REVIEW.json');assert review['completed'] and review['PASS']
    control=read(check(review['control_complete']));p=read(check(control['protocol']))
    parent=read(check(p['parent_protocol']));limits=read(check(p['original_native35']))['native35_limits']
    assert len(limits)==35
    native=[native_meta(lines(check(b['GT'])),n)for b,n in zip(parent['native'],(6144,20480))]
    mrrows=lines(check(p['source_labels']));pubrows=lines(check(parent['public_labels']))
    mrtruth=np.array([r['role_target'] for r in mrrows]);mrval=np.array([r['split']=='SOURCE_VALIDATION'for r in mrrows])
    pubtruth=target(np.array([r['semantic_state']for r in pubrows]));pubval=np.array([r['split']=='VAL_HALL'for r in pubrows])
    results={};score_bindings=[]
    for arm in ARMS:
        packs=[]
        for name in('OLD6144','NEW20480','PALLOX512','MR1840'):
            path=RUN/arm/f'FINAL_{name}_OUTPUTS.npz';packs.append(packet(path));score_bindings.append(bind(path))
        native_constraints=[]
        for name,b in limits.items():
            new=name.startswith('NEW_NATIVE::');tail=name[len('NEW_NATIVE::'if new else'NATIVE::'):]
            g,state=tail.rsplit('::',1);i=int(new);a=packs[i];z=a['raw_three_logits'][:,0]
            native_constraints.append(make_constraint(name,state,b['limit'],z,a['query_available'],native[i]['target'],group(native[i],g)))
        a=packs[2];z=a['raw_three_logits'][:,0]
        pub_constraints=[make_constraint('PALLOX_HALL_INPUT_VALID::D','D',30/33,z,a['query_available'],pubtruth,pubval&a['query_available']),
                         make_constraint('PALLOX_HALL_INPUT_VALID::E','E',5/6,z,a['query_available'],pubtruth,pubval&a['query_available'])]
        a=packs[3];z=a['raw_three_logits'][:,0];av=a['query_available'];m=mrval&av
        mr_constraints=[make_constraint('MR_HELD_INPUT_VALID::D','D',.90,z,av,mrtruth,m),
                        make_constraint('MR_HELD_INPUT_VALID::E','E',.80,z,av,mrtruth,m)]
        native_answer=feasible(native_constraints);pub_answer=feasible(pub_constraints);mr_answer=feasible(mr_constraints)
        combined=feasible(native_constraints+pub_constraints+mr_constraints)
        results[arm]=dict(native35=native_answer,Pallox_HALL=pub_answer,MR_held_inputvalid=mr_answer,
            ALL_fixed_visibility_requirements=combined,
            MR_order_statistic_E_capacity_given_D=E_capacity_at_D_requirement(*mr_constraints,z,av,np.where(m,mrtruth,-1)))
        assert not mr_answer['any_one_global_threshold_feasible']
        assert not combined['any_one_global_threshold_feasible']
    # Explicit tied-score fixture: a D and E at the same value cannot both pass.
    toy_z=np.array([1.,1.,5.]);toy_av=np.array([True,True,False]);toy_y=np.array([0,1,0]);mask=np.ones(3,bool)
    td=make_constraint('TOY_D','D',.5,toy_z,toy_av,toy_y,mask)
    te=make_constraint('TOY_E','E',1.,toy_z,toy_av,toy_y,mask)
    assert not feasible([td,te])['any_one_global_threshold_feasible']
    impossible=make_constraint('TOY_U_CAPACITY','D',1.,toy_z,toy_av,toy_y,mask)
    assert not impossible['visible']['capacity_possible']
    result=dict(completed=True,PASS=True,diagnostic_only=True,source_only=True,
        code=bind(__file__),independent_saved_review=bind(H/'REVIEW.json'),control_complete=review['control_complete'],
        immutable_native35_limits=p['original_native35'],source_scores=score_bindings,
        method=dict(D_rule='strict z<threshold',E_rule='inclusive z>=threshold',
            D_requirement='threshold strictly above the kth ascending available-D logit',
            E_requirement='threshold at most the kth descending available-E logit',
            feasibility='all original capacities sufficient and max strict D bound < min inclusive E bound',
            ties='equal opposing bounds are infeasible',U='original unavailable inputs remain U in original known-class denominators',
            required_counts='smallest integer k with original Python k/N>=unchanged limit; no rounding of gates'),
        arms=results,generated_tie_and_input_U_capacity_checks=True,
        numeric_threshold_bounds_exported=False,threshold_selection_calibration_or_application=False,
        frozen_raw0_and_original_gates_changed=False,model_reinterpreted_as_pass=False,
        source_weak_labels_not_human_gold=True,selected_model=None,goal_status='active_unsolved',
        model_feature_values_weight_RGB_private_or_fit_calls=0)
    output=H/'GLOBAL_THRESHOLD_FEASIBILITY_DIAGNOSTIC.json'
    with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    compact={a:dict(native35_any_threshold=x['native35']['any_one_global_threshold_feasible'],
        public_HALL_any_threshold=x['Pallox_HALL']['any_one_global_threshold_feasible'],
        MR_any_threshold=x['MR_held_inputvalid']['any_one_global_threshold_feasible'],
        ALL_any_threshold=x['ALL_fixed_visibility_requirements']['any_one_global_threshold_feasible'],
        MR_capacity=x['MR_order_statistic_E_capacity_given_D'])for a,x in results.items()}
    print(json.dumps(dict(output=bind(output),findings=compact)))
if __name__=='__main__':main()
