"""Independent selected-pose reconstruction, without images or references.

Read inputs/predictions as streams. Rebuild the frozen native observation
problem directly and evaluate each model regularizer through its pure API;
do not use the new wrapper to certify its own energy decomposition.
"""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import math
import numpy as np

from . import evaluate_bounded_geometric_priors_v4_22 as V
from . import refine_grazing_image_chain_mixtures_v4_18c as NATIVE
from . import calibrate_bounded_o_localization_v4_21 as OFIT
from . import bounded_o_localization_prior_v4_21 as O
from . import bounded_coarse_visibility_prior_v4_20 as S
from . import calibrate_coarse_visibility_prior_v4_20 as SFIT


def main(output=None):
    C,E=V.C,V.E
    output=Path(V.OUTPUT if output is None else output)
    dest=output/'independent_root_audit_v1'
    if dest.exists():raise FileExistsError('Preserve previous root audits')
    protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    if (seal['rows']!=238 or seal['predictions_sha256']!=C.sha(output/'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')):
        raise ValueError('Complete sealed predictions required before audit')
    inputseal=C.read(output/'INPUT_SEAL.json');parentseal=C.read(V.V.OUTPUT/'PREDICTION_SEAL.json')
    if (inputseal['input_sha256']!=C.sha(output/'INPUTS.jsonl.gz')
            or inputseal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')
            or parentseal['rows']!=238
            or parentseal['predictions_sha256']!=C.sha(V.V.OUTPUT/'PREDICTIONS.jsonl.gz')):
        raise ValueError('Exact current inputs and sealed parent predictions required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    nmodel=NATIVE._verified_model()
    omodel=C.read(OFIT.OUTPUT/'MODEL.json');oparams=OFIT.prior_parameters(omodel)
    smodel=C.read(SFIT.OUTPUT/'MODEL.json');S.validate_model(smodel)
    profiles=V.PROFILE_ARMS
    checks=0;failures=[];maxdiff=0.;rows=0;counts=Counter();selected=0;fallback=0
    def check(ok,what,ident,arm=None):
        nonlocal checks
        checks+=1
        if not ok:failures.append(dict(id=ident,arm=arm,check=what))
    def near(a,b):
        nonlocal maxdiff
        delta=abs(float(a)-float(b));maxdiff=max(maxdiff,delta)
        return delta<=1e-8*max(1.,abs(float(a)),abs(float(b)))
    tasks=C.rows(output/'INPUTS.jsonl.gz');predictions=C.rows(output/'PREDICTIONS.jsonl.gz')
    previous_rows=C.rows(V.V.OUTPUT/'PREDICTIONS.jsonl.gz')
    for task,row,previous in zip_longest(tasks,predictions,previous_rows):
        if task is None or row is None or previous is None:raise ValueError('Input/prediction lengths differ')
        rows+=1;ident=task['id'];counts[task['partition']]+=1
        check(ident==row['id']==previous['id']==protocol['ids'][rows-1],'ordered complete population',ident)
        check(row['arms'][V.ARMS[0]]==task['baseline'],'exact frozen e baseline',ident)
        check(set(row['arms'])==set(V.ARMS),'all five declared arms',ident)
        check(task['states']==row['states'],'authoritative states unchanged',ident)
        original=task['families']['ORIGINAL_WD'];coarse=original['coarse_pose']
        sprepared=S.prepare_bounded_coarse_visibility_prior(original['native_xyz'],coarse['R'],coarse['t'],
            np.asarray(task['initial_self_hidden'],dtype=bool),smodel)
        archives=[r['original_domain']for r in task['O_localization_abstention']['records']]
        for record in archives:
            check(record['binding']['calibration_model_sha256']==omodel['source_O_model_digest'],
                'same original O ellipse calibration model',ident)
        regions=O.prepare_o_localization_regions(archives,task['states'],task['initial_self_hidden'])
        bases={}
        for profile,arms in profiles.items():
            include_s=profile=='OS'
            if profile not in ('O','OS'):raise ValueError('Explicit fixed O/OS profile declaration required')
            oldsearches=previous['native_refinement_audit'][V.V.ARMS[1]]['family_refinements']
            newsearches=row['bounded_prior_refinement_audit'][arms[0]]['family_refinements']
            check(set(oldsearches)==set(newsearches)==set(task['families']),'same WD search families',ident,arms[0])
            for family_name in newsearches:
                check(newsearches[family_name]['start_selection']['selected_start_names']==
                    oldsearches[family_name]['start_selection']['selected_start_names'],
                    'exact original v19 unregularized native start names',ident,arms[0])
            for arm in arms:
                result=row['arms'][arm];pose=result['actual_pose'];solution=result['solution']
                family=task['families'][result['WD_family']]
                key=result['WD_family']
                if key not in bases:bases[key]=NATIVE._problem(task,family,nmodel)[0]
                base=bases[key];R=np.asarray(pose['R_cf']);t=np.asarray(pose['centroid'])
                fresh=base.evaluate(R,t)
                check(fresh['accepted']==result['accepted'],'fresh native hard-guard acceptance',ident,arm)
                check(np.allclose(R@np.asarray(family['physical_registration_Q']),pose['R_physical'],atol=1e-12,rtol=0),
                    'physical registration unchanged',ident,arm)
                if not result['accepted']:
                    fallback+=1
                    check(pose['fallback']and np.array_equal(R,np.asarray(coarse['R']))
                        and np.array_equal(t,np.asarray(coarse['t'])),'own original coarse fallback only',ident,arm)
                    continue
                selected+=1
                p=NATIVE.BASE.NATIVE.V._pose_vector(R,t,base.scale)
                ov,_,oa=O.bounded_o_localization_value_and_gradient(p,base.xyz,base.K,base.scale,regions,**oparams)
                sv=0.;sa=None
                if include_s:
                    rebound=S.rebind_bounded_visibility_trial_geometry(sprepared,base.xyz)
                    sv,_,sa=S.coarse_visibility_prior_value_and_gradient(p,base.xyz,base.scale,rebound)
                native=float(fresh['objective']);prior=math.fsum((ov,sv));total=math.fsum((native,prior))
                check(near(native,solution['native_observation_objective']),'independent native energy',ident,arm)
                check(near(prior,solution['regularizer_objective']),'independent regularizer energy',ident,arm)
                check(near(total,solution['objective']),'independent total objective',ident,arm)
                actual={r['kind']:r['weighted_cost']for r in solution['regularizer_costs']}
                check(set(actual)=={'O_COARSE_LOCALIZATION_PRIOR','S_COARSE_VISIBILITY_PRIOR'},
                    'separate named priors',ident,arm)
                check(near(actual['O_COARSE_LOCALIZATION_PRIOR'],ov)and near(actual['S_COARSE_VISIBILITY_PRIOR'],sv),
                    'one cost per frame per prior',ident,arm)
                check(ov<=-2*math.log(oparams['epsilon'])+1e-10 and sv<=-2*math.log(smodel['epsilon'])+1e-10,
                    'finite one-frame caps',ident,arm)
                check(oa['O_point_terms']==0 and (sa is None or sa['S_image_or_2D_terms']==0),
                    'no hidden-point image correspondence',ident,arm)
                readout=result['keypoint_readout'];projected=C.project(base.xyz,R,t,base.K)
                for k,state in enumerate(task['states']):
                    if state=='S':
                        check(np.allclose(readout['reported_points'][k],projected[k],atol=1e-8,rtol=0),
                            'S output is model projection only',ident,arm)
                    elif state=='U':
                        check(readout['reported_points'][k]is None,'U remains region only',ident,arm)
                selection=result['candidate_selection'];candidates=selection['candidates']
                chosen=[c for c in candidates if c['name']==selection['selected_name']]
                check(len(chosen)==1,'one selected authenticated representative',ident,arm)
                if chosen:
                    check(near(chosen[0]['regularized_native_energy'],total),
                        'selector uses same independently reconstructed total',ident,arm)
                    minimum=min(c['total_energy']for c in candidates)
                    tolerance=32*np.finfo(float).eps*max(1.,abs(minimum),abs(chosen[0]['total_energy']))
                    check(chosen[0]['total_energy']<=minimum+tolerance,'minimum declared selection energy',ident,arm)
    check(rows==238 and counts==Counter(CAL=46,TEST=49,REMAINDER=143),'all238 severe0',None)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    receipt=dict(schema='INDEPENDENT_BOUNDED_O_S_NATIVE_CONTROL_AUDIT_V22',rows=rows,
        partitions=dict(counts),checks=checks,failure_count=len(failures),failures=failures,
        selected_arm_poses=selected,fallback_arm_poses=fallback,maximum_energy_difference=maxdiff,
        all238_kept=True,primary192=192,severe=0,source=E._binding(Path(__file__)),
        prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        parent_prediction_seal=E._binding(V.V.OUTPUT/'PREDICTION_SEAL.json'),
        model_bindings=[E._binding(p)for p in (OFIT.OUTPUT/'MODEL.json',SFIT.OUTPUT/'MODEL.json')],
        regularizer_wrapper_not_used_to_certify_own_objective=True,
        new_pose_fits_RGB_NN_or_reference_decodes=0,no_default_adoption=True)
    C.write(dest/'AUDIT.json',receipt)
    C.write(dest/'AUDIT_SEAL.json',dict(receipt_sha256=C.sha(dest/'AUDIT.json')))
    print(dict(rows=rows,checks=checks,failures=len(failures),max_energy_difference=maxdiff),flush=True)
    if failures:raise ValueError('Independent bounded-prior control audit failed')


if __name__=='__main__':main()
