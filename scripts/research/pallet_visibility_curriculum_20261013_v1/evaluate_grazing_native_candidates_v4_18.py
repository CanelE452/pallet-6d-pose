"""Uniform cached238 grazing-BI->U controls, no new pose optimization.

Use the identical v17 physical proposal bank and old authenticated-line CAL
model. Compare native-only and native+the unchanged frozen outline weight.
Only original-geometry B/I uncertainty changes; all S/U/O/T are preserved.
Historical observations are cached, not a new full-RGB runtime execution.
"""
import os
for _key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from copy import deepcopy
from collections import Counter
from pathlib import Path
import argparse
import shutil
import time
import types
import numpy as np

from . import common as C
from . import evaluate_cached_native_edges_v4_17 as B
from . import grazing_BI_uncertainty_v4_18 as G

OUTPUT=B.OUTPUT.parent/'grazing_native_candidates_v4_18'
INVENTORY=B.OUTPUT.parent/'grazing_BI_uncertainty_v4_18/INVENTORY_V1.json'
REGION_MODEL=B.E.FIX/'truncation_interval_model_CAL_v4_2/MODEL.json'
ARMS=('SAME_FROZEN_E','GRAZING_U_NATIVE_DMERGED','GRAZING_U_NATIVE_PLUS_MASK_DMERGED')


def adapt_cached_task(task,source,localization_model):
    geometry=source['initial_geometry'];before_states=list(task['states'])
    gate=G.apply_grazing_BI_uncertainty(geometry,before_states,
        geometry_tolerance_deg=task['spec']['config']['geometry_tolerance_deg'])
    for a,b in((geometry['native_xyz'],task['families']['ORIGINAL_WD']['native_xyz']),
        (geometry['R_cf'],task['families']['ORIGINAL_WD']['coarse_pose']['R']),
        (geometry['centroid'],task['families']['ORIGINAL_WD']['coarse_pose']['t'])):
        if not np.allclose(a,b,atol=1e-12,rtol=0):raise ValueError('Original initial geometry, never candidate geometry, defines the band')
    filtered=G.filter_grazing_BI_reuse(task['spec']['visible'],task['boundary_merged'],gate['states'],
        gate['new_U_native_ids'],initial_self_hidden=geometry['initial_self_hidden'])
    regions=G.append_grazing_U_regions(geometry,gate['states'],gate['new_U_native_ids'],
        raw_native_points=source['original_raw_points'],
        nonself_prediction_scale_px=source['ot_domains']['nonself_prediction_scale_px'],
        localization_model=localization_model,existing_u_regions=task['u_regions'])
    result={**task,'spec':{**task['spec'],'states':gate['states'],'visible':filtered['visible']},
        'states':gate['states'],'boundary_merged':filtered['boundary_observations'],'u_regions':regions['u_regions']}
    result.pop('boundary_original',None)
    audit=dict(original_states=before_states,geometry_gate=gate,
        precise_factor_reuse_filter={k:v for k,v in filtered.items()if k not in('visible','boundary_observations')},
        new_U_regions=regions['new_u_regions'],missing_new_U_raw_native_ids=regions['missing_new_U_raw_native_ids'],
        U_radius_px=regions['radius_px'],borrowed_CAL264_radius_ratio=regions['radius_ratio'],
        original_U_domains_preserved=True,
        original_OT_numeric_domains_sha256=B.E._digest(task['spec']['domains']),
        adapted_OT_numeric_domains_sha256=B.E._digest(result['spec']['domains']),
        retained_OT_original_validated_bundle_numeric_inputs_only=True,
        full_RGB_runtime_bundle_or_fresh_image_binding_not_claimed=True,
        new_state_and_domain_math_binding=B.E._digest(dict(states=gate['states'],domains=result['spec']['domains'],u_regions=regions['u_regions'])),
        raw_initial_S_queries=0,new_RGB_queries=0,new_NN_forwards=0,
        historical_cached_grazing_queries_may_have_occurred=True,
        old_raw_nonS_centers_from_saved_prediction_archive_only=True)
    if audit['original_OT_numeric_domains_sha256']!=audit['adapted_OT_numeric_domains_sha256']:
        raise ValueError('Existing O/T numeric inputs changed')
    result['grazing_adaptation']=audit
    return C.finite(result)


def _load():
    parent=C.read(B.OUTPUT/'PROTOCOL.json');seal=C.read(B.OUTPUT/'INPUT_SEAL.json')
    if (seal['rows']!=238 or C.sha(B.OUTPUT/'INPUTS.jsonl.gz')!=seal['input_sha256']
        or C.sha(B.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Exact cached v17 inputs required')
    old=B.ARCHIVE.OUTPUT;oldseal=C.read(old/'INPUT_SEAL.json')
    if (oldseal['rows']!=238 or C.sha(old/'INPUTS.jsonl.gz')!=oldseal['input_sha256']
        or C.sha(old/'PROTOCOL.json')!=oldseal['protocol_sha256']):raise ValueError('Original raw-coordinate whitelist source seal required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):B.E._verify_sources(parent[key])
    model=B._verified_model();localization=C.read(REGION_MODEL);inventory=C.read(INVENTORY)
    expected={r['id']:r['new_U_native_ids']for r in inventory['rows']}
    tasks=list(C.rows(B.OUTPUT/'INPUTS.jsonl.gz'));by_id={r['id']:r for r in tasks};adapted={}
    for archived in C.rows(old/'INPUTS.jsonl.gz'):
        ident=archived['id'];task=by_id[ident]
        if task['states']!=archived['spec']['states']:raise ValueError('Same original observations required')
        # Whitelist access: initial model geometry, only new-nonS raw centers,
        # and the immutable nonS scale. No scores/targets/human state fields.
        adapted[ident]=adapt_cached_task(task,archived['finish_input']['_original_source'],localization)
        if adapted[ident]['grazing_adaptation']['geometry_gate']['new_U_native_ids']!=expected[ident]:
            raise ValueError('Predeclared geometry-only 31-point inventory changed')
    result=[adapted[r['id']]for r in tasks]
    if (len(result)!=238 or[r['id']for r in result]!=parent['ids']
        or Counter(r['partition']for r in result)!=Counter(CAL=46,TEST=49,REMAINDER=143)
        or sum(len(r['grazing_adaptation']['geometry_gate']['new_U_native_ids'])for r in result)!=31):
        raise ValueError('All original238 and fixed31 grazing flags required')
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    for filename in(__file__,G.__file__,G.OLD.__file__,G.U.__file__):
        p=Path(filename);sources[str(p.relative_to(C.ROOT))]=C.sha(p)
    for folder in(B.OUTPUT,old):
        for name in('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz'):
            p=folder/name;inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    for p in(INVENTORY,REGION_MODEL,INVENTORY.parent/'PROPOSED_CONTROL_PLAN_V1.json'):
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    return result,model,sources,inputs,parent['protected_user_sha256']


def select_native_plus_mask(candidates,weight):
    """Explicit combined-energy control, no fitted weight or GT routing."""
    weight=float(weight)
    if not np.isfinite(weight)or weight<0:raise ValueError('Unchanged finite nonnegative outline weight required')
    rows=[]
    for index,c in enumerate(candidates):
        if c['accepted']is not True or c['all_joint_constraints_satisfied']is not True:continue
        mask=c.get('global_mask_cost')
        if weight and mask is None:continue
        native=sum(r['weighted_robust_cost']for r in c['native_costs'])
        total=native+weight*(0. if mask is None else mask)
        if not np.isfinite(total)or total<0:raise ValueError('Finite nonnegative combined energy required')
        rows.append(dict(index=index,name=c['name'],family=c['family'],native_energy=native,
            global_mask_cost=mask,outline_factor_weight=weight,total_selection_energy=total))
    selected=None;ties=[]
    if rows:
        minimum=min(r['total_selection_energy']for r in rows)
        ties=[r for r in rows if abs(r['total_selection_energy']-minimum)<=32*np.finfo(float).eps*max(1.,abs(minimum),abs(r['total_selection_energy']))]
        selected=ties[0]
    return dict(selected_name=None if selected is None else selected['name'],
        selected_family=None if selected is None else selected['family'],candidates=rows,
        machine_numeric_tie_indices=[r['index']for r in ties],
        objective='native energy + frozen outline_factor_weight * frozen outline energy',
        global_mask_may_override_contrary_native_cost_in_this_explicit_control=True,
        no_default_adoption=True,global_uniqueness_proved=False)


def _run(task,model):
    started=time.monotonic();before=B.E._digest(task);records=task['boundary_merged']
    evaluations=[];full={};bundles={};ocfg=B.OUTLINE._outline_config(task['outline']['configuration'])
    angle=B.ANGLE.authentication_angular_parameters(model)
    for name,family in task['families'].items():
        spec=deepcopy(task['spec']);spec.update(xyz=family['native_xyz'],coarse_pose=family['coarse_pose'])
        spec['config']['prior_weight']=0.
        bundle=B.AUTH.authenticate_supported_native_edges(spec['xyz'],spec['states'],spec['visible'],records,spec['image_hw'],
            initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
            config=spec['config'],**angle)
        problem=B.NATIVE.prepare_native_edge_problem(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],
            spec['coarse_pose'],config=spec['config'],boundary_observations=records,u_regions=task['u_regions'],edge_bundle=bundle)
        Q=np.asarray(family['physical_registration_Q'])
        if not np.allclose(Q,problem.Q,atol=1e-12,rtol=0):raise ValueError('Unchanged physical family Q required')
        bundles[name]=bundle
        for index,proposal in enumerate(task['proposals']):
            R=np.asarray(proposal['R_physical'])@Q.T;t=np.asarray(proposal['t']);checked=problem.evaluate(R,t)
            token=name+'/'+str(index);mask=0. if not task['outline'].get('supported')else None;mask_audit=None
            if checked['accepted']and task['outline'].get('supported'):
                try:
                    p=B.NATIVE.V._pose_vector(R,t,problem.scale)
                    value,_,mask_audit=B.OUTLINE.outline_value_and_gradient(p,problem.xyz,problem.K,problem.scale,task['outline'],ocfg)
                    if mask_audit.get('valid_projection')and np.isfinite(value):mask=float(value)
                except(ValueError,FloatingPointError,np.linalg.LinAlgError)as error:
                    mask_audit=dict(valid_projection=False,reason=str(error))
            evaluations.append(dict(name=token,family=name,accepted=checked['accepted'],all_joint_constraints_satisfied=checked['accepted'],
                native_costs=B._bridge_costs(checked.get('native_costs',[])),global_mask_cost=mask,
                physical_pose_digest=proposal['pose_digest'],minimum_joint_slack=checked.get('minimum_joint_slack'),
                rejected_reason=checked.get('reason')))
            if checked['accepted']:full[token]=dict(evaluation=checked,family=family,mask_audit=mask_audit,proposal_index=index)
    native=B._compact_selection(B.AUTH.select_hierarchical_native_candidates(task['states'],evaluations))
    combined=select_native_plus_mask(evaluations,ocfg['outline_factor_weight'])
    arms={ARMS[0]:deepcopy(task['baseline'])};tangents={};audits={}
    for arm,selection in((ARMS[1],native),(ARMS[2],combined)):
        token=selection['selected_name']
        if token is None:
            family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
            selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,native_costs=[],
                reason='no_cached_proposal_with_current_grazing_U_joint_guards_and_defined_control_energy')
        else:
            item=full[token];selected={**item['evaluation'],'family':item['family'],'proposal_index':item['proposal_index'],
                'global_mask_audit':item['mask_audit'],'frozen_outline_factor_weight':ocfg['outline_factor_weight']}
        pose=B._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
        arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=task['states'],solution=selected,
            keypoint_readout=B._readout(task,records,selected,pose),candidate_selection=selection,
            ambiguity_audit=dict(native_frontier=deepcopy(native['ambiguity']),global_uniqueness_proved=False),
            partial_boundary_observation=dict(boundary_observations=records,edge_authentication_by_family=bundles),
            fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False),
            method=arm,WD_family=selected['family']['name'])
        tangents[arm]=dict(variance_inflation=4.,active_native_ids=sorted({r['native_corner']for r in selected['native_costs']
            if r['kind']=='UNCERTAIN_ARCHIVED_TANGENT'}),archived_tangent_is_not_precise_RGB_point=True)
        audits[arm]=dict(selected_name=token,selection_sha256=B.E._digest(selection),
            selected_native_objective=selected['objective'],pose_optimizations=0)
    if B.E._digest(task)!=before:raise ValueError('Frozen adapted inputs mutated')
    return C.finite(dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],image_sha256=task['image_sha256'],
        states=task['states'],original_states=task['grazing_adaptation']['original_states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family',task['baseline'].get('WD_family'))for a in ARMS},
        tangent_control_audit=tangents,grazing_adaptation=task['grazing_adaptation'],grazing_candidate_audit=audits,
        candidate_evaluations=evaluations,input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(all238_uniform=True,original_initial_S_preserved=True,previous_U_O_T_preserved=True,
            new_U_range_only=True,new_U_precise_BI_or_partial_costs=0,new_RGB_or_NN_or_pose_fits=0,
            old_angular_CAL_model_not_new_NMS_model=True,old_BI_covariance_and_reliability_preserved=True,
            all_current_joint_guards_recomputed=True,GT_or_ID_or_partition_routing=False,
            observations_cached_not_fresh_RGB_API=True,no_default_adoption=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior versioned artifacts')
    tasks,model,sources,inputs,protected=_load()
    for hashes in(sources,inputs,protected):B.E._verify_sources(hashes)
    output.mkdir(parents=True)
    protocol=dict(schema='GRAZING_BI_U_CACHED_NATIVE_ALL238_V4_18',ids=[t['id']for t in tasks],arms=list(ARMS),
        population=dict(CAL=46,TEST=49,REMAINDER=143,severe=0),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,angular_model_binding=B.E._binding(B.ANGLE.OUTPUT/'MODEL.json'),
        U_localization_model_binding=B.E._binding(REGION_MODEL),predeclared_grazing_points=31,
        existing_geometry_tolerance_only=True,old_CAL264_radius_ratio=.544750754221928,
        baseline='immutable uniformly applied e',proposal_bank='byte-values identical v17 saved physical poses, all bothfamilies',
        native_control='native minsum/Pareto audit, mask only machine ties',
        combined_control='native+unchanged per-task frozen outline_factor_weight times same mask cost',
        old_NMS_generation_used=False,no_new_optimization_or_RGB_or_NN=True,
        retained_OT_original_validated_numeric_domains_only=True,new_full_RGB_runtime_binding_claimed=False,
        all238_before_new_metrics=True,reused_TEST_and_remaining_exploratory=True,no_default_adoption=True)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for rel,digest in sources.items():
        p=output/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/rel,p)
        if C.sha(p)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_predictions_and_metrics=True))
    print('PREPARED238 grazing31',flush=True)


def predict(output=OUTPUT,workers=8):
    runner=types.FunctionType(B.predict.__code__,{**B.predict.__globals__,'_run':_run},B.predict.__name__)
    return runner(Path(output),workers)


def score(output=OUTPUT):
    scorer=types.FunctionType(B.ARCHIVE.score.__code__,{**B.ARCHIVE.score.__globals__,'ARMS':ARMS},B.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','predict','score'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='predict':predict(a.output,a.workers)
    else:score(a.output)
