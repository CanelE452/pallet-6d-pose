"""Sealed CAL46-only V30 association-selected line and tangent noise.

The raw observations remain V29. This NEW source reuses the frozen V29 driver
mathematics with explicit V30 versions/profile bindings; parent code is unchanged.

`plan`/`prepare` never decode CAL reference coordinates. `fit` is separate and
requires explicit root authorization. Every retained group and selected-line
axis is bound before labels. Original CAL44 geometry-only alignment is reused;
all46 inventory/zero-support frames and four LOGO folds are retained. No image,
NN, point correction, pose solve, TEST or remaining reference is read.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import numpy as np
from . import common as C
from . import geometry_guided_edge_noise_calibration_v4_30 as M
from . import interior_observation as INTERIOR
NEIGHBORS=INTERIOR.NEIGHBORS

OLD=M.LINE.OLD
OUTPUT=M.LINE.OUTPUT.parent/'CAL_geometry_guided_edge_noise_v4_30'
PREPARED=M.LINE.OUTPUT.parent/'geometry_guided_native_evidence_v4_30_checks'
PLAN_OUTPUT=M.LINE.OUTPUT.parent/'CAL_geometry_guided_edge_noise_plan_v4_30'
LABELS=OLD.LABELS
CENTRAL_PLAN=OUTPUT.parent/'observed_bi_associated_edges_plan_v4_30/PREDECLARATION.json'
CENTRAL_PLAN_SHA256='1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'
PROTOCOL_SCHEMA='SEALED_MULTIGUIDE_GRAPH_SELECTED_CAL46_EDGE_TANGENT_PROTOCOL_V30'


def _verify(mapping):
    for relative,digest in mapping.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen input/source changed: '+relative)


def _union_maps(*mappings):
    result={}
    for mapping in mappings:
        for relative,digest in mapping.items():
            if relative in result and result[relative]!=digest:
                raise ValueError('Source/input/protected ancestry conflict: '+relative)
            result[relative]=digest
    return result


def _central_declaration():
    if C.sha(CENTRAL_PLAN)!=CENTRAL_PLAN_SHA256:
        raise ValueError('Frozen central V30 declaration required')
    central=C.read(CENTRAL_PLAN)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        _verify(central[key])
    # Also fail closed on inconsistent cross-class path bindings.
    _union_maps(*(central[k]for k in ('source_sha256','input_sha256','protected_user_sha256')))
    return central


def _sources():
    modules=[sys.modules[__name__],M,M.PARENT,M.CORE,M.CORE.ASSOCIATION,C,M.LINE,M.TANGENT,M.TANGENT.DIAG,OLD,OLD.OLD_CAL,
        OLD.ALIGN,OLD.FACTOR,OLD.N,OLD.N.V,OLD.N.J,OLD.N.P,INTERIOR]
    return {str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__) for m in modules}


def _snapshot(output,sources):
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Snapshot differs')


def plan(output=PLAN_OUTPUT):
    output=Path(output).resolve()
    central=_central_declaration()
    if output.exists():raise FileExistsError('Preserve prior noise declarations')
    sources=_union_maps(central['source_sha256'],_sources())
    inputs=_union_maps(central['input_sha256'],
        {str(CENTRAL_PLAN.relative_to(C.ROOT)):CENTRAL_PLAN_SHA256})
    output.mkdir(parents=True)
    value=dict(schema='NEW_MULTIGUIDE_GRAPHSELECTED_NOISE_CAL_PLAN_V30',UTC=datetime.now(timezone.utc).isoformat(),
        observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,noise_population=M.POPULATION,selection_profile=M.SELECTION_PROFILE,source_policies=M.SOURCE_POLICIES,
        line_recipes={p:M.recipe_for(p) for p in M.SOURCE_POLICIES},
        tangent_recipes={p:M.recipe_for(p,tangent=True) for p in M.SOURCE_POLICIES},
        metadata_contract='CAL46 ordered unique profiles each groups(1 graphselected component/physicaledge) and tangent_records(selected actual line axis)',
        CAL_reference=str(LABELS.relative_to(C.ROOT)),CAL_reference_bytes_sha256=C.sha(LABELS),
        sources=sources,input_sha256=inputs,
        central_source_sha256=deepcopy(central['source_sha256']),
        central_input_sha256=deepcopy(central['input_sha256']),
        central_plan_path=str(CENTRAL_PLAN.relative_to(C.ROOT)),central_plan_sha256=CENTRAL_PLAN_SHA256,
        protected_user_sha256=central['protected_user_sha256'],source_and_metadata_before_new_CAL_reference_decode=True,
        labels_historically_exposed_exploratory=True,old_model_relabelled=False,
        no_support_policy='zero supported line/tangent population => explicit MODEL_UNAVAILABLE; no old model substitution; single-session scale marked descriptive/weak',
        unseen_fold_policy='held session excluded before numerical errors; empty training fold explicit MODEL_UNAVAILABLE',
        non_Cal_numeric_reference_decodes=0,image_NN_point_or_pose_fits=0,default_adoption=False)
    C.write(output/'PLAN.json',value);_snapshot(output,sources)
    C.write(output/'PLAN_SEAL.json',dict(plan_sha256=C.sha(output/'PLAN.json'),sources=sources))
    return value


def _native_index(value):
    if isinstance(value,(bool,np.bool_))or not isinstance(value,(int,np.integer))or not 0<=int(value)<8:
        raise ValueError('Native integer index required')
    return int(value)


def validate_prepared_row(row):
    """Reference-free shape/owner/selection/axis checks; no RGB/KP synthesis."""
    if (row.get('partition')!='CAL' or row.get('observation_generation')!=M.OBSERVATION_FAMILY
            or row.get('raw_observation_generation')!=M.RAW_OBSERVATION_GENERATION
            or row.get('noise_population')!=M.POPULATION
            or row.get('selection_profile')!=M.SELECTION_PROFILE):
        raise ValueError('New-generation CAL metadata only')
    if set(row['profiles'])!=set(M.SOURCE_POLICIES):raise ValueError('Both uniform source policies required')
    states=row['states'];hidden=row['initial_geometry']['initial_self_hidden']
    if len(states)!=8 or len(hidden)!=8:raise ValueError('Authoritative states and initial S gate required')
    result={k:deepcopy(row[k])for k in ('id','partition','session','initial_geometry','states','observation_generation','raw_observation_generation','noise_population','selection_profile')};result['profiles']={}
    for policy in M.SOURCE_POLICIES:
        profile=row['profiles'][policy]
        if profile.get('selection_profile')!=M.SELECTION_PROFILE:
            raise ValueError('Each source-policy profile must bind the V30 selection profile')
        groups=[];components={};seen=set()
        for group in profile['groups']:
            if len(group['components'])!=1:raise ValueError('Only deterministic selected line may enter CAL likelihood')
            edge=group['components'][0];k=_native_index(edge.get('native_corner'));mate=_native_index(edge.get('native_neighbor'))
            # Gate before saved line/support fields, including excluded S targets.
            if hidden[k]or states[k]not in ('B','I')or hidden[mate]or states[mate]=='S':
                raise ValueError('Excluded S/U/O/T anchor line cannot enter prepared noise population')
            pair=tuple(sorted([k,mate]))
            if mate not in NEIGHBORS[k]or pair in seen or pair!=tuple(group['physical_native_edge']):
                raise ValueError('Unique named incident physical edge required')
            seen.add(pair);M._validate_support_priors(group['components'])
            weight=float(group['reliability_weight'])
            if not np.isfinite(weight)or weight<0:raise ValueError('Fixed nonnegative reliability required')
            n=np.asarray(edge['normal_unit_xy'],float);anchor=np.asarray(edge['support_anchor_xy'],float);offset=float(edge['offset_px'])
            if (n.shape!=(2,)or anchor.shape!=(2,)or not np.isfinite([*n,*anchor,offset]).all()
                    or not np.isclose(n@n,1.,atol=1e-10,rtol=0)or abs(n@anchor+offset)>1e-7):
                raise ValueError('Measured unit TLS normal/offset and on-line support anchor required')
            floor=M.conditional_covariance_floors(edge['actual_inlier_span_px'],edge['normal_floor_px'])
            if not np.allclose(floor,[edge['normal_variance_px2'],edge['angular_variance_rad2']],atol=1e-14,rtol=1e-12):
                raise ValueError('Zero-CAL exact per-line floors required before total-error fitting')
            if edge.get('normal_angle_correlation',0.)!=0.:raise ValueError('Fixed zero correlation required')
            if edge.get('selected_chain_ref')!=group.get('selected_chain_ref') or edge.get('selected_chain_ref')is None:
                raise ValueError('Exact graphselected chain reference required')
            if edge['component_id'] in components:raise ValueError('Component identity must be unique')
            components[edge['component_id']]=edge;groups.append(deepcopy(group))
        tangents=[];ids=set()
        for record in profile['tangent_records']:
            point=M.prepared_tangent(record,states,hidden)
            if point is None:continue
            k=point['native_corner']
            if k in ids:raise ValueError('One archived tangent per active one-direction corner')
            ids.add(k);ref=point['selected_line_ref']
            if not isinstance(ref,str)or ref not in components:raise ValueError('Tangent axis must join one active selected component')
            edge=components[ref]
            if k not in (edge['native_corner'],edge['native_neighbor']):raise ValueError('Tangent must be incident to selected physical line')
            t=np.asarray(point['selected_line_tangent_unit_xy']);n=np.asarray(edge['normal_unit_xy'])
            if abs(t@n)>1e-10:raise ValueError('Actual-line tangent must be orthogonal to measured normal; original-guide tangent is forbidden')
            tangents.append(point)
        result['profiles'][policy]=dict(groups=groups,tangents=tangents,
            runtime_plan_digest=profile['runtime_plan_digest'],source_policy=policy,
            selection_profile=M.SELECTION_PROFILE)
    return result


def prepare(output=OUTPUT,prepared=PREPARED,plan_folder=PLAN_OUTPUT):
    output,prepared,plan_folder=map(lambda p:Path(p).resolve(),(output,prepared,plan_folder))
    if output.exists():raise FileExistsError('Preserve previous CAL preparations')
    declaration=C.read(plan_folder/'PLAN.json');pseal=C.read(plan_folder/'PLAN_SEAL.json')
    central=_central_declaration()
    if (C.sha(plan_folder/'PLAN.json')!=pseal['plan_sha256']
            or declaration['sources']!=_union_maps(central['source_sha256'],_sources())
            or declaration['central_source_sha256']!=central['source_sha256']
            or declaration['central_input_sha256']!=central['input_sha256']
            or declaration['input_sha256']!=_union_maps(central['input_sha256'],
                {str(CENTRAL_PLAN.relative_to(C.ROOT)):CENTRAL_PLAN_SHA256})
            or declaration['protected_user_sha256']!=central['protected_user_sha256']):
        raise ValueError('Frozen fitter declaration/source required before labels')
    for key in ('sources','input_sha256','protected_user_sha256'):_verify(declaration[key])
    if C.sha(CENTRAL_PLAN)!=declaration['central_plan_sha256']:raise ValueError('Central declaration changed')
    if declaration['CAL_reference_bytes_sha256']!=C.sha(LABELS):raise ValueError('CAL reference bytes changed')
    prep=C.read(prepared/'PREPARATION_SEAL.json');seal=C.read(prepared/'CAL_PREFLIGHT_SEAL_V1.json')
    report=C.read(prepared/'CAL_PREFLIGHT_V1.json');journal=prepared/'CAL_PREPARED_GROUPS.jsonl.gz'
    if (C.sha(prepared/'PREPARATION_SEAL.json')!=seal['preparation_sha256']
            or C.sha(prepared/'CAL_PREFLIGHT_V1.json')!=seal['report_sha256']
            or C.sha(journal)!=seal['prepared_groups_sha256'] or report['error_count']!=0
            or report['successful_metadata_rows']!=46):raise ValueError('Sealed successful new CAL46 metadata required')
    for name in ('source_sha256','input_sha256','protected_user_sha256'):_verify(prep[name])
    rows=list(C.rows(journal));ids=prep['ids']
    if len(rows)!=46 or len(set(ids))!=46 or [r['id']for r in rows]!=ids:raise ValueError('Complete unique ordered CAL46 required')
    rows=[validate_prepared_row(r)for r in rows]
    sources=_union_maps(prep['source_sha256'],declaration['sources'])
    bound=[journal,prepared/'PREPARATION_SEAL.json',prepared/'CAL_PREFLIGHT_SEAL_V1.json',prepared/'CAL_PREFLIGHT_V1.json',plan_folder/'PLAN.json',plan_folder/'PLAN_SEAL.json',CENTRAL_PLAN,LABELS]
    inputs=_union_maps(prep['input_sha256'],declaration['input_sha256'],
        {str(p.relative_to(C.ROOT)):C.sha(p) for p in bound})
    protected=_union_maps(prep['protected_user_sha256'],declaration['protected_user_sha256'])
    _union_maps(sources,inputs,protected)
    output.mkdir(parents=True);_snapshot(output,sources)
    protocol=dict(schema=PROTOCOL_SCHEMA,UTC=datetime.now(timezone.utc).isoformat(),ids=ids,rows=46,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,noise_population=M.POPULATION,selection_profile=M.SELECTION_PROFILE,
        line_recipes={p:M.recipe_for(p)for p in M.SOURCE_POLICIES},tangent_recipes={p:M.recipe_for(p,tangent=True)for p in M.SOURCE_POLICIES},
        CAL_label_path=str(LABELS.relative_to(C.ROOT)),numeric_CAL_reference_decodes=0,
        old_noise_or_gamma_relabelled=False,new_image_NN_point_pose_fits=0)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz',rows)
    for mapping in (sources,inputs,protected):_verify(mapping)
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'),rows=46,
        selected_groups_tangent_axes_and_policy_populations_fixed_before_labels=True,
        before_new_CAL_reference_decode=True,no_TEST_or_remaining_decode=True))
    print(dict(prepared_CAL46=46,per_policy={p:dict(groups=sum(len(r['profiles'][p]['groups'])for r in rows),tangents=sum(len(r['profiles'][p]['tangents'])for r in rows))for p in M.SOURCE_POLICIES},labels=0),flush=True)


def _verify_preparation(output):
    protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if (protocol['schema']!=PROTOCOL_SCHEMA or protocol['observation_generation']!=M.OBSERVATION_FAMILY or protocol['raw_observation_generation']!=M.RAW_OBSERVATION_GENERATION
            or protocol.get('noise_population')!=M.POPULATION or protocol.get('selection_profile')!=M.SELECTION_PROFILE
            or protocol['line_recipes']!={p:M.recipe_for(p)for p in M.SOURCE_POLICIES}
            or protocol['tangent_recipes']!={p:M.recipe_for(p,tangent=True)for p in M.SOURCE_POLICIES}
            or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz')!=seal['CAL_inputs_sha256']):raise ValueError('New v30 prepared generation/source recipes changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    return protocol


def _fit_or_status(records,policy,kind,bindings,held=None):
    selected=[r for r in records if r['compatible'] and r['session']!=held]
    key='groups' if kind=='line' else 'tangents'
    if not any(r[key]for r in selected):return dict(status='MODEL_UNAVAILABLE_NO_SUPPORTED_CAL_TRAINING',model=None,old_model_substituted=False)
    fn=M.fit_geometry_guided_line_model if kind=='line' else M.fit_geometry_guided_tangent_model
    model=fn(records,source_policy=policy,heldout_session=held,input_bindings=bindings)
    return dict(status='FITTED_SINGLE_SESSION_DESCRIPTIVE' if model['single_session_scale_caveat'] else 'FITTED_CAL_ONLY',model=model)



def _heldout_description(rows,session,line_result,tangent_result):
    held=[r for r in rows if r['compatible'] and r['session']==session]
    answer=dict(heldout_supported_line_frames=sum(bool(r['groups'])for r in held),
        heldout_supported_tangent_frames=sum(bool(r['tangents'])for r in held),
        heldout_line_NLL=None,heldout_tangent_NLL=None,
        selected_population_descriptive_only=True,independent_coverage_claim=False)
    model=line_result['model'];supported=[r for r in held if r['groups']]
    if model is not None and supported:
        groups=[];weights=[]
        for row in supported:
            for group in row['groups']:
                groups.append([dict(error=np.array([c['normal_error_px'],c['angular_sine_error']]),
                    floor=np.asarray(c['conditional_variance_floor']),mixture_component_prior=c['mixture_component_prior'])for c in group['components']])
                weights.append(1./len(supported)/len(row['groups']))
        answer['heldout_line_NLL']=M.mixture_noise_nll_and_gradient(
            [model['normal_variance_component_px2'],model['angular_variance_component_sine2']],groups,weights)[0]
    model=tangent_result['model'];supported=[r for r in held if r['tangents']]
    if model is not None and supported:
        points=[];weights=[]
        for row in supported:
            points.extend(row['tangents']);weights.extend([1./len(supported)/len(row['tangents'])]*len(row['tangents']))
        answer['heldout_tangent_NLL']=M.scalar_tangent_nll_and_gradient(model['gamma'],
            [p['signed_tangent_error_px']for p in points],[p['original_variance_axis_px2']for p in points],weights)[0]
    return answer

def fit(output=OUTPUT,*,root_CAL_fit_GO=False):
    if root_CAL_fit_GO is not True:raise PermissionError('Separate explicit root CAL-reference/fit GO is required')
    output=Path(output).resolve();protocol=_verify_preparation(output)
    if (output/'MODEL_SEAL.json').exists() or any((output/p/'MODEL.json').exists()for p in M.SOURCE_POLICIES):raise FileExistsError('Preserve prior fitted models')
    prepared=list(C.rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'))
    labels=C.read(C.ROOT/protocol['CAL_label_path'])['rows'];records={p:[]for p in M.SOURCE_POLICIES}
    for row in prepared:
        label=labels[row['id']]
        if label['metadata'].get('partition')!='CAL':raise ValueError('CAL reference only')
        geometry=row['initial_geometry'];alignment=OLD.ALIGN._reference_alignment({'runtime':{'initial_geometry':geometry}},label)
        xyz=np.asarray(geometry['native_xyz']);K=np.asarray(geometry['K']);scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
        if alignment['compatible']:
            p=OLD.N.V._pose_vector(alignment['R_reference_native'],alignment['t_reference'],scale)
            projection=C.project(xyz,alignment['R_reference_native'],alignment['t_reference'],K)
        for policy in M.SOURCE_POLICIES:
            record=dict(id=row['id'],partition='CAL',session=row['session'],compatible=bool(alignment['compatible']),
                observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,noise_population=M.POPULATION,selection_profile=M.SELECTION_PROFILE,source_policy=policy,
                fixed_geometry_phase=int(alignment['phase']),geometry_gap_m=float(alignment['geometry_gap_m']),
                runtime_plan_digest=row['profiles'][policy]['runtime_plan_digest'],groups=[],tangents=[])
            if alignment['compatible']:
                for group in row['profiles'][policy]['groups']:
                    edge=group['components'][0];_,_,audit=OLD.FACTOR.projected_edge_residual_and_jacobian(p,xyz,K,scale,edge)
                    floor=M.conditional_covariance_floors(edge['actual_inlier_span_px'],edge['normal_floor_px'])
                    component=dict(normal_error_px=audit['normal_residual_px'],angular_sine_error=audit['angular_sine_residual'],
                        actual_inlier_span_px=edge['actual_inlier_span_px'],normal_floor_px=edge['normal_floor_px'],conditional_variance_floor=floor.tolist(),
                        source_component=deepcopy(edge),support_id=edge['support_id'],fit_variant_id=edge['fit_variant_id'],component_id=edge['component_id'],mixture_component_prior=1.)
                    record['groups'].append(dict(physical_native_edge=group['physical_native_edge'],components=[component]))
                for point in row['profiles'][policy]['tangents']:
                    t=np.asarray(point['selected_line_tangent_unit_xy']);q=np.asarray(point['archived_q_xy']);k=point['native_corner']
                    record['tangents'].append({**deepcopy(point),'signed_tangent_error_px':float(t@(q-projection[k]))})
            records[policy].append(record)
    if len(prepared)!=46 or any(sum(r['compatible']for r in rs)!=44 for rs in records.values()):raise ValueError('Expected original CAL46/fixed44 inventory')
    models={};report={};bound=protocol['input_sha256']
    for policy,rows in records.items():
        folder=output/policy;folder.mkdir();models[policy]={};folds=[]
        for kind in ('line','tangent'):
            status=_fit_or_status(rows,policy,kind,bound);models[policy][kind]=status
            name='MODEL.json' if kind=='line' else 'TANGENT_MODEL.json'
            C.write(folder/name,status['model'] if status['model']is not None else status)
        for session in sorted({r['session']for r in rows}):
            line=_fit_or_status(rows,policy,'line',bound,session);tangent=_fit_or_status(rows,policy,'tangent',bound,session)
            folds.append(dict(heldout_session=session,heldout_ids=[r['id']for r in rows if r['session']==session],
                line=line,tangent=tangent,**_heldout_description(rows,session,line,tangent)))
        C.save_rows(folder/'CAL_ERRORS.jsonl.gz',rows)
        report[policy]=dict(line=models[policy]['line']['status'],tangent=models[policy]['tangent']['status'],
            inventory46=46,compatible44=44,LOGO_folds=folds,no_error_or_size_trimming=True,old_model_substituted=False)
    C.write(output/'CAL_LOGO_REPORT.json',report);_verify_preparation(output)
    assets={str(p.relative_to(output)):C.sha(p)for p in output.rglob('*')if p.is_file()and p.name in ('MODEL.json','TANGENT_MODEL.json','CAL_ERRORS.jsonl.gz','CAL_LOGO_REPORT.json')}
    C.write(output/'MODEL_SEAL.json',dict(assets=assets,preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],fit_partition='CAL',
        observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        noise_population=M.POPULATION,selection_profile=M.SELECTION_PROFILE,
        CAL_inventory=46,compatible_inventory=44,new_image_NN_point_or_pose_fits=0,old_model_relabelled=False,default_adoption=False))
    print({policy:{kind:dict(status=s['status'],sigma_normal_px=s['model'].get('sigma_normal_px')if s['model']else None,sigma_sine=s['model'].get('sigma_sine')if s['model']else None,gamma=s['model'].get('gamma')if s['model']else None)for kind,s in kinds.items()}for policy,kinds in models.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--plan-folder',type=Path,default=PLAN_OUTPUT);parser.add_argument('--root-CAL-fit-GO',action='store_true')
    args=parser.parse_args()
    if args.stage=='plan':plan(args.plan_folder)
    elif args.stage=='prepare':prepare(args.output,args.prepared,args.plan_folder)
    else:fit(args.output,root_CAL_fit_GO=args.root_CAL_fit_GO)
