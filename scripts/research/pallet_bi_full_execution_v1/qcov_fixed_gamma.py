"""Technical repair: q/covariance transport conditional on fixed COMMON Gamma.

The preceding diagnostic remains an immutable Sigma+Gamma donor experiment.
This separately registered adapter holds reliability at COMMON in every cell.
"""
from copy import deepcopy
from pathlib import Path
import time

from . import qcov as Q


def verify_old(old_diagnostic):
    seal=Q.read(old_diagnostic/'PREDICTION_SEAL.json')
    for name,digest in seal['artifacts'].items():
        if Q.sha(old_diagnostic/name)!=digest:raise ValueError('Original sealed D artifact changed')
    if sum(name.startswith('packets/')for name in seal['artifacts'])!=238:raise ValueError('Every original D packet must be sealed')
    return Q.sha(old_diagnostic/'PREDICTION_SEAL.json')


def fixed_gamma_packet(frame,common,bi,q_bit,covariance_bit):
    packet,provenance=Q.fixed_common_packet(frame,common,bi,q_bit,covariance_bit)
    complete,_,_=Q.effective(common)
    for point in packet['full_points']:
        k=point['native_corner']
        if k not in complete:continue
        point.pop('fit_reliability_weight',None)
        if 'fit_reliability_weight'in complete[k]:point['fit_reliability_weight']=deepcopy(complete[k]['fit_reliability_weight'])
    provenance['schema']='COUNTERFACTUAL_FIXED_COMMON_GRAPH_GAMMA_V1'
    provenance['Gamma_donor']='COMMON_FIXED_IN_ALL4CELLS'
    provenance['Sigma_donor_fields']=['covariance_px2','fit_covariance_px2']
    provenance['interpretation']='algorithmic q/Sigma sensitivity conditional on COMMON logical factor graph and COMMON reliability Gamma'
    for factor in provenance['factors']:factor['Gamma_donor']='COMMON'
    Q.validate_fixed_packet(common,packet,provenance)
    return packet,provenance


def review(prior_run,old_diagnostic,output):
    prior_run,old_diagnostic,output=map(Path,(prior_run,old_diagnostic,output));rows=[];missing=0;allcells=0
    old_seal_sha256=verify_old(old_diagnostic)
    inventory=Q.read(prior_run/'P0_INPUT_CONTRACT.json')['real']['inventory']
    for ordinal,frame in enumerate(inventory):
        saved=Q.read(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')['current_observation_and_solver_records']
        common=saved['OBS_COMMON_s20261018']['selection'];bi=saved['OBS_BI_s20261018']['selection'];old=Q.read(old_diagnostic/'packets'/f'{ordinal:04d}.json.gz')
        cache={Q.P.digest(Q.parsed_solver_payload(frame,p['selection']))for p in old.values()};changed=[]
        for name,qbit,cbit in [('Q0S0',0,0),('Q1S0',1,0),('Q0S1',0,1),('Q1S1',1,1)]:
            packet,prov=fixed_gamma_packet(frame,common,bi,qbit,cbit);signature=Q.P.digest(Q.parsed_solver_payload(frame,packet));allcells+=1
            if signature not in cache:changed.append(name);missing+=1
            cache.add(signature)
        if changed:rows.append(dict(ordinal=ordinal,id=frame['id'],new_numerical_cells=changed))
    report=dict(schema='FIXED_COMMON_GAMMA_TECHNICAL_REPAIR_REVIEW_V1',epoch=time.time(),source_sha256=Q.sha(__file__),
        original_qcov_source_sha256=Q.sha(Q.__file__),typed_solver_source_sha256=Q.sha(Q.S.__file__),
        old_D_prediction_seal_sha256=old_seal_sha256,old_D_root=str(old_diagnostic),full_denominator=238,total_cells=allcells,
        changed_frames=len(rows),new_numerical_packets=missing,max_additional_family_dispatches=2*missing,
        exact_previous_cell_reuses=allcells-missing,rows=rows,root_GO_required_before_new_calls=True,
        Gamma_fixed_COMMON=True,typed_partials_unchanged=True,numeric_reference_inputs=False,new_optimizer_calls=0)
    Q.write(output/'REVIEW.json',report);return report


def run(prior_run,old_diagnostic,output,*,root_GO=False):
    if root_GO is not True:raise PermissionError('Root review GO required for fixed-Gamma diagnostic repair')
    prior_run,old_diagnostic,output=map(Path,(prior_run,old_diagnostic,output));report=Q.read(output/'REVIEW.json')
    if report['source_sha256']!=Q.sha(__file__):raise ValueError('Reviewed adapter changed')
    if report['old_D_prediction_seal_sha256']!=verify_old(old_diagnostic):raise ValueError('Reviewed D seal changed')
    if report['original_qcov_source_sha256']!=Q.sha(Q.__file__)or report['typed_solver_source_sha256']!=Q.sha(Q.S.__file__):raise ValueError('Reviewed numerical source changed')
    start=time.monotonic();inventory=Q.read(prior_run/'P0_INPUT_CONTRACT.json')['real']['inventory'];rows=[];calls=0;dispatches=0
    for ordinal,frame in enumerate(inventory):
        saved=Q.read(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')['current_observation_and_solver_records']
        common=saved['OBS_COMMON_s20261018']['selection'];bi=saved['OBS_BI_s20261018']['selection']
        original=Q.read(old_diagnostic/'frames'/f'{ordinal:04d}.json.gz');old=Q.read(old_diagnostic/'packets'/f'{ordinal:04d}.json.gz');cache={}
        for name,p in old.items():
            payload=Q.parsed_solver_payload(frame,p['selection']);cache[Q.P.digest(payload)]=(payload,p['solver_result'],'OLD_SIGMA_GAMMA_'+name)
        cells={};packets={};framecalls=0;framedispatches=0;reuses=0
        for name,qbit,cbit in [('Q0S0',0,0),('Q1S0',1,0),('Q0S1',0,1),('Q1S1',1,1)]:
            selection,prov=fixed_gamma_packet(frame,common,bi,qbit,cbit);payload=Q.parsed_solver_payload(frame,selection);signature=Q.P.digest(payload)
            if signature in cache:
                previous,result,origin=cache[signature]
                if previous!=payload:raise ValueError('Cache hash collision')
                currentcalls=0;reuse=True;reuses+=1
            else:
                result=Q.S.solve_frame(frame,selection);currentcalls=result['optimizer_calls'];framedispatches+=2;origin=name;reuse=False
                cache[signature]=(payload,result,name)
            framecalls+=currentcalls
            cells[name]=dict(available=True,actual_pose=result['actual_pose'],accepted=result['accepted'],fallback=result['fallback'],
                status=result['status'],selected_family=(result.get('solution')or{}).get('selected_family'),family_readouts=Q.family_readouts(result),
                exact_numerical_solver_payload_sha256=signature,exact_after_typed_parse_cache_reused=reuse,cache_origin=origin,
                actual_new_family_optimizer_calls=currentcalls,source_transport_provenance=prov)
            packets[name]=dict(selection=selection,solver_result=result)
        record=deepcopy(original);record.update(cells=cells,actual_new_family_optimizer_calls=framecalls,
            new_family_dispatches=framedispatches,exact_numerical_packet_reuses=reuses,
            interpretation='algorithmic q/Sigma sensitivity conditional on COMMON logical graph and fixed COMMON Gamma',
            transported_Q1S1_matches_actual_BI_parsed_payload=Q.parsed_solver_payload(frame,packets['Q1S1']['selection'])==Q.parsed_solver_payload(frame,bi))
        Q.write(output/'packets'/f'{ordinal:04d}.json.gz',packets);Q.write(output/'frames'/f'{ordinal:04d}.json.gz',record)
        rows.append(record);calls+=framecalls;dispatches+=framedispatches
        if dispatches>report['max_additional_family_dispatches']:raise RuntimeError('Reviewed additional dispatch budget exceeded')
    summary=dict(schema='COUNTERFACTUAL_FIXED_COMMON_GRAPH_GAMMA_V1',status='DIAGNOSTIC_ONLY',full_denominator=238,
        actual_new_family_optimizer_calls=calls,actual_new_family_dispatches=dispatches,
        exact_numerical_packet_reuses=sum(r['exact_numerical_packet_reuses']for r in rows),
        source_sha256=Q.sha(__file__),review_sha256=Q.sha(output/'REVIEW.json'),old_D_root=str(old_diagnostic),
        Gamma_fixed_COMMON=True,original_D_retained_as_Sigma_plus_Gamma=True,all_family_endpoints_preserved=True,
        numeric_reference_inputs=False,wall_seconds=time.monotonic()-start)
    Q.write(output/'SUMMARY.json',summary)
    paths=[p for p in output.rglob('*')if p.is_file()and p.name!='PREDICTION_SEAL.json']
    Q.write(output/'PREDICTION_SEAL.json',dict(complete=True,full_denominator=238,before_new_scoring=True,numeric_reference_inputs=False,
        artifacts={str(p.relative_to(output)):Q.sha(p)for p in paths}))
    return summary
