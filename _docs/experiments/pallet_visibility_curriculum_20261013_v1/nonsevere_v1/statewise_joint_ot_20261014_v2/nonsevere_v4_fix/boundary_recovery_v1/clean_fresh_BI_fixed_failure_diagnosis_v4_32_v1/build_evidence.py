"""Postscore, fixed-three diagnosis; stored numeric data only, no references."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import audit_clean_fresh_BI_poses_v4_32 as A

OUT = Path(__file__).parent
RUN = OUT.parent/'clean_fresh_BI_pose_eval_v4_32_r2'
CASES = ('wood_night_01:031344', 'wood_night_01:031426', 'eval_noapril:1775201428535961600')


def selected(path, *, components=False):
    answer = [] if components else {}
    for row in C.rows(path):
        identifier = row.get('id', row.get('identifiers', {}).get('id'))
        if identifier not in CASES:
            continue
        if components:
            answer.append(row)
        else:
            if identifier in answer: raise ValueError('Duplicate fixed case')
            answer[identifier] = row
    if not components and set(answer) != set(CASES): raise ValueError('All fixed cases retained')
    return answer


def angle(first, second):
    a,b = np.asarray(first),np.asarray(second)
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0.,1.))))


def project(family, R, t, K):
    camera = np.asarray(family['native_xyz'])@np.asarray(R).T+np.asarray(t)
    hom = camera@np.asarray(K).T
    return hom[:,:2]/hom[:,2:3]


def main():
    if (OUT/'EVIDENCE.json').exists(): raise FileExistsError('Preserve the diagnosis attempt')
    if C.sha(RUN/'SCORE_SEAL.json') != '7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80':
        raise ValueError('Exact ROOT supplied score seal required')
    plan=C.read(OUT/'PLAN.json');seal=C.read(RUN/'SCORE_SEAL.json')
    pred=C.read(RUN/'PREDICTION_SEAL.json');inputs=C.read(RUN/'INPUT_SEAL.json');protocol=C.read(RUN/'PROTOCOL.json')
    audit=C.read(RUN/'independent_root_audit_v1/AUDIT.json');aseal=C.read(RUN/'independent_root_audit_v1/AUDIT_SEAL.json')
    bindings={str((RUN/'SCORE_SEAL.json').relative_to(ROOT)):C.sha(RUN/'SCORE_SEAL.json'),
        str((RUN/'PREDICTION_SEAL.json').relative_to(ROOT)):plan['prediction_seal_sha256']}
    for name,expected in [('INPUTS.jsonl.gz',inputs['input_sha256']),('PREDICTIONS.jsonl.gz',pred['predictions_sha256']),
        ('independent_root_audit_v1/AUDIT.json',aseal['receipt_sha256']),
        ('independent_root_audit_v1/SELECTED_POSE_COMPONENTS.jsonl.gz',aseal['components_sha256'])]:
        if C.sha(RUN/name)!=expected:raise ValueError('Sealed source changed:'+name)
        bindings[str((RUN/name).relative_to(ROOT))]=expected
    if audit['rows']!=238 or audit['failure_count'] or audit['prediction_seal']!=seal['prediction_seal']:
        raise ValueError('Closed all238 independent zero-failure audit required')
    packets=selected(RUN/'INPUTS.jsonl.gz');results=selected(RUN/'PREDICTIONS.jsonl.gz')
    components=selected(RUN/'independent_root_audit_v1/SELECTED_POSE_COMPONENTS.jsonl.gz',components=True)
    score={}
    global_summary={}
    for part in ('TEST','REMAINDER','PRIMARY'):
        path=RUN/(part+'_SCORE.json')
        if C.sha(path)!=seal['score_sha256'][part]:raise ValueError('Exact stored metric seal required')
        value=C.read(path);bindings[str(path.relative_to(ROOT))]=C.sha(path)
        global_summary[part]=value['summary']
        score.update({r['id']:r for r in value['rows'] if r['id'] in CASES})
    rows=[]
    for identifier in CASES:
        packet,result=packets[identifier],results[identifier]
        source=packet['clean_input'];families=A.geometry_families(packet);coarse=families['ORIGINAL_WD']['coarse_pose']
        initial=project(families['ORIGINAL_WD'],coarse['R'],coarse['t'],source['K'])
        rgb=[p for p,h in protocol['input_sha256'].items() if h==packet['identifiers']['image_sha256']
            and '/sessions/'+packet['identifiers']['session']+'/rgb/' in p and Path(p).suffix.lower() in ('.png','.jpg','.jpeg')]
        if len(rgb)!=1:raise ValueError('One exact fixed original RGB path required')
        image=ROOT/rgb[0]
        if C.sha(image)!=packet['identifiers']['image_sha256']:raise ValueError('Original RGB changed')
        destination=OUT/(identifier.replace(':','__')+image.suffix);shutil.copyfile(image,destination)
        bindings[rgb[0]]=C.sha(image)
        points=[]
        for m in source['visible_measurements']:
            k=m['native_corner']
            if source['initial_self_hidden'][k] or source['states'][k] not in ('B','I'):continue
            points.append(dict(native_corner=k,state=source['states'][k],source=m['source'],xy=m['xy'],
                covariance_px2=m.get('covariance_px2'),fit_covariance_px2=m.get('fit_covariance_px2'),
                fit_reliability_weight=m.get('fit_reliability_weight'),initial_model_projection_xy=initial[k].tolist(),
                initial_prediction_to_archived_q_px=float(np.linalg.norm(initial[k]-m['xy']))))
        arms={}
        for arm in A.ARMS:
            entry=result['arms'][arm]
            metrics=score[identifier]['metrics'][arm]
            if arm==A.ARMS[0]:
                arms[arm]=dict(accepted=entry['accepted'],actual_pose=entry['actual_pose'],metrics=metrics,
                    current_v32_objective_not_evaluated=True,e_is_prediction_not_truth=True)
                continue
            name=entry['WD_family'];family=families[name];solution=entry['solution'];plan=entry['new_geometry_edge_cost_plans'][name]
            q=project(family,entry['R'],entry['t'],source['K']);xyz=np.asarray(family['native_xyz'])
            groups=[]
            for group in plan['active_groups']:
                k,mate=group['native_corner'],group['native_neighbor'];n=np.asarray(group['normal_unit_xy']);t=np.array([-n[1],n[0]])
                support=A.INDEPENDENT.KERNEL.original_support_slacks(q,group)
                slacks=support['coverage_slacks']+[x for r in support['directed']for x in r['slacks']]
                v=q[mate]-q[k];normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
                reference=group['selected_chain_ref']['guide_reference']
                groups.append(dict(native_edge=[k,mate],physical_axis='XYZ'[int(np.argmax(np.abs(xyz[mate]-xyz[k])))],
                    owners=group['activation_owners'],proposal_grade=group['proposal_grade'],guide_kind=reference['guide_kind'],
                    guide_id=reference['guide_id'],component_id=group['component_id'],
                    measured_tangent_unit_xy=t.tolist(),actual_support_span_px=group['actual_inlier_span_px'],
                    actual_support_tangent_interval_px=group['actual_support_tangent_interval_px'],
                    actual_inlier_count=len(group['actual_inlier_xy']),actual_inlier_xy=group['actual_inlier_xy'],
                    actual_support_anchor_xy=group['support_anchor_xy'],normal_unit_xy=group['normal_unit_xy'],offset_px=group['offset_px'],
                    normal_variance_px2=group['normal_variance_px2'],angular_variance_sine2=group['angular_variance_rad2'],
                    fit_reliability_weight=group['fit_reliability_weight'],model_endpoint_projection_xy=q[[k,mate]].tolist(),
                    model_projected_edge_length_px=float(np.linalg.norm(v)),model_vs_measured_direction_undirected_deg=angle(v,t),
                    normal_residual_px=float(normal@(np.asarray(group['support_anchor_xy'])-q[k])),
                    support_slacks=support,minimum_support_slack_px=float(min(slacks))))
            selection=entry['candidate_selection'];candidates=selection['candidates'];totals=[r['total_energy']for r in candidates]
            costs=solution.get('native_costs',[]);bykind=Counter()
            for c in costs:bykind[c['kind']]+=c['weighted_robust_cost']
            independent=[r for r in components if r['id']==identifier and r['arm']==arm]
            if entry['accepted'] and (len(independent)!=1 or not A.near(independent[0]['total'],solution['objective'])):
                raise ValueError('Stored independent chosen scalar mismatch')
            arms[arm]=dict(accepted=entry['accepted'],pose_available=entry['pose_available'],WD_family=name,
                actual_pose=entry['actual_pose'],metrics=metrics,active_new_physical_edges=len(groups),
                active_new_owner_ids=plan['new_owner_ids'],corner_modes=[dict(native_corner=p['native_corner'],mode=p['mode'])for p in plan['corner_plans']],
                actual_active_groups=groups,native_costs=costs,native_cost_by_kind=dict(bykind),
                selected_name=selection['selected_name'],selected_endpoint_origin=entry['selected_endpoint_origin'],
                containment_weight=selection['containment_weight'],stored_feasible_candidate_count=len(candidates),
                stored_candidate_counts_by_family=dict(Counter(c['family']for c in candidates)),
                stored_feasible_total_min=min(totals)if totals else None,stored_feasible_total_max=max(totals)if totals else None,
                selected_candidate=next((c for c in candidates if c['name']==selection['selected_name']),None),
                lowest_three_stored_feasible=sorted(candidates,key=lambda c:(c['total_energy'],c['index']))[:3],
                independent_selected_components=independent[0]if independent else None,
                stored_minimum_joint_slack=solution.get('minimum_joint_slack'),readout=entry['readout'],
                new_corner_measurements_created=0,all_unstored_candidate_costs_or_seed_search_not_replayed=True)
        rows.append(dict(id=identifier,partition=result['partition'],states=source['states'],initial_self_hidden=source['initial_self_hidden'],
            source_image=dict(path=rgb[0],sha256=C.sha(image),local_copy=destination.name),
            original_coarse_native_pose=coarse,original_coarse_model_center=(np.asarray(coarse['R'])@np.asarray(source['native_xyz']).mean(axis=0)+coarse['t']).tolist(),
            original_BI_archive=points,old_partial_native_corners=[b['native_corner']for b in packet['physics']['boundary_merged']],
            authoritative_u_regions=packet['physics']['u_regions'],O_region_count=len(packet['regularizers']['o_regions']),
            S_eligible_ids=[]if packet['regularizers']['original_S_prepared']is None else [r['native_corner']for r in packet['regularizers']['original_S_prepared']['eligible_initial_S']],
            arms=arms))
    output=dict(schema='SEALED_FIXED_THREE_FAILURE_MECHANISM_EVIDENCE_V32',cases=rows,global_stored_summaries=global_summary,
        source_sha256={str(Path(__file__).relative_to(ROOT)):C.sha(__file__),str(Path(A.__file__).relative_to(ROOT)):C.sha(A.__file__)},
        input_sha256=bindings,scope=dict(new_reference_or_GT_decodes=0,new_queries_seeds_NN_TLS_PnP_optimizer_or_fits=0,
            final_GT_visualization_owned_by_ACTUAL=True,default_adoption=False,e_is_not_truth=True))
    C.write(OUT/'EVIDENCE.json',output)
    print(json.dumps({row['id']:{arm:dict(T=value['metrics']['translation_cm'],R=value['metrics']['rotation_deg'],
        accepted=value['accepted'],family=value.get('WD_family'),edges=value.get('active_new_physical_edges'),
        native=value.get('native_cost_by_kind'),selected=value.get('selected_candidate'),groups=value.get('actual_active_groups'))
        for arm,value in row['arms'].items()}for row in rows},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
