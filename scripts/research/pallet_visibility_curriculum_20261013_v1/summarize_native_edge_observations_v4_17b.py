"""Read-only full238 observer contracts, coverage and abstention counts."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil

import numpy as np

from . import observe_native_edge_chains_v4_17b as D


OUTPUT=D.BOUNDARY/'native_edge_observation_coverage_v4_17b'


def main(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all earlier coverage/audits')
    tasks,protocol,prep=D._load(D.OUTPUT)
    full=D.C.read(D.OUTPUT/'OBSERVATION_SEAL.json')
    for path,key in ((D.OUTPUT/'OBSERVATIONS.jsonl.gz','observations_sha256'),
                     (D.OUTPUT/'SUMMARY.json','summary_sha256'),
                     (D.OUTPUT/'PREPARATION_SEAL.json','preparation_seal_sha256'),
                     (D.OUTPUT/'PROTOCOL.json','protocol_sha256')):
        if D.C.sha(path)!=full[key]:raise ValueError('Full observation seal changed')
    for part in ('CAL','REST'):
        D._verify_observation_seal(D.OUTPUT,part)
        if D.C.sha(D.OUTPUT/(part+'_OBSERVATION_SEAL.json'))!=full['partition_seals'][part]:raise ValueError('Partition seal changed')
    rows=list(D.C.rows(D.OUTPUT/'OBSERVATIONS.jsonl.gz'))
    if len(rows)!=238 or [r['id'] for r in rows]!=[t['id'] for t in tasks]:raise ValueError('Complete ordered238 required')
    checks=0;counts=defaultdict(Counter);edge_reasons=defaultdict(Counter);line_reasons=defaultdict(Counter)
    overflows=[];zero=[];joint_inventory=[]
    for task,row in zip(tasks,rows):
        part=row['partition'];initial=task['initial_geometry']['initial_self_hidden'];states=task['states']
        assert row['visible_measurements']==task['visible_measurements'];checks+=1
        for key in ('states','initial_geometry','domains','u_regions','nonself_prediction_scale_px'):
            assert row[key]==task[key];checks+=1
        assert row['unchanged_observation_math_sha256']==task['unchanged_observation_math_sha256'];checks+=1
        chain=row['chain_observation'];proposal=row['junction_proposals'];cfg=chain['configuration']
        assert row['contract']['precise_KP_adoptions']==row['contract']['pose_calls']==row['contract']['NN_calls']==0;checks+=1
        assert proposal['precise_KP_adoptions']==proposal['old_q_covariance_state_changes']==0;checks+=1
        counts[part]['frames']+=1
        if row['summary']['zero_supported_edges']:
            counts[part]['zero_edge_frames']+=1;zero.append(dict(id=row['id'],partition=part,states=states))
        else:counts[part]['supported_frames']+=1
        frame_overflow=[]
        for ray in chain['edge_records']:
            edge_reasons[part][ray['reason']]+=1
            if ray.get('pixel_queried'):
                k,mate=ray['native_corner'],ray['native_neighbor']
                assert not initial[k] and states[k] in ('B','I') and not initial[mate] and states[mate]!='S';checks+=1
                assert ray['neighbor_unknown_corner_queries']==ray['S_local_queries']==0;checks+=1
            assert 'selected_chain_index' not in ray;checks+=1
            if ray.get('supported'):
                counts[part]['directed_edges']+=1
                for actual in ray['chains']:
                    good=np.asarray(actual['inlier_mask'],bool)
                    assert good.sum()==actual['inlier_count'] and good.sum()>=cfg['minimum_inlier_count'];checks+=1
                    for key,minimum in (('RGB_gradient_energy',cfg['minimum_gradient']),
                        ('RGB_alignment',cfg['minimum_gradient_alignment']),('RGB_contrast',cfg['minimum_color_transition'])):
                        assert np.asarray(actual[key])[good].min()>=minimum;checks+=1
                    assert actual['conditional_uncertainty']['normal_sigma_px']>=1 and actual['conditional_uncertainty']['angle_sigma_rad']>0;checks+=1
                    assert actual['invented_bridge_points']==0;checks+=1
                    counts[part]['actual_chain_hypotheses']+=1
            audit=ray.get('component_linking_audit',{})
            if audit.get('combinatorial_overflow_abstained'):
                assert not ray['supported'] and not ray['chains'];checks+=1
                counts[part]['overflow_directed_edges']+=1
                frame_overflow.append(dict(native_corner=ray['native_corner'],native_neighbor=ray['native_neighbor'],
                    actual_components=audit['actual_components'],admissible_links=len(audit['links']),
                    raw_NMS_peak_count=ray.get('image_NMS_peak_count'),RGB_supported_peak_count=ray.get('RGB_context_peak_count')))
            for rejected in ray.get('rejected_chains',[]):line_reasons[part][rejected['reason']]+=1
        counts[part]['unique_native_edges']+=len({tuple(sorted(r['native_edge'])) for r in chain['edge_records'] if r.get('supported')})
        if frame_overflow:overflows.append(dict(id=row['id'],partition=part,edges=frame_overflow))
        for record in proposal['records']:
            assert not record['full_KP_adopted'];checks+=1
            if record['candidates']:
                assert record['state']=='B' and not initial[record['native_corner']];checks+=1
                counts[part]['B_junction_candidates']+=len(record['candidates'])
                joint_inventory.append(dict(id=row['id'],partition=part,native_corner=record['native_corner'],
                    candidates=len(record['candidates']),disagreement_px=record['candidate_disagreement_px'],
                    geometric_unique_3px=record['geometrically_unique_within_existing_3px_rule']))
        if any(r['candidates'] for r in proposal['records']):counts[part]['frames_with_B_junction_candidate']+=1
    output.mkdir(parents=True)
    source=D._binding(Path(__file__));shutil.copyfile(__file__,output/Path(__file__).name)
    report=dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_COVERAGE_V4_17B',UTC=datetime.now(timezone.utc).isoformat(),
        passed=True,checks=checks,failures=0,source=source,
        input_bindings=[D._binding(D.OUTPUT/name) for name in ('PROTOCOL.json','PREPARATION_SEAL.json','OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz')],
        partition_coverage={k:dict(v) for k,v in counts.items()},
        top_edge_outcomes={k:v.most_common() for k,v in edge_reasons.items()},
        rejected_chain_hypothesis_outcomes={k:v.most_common() for k,v in line_reasons.items()},
        hypothesis_rejection_counts_are_not_independent_edges=True,
        zero_edge_frames=zero,overflow_frame_inventory=overflows,B_junction_inventory=joint_inventory,
        overflow_scope='Whole edge abstains explicitly on >64 path hypotheses. No partial path set is secretly chosen. Raw peak arrays are not archived on these abstaining edges; a future compact DAG investigation is separate.',
        alternative_preservation_scope='Every returned non-overflow chain and topology candidate is preserved. Producer reference chain index is display-only and is not a selected measurement.',
        new_native_identity_or_coverage_certification=False,
        full_point_conditional_bootstrap_and_total_error_calibration_run=False,
        old_q_covariance_states_domains_U_regions_unchanged=True,
        original49_default_or_pose_predictions_changed=False,GT_numeric_decodes=0,NN_calls=0,pose_calls=0,
        future_DAG_investigation_not_run_here=True)
    D.C.write(output/'COVERAGE_V1.json',report)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):D._verify_map(protocol[key])
    D.C.write(output/'COVERAGE_SEAL_V1.json',dict(report=D._binding(output/'COVERAGE_V1.json'),source=source,
        observation_seal=D._binding(D.OUTPUT/'OBSERVATION_SEAL.json'),all_current_source_input_protected_bindings_pass=True))
    print(json.dumps(dict(output=str(output),checks=checks,partition_coverage=report['partition_coverage']),ensure_ascii=False))


if __name__=='__main__':main()
