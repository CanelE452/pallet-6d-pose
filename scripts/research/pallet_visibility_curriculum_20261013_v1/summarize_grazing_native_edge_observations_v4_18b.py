"""Sealed238 observation/output audit; no image, label, NN or fitting calls."""
from collections import Counter
from pathlib import Path

from . import common as C
from . import observe_grazing_native_edge_chains_v4_18b as D


OUTPUT = D.OUTPUT.parent/'grazing_native_edge_observation_coverage_v4_18b'


def main(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every prior observation audit')
    folder = D.OUTPUT
    tasks,protocol,_ = D.OLD._load(folder)
    for partition in ('CAL','REST'):
        D.OLD._verify_observation_seal(folder,partition)
    seal = C.read(folder/'OBSERVATION_SEAL.json')
    if (C.sha(folder/'OBSERVATIONS.jsonl.gz') != seal['observations_sha256']
            or C.sha(folder/'PROTOCOL.json') != seal['protocol_sha256']
            or C.sha(folder/'SUMMARY.json') != seal['summary_sha256']
            or C.sha(folder/'PREPARATION_SEAL.json') != seal['preparation_seal_sha256']):
        raise ValueError('Complete238 observation seal changed')
    rows = list(C.rows(folder/'OBSERVATIONS.jsonl.gz'))
    old_rows = {r['id']:r for r in C.rows(D.OLD.OUTPUT/'OBSERVATIONS.jsonl.gz')}
    checks = 0
    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError(message)
    check(len(rows)==238 and [r['id'] for r in rows]==seal['ordered_ids']==[t['id'] for t in tasks],
        'All ordered238 observations, including zero-edge abstentions, required')
    evidence = Counter()
    changed_inventory = []
    chain_count_changes = []
    for task,row in zip(tasks,rows):
        old = old_rows[row['id']]
        for key in ('states','initial_geometry','visible_measurements','domains','u_regions','nonself_prediction_scale_px',
                'new_grazing_U_native_ids','original_states','observation_generation','unchanged_observation_math_sha256'):
            check(task[key]==row[key],row['id']+' changed sealed input '+key)
        check(row['initial_geometry']==old['initial_geometry'], 'Original geometry changed')
        check(row['domains']==old['domains'], 'Original O/T numeric domains changed')
        for k,(before,after) in enumerate(zip(old['states'],row['states'])):
            check(before==after or (before in ('B','I') and after=='U' and k in row['new_grazing_U_native_ids']),
                'Only fixed initial-grazing B/I->U adaptation allowed')
        expected = [r for r in old['visible_measurements'] if D.NMS.V3._id(r) not in row['new_grazing_U_native_ids']]
        check(expected==row['visible_measurements'], 'Retained complete B/I measurement/noise changed')
        obs = row['chain_observation']
        check(D.OLD._digest(obs['configuration'])==D.OLD._digest(D.NMS.DEFAULT_CONFIGURATION),
            'Frozen17c configuration changed after JSON tuple/list normalization')
        for key in ('S_local_queries','S_coordinate_reads','U_coordinate_reads','U_corner_queries',
                'O_T_coordinate_reads','old_corner_observations_changed','full_2D_corner_measurements_created','pose_calls'):
            check(obs[key]==0,row['id']+' producer contract violation '+key)
        check(obs['GT_inputs'] is False and obs['states_changed'] is False,'GT or state mutation in NMS producer')
        causes = Counter()
        for ray in obs['edge_records']:
            k,mate = ray['native_corner'],ray['native_neighbor']
            check(row['states'][k] in ('B','I') and not row['initial_geometry']['initial_self_hidden'][k],
                'Excluded state used as image anchor')
            if ray.get('pixel_queried'):
                check(row['states'][mate]!='S' and not row['initial_geometry']['initial_self_hidden'][mate],
                    'Permanent S endpoint participated in a local query')
                if row['states'][mate] in ('U','O','T'):
                    check(ray['neighbor_image_coordinate_used'] is False and ray['neighbor_unknown_corner_queries']==0,
                        'Unknown mate was used as an image point')
                    evidence['queried_edges_with_model_only_unknown_mate'] += 1
                    if mate in row['new_grazing_U_native_ids']:
                        evidence['queried_edges_with_new_grazing_U_model_only_mate'] += 1
            if not ray.get('supported'):
                causes[ray.get('reason','unspecified')] += 1
            for chain in ray.get('chains',()):
                check(chain['measurement_source']=='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',
                    'Actual-NMS chain source changed')
        for cause,count in causes.items():
            evidence['abstain:'+cause] += count
        if row['new_grazing_U_native_ids']:
            changed_inventory.append(dict(id=row['id'],partition=row['partition'],native_ids=row['new_grazing_U_native_ids']))
        old_count=old['summary']['actual_chain_hypotheses'];new_count=row['summary']['actual_chain_hypotheses']
        if old_count!=new_count:
            chain_count_changes.append(dict(id=row['id'],partition=row['partition'],old_alternatives=old_count,new_alternatives=new_count))
    check(sum(len(r['new_grazing_U_native_ids']) for r in rows)==31,'Predeclared31 new-U population differs')
    summaries={partition:D._summary([r for r in rows if r['partition']==partition])
        for partition in ('CAL','TEST','REMAINDER')}
    sources = {str(Path(p).resolve().relative_to(C.ROOT)):C.sha(Path(p))
        for p in (__file__,D.__file__,D.OLD.__file__,D.NMS.__file__,D.OLD.TOPOLOGY.__file__)}
    inputs = {str((folder/name).relative_to(C.ROOT)):C.sha(folder/name)
        for name in ('PROTOCOL.json','PREPARATION_SEAL.json','OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz','SUMMARY.json',
            'CAL_OBSERVATION_SEAL.json','REST_OBSERVATION_SEAL.json')}
    inputs[str((D.OLD.OUTPUT/'OBSERVATION_SEAL.json').relative_to(C.ROOT))]=C.sha(D.OLD.OUTPUT/'OBSERVATION_SEAL.json')
    output.mkdir(parents=True)
    report=dict(schema='GRAZING_U_FRESH_NMS_COVERAGE_AND_CONTRACT_AUDIT_V4_18B',passed=True,
        checks=checks,failures=0,source_sha256=sources,input_sha256=inputs,
        all238_summary=D._summary(rows),partition_summaries=summaries,
        new_grazing_U_inventory=changed_inventory,query_and_abstention_counts=dict(evidence),
        actual_chain_count_changes_from_original_fresh_observer=chain_count_changes,
        old_parent_boundary_digest_is_original_archive=True,current_filtered_profile_digest_separate=True,
        source_review_and_freeze_receipt=str((D.OUTPUT.parent/'grazing_native_edge_observation_implementation_v4_18b/SOURCE_FREEZE_V1.json').relative_to(C.ROOT)),
        all_actual_queries_executed_again_after_new_gate=True,old_NMS_result_reuse=False,
        existing_CAL_noise_model_validated_for_new_generation=False,new_CAL_noise_refit_required=True,
        mathematical_DAG_or_threshold_or_mask_or_geometry_changes=0,
        new_GT_decodes=0,new_RGB_or_NN_calls=0,new_noise_or_pose_fits=0,
        full_KP_adoptions=0,default_pipeline_changed=False)
    C.write(output/'COVERAGE_AND_CONTRACT_V1.json',report)
    C.write(output/'REVIEW_SEAL_V1.json',dict(review_sha256=C.sha(output/'COVERAGE_AND_CONTRACT_V1.json'),
        source_sha256=sources,input_sha256=inputs))
    print(dict(output=str(output),passed=True,checks=checks,chain_count_changes=chain_count_changes))


if __name__=='__main__':
    main()
