"""Synthetic V33 transition/stream/metadata and one toy numeric replay only."""
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from . import audit_largest_component_global_outline_poses_v4_33 as A


def forbidden(*args, **kwargs):
    raise AssertionError('No actual rows/masks/models/GT or new numerical path')


def test_root_GO_and_explicit_current_pins_precede_IO(monkeypatch):
    monkeypatch.setattr(A.C,'read',forbidden);monkeypatch.setattr(A.C,'sha',forbidden)
    monkeypatch.setattr(Path,'exists',forbidden)
    with pytest.raises(PermissionError):A.main()
    with pytest.raises(ValueError,match='before IO'):A.main(root_independent_audit_GO=True)


def test_numeric_replay_is_exact_old_independent_function_without_controller_import():
    assert A.audit_row is A.A32.audit_row and A.digest is A.A32.digest
    assert A.C.sha(A.A32.__file__)==A.A32_SHA256
    assert A.C.sha(A.CONTROLLER)==A.CONTROLLER_SHA256
    source=inspect.getsource(A)
    assert 'solve_worker_packet(' not in source and 'replace_outline_observation(' not in source
    assert 'generate_fresh_observed_BI_pnp_proposals(' not in source
    assert not any(name in A.__dict__ for name in ('M','G','AUTH','SOLVER','FACTORY','CLEAN'))


def pair():
    from .test_evaluate_largest_component_global_outline_poses_v4_33 import toy,V
    original,outline=toy()
    changed=V.replace_outline_observation(original,outline)
    record=dict(id=outline['id'],partition=outline['partition'],ready=True,
        parent_positive_packet_sha256=A.digest(original),positive_packet_sha256=A.digest(changed),
        outline_row_sha256=A.digest(outline),changed_field='regularizers.outline_observation')
    return original,changed,outline,record


def test_independent_C_only_transition_without_controller_replacement_call(monkeypatch):
    original,changed,outline,record=pair();before=deepcopy((original,changed,outline,record))
    from . import evaluate_largest_component_global_outline_poses_v4_33 as V
    monkeypatch.setattr(V,'replace_outline_observation',forbidden)
    checks=A._check_transition(original,changed,outline,record)
    assert all(item['passed'] for item in checks),checks
    assert before==(original,changed,outline,record)


@pytest.mark.parametrize('corruption',['q','source','Cconfiguration','union','record','newmask'])
def test_transition_corruptions_detected(corruption):
    original,changed,outline,record=deepcopy(pair())
    if corruption=='q':changed['clean_input']['visible_measurements'][0]['xy'][0]+=1.
    elif corruption=='source':changed['source_contract']['old_model_substituted']=True
    elif corruption=='Cconfiguration':changed['regularizers']['outline_configuration']['normal_sigma_px']+=1.
    elif corruption=='union':outline['original_union']['polygon_xy'][0][0]+=1.
    elif corruption=='record':record['parent_positive_packet_sha256']='0'*64
    else:changed['regularizers']['outline_observation']['image_binding']['mask_file_sha256']='0'*64
    assert any(not item['passed'] for item in A._check_transition(original,changed,outline,record))


def _streams():
    ids=[f'SYNTHETIC:木{i}' for i in range(238)]
    originals=[];outlines=[];packets=[];comparisons=[];predictions=[];parents=[];records=[];indices=[];outline_entries=[]
    for i,identifier in enumerate(ids):
        part='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER'
        packet={key:None for key in A.A32.REPLAY.WORKER_KEYS}
        packet.update(schema=A.A32.REPLAY.WORKER_SCHEMA,identifiers=dict(id=identifier,partition=part),
            source_contract=dict(ready=False))
        outline=dict(id=identifier,partition=part,selection_profile=A.PROFILE,baseline_profile=A.BASELINE_PROFILE,
            positive_packet_sha256=A.digest(packet),union_exact_match=True)
        worker=dict(id=identifier,partition=part,arms={name:dict(accepted=False,pose_available=False) for name in A.ARMS[1:]})
        row={**worker,'arms':{'SAME_FROZEN_E':dict(accepted=False),**worker['arms']},
            'e_comparison_parent_only_no_numeric_exposure':True,'legacy_e_pose_semantics':dict(comparison_only=True)}
        record=dict(id=identifier,partition=part,ready=False,positive_packet_sha256=A.digest(packet),
            parent_positive_packet_sha256=A.digest(packet),outline_row_sha256=A.digest(outline),changed_field=None)
        originals.append(deepcopy(packet));packets.append(packet);outlines.append(outline);records.append(record)
        predictions.append(row);parents.append(deepcopy(row))
        comparisons.append(dict(id=identifier,partition=part,prediction=row['arms']['SAME_FROZEN_E']))
        indices.append(dict(id=identifier,partition=part,arm_count=4,canonical_row_sha256=A.digest(worker)))
        outline_entries.append(dict(id=identifier,partition=part,canonical_row_sha256=A.digest(outline)))
    return packets,comparisons,predictions,records,indices,ids,originals,outlines,parents,outline_entries


def test_all238_ordered_zero_no_pose_stream_and_entire_native_e_parity():
    values=_streams();results=list(A.checked_rows(*values))
    assert len(results)==238 and all(all(item['passed'] for item in row[3]) for row in results)
    wrong=deepcopy(values);wrong[8][1]['arms'][A.NATIVE_ARMS[0]]['accepted']=True
    results=list(A.checked_rows(*wrong))
    assert any(not item['passed'] for item in results[1][3])
    wrong=deepcopy(values);wrong[7].pop()
    with pytest.raises(ValueError,match='All238'):list(A.checked_rows(*wrong))
    wrong=deepcopy(values);wrong[9][96]['partition']='CAL'
    with pytest.raises(ValueError,match='ordered238'):list(A.checked_rows(*wrong))


def metadata(monkeypatch):
    ids=[f'SYNTHETIC:{i}' for i in range(238)];values={};hashes={}
    def add(path,value):values[path]=value;hashes.setdefault(path,'a'*64)
    output=A.OUTPUT
    maps=dict(source_sha256={},input_sha256={},protected_user_sha256={str(i):'b'*64 for i in range(6)},
        original_CAL_file_sha256={str(i):'c'*64 for i in range(53)},
        immutable_LCC_CAL_file_sha256={str(i):'d'*64 for i in range(53)})
    protocol=dict(maps,ordered_ids=ids,arms=list(A.ARMS),schema='SEALED_OUTLINE_ONLY_POSITIVE_PNP_INPUTS_V33',
        partitions=A.PARTITIONS,population_inventory=A.POPULATION_INVENTORY,central_plan_sha256=A.CENTRAL_SHA256,
        native_model_seal_sha256=A.MODEL_SHA256,outline_seal_sha256='f'*64,outline_selection_profile=A.PROFILE,
        native_observation_family=A.A32.REPLAY.FAMILY,native_selection_profile=A.A32.REPLAY.PROFILE,
        native_noise_population=A.A32.REPLAY.POPULATION,native_CAL32_model_relabelled=False,
        only_positive_packet_field_replaced='regularizers.outline_observation',worker_keys=sorted(A.A32.REPLAY.WORKER_KEYS),
        e_comparison_parent_only=True,e_comparison_sha256='a'*64,execution_plan=dict(path='toy_plan.json',sha256='a'*64),
        records=[dict(id=identifier) for identifier in ids])
    add(output/'PROTOCOL.json',protocol)
    add(A.CENTRAL,dict(ordered_ids=ids,partitions=A.POPULATION_INVENTORY,selection_profile=A.PROFILE,arms=list(A.ARMS)))
    hashes[A.CENTRAL]=A.CENTRAL_SHA256;hashes[A.MODEL/'MODEL_SEAL.json']=A.MODEL_SHA256
    hashes[A.CONTROLLER]=A.CONTROLLER_SHA256;hashes[Path(A.A32.__file__)]=A.A32_SHA256
    hashes[A.A32.HELPER]=A.A32.HELPER_SHA256
    add(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=ids,arms=list(A.ARMS),arm_count=5,
        outline_selection_profile=A.PROFILE,protocol_sha256='a'*64,input_sha256='a'*64,e_comparison_sha256='a'*64))
    add(output/'PREDICTION_SEAL.json',dict(rows=238,ordered_ids=ids,arms=list(A.ARMS),arm_count=5,
        outline_selection_profile=A.PROFILE,all_declared_arms_complete=True,before_fixed_reference_scoring=True,
        no_population_drop=True,native_CAL32_model_relabelled=False,**{key:'a'*64 for key in (
            'protocol_sha256','input_seal_sha256','predictions_sha256','separate_e_comparison_sha256',
            'execution_protocol_sha256','row_index_sha256','native_and_e_parity_sha256')}))
    hashes[output/'PREDICTION_SEAL.json']='e'*64
    add(output/'NATIVE_AND_E_PARITY_V1.json',dict(rows=238,native_arm_payload_equalities=476,
        e_payload_equalities=238,failure_count=0,numeric_tolerance=0.))
    add(output/'PREDICTION_EXECUTION_PROTOCOL.json',dict(controller_source_sha256=A.CONTROLLER_SHA256,
        worker_source_sha256=A.WORKER_SHA256,solver_source_sha256=A.SOLVER_SHA256,outline_selection_profile=A.PROFILE,
        positive_worker_keys=sorted(A.A32.REPLAY.WORKER_KEYS),workers=3,maximum_pending=3,e_numeric_worker_inputs=0,reference_or_score_inputs=0))
    entries=[dict(id=identifier,partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER')for i,identifier in enumerate(ids)]
    add(A.OUTLINES/'OUTLINE_ROW_INDEX.json',dict(entries=entries))
    add(A.OUTLINES/'FINAL_PROTOCOL.json',dict(central_plan_sha256=A.CENTRAL_SHA256,ordered_ids=ids,selection_profile=A.PROFILE))
    add(A.OUTLINES/'OUTLINE_OBSERVATION_SEAL.json',dict(rows=238,ordered_ids=ids,partitions=A.PARTITIONS,
        selection_profile=A.PROFILE,canonical_validation=dict(rows=238,canonical_row_digests_verified=238,missing_duplicate_or_extra_rows=0),
        protocol_sha256='a'*64,observations_sha256='a'*64,row_index_sha256='a'*64,partition_seals={'CAL':'a'*64,'REST':'a'*64}))
    hashes[A.OUTLINES/'OUTLINE_OBSERVATION_SEAL.json']='f'*64
    for part,count,selected,subset in (('CAL',46,ids[:46],entries[:46]),('REST',192,ids[46:],entries[46:])):
        add(A.OUTLINES/(part+'_OBSERVATION_SEAL.json'),dict(rows=count,ordered_ids=selected,
            selection_profile=A.PROFILE,union_exact_match_rows=count,row_index_sha256='a'*64,observations_sha256='a'*64))
        add(A.OUTLINES/(part+'_ROW_INDEX.json'),dict(entries=subset))
    add(A.PARENT/'PREDICTION_SEAL.json',dict(rows=238,ordered_ids=ids,arms=list(A.ARMS),predictions_sha256='a'*64))
    hashes[A.PARENT/'PREDICTION_SEAL.json']=A.PARENT_PREDICTION_SHA256
    add(A.PARENT/'INPUT_SEAL.json',dict(input_sha256='a'*64))
    add(output/'PREDICTION_ROW_INDEX.json',dict(entries=[dict(id=identifier)for identifier in ids]))
    monkeypatch.setattr(A.C,'sha',lambda path:hashes.get(Path(path),'a'*64))
    monkeypatch.setattr(A.C,'read',lambda path:deepcopy(values[Path(path)]))
    monkeypatch.setattr(A.A32,'_verify',lambda *maps:{})
    return output,values


def test_new33_metadata_seals_and_unchanged32_noise_model_contract(monkeypatch):
    output,values=metadata(monkeypatch)
    result=A._metadata(output,'e'*64,A.CENTRAL_SHA256,A.MODEL_SHA256,'f'*64)
    assert len(result['ids'])==len(result['outline_entries'])==238
    values[output/'PROTOCOL.json']['native_CAL32_model_relabelled']=True
    with pytest.raises(ValueError,match='native32'):A._metadata(output,'e'*64,A.CENTRAL_SHA256,A.MODEL_SHA256,'f'*64)


@pytest.mark.parametrize('corruption',['parity','CAL','worker','inventory'])
def test_metadata_rejects_missing_parity_partition_wrong_worker_or_population(monkeypatch,corruption):
    output,values=metadata(monkeypatch)
    if corruption=='parity':values[output/'NATIVE_AND_E_PARITY_V1.json']['failure_count']=1
    elif corruption=='CAL':values[A.OUTLINES/'CAL_OBSERVATION_SEAL.json']['union_exact_match_rows']=45
    elif corruption=='worker':values[output/'PREDICTION_EXECUTION_PROTOCOL.json']['e_numeric_worker_inputs']=1
    else:values[output/'PROTOCOL.json']['population_inventory']=A.PARTITIONS
    with pytest.raises(ValueError):A._metadata(output,'e'*64,A.CENTRAL_SHA256,A.MODEL_SHA256,'f'*64)


def test_genuine_toy_LCC_selected_arithmetic_uses_old_independent_native_OSC(monkeypatch):
    from .test_evaluate_largest_component_global_outline_poses_v4_33 import V
    original,changed,outline,record=pair()
    result=V.solve_worker_packet(changed);identifiers=changed['identifiers']
    comparison=dict(id=identifiers['id'],partition=identifiers['partition'],prediction=dict(accepted=True),
        legacy_e_pose_semantics=dict(numeric_worker_exposure=False,comparison_only=True))
    row=dict(result,id=identifiers['id'],partition=identifiers['partition'],image_sha256=identifiers['image_sha256'],
        legacy_e_pose_semantics=comparison['legacy_e_pose_semantics'],e_comparison_parent_only_no_numeric_exposure=True)
    row['arms']={'SAME_FROZEN_E':deepcopy(comparison['prediction']),**row['arms']}
    monkeypatch.setattr(V.M,'build_clean_native_cores',forbidden)
    monkeypatch.setattr(V.M.G.GeometryGuidedEdgeProblem,'objective',forbidden)
    monkeypatch.setattr(V.M.G.GeometryGuidedEdgeProblem,'evaluate',forbidden)
    audit=A.audit_row(changed,comparison,row)
    assert all(item['passed']for item in audit['checks']),[item for item in audit['checks']if not item['passed']]
    assert audit['accepted']+audit['fallback']==4
