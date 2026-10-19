"""Constructed metadata only: no producer, reference or pose calculation."""
from copy import deepcopy
import json

import pytest

from scripts.research.pallet_bi_full_execution_v1 import post_core_diagnostics as D


def coverage(fid='frame',partition='TEST'):
    return dict(arm='S2_C0',id=fid,partition=partition,session='session',cluster_id='group',
        generated_good=True,unique_good=True,eligible_good=True,generated_candidates=9,unique_candidates=5,eligible_candidates=3,
        selected={seed:dict(metric=dict(success=seed==D.SEEDS[0]),failure_stage='SUCCESS'if seed==D.SEEDS[0]else'SELECTION_LOSS',
            transition='BOTH_SUCCESS'if seed==D.SEEDS[0]else'DAMAGE',selected_candidate_id='stored',fallback=False,prediction_status='saved')for seed in D.SEEDS})


def saved_result(fid='frame'):
    states=['B','I','B','U','S','O','T','I']
    visible_rank=dict(rank=4,parameter_count=6,singular_values=[8.,7.,6.,5.],condition=8/5,rank_proves_global_uniqueness=False)
    combined_rank=dict(rank=5,parameter_count=6,singular_values=[8.,7.,6.,5.,4.],condition=2.,rank_proves_global_uniqueness=False)
    accepted=dict(accepted=True,status='joint_feasible_ambiguous',fallback_reason=None,rank=dict(visible=visible_rank,BI_and_partial_boundary=combined_rank),
        optimizer_calls=0,outline_stage_exact_reused_without_outline=True,ordered_actual_initializations=[{},{}],
        ordered_actual_joint_initializations=[{},{},{}],search=dict(joint_initializations_per_branch=3,T_union_branches_searched=1),
        minimum_joint_slack=.1,all_visited_endpoints=[],OT_constraints_satisfied=True,U_constraints_satisfied=True)
    rejected=dict(accepted=False,status='no_admissible_P1_WD0_start',fallback_reason='no_saved_fullsearch_feasible_start',optimizer_calls=0,
        ordered_actual_initializations=[],all_visited_endpoints=[],fallback_pose=dict(stored=True))
    return dict(id=fid,packet=dict(solver_input=dict(states=states,
        visible=[dict(native_corner=k,state=states[k])for k in (0,1,2)],
        boundary_observations=[dict(native_corner=1,state='I',lines=[{}])],config=dict(max_joint_iterations=250))),
        selected_family='ORIGINAL_WD',stage_results=dict(G10=dict(selected_family='ORIGINAL_WD',families=[
            dict(name='ORIGINAL_WD',accepted=True,solution=accepted),dict(name='SWAPPED_WD',accepted=False,solution=rejected)])),
        operations=dict(pose_optimizer_calls=17),proof=dict(G13_exact_reuses_G10=True))


def test_revoked_2D_is_excluded_and_missing_rank_inherited_starts_stay_honest():
    row=D.extract(saved_result(),coverage())
    assert row['factors']['full2D']==2
    assert row['factors']['normal1D']==1
    assert row['factors']['observed_scalar_dimension']==5
    assert row['factors']['revoked_full_archive_native_ids']==[1]
    assert row['factors']['B_I_anchor_dimension_counts']=={'0':1,'1':1,'2':2}
    assert row['selected_G10']['ordered_initialization_n']==2
    assert row['selected_G10']['stored_optimizer_calls']==0
    assert row['selected_G10']['outline_stage_exact_reused_without_outline']
    missing=row['G10_families']['SWAPPED_WD']['rank']['BI_and_partial_boundary']
    assert missing['rank']is None and missing['singular_values']is None
    assert missing['category']=='NOT_RECORDED'and not missing['global_uniqueness_inferred']
    assert row['stored_full_pipeline_pose_optimizer_calls']==17


def test_losses_are_already_scored_and_all_three_seeds_required():
    absent=coverage();absent.update(generated_good=False,unique_good=False,eligible_good=False)
    for selected in absent['selected'].values():selected.update(metric=dict(success=False),failure_stage='GENERATED_ABSENT')
    admission=coverage();admission['eligible_good']=False
    for selected in admission['selected'].values():selected.update(metric=dict(success=False),failure_stage='ADMISSION_LOSS')
    group=[D.extract(saved_result('absent'),dict(absent,id='absent')),D.extract(saved_result('admit'),dict(admission,id='admit')),
        D.extract(saved_result('rank'),coverage('rank'))]
    summary=D.summarize(group)
    assert summary['n']==3
    assert summary['seeds'][D.SEEDS[1]]['failure_stages']=={'GENERATED_ABSENT':1,'ADMISSION_LOSS':1,'SELECTION_LOSS':1}
    bad=coverage();bad['selected'].pop(D.SEEDS[2])
    with pytest.raises(ValueError,match='three'):D.extract(saved_result(),bad)


def constructed_tree(root):
    core=root/'core';bank=root/'postbank';prediction=root/'prediction';frame_hashes={};bindingsource={'frozen.py':'source'}
    ledger=[]
    for i in range(238):
        partition='CAL'if i<46 else'TEST'if i<95 else'REMAINDER';fid=f'frame{i:04d}';folder=core/f'{i:04d}'
        row=coverage(fid,partition);ledger.append(row)
        D.write(folder/'RESULT.json.gz',saved_result(fid))
        D.write(folder/'FEATURE.json.gz',{k:row[k]for k in ('id','partition','session','cluster_id')})
        # The stored seal's original arm may differ after exact reuse. The
        # requested bank is authenticated through its own ancestor seal.
        D.write(folder/'FRAME_SEAL.json',dict(complete=True,id=fid,arm='ORIGINAL_REUSE_ORIGIN',domain='real',ordinal=i,
            source_binding=bindingsource,artifacts={n:D.sha(folder/n)for n in ('RESULT.json.gz','FEATURE.json.gz')}))
        frame_hashes[f'{i:04d}']=D.sha(folder/'FRAME_SEAL.json')
    features=root/'FEATURES.jsonl.gz';D.write(features,[dict(id=r['id'])for r in ledger],jsonl=True)
    feature_seal=root/'FEATURE_SEAL.json';D.write(feature_seal,dict(complete=True,rows=238,source_binding=bindingsource,
        features_sha256=D.sha(features),head_features_sha256=D.sha(features),frame_seal_sha256=frame_hashes))
    D.write(prediction/'PREDICTIONS.jsonl.gz',[dict(id=r['id'])for r in ledger],jsonl=True)
    D.write(prediction/'PREDICTION_SEAL.json',dict(complete=True))
    manifest=root/'MANIFEST.json';D.write(manifest,dict(arms=dict(S2_C0=dict(backend='fullnormal',features=str(features),
        feature_seal=str(feature_seal),prediction_dir=str(prediction),result_root=str(core)))))
    D.write(bank/'INPUT_BINDINGS.json',dict(S2_C0=dict(features=dict(path=str(features),sha256=D.sha(features)),
        feature_seal=dict(path=str(feature_seal),sha256=D.sha(feature_seal)),generated_feature_seal=dict(path=str(feature_seal),sha256=D.sha(feature_seal)),
        generated_frame_seals=frame_hashes,prediction=dict(path=str(prediction/'PREDICTIONS.jsonl.gz'),sha256=D.sha(prediction/'PREDICTIONS.jsonl.gz'),
            seal_sha256=D.sha(prediction/'PREDICTION_SEAL.json')),extras=[])))
    # Reversed coverage proves an ID join rather than a positional inner zip.
    D.write(bank/'BANK_COVERAGE_ROWS.jsonl.gz',list(reversed(ledger)),jsonl=True)
    D.write(bank/'SUMMARY.json',dict(complete=True))
    D.write(bank/'ANALYSIS_SEAL.json',dict(complete=True,manifest_sha256=D.sha(manifest),
        artifacts={n:D.sha(bank/n)for n in ('INPUT_BINDINGS.json','BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json')}))
    return manifest,bank,core


def test_full238_authenticated_id_join_and_population_groups_without_reference_reader(tmp_path):
    manifest,bank,core=constructed_tree(tmp_path)
    seal=D.run(manifest,bank,tmp_path/'diagnostics')
    assert seal['reference_files_read']==seal['new_pose_optimizer_calls']==seal['pose_metrics_calculated']==0
    assert seal['populations']==D.EXPECTED
    rows=list(D.rows(tmp_path/'diagnostics/CORE_DIAGNOSTIC_ROWS.jsonl.gz'))
    assert len(rows)==238 and rows[0]['id']=='frame0000'and rows[-1]['id']=='frame0237'
    summaries=D.read(tmp_path/'diagnostics/SUMMARY_BY_POPULATION_AND_STRATUM.json')['S2_C0']
    assert {key:value['summary']['n']for key,value in summaries.items()}==D.EXPECTED
    assert summaries['PRIMARY192']['strata']['selected_local_rank/BI_and_partial_boundary']['LOCAL_5OF6']['n']==192
    assert D.read(core/'0000/FRAME_SEAL.json')['arm']=='ORIGINAL_REUSE_ORIGIN'


def test_changed_result_or_postbank_bytes_are_rejected(tmp_path):
    manifest,bank,core=constructed_tree(tmp_path)
    (core/'0000/RESULT.json.gz').write_bytes(b'changed archive')
    with pytest.raises(ValueError,match='artifact changed'):D.run(manifest,bank,tmp_path/'rejected')
    with (bank/'SUMMARY.json').open('a')as stream:stream.write(' ')
    with pytest.raises(ValueError,match='post-bank artifact'):D.authenticate_postbank(manifest,bank)


def test_union_uses_explicit_core_ancestry_and_rejects_changed_union_row(tmp_path):
    manifest,bank,core=constructed_tree(tmp_path)
    plan=D.read(manifest);binding=D.read(bank/'INPUT_BINDINGS.json');spec=plan['arms']['S2_C0'];entries=[]
    for i in range(238):
        fid=f'frame{i:04d}';path=tmp_path/'union_rows'/f'{i:04d}.json'
        D.write(path,dict(dependencies=dict(core_frame_seal_sha256=D.sha(core/f'{i:04d}'/'FRAME_SEAL.json'),
            result_sha256=D.sha(core/f'{i:04d}'/'RESULT.json.gz'))))
        entries.append(dict(id=fid,seal_path=str(path),sha256=D.sha(path)))
    union_features=tmp_path/'UNION_FEATURES.jsonl.gz';D.write(union_features,[dict(id=e['id'])for e in entries],jsonl=True)
    union_seal=tmp_path/'UNION_FEATURE_SEAL.json';D.write(union_seal,dict(complete=True,rows=238,
        schema='OWN_FULL_NORMAL_KERNEL_EXTRA_FEATURE_SEAL_V1',head_features_sha256=D.sha(union_features),frame_seals=entries))
    spec['generated_feature_seal']=spec['feature_seal'];spec.update(features=str(union_features),feature_seal=str(union_seal))
    binding['S2_C0'].update(features=dict(path=str(union_features),sha256=D.sha(union_features)),
        feature_seal=dict(path=str(union_seal),sha256=D.sha(union_seal)))
    union_manifest=tmp_path/'UNION_MANIFEST.json';D.write(union_manifest,plan);union_bank=tmp_path/'union_postbank'
    import shutil
    union_bank.mkdir()
    for name in ('BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json'):shutil.copy2(bank/name,union_bank/name)
    D.write(union_bank/'INPUT_BINDINGS.json',binding)
    D.write(union_bank/'ANALYSIS_SEAL.json',dict(complete=True,manifest_sha256=D.sha(union_manifest),
        artifacts={n:D.sha(union_bank/n)for n in ('INPUT_BINDINGS.json','BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json')}))
    assert D.run(union_manifest,union_bank,tmp_path/'union_diagnostics')['complete']
    with open(entries[0]['seal_path'],'a')as stream:stream.write(' ')
    with pytest.raises(ValueError,match='Union row seal changed'):D.run(union_manifest,union_bank,tmp_path/'rejected_union')


def test_module_has_no_producer_metric_or_reference_loader_imports():
    import ast
    source=ast.parse(open(D.__file__).read())
    modules=[node.module for node in ast.walk(source)if isinstance(node,ast.ImportFrom)]
    assert modules==['__future__','collections','pathlib']
