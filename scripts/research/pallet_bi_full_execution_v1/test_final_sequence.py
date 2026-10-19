"""Constructed metadata only: no actual child processes, GT, models or poses."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.research.pallet_bi_full_execution_v1 import final_sequence as F


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')


def forbidden(*args,**kwargs):raise AssertionError('No scientific/child command is allowed in constructed tests')


def test_ROOT_GO_rejected_before_wait_or_side_effects(tmp_path,monkeypatch):
    pp,gp=tmp_path/'PLAN.json',tmp_path/'GO.json';put(pp,{'output':str(tmp_path/'never')});put(gp,{'approved':False})
    monkeypatch.setattr(F,'guard',forbidden);monkeypatch.setattr(F,'wait_for',forbidden)
    monkeypatch.setattr(F.subprocess,'call',forbidden)
    with pytest.raises(PermissionError):F.run(pp,gp)
    assert not (tmp_path/'never').exists()


def test_completed_native_adoption_then_skip_never_calls_runner(tmp_path):
    folder=tmp_path/'native';artifact=folder/'RESULT.json';put(artifact,{'constructed':True});terminal=folder/'SEAL.json';put(terminal,{'complete':True})
    validator=lambda:{str(p):F.sha(p)for p in [artifact,terminal]}
    args=(tmp_path/'controller','stage',['literal','argv'],{'sealed_input':'digest'},validator,terminal,folder,forbidden)
    assert F.stage(*args)=='ADOPT'
    assert F.stage(*args)=='SKIP'
    put(artifact,{'changed':True})
    with pytest.raises(ValueError,match='Frozen'):F.stage(*args)


def test_changed_checkpoint_input_and_partial_output_are_preserved(tmp_path):
    folder=tmp_path/'complete';terminal=folder/'SEAL.json';put(terminal,{'complete':True})
    validator=lambda:{str(terminal):F.sha(terminal)}
    F.stage(tmp_path/'controller','one',['argv'],{'input':'a'},validator,terminal,folder,forbidden)
    with pytest.raises(ValueError,match='input/argv'):F.stage(tmp_path/'controller','one',['argv'],{'input':'b'},validator,terminal,folder,forbidden)
    partial=tmp_path/'partial';sentinel=partial/'earlier.json';put(sentinel,{'keep':True});before=sentinel.read_bytes()
    with pytest.raises(RuntimeError,match='Partial output preserved'):
        F.stage(tmp_path/'controller','two',['argv'],{},forbidden,partial/'SEAL.json',partial,forbidden)
    assert sentinel.read_bytes()==before


def extra_fixture(root):
    output=root/'F_FINALIZE';decl=[];ordered=[]
    for index,methods in enumerate([['POSEFIX_SUBPIX_YOLO_ANCHOR','POSEFIX_SUBPIX_PF_ANCHOR'],['S4_PL']]):
        path,sp=root/f'extra{index}.jsonl.gz',root/f'extra{index}_SEAL.json'
        path.write_text('opaque fixture; PF archive may also contain unused POSEFIX_RAW')
        put(sp,{'constructed':True});decl.append(dict(path=str(path),seal=str(sp),methods=methods))
        ordered.append(dict(input_index=index,extra_path=str(path),producer_seal_path=str(sp),admitted_methods=sorted(methods),
            extra_sha256=F.sha(path),producer_seal_sha256=F.sha(sp)))
    put(output/'real_EXTRA_INPUTS.json',dict(extra_inputs=decl));features={}
    for arm in ['F_COMMON','F_BI']:
        folder=output/arm/'real_union';put(folder/'FEATURE_SEAL.json',dict(complete=True,rows=238,
            dependencies={'extra_producer_binding':{'ordered_inputs':ordered,
                'declaration_manifest':dict(path=str(output/'real_EXTRA_INPUTS.json'),sha256=F.sha(output/'real_EXTRA_INPUTS.json'))}}))
        features[arm]={'real':dict(path=str(folder/'HEAD_FEATURES.jsonl.gz'),seal=str(folder/'FEATURE_SEAL.json'))}
    put(output/'FINISH_RECEIPT.json',dict(complete=True,feature_inputs=features))
    return output,decl


def test_F_manifest_preserves_exact_methods_and_actual_union_parent_order(tmp_path):
    output,decl=extra_fixture(tmp_path);plan=dict(run=str(tmp_path),F_finish=str(output),F_core=str(tmp_path/'core'))
    mp=F.manifest(plan,'F');value=F.read(mp)
    assert set(value['arms'])=={'F_COMMON','F_BI'}
    for arm,spec in value['arms'].items():
        assert spec['extra_rows']==decl
        assert spec['features']==F.read(output/'FINISH_RECEIPT.json')['feature_inputs'][arm]['real']['path']
        assert spec['result_root']==str(tmp_path/'core'/arm/'real')
        assert all('POSEFIX_RAW'not in row['methods']for row in spec['extra_rows'])
    # A method omission would expose all PF archive methods; reject it.
    bad=deepcopy(decl);bad[0].pop('methods');put(output/'real_EXTRA_INPUTS.json',dict(extra_inputs=bad))
    with pytest.raises(ValueError,match='methods differ'):F.extra_declarations(output,value['arms']['F_COMMON']['feature_seal'])


def test_F_method_mismatch_or_changed_parent_bytes_rejected(tmp_path):
    output,decl=extra_fixture(tmp_path);sp=output/'F_COMMON/real_union/FEATURE_SEAL.json'
    own=F.read(sp);own['dependencies']['extra_producer_binding']['ordered_inputs'][0]['admitted_methods']+=['POSEFIX_RAW'];put(sp,own)
    with pytest.raises(ValueError,match='methods differ'):F.extra_declarations(output,sp)
    own['dependencies']['extra_producer_binding']['ordered_inputs'][0]['admitted_methods']=sorted(decl[0]['methods']);put(sp,own)
    Path(decl[0]['path']).write_text('changed parent bytes')
    with pytest.raises(ValueError,match='bytes changed'):F.extra_declarations(output,sp)


def test_S2_manifest_maps_prefixes_to_actual_core_arms(tmp_path):
    plan=dict(run=str(tmp_path),S2_core=str(tmp_path/'core'),S2_heads=str(tmp_path/'heads'))
    value=F.read(F.manifest(plan,'S2'))
    assert set(value['arms'])=={'S2_C0','S2_CP','S2_CB'}
    for arm,spec in value['arms'].items():
        name=arm.removeprefix('S2_');assert spec['result_root']==str(tmp_path/'core'/name/'real')
        assert spec['prediction_dir']==str(tmp_path/'heads'/name/'prediction')
        assert spec['generated_feature_seal']==str(tmp_path/'core'/name/'real/FEATURE_SEAL.json')


def test_core_native_checkpoint_uses_string_seeds_and_all_five_populations(tmp_path):
    mp=tmp_path/'manifest.json';put(mp,{'arms':{'S2_C0':{}}});bank=tmp_path/'S2_BANK_ANALYSIS'
    for name in ['ANALYSIS_SEAL.json','INPUT_BINDINGS.json','BANK_COVERAGE_ROWS.jsonl.gz']:put(bank/name,{'opaque':True})
    folder=tmp_path/'S2_CORE_DIAGNOSTICS';names=['CORE_DIAGNOSTIC_ROWS.jsonl.gz','SUMMARY_BY_POPULATION_AND_STRATUM.json','INPUT_BINDINGS.json']
    for name in names:put(folder/name,{'opaque':True})
    source=Path(F.__file__).parent/'post_core_diagnostics.py';value=dict(complete=True,source_sha256=F.sha(source),arms=['S2_C0'],
        rows_per_arm=238,populations={'ALL238':238,**F.POPS},seeds=list(map(str,F.SEEDS)),
        inputs=dict(manifest_sha256=F.sha(mp),post_bank_analysis_seal_sha256=F.sha(bank/'ANALYSIS_SEAL.json'),
            post_bank_input_bindings_sha256=F.sha(bank/'INPUT_BINDINGS.json'),post_bank_coverage_sha256=F.sha(bank/'BANK_COVERAGE_ROWS.jsonl.gz')),
        artifacts={name:F.sha(folder/name)for name in names})
    put(folder/'DIAGNOSTIC_SEAL.json',value);plan=dict(run=str(tmp_path),sources={str(source):F.sha(source)})
    assert len(F.native(plan,'S2_core',mp))==4
    value['seeds']=F.SEEDS;put(folder/'DIAGNOSTIC_SEAL.json',value)
    with pytest.raises(ValueError,match='ancestry/population'):F.native(plan,'S2_core',mp)


def test_frozen_source_changes_rejected_and_exact_stage_order():
    assert F.STAGE_ORDER==['S2_evaluation','S2_independent','S2_bank','S2_selection','S2_core','F_choose','F_finish',
                          'F_evaluation','F_independent','F_bank','F_selection','F_core']
    import ast
    tree=ast.parse(Path(F.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom)and n.module and any(x in n.module for x in ['numpy','torch','evaluation','f_finish'])for n in tree.body)


def test_prepare_freezes_metadata_without_opening_raw_references(tmp_path,monkeypatch):
    run=tmp_path/'run';d1=tmp_path/'d1';refs=tmp_path/'references';source=tmp_path/'frozen.py';source.write_text('# constructed\n')
    put(run/'S2_FULL_UNIT_WEIGHT_FOUR_CPU/PRODUCER_LOCK.json',{'numerical_kernel_source_binding':{str(source):F.sha(source)}})
    waiter=run/'S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py';waiter.write_text('# idle fixture\n')
    fc=run/'F_CORE_FOUR_CPU_CONTROLLER.py';fc.write_text('# idle fixture\n')
    put(run/'S2_UNIT_WEIGHT_FOUR_CPU_H1_AUTOMATION_PLAN.json',dict(head_source_sha256={str(source):F.sha(source)},source_sha256=F.sha(waiter)))
    put(run/'F_FOUR_CPU_CONTROLLER_PLAN.json',dict(source_sha256=F.sha(fc)))
    for name in ['CAMPAIGN_PLAN.json','ENVIRONMENT_INITIAL.json','F_SELECTION_RULE_CLARIFICATION.json',
        'F_AUXILIARY_MAPPING_IMPLEMENTATION_NOTE.json','F_OBSERVER_POLICY_LOCK_V2.json','F_CORE_PRECOMPUTE_LOCK.json']:
        put(run/name,{'constructed':True})
    put(d1/'PREDICTION_SEAL.json',{'constructed':True});put(refs/'SCORE_SEAL.json',{'artifacts':{'FIXED_REFERENCES.json':'hash-only'}})
    fixed=run/'FIXED_SETS.json';put(fixed,{'sets':{}});raw=refs/'FIXED_REFERENCES.json';raw.write_text('NEVER_OPEN_RAW_REFERENCE')
    original=F.sha
    def guarded(path):
        if Path(path).name=='FIXED_REFERENCES.json':raise AssertionError('Preparation opened raw references')
        return original(path)
    monkeypatch.setattr(F,'sha',guarded);plan=F.prepare(run,d1,tmp_path/'old',tmp_path/'prior',refs,fixed,tmp_path/'controller')
    assert str(raw)not in plan['fixed_inputs']and plan['child_H1_fits']==6
    source.write_text('# changed\n')
    with pytest.raises(ValueError,match='Frozen'):F.guard(plan)


def test_completed_controller_second_run_authenticates_without_any_write(tmp_path,monkeypatch):
    output=tmp_path/'controller';output.mkdir();pp,gp=tmp_path/'PLAN.json',tmp_path/'GO.json'
    plan=dict(output=str(output),run=str(tmp_path),sources={},fixed_inputs={});put(pp,plan)
    put(gp,dict(approved=True,plan_sha256=F.sha(pp),controller_sha256=F.sha(F.__file__),F_H1_fits_authorized=6,
        S2_H1_fits_from_this_controller=0,max_scientific_CPU_workers=8))
    ledger=output/'ACTUAL_COMMAND_LEDGER.jsonl';ledger.write_text('preserved actual ledger\n');checks={};files={}
    for key in F.STAGE_ORDER:
        item=tmp_path/(key+'_artifact.json');put(item,{'opaque':key});files[key]={str(item):F.sha(item)}
        cp=output/'checkpoints'/(key+'.json');put(cp,dict(complete=True,identity=dict(input_bindings={'PLAN.json':F.sha(pp),'ROOT_GO':F.sha(gp)}),completed_files=files[key]))
        checks[cp.name]=F.sha(cp)
    put(output/'FINAL_SEQUENCE_SEAL.json',dict(complete=True,plan_sha256=F.sha(pp),ROOT_GO_sha256=F.sha(gp),source_sha256=F.sha(F.__file__),
        stage_order=F.STAGE_ORDER,checkpoints=checks,actual_command_ledger_sha256=F.sha(ledger)))
    before={str(p):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}
    monkeypatch.setattr(F,'native',lambda plan,key,mp=None:files[key]);monkeypatch.setattr(F,'append_ledger',forbidden)
    monkeypatch.setattr(F,'wait_for',forbidden);monkeypatch.setattr(F.subprocess,'call',forbidden)
    assert F.run(pp,gp)['complete']
    assert {str(p):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}==before
