"""V2 one-shot assembly: completed science commands plus preserved repair history.

No IO on import; no subprocess, watcher, model import, checkpoint load/hash,
RGB/annotation/reference/prediction cache read. The only outputs are RUN_LOG.json
and preflight/RUN_LOG_QA.json. Original command-record field names are preserved.

ROOT supplies preflight/RUN_LOG_INPUTS_V2.json with schema run_log_inputs_v2 and
input_files={absolute_path:{kind: exact REQUIRED[path], sha256:64hex}}. The key set
must equal REQUIRED (metadata bodies, command-event journals, stdout and sources).
Sources must also match FROZEN. ROOT_REVIEW at RUN_LOG_ROOT_REVIEW_V2.json contains
passed:true, source_sha256:{self:SHA, **FROZEN}. RUN_LOG_GO_V2.json requires schema
run_log_go_v2, passed:true, assembly_authorized:true, input_manifest_sha256,
ROOT_review_sha256 and source_sha256 with that exact map. Invoke --run once after
all upstream producers finish. Missing/partial/nonzero records fail; no retry or
planned command is substituted. Upstream input/CP/manifest/SUBSETS hashes are
declarations only: those target files are never opened or rehashed here.

One pre-existing CPU queue omitted returncode from its QA. That raw field remains
absent; source-gated zero-exit evidence is disclosed, never a fabricated stored0.
Private training histories and per-frame identifiers/point arrays are not copied.
"""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import math
import re
import sys

OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent;SOURCE=Path(__file__).resolve()
MAIN=BASE/'dope-filter-ablation-per-type-20261027-v1'
WOOD=BASE/'dope-wood640-eval80-20261027-v1';COMBINED=WOOD/'combined372'
RATIO=BASE/'dope-r2-ratio-per-type-20261027-v1'
KINDS=('PLASTIC','WOOD','GREEN');CONDITIONS=('no_confidence','no_ransac','no_loo','no_final_support','confidence_only')
FIRST_STAGES=['wood80_seal','wood80_infer','wood80_score','ratio_seal_go',
    'ratio_original_wood_reproduction','ratio_original_wood_parity',*['ratio_wood_'+c for c in 'ABCDE']]
FINISH_STAGES=['merged_baseline_caches','ratio_evaluation_seal','merged_primary372_points',
    'merged_old45_pose_golden','merged_primary372_supplemental_pose','wood_ratios_infer','wood_ratios_score',
    *['ratio_'+k+'_'+c for k in ('GREEN','PLASTIC') for c in 'ABCDE'],
    'all_ratios_infer','all_ratios_score','all_ratios_plots','all_ratios_future_teacher_select','all_ratios_future_teacher_score']
FROZEN={
    str(MAIN/'code/run_campaign.py'):'b39b2e506590b6de792e8c9f3785f34e7352e7bb717b89b805b209845cb51f65',
    str(MAIN/'code/experiment_driver.py'):'5bdff0e7c8fd90b2d8c3b16affcb6ae74dee2c10cb9e93cc3553961a3e5907fa',
    str(OUT/'code/finalize_ablation.py'):'963e44708ba4ceab5734926b53c7eb3f5433497dbbbb0b3a08a514eaaab6e751',
    str(OUT/'code/advance_merged_workflow.py'):'ce2509d14f94e415243820341eac6e1203bbeae5cb6769251757403b5491a098',
    str(OUT/'code/finish_merged_experiments.py'):'4b11205c5cbbe7b29f6a032a1a57313ac4143010cc0b2e946bb73b4910928275',
    str(OUT/'code/run_main_model_comparisons.py'):'53fb942a283c4d37b3c3e64d6816bc28c65074399f04be60c519d9961cc028c7',
    str(OUT/'code/run_merged_filter_diagnostics.py'):'6b2daf082e2bf31bc60b292d1417e7548e147eae308bdc1357004211956d44c8',
    str(OUT/'code/run_filter_statistics_queue.py'):'599e543e463d1bb2492ef8e8984650617c93a01e6899428c6987586e4e68a472',
    str(RATIO/'code/experiment_driver.py'):'d54a3a300e65f7b247cda9444123c59b0b84afaf7181c4e111384a9dcf56709a',
    str(RATIO/'code/completion_policy.py'):'9e9cb77d418d366a6fba8171045732515d9ac4eb3058a464f26b155cb3b50634',
    str(WOOD/'code/main_model_comparisons.py'):'96fcb6c233bc4f768f558bc1bc8d57cbb9f813f308fde63fe0ba19dcfc73a46e',
    str(WOOD/'code/merged_filter_diagnostics.py'):'e83f23bfe3f1e1429654ed7e2036993abd68849498ff0cb0b30a1f9693f6832a'}
# Keep original source identities as historical evidence; bind all actual V2 entrypoints.
FROZEN.update({
    str(OUT/'code/finish_merged_experiments_v2.py'):'3552f73710797fe69fca4990f06202b89db1d7f399fba3c5f0f7d84abe51b9f1',
    str(OUT/'code/run_main_model_comparisons_v2.py'):'f6691ef509d3c16d435ac0e1b49eb7a62cc7756050c070b53d7c48e8fb656585',
    str(OUT/'code/run_merged_filter_diagnostics_v2.py'):'9b08219e7c9edef23b149612d93a790661d4ee677bcd709f49dbbb372869e3ce',
    str(OUT/'code/run_filter_statistics_queue_v2.py'):'8ee4d51539d156244ebdd3893e9477f734d5bb8afdb1b5165c413c3cde7b4431',
    str(OUT/'code/prepare_ratio_evaluation_v2.py'):'11278e1b623c251c98276305390de32cb3cd030cd740c25c52aeeb19da959322',
    str(OUT/'code/run_filter_diagnostics_with_metadata_v2.py'):'ca4df1f250ae8db1fae7dedfa50c77d4cae1cfa9939fda4fc7c3438ff14ab837',
    str(OUT/'code/assemble_run_log.py'):'610a3166db9aea685491c61e817507fc7443d4622e24b10ffd044cbfd0f7df89',
    str(WOOD/'code/prepare_combined.py'):'9430e48df9ff61da67d3d1df1aaf799deda1c0294d8bf739f2eac19d42c5ab53',
    str(OUT/'code/prepare_ratio_evaluation.py'):'3b8f18bed78e2bbece33f0488d0bdfe8ee0d0a4890f83af639c4924d6f4d02f7'})
ADAPTER=OUT/'code/run_filter_diagnostics_with_metadata_v2.py'
NORMALIZER=OUT/'code/prepare_ratio_evaluation_v2.py'
SCIENCE=WOOD/'code/merged_filter_diagnostics.py'
FILTER_FOLDER=COMBINED/'filter_diagnostics/filter372_v1'
REPAIRS={
    'failed_finish':OUT/'FINISH_EXPERIMENTS_FAILURE.json',
    'failed_diagnostics':OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',
    'failed_statistics':OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE.json',
    'failed_comparisons':OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE.json',
    'finish_authority':OUT/'preflight/FINISH_WORKFLOW_RESUME_AUTHORITY_V2.json',
    'outer_authority':OUT/'preflight/OUTER_WORKFLOW_RESUME_AUTHORITY_V2.json',
    'ROOT_observed_prior_comparison_exit':OUT/'preflight/MAIN_COMPARISON_QUEUE_RESTART_20261027T1100.json'}
META={
    'campaign':MAIN/'CAMPAIGN_COMPLETION.json',
    'ablation_finalizer':OUT/'ABLATION_FINALIZE_COMPLETE.json',
    'first_queue':OUT/'MERGED_WORKFLOW_COMPLETE.json',
    'finish_queue':OUT/'FINISH_EXPERIMENTS_COMPLETE.json',
    'main_comparisons_queue':OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_QA.json',
    'filter_workflow':OUT/'FILTER_DIAGNOSTIC_WORKFLOW_COMPLETE.json',
    'filter_statistics_queue':OUT/'preflight/FILTER_STATISTICS_QUEUE_QA.json',
    'main_infer':MAIN/'evaluation/trained_v1/infer/COMPLETE.json',
    'WOOD80_infer':WOOD/'inference/COMPLETE.json',
    'main_scores':MAIN/'evaluation/final_v1/final/COMPLETE.json',
    'main_bootstrap':MAIN/'evaluation/final_v1/final/BOOTSTRAP_RECEIPT.json',
    'merged_scores':COMBINED/'evaluation/final_v1/final/COMPLETE.json',
    'comparison_complete':COMBINED/'comparisons/main_v1/COMPLETE.json',
    'comparison_bootstrap':COMBINED/'comparisons/main_v1/BOOTSTRAP_RECEIPT.json',
    'ratio_infer_wood':RATIO/'cache/ratios_v1/COMPLETE_wood.json',
    'ratio_infer_all':RATIO/'cache/ratios_v1/COMPLETE_all.json',
    'ratio_scores':RATIO/'evaluation/ratios_v1/all/COMPLETE.json',
    'ratio_bootstrap':RATIO/'evaluation/ratios_v1/all/BOOTSTRAP_RECEIPT.json',
    'future_teacher':RATIO/'diagnostics/ratios_v1/all/COMPLETE.json',
    'technical_reproduction':RATIO/'reproduction/WOOD/TRAIN_COMPLETE.json',
    'technical_protocol':RATIO/'reproduction/WOOD/PROTOCOL.json',
    'technical_parity':RATIO/'reproduction/WOOD/CANONICAL_PARITY_QA.json',
    'filter_infer':COMBINED/'filter_diagnostics/filter372_v1/infer/COMPLETE.json',
    'filter_equivalence':COMBINED/'filter_diagnostics/filter372_v1/select/EQUIVALENCE.json',
    'filter_score':COMBINED/'filter_diagnostics/filter372_v1/score/COMPLETE.json',
    'filter_statistics':OUT/'preflight/MERGED_FILTER_STATISTICS_QA.json',
    'ratio_exposures':OUT/'results/ratio_exposures/QA.json'}
JOURNALS={'first_queue':OUT/'logs/MERGED_COMMANDS.jsonl','finish_queue':OUT/'logs/FINISH_COMMANDS_V2.jsonl','original_finish':OUT/'logs/FINISH_COMMANDS.jsonl',
    'ratio_driver':RATIO/'logs/COMMANDS.jsonl','main_forward_events':MAIN/'evaluation/trained_v1/infer/FORWARDS.jsonl',
    'WOOD80_forward_events':WOOD/'inference/FORWARDS.jsonl',
    'ratio_forward_events':RATIO/'cache/ratios_v1/FORWARDS.jsonl',
    'filter_forward_events':COMBINED/'filter_diagnostics/filter372_v1/infer/FORWARDS.jsonl'}
META.update({'diagnostic_GO':WOOD/'preflight/FILTER_DIAGNOSTIC_GO.json'})
for stage in ('infer','select','score'):
    META['adapter_'+stage]=FILTER_FOLDER/('METADATA_ADAPTER_'+stage+'_COMPLETE.json')
    META['adapter_preflight_'+stage]=FILTER_FOLDER/('METADATA_ADAPTER_'+stage+'.json')
META.update(REPAIRS)
MAIN_ARMS={(c,k):MAIN/c/k for c in CONDITIONS for k in KINDS}
RATIO_ARMS={(k,c):RATIO/'trials'/k/c for k in ('WOOD','GREEN','PLASTIC') for c in 'ABCDE'}
LOGS=[MAIN/'logs'/f'{c}_{k}.stdout.log' for c,k in MAIN_ARMS]
LOGS += [OUT/'logs'/f'ablation_{s}.stdout.log' for s in ('infer','final','plot','integrity','audit','package')]
LOGS += [OUT/'logs'/f'merged_{s}.stdout.log' for s in FIRST_STAGES]
LOGS += [OUT/'logs/finish_merged_baseline_caches.stdout.log']
LOGS += [OUT/'logs'/f'finish_v2_{s}.stdout.log' for s in FINISH_STAGES[1:]]
REPAIR_LOG=OUT/'logs/finish_ratio_evaluation_seal.stdout.log'
LOGS += [OUT/'logs/main_model_comparisons_main_v1.stdout.log']
LOGS += [OUT/'logs'/f'filter372_filter372_v1_{s}.stdout.log' for s in ('infer','select','score')]
LOGS += [OUT/'logs'/f'filter_statistics_queue_{s}.stdout.log' for s in ('statistics','exposures')]
REQUIRED={**{str(p):'json' for p in META.values()},**{str(p):'jsonl' for p in JOURNALS.values()},
    **{str(p):'stdout' for p in [*LOGS,REPAIR_LOG]},**{p:'source' for p in FROZEN}}
for p in MAIN_ARMS.values():
    for name in ('PROTOCOL.json','TRAIN_COMPLETION.json'):REQUIRED[str(p/name)]='json'
    REQUIRED[str(p/'COMMANDS.jsonl')]='jsonl'
for p in RATIO_ARMS.values():
    for name in ('PROTOCOL.json','TRAIN_COMPLETE.json'):REQUIRED[str(p/name)]='json'
INPUTS=OUT/'preflight/RUN_LOG_INPUTS_V2.json';REVIEW=OUT/'preflight/RUN_LOG_ROOT_REVIEW_V2.json'
GO=OUT/'preflight/RUN_LOG_GO_V2.json';TARGET=OUT/'RUN_LOG.json';QA=OUT/'preflight/RUN_LOG_QA.json'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def raw(path):
    p=Path(path);require(p.resolve()==p and not p.is_symlink() and p.is_file(),'Missing/unsafe exact metadata: '+str(p))
    return p.read_bytes()


def sha(path):return hashlib.sha256(raw(path)).hexdigest()


class Inputs:
    def __init__(self):
        self.bootstrap={str(p):sha(p) for p in (SOURCE,INPUTS,REVIEW,GO)}
        manifest=json.loads(raw(INPUTS));review=json.loads(raw(REVIEW));go=json.loads(raw(GO))
        sources={**FROZEN,str(SOURCE):self.bootstrap[str(SOURCE)]}
        require(review['passed'] is True and review['source_sha256']==sources,'ROOT source review stale')
        require(manifest['schema']=='run_log_inputs_v2' and set(manifest['input_files'])==set(REQUIRED),'Exact metadata whitelist differs')
        require(go['schema']=='run_log_go_v2' and go['passed'] is True and go['assembly_authorized'] is True
            and go['input_manifest_sha256']==self.bootstrap[str(INPUTS)] and go['ROOT_review_sha256']==self.bootstrap[str(REVIEW)]
            and go['source_sha256']==sources,'Explicit ROOT assembly GO absent/stale')
        self.files=manifest['input_files'];self.used={};self.values={}
        for path,kind in REQUIRED.items():
            item=self.files[path];require(set(item)=={'kind','sha256'} and item['kind']==kind and digest(item['sha256']),'Invalid approved metadata entry')
        for path,expected in FROZEN.items():require(self.files[path]['sha256']==expected,'Frozen producer source differs')
        for p,expected in self.bootstrap.items():require(sha(p)==expected,'Authorization changed during read')

    def read(self,path):
        p=str(path);require(p in REQUIRED and REQUIRED[p] in ('json','jsonl'),'Unlisted metadata body read')
        if p in self.values:return self.values[p]
        data=raw(path);d=hashlib.sha256(data).hexdigest();require(d==self.files[p]['sha256'],'ROOT-approved metadata bytes differ: '+p)
        result=json.loads(data) if REQUIRED[p]=='json' else [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        require(sha(path)==d,'Metadata changed during read');self.used[p]=d;self.values[p]=result;return result

    def hash(self,path):
        p=str(path);require(p in REQUIRED,'Unlisted byte/hash access')
        d=sha(path);require(d==self.files[p]['sha256'],'Approved metadata/stdout/source SHA differs: '+p);self.used[p]=d;return d

    def unchanged(self):
        for path,d in {**self.bootstrap,**self.used}.items():require(sha(path)==d,'Run-log input changed: '+path)


def declared_hashes(value,pointer=''):
    """Preserve hash declarations and pointers only; never follow their paths."""
    result=[]
    if isinstance(value,dict):
        for key,item in value.items():
            token=str(key).replace('~','~0').replace('/','~1');ptr=pointer+'/'+token
            if digest(item):result.append(dict(json_pointer=ptr,sha256=item,declared_path=key if str(key).startswith('/') else None))
            elif isinstance(item,dict):result.extend(declared_hashes(item,ptr))
    return result  # Lists (including private per-update histories) are never copied/traversed.


def projection(value,fields):return {key:value[key] for key in fields if key in value}


def command(io,group,path,pointer,record,logkey='log',implicit_zero=False):
    require(isinstance(record['command'],list) and record['command'] and all(isinstance(t,str) for t in record['command']),'Command argv absent')
    require(isinstance(record['elapsed_seconds'],(int,float)) and math.isfinite(record['elapsed_seconds']) and record['elapsed_seconds']>=0,'Invalid actual elapsed time')
    if implicit_zero:require('returncode' not in record,'Previously missing returncode schema changed')
    else:require(type(record['returncode']) is int and record['returncode']==0,'Incomplete/nonzero actual command')
    log=Path(record[logkey]);require(log in LOGS,'Unexpected actual stdout path')
    stdout_sha=io.hash(log)
    for key in ('stdout_sha256',):
        if key in record:require(record[key]==stdout_sha,'Stored stdout SHA differs')
    item=dict(group=group,source_path=str(path),json_pointer=pointer,record=record,verified_stdout_sha256=stdout_sha)
    if implicit_zero:item['returncode_evidence']=dict(recorded_returncode_field_present=False,
        source_path=str(OUT/'code/run_main_model_comparisons_v2.py'),source_sha256=FROZEN[str(OUT/'code/run_main_model_comparisons_v2.py')],
        statement="require(result.returncode==0,'CPU comparison child failed') precedes successful QA creation",
        interpretation='Completion is source-gated by zero exit; this is not a stored returncode measurement.')
    return item


def complete(value,passed=False):
    require(value['complete'] is True and (not passed or value['passed'] is True),'Upstream completion failed/partial')


def forward_counts(rows,keys,expected,completed_event='complete'):
    require(all(r['event'] in ('attempt',completed_event) for r in rows),'Unexpected forward journal event')
    attempts=[tuple(r[k] for k in keys) for r in rows if r['event']=='attempt'];done=[tuple(r[k] for k in keys) for r in rows if r['event']==completed_event]
    require(len(attempts)==len(done)==expected and Counter(attempts)==Counter(done)
        and len(set(attempts))==expected,'Actual forward attempt/completion identity differs')
    return dict(attempt_events=len(attempts),completion_events=len(done),unique_pairs=len(set(attempts)),identity_fields=list(keys))


def repair_history(io,meta,prior):
    failed={str(REPAIRS[k]):io.hash(REPAIRS[k]) for k in
        ('failed_finish','failed_diagnostics','failed_statistics','failed_comparisons')}
    authorities={key:meta[key] for key in ('finish_authority','outer_authority')}
    for key,value in authorities.items():
        require(value['schema']=='outer_workflow_resume_authority_v2' and value['passed'] is True
            and value['technical_repair_only'] is True and value['scientific_core_or_target_tags_changed'] is False
            and value['old_failures']==failed,'Preserved four-failure ROOT authority differs: '+key)
    expected_rebindings={
        'finish_authority':{'finish_merged_experiments_v2.py':'advance_merged_workflow.py'},
        'outer_authority':{'run_main_model_comparisons_v2.py':'finish_merged_experiments_v2.py',
            'run_merged_filter_diagnostics_v2.py':'finish_merged_experiments_v2.py',
            'run_filter_statistics_queue_v2.py':'run_merged_filter_diagnostics_v2.py'}}
    for key,pairs in expected_rebindings.items():
        entries=authorities[key]['approved_rebindings']
        require(set(entries)=={str(OUT/'code'/n) for n in pairs},'Exact reviewed resume source set differs')
        for name,parent in pairs.items():
            source=OUT/'code'/name;producer=OUT/'code'/parent;entry=entries[str(source)]
            require(entry['source_sha256']==FROZEN[str(source)] and entry['producer']==dict(
                source_path=str(producer),source_sha256=FROZEN[str(producer)])
                and set(entry['allowed_old_failures']).issubset(failed),'Resume static source/producer binding differs')
    finish_sha=io.hash(REPAIRS['finish_authority']);outer_sha=io.hash(REPAIRS['outer_authority'])
    finished=meta['finish_queue'];require(finished['repair_authority_sha256']==finish_sha
        and finished['reused_completed_stages']==['merged_baseline_caches']
        and finished['original_attempt_journal']==str(JOURNALS['original_finish'])
        and finished['resume_journal']==str(JOURNALS['finish_queue']),'Actual resumed finish receipt differs')
    for key in ('main_comparisons_queue','filter_workflow','filter_statistics_queue'):
        require(meta[key]['resume_authority_sha256']==outer_sha,'Actual outer queue omitted ROOT resume authority')
    reused=authorities['finish_authority']['reused_completed_stage']
    baseline_paths={str(COMBINED/'cache'/f'BASE_{k}_{r}.json') for k in KINDS for r in ('R0','R1','control_R2','oldfull_R2')}
    require(reused['stage']=='merged_baseline_caches' and reused['journal_path']==str(JOURNALS['original_finish'])
        and reused['journal_sha256']==io.hash(JOURNALS['original_finish'])
        and set(reused['output_sha256'])==baseline_paths and all(digest(h) for h in reused['output_sha256'].values())
        and set(reused['stdout_sha256'])=={r['log'] for r in prior},'Reused successful baseline authority differs')
    for path,expected in reused['stdout_sha256'].items():require(io.hash(path)==expected,'Preserved original success/failure stdout differs')
    attempted=prior[1]
    require(attempted['stage']=='ratio_evaluation_seal' and type(attempted['returncode']) is int
        and attempted['returncode']==1 and attempted['log']==str(REPAIR_LOG)
        and attempted['command'][2:]==[str(OUT/'code/prepare_ratio_evaluation.py'),'--seal']
        and attempted['script_sha256']==FROZEN[str(OUT/'code/prepare_ratio_evaluation.py')]
        and math.isfinite(attempted['elapsed_seconds']) and attempted['elapsed_seconds']>=0,
        'Preserve the actual failed sealer command separately from successful60')
    require(prior[0]['script_sha256']==FROZEN[str(WOOD/'code/prepare_combined.py')],
        'Reused baseline scientific-cache preparation source differs')
    observation=meta['ROOT_observed_prior_comparison_exit']
    require(observation['observed_prior_session_exit_code']==143 and observation['prior_session']==90925
        and observation['prior_pid']==453979 and 'prior_terminal_state' in observation,
        'ROOT observed prior outer-session termination evidence differs')
    return dict(failed_attempted_command_records=[dict(source_path=str(JOURNALS['original_finish']),json_pointer='JSONL:2',
            record=attempted,verified_stdout_sha256=io.hash(REPAIR_LOG),included_in_successful60=False)],
        cascade_failure_records=[dict(source_path=str(REPAIRS[k]),sha256=failed[str(REPAIRS[k])],
            record=projection(meta[k],['utc','error','passed','automatic_retry','published',
                'input_or_performance_tuning_attempted','root_repair_required']))
            for k in ('failed_finish','failed_diagnostics','failed_statistics','failed_comparisons')],
        ROOT_observed_prior_outer_session=dict(source_path=str(REPAIRS['ROOT_observed_prior_comparison_exit']),
            sha256=io.hash(REPAIRS['ROOT_observed_prior_comparison_exit']),
            record=projection(observation,['observed_prior_session_exit_code','prior_session','prior_pid','prior_terminal_state']),
            observed_exit_JSON_pointer='/observed_prior_session_exit_code',
            scope='ROOT observation of outer session90925; not a stored scientific child-command returncode',
            termination_cause='unknown; no cause inferred from exit143 or absent process'),
        resume_authorities=[dict(source_path=str(REPAIRS[key]),sha256=io.hash(REPAIRS[key]),
            record=projection(value,['schema','passed','technical_repair_only','scientific_core_or_target_tags_changed',
                'old_failures','approved_rebindings','reused_completed_stage'])) for key,value in authorities.items()],
        reused_baseline_cache_SHA_declarations_followed=False,
        scientific_condition_or_seed_changed=False)


def metadata_adapter_chain(io,meta):
    go=meta['diagnostic_GO'];sources=go['source_sha256']
    require(go['passed'] is True and go['tag']=='filter372_v1' and len(sources)==15
        and sources.get(str(SCIENCE))==FROZEN[str(SCIENCE)] and sources.get(str(ADAPTER))==FROZEN[str(ADAPTER)]
        and sources.get(str(NORMALIZER))==FROZEN[str(NORMALIZER)]
        and all(digest(h) for h in sources.values()),'Explicit unchanged e83+original12+adapter+normalizer source15 differs')
    post_hashes={str(META['adapter_'+s]):io.hash(META['adapter_'+s]) for s in ('infer','select','score')}
    require(meta['filter_workflow']['metadata_adapter_receipt_sha256']==post_hashes,
        'Completed workflow omitted actual metadata adapter stages')
    result=[]
    for stage,key in (('infer','filter_infer'),('select','filter_equivalence'),('score','filter_score')):
        receipt=meta['adapter_'+stage];preflight=meta['adapter_preflight_'+stage]
        complete(receipt,True);complete(preflight,True)
        expected_path=META[key];binding_key='source_input_sha256' if stage=='select' else 'input_sha256'
        require(receipt['stage']==preflight['stage']==stage and receipt['tag']==preflight['tag']=='filter372_v1'
            and receipt['scientific_kernel_path']==str(SCIENCE) and receipt['scientific_kernel_sha256']==FROZEN[str(SCIENCE)]
            and receipt['metadata_adapter_path']==str(ADAPTER) and receipt['metadata_adapter_sha256']==FROZEN[str(ADAPTER)]
            and receipt['pure_normalizer_path']==str(NORMALIZER) and receipt['pure_normalizer_sha256']==FROZEN[str(NORMALIZER)]
            and receipt['original_function_objects_unchanged'] is True and receipt['original_SOURCE_unchanged'] is True
            and receipt['metadata_only_override'] is True and receipt['original_files_modified'] is False
            and receipt['scientific_functions_replaced'] is False and receipt['source_sha256']==sources
            and receipt['original_stage_receipt_path']==str(expected_path)
            and receipt['original_stage_receipt_sha256']==io.hash(expected_path)
            and receipt['metadata_preflight_path']==str(META['adapter_preflight_'+stage])
            and receipt['metadata_preflight_sha256']==io.hash(META['adapter_preflight_'+stage])
            and preflight['explicit_GO_source_sha256']==sources,'Actual metadata-only adapter provenance chain differs')
        for path,expected in {str(SCIENCE):FROZEN[str(SCIENCE)],str(ADAPTER):FROZEN[str(ADAPTER)],
            str(NORMALIZER):FROZEN[str(NORMALIZER)],str(META['adapter_preflight_'+stage]):io.hash(META['adapter_preflight_'+stage])}.items():
            require(meta[key][binding_key].get(path)==expected,'Original scientific receipt omitted actual adapter/source/preflight identity')
        result.append(dict(stage=stage,source_path=str(META['adapter_'+stage]),record=receipt,
            metadata_preflight_path=str(META['adapter_preflight_'+stage]),metadata_preflight_sha256=io.hash(META['adapter_preflight_'+stage])))
    require(all(r['command'][2]==str(ADAPTER) for r in meta['filter_workflow']['stages']),
        'Actual diagnostic command does not name the declared metadata adapter')
    return dict(scientific_kernel_path=str(SCIENCE),scientific_kernel_sha256=FROZEN[str(SCIENCE)],
        original_dependency_count=12,explicit_source_count=15,stages=result)


def assemble(io):
    meta={key:io.read(p) for key,p in META.items()};records=[];training=[];seed_evidence=[]
    campaign=meta['campaign'];require(campaign['finished'] is True and campaign['all_arms_complete'] is True
        and campaign['total_arms']==campaign['passed']==15 and campaign['failed']==0 and campaign['frozen_inputs_unchanged'] is True,'MAIN15 completion required')
    require([(r['condition'],r['kind']) for r in campaign['attempts']]==list(MAIN_ARMS),'MAIN15 order differs')
    for n,row in enumerate(campaign['attempts']):
        c,k=row['condition'],row['kind'];arm=MAIN_ARMS[c,k];receipt=io.read(arm/'TRAIN_COMPLETION.json');protocol=io.read(arm/'PROTOCOL.json')
        complete(receipt);require(row['complete'] is True and receipt['condition']==c and receipt['kind']==k
            and receipt['main_optimizer_updates']==1656 and receipt['optimizer_steps']==[1656]
            and protocol['seed']==42 and receipt['protocol_sha256']==io.hash(arm/'PROTOCOL.json'),'MAIN arm identity/update/seed binding differs')
        require(row['command'][1:]==[str(MAIN/'code/experiment_driver.py'),'--condition',c,'--kind',k],'MAIN actual argv differs')
        calls=io.read(arm/'COMMANDS.jsonl');normal=[(i,r) for i,r in enumerate(calls) if '--seal-only' not in r['argv']]
        require(len(normal)==1 and normal[0][1]['argv']==row['command'] and normal[0][1]['seed']==42,'MAIN invocation journal does not match actual completed command')
        rounds=receipt['rounds'];require([r['round'] for r in rounds]==[1,2] and all(r['complete'] is True for r in rounds),'Both actual MAIN rounds required')
        records.append(command(io,'MAIN_training',META['campaign'],f'/attempts/{n}',row))
        training.append(dict(experiment='MAIN',kind=k,condition=c,receipt_path=str(arm/'TRAIN_COMPLETION.json'),
            receipt_sha256=io.hash(arm/'TRAIN_COMPLETION.json'),invocation_source=str(arm/'COMMANDS.jsonl'),
            invocation_jsonl_line=normal[0][0]+1,invocation_record=normal[0][1],
            completion=projection(receipt,['complete','condition','kind','protocol_sha256','main_optimizer_updates','optimizer_steps']),
            rounds=[projection(r,['round','complete','teacher_checkpoint_sha256','checkpoint_path','checkpoint_sha256','resume_checkpoint_sha256','pseudo_count']) for r in rounds],
            protocol_sha256_declarations=declared_hashes(protocol)))
        seed_evidence.append(dict(purpose='MAIN training',seed=42,path=str(arm/'PROTOCOL.json'),json_pointer='/seed'))
    for key in ('ablation_finalizer','first_queue','finish_queue','main_comparisons_queue','filter_workflow','filter_statistics_queue'):complete(meta[key],key in ('main_comparisons_queue','filter_workflow','filter_statistics_queue'))
    for key,field,source in (('first_queue','queue_sha256','advance_merged_workflow.py'),
        ('finish_queue','workflow_sha256','finish_merged_experiments_v2.py'),
        ('main_comparisons_queue','queue_source_sha256','run_main_model_comparisons_v2.py')):
        require(meta[key][field]==FROZEN[str(OUT/'code'/source)],'Actual queue completion source differs: '+key)
    require(meta['filter_workflow']['source_sha256'][str(OUT/'code/run_merged_filter_diagnostics_v2.py')]
        ==FROZEN[str(OUT/'code/run_merged_filter_diagnostics_v2.py')]
        and meta['filter_statistics_queue']['source_sha256'][str(OUT/'code/run_filter_statistics_queue_v2.py')]
        ==FROZEN[str(OUT/'code/run_filter_statistics_queue_v2.py')],'Diagnostic completion producer differs')
    finalizer=meta['ablation_finalizer'];require(finalizer['campaign_sha256']==io.hash(META['campaign'])
        and finalizer['code_sha256']==FROZEN[str(OUT/'code/finalize_ablation.py')],'Finalizer source/campaign binding differs')
    require([r['stage'] for r in finalizer['stages']]==['infer','final','plot','integrity','audit','package'],'Finalizer stage catalogue differs')
    for i,r in enumerate(finalizer['stages']):records.append(command(io,'MAIN_finalization',META['ablation_finalizer'],f'/stages/{i}',r))
    first=io.read(JOURNALS['first_queue']);require([r['stage'] for r in first]==FIRST_STAGES,'Actual first-queue catalogue differs')
    for i,r in enumerate(first):records.append(command(io,'first_queue',JOURNALS['first_queue'],f'JSONL:{i+1}',r))
    prior=io.read(JOURNALS['original_finish']);resumed=io.read(JOURNALS['finish_queue'])
    require([r['stage'] for r in prior]==FINISH_STAGES[:2] and [r['returncode'] for r in prior]==[0,1],
        'Preserve exactly the original successful baseline and failed metadata sealer')
    require([r['stage'] for r in resumed]==FINISH_STAGES[1:],'Actual21 resumed finish stages differ')
    records.append(command(io,'finish_queue',JOURNALS['original_finish'],'JSONL:1',prior[0]))
    for i,r in enumerate(resumed):records.append(command(io,'finish_queue',JOURNALS['finish_queue'],f'JSONL:{i+1}',r))
    repair=repair_history(io,meta,prior)
    adapter_chain=metadata_adapter_chain(io,meta)
    comparison=meta['main_comparisons_queue'];require(comparison['total_CI_rows']==210,'Main comparison queue incomplete')
    require(comparison['comparison_COMPLETE_sha256']==io.hash(META['comparison_complete'])
        and meta['filter_workflow']['score_complete_path']==str(META['filter_score'])
        and meta['filter_workflow']['score_complete_sha256']==io.hash(META['filter_score'])
        and meta['filter_statistics_queue']['statistics_QA_sha256']==io.hash(META['filter_statistics'])
        and meta['filter_statistics_queue']['exposure_QA_sha256']==io.hash(META['ratio_exposures']),
        'Queue final-receipt handoff differs')
    # Do not copy input_sha256 into the command record: retain the original field set via projection,
    # while the complete original QA remains bound by its file SHA and metadata declarations below.
    cmprecord=projection(comparison,['command','elapsed_seconds','stdout_path','stdout_sha256'])
    records.append(command(io,'main_comparisons',META['main_comparisons_queue'],'/',cmprecord,'stdout_path',True))
    for key,stages,logkey in (('filter_workflow',['infer','select','score'],'stdout_log'),('filter_statistics_queue',['statistics','exposures'],'stdout_path')):
        rows=meta[key]['stages'];require([r['stage'] for r in rows]==stages,'Completed diagnostic stages differ')
        for i,r in enumerate(rows):records.append(command(io,key,META[key],f'/stages/{i}',r,logkey))
    ratio_calls=io.read(JOURNALS['ratio_driver']);expected_calls=[['reproduce']]+[['trial','--kind',k,'--condition',c] for k,c in RATIO_ARMS]
    require([r['argv'][1:] for r in ratio_calls]==expected_calls,'Actual ratio16 invocation catalogue differs')
    require(all(r['argv'][0]==str(RATIO/'code/experiment_driver.py') for r in ratio_calls),'Ratio inner driver path differs')
    for (k,c),arm in RATIO_ARMS.items():
        receipt=io.read(arm/'TRAIN_COMPLETE.json');protocol=io.read(arm/'PROTOCOL.json');complete(receipt)
        stats=receipt['stats'];require(receipt['kind']==k and receipt['condition']==c and receipt['round']==2
            and receipt['optimizer_steps']==[1656] and stats['updates_completed']==stats['updates_required']==828
            and receipt['protocol_sha256']==io.hash(arm/'PROTOCOL.json') and protocol['seed']==42
            and protocol['subset_seed']==20261027 and receipt['neural_teacher_forwards']==0 and receipt['PnP_fits']==0,'Ratio completed trial contract differs')
        training.append(dict(experiment='RATIO',kind=k,condition=c,receipt_path=str(arm/'TRAIN_COMPLETE.json'),
            receipt_sha256=io.hash(arm/'TRAIN_COMPLETE.json'),completion=projection(receipt,[
                'complete','kind','condition','round','synthetic_percent','real_unique_count','synthetic_unique_count',
                'protocol_sha256','checkpoint_path','checkpoint_sha256','resume_checkpoint_sha256','teacher_checkpoint_sha256',
                'R1_resume_checkpoint_sha256','R1_resume_SHA_authority','frozen_R2_pseudo_sha256','training_go_sha256',
                'subsets_sha256','exposure_log_sha256','optimizer_steps','neural_teacher_forwards','PnP_fits','evaluation_performed']),
            update_counts=projection(stats,['updates_completed','updates_required','persistent_Adam_steps','epochs']),
            protocol_sha256_declarations=declared_hashes(protocol)))
        seed_evidence.extend(dict(purpose=purpose,seed=seed,path=str(arm/'PROTOCOL.json'),json_pointer=pointer)
            for purpose,seed,pointer in (('ratio training',42,'/seed'),('fixed real/synthetic subset',20261027,'/subset_seed')))
    repro=meta['technical_reproduction'];parity=meta['technical_parity'];complete(repro);require(parity['passed'] is True
        and parity['frames']==45 and parity['reference_points']==346 and parity['training_receipt_sha256']==io.hash(META['technical_reproduction'])
        and parity['image_forwards_completed']==45
        and parity['reproduced_checkpoint_sha256']==repro['checkpoint_sha256'] and parity['pck_count_exact'] is True
        and parity['coverage_exact'] is True and parity['valid_exact'] is True
        and 0<=parity['maximum_native_xy_abs_difference_px']<=.01,'Technical WOOD45 reproduction/parity failed')
    require(repro['condition']=='ORIGINAL_REPRODUCTION' and repro['optimizer_steps']==[1656] and repro['stats']['updates_completed']==828,'Technical reproduction training incomplete')
    require(meta['technical_protocol']['seed']==42 and repro['protocol_sha256']==io.hash(META['technical_protocol']),'Technical reproduction seed/protocol binding differs')
    seed_evidence.append(dict(purpose='technical reproduction training',seed=42,path=str(META['technical_protocol']),json_pointer='/seed'))
    training.append(dict(experiment='technical_reproduction',receipt_path=str(META['technical_reproduction']),
        completion=projection(repro,['complete','kind','condition','round','checkpoint_path','checkpoint_sha256',
            'reference_checkpoint_sha256','resume_checkpoint_sha256','teacher_checkpoint_sha256','optimizer_steps']),
        source_sha256_declarations=declared_hashes(repro),parity_receipt_path=str(META['technical_parity'])))
    for key in ('main_bootstrap','comparison_bootstrap','ratio_bootstrap'):
        value=meta[key];require(value['seed']==20261026 and value['replicates']==10000,'Actual bootstrap seed/replicate differs')
        seed_evidence.append(dict(purpose=key,seed=value['seed'],replicates=value['replicates'],path=str(META[key]),json_pointer='/seed'))
    for key in ('main_infer','WOOD80_infer','main_scores','merged_scores','comparison_complete','ratio_infer_wood','ratio_infer_all','ratio_scores','future_teacher','filter_infer','filter_equivalence','filter_score','filter_statistics','ratio_exposures'):
        complete(meta[key],key in ('merged_scores','comparison_complete','filter_infer','filter_equivalence','filter_score','filter_statistics','ratio_exposures'))
    require(meta['finish_queue']['count_ratio_trials_completed']==15 and meta['finish_queue']['primary_frames']==372
        and meta['finish_queue']['WOOD_frames']==125 and meta['finish_queue']['future_R3_probe_completed'] is True,'Finish15/372 scope differs')
    require(meta['merged_scores']['main_groups']==51 and meta['merged_scores']['frame_rows']==6324
        and meta['merged_scores']['primary_reference_points']==2891,'Combined primary completion differs')
    require(meta['ratio_scores']['main_groups']==15 and meta['ratio_scores']['main_CI_rows']==135
        and meta['ratio_scores']['GREEN_manual_CI_rows']==45 and meta['future_teacher']['metric_rows']==20
        and meta['future_teacher']['future_R3_probe_only'] is True,'Ratio/probe completed catalogue differs')
    require(meta['filter_workflow']['new_forwards']==320 and meta['filter_infer']['new_forwards']==meta['filter_infer']['attempts']==meta['filter_infer']['completed']==320
        and meta['filter_score']['component_rows']==35 and meta['filter_score']['T3_stage_rows']==165,'Additional320/diagnostic catalogue differs')
    require(meta['main_infer']['available_checkpoints']==30 and meta['main_infer']['unavailable_not_inferred']==0
        and meta['WOOD80_infer']['groups']==17 and meta['WOOD80_infer']['forward_log_sha256']==io.hash(JOURNALS['WOOD80_forward_events']),
        'Completed MAIN/authorized80 model scope differs')
    for key,models,actual in (('ratio_infer_wood',5,1250),('ratio_infer_all',15,2470)):
        value=meta[key];require(value['models']==models and value['actual_image_forward_attempts']==actual
            and sum(r['actual_image_forwards_completed'] for r in value['current_run'])==actual,
            'Actual ratio invocation versus reused-cache forward scope differs')
    require(meta['future_teacher']['neural_forwards']==0 and meta['filter_score']['NN_forwards']==0
        and meta['filter_score']['training_updates']==0 and meta['filter_score']['PnP_fits']==0,'Cached diagnostic phase scope differs')
    forwards={'MAIN_evaluation':forward_counts(io.read(JOURNALS['main_forward_events']),('group','id'),2920),
        'authorized_WOOD80_evaluation':forward_counts(io.read(JOURNALS['WOOD80_forward_events']),('group','id'),1360,'completed'),
        'RATIO_R2_two_view_evaluation':forward_counts(io.read(JOURNALS['ratio_forward_events']),('group','id','view'),3720),
        'added_WOOD80_filter_diagnostics':forward_counts(io.read(JOURNALS['filter_forward_events']),('teacher','id','view'),320)}
    require(len(records)==60 and len(training)==31,'Completed command/training scope differs')
    require(len({r['record'][next(k for k in ('log','stdout_log','stdout_path') if k in r['record'])] for r in records})==60,'Command stdout duplicated')
    declarations=[dict(source_path=str(p),sha256=io.hash(p),declared_sha256_fields=declared_hashes(io.read(p))) for p in META.values()]
    for p in FROZEN:io.hash(p)
    require(set(io.used)==set(REQUIRED),'One exact required source/receipt/stdout was not consumed')
    return dict(schema='completed_DOPE_execution_run_log_v2',complete=True,utc=utc(),
        actual_command_records=records,repair_history=repair,metadata_adapter_provenance=adapter_chain,
        actual_inner_ratio_invocations=dict(source_path=str(JOURNALS['ratio_driver']),records=ratio_calls),
        completed_training=training,seed_evidence=seed_evidence,actual_forward_event_counts=forwards,
        phase_scope=dict(MAIN_training='15 independent arms; two rounds each,1656 updates per arm',
            RATIO_training='15 additional R2 trials,828 updates each; same frozen R1 teachers and pseudo-label targets',
            technical_reproduction='one original WOOD R2 reproduction,828 updates; canonical evaluation45 frames only',
            MAIN_evaluation='canonical single-view evaluation; original292 remains unchanged',
            WOOD80_evaluation='17 completed WOOD models on authorized added80,1360 canonical single-view forwards; cached for PRIMARY372',
            RATIO_evaluation='PRIMARY372 own-kind R2 models, original+flip cached once and reused by hypothetical R3 probe',
            added_filter_infer='320 additional neural views on exactly authorized WOOD80; zero training updates',
            filter_select='CPU geometric filtering; no neural forward or training, PnP count not stored',
            filter_score_and_statistics='cached counter/scalar analysis; no training/neural/PnP',
            future_teacher='hypothetical R3 filter probe only; no R3 training; fixed R2 training pseudo-label quality is not varied'),
        declared_input_and_checkpoint_SHA_provenance=declarations,
        input_sha256=dict(io.used),required_input_files=len(REQUIRED),authorization_sha256=dict(io.bootstrap),
        limitations=['59 records directly store returncode0; one main-comparison QA records source-gated completion without a returncode field.',
            'Hashes declared by upstream receipts are preserved with JSON pointers and are not independently rehashed here.',
            'Wall times are recorded command durations; overlapping CPU/GPU stages must not be summed as end-to-end elapsed time.',
            'The failed V1 sealer and original queue terminations are separate repair history, not part of the60 successful command records.',
            'The observed outer-session exit143 is a ROOT observation, not a stored scientific child-command returncode; its cause is not inferred.',
            'This run log covers the explicitly listed completed execution graph, not unrecorded interactive preparation or publication.'],
        private_point_arrays_copied=False,raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,
        reference_or_prediction_cache_reads=0,training_updates=0,neural_forwards=0,PnP_fits=0,published=False)


def write(path,value):
    require(path in (TARGET,QA) and path.resolve()==path and not path.exists(),'Preserve existing run-log output')
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    require(sys.argv[1:]==['--run'],'Explicit --run required')
    require(not TARGET.exists() and not QA.exists(),'Preserve prior/partial run-log artifacts')
    io=Inputs()
    try:
        result=assemble(io);io.unchanged();write(TARGET,result);io.unchanged()
        write(QA,dict(passed=True,complete=True,utc=utc(),run_log_sha256=sha(TARGET),source_sha256=sha(SOURCE),
            actual_command_records=60,stored_returncode_zero_records=59,source_gated_returncode_records=1,
            MAIN_training_arms=15,RATIO_R2_trials=15,technical_reproductions=1,
            actual_forward_event_counts=result['actual_forward_event_counts'],input_sha256=io.used,
            authorization_sha256=io.bootstrap,required_input_files=len(REQUIRED),repair_failed_command_records=1,
            repair_failure_records=4,ROOT_observed_prior_outer_exit_code=143,metadata_adapter_stage_receipts=3,raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,
            reference_or_prediction_cache_reads=0,neural_forwards=0,training_updates=0,PnP_fits=0,published=False))
        print('RUN_LOG_COMPLETE commands60 training15+15+1 metadata_only',flush=True)
    except Exception as error:
        if not QA.exists():write(QA,dict(passed=False,complete=False,utc=utc(),error=repr(error),automatic_retry=False,
            source_sha256=sha(SOURCE),input_sha256=io.used,published=False))
        raise


if __name__=='__main__':main()
