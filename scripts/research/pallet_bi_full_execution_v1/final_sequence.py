"""Root-GO-only sequential postseal S2/F completion and analysis controller.

Import/prepare/checkpoint validation use stdlib metadata only. No production
worker is changed. Child science/GT readers run only after a separately frozen
plan/GO and their own completed prediction gates. Existing partial outputs are
preserved and require root intervention; completed stages are never rerun.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

NS='scripts.research.pallet_bi_full_execution_v1.'
SEEDS=[20261018,20261019,20261020]
POPS={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
BASE=['N0','E0',*[f'D1_H1_MLP_s{s}'for s in SEEDS],'POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR']
F_FINISH_SHA='0b75f02fcd58ab4fcef8586afebc071a4739bfb11ec7c2ed5d872e647adbf7b4'
POST_BANK_SHA='f5fa42629e6848b1650cd90c61a16b2d03f543ae05fe13a58887477316676405'
INDEPENDENT=['INDEPENDENT_ROWS.jsonl.gz','INDEPENDENT_SUMMARY.json','INDEPENDENT_PAIRED.json','INDEPENDENT_VERIFICATION.json']
STAGE_ORDER=['S2_evaluation','S2_independent','S2_bank','S2_selection','S2_core','F_choose','F_finish',
             'F_evaluation','F_independent','F_bank','F_selection','F_core']


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);body=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=body:raise ValueError('Preserved metadata differs: '+str(path))
    else:
        with path.open('x')as stream:stream.write(body)
def bound(path,bindings):
    path=Path(path).resolve();bindings[str(path)]=sha(path);return read(path)
def verify_map(mapping):
    for path,value in mapping.items():
        if sha(path)!=value:raise ValueError('Frozen input/source/artifact changed: '+str(path))
def artifacts(folder,seal,bindings):
    for name,value in seal['artifacts'].items():
        path=Path(folder)/name
        if sha(path)!=value:raise ValueError('Completed artifact changed: '+str(path))
        bindings[str(path.resolve())]=value


def prepare(run,d1,old_run,prior_run,reference_dir,fixed_sets,output):
    """Freeze source/header inputs only; never open FIXED_REFERENCES.json."""
    run,d1,old_run,prior_run,reference_dir,fixed_sets,output=map(lambda p:Path(p).resolve(),
        (run,d1,old_run,prior_run,reference_dir,fixed_sets,output))
    if output.exists():raise FileExistsError('Preparation requires a new controller folder')
    repo=Path(__file__).resolve().parents[3];folder=Path(__file__).parent
    core=read(run/'S2_FULL_UNIT_WEIGHT_FOUR_CPU/PRODUCER_LOCK.json')
    heads=read(run/'S2_UNIT_WEIGHT_FOUR_CPU_H1_AUTOMATION_PLAN.json')
    sources={str((repo/k).resolve()):v for k,v in core['numerical_kernel_source_binding'].items()}
    sources.update(heads['head_source_sha256'])
    for name in ['final_sequence.py','f_finish.py','evaluation.py','independent_check.py','post_bank.py','post_selection.py',
                 'post_core_diagnostics.py','s1_union.py','kernel_extra.py','extra_sources.py','material_finish.py',
                 'image_batch.py','material_assets.py','material_support.py','render_refine.py',
                 'fullnormal_dispatch_four.py','fullnormal_dispatch.py','fullnormal_parallel.py']:
        sources[str((folder/name).resolve())]=sha(folder/name)
    for relative in ['scripts/research/pallet_accuracy_closeout_20261017_v1/evaluation.py',
                     'scripts/research/pallet_accuracy_closeout_20261017_v1/baseline_audit.py',
                     'scripts/research/pallet_statewise_end_to_end_recovery_v1/evaluator.py']:
        sources[str(repo/relative)]=sha(repo/relative)
    if sources[str(folder/'f_finish.py')]!=F_FINISH_SHA or sources[str(folder/'post_bank.py')]!=POST_BANK_SHA:
        raise ValueError('Reviewed finalization/post-bank source differs')
    verify_map(sources);fixed={}
    for path in [run/'CAMPAIGN_PLAN.json',run/'ENVIRONMENT_INITIAL.json',reference_dir/'SCORE_SEAL.json',fixed_sets,
                 run/'F_SELECTION_RULE_CLARIFICATION.json',run/'F_AUXILIARY_MAPPING_IMPLEMENTATION_NOTE.json',
                 run/'F_OBSERVER_POLICY_LOCK_V2.json',run/'F_CORE_PRECOMPUTE_LOCK.json',d1/'PREDICTION_SEAL.json',
                 run/'S2_UNIT_WEIGHT_FOUR_CPU_H1_AUTOMATION_PLAN.json',run/'S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py',
                 run/'F_FOUR_CPU_CONTROLLER_PLAN.json',run/'F_CORE_FOUR_CPU_CONTROLLER.py']:
        fixed[str(path)]=sha(path)
    plan=dict(schema='FINAL_SEQUENCE_EXECUTION_PLAN_V1',run=str(run),d1=str(d1),old_run=str(old_run),prior_run=str(prior_run),
        reference_dir=str(reference_dir),fixed_sets=str(fixed_sets),output=str(output),worktree=str(repo),python=sys.executable,
        sources=sources,fixed_inputs=fixed,numeric_source_binding=core['numerical_kernel_source_binding'],controller_sha256=sha(__file__),stage_order=STAGE_ORDER,
        S2_core=str(run/'S2_FULL_UNIT_WEIGHT_FOUR_CPU'),S2_heads=str(run/'S2_UNIT_WEIGHT_FOUR_CPU_H1'),
        F_core=str(run/'F_CORE_FOUR_CPU'),F_finish=str(run/'F_FINALIZE'),
        S2_waiter_source_sha256=heads['source_sha256'],F_controller_source_sha256=read(run/'F_FOUR_CPU_CONTROLLER_PLAN.json')['source_sha256'],
        resource=dict(sequential_child_workers=1,BLAS_OMP_MKL=1,max_scientific_CPU_workers=8),
        child_H1_fits=6,S2_waiter_owns_nine_fits=True,GT_reader_gate='full238 own prediction seals before evaluation/independent/post_bank',
        source_guard='all frozen sources checked before each stage,including completed-stage adoption;no source guard changes')
    write(output/'PLAN.json',plan);return plan


def prediction(folder,prefix,bindings,real_features=None,synthetic_features=None):
    folder=Path(folder);seal=bound(folder/'PREDICTION_SEAL.json',bindings)
    names=[prefix+'_s'+str(s)for s in SEEDS]
    if (seal.get('complete')is not True or seal.get('rows')!=238 or seal.get('before_reference_decode')is not True
        or seal.get('current_real_reference_reads')!=0 or len(seal.get('ordered_ids',[]))!=238
        or len(set(seal['ordered_ids']))!=238 or set(seal.get('checkpoints',{}))!=set(names)):
        raise ValueError('All238 own predictions and exactly three registered seeds required')
    pp=folder/'PREDICTIONS.jsonl.gz'
    if sha(pp)!=seal['predictions_sha256']:raise ValueError('Prediction bytes changed')
    bindings[str(pp.resolve())]=seal['predictions_sha256']
    training=bound(folder.parent/'training/TRAINING_MANIFEST.json',bindings)
    if (not training.get('complete')or len(training['runs'])!=3 or {x['seed']for x in training['runs']}!=set(SEEDS)
        or sha(folder.parent/'training/TRAINING_MANIFEST.json')!=seal['training_manifest_sha256']):
        raise ValueError('Own three-fit prediction ancestry differs')
    for item in training['runs']:
        if not item['complete']or item['arm']!=prefix or sha(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Own checkpoint changed')
        bindings[str(Path(item['checkpoint']).resolve())]=item['checkpoint_sha256']
    if real_features is not None:
        path,sp=map(Path,real_features);own=bound(sp,bindings)
        if (not own['complete']or own['rows']!=238 or sha(path)!=seal['features_sha256']or sha(sp)!=seal['feature_seal_sha256']
            or sha(path)!=own['head_features_sha256']):raise ValueError('Prediction uses another own real feature bank')
        bindings[str(path.resolve())]=sha(path)
    if synthetic_features is not None:
        path,sp=map(Path,synthetic_features);own=bound(sp,bindings)
        if not own['complete']or own['rows']!=1003 or sha(path)!=own['head_features_sha256']:raise ValueError('Full own synthetic feature bank required')
        for item in training['runs']:
            if item['bindings']['features_sha256']!=sha(path)or item['bindings']['feature_seal_sha256']!=sha(sp):raise ValueError('Own H1 training bank differs')
        bindings[str(path.resolve())]=sha(path)
    return dict(path=str(pp.resolve()),sha256=sha(pp),seal_sha256=sha(folder/'PREDICTION_SEAL.json'))


def s2_ready(plan):
    root=Path(plan['S2_heads']);bindings={};seal=bound(root/'ALL_REAL_PREDICTIONS_SEAL.json',bindings)
    if (not seal.get('complete')or seal['rows_per_arm']!=238 or seal['seeds']!=SEEDS or seal['arms']!=['S2_C0','S2_CP','S2_CB']
        or seal['source_sha256']!=plan['S2_waiter_source_sha256']or seal['current_real_reference_reads']!=0 or not seal['before_new_scoring']):
        raise ValueError('All nine S2 predictions required')
    for arm in ['C0','CP','CB']:
        core=Path(plan['S2_core'])/arm
        prediction(root/arm/'prediction','S2_'+arm,bindings,
            (core/'real/FEATURES.jsonl.gz',core/'real/FEATURE_SEAL.json'),
            (core/'synthetic/FEATURES.jsonl.gz',core/'synthetic/FEATURE_SEAL.json'))
        if seal['prediction_seal_sha256'][arm]!=sha(root/arm/'prediction/PREDICTION_SEAL.json'):raise ValueError('S2 aggregate ancestry changed')
    return bindings


def f_core_ready(plan):
    run,core=Path(plan['run']),Path(plan['F_core']);bindings={}
    done=bound(run/'F_FOUR_CPU_CONTROLLER_COMPLETION.json',bindings)
    summary=bound(core/'SUMMARY.json',bindings);dispatch=bound(core/'DISPATCH_COMPLETION.json',bindings);lock=bound(core/'PRODUCER_LOCK.json',bindings)
    if (not done.get('complete')or not done['unconditional_core_only']or done['H1_fits']!=0 or done['F_choice_calls']!=0
        or summary['status']!='COMPLETE_NEW'or summary['missing_rows']or summary['technical_failures']
        or not dispatch['active_rows_drained_before_exit']or dispatch['submitted_jobs']!=1241 or dispatch['unsubmitted_jobs']!=0
        or done['source_sha256']!=plan['F_controller_source_sha256']):
        raise ValueError('Both full F cores and exited production pool required')
    for key,name in [('producer_summary_sha256','SUMMARY.json'),('producer_dispatch_sha256','DISPATCH_COMPLETION.json'),('producer_lock_sha256','PRODUCER_LOCK.json')]:
        if done[key]!=sha(core/name):raise ValueError('F controller ancestry differs')
    if (dispatch['summary_sha256']!=sha(core/'SUMMARY.json')or lock['arms']!=['F_COMMON','F_BI']
        or lock['workers']!=4 or lock['schema']!='FULL_NORMAL_FOUR_CPU_DISPATCH_V1'
        or lock['numerical_kernel_source_binding']!=plan['numeric_source_binding']):raise ValueError('F core identity differs')
    for arm in ['F_COMMON','F_BI']:
        for domain,n in [('real',238),('synthetic',1003)]:
            folder=core/arm/domain;seal=bound(folder/'FEATURE_SEAL.json',bindings)
            if not seal['complete']or seal['rows']!=n or seal['producer_lock_sha256']!=sha(core/'PRODUCER_LOCK.json')or sha(folder/'FEATURES.jsonl.gz')!=seal['head_features_sha256']:
                raise ValueError('Full F own feature bank differs')
            bindings[str((folder/'FEATURES.jsonl.gz').resolve())]=seal['head_features_sha256']
    return bindings


def extra_declarations(output,feature_seal):
    """Copy exact method filters; compare with the actual union metadata."""
    output=Path(output);path=output/'real_EXTRA_INPUTS.json'
    declared=deepcopy(read(path)['extra_inputs'])if path.exists()else[]
    own=read(feature_seal);binding=own.get('dependencies',{}).get('extra_producer_binding',{})
    if binding and 'ordered_inputs'not in binding:raise ValueError('Untracked legacy union extra parent')
    ordered=binding.get('ordered_inputs',[])
    if len(declared)!=len(ordered):raise ValueError('Declared/actual union extra parent count differs')
    for source,actual in zip(declared,ordered):
        methods=source.get('methods')
        if (not isinstance(methods,list)or not methods or len(methods)!=len(set(methods))
            or Path(source['path']).resolve()!=Path(actual['extra_path']).resolve()
            or Path(source['seal']).resolve()!=Path(actual['producer_seal_path']).resolve()
            or sorted(methods)!=actual['admitted_methods']):raise ValueError('Exact union extra paths/methods differ')
        if sha(source['path'])!=actual['extra_sha256']or sha(source['seal'])!=actual['producer_seal_sha256']:
            raise ValueError('Actual union extra producer bytes changed')
    if declared:
        declaration=binding.get('declaration_manifest',{})
        if declaration.get('path')!=str(path.resolve())or declaration.get('sha256')!=sha(path):raise ValueError('Actual union declaration differs')
    return declared


def manifest(plan,stage):
    run=Path(plan['run']);arms={}
    if stage=='S2':
        for name in ['C0','CP','CB']:
            folder=Path(plan['S2_core'])/name/'real'
            arms['S2_'+name]=dict(backend='fullnormal',features=str(folder/'FEATURES.jsonl.gz'),feature_seal=str(folder/'FEATURE_SEAL.json'),
                generated_feature_seal=str(folder/'FEATURE_SEAL.json'),result_root=str(folder),prediction_dir=str(Path(plan['S2_heads'])/name/'prediction'))
    else:
        output=Path(plan['F_finish']);receipt=read(output/'FINISH_RECEIPT.json')
        for arm in ['F_COMMON','F_BI']:
            actual=receipt['feature_inputs'][arm]['real'];folder=Path(plan['F_core'])/arm/'real'
            arms[arm]=dict(backend='fullnormal',features=actual['path'],feature_seal=actual['seal'],generated_feature_seal=str(folder/'FEATURE_SEAL.json'),
                result_root=str(folder),prediction_dir=str(output/arm/'prediction'),extra_rows=extra_declarations(output,actual['seal']))
    path=run/(stage+'_POST_BANK_MANIFEST.json');write(path,dict(schema='FINAL_SEQUENCE_FULL_POPULATION_MANIFEST_V1',arms=arms))
    return path


def native(plan,key,mp=None):
    """Authenticate a native completed stage; return its immutable file map."""
    run=Path(plan['run']);bindings={};prefix=key.split('_')[0];module=key.split('_',1)[1]
    source=lambda name:plan['sources'][str(Path(__file__).parent/(name+'.py'))]
    if key=='F_choose':
        value=bound(run/'F_SYSTEM_CHOICE.json',bindings)
        if not value.get('complete')or value['controller_sha256']!=source('f_finish')or value['primary_configuration_ranking']is not False or not value['before_F_optional_union_and_H1']:
            raise ValueError('Completed frozen F choice differs')
        verify_map(value['input_bindings']);bindings.update(value['input_bindings'])
        for path in [Path(plan['S2_core'])/'SUMMARY.json',Path(plan['S2_heads'])/'ALL_REAL_PREDICTIONS_SEAL.json']:
            if value['input_bindings'].get(str(path.resolve()))!=sha(path):raise ValueError('F choice uses another S2 bank')
        return bindings
    if key=='F_finish':
        folder=Path(plan['F_finish']);value=bound(folder/'FINISH_RECEIPT.json',bindings)
        if (not value.get('complete')or value['H1_fits']!=6 or value['controller_sha256']!=source('f_finish')
            or value['F_choice_sha256']!=sha(run/'F_SYSTEM_CHOICE.json')or value['current_real_reference_reads']!=0
            or not value['all_real_predictions_sealed_before_reference_decode']):raise ValueError('Completed F own-six-fit finalization differs')
        verify_map(value['sealed_artifact_paths']);bindings.update(value['sealed_artifact_paths'])
        for field,name in [('actual_command_ledger_sha256','POSTPROCESS_LEDGER.jsonl'),('postprocess_source_lock_sha256','POSTPROCESS_SOURCE_LOCK.json')]:
            if value[field]!=sha(folder/name):raise ValueError('F finalization execution/source lock changed')
            bindings[str((folder/name).resolve())]=value[field]
        closure=read(folder/'POSTPROCESS_SOURCE_LOCK.json')['sources']
        verify_map({str(Path(plan['worktree'])/p):v for p,v in closure.items()})
        for arm in ['F_COMMON','F_BI']:
            real,synthetic=value['feature_inputs'][arm]['real'],value['feature_inputs'][arm]['synthetic']
            prediction(folder/arm/'prediction',arm,bindings,(real['path'],real['seal']),(synthetic['path'],synthetic['seal']))
        return bindings
    names=dict(evaluation='EVALUATION',independent='INDEPENDENT',bank='BANK_ANALYSIS',selection='SELECTION_DIAGNOSTIC',core='CORE_DIAGNOSTICS')
    folder=run/(prefix+'_'+names[module]);terminal=dict(evaluation='EVALUATION_SEAL.json',independent='INDEPENDENT_VERIFICATION.json',
        bank='ANALYSIS_SEAL.json',selection='SELECTION_DIAGNOSTIC_SEAL.json',core='DIAGNOSTIC_SEAL.json')[module]
    value=bound(folder/terminal,bindings);ev=run/(prefix+'_EVALUATION');bank=run/(prefix+'_BANK_ANALYSIS')
    if module=='independent':
        canonical=read(ev/'EVALUATION_SEAL.json')
        if (value.get('PASS')is not True or value.get('errors')!=[]or value['rows']!=238 or value['populations']!=POPS
            or value['source_sha256']!=source('independent_check')or value['canonical_evaluation_seal_sha256']!=sha(ev/'EVALUATION_SEAL.json')
            or value['arms']!=canonical['arms']or value['reference_sha256']!=canonical['reference_sha256']
            or value['prediction_bindings']!=canonical['prediction_bindings']or not value['success_classifications_exact']):raise ValueError('Independent verification ancestry differs')
        for name in INDEPENDENT:bindings[str((folder/name).resolve())]=sha(folder/name)
        return bindings
    if value.get('complete')is not True:raise ValueError('Incomplete native stage')
    required=dict(evaluation={'SCORED_ROWS.jsonl.gz','SUMMARY.json','PAIRED.json'},
        bank={'BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json','ALL_FAILURES.csv','INPUT_BINDINGS.json'},
        selection={'ACTION_RANK_ROWS.jsonl.gz','SUMMARY.json','FIXED_COHORT_ACTIONS.json'},
        core={'CORE_DIAGNOSTIC_ROWS.jsonl.gz','SUMMARY_BY_POPULATION_AND_STRATUM.json','INPUT_BINDINGS.json'})[module]
    if set(value['artifacts'])!=required:raise ValueError('Complete native artifact set differs')
    artifacts(folder,value,bindings)
    if module=='evaluation':
        access=bound(folder/'REFERENCE_ACCESS.json',bindings)
        expected=expected_predictions(plan,prefix)
        wanted=BASE+[arm+'_s'+str(s)for arm in (['S2_C0','S2_CP','S2_CB']if prefix=='S2'else['F_COMMON','F_BI'])for s in SEEDS]
        if (value['status']!='COMPLETE_SAVED_POSE_SCORING'or value['populations']!=POPS or value['arms']!=wanted
            or value['prediction_bindings']!=expected or access['prediction_bindings']!=expected
            or access['scorer_sha256']!=source('evaluation')or access['reference_sha256']!=value['reference_sha256']):raise ValueError('Canonical evaluation source/predictions differ')
    else:
        if value['source_sha256']!=source('post_'+('core_diagnostics'if module=='core'else module)):
            raise ValueError('Analysis source differs')
        if module in ('bank','selection')and(value['manifest_sha256']!=sha(mp)or value['fixed_sets_sha256']!=sha(plan['fixed_sets'])):
            raise ValueError('Analysis manifest/fixed sets differ')
        if module=='bank':
            access=bound(folder/'REFERENCE_ACCESS.json',bindings)
            expected=[prediction(spec['prediction_dir'],arm,{})for arm,spec in read(mp)['arms'].items()]+[base_prediction(plan)]
            if (access['prediction_bindings']!=expected or access['source_sha256']!=source('post_bank')
                or access['reference_sha256']!=value['reference_sha256']):raise ValueError('Bank baseline/own prediction ancestry differs')
            inputs=read(folder/'INPUT_BINDINGS.json')
            if set(inputs)!=set(read(mp)['arms']):raise ValueError('Post-bank arm population differs')
            for arm,spec in read(mp)['arms'].items():
                if inputs[arm]['prediction']!=prediction(spec['prediction_dir'],arm,{}):raise ValueError('Bank own prediction binding differs')
                for field in ['features','feature_seal']:
                    if Path(inputs[arm][field]['path']).resolve()!=Path(spec[field]).resolve()or inputs[arm][field]['sha256']!=sha(spec[field]):raise ValueError('Bank feature ancestry differs')
                generated=spec.get('generated_feature_seal',spec['feature_seal']);gb=inputs[arm]['generated_feature_seal']
                if Path(gb['path']).resolve()!=Path(generated).resolve()or gb['sha256']!=sha(generated):raise ValueError('Generated bank ancestor differs')
                extra=inputs[arm]['extras'];declared=spec.get('extra_rows',[])
                if len(extra)!=len(declared):raise ValueError('Post-bank extra parents differ')
                for item,decl in zip(extra,declared):
                    if (Path(item['extra_path']).resolve()!=Path(decl['path']).resolve()or item['extra_sha256']!=sha(decl['path'])
                        or Path(item['producer_seal_path']).resolve()!=Path(decl['seal']).resolve()or item['producer_seal_sha256']!=sha(decl['seal'])):
                        raise ValueError('Post-bank extra producer ancestry differs')
                    if item.get('declared_methods')!=decl['methods']or not item.get('unused_parent_methods_excluded')or not item.get('selected_rows_payload_sha256'):
                        raise ValueError('Post-bank exact extra method filter differs')
        if module=='selection':
            expected={arm:entry['prediction']for arm,entry in read(bank/'INPUT_BINDINGS.json').items()}
            if value['bank_analysis_seal_sha256']!=sha(bank/'ANALYSIS_SEAL.json')or value['prediction_bindings']!=expected:raise ValueError('Selection bank ancestry differs')
        if module=='core':
            expected=dict(manifest_sha256=sha(mp),post_bank_analysis_seal_sha256=sha(bank/'ANALYSIS_SEAL.json'),
                post_bank_input_bindings_sha256=sha(bank/'INPUT_BINDINGS.json'),post_bank_coverage_sha256=sha(bank/'BANK_COVERAGE_ROWS.jsonl.gz'))
            if (value['inputs']!=expected or value['rows_per_arm']!=238 or value['populations']!={'ALL238':238,**POPS}
                or value['seeds']!=list(map(str,SEEDS))or value['arms']!=list(read(mp)['arms'])):
                raise ValueError('Core diagnostic ancestry/population differs')
    if module in ('evaluation','bank'):
        expected_ref=read(Path(plan['reference_dir'])/'SCORE_SEAL.json')['artifacts']['FIXED_REFERENCES.json']
        if value['reference_sha256']!=expected_ref:raise ValueError('Canonical reference seal differs')
    return bindings


def base_prediction(plan):
    folder=Path(plan['d1']);seal=read(folder/'PREDICTION_SEAL.json');pp=folder/'PREDICTIONS.jsonl.gz';value=sha(pp)
    if not seal.get('complete')or value not in [seal.get('predictions_sha256'),seal.get('prediction_sha256'),*seal.get('artifacts',{}).values()]:raise ValueError('Completed baseline prediction differs')
    return dict(path=str(pp),sha256=value,seal_sha256=sha(folder/'PREDICTION_SEAL.json'))
def expected_predictions(plan,prefix):
    own=[(Path(plan['S2_heads'])/a/'prediction','S2_'+a)for a in ['C0','CP','CB']]if prefix=='S2'else[(Path(plan['F_finish'])/a/'prediction',a)for a in ['F_COMMON','F_BI']]
    return [base_prediction(plan)]+[prediction(path,arm,{})for path,arm in own]


def guard(plan):verify_map(plan['sources']);verify_map(plan['fixed_inputs'])
def wait_for(path,validator):
    while True:
        if Path(path).exists():
            try:return validator()
            except (json.JSONDecodeError,FileNotFoundError):pass
        print(json.dumps(dict(event='WAIT',path=str(path),scientific_calls=0)),flush=True);time.sleep(20)
def stage(output,key,argv,input_bindings,validator,terminal,output_dir,runner):
    """Adopt only authenticated completion; refuse partial reruns."""
    cp=Path(output)/'checkpoints'/(key+'.json');identity=dict(argv=list(map(str,argv)),input_bindings=input_bindings)
    if cp.exists():
        previous=read(cp)
        if previous['identity']!=identity:raise ValueError('Stage checkpoint input/argv differs')
        verify_map(previous['completed_files']);current=validator()
        if current!=previous['completed_files']:raise ValueError('Completed checkpoint ancestry differs')
        return 'SKIP'
    if Path(terminal).exists():
        files=validator();adopted=True
    else:
        if Path(output_dir).exists()and any(Path(output_dir).iterdir()):raise RuntimeError('Partial output preserved;root review required: '+str(output_dir))
        runner(key,argv);files=validator();adopted=False
    write(cp,dict(complete=True,identity=identity,completed_files=files,adopted_without_rerun=adopted));return 'ADOPT'if adopted else'COMPLETE'


def append_ledger(output,value):
    with (Path(output)/'ACTUAL_COMMAND_LEDGER.jsonl').open('a')as stream:
        stream.write(json.dumps(value,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def execute(key,argv,output,worktree):
    output=Path(output)
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    append_ledger(output,dict(event='START',phase=key,argv=list(map(str,argv)),epoch=time.time()))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(worktree))
    with (output/(key+'.log')).open('a')as stream:code=subprocess.call(list(map(str,argv)),cwd=worktree,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    append_ledger(output,dict(event='COMPLETE'if code==0 else'TECHNICAL_FAILURE',phase=key,argv=list(map(str,argv)),return_code=code,
        wall_seconds=time.monotonic()-started,child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,epoch=time.time()))
    if code:raise RuntimeError('Child failed;all outputs/logs preserved: '+key)


def completed_sequence(plan_path,GO_path,plan):
    """Return a completed controller unchanged, before any ledger write."""
    output=Path(plan['output']);seal=read(output/'FINAL_SEQUENCE_SEAL.json')
    if (seal.get('complete')is not True or seal['plan_sha256']!=sha(plan_path)or seal['ROOT_GO_sha256']!=sha(GO_path)
        or seal['source_sha256']!=sha(__file__)or seal['stage_order']!=STAGE_ORDER
        or seal['actual_command_ledger_sha256']!=sha(output/'ACTUAL_COMMAND_LEDGER.jsonl')):
        raise ValueError('Completed final sequence binding differs')
    wanted={key+'.json'for key in STAGE_ORDER}
    if set(seal['checkpoints'])!=wanted:raise ValueError('All12 completed stage checkpoints required')
    for key in STAGE_ORDER:
        path=output/'checkpoints'/(key+'.json')
        if sha(path)!=seal['checkpoints'][path.name]:raise ValueError('Completed stage checkpoint changed')
        cp=read(path)
        if not cp['complete']or cp['identity']['input_bindings']['PLAN.json']!=sha(plan_path)or cp['identity']['input_bindings']['ROOT_GO']!=sha(GO_path):
            raise ValueError('Completed checkpoint input binding differs')
        verify_map({p:v for p,v in cp['identity']['input_bindings'].items()if Path(p).is_absolute()})
        verify_map(cp['completed_files'])
        mp=Path(plan['run'])/(key.split('_')[0]+'_POST_BANK_MANIFEST.json')if key not in ('F_choose','F_finish')else None
        if native(plan,key,mp)!=cp['completed_files']:raise ValueError('Completed native stage ancestry differs')
    return seal


def run(plan_path,GO_path):
    plan=read(plan_path);go=read(GO_path)
    if (go.get('approved')is not True or go.get('plan_sha256')!=sha(plan_path)or go.get('controller_sha256')!=sha(__file__)
        or go.get('F_H1_fits_authorized')!=6 or go.get('S2_H1_fits_from_this_controller')!=0
        or go.get('max_scientific_CPU_workers')!=8):raise PermissionError('Reviewed plan/source-bound ROOT GO required')
    guard(plan);output=Path(plan['output']);runroot=Path(plan['run'])
    if (output/'FINAL_SEQUENCE_SEAL.json').exists():return completed_sequence(plan_path,GO_path,plan)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'OWNER.lock').open('a+')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(output/'ROOT_GO_BINDING.json',dict(path=str(Path(GO_path).resolve()),sha256=sha(GO_path),plan_sha256=sha(plan_path),controller_sha256=sha(__file__)))
        append_ledger(output,dict(event='CONTROLLER_START',argv=[v.decode()for v in Path('/proc/self/cmdline').read_bytes().split(b'\0')if v],
            plan_sha256=sha(plan_path),ROOT_GO_sha256=sha(GO_path),source_sha256=sha(__file__),epoch=time.time()))
        s2=wait_for(Path(plan['S2_heads'])/'ALL_REAL_PREDICTIONS_SEAL.json',lambda:s2_ready(plan))
        def invoke(key,argv,mp,terminal,folder,inputs):
            guard(plan);inputs={**inputs,'PLAN.json':sha(plan_path),'ROOT_GO':sha(GO_path)}
            action=stage(output,key,argv,inputs,lambda:native(plan,key,mp),terminal,folder,
                lambda k,a:execute(k,a,output,plan['worktree']))
            append_ledger(output,dict(event='CHECKPOINT_ACTION',phase=key,action=action,argv=list(map(str,argv)),
                checkpoint_sha256=sha(output/'checkpoints'/(key+'.json')),epoch=time.time()))
            return action
        def analysis(prefix,inputs):
            guard(plan);mp=manifest(plan,prefix);inputs={**inputs,str(mp):sha(mp)};python=plan['python']
            folder=runroot/(prefix+'_EVALUATION');own=[str(Path(plan['S2_heads'])/a/'prediction')for a in ['C0','CP','CB']]if prefix=='S2'else[str(Path(plan['F_finish'])/a/'prediction')for a in ['F_COMMON','F_BI']]
            commands=[('evaluation',[python,'-B','-m',NS+'evaluation','score','--d1',plan['d1'],'--reference-dir',plan['reference_dir'],'--output',str(folder),'--additional',*own],folder,'EVALUATION_SEAL.json'),
                ('independent',[python,'-B','-m',NS+'independent_check','--evaluation-dir',str(folder),'--reference-dir',plan['reference_dir'],'--output',str(runroot/(prefix+'_INDEPENDENT'))],runroot/(prefix+'_INDEPENDENT'),'INDEPENDENT_VERIFICATION.json'),
                ('bank',[python,'-B','-m',NS+'post_bank','--manifest',str(mp),'--d1',plan['d1'],'--reference-dir',plan['reference_dir'],'--fixed-sets',plan['fixed_sets'],'--output',str(runroot/(prefix+'_BANK_ANALYSIS'))],runroot/(prefix+'_BANK_ANALYSIS'),'ANALYSIS_SEAL.json'),
                ('selection',[python,'-B','-m',NS+'post_selection','--manifest',str(mp),'--bank-analysis',str(runroot/(prefix+'_BANK_ANALYSIS')),'--fixed-sets',plan['fixed_sets'],'--output',str(runroot/(prefix+'_SELECTION_DIAGNOSTIC'))],runroot/(prefix+'_SELECTION_DIAGNOSTIC'),'SELECTION_DIAGNOSTIC_SEAL.json'),
                ('core',[python,'-B','-m',NS+'post_core_diagnostics','--manifest',str(mp),'--post-bank',str(runroot/(prefix+'_BANK_ANALYSIS')),'--output',str(runroot/(prefix+'_CORE_DIAGNOSTICS'))],runroot/(prefix+'_CORE_DIAGNOSTICS'),'DIAGNOSTIC_SEAL.json')]
            for name,argv,destination,seal in commands:
                invoke(prefix+'_'+name,argv,mp,destination/seal,destination,inputs)
                inputs={**inputs,**native(plan,prefix+'_'+name,mp)}
            return inputs
        inputs=analysis('S2',s2)
        argv=[plan['python'],'-B','-m',NS+'f_finish','choose','--run',plan['run'],'--s2-core',plan['S2_core'],'--s2-heads',plan['S2_heads']]
        invoke('F_choose',argv,None,runroot/'F_SYSTEM_CHOICE.json',output/'choose_partial_guard',inputs)
        core=wait_for(runroot/'F_FOUR_CPU_CONTROLLER_COMPLETION.json',lambda:f_core_ready(plan))
        argv=[plan['python'],'-B','-m',NS+'f_finish','finish','--run',plan['run'],'--d1',plan['d1'],'--old-run',plan['old_run'],'--prior-run',plan['prior_run'],'--core',plan['F_core'],'--output',plan['F_finish']]
        invoke('F_finish',argv,None,Path(plan['F_finish'])/'FINISH_RECEIPT.json',Path(plan['F_finish']),{**core,**native(plan,'F_choose')})
        analysis('F',native(plan,'F_finish'))
        write(output/'FINAL_SEQUENCE_SEAL.json',dict(complete=True,scope='S2/F analysis andown6Ffits only;figures/contrasts/export/report remain root work',
            plan_sha256=sha(plan_path),ROOT_GO_sha256=sha(GO_path),source_sha256=sha(__file__),stage_order=STAGE_ORDER,
            checkpoints={p.name:sha(p)for p in sorted((output/'checkpoints').glob('*.json'))},actual_command_ledger_sha256=sha(output/'ACTUAL_COMMAND_LEDGER.jsonl')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare')
    for name in ['run','d1','old-run','prior-run','reference-dir','fixed-sets','output']:p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--root-GO',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.run,args.d1,args.old_run,args.prior_run,args.reference_dir,args.fixed_sets,args.output)
    else:run(args.plan,args.root_GO)
