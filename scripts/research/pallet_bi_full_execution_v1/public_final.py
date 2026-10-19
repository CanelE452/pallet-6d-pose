"""Final exact-file publication policy; standard library and saved metadata only.

This module never produces scientific results. It refuses a final allowlist
until the real terminal seals,33 fit receipts and11 prediction seals exist.
"""
from __future__ import annotations
import ast
import csv
import json
import math
import io
from pathlib import Path
from . import public_package as P

SCHEMA='EXPLICIT_FINAL_PUBLIC_PACKAGE_ALLOWLIST_V1'
PUBLICATION_STATUS='PREPARED_NOT_YET_COMMITTED_OR_PUSHED'
BASE='3553202a3682faeb7be2e7df051b230683965382'
SPEC='FINAL_PUBLIC_SELECTION_SPEC_V1'
SEEDS=(20261018,20261019,20261020)
ARMS=('S1_UNION','S2_C0','S2_CP','S2_CB','S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE','F_COMMON','F_BI')
POPS={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
STATUSES={'COMPLETE_NEW','COMPLETE_EXACT_REUSE','DIAGNOSTIC_ONLY','BLOCKED_INPUT','BLOCKED_TECHNICAL','INCOMPLETE','NOT_APPLICABLE_WITH_REASON'}
NEW_GALLERIES=('S2','F','F_BI_VS_COMMON','S2_CP_VS_C0','S2_CB_VS_CP')
GROUPS=('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')
REPORT_FILES={'FULL40_PRIMARY_METRICS.csv','REPRESENTATIVE_PRIMARY_METRICS.csv','PAIRED30_PRIMARY_COMPACT.csv','FIXED_COHORTS.csv','FIXED_COHORTS.json'}
REPORT_DOCUMENTS={'FINAL_REPORT_KO.md','S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md'}
REPORT_METADATA_DOCUMENTS={'METHOD_DIFFERENCE_LEDGER.json','METHOD_DIFFERENCE_LEDGER.csv','STAGE_STATUS.json','STAGE_STATUS.csv'}
POSTBANK_PATHS={'S1':'POST_BANK_S1_VALIDATION_V2','S3_S4':'S3_S4_BANK_ANALYSIS','S2':'S2_BANK_ANALYSIS','F':'F_BANK_ANALYSIS'}
SEQUENCE_STAGES=('S2_evaluation','S2_independent','S2_bank','S2_selection','S2_core','F_choose',
 'F_finish','F_evaluation','F_independent','F_bank','F_selection','F_core')
DERIVED_STAGES=('contrasts','export','gallery_S2','gallery_F','gallery_F_BI_VS_COMMON',
 'gallery_S2_CP_VS_C0','gallery_S2_CB_VS_CP','scatter')
SAFE_DOCUMENTS={'FINAL_REPORT_KO.md','METHODOLOGY_KO.md','STRUCTURE_REVIEW_NOTES_KO.md','REPRODUCE.md',
 'STAGE_STATUS.json','STAGE_STATUS.csv','METHOD_DIFFERENCE_LEDGER.json','METHOD_DIFFERENCE_LEDGER.csv',
 'REQUIREMENTS_MATRIX.json','REQUIREMENTS_MATRIX.csv','CACHE_DEPENDENCIES.json','CACHE_DEPENDENCIES.csv',
 'EXECUTION_INDEX.json','EXECUTION_INDEX.csv','PROVENANCE_INDEX.json','PROVENANCE_INDEX.csv',
 'D_PUBLIC_SCALARS_README.md','MASTER_COSTS_README.md','S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md'}


def require(value,message):
    if not value:raise ValueError(message)


def scalar_tree(value):
    """Scalar export whitelist: no matrix/list-of-vector payloads or NaN."""
    P.guard_json(value)
    if isinstance(value,dict):
        for item in value.values():scalar_tree(item)
    elif isinstance(value,list):
        require(not any(isinstance(item,list)for item in value),'Numeric vector/matrix archive is not a public scalar table')
        for item in value:scalar_tree(item)
    else:
        require(value is None or type(value)in(str,bool,int,float),'Non-scalar public export value')
        if isinstance(value,float):require(math.isfinite(value),'Nonfinite public scalar')


def path_inside(root,relative):
    relative=P.clean_relative(relative);root=Path(root).resolve();path=(root/relative).resolve()
    require(root in path.parents,'Input escaped declared root')
    return path


class Inputs:
    def __init__(self):self.hashes={}
    def bind(self,path,digest=None):
        path=P.no_symlink(path);actual=P.sha(path)
        require(digest is None or digest==actual,'Input hash differs: '+path.name)
        require(str(path)not in self.hashes or self.hashes[str(path)]==actual,'Input changed during final selection')
        self.hashes[str(path)]=actual;return path
    def read(self,path,digest=None):return P.read(self.bind(path,digest))
    def stable(self):
        require(all(P.sha(path)==digest for path,digest in self.hashes.items()),'Final-selection input changed')


def fit_accounts(run,inputs):
    path=inputs.bind(run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl')
    rows=[json.loads(line)for line in path.read_text().splitlines()if line.strip()]
    wanted={(arm,seed)for arm in ARMS for seed in SEEDS}
    require(len(rows)==33 and {(r['arm'],r['seed'])for r in rows}==wanted,'Final publication requires actual33 registered fits,not planned or partial fits')
    result=[]
    for row in rows:
        require(row['registered_before_optimizer']is True,'Fit was not registered before optimization')
        folder=Path(row['output'])/f"{row['arm']}_s{row['seed']}";receipt=inputs.read(folder/'RUN_RECEIPT.json')
        require(receipt['complete']is True and all(receipt[k]==row[k]for k in ('arm','seed','attempt','bindings')),'Completed fit/reservation binding differs')
        require(receipt['actual_new_training_runs']==1 and receipt['parameters']==16321,'Wrong H1 fit contract')
        for name,key in [('best.pt','checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('HISTORY.json','history_sha256')]:inputs.bind(folder/name,receipt[key])
        for source,digest in row['bindings']['sources'].items():inputs.bind(source,digest)
        result.append({k:receipt[k]for k in ('arm','seed','attempt','optimizer_steps','training_frame_exposures','resumed_same_optimizer','checkpoint_sha256','last_checkpoint_sha256','history_sha256')}|
            dict(receipt_sha256=P.sha(folder/'RUN_RECEIPT.json'),input_hashes={k:v for k,v in row['bindings'].items()if k!='sources'}))
    return result


def prediction_paths(run):
    return {'S1_UNION':run/'s1_predictions',**{a:run/'s3_s4_predictions'/a for a in ARMS[4:9]},
        **{'S2_'+a:run/'S2_UNIT_WEIGHT_FOUR_CPU_H1'/a/'prediction'for a in ('C0','CP','CB')},
        **{a:run/'F_FINALIZE'/a/'prediction'for a in ('F_COMMON','F_BI')}}


def predictions(run,inputs,fits):
    fitmap={(r['arm'],r['seed']):r for r in fits};result=[];ordered=None
    for arm,folder in prediction_paths(run).items():
        seal=inputs.read(folder/'PREDICTION_SEAL.json')
        require(seal['complete']is True and seal['rows']==238 and seal['before_reference_decode']is True and seal['current_real_reference_reads']==0,'All238 pre-reference predictions required')
        ids=seal['ordered_ids'];require(len(ids)==len(set(ids))==238,'Prediction identity denominator differs')
        if ordered is not None:require(ordered==ids,'Cross-arm prediction order differs')
        ordered=ids
        require(set(seal['checkpoints'])=={arm+'_s'+str(seed)for seed in SEEDS},'Exactly three saved prediction seeds required')
        for seed in SEEDS:require(seal['checkpoints'][arm+'_s'+str(seed)]['checkpoint_sha256']==fitmap[(arm,seed)]['checkpoint_sha256'],'Prediction checkpoint differs from actual completed fit')
        require(seal['tiny_head_forward_calls']==714 and seal['new_backbone_forwards']==0,'Head/backbone forward units differ')
        # Predictions are hash authenticated only,never JSON/decompressed.
        inputs.bind(folder/'PREDICTIONS.jsonl.gz',seal['predictions_sha256'])
        result.append(dict(arm=arm,rows=238,seeds=list(SEEDS),prediction_seal_sha256=P.sha(folder/'PREDICTION_SEAL.json'),
            predictions_sha256=seal['predictions_sha256'],tiny_head_forward_calls=714,new_backbone_forwards=0))
    return result


def paired(value):
    scalar_tree(value);require(set(value)==set(POPS),'All four paired populations required')
    names=None
    for pop,n in POPS.items():
        table=value[pop];require(len(table)==30,'All30 contrasts required in every population')
        if names is not None:require(set(table)==names,'Contrast set differs across populations')
        names=set(table)
        for record in table.values():
            require(record['denominator']==n and set(record['ids'])==set(GROUPS),'Paired full denominator/categories differ')
            ids=[fid for key in GROUPS for fid in record['ids'][key]]
            require(len(ids)==len(set(ids))==n and all(len(record['ids'][key])==record['counts'][key]for key in GROUPS),'Paired IDs omitted or duplicated')
            require(record['success_difference_count']==record['counts']['RECOVERY']-record['counts']['DAMAGE'],'Paired success count differs')
            require(set(record['deltas'])=={'success_rate','translation_cm','rotation_deg'},'Paired delta axes differ')
            if pop=='PRIMARY192':
                require(len(record['recording_clusters'])==len(record['leave_one_cluster_out'])==6,'PRIMARY six-cluster/leaveone results missing')
            for delta in record['deltas'].values():
                for key in ('recording_cluster_CI','named_session_cluster_sensitivity_CI'):
                    ci=delta[key]
                    if ci['cluster_count']>=2:require(ci['resamples']==10000 and ci['seed']==20260917,'Registered10000-draw cluster uncertainty required')


def error_budget40(repo,run,item,inputs,metrics_seal,metrics_sha256,add):
    """Authenticate the separate final40 saved-scalar diagnostic,not initial25."""
    sp=path_inside(run,item['path']);seal=inputs.read(sp,item['sha256']);scalar_tree(seal)
    require(sp.name=='ERROR_BUDGET_SEAL.json'and seal['schema']=='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1'and seal['complete']is True,'Final error-budget terminal required')
    require(seal['campaign_scope']=='ALL40_COMPLETED_ARMS'and len(seal['arms'])==40 and seal['arms']==metrics_seal['arms']and seal['populations']==POPS,'Final40/four-population diagnostic scope differs')
    require(seal['joint_rows']==800 and seal['band_rows']==1120 and seal['conservation_rows']==320,'All final error-budget rows required')
    require(seal['input_export_seal_sha256']==metrics_sha256,'Error budget must bind exact closed PUBLIC_METRICS40 exporter')
    require(seal['posthoc_descriptive']is True and seal['preregistered_test']is False and seal['selector_performance_claim']is False and seal['no_outlier_exclusion']is True and seal['independent_real_test_n']==0,'Posthoc descriptive scope/limitations differ')
    require(all(seal[k]==0 for k in ('raw_reference_reads','pose_or_RGB_reads','new_pose_metric_calls','new_solves','new_fits')),'Saved-scalar-only diagnostic required')
    inputs.bind(Path(repo)/P.CODE_PREFIX/'error_budget.py',seal['source_sha256'])
    inputs.bind(Path(repo)/P.CODE_PREFIX/'export_results.py',seal['scalar_exporter_source_sha256'])
    names={'JOINT_CATEGORIES.csv','ERROR_BANDS.csv','CONSERVATION_CHECKS.csv','ERROR_BUDGET_SCHEMA.json','INPUT_BINDINGS.json'}
    require(set(seal['artifacts'])==names,'Exact five scalar artifacts plus final seal required')
    bound=inputs.read(sp.parent/'INPUT_BINDINGS.json',seal['artifacts']['INPUT_BINDINGS.json'])
    require(bound['export_seal_sha256']==metrics_sha256 and bound['artifacts']==metrics_seal['artifacts'],'Diagnostic input artifact ancestry differs')
    schema=inputs.read(sp.parent/'ERROR_BUDGET_SCHEMA.json',seal['artifacts']['ERROR_BUDGET_SCHEMA.json']);scalar_tree(schema)
    require(schema['schema']=='POSTHOC_SAVED_SCALAR_ERROR_BUDGET_V1'and schema['populations']==POPS and schema['no_outlier_exclusion']is True and schema['no_metric_recalculation']is True,'Saved-error diagnostic schema differs')
    counts={'JOINT_CATEGORIES.csv':800,'ERROR_BANDS.csv':1120,'CONSERVATION_CHECKS.csv':320}
    for name in sorted(names):
        path=inputs.bind(sp.parent/name,seal['artifacts'][name])
        if path.suffix=='.json':scalar_tree(P.read(path))
        else:
            with path.open(newline='')as stream:require(sum(1 for _ in csv.DictReader(stream))==counts[name],'Final diagnostic scalar row count differs')
        add(path,P.DOC_PREFIX+'tables/ERROR_BUDGET40_V1/'+name,'saved_metric_metadata'if path.suffix=='.json'else'saved_metric_table',dict(seal_sha256=P.sha(sp),artifact_key=name))
    add(sp,P.DOC_PREFIX+'tables/ERROR_BUDGET40_V1/'+sp.name,'saved_metric_metadata',dict(seal_sha256=P.sha(sp),artifact_key=sp.name))
    return P.sha(sp)


def report_appendices(repo,run,item,inputs,expected,closed,documents,statuses,add):
    """Only five reviewed report scalar appendices; private receipt stays private."""
    sp=path_inside(run,item['path']);seal=inputs.read(sp,item['sha256'])
    require(sp.name=='REPORT_SEAL.json'and seal['schema']=='FINAL_KO_REPORT_SAVED_SCALAR_SEAL_V1'and seal['complete']is True,'Completed scalar-report receipt required')
    require(seal['scientific_and_derived_complete']is True and seal['report_ready']is True and seal['whole_campaign_complete']is False and seal['publication_pending']is True,'Report/analysis ready;publication remains pending')
    require(set(seal['artifacts'])==REPORT_FILES and set(seal['documents'])==REPORT_DOCUMENTS,'Exact five appendices and three report documents required')
    require(set(seal['metadata_documents'])==REPORT_METADATA_DOCUMENTS,'Exact completed method/stage JSON/CSV documents required')
    require(all(seal[k]==v for k,v in expected.items()),'Report source scalar-export/paired/errorbudget/cost ancestry differs')
    require(all(seal[k]==0 for k in ('raw_reference_reads','pose_payload_reads','model_tensor_reads','new_pose_metric_calls','new_solves','new_fits','new_forwards','new_renders')),'Report must consume saved scalars only')
    inputs.bind(Path(repo)/P.CODE_PREFIX/'final_report.py',seal['source_sha256'])
    require(set(seal['postbank_seal_sha256'])==set(POSTBANK_PATHS),'All four completed bank-analysis receipts required')
    for alias,folder in POSTBANK_PATHS.items():
        path=Path(run)/folder/'ANALYSIS_SEAL.json';digest=seal['postbank_seal_sha256'][alias]
        bank=inputs.read(path,digest);require(bank['complete']is True and bank['actual_new_pose_calls']==0,'Completed postseal bank metadata required')
        require(seal['input_bindings'][str(path)]==digest,'Report postbank source omitted from input bindings')
        if alias in ('S2','F'):require(str(path.resolve())in closed,'New bank analysis lacks exact completed sequence binding')
        for name,actual in bank['artifacts'].items():inputs.bind(path.parent/name,actual)
    for source,digest in seal['input_bindings'].items():inputs.bind(source,digest)
    supplied={item['name']:item['sha256']for item in documents}
    for name,digest in {**seal['documents'],**seal['metadata_documents']}.items():
        require(supplied.get(name)==digest,'ROOT must copy and explicitly select exact sealed report documents before final build')
        inputs.bind(Path(repo)/P.DOC_PREFIX/name,digest)
    with (Path(repo)/P.DOC_PREFIX/'STAGE_STATUS.csv').open(newline='')as stream:stage_rows=list(csv.DictReader(stream))
    json_rows=P.read(Path(repo)/P.DOC_PREFIX/'STAGE_STATUS.json')['stages']
    for rows in (stage_rows,json_rows):
        require(len(rows)==len(statuses)and {r['stage']:r['status']for r in rows}==statuses and all(r['status']in STATUSES for r in rows),'Final stage documents must match exact eight-stage/seven-enum selection status index')
    with (Path(repo)/P.DOC_PREFIX/'METHOD_DIFFERENCE_LEDGER.csv').open(newline='')as stream:method_rows=list(csv.DictReader(stream))
    method_json=P.read(Path(repo)/P.DOC_PREFIX/'METHOD_DIFFERENCE_LEDGER.json')
    for rows in (method_rows,method_json):
        require(isinstance(rows,list)and len(rows)==38 and all(r['status']in STATUSES for r in rows),'All38 method rows and registered status enums required;source-proof status belongs in a separate field')
    rows={'FULL40_PRIMARY_METRICS.csv':40,'REPRESENTATIVE_PRIMARY_METRICS.csv':16,'PAIRED30_PRIMARY_COMPACT.csv':30,'FIXED_COHORTS.csv':165}
    for name in sorted(REPORT_FILES):
        path=inputs.bind(sp.parent/name,seal['artifacts'][name])
        if path.suffix=='.json':scalar_tree(P.read(path))
        else:
            with path.open(newline='')as stream:require(sum(1 for _ in csv.DictReader(stream))==rows[name],'All final40/16representative/30paired/165cohort report rows required')
        add(path,P.DOC_PREFIX+'tables/final_report/'+name,'saved_metric_metadata'if path.suffix=='.json'else'saved_metric_table',dict(seal_sha256=P.sha(sp),artifact_key=name))
    return P.sha(sp)


def collect(repo,run,spec,inputs):
    repo,run=Path(repo).resolve(),Path(run).resolve();require(spec['schema']==SPEC and spec.get('template_only')is not True,'Final selection schema required;template remains incomplete')
    policy=inputs.read(path_inside(run,spec['size_policy']['path']),spec['size_policy']['sha256'])
    require(policy['schema']=='ROOT_REVIEWED_PUBLIC_SIZE_POLICY_V1'and policy['approved']is True and policy['author']=='ROOT','Explicit ROOT-reviewed size policy required')
    file_cap,total_cap=policy['maximum_file_bytes'],policy['maximum_total_bytes']
    require(type(file_cap)is int and 0<file_cap<=90*1024*1024 and type(total_cap)is int and total_cap>0,'Positive explicit caps;per-file maximum90MiB,below Git100MB')
    require(policy['no_case_or_outlier_omission']is True and bool(policy['reason']),'Caps cannot justify dropped cases/outliers')
    stage=spec['stage_statuses'];require(set(stage)=={'P0','S1','D_QCOV','S2','S3','S4','F','R'}and all(s in STATUSES for s in stage.values()),'Allowed seven-enum stage status index required')
    require(stage['D_QCOV']=='DIAGNOSTIC_ONLY'and all(stage[s]in ('COMPLETE_NEW','COMPLETE_EXACT_REUSE')for s in stage if s!='D_QCOV'),'Final campaign cannot contain unfinished stages')
    def terminal(item):
        path=path_inside(run,item['path']);value=inputs.read(path,item['sha256']);require(value['complete']is True,'Complete terminal prerequisite required');return path,value
    seqpath,seq=terminal(spec['final_sequence']);derivedpath,derived=terminal(spec['derived_completion'])
    require(derived['schema']=='FULL_CAMPAIGN_DERIVED_COMPLETION_SEAL_V1'and derived['upstream_sequence_seal_sha256']==P.sha(seqpath),'Derived/final-sequence ancestry differs')
    require(all(derived[k]==v for k,v in dict(contrasts=30,scalar_arms=40,scalar_population_rows=160,scalar_frame_arm_rows=9520,gallery_groups=5,gallery_seed_count=3,scatter_datasets=69).items()),'Complete30contrasts/40arms/5galleries/69scatter contract required')
    require(derived['raw_RGB_copied']is False and derived['public_poses_or_GT']is False and derived['all_outliers_and_failures_retained']is True,'Derived public privacy/coverage differs')
    closed=set()
    for terminal_path,value,stages in ((seqpath,seq,SEQUENCE_STAGES),(derivedpath,derived,DERIVED_STAGES)):
        plan=inputs.read(terminal_path.parent/'PLAN.json',value['plan_sha256'])
        require(tuple(plan['stage_order'])==stages and tuple(value['stage_order'])==stages,'Registered controller stage order differs')
        source=Path(repo/P.CODE_PREFIX/('final_sequence.py'if terminal_path==seqpath else'derived_completion.py'))
        inputs.bind(source,value['source_sha256']);require(plan['controller_sha256']==P.sha(source),'Controller numerical/source identity differs')
        root_go=inputs.read(path_inside(run,spec['final_sequence_GO'if terminal_path==seqpath else'derived_completion_GO']['path']),value['ROOT_GO_sha256'])
        require(root_go['approved']is True and root_go['plan_sha256']==P.sha(terminal_path.parent/'PLAN.json')and root_go['controller_sha256']==P.sha(source),'Controller ROOT GO/plan/source ancestry differs')
        for source,digest in plan['sources'].items():inputs.bind(source,digest)
        for source,digest in plan.get('fixed_inputs',{}).items():inputs.bind(source,digest)
        require(set(value['checkpoints'])=={s+'.json'for s in stages},'Every exact completed controller checkpoint required')
        inputs.bind(terminal_path.parent/'ACTUAL_COMMAND_LEDGER.jsonl',value['actual_command_ledger_sha256'])
        for name,digest in value['checkpoints'].items():
            checkpoint=inputs.read(terminal_path.parent/'checkpoints'/name,digest);require(checkpoint['complete']is True,'Incomplete controller checkpoint')
            require(checkpoint['identity']['input_bindings']['PLAN.json']==P.sha(terminal_path.parent/'PLAN.json')and checkpoint['identity']['input_bindings']['ROOT_GO']==value['ROOT_GO_sha256'],'Checkpoint PLAN/GO ancestry differs')
            for source,actual in checkpoint['completed_files'].items():inputs.bind(source,actual);closed.add(str(Path(source).resolve()))
    for source,digest in derived['public_contrasts_files'].items():inputs.bind(source,digest);closed.add(str(Path(source).resolve()))
    oldvisual=inputs.read(run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json');oldexec=inputs.read(run/'DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json')
    require(oldvisual['complete']is True and oldexec['artifact_seals'][str(run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json')]==P.sha(run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json'),'Preserved previous visual closure differs')
    oldgallery={g['name']:g['manifest_sha256']for g in oldvisual['galleries']}
    rights_path=Path(plan['d1'])/'RIGHTS_SCOPE_BINDING.json';rights=inputs.read(rights_path);approved=set(rights['approved_ids'])
    require(len(approved)==192,'Original approved PRIMARY192 display scope required')
    privacy_path=inputs.bind(Path(plan['d1'])/'DISPLAY_PRIVACY_PLAN.json')
    costpath=path_inside(run,spec['cost_summary']['path']);cost=inputs.read(costpath,spec['cost_summary']['sha256']);scalar_tree(cost)
    require(cost['schema']=='PUBLIC_RECOVERY_AWARE_MASTER_COST_SUMMARY_V1'and cost['all_declared_H1_fits_complete']is True and cost['all_declared_H1_prediction_seals_complete']is True,'Completed actual H1 cost/readiness projection required')
    require(cost['H1']['completed_new_fits']==cost['H1']['reservations']==33 and cost['H1']['tiny_head_real_prediction_forwards']==7854,'Actual33 fits/11head prediction seals required')
    require(cost['complete_cost_measurement']is False and cost['unknown_cost_token']=='UNKNOWN_NOT_ZERO','Interrupted/uncounted work must remain honestly unknown')
    fits=fit_accounts(run,inputs);pred=predictions(run,inputs,fits)
    visualpath,visual=terminal(spec['final_visual_review'])
    require(visual['schema']=='ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW_V1'and visual['approved_for_public_display']is True and visual['review_scope']=='LIMITED_SPOT_REVIEW','Actual ROOT limited pixel review prerequisite required')
    require(visual['human_owner_certifications']==0 and visual['raw_RGB_copied']is False,'No fabricated human owner or raw-image release')
    entries=[]
    def add(path,destination,kind,evidence=None):
        path=inputs.bind(path);P.clean_relative(destination)
        entries.append(dict(source=str(path),destination=destination,kind=kind,bytes=path.stat().st_size,sha256=P.sha(path),evidence=evidence))
    names=spec['code_filenames'];require(len(names)==len(set(names))and all('/'not in n and Path(n).suffix in ('.py','.md')for n in names),'Explicit unique code filenames required')
    for name in names:
        path=P.checked_source(repo/P.CODE_PREFIX/name,repo/P.CODE_PREFIX)
        if path.suffix=='.py':ast.parse(path.read_text())
        add(path,P.CODE_PREFIX+name,'code')
    execution_sources=spec['execution_sources'];execution_names=set()
    for source in execution_sources:
        path=path_inside(run,source['path']);require(path.suffix=='.py'and path.name not in execution_names,'Explicit unique small execution-source .py files required')
        require(path.stat().st_size<=1024*1024,'Execution wrappers must be small reviewed code,not embedded data arrays')
        execution_names.add(path.name);inputs.bind(path,source['sha256']);ast.parse(path.read_text())
        add(path,P.DOC_PREFIX+'execution_sources/'+path.name,'execution_source')
    require({'S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py','F_CORE_FOUR_CPU_CONTROLLER.py','ROOT_LAUNCH_RECOVERY_SERVICES.py','ROOT_CAPTURE_SERVICE_LIFECYCLE.py'}<=execution_names,'Actual RUN-owned controllers and ROOT service/lifecycle source must be reconstructible')
    docs=spec['documents'];require({'FINAL_REPORT_KO.md','METHODOLOGY_KO.md','STRUCTURE_REVIEW_NOTES_KO.md','REPRODUCE.md'}<={d['name']for d in docs},'Final report/methodology/structure/reproduction documents required')
    for doc in docs:
        name=doc['name'];require(name in SAFE_DOCUMENTS,'Unreviewed final document filename')
        path=P.checked_source(repo/P.DOC_PREFIX/name,repo/P.DOC_PREFIX);inputs.bind(path,doc['sha256']);add(path,P.DOC_PREFIX+name,'documentation')
    gallery_counts={};gallery_seals={}
    roots=[(run/key,label)for key,label in P.GALLERIES.items()]+[(derivedpath.parent/('GALLERY_'+key),key)for key in NEW_GALLERIES]
    for root,label in roots:
        mp=root/'GALLERY_MANIFEST.json';m=inputs.read(mp)
        if root.parent==derivedpath.parent:require(str(mp.resolve())in closed,'Gallery lacks exact completed derived checkpoint binding')
        else:require(oldgallery[root.name]==P.sha(mp),'Preserved final gallery differs from its old completed review')
        require(m['complete']is True and m['CAL_RGB_opened']==0 and m['raw_RGB_copied']is False and m['public_reference_pose_or_GT']is False,'Completed PRIMARY redacted gallery required')
        require(visual['gallery_seals'][str(mp.relative_to(run))]==P.sha(mp),'Actual limited review does not bind this gallery')
        cases=m['cases'];require(len(cases)==m['case_count']and set(m['all_PRIMARY_transition_ids'])<={c['id']for c in cases},'Every all-seed transition ID must remain in gallery')
        require(all(c['id']in approved for c in cases)and m['rights_scope_sha256']==P.sha(rights_path)and m['privacy_plan_sha256']==P.sha(privacy_path),'Gallery rights/privacy scope differs')
        expected={'index.html'}|{c[key]for c in cases for key in ('panel','family_panel','context_panel','json')}
        require(set(m['artifact_sha256'])==expected,'Whole gallery artifact set cannot be filtered or extended')
        for name,digest in m['artifact_sha256'].items():
            path=P.checked_source(root/name,root);inputs.bind(path,digest)
            kind='gallery_html'if name=='index.html'else'gallery_derived_metadata'if name.endswith('.json')else'gallery_derived_jpeg'
            require(name=='index.html'or __import__('re').fullmatch(r'case_\d{4}(?:_context|_families)?\.(?:jpg|json)',name),'Unexpected gallery file')
            if kind=='gallery_derived_metadata':
                v=P.read(path);P.guard_json(v);require(v['partition']in ('TEST','REMAINDER')and v['raw_RGB_copied']is False and v['reference_pose_or_GT_published']is False,'Only public PRIMARY derived metadata allowed')
            add(path,P.DOC_PREFIX+'galleries/'+label+'/'+name,kind,dict(seal_sha256=P.sha(mp),artifact_key=name))
        gallery_counts[label]=len(cases);gallery_seals[label]=P.sha(mp)
    for root,label in [(run/'S1_S3_S4_COMPARISON_FIGURES','S1_S3_S4_T_horizontal_R_vertical'),(derivedpath.parent/'FULL_SCATTER','FULL_T_horizontal_R_vertical')]:
        mp=root/'FIGURE_MANIFEST.json';m=inputs.read(mp);require(m['complete']is True and m['all_outliers_retained']is True and m['PRIMARY_frames_per_dataset']==192,'Full-outlier PRIMARY scatter required')
        if root.parent==derivedpath.parent:require(str(mp.resolve())in closed,'Scatter lacks completed derived checkpoint binding')
        else:require(oldexec['artifact_seals'][str(mp)]==P.sha(mp),'Preserved scatter differs from final execution seal')
        for name,kind in [('index.html','scatter_html'),('SCATTER_DATA.json','scatter_derived_data'),('REPRESENTATIVE_FULL.png','scatter_figure'),('REPRESENTATIVE_FULL.svg','scatter_figure')]:
            path=P.checked_source(root/name,root);inputs.bind(path,m['artifacts'][name]);add(path,P.DOC_PREFIX+'figures/'+label+'/'+name,kind,dict(seal_sha256=P.sha(mp),artifact_key=name))
    orientation=run/'S1_ORIENTATION_SUPPLEMENT';m=inputs.read(orientation/'DISPLAY_SEAL.json');name='ORIENTATION_NATIVE_AXES.jpg'
    require(m['complete']is True and m['raw_RGB_copied']is False and m['GT_numeric_JSON_exported']is False,'Safe saved orientation supplement required')
    require(oldexec['artifact_seals'][str(orientation/'DISPLAY_SEAL.json')]==P.sha(orientation/'DISPLAY_SEAL.json'),'Preserved orientation differs')
    inputs.bind(orientation/name,m['artifact_sha256'][name]);add(orientation/name,P.DOC_PREFIX+'figures/S1_orientation/'+name,'orientation_derived_jpeg',dict(seal_sha256=P.sha(orientation/'DISPLAY_SEAL.json'),artifact_key=name))
    for root,kind in [(derivedpath.parent/'PUBLIC_METRICS40','metrics'),(derivedpath.parent/'PUBLIC_CONTRASTS30','contrasts'),(run/'D_PUBLIC_SCALARS_V3','D')]:
        terminal_name='PUBLIC_CONTRASTS_SEAL.json'if kind=='contrasts'else'EXPORT_SEAL.json';sp=root/terminal_name;s=inputs.read(sp);require(s['complete']is True,'Complete scalar export required')
        if root.parent==derivedpath.parent:require(str(sp.resolve())in closed,'Scalar export lacks completed derived checkpoint binding')
        if kind=='metrics':require(s['schema']=='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1'and len(s['arms'])==40 and s['summary_rows']==160 and s['scalar_row_rows']==9520 and s['metrics_recomputed']is False,'Complete40-arm metric whitelist required')
        if kind=='metrics':metric_seal=s;metric_sha256=P.sha(sp)
        if kind=='contrasts':require(s['public_poses_or_GT']is False and s['new_metric_or_statistical_calculations']==0,'Saved contrast whitelist required')
        if kind=='D':require(s['schema']=='PUBLIC_D_SAVED_SCALAR_EXPORT_SEAL_V1'and s['raw_reference_decodes']==0 and s['mixed_pose_rows_decoded']==0,'D saved-scalar export required')
        for name,digest in s['artifacts'].items():
            path=P.checked_source(root/name,root);inputs.bind(path,digest)
            require(path.suffix in ('.json','.csv'),'Only scalarJSON/CSV export artifacts')
            if path.suffix=='.json':
                v=P.read(path);scalar_tree(v)
                if name=='ALL_POPULATION_PAIRED_CONTRASTS.json':paired(v)
            add(path,P.DOC_PREFIX+'tables/'+root.name+'/'+name,'saved_metric_metadata'if path.suffix=='.json'else'saved_metric_table',dict(seal_sha256=P.sha(sp),artifact_key=name))
    error_budget_sha256=error_budget40(repo,run,spec['error_budget40'],inputs,metric_seal,metric_sha256,add)
    add(costpath,P.DOC_PREFIX+'tables/costs/PUBLIC_COST_SUMMARY.json','saved_metric_metadata',dict(input_sha256=P.sha(costpath)))
    fifthpath=path_inside(run,spec['derived_service_cost']['path']);fifth=inputs.read(fifthpath,spec['derived_service_cost']['sha256']);scalar_tree(fifth)
    require(set(fifth)=={'complete','CPU_seconds','elapsed_wall_seconds','Result','ExecMainStatus','NRestarts','source_sha256','launch_receipt_sha256','relation'},'Fifth-service public scalar whitelist differs')
    require(fifth['complete']is True and fifth['Result']=='success'and int(fifth['ExecMainStatus'])==int(fifth['NRestarts'])==0,'Derived service capture must be completed successfully')
    require(all(type(fifth[k])in(int,float)and math.isfinite(fifth[k])and fifth[k]>=0 for k in ('CPU_seconds','elapsed_wall_seconds')),'Finite nonnegative service counters required')
    inputs.bind(repo/P.CODE_PREFIX/'cost_waiter.py',fifth['source_sha256'])
    add(fifthpath,P.DOC_PREFIX+'tables/costs/DERIVED_SERVICE_PUBLIC.json','saved_metric_metadata',dict(input_sha256=P.sha(fifthpath)))
    paired_seal_sha256=P.sha(derivedpath.parent/'PUBLIC_CONTRASTS30/PUBLIC_CONTRASTS_SEAL.json')
    report_sha256=report_appendices(repo,run,spec['report_appendices'],inputs,
        dict(metrics_export_seal_sha256=metric_sha256,paired_export_seal_sha256=paired_seal_sha256,error_budget_seal_sha256=error_budget_sha256,cost_summary_sha256=P.sha(costpath)),
        closed,docs,stage,add)
    require(len({e['destination']for e in entries})==len(entries),'Duplicate final package destination')
    receipt_index=spec['public_receipt_index'];roles={'provenance','cache','calls','argv','environment','model','calibration'}
    require(roles<={r['role']for r in receipt_index},'Provenance/cache/calls/argv/environment/model/calibration receipt index required')
    public_receipts=[]
    for receipt in receipt_index:
        require(receipt['role']in roles and bool(receipt['scope']),'Typed receipt scope required')
        path=path_inside(run,receipt['path']);require(path.suffix in ('.json','.jsonl'),'Only named metadata receipts in public hash index')
        inputs.bind(path,receipt['sha256']);public_receipts.append({k:receipt[k]for k in ('alias','role','scope','sha256')})
    public_index=dict(schema='FINAL_PUBLIC_PROVENANCE_INDEX_V1',whole_campaign_complete=False,
        scientific_and_derived_complete=True,publication_ready=True,publication_status=PUBLICATION_STATUS,stage_statuses=stage,
        R_stage_scope='Completed report/analysis/reproduction metadata;publication and Git remain separate and pending',
        publication_snapshot_scope='Build-time snapshot;ROOT post-push remote commit-SHA completion is outside this commit to avoid circular self-attestation',
        actual_new_H1_fits=33,actual_fit_attempts_registered=33,fit_budget=36,fit_receipts=fits,prediction_receipts=pred,
        representative_seed=20261018,populations=POPS,independent_real_test_n=0,human_owner_certifications=0,
        final_sequence_seal_sha256=P.sha(seqpath),derived_completion_seal_sha256=P.sha(derivedpath),
        public_cost_summary_sha256=P.sha(costpath),derived_service_public_sha256=P.sha(fifthpath),cost_measurement_complete=False,
        final40_error_budget_seal_sha256=error_budget_sha256,final40_error_budget_scope='Posthoc saved-scalar diagnostic,not a preregistered hypothesis test or selector-performance estimate;initial25tables excluded',
        scalar_report_seal_sha256=report_sha256,
        limited_pixel_review_sha256=P.sha(visualpath),limited_pixel_review_scope='LIMITED_SPOT_REVIEW',
        gallery_case_counts=gallery_counts,gallery_seals=gallery_seals,unknown_costs='UNKNOWN_NOT_ZERO',
        receipt_hash_index=public_receipts,source_hash_index={name:P.sha(repo/P.CODE_PREFIX/name)for name in names},
        execution_source_hash_index={Path(s['path']).name:s['sha256']for s in execution_sources},
        execution_source_path_mapping='Preserved exact RUN-owned scripts contain original local paths;map the declared private run/worktree/interpreter paths before independent reproduction. Do not treat reconstructed API calls as historically executed CLI.',
        interpretation='Completed development campaign;Dconditional diagnostic;MATCHEDF notoriginal120parity;no independent physical-owner or new-real confirmation claim')
    return entries,public_index,dict(schema='EXPLICIT_PUBLIC_CODE_FILENAMES_V1',filenames=names),policy


def paired_csvs(value):
    """Copy saved paired fields/IDs into portable CSV;no statistical fit."""
    summaries=[];ids=[]
    for pop,table in value.items():
        for name,record in table.items():
            row=dict(population=pop,contrast=name,before=record['before'],after=record['after'],denominator=record['denominator'],
                success_difference_count=record['success_difference_count'],success_difference_percentage_points=record['success_difference_percentage_points'],**record['counts'])
            for axis,d in record['deltas'].items():
                row[axis+'_mean']=d['mean'];row[axis+'_common_available_n']=d['common_available_n']
                for unit in ('recording_cluster_CI','named_session_cluster_sensitivity_CI'):
                    ci=d[unit];bounds=ci['CI95'];row[axis+'_'+unit+'_lo']=None if bounds is None else bounds[0];row[axis+'_'+unit+'_hi']=None if bounds is None else bounds[1]
            summaries.append(row)
            for category,values in record['ids'].items():
                ids.extend(dict(population=pop,contrast=name,category=category,id=fid)for fid in values)
    def encoded(rows):
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);return stream.getvalue()
    return {'PAIRED_CONTRAST_SUMMARY.csv':encoded(summaries),'ALL_PAIRED_CATEGORY_IDS.csv':encoded(ids)}


def generated_values(run,spec,index,catalog):
    derived=path_inside(run,spec['derived_completion']['path']).parent
    paired_value=P.read(derived/'PUBLIC_CONTRASTS30/ALL_POPULATION_PAIRED_CONTRASTS.json')
    result={'PUBLIC_PROVENANCE_INDEX.json':(P.DOC_PREFIX+'PUBLIC_PROVENANCE_INDEX.json',index),
            'PUBLIC_CODE_ALLOWLIST.json':(P.DOC_PREFIX+'PUBLIC_CODE_ALLOWLIST.json',catalog)}
    result.update({name:(P.DOC_PREFIX+'tables/PUBLIC_CONTRASTS30/'+name,value)for name,value in paired_csvs(paired_value).items()})
    return result


def validate_files(entries,file_cap,total_cap):
    destinations={e['destination']for e in entries};total=0;largest=[];links=0
    for e in entries:
        path=P.no_symlink(e['source']);require(P.sha(path)==e['sha256']and path.stat().st_size==e['bytes'],'Final selected bytes changed')
        suffix=path.suffix.lower();require(suffix not in P.FORBIDDEN_SUFFIXES,'Private binary/cache asset prohibited')
        require(e['destination'].startswith((P.CODE_PREFIX,P.DOC_PREFIX)),'Only new namespace/report destinations')
        if suffix=='.json':P.guard_json(P.read(path))
        elif suffix=='.csv':
            with path.open(newline='')as stream:require(not(set(next(csv.reader(stream),[]))&P.FORBIDDEN_KEYS),'Raw pose/reference CSV header')
        elif suffix in ('.md','.html'):
            for link in P.links_in(path.read_text(),suffix):
                url=P.urlsplit(P.html.unescape(link))
                if url.scheme in ('http','https','mailto')or not url.path:continue
                require(not url.scheme and not url.netloc and not url.path.startswith('/'),'Private/nonportable file link')
                target=P.posixpath.normpath(P.posixpath.join(P.posixpath.dirname(e['destination']),P.unquote(url.path)))
                require(target in destinations,'Missing local final-package link: '+target);links+=1
        elif suffix in ('.jpg','.png'):
            with path.open('rb')as stream:header=stream.read(16)
            require(header.startswith(b'\xff\xd8\xff')if suffix=='.jpg'else header.startswith(b'\x89PNG\r\n\x1a\n'),'Invalid derived image header')
        total+=e['bytes'];largest.append(dict(destination=e['destination'],bytes=e['bytes']))
    inventory=P.contents(dict(schema=SCHEMA,whole_campaign_complete=False,package_base_execution_sha=BASE,entries=entries),'0'*64)
    inventory_bytes=len((json.dumps(inventory,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    audit=dict(schema='FINAL_PUBLIC_SIZE_AUDIT_V1',files=len(entries),total_bytes=total,maximum_file_bytes=file_cap,maximum_total_bytes=total_cap,
        staged_files=len(entries)+1,PUBLIC_CONTENTS_bytes=inventory_bytes,actual_staging_total_bytes=total+inventory_bytes,
        inventory_length_method='Exact serialized PUBLIC_CONTENTS shape;manifest binding uses a fixed64hex placeholder of identical byte length',
        largest_files=sorted(largest,key=lambda r:-r['bytes'])[:20],every_case_and_outlier_retained=True,no_files_truncated=True,
        per_file_git_safety_limit_bytes=90*1024*1024,local_links_verified=links,
        limits_satisfied=total+inventory_bytes<=total_cap and inventory_bytes<=file_cap and all(e['bytes']<=file_cap for e in entries))
    return audit


def build(repo,run,spec_path,manifest):
    manifest=Path(manifest);require(not manifest.exists(),'Fresh final manifest path required')
    inputs=Inputs();spec=inputs.read(spec_path);entries,index,catalog,policy=collect(repo,run,spec,inputs)
    generated=manifest.parent/(manifest.stem+'_public_metadata');require(not generated.exists(),'Fresh generated public metadata folder required');generated.mkdir(parents=True)
    for name,(destination,value)in generated_values(Path(run),spec,index,catalog).items():
        if name.endswith('.json'):P.write_new(generated/name,value)
        else:
            with (generated/name).open('x',newline='')as stream:stream.write(value)
        path=generated/name;entries.append(dict(source=str(path.resolve()),destination=destination,kind='publication_metadata',bytes=path.stat().st_size,sha256=P.sha(path),evidence=None))
    audit=validate_files(entries,policy['maximum_file_bytes'],policy['maximum_total_bytes'])
    audit_path=manifest.with_name(manifest.stem+'_SIZE_AUDIT.json');P.write_new(audit_path,audit)
    require(audit['limits_satisfied'],'Explicit size caps exceeded;review SIZE_AUDIT and issue a fresh ROOT policy/manifest,never truncate cases')
    inputs.stable()
    value=dict(schema=SCHEMA,draft=False,whole_campaign_complete=False,scientific_and_derived_complete=True,
        publication_ready=True,publication_status=PUBLICATION_STATUS,root_git_commit_push_only=True,
        package_base_execution_sha=BASE,repo=str(Path(repo).resolve()),run=str(Path(run).resolve()),
        selection_spec=dict(path=str(Path(spec_path).resolve()),sha256=P.sha(spec_path)),source_evidence_bindings=inputs.hashes,
        source_roots=sorted({str(Path(e['source']).parent.resolve())for e in entries}),entries=entries,
        maximum_file_bytes=policy['maximum_file_bytes'],maximum_total_bytes=policy['maximum_total_bytes'],
        declared_files=len(entries),declared_total_bytes=audit['total_bytes'],size_audit=dict(path=str(audit_path.resolve()),sha256=P.sha(audit_path)),
        builder_source_sha256=P.sha(__file__),public_package_source_sha256=P.sha(P.__file__),
        actual_new_H1_fits=33,all_gallery_cases_and_outliers_retained=True,public_copy_executed=False)
    P.write_new(manifest,value);return verify(manifest)


def verify(manifest):
    m=P.read(manifest);require(m['schema']==SCHEMA and m['whole_campaign_complete']is False and m['draft']is False
        and m['scientific_and_derived_complete']is True and m['publication_ready']is True and m['publication_status']==PUBLICATION_STATUS,'Prepared final schema with publication pending required')
    require(m['package_base_execution_sha']==BASE,'Final package base commit differs')
    require(m['builder_source_sha256']==P.sha(__file__)and m['public_package_source_sha256']==P.sha(P.__file__),'Reviewed final selector source changed')
    inputs=Inputs();spec=inputs.read(m['selection_spec']['path'],m['selection_spec']['sha256'])
    expected,index,catalog,policy=collect(m['repo'],m['run'],spec,inputs)
    selected={e['destination']:e for e in m['entries']};require(len(selected)==len(m['entries']),'Duplicate manifest entries')
    for e in expected:require(selected.get(e['destination'])==e,'Final manifest changed selection/source/coverage')
    extra=set(selected)-{e['destination']for e in expected};generated=generated_values(Path(m['run']),spec,index,catalog)
    require(extra=={dest for dest,_ in generated.values()},'Unexpected extra final public file')
    for name,(destination,value)in generated.items():
        e=selected[destination];source=P.checked_source(e['source'],Path(manifest).parent/(Path(manifest).stem+'_public_metadata'))
        require(source.name==name and e['kind']=='publication_metadata','Generated public metadata identity differs')
        require((P.read(source)==value)if name.endswith('.json')else source.read_text()==value,'Generated public provenance/catalog/pairedCSV differs')
    require(m['maximum_file_bytes']==policy['maximum_file_bytes']and m['maximum_total_bytes']==policy['maximum_total_bytes'],'Manifest silently changed reviewed caps')
    audit=validate_files(m['entries'],m['maximum_file_bytes'],m['maximum_total_bytes']);inputs.bind(m['size_audit']['path'],m['size_audit']['sha256'])
    require(P.read(m['size_audit']['path'])==audit and audit['limits_satisfied'],'Final size audit differs/exceeds reviewed cap')
    require(audit['files']==m['declared_files']and audit['total_bytes']==m['declared_total_bytes'],'Final file/byte count differs')
    for path,digest in m['source_evidence_bindings'].items():inputs.bind(path,digest)
    inputs.stable()
    return dict(PASS=True,manifest_sha256=P.sha(manifest),whole_campaign_complete=False,scientific_and_derived_complete=True,
        publication_ready=True,publication_status=PUBLICATION_STATUS,actual_new_H1_fits=33,
        allowlisted_files=audit['files'],bytes=audit['total_bytes'],local_links_verified=audit['local_links_verified'],
        actual_gallery_files_copied=0,new_scientific_calls=0,raw_references_or_pose_packets_decoded=0)


def template(repo,run,output):
    """Incomplete ROOT-editable spec;not a final manifest or execution claim."""
    repo,run=Path(repo).resolve(),Path(run).resolve();catalog=P.read(repo/P.DOC_PREFIX/'PUBLIC_CODE_ALLOWLIST_DRAFT.json')
    additions=['public_final.py','test_public_final.py','fullnormal_recovery.py','test_fullnormal_recovery.py','final_sequence.py','test_final_sequence.py',
        'derived_completion.py','test_derived_completion.py','master_costs.py','test_master_costs.py','test_f_finish.py','export_d_scalars.py','test_export_d_scalars.py']
    additions+=['cost_waiter.py','test_cost_waiter.py','error_budget.py','test_error_budget.py',
        'MASTER_COSTS_README.md','COST_WAITER_README.md','FINAL_PUBLIC_PACKAGE_README.md','final_report.py','test_final_report.py','FINAL_REPORT_README.md',
        'error40_waiter.py','test_error40_waiter.py']
    names=sorted(set(catalog['filenames'])|{n for n in additions if (repo/P.CODE_PREFIX/n).is_file()})
    ref=lambda path:dict(path=path,sha256='ROOT_FILL_AFTER_ACTUAL_COMPLETE')
    cost_relative='ROOT_MASTER_COST_AUDIT_V1/PUBLIC_COST_SUMMARY.json'
    if (run/'MASTER_COSTS_WAIT_V2/PLAN.json').is_file():
        cost_relative=str((Path(P.read(run/'MASTER_COSTS_WAIT_V2/PLAN.json')['master_output'])/'PUBLIC_COST_SUMMARY.json').relative_to(run))
    value=dict(schema=SPEC,template_only=True,code_filenames=names,
        documents=[dict(name=n,sha256='ROOT_FILL_AFTER_FINAL_REWRITE')for n in ('FINAL_REPORT_KO.md','METHODOLOGY_KO.md','STRUCTURE_REVIEW_NOTES_KO.md','REPRODUCE.md','S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md','METHOD_DIFFERENCE_LEDGER.json','METHOD_DIFFERENCE_LEDGER.csv','STAGE_STATUS.json','STAGE_STATUS.csv')],
        stage_statuses={s:'INCOMPLETE'for s in ('P0','S1','D_QCOV','S2','S3','S4','F','R')},
        size_policy=ref('ROOT_FINAL_PUBLIC_SIZE_POLICY.json'),
        final_sequence=ref('FINAL_SEQUENCE_RECOVERY_V1/FINAL_SEQUENCE_SEAL.json'),derived_completion=ref('DERIVED_COMPLETION_V1/DERIVED_COMPLETION_SEAL.json'),
        final_sequence_GO=ref('FINAL_SEQUENCE_RECOVERY_ROOT_GO_V1.json'),derived_completion_GO=ref('DERIVED_COMPLETION_ROOT_GO_V1.json'),
        cost_summary=ref(cost_relative),
        derived_service_cost=ref('ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json'),
        error_budget40=ref('ERROR_BUDGET40_V1/ERROR_BUDGET_SEAL.json'),
        report_appendices=ref('FINAL_REPORT_V1/REPORT_SEAL.json'),
        final_visual_review=ref('ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW.json'),
        execution_sources=[ref(n)for n in ('S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py','F_CORE_FOUR_CPU_CONTROLLER.py',
            'ROOT_LAUNCH_RECOVERY_SERVICES.py','ROOT_CAPTURE_SERVICE_LIFECYCLE.py',
            'ROOT_LAUNCH_POST_CORE_RECOVERY_SERVICES.py','ROOT_LAUNCH_DERIVED_COMPLETION.py',
            'ROOT_LAUNCH_COST_WAITER_V2.py','ROOT_RUN_ERROR_BUDGET25.py','ROOT_MASTER_METADATA_COMPONENT_PREFLIGHT.py','ROOT_LAUNCH_ERROR40_WAITER_V1.py')],
        public_receipt_index=[dict(alias=role,role=role,path='ROOT_FILL_METADATA_RECEIPT.json',sha256='ROOT_FILL_ACTUAL_SHA',scope='ROOT_FILL_EXACT_EXECUTED_SCOPE')for role in ('provenance','cache','calls','argv','environment','model','calibration')],
        size_policy_contract=dict(schema='ROOT_REVIEWED_PUBLIC_SIZE_POLICY_V1',approved=False,author='ROOT',maximum_file_bytes=90*1024*1024,
            maximum_total_bytes='ROOT_SET_AFTER_FULL_SIZE_REVIEW',no_case_or_outlier_omission=True,reason='ROOT explicit operational cap;never omit cases or outliers'),
        visual_review_contract=dict(schema='ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW_V1',complete=False,approved_for_public_display=False,
            review_scope='LIMITED_SPOT_REVIEW',human_owner_certifications=0,raw_RGB_copied=False,gallery_seals='ROOT actual11galleryseals, keyed by run-relative GALLERY_MANIFEST path'),
        notes=['Final-build refuses this template until actual terminal hashes/stage statuses/doc hashes/ROOT policy/review are supplied.',
            'Review final code filenames,including any later cost-waiter helper;do not add private assets or weights.',
            'PUBLIC_COST path is a placeholder;bind the actual completed master public projection path.',
            'Previous draft manifests/maps/sources stay immutable;no public copy/Git operation is performed by template or build.'])
    P.write_new(output,value);return dict(template_written=str(output),ready_for_final_build=False,new_scientific_calls=0)
