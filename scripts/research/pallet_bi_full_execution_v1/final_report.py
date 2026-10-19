"""Korean final-report materialization from closed saved scalar metadata only.

Standard library only. No row-error, pose, reference, feature, image or tensor
payload is decoded; no metric, bootstrap, model, optimizer or renderer runs.
Actual materialization needs a source/spec-bound ROOT GO and fresh output.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
import posixpath
from pathlib import Path
import re
import time
from urllib.parse import unquote,urlsplit

SCHEMA='FINAL_KO_REPORT_SPEC_V1'
SEAL='FINAL_KO_REPORT_SAVED_SCALAR_SEAL_V1'
BASE='https://github.com/CanelE452/pallet-6d-pose/blob/3553202a3682faeb7be2e7df051b230683965382/'
SEEDS=(20261018,20261019,20261020)
PREFIXES=('S1_UNION','S2_C0','S2_CP','S2_CB','S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE','F_COMMON','F_BI')
POPS={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
GROUPS=('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')
COHORTS={'original31_selection_failure':31,'original41_candidate_absent':41,'PF_union38_candidate_absent':38,
 'PF_new_good':3,'PF_good_old_bank_good_H1_fail':12}
STATUS={'COMPLETE_NEW','COMPLETE_EXACT_REUSE','DIAGNOSTIC_ONLY','BLOCKED_INPUT','BLOCKED_TECHNICAL','INCOMPLETE','NOT_APPLICABLE_WITH_REASON'}
SUMMARY=('population','arm','denominator','successes','success_rate','no_pose','T_gt50cm','T_gt100cm','R_gt45deg',
 'fallback_known_n','fallback_true_n','fallback_false_n','fallback_unknown_n')
STATS=('n','missing','mean','sample_variance','sample_std','median','P90','P95','max')
FIELDS=SUMMARY+tuple(a+'_'+k+(''if k in('n','missing')else'_'+u+('2'if k=='sample_variance'else''))
 for a,u in [('T','cm'),('R','deg')]for k in STATS)
PAIR_DEFS=(('S2_C0','S2_CP','covariance_floor'),('S2_CP','S2_CB','role_calibration'),
 ('S1_UNION','S3_COMMON','common_material_proposals'),('S1_UNION','S3_BI','role_material_proposals'),
 ('S3_COMMON','S3_BI','S3_role_policy'),('S1_UNION','S4_PL','PL_proposals'),
 ('S4_PL','S4_CONTOUR','contour_minus_PL'),('S4_CONTOUR','S4_IMAGE','image_minus_contour'),
 ('S4_PL','S4_IMAGE','image_minus_PL'),('F_COMMON','F_BI','final_matched_BI_effect'))
APPENDICES=('FULL40_PRIMARY_METRICS.csv','REPRESENTATIVE_PRIMARY_METRICS.csv','PAIRED30_PRIMARY_COMPACT.csv',
 'FIXED_COHORTS.csv','FIXED_COHORTS.json')
DOCUMENTS=('FINAL_REPORT_KO.md','S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md')
DERIVED_STAGES=['contrasts','export','gallery_S2','gallery_F','gallery_F_BI_VS_COMMON','gallery_S2_CP_VS_C0','gallery_S2_CB_VS_CP','scatter']
SEQUENCE_STAGES=['S2_evaluation','S2_independent','S2_bank','S2_selection','S2_core','F_choose','F_finish',
 'F_evaluation','F_independent','F_bank','F_selection','F_core']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def arms40():
    return sorted(['N0','E0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR',
        *['D1_H1_MLP_s'+str(s)for s in SEEDS],*[a+'_s'+str(s)for a in PREFIXES for s in SEEDS]])


class Inputs:
    def __init__(self):self.bindings={}
    def bind(self,path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        require(expected is None or actual==expected,'Changed saved input: '+path.name)
        require(str(path)not in self.bindings or self.bindings[str(path)]==actual,'Input changed during report')
        self.bindings[str(path)]=actual;return path
    def read(self,path,expected=None):
        path=Path(path).resolve()
        require(path.suffix=='.json'and not any(x in path.name for x in
            ('FIXED_REFERENCES','SYNTH_GT_LABELS','GEOMETRY_SIDETABLE','E_RUN','PREDICTIONS')),
            'Only explicitly declared saved scalar/receipt JSON may be decoded')
        path=self.bind(path,expected)
        return json.loads(path.read_text())
    def stable(self):
        for p,d in self.bindings.items():require(sha(p)==d,'Input changed before report seal')


def number(value):
    if value in('',None):return None
    n=float(value);require(math.isfinite(n),'Nonfinite saved scalar');return n


def integer(value):
    n=int(value);require(str(n)==str(value)or type(value)is int,'Noninteger saved count');return n


def csv_read(path,fields=None):
    with Path(path).open(newline='')as f:
        reader=csv.DictReader(f)
        require(fields is None or reader.fieldnames==list(fields),'Saved CSV columns differ')
        result=list(reader)
    require(all(None not in row and all(v is not None for v in row.values())for row in result),'Malformed CSV')
    return result


def write_json(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def write_csv(path,fields,rows):
    with Path(path).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def bundle(inputs,spec,terminal):
    root=Path(spec['path']);seal=inputs.read(root/terminal,spec['sha256'])
    require(seal['complete']is True,'Saved bundle is incomplete')
    for name,d in seal['artifacts'].items():
        require(Path(name).name==name,'Bundle artifact escaped root');inputs.bind(root/name,d)
    return root,seal


def metrics(inputs,spec):
    root,seal=bundle(inputs,spec,'EXPORT_SEAL.json');expected=arms40()
    require(seal['schema']=='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1'and seal['arms']==expected
        and seal['populations']==POPS and seal['summary_rows']==160 and seal['scalar_row_rows']==9520
        and seal['identities']==238 and seal['all_outliers_and_failures_retained']is True
        and seal['metrics_recomputed']is False and seal['raw_references_decoded']==seal['solve_fit_forward_calls']==0,
        'Closed all40/four-population scalar export required')
    require(set(seal['artifacts'])=={'ALL_POPULATION_METRICS.csv','ROW_METRICS.csv','METRICS_SCHEMA.json','INPUT_BINDINGS.json'},
        'Exact scalar-export bundle required')
    schema=inputs.read(root/'METRICS_SCHEMA.json')
    require(schema['sample_variance_ddof']==schema['sample_std_ddof']==1 and schema['populations']==POPS
        and schema['summary_columns']==list(FIELDS) and schema['independent_real_test_n']==0,'Saved metric ddof/scope differs')
    rows=csv_read(root/'ALL_POPULATION_METRICS.csv',FIELDS);table={}
    for r in rows:
        key=r['population'],r['arm'];require(key not in table,'Duplicate metric arm/population')
        require(key[0]in POPS and key[1]in expected and integer(r['denominator'])==POPS[key[0]],'Metric identity/denominator differs')
        n=POPS[key[0]];success=integer(r['successes']);missing=integer(r['no_pose'])
        require(0<=success<=n-missing and number(r['success_rate'])==success/n,'Saved success/rate differs')
        require(sum(integer(r['fallback_'+v+'_n'])for v in('true','false','unknown'))==n,'Fallback missingness was lost')
        for a,u in [('T','cm'),('R','deg')]:
            available=integer(r[a+'_n']);require(available+integer(r[a+'_missing'])==n and integer(r[a+'_missing'])==missing,'Available-only count differs')
            for k in STATS[2:]:
                value=number(r[a+'_'+k+'_'+u+('2'if k=='sample_variance'else'')])
                require((value is None)==(available<(2 if k in('sample_variance','sample_std')else 1)), 'Undefined statistic must remain missing')
                require(value is None or value>=0,'Negative saved error moment')
        table[key]=r
    require(set(table)=={(p,a)for p in POPS for a in expected},'Every40 arm and four populations required')
    return table,seal,inputs.read(root/'INPUT_BINDINGS.json')


def authorities(inputs,spec,public_bindings,metric_seal):
    require(len(spec)==len(public_bindings['sources'])==4,'Four completed evaluation/independent bundles required')
    seen=set();reference=None
    for item,saved in zip(spec,public_bindings['sources']):
        ev,ind=Path(item['evaluation']),Path(item['independent'])
        es=inputs.read(ev/'EVALUATION_SEAL.json',saved['evaluation_seal_sha256'])
        check=inputs.read(ind/'INDEPENDENT_VERIFICATION.json',saved['independent_artifacts']['INDEPENDENT_VERIFICATION.json'])
        require(es['complete']is True and check['PASS']is True and check['errors']==[] and check['rows']==238
            and check['arms']==es['arms']==saved['arms'] and check['populations']==es['populations']==POPS
            and check['canonical_evaluation_seal_sha256']==sha(ev/'EVALUATION_SEAL.json')
            and check['prediction_bindings']==es['prediction_bindings']
            and check['reference_sha256']==es['reference_sha256']
            and check['success_classifications_exact']is True,'Canonical/independent ancestry differs')
        require(saved['canonical_artifacts']=={n:es['artifacts'][n]for n in('SCORED_ROWS.jsonl.gz','SUMMARY.json','PAIRED.json')},
            'Public scalar source binding differs')
        for name in('SUMMARY.json','PAIRED.json'):inputs.bind(ev/name,es['artifacts'][name])
        require(set(saved['independent_artifacts'])=={'INDEPENDENT_ROWS.jsonl.gz','INDEPENDENT_SUMMARY.json',
            'INDEPENDENT_PAIRED.json','INDEPENDENT_VERIFICATION.json'},'Full independent artifact binding required')
        for name,digest in saved['independent_artifacts'].items():
            if name.endswith('.json'):inputs.bind(ind/name,digest)
        # Mixed pose-valued scored rows and prediction payloads are never read.
        if reference is not None:require(reference==es['reference_sha256'],'Cross-stage references differ')
        reference=es['reference_sha256'];seen.update(es['arms'])
    require(seen==set(metric_seal['arms'])and reference==public_bindings['reference_sha256'],'Canonical union differs from public40')


def paired(inputs,spec,table):
    root,seal=bundle(inputs,spec,'PUBLIC_CONTRASTS_SEAL.json')
    require(seal['public_poses_or_GT']is False and seal['new_metric_or_statistical_calculations']==0,
        'Saved paired whitelist required')
    require(set(seal['artifacts'])=={'ALL_POPULATION_PAIRED_CONTRASTS.json','CONTRAST_SCHEMA.json'},'Exact paired bundle required')
    data=inputs.read(root/'ALL_POPULATION_PAIRED_CONTRASTS.json');require(set(data)==set(POPS),'All four paired populations required')
    expected={name+'_s'+str(s):(before+'_s'+str(s),after+'_s'+str(s))for before,after,name in PAIR_DEFS for s in SEEDS}
    for pop,n in POPS.items():
        require(set(data[pop])==set(expected),'All30 registered paired contrasts required')
        for name,r in data[pop].items():
            require((r['before'],r['after'])==expected[name]and r['denominator']==n,'Paired arm/denominator differs')
            ids=[fid for g in GROUPS for fid in r['ids'][g]]
            require(len(ids)==len(set(ids))==n and all(len(r['ids'][g])==r['counts'][g]for g in GROUPS),'Paired full identities lost')
            require(r['success_difference_count']==r['counts']['RECOVERY']-r['counts']['DAMAGE']
                ==integer(table[(pop,r['after'])]['successes'])-integer(table[(pop,r['before'])]['successes']), 'Paired count identity differs')
            require(set(r['deltas'])=={'success_rate','translation_cm','rotation_deg'},'Paired axes differ')
            if pop=='PRIMARY192':require(len(r['recording_clusters'])==len(r['leave_one_cluster_out'])==6,'All six recording groups required')
            for d in r['deltas'].values():
                for key in('recording_cluster_CI','named_session_cluster_sensitivity_CI'):
                    ci=d[key]
                    if ci['cluster_count']>=2:require(ci['resamples']==10000 and ci['seed']==20260917,'Registered paired uncertainty differs')
    return data,seal


def budget(inputs,spec,metrics_sha):
    root,seal=bundle(inputs,spec,'ERROR_BUDGET_SEAL.json')
    require(seal['schema']=='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1'and seal['campaign_scope']=='ALL40_COMPLETED_ARMS'
        and seal['arms']==arms40()and seal['populations']==POPS and seal['input_export_seal_sha256']==metrics_sha
        and (seal['joint_rows'],seal['band_rows'],seal['conservation_rows'])==(800,1120,320)
        and seal['posthoc_descriptive']is True and seal['preregistered_test']is False
        and seal['selector_performance_claim']is False and seal['no_outlier_exclusion']is True
        and seal['independent_real_test_n']==0,'Final40 descriptive error-budget required')
    require(all(seal[k]==0 for k in('raw_reference_reads','pose_or_RGB_reads','new_pose_metric_calls','new_solves','new_fits')),'Error budget not scalar-only')
    require(set(seal['artifacts'])=={'JOINT_CATEGORIES.csv','ERROR_BANDS.csv','CONSERVATION_CHECKS.csv',
        'ERROR_BUDGET_SCHEMA.json','INPUT_BINDINGS.json'},'Final40 exact five scalar artifacts required')
    bindings=inputs.read(root/'INPUT_BINDINGS.json')
    require(bindings['export_seal_sha256']==metrics_sha,'Error-budget export binding differs')
    counts={'JOINT_CATEGORIES.csv':800,'ERROR_BANDS.csv':1120,'CONSERVATION_CHECKS.csv':320}
    for name,n in counts.items():require(len(csv_read(root/name))==n,'Error-budget rows missing')
    return seal


def frozen_choice(inputs,item):
    d=inputs.read(item['path'],item['sha256'])
    require(d['schema']=='F_FROZEN_SYSTEM_CHOICE_V1'and d['complete']is True
        and d['before_F_optional_union_and_H1']is True and d['primary_configuration_ranking']is False
        and d['primary_development_fallback_bit_used']is True and d['original120_direct_parity']is False
        and d['independent_real_test_n']==0,'Actual frozen VAL-selected F choice required')
    wanted={'S1','S2_C0','S2_CP','S2_CB','S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE'}
    require(set(d['own_validation'])==wanted and set(d['ranked_configuration_names'])==wanted,'Actual9 configuration choices required')
    require(d['all_negative_fallback']is False and d['selected_configuration']==d['ranked_configuration_names'][0],
        'Unsupported/unexecuted fallback or choice binding; never select a substitute in reporter')
    for value in d['own_validation'].values():
        require(value['validation_frames']==235 and set(value['by_seed'])==set(map(str,SEEDS)),'All3 own-VAL235 receipts required')
    return d


def fixed_cohorts(inputs,spec):
    fixed=inputs.read(spec['fixed_sets']['path'],spec['fixed_sets']['sha256'])['sets']
    require(set(fixed)==set(COHORTS)and all(len(set(fixed[k]))==len(fixed[k])==n for k,n in COHORTS.items()),'Original31/41/38/new3/old12 changed')
    require(set(fixed['PF_union38_candidate_absent'])<=set(fixed['original41_candidate_absent'])
        and set(fixed['PF_new_good'])==set(fixed['original41_candidate_absent'])-set(fixed['PF_union38_candidate_absent'])
        and set(fixed['PF_good_old_bank_good_H1_fail'])<=set(fixed['original31_selection_failure']), 'Original fixed cohort relationships differ')
    result={};seals={}
    for item in spec['banks']:
        seal=inputs.read(item['seal'],item['sha256']);require(seal['complete']is True,'Completed post-bank metadata required')
        seals[item['name']]=item['sha256']
        if item.get('kind')=='S1':
            require(seal['independent_fixed_set_equality']is True and seal['fixed_counts']==COHORTS,'S1 fixed-set proof differs')
            raw=inputs.read(item['summary'],item['summary_sha256']);source={'S1_UNION':{'fixed_cohorts':raw}}
        else:
            require(seal['fixed_sets_sha256']==spec['fixed_sets']['sha256']and seal['actual_new_pose_calls']==0,
                'Post-bank original fixed-set ancestry/new-call contract differs')
            raw=inputs.read(item['summary'],seal['artifacts']['SUMMARY.json']);source=raw
        for arm,record in source.items():
            require(arm in PREFIXES and arm not in result,'Unknown/duplicate post-bank arm')
            normalized={}
            for name,n in COHORTS.items():
                c=record.get('fixed_cohorts',{}).get(name)
                if c is None:
                    normalized[name]=dict(available=False,reason='MISSING_SAVED_COHORT_FIELD',n=n);continue
                require(c['n']==n,'Saved cohort denominator differs')
                generated=c.get('generated_good',c.get('union_generated_good'))
                eligible=c.get('eligible_good',c.get('union_eligible_good'))
                require(type(generated)is int and type(eligible)is int and 0<=eligible<=generated<=n,'Saved cohort admission counts differ')
                seeds={}
                for seed in map(str,SEEDS):
                    s=c['seeds'][seed]
                    ids=s['transition_ids']if 'transition_ids'in s else s['transitions']
                    require(set(ids)==set(GROUPS)and all(isinstance(ids[g],list)for g in GROUPS),'Saved transition ID lists missing')
                    allids=[fid for g in GROUPS for fid in ids[g]]
                    require(len(allids)==n and set(allids)==set(fixed[name]),'Cohort IDs redefined across arms/seeds')
                    selected=s.get('selected_good',s.get('union_selected_good'))
                    require(selected==len(ids['RECOVERY'])+len(ids['BOTH_SUCCESS'])and selected<=eligible,'Cohort selected/admission mismatch')
                    seeds[seed]=dict(selected_good=selected,transition_ids=ids)
                normalized[name]=dict(available=True,n=n,generated_good=generated,eligible_good=eligible,seeds=seeds)
            result[arm]=normalized
    require(set(result)==set(PREFIXES),'Post-bank metadata must cover all11 new systems')
    require(set(seals)=={'S1','S2','S3_S4','F'},'Exactly four completed post-bank aliases required')
    return dict(sets=fixed,arms=result),seals


def cost_summary(inputs,item):
    d=inputs.read(item['path'],item['sha256'])
    require(d['schema']=='PUBLIC_RECOVERY_AWARE_MASTER_COST_SUMMARY_V1'and d['complete_cost_measurement']is False
        and d['unknown_cost_token']=='UNKNOWN_NOT_ZERO'and d['all_declared_H1_fits_complete']is True
        and d['all_declared_H1_prediction_seals_complete']is True
        and d['H1']['completed_new_fits']==d['H1']['reservations']==33
        and d['H1']['tiny_head_real_prediction_forwards']==7854 and d['H1']['new_backbone_prediction_forwards']==0,
        'Actual33/11 prediction seals and preserved unknown costs required')
    return d


def saved_diagnostic(inputs,item):
    seal=inputs.read(item['seal'],item['sha256'])
    require(seal['complete']is True and seal['all3_selected_PL_endpoint_ancestries_checked']is True
        and all(seal[k]==0 for k in('new_reference_file_reads','new_pose_calls','new_model_forwards','new_fits','new_pose_scoring_calls')),
        'Closed finite saved-output diagnostic required')
    p=Path(item['path']);d=inputs.read(p,seal['artifacts'][p.name])
    require(d['complete']is True and d['observational_only']is True
        and d['cannot_identify_feature_cause_without_intervention']is True,'Observational diagnostic scope differs')
    proofs=d['PL_three_genuine_selected_update_support_counts']
    require(len(proofs)==3 and len({r['id']for r in proofs})==3
        and all(r['prior_pose_matches']==[] and r['producer_available']is True for r in proofs),
        'PL genuine endpoint ancestry differs')
    return dict(PL=proofs,IMAGE_feature_associations={k:d['IMAGE_named_feature_comparisons'][k]
        for k in('fixed_energy','full_energy','point_residual_mean_px')})


def statuses(spec):
    value=spec['stage_statuses'];require(set(value)=={'P0','S1','D_QCOV','S2','S3','S4','F','R'}and all(v in STATUS for v in value.values()),'Allowed7-enum stage status contract required')
    require(value['P0']=='COMPLETE_EXACT_REUSE'and value['D_QCOV']=='DIAGNOSTIC_ONLY'
        and all(value[k]=='COMPLETE_NEW'for k in('S1','S2','S3','S4','F','R')),'Final scalar report cannot silently claim incomplete scientific stages')
    return value


def terminal(inputs,item,source):
    p=Path(item['path']);d=inputs.read(p,item['sha256'])
    require(d['complete']is True and d['source_sha256']==sha(source),'Completed authentic controller source required')
    plan=inputs.read(p.parent/'PLAN.json',d['plan_sha256'])
    go=inputs.read(item['root_GO'],d['ROOT_GO_sha256'])
    require(go['approved']is True and go['plan_sha256']==d['plan_sha256']and go['controller_sha256']==sha(source),
        'Controller ROOT GO/plan/source binding differs')
    inputs.bind(p.parent/'ACTUAL_COMMAND_LEDGER.jsonl',d['actual_command_ledger_sha256'])
    for field in('sources','fixed_inputs'):
        for path,digest in plan[field].items():inputs.bind(path,digest)
    expected=DERIVED_STAGES if Path(source).name=='derived_completion.py'else SEQUENCE_STAGES
    require(d['stage_order']==plan['stage_order']==expected and plan['controller_sha256']==sha(source),'Controller full stages/source differs')
    require(set(d['checkpoints'])=={s+'.json'for s in plan['stage_order']},'Every declared checkpoint required')
    closed={}
    for name,digest in d['checkpoints'].items():
        cp=inputs.read(p.parent/'checkpoints'/name,digest);require(cp['complete']is True,'Incomplete source checkpoint')
        require(cp['identity']['input_bindings']['PLAN.json']==d['plan_sha256']
            and cp['identity']['input_bindings']['ROOT_GO']==d['ROOT_GO_sha256'], 'Checkpoint PLAN/GO binding differs')
        for path,value in cp['completed_files'].items():
            require(path not in closed or closed[path]==value,'Contradictory completed artifact binding')
            inputs.bind(path,value)  # byte authentication only, never payload decode
            closed[path]=value
    d['_authenticated_checkpoint_artifact_hashes']=closed
    return d


def method_source(old,repo):
    """Source-only resolution; this does not assert execution completion."""
    module_callables={20:'choose_asset',21:'extract_supports',23:'generic_residuals',24:'refine_s4',25:'refine_s4',
        26:'_image_update',27:'choose_starts',28:'hardware_info',29:'unproject',30:'run',31:'choose',32:'run',34:'train',35:'verify'}
    r=dict(old)
    if old['callable']=='MODULE':
        ordinal=int(old['id'].split('-')[-1])
        if ordinal==33:r['source']='scripts/research/pallet_bi_full_execution_v1/f_finish.py';r['callable']='choose'
        else:r['callable']=module_callables[ordinal]
    source=Path(repo)/r['source'];tree=ast.parse(source.read_text());name=r['callable'].rsplit('.',1)[-1]
    matches=[n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))and n.name==name]
    require(matches,'Method callable is absent from bound source: '+old['id']+'/'+name)
    r.update(source_sha256=sha(source),line=matches[0].lineno)
    return r


def method_ledger(inputs,item,repo,completion_bindings):
    """Preserve historical draft bytes and bind final execution to real seals."""
    rows=inputs.read(item['path'],item['sha256']);require(isinstance(rows,list)and rows,'Method-difference ledger required')
    result=[]
    for old in rows:
        r=method_source(old,repo);source=Path(repo)/r['source'];inputs.bind(source)
        stage=old['stage'];alias='S1'if stage in('BASELINE','S1')else'D'if stage=='D'else'F'if stage=='F'else'S2'if stage in('S2','FULLNORMAL')else'R'if stage=='R'else'S3_S4'
        status='COMPLETE_EXACT_REUSE'if stage=='BASELINE'else'DIAGNOSTIC_ONLY'if stage=='D'else'COMPLETE_NEW'
        r.update(historical_source=old['source'],historical_callable=old['callable'],historical_source_sha256=old['source_sha256'],historical_line=old['line'],
            historical_evidence_sha256=old['evidence_sha256'],historical_status=old.get('status'),
            source_verification_status='SOURCE_CONFIRMED;COMPLETED_BOUND_OUTPUT',status=status,execution_status=status,
            status_scope='Scientific/report metadata complete;publication/Git remain pending;whole_campaign_complete=false',
            completion_binding_alias=alias,completion_binding_sha256=completion_bindings[alias])
        updates={
            'METHOD-06':'Own CAL and corrected full-normal production, own9 H1 fits and all238 predictions/evaluation are complete. Calibration coverage alone is not the end-to-end effect.',
            'METHOD-07':'Recovery changed scheduling/receipts only. Completed corrected banks/predictions are bound; original lost in-memory work remains UNKNOWN_NOT_ZERO.',
            'METHOD-32':'Policy bundle1–4 with4 tied; downstream B/I summaries remain both. Full core, frozen own-VAL choice, own6 H1 fits and evaluated238 predictions are complete.',
            'METHOD-33':'The actual own-VAL frozen choice and registered development fallback bit precede F union/H1. No PRIMARY winner ranking and no new independent-real confirmation.',
            'METHOD-35':'All four canonical/independent bundles and four populations/40arms are complete. PRIMARY remains development;independent_real_test_n=0.'}
        if old['id']in updates:r['interpretation_or_limit']=updates[old['id']]
        require(not any(word in(r['actual_method']+' '+r['interpretation_or_limit']).lower()for word in('evaluationpending','predictions/evaluation pending','claim yet','recovery active')),
            'Stale unfinished method wording requires reviewed update: '+r['id'])
        result.append(r)
    return result


def format_n(value,digits=3):
    value=number(value);return 'NA'if value is None else f'{value:.{digits}f}'


def prefix_arm(prefix,seed=SEEDS[0]):return prefix+'_s'+str(seed)


def render(table,contrasts,choice,cost,cohorts,d_rows,stage,links,diagnostic=None):
    def metric(a,p='PRIMARY192'):return table[(p,a)]
    def counts(a,p='PRIMARY192'):return '/'.join(metric(prefix_arm(a,s),p)['successes']for s in SEEDS)
    old=metric(prefix_arm('D1_H1_MLP'));bi=metric(prefix_arm('F_BI'));common=metric(prefix_arm('F_COMMON'))
    delta=integer(bi['successes'])-integer(old['successes'])
    common_delta=integer(common['successes'])-integer(old['successes'])
    finalpairs=[contrasts['PRIMARY192']['final_matched_BI_effect_s'+str(s)]for s in SEEDS]
    policy='/'.join(f'{p["success_difference_count"]:+d}'for p in finalpairs)
    pd=[p['success_difference_count']for p in finalpairs]
    policy_verdict=('세 seed 모두 B/I 성공 수 증가가 관찰됐다'if all(v>0 for v in pd)else
        '세 seed 모두 성공 수가 같았다'if all(v==0 for v in pd)else
        '일관된 B/I 추가 성공 이득은 관찰되지 않았다')
    pf_differences=[]
    for prefix in('F_COMMON','F_BI'):
        final=metric(prefix_arm(prefix))
        for pf in('POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR'):
            reference=metric(pf);success_delta=integer(final['successes'])-integer(reference['successes'])
            if all(integer(r[a+'_n'])==192 for r in(final,reference)for a in('T','R')):
                dt=number(final['T_mean_cm'])-number(reference['T_mean_cm']);dr=number(final['R_mean_deg'])-number(reference['R_mean_deg'])
                description=f'Δ성공{success_delta:+d}, ΔT평균{dt:+.3f}cm/ΔR평균{dr:+.3f}°'
            else:description=f'Δ성공{success_delta:+d}; 공통 available paired 평균 저장값 없음(NA)'
            pf_differences.append(prefix+'−'+pf+': '+description)
    lines=['# 팔레트 B/I 전체 실행 결과','',
        f'**전체 성능:** 최종 MATCHED_BI의 대표 seed20261018 성공은 {bi["successes"]}/192로 원 H1의 {old["successes"]}/192 대비 {delta:+d}건이다. T/R 평균은 {format_n(bi["T_mean_cm"]) }cm/{format_n(bi["R_mean_deg"])}°이며, 원 H1은 {format_n(old["T_mean_cm"])}cm/{format_n(old["R_mean_deg"])}°다. 성공·평균·tail을 함께 비교하며 한 지표의 개선을 전체 지배로 해석하지 않는다.',
        f'MATCHED_COMMON은 {common["successes"]}/192로 원 H1 대비 {common_delta:+d}건이며, T/R 평균은 {format_n(common["T_mean_cm"])}cm/{format_n(common["R_mean_deg"])}°다.',
        '대표 seed의 PF 대비 절대오차 차이: '+'; '.join(pf_differences)+'. 두 군 모두192 available이면 저장 평균차는 같은192 ID의 paired mean delta와 대수적으로 같다. 별도 PF confidence interval은 새로 계산하지 않았다.',
        f'**B/I 추가 효과:** MATCHED_BI−MATCHED_COMMON의 PRIMARY 성공 차이는 세 seed 순서로 **{policy}건**이다. {policy_verdict}. 이는 고정 정책 묶음의 관측 결과이며, 원 H1 대비 변화와 구별한다. 아래 촬영군·paired interval을 함께 읽어야 하며 독립 신규 실사 확증은 **0건**이다.',
        '현재 권고는 N3/SubPix를 유지하는 것이다. 제거군 근거가 없으므로 이 실행으로 제거를 권하지 않는다. 등록된 최종 구성을 저장 지표대로 평가하고 새 실사에서 독립 확증한다.','',
        '과학·파생 산출물과 이 보고서의 준비 완료를 나타낸다. 공개 패키지·Git commit/push·원격 SHA 확인은 별도 ROOT 단계이며 `whole_campaign_complete=false`다. 대표 seed는 사전 지정20261018이고 좋은 seed로 바꾸지 않았다.','',
        '## 1. 실제 완료와 재사용','',
        '| 단계 | 상태 | 실제 범위 |','|---|---|---|']
    scope={'P0':'원 입력·기준·고정 ID·PF tensor 정확 재사용','S1':'D1+PF2 union과 own H1','D_QCOV':'고정 공통 graph q/Σ 진단; 전체 방법과 분리',
        'S2':'전체 normal C0/CP/CB와 own H1','S3':'COMMON/BI generic support 제안+union','S4':'PL/CONTOUR/IMAGE 실제 새 pose+union',
        'F':'고정 VAL 선택 auxiliary를 양쪽에 동일 적용한 matched 정책','R':'독립 재집계·전체 gallery·scatter·비용·재현/보고서; 공개는 대기'}
    lines += [f'| {k} | {stage[k]} | {scope[k]} |'for k in scope]
    lines += ['', '실사238=PRIMARY192(TEST49+REMAINDER143)+CAL46을 모두 유지했다. 실패/no-pose/fallback은 이진 분모에서 빠지지 않는다. 연속오차는 available-only, sample variance/SD는 ddof1이며 trimming은 없다. CAL 및 반복 사용 PRIMARY는 개발 자료다. 독립 checker 일치는 같은 자료의 계산 검증이며 새 독립 test를 만들지 않는다.','',
        '## 2. 생략·고정한 것과 비교 범위','',
        '원 spine의 N3/SubPix를 고정한 채 추가 후보·관측 정책을 비교했다. 제거군을 실행하지 않았으므로 이 결과로 N3/SubPix가 필요 없거나 이득의 원인이라고 말할 수 없다. 큰 관측 모델의 재학습, 새 시퀀스 수집, human material-owner annotation은 하지 않았다. 현재 라이브러리·등록 자산과 비학습 grayscale DIS를 사용했으며 DeepIM/SRT3D 논문 전체 재현을 주장하지 않는다.',
        '새 MATCHED_COMMON/BI는 동일한 가용 cue·mask·geometry 및 후단 역할 특징을 가진 강한 대조군이다. 원120 H1과 full-normal 후보/특징·일부 입력 채널 범위가 달라 원 방법의 직접 parity나 모든 B/I 정보 제거군으로 부르지 않는다. native S는 모델 투영만 유지하며 새 이미지 측정·regression·seed 관측으로 추론하지 않았다. ['+'방법 상세]('+links['methodology']+')와 [구조 검토]('+links['structure']+')를 함께 본다.','',
        '## 3. 전체 방법의 성능과 tradeoff','',
        '성공은 T≤5cm이고 C2 회전오차≤5°다. C2는 Y축180°만 허용하며90° 혼동은 오류다. 아래 성공 열은20261018/19/20이며 정적 N0/E0/PF는 한 번의 동결 결과다. T/R 평균과 tail은 대표 seed의 저장값이다.','',
        '| 군 | PRIMARY 성공 | T 평균 cm | R 평균 ° | T>50 / >100 / R>45 | no-pose |','|---|---:|---:|---:|---:|---:|']
    representatives=['N0','E0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR',prefix_arm('D1_H1_MLP')]+[prefix_arm(a)for a in PREFIXES]
    for arm in representatives:
        r=metric(arm);prefix=arm.rsplit('_s',1)[0]if '_s2026'in arm else None
        successes=counts(prefix)if prefix else r['successes']
        lines.append(f'| {prefix or arm} | {successes} | {format_n(r["T_mean_cm"])} | {format_n(r["R_mean_deg"])} | {r["T_gt50cm"]}/{r["T_gt100cm"]}/{r["R_gt45deg"]} | {r["no_pose"]} |')
    lines += ['',f'[전체40 군 PRIMARY: 평균·sample variance·SD·중앙값·P90/P95·최댓값·tail]({links["full40"]}) · [네 population 전체160행 원정밀 값]({links["metrics"]}) · [오차 예산40]({links["budget"]}). PF 두 anchor의 한 결과를 세 독립 PF 실험으로 늘리지 않았다. 오차 예산은 저장 scalar의 기술적 분해이며 새 사전등록 검정이나 selector 이득 추정이 아니다.','',
        '## 4. B/I−COMMON과 촬영군 차이','',
        '| 비교 | seed | recovery / damage | Δ 성공 | Δ percentage point | 6촬영군95% CI, pp |','|---|---:|---:|---:|---:|---:|']
    for name in ('S3_role_policy','final_matched_BI_effect'):
        for seed in SEEDS:
            p=contrasts['PRIMARY192'][name+'_s'+str(seed)];ci=p['deltas']['success_rate']['recording_cluster_CI']['CI95']
            interval='NA'if ci is None else f'[{100*ci[0]:.3f}, {100*ci[1]:.3f}]'
            lines.append(f'| {name} | {seed} | {p["counts"]["RECOVERY"]}/{p["counts"]["DAMAGE"]} | {p["success_difference_count"]:+d} | {p["success_difference_percentage_points"]:.3f} | {interval} |')
    cluster_text=[]
    for p in finalpairs:
        cluster_text.append(', '.join(f'{k}:{v["success_difference_count"]:+d}'for k,v in p['recording_clusters'].items()))
    lines += ['', 'MATCHED BI−COMMON의 촬영군별 Δ성공(세 seed 순): '+' / '.join(cluster_text)+'.',
        f'[모든30 paired 비교·4 population·각 recording/named-session interval·leave-one-cluster 결과]({links["paired"]})에 전체 회복/손상 ID를 유지한다. 6 recording cluster,10000 draws의 저장 CI를 복사했으며 세 seed가 새 독립 이미지 수를 늘리지는 않는다.',
        f'최종 auxiliary 구성은 **{choice["selected_configuration"]}**다. actual own-VAL235 세 seed의 median success→median loss→실측 production-wall 순서로 사전 고정한 결과를 사용했다. PRIMARY로 후보를 재순위화하지 않았다. 등록된 all-negative 개발 fallback bit는 평가됐고 실제 선택의 `all_negative_fallback={str(choice["all_negative_fallback"]).lower()}`다. 원120 fallback을 새 이름으로 실행한 것으로 쓰지 않는다.','',
        '## 5. 확인된 연결: 후보 생성→준입→선택','',
        '| 군 | 원31 선택18/19/20 | 원41 generated/eligible; 선택 | PF후38 generated/eligible; 선택 |','|---|---:|---:|---:|']
    def ccell(arm,name,full=False):
        c=cohorts['arms'][arm][name]
        if not c['available']:return 'NA: saved field 없음'
        selected='/'.join(str(c['seeds'][str(s)]['selected_good'])for s in SEEDS)
        return f'{c["generated_good"]}/{c["eligible_good"]}; {selected}'if full else selected
    for arm in PREFIXES:lines.append(f'| {arm} | {ccell(arm,"original31_selection_failure")} | {ccell(arm,"original41_candidate_absent",True)} | {ccell(arm,"PF_union38_candidate_absent",True)} |')
    lines += ['',f'집합31/41/38와 PF-new-good3/old12는 원래 ID로 고정했다. [다섯 cohort 모든 군·seed 수치와 transition ID]({links["cohorts"]})를 남기며 새 군의 자기 실패 평균끼리 빼지 않는다. good 후보 coverage가 늘어도 실제 H1 선택이 좋아진다는 보장은 없다.',
        'PL은 기존 native point/typed-normal과 pose에 따라 갱신한 native-outline evidence를 쓰며 이 branch는 CAD/proxy renderer를 쓰지 않는다. 따라서 PL 개선을 인증 material CAD·전체 RGB·물리 owner 정합의 성공으로 설명할 수 없다. IMAGE의 good endpoint 존재 후 낮은 H1 순위는 저장된 feature/logit 연관으로 설명 가능한 손실 위치이며 원인 개입 실험은 아니다.',
        '[S1/D 짧은 근거](S1_D_EVIDENCE.md) · [S3/S4 짧은 근거](S3_S4_EVIDENCE.md). q/Σ/가족 전환 D는 제한 공통 graph 진단이며 전체 H1 B/I 효과와 합치지 않는다. 원 D는Σ+Γ donor이고 corrected D는COMMON reliability Γ를 고정한다.','',
        '## 6. 미식별 원인과 최소 추가 증거','',
        '남은 오차를 feature 하나, proxy, 관측 cue 또는 학습 분포의 단일 원인으로 식별하지 못했다. 필요한 최소 증거는 독립 physical correspondence/owner 확인, signed axes·instance geometry의 독립 등록 검사와 새 촬영 실사 검증이다. 특정 loss feature의 원인 효과에는 다른 요소를 고정한 추가 개입이 필요하다. 현재 결과를 본 뒤 threshold·seed·pose·F 구성을 다시 고르지 않았다.',
        'S3 kept-original의 정확 factor ID와 PL 매 update outline 원좌표가 모두 저장된 것은 아니므로 완전한 좌표 ancestry를 주장하지 않는다. raw support가 없는 old factor는 UNKNOWN_CORRELATION으로 유지될 수 있으며 source-ID/pixel8px dedup은 인증된 물리 독립성이 아니다. 고정 row/endpoint/support/cost binding의 기록 범위를 방법 문서에 명시했다.','',
        '## 7. 독립 확증·자산·공개 한계','',
        'human owner 인증은0이며 exact real CAD identity와 signed axes도 미인증이다. 합성 VAL235의 exact named GLB 부재/registered proxy 사용은 확인된 연결·분포 한계이지 이 결과만으로 형상이 실제 틀렸다는 실증이 아니다. 자동 structural proxy를 bare cuboid material CAD나 독립 physical truth로 승격하지 않는다. 물리 owner/자산 인증은 `NOT_APPLICABLE_WITH_REASON`의 별도 한계이고 실제 자동 S3/S4 전체를 BLOCKED로 바꾸지 않는다.',
        f'[전체 transition gallery]({links["gallery"]})와 [full-axis scatter]({links["scatter"]})는 파생 시각자료다. 제한된 pixel/AI 검토는 human 실사 정답이나 전량 pixel 검토가 아니다. 원24/private RGB·GT/Rt·weights·banks·등록 자산은 공개 패키지에서 제외하고 기존 승인 PRIMARY derived 범위만 공개한다.','',
        '## 8. 실제 실행 비용·재사용·완료 범위','']
    core=cost['fullnormal_runtime_minimize_calls'];d=cost['D_diagnostic_runtime_minimize_calls'];h=cost['H1']
    lines += [f'새 H1 fit은 실제 **{h["completed_new_fits"]}회/등록{h["reservations"]}회**이며 tiny real H1 prediction forwards는 **{h["tiny_head_real_prediction_forwards"]}**, 새 backbone prediction forward는 **{h["new_backbone_prediction_forwards"]}**다. full-normal minimize 알려진 lower bound는 **{core["known_lower_bound"]}**이고 D는 별개 {d["old_Sigma_plus_Gamma"]}+{d["fixed_Gamma"]}회다. S3 objective/Jacobian·DIS/PnP·render는 다른 단위로 분리하여 하나의 총 호출 수로 더하지 않는다.',
        f'원 D4 prefix {core["old_D4_prefix_included_not_added_again"]}회는 복구된 base 누적값에 이미 포함되어 다시 더하지 않는다. exact reuse는 복사된 FRAME의 intrinsic 비용을 새 optimizer 비용으로 청구하지 않았다. 원 중단{core["original_interrupted_arm_directories"]} arm의 미기록 작업, 구형 import/finalization 등은 성공적 복구 뒤에도 **UNKNOWN_NOT_ZERO**다.',
        f'문서화된 START→비용 측정 cutoff elapsed는 **{cost.get("elapsed_at_measurement_cutoff_seconds","UNKNOWN_NOT_ZERO")}초**이며 범위는 `{cost.get("measurement_cutoff_scope")}`다. 공개/Git 후 최종 workflow elapsed가 아니다. child wall 합·서로 겹친 service elapsed·내부 fit timer를 campaign elapsed로 쓰지 않고, inclusive service CPU와 그 자식/내부 타이머도 중복 합산하지 않는다.',
        f'추가 파생 작업 서비스의 inclusive CPU/elapsed는 {cost["_derived_service"]["CPU_seconds"]}초/{cost["_derived_service"]["elapsed_wall_seconds"]}초다. master 감사 child의 외부 CPU/wall은 {cost["_master_child"]["child_CPU_seconds"]}초/{cost["_master_child"]["wall_seconds"]}초로 내부 timer 밖 출력 write/import/teardown을 포함한다. 이 값들도 자식·내부 범위와 중복 합산하지 않는다.',
        f'[Typed 비용 원장]({links["cost"]}) · [파생 서비스 비용]({links["cost_fifth"]}) · [cache/환경/모델/교정기/실제argv/source 인덱스]({links["provenance"]}) · [재현 절차]({links["reproduce"]}). 이 보고서는 저장 scalar와 봉인 메타데이터만 연결했다. 새 pose/GT 오차계산·fit·forward·render·bootstrap·선택은0회다.']
    return '\n'.join(lines)+'\n'


def validate_links(text,allowed):
    for target in re.findall(r'\]\(([^)]+)\)',text):
        u=urlsplit(target)
        if u.scheme:
            require(target.startswith(BASE),'Only official frozen GitHub links allowed');continue
        require(not u.netloc and not u.path.startswith('/'),'No private/absolute report links')
        path=posixpath.normpath(unquote(u.path))
        require(path in allowed,'Report links outside selected manifest: '+path)


def materialize(spec_path,GO_path,output):
    inputs=Inputs();spec=inputs.read(spec_path);go=inputs.read(GO_path)
    require(spec['schema']==SCHEMA and spec.get('template_only')is not True,'Reviewed actual report spec required')
    require(go.get('approved')is True and go.get('metadata_only')is True and go['source_sha256']==sha(__file__)
        and go['spec_sha256']==sha(spec_path),'Source/spec-bound ROOT report GO required')
    output=Path(output);require(not output.exists(),'Final-report outputs must be fresh; preserve every draft')
    for p,d in spec['source_bindings'].items():inputs.bind(p,d)
    stage=statuses(spec);table,ms,mb=metrics(inputs,spec['metrics']);authorities(inputs,spec['authorities'],mb,ms)
    paired_values,ps=paired(inputs,spec['paired'],table);bs=budget(inputs,spec['budget'],spec['metrics']['sha256'])
    choice=frozen_choice(inputs,spec['f_choice']);cost=cost_summary(inputs,spec['cost'])
    waiter=inputs.read(spec['cost_waiter']['path'],spec['cost_waiter']['sha256'])
    require(waiter['complete']is True and waiter['campaign_complete']is False
        and waiter['source_sha256']==sha(Path(spec['repo'])/'scripts/research/pallet_bi_full_execution_v1/cost_waiter.py')
        and waiter['output_bindings'][str(Path(spec['cost']['path']).resolve())]==spec['cost']['sha256'],
        'Completed external cost-wrapper parent required')
    fifth=inputs.read(spec['fifth_service']['path'],spec['fifth_service']['sha256'])
    require(fifth['complete']is True and fifth['Result']=='success'and int(fifth['ExecMainStatus'])==int(fifth['NRestarts'])==0
        and waiter['output_bindings'][str(Path(spec['fifth_service']['path']).resolve())]==spec['fifth_service']['sha256'],
        'Completed fifth-service external accounting required')
    cost['_derived_service']=fifth;cost['_master_child']=waiter['actual_master_child']
    cohorts,bank_seals=fixed_cohorts(inputs,spec['cohorts'])
    diagnostic=saved_diagnostic(inputs,spec['saved_diagnostic'])
    derived=terminal(inputs,spec['derived'],Path(spec['repo'])/'scripts/research/pallet_bi_full_execution_v1/derived_completion.py')
    sequence=terminal(inputs,spec['sequence'],Path(spec['repo'])/'scripts/research/pallet_bi_full_execution_v1/final_sequence.py')
    require(derived['upstream_sequence_seal_sha256']==spec['sequence']['sha256']and derived['scalar_arms']==40
        and derived['contrasts']==30,'Derived/sequence all40 ancestry differs')
    metric_path=str((Path(spec['metrics']['path'])/'EXPORT_SEAL.json').resolve())
    require(derived['_authenticated_checkpoint_artifact_hashes'].get(metric_path)==spec['metrics']['sha256'],
        'Metrics export is not bound to closed derived checkpoint')
    paired_path=str((Path(spec['paired']['path'])/'PUBLIC_CONTRASTS_SEAL.json').resolve())
    require(derived['public_contrasts_files'].get(paired_path)==spec['paired']['sha256'],
        'Paired export is not bound to closed derived terminal')
    droot,ds=bundle(inputs,spec['D'],'EXPORT_SEAL.json')
    require(ds['schema']=='PUBLIC_D_SAVED_SCALAR_EXPORT_SEAL_V1'and ds['raw_reference_decodes']==ds['mixed_pose_rows_decoded']==0,'D scalar-only export required')
    d_rows=csv_read(droot/'POPULATION_METRICS.csv')
    d_contrasts=csv_read(droot/'CONTRAST_COUNTS.csv')
    completion_bindings={**bank_seals,'D':spec['D']['sha256'],'R':spec['derived']['sha256']}
    ledger=method_ledger(inputs,spec['method_ledger_draft'],spec['repo'],completion_bindings)
    links=spec['links'];allowed=set(spec['selected_relative_files'])
    text=render(table,paired_values,choice,cost,cohorts,d_rows,stage,links,diagnostic)
    validate_links(text,allowed)
    s1calls=cost['S1_candidate_cost_evaluations']
    s1=['# S1/D 저장 근거','',f'S1 PRIMARY 성공(세 seed): '+ '/'.join(table[('PRIMARY192',prefix_arm('S1_UNION',s))]['successes']for s in SEEDS)+'.',
        f'원 D1+두 PF anchor union과 own-H1 결과다. 신규PF cost{s1calls["new_unique_PF"]}과 실제 old-cost fallback{s1calls["old_cost_fallback_reconstruction"]}를 포함한{s1calls["scientific_actual"]} 평가를 constructed work로 빼지 않는다. 별도 postseal diagnostic{s1calls["postseal_diagnostic"]}는 다른 범위다.',
        'D는 oldΣ+Γ와 COMMONΓ고정 진단을 구별한다. 제한 WD 공통 graph에서 source-compatible103/source-changing135를 분리하며 전체 H1 정책의 인과 분해로 바꾸지 않는다.','',
        '| D variant | PRIMARY cell | 성공/192 |','|---|---|---:|']
    for r in d_rows:
        if r['population']=='PRIMARY192':s1.append(f'| {r["variant"]} | {r["cell"]} | {r["successes"]}/192 |')
    s1+=['','| D variant | contrast | family switches | within-before-family Δsuccess | family-readout Δsuccess | total Δsuccess |',
        '|---|---|---:|---:|---:|---:|']
    for r in d_contrasts:
        if r['population']=='PRIMARY192':s1.append(f'| {r["variant"]} | {r["contrast"]} | {r["family_switches"]} | {r["within_before_family_success_delta"]} | {r["family_readout_success_delta"]} | {r["joint_success_delta"]} |')
    s1+=['',f'[전체 D scalar]({links["D"]}) · [고정 cohort]({links["cohorts"]}) · [전체 보고서](FINAL_REPORT_KO.md).']
    s34=['# S3/S4 저장 근거','',
        'COMMON/BI는 같은 cue·mask·geometry 접근에서 support 채택 정책을 달리한다. 후단 역할 특징은 양쪽에 남는다. PL은 native point/typed-normal+pose-updated outline이며 renderer를 쓰지 않는다. CONTOUR/IMAGE는 registered structured proxy와 단일영상 최대2 start/4 update의 실제 pose 제안이다. IMAGE는 native grayscale DIS+render-depth/PnP이며 RGB 학습 정합 전반의 재현이 아니다.',
        '원41/후38에서 good endpoint 생성·준입과 실제 선택을 분리한다. 저장된 rank/feature/logit 패턴은 선택 손실 위치의 관찰이지 원인 귀속이 아니다. physical owner UNKNOWN/human0와 미인증 asset/signed axes를 유지한다.',
        'S3는 raw support가 없는 old factor를 UNKNOWN_CORRELATION으로 유지할 수 있다. source-ID/pixel8px dedup과 저장된 support/raw-factor ancestry는 물리적 독립성의 인증이 아니며, kept-original 정확 ID가 모두 저장된 것으로 쓰지 않는다.',
        'PL의 고정41 선택 회복3개는 '+', '.join(r['id']+' N0/update'+str(r['update'])for r in diagnostic['PL'])+'이며 prior S1/PF pose 일치가 없는 genuine 새 endpoint다. 모든3개 ancestry를 함께 보존한 결과이며 정확 material ownership의 인증은 아니다.',
        'IMAGE best-good−selected의 저장 feature 연관: '+', '.join(k+': best-good-higher '+str(v['best_good_higher'])+'/'+str(v['valid_pairs'])for k,v in diagnostic['IMAGE_feature_associations'].items())+'. 같은 이미지의 세 seed 반복이며 feature 원인 효과로 해석하지 않는다.',
        f'[전체40 원정밀 수치]({links["full40"]}) · [전체 paired]({links["paired"]}) · [고정 cohort]({links["cohorts"]}) · [전체 보고서](FINAL_REPORT_KO.md).']
    for doc in ('\n'.join(s1),'\n'.join(s34)):validate_links(doc,allowed)
    inputs.stable();output.mkdir(parents=True,exist_ok=False)
    primary=[table[('PRIMARY192',a)]for a in arms40()]
    representative=[r for r in primary if not re.search(r'_s202610(?:19|20)$',r['arm'])]
    write_csv(output/APPENDICES[0],FIELDS,primary);write_csv(output/APPENDICES[1],FIELDS,representative)
    compact=[]
    for name,r in paired_values['PRIMARY192'].items():
        row=dict(contrast=name,before=r['before'],after=r['after'],denominator=r['denominator'],
            success_difference_count=r['success_difference_count'],success_difference_percentage_points=r['success_difference_percentage_points'],**r['counts'])
        for axis,d in r['deltas'].items():
            ci=d['recording_cluster_CI']['CI95'];row.update({axis+'_mean':d['mean'],axis+'_CI95_low':None if ci is None else ci[0],axis+'_CI95_high':None if ci is None else ci[1]})
        compact.append(row)
    write_csv(output/APPENDICES[2],list(compact[0]),compact)
    cohort_rows=[]
    for arm,cc in cohorts['arms'].items():
        for name,c in cc.items():
            for seed in map(str,SEEDS):
                cohort_rows.append(dict(arm=arm,cohort=name,seed=seed,n=c['n'],available=c['available'],
                    generated_good=c.get('generated_good'),eligible_good=c.get('eligible_good'),
                    selected_good=c['seeds'][seed]['selected_good']if c['available']else None,reason=c.get('reason')))
    write_csv(output/APPENDICES[3],list(cohort_rows[0]),cohort_rows);write_json(output/APPENDICES[4],cohorts)
    for name,value in zip(DOCUMENTS,(text,'\n'.join(s1)+'\n','\n'.join(s34)+'\n')):
        with(output/name).open('x')as f:f.write(value)
    write_json(output/'METHOD_DIFFERENCE_LEDGER.json',ledger)
    write_csv(output/'METHOD_DIFFERENCE_LEDGER.csv',list(dict.fromkeys(k for r in ledger for k in r)),ledger)
    stage_rows=[dict(stage=k,status=v,scope='Scientific/report component complete;publication/Git remain pending')for k,v in stage.items()]
    write_json(output/'STAGE_STATUS.json',dict(schema='FINAL_REPORT_STAGE_STATUS_V1',stages=stage_rows,
        scientific_and_derived_complete=True,report_ready=True,whole_campaign_complete=False,publication_pending=True,
        physical_owner_status='NOT_APPLICABLE_WITH_REASON',physical_owner_reason='Independent human owner/instance/signed-axis certification absent;automatic stages executed',
        human_owner_certifications=0,independent_real_test_n=0))
    write_csv(output/'STAGE_STATUS.csv',('stage','status','scope'),stage_rows)
    inputs.stable()
    receipt=dict(schema=SEAL,complete=True,source_sha256=sha(__file__),scientific_and_derived_complete=True,
        report_ready=True,whole_campaign_complete=False,publication_pending=True,
        metrics_export_seal_sha256=spec['metrics']['sha256'],paired_export_seal_sha256=spec['paired']['sha256'],
        error_budget_seal_sha256=spec['budget']['sha256'],cost_summary_sha256=spec['cost']['sha256'],
        postbank_seal_sha256=bank_seals,artifacts={n:sha(output/n)for n in APPENDICES},documents={n:sha(output/n)for n in DOCUMENTS},
        metadata_documents={n:sha(output/n)for n in('METHOD_DIFFERENCE_LEDGER.json','METHOD_DIFFERENCE_LEDGER.csv','STAGE_STATUS.json','STAGE_STATUS.csv')},
        input_bindings=inputs.bindings,all_population_summary_rows=160,primary_rows=len(primary),representative_rows=len(representative),
        paired_primary_rows=len(compact),fixed_cohort_rows=len(cohort_rows),independent_real_test_n=0,human_owner_certifications=0,
        raw_reference_reads=0,pose_payload_reads=0,model_tensor_reads=0,new_pose_metric_calls=0,new_solves=0,new_fits=0,
        new_forwards=0,new_renders=0,new_bootstrap_calls=0,F_choice_calls=0,created_epoch=time.time())
    receipt['payload_read_counter_scope']='Numeric payload decoding; recorded byte-only SHA authentication is not pose/reference/model deserialization'
    write_json(output/'REPORT_SEAL.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in('spec','root-GO','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();result=materialize(a.spec,a.root_GO,a.output)
    print(json.dumps(dict(complete=result['complete'],whole_campaign_complete=False,output=a.output,new_scientific_calls=0)))


if __name__=='__main__':main()
