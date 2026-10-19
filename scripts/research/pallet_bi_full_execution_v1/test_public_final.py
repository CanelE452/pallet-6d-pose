"""Constructed final-publication metadata fixtures; no dataset or model loads."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from . import public_package as P
from . import public_final as F


def put(path,value):
    path=Path(path);P.write_new(path,value);return path


def binary(path,content):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);return path


def fixture(root):
    repo=root/'repo';run=root/'run';code=repo/P.CODE_PREFIX;docs=repo/P.DOC_PREFIX
    code.mkdir(parents=True);docs.mkdir(parents=True);run.mkdir()
    for name in ('final_sequence.py','derived_completion.py','cost_waiter.py','public_final.py','error_budget.py','export_results.py','final_report.py'):
        (code/name).write_text('"""Constructed code, no scientific execution."""\n')
    for name in ('FINAL_REPORT_KO.md','METHODOLOGY_KO.md','STRUCTURE_REVIEW_NOTES_KO.md','REPRODUCE.md','S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md'):
        (docs/name).write_text('Constructed final document.\n')
    stage_statuses={s:'DIAGNOSTIC_ONLY'if s=='D_QCOV'else'COMPLETE_NEW'for s in ('P0','S1','D_QCOV','S2','S3','S4','F','R')}
    stage_rows=[dict(stage=stage,status=status,scope='Constructed analysis;publicationpending')for stage,status in stage_statuses.items()]
    put(docs/'STAGE_STATUS.json',dict(stages=stage_rows))
    (docs/'STAGE_STATUS.csv').write_text('stage,status,scope\n'+''.join(r['stage']+','+r['status']+','+r['scope']+'\n'for r in stage_rows))
    put(docs/'METHOD_DIFFERENCE_LEDGER.json',[dict(stage='S1',status='COMPLETE_NEW',difference='Constructed')for _ in range(38)])
    (docs/'METHOD_DIFFERENCE_LEDGER.csv').write_text('stage,status,difference\n'+''.join('S1,COMPLETE_NEW,Constructed\n'for _ in range(38)))
    d1=root/'d1';rights=put(d1/'RIGHTS_SCOPE_BINDING.json',dict(approved_ids=['id'+str(i)for i in range(192)]))
    privacy=put(d1/'DISPLAY_PRIVACY_PLAN.json',dict(constructed=True))
    checkpoint_fits={};ledger=[];events=[]
    for arm in F.ARMS:
        output=run/'training'/arm
        for seed in F.SEEDS:
            binding=dict(features_sha256='CONSTRUCTED',normalizer_sha256='HASH_ONLY',sources={str(code/'final_sequence.py'):P.sha(code/'final_sequence.py')})
            row=dict(arm=arm,seed=seed,attempt=len(ledger)+1,output=str(output),bindings=binding,registered_before_optimizer=True)
            ledger.append(row);folder=output/(arm+'_s'+str(seed))
            best=binary(folder/'best.pt',b'OPAQUE_CONSTRUCTED_NOT_A_MODEL')
            last=binary(folder/'last.pt',b'OPAQUE_CONSTRUCTED_NOT_A_MODEL')
            hist=put(folder/'HISTORY.json',[dict(epoch=1)])
            receipt=dict(complete=True,**{k:row[k]for k in ('arm','seed','attempt','bindings')},actual_new_training_runs=1,
                parameters=16321,optimizer_steps=1,training_frame_exposures=32,resumed_same_optimizer=False,
                checkpoint_sha256=P.sha(best),last_checkpoint_sha256=P.sha(last),history_sha256=P.sha(hist))
            rp=put(folder/'RUN_RECEIPT.json',receipt)
            checkpoint_fits[(arm,seed)]=receipt
            events.append(dict(event='COMPLETE',arm=arm,seed=seed,attempt=row['attempt'],optimizer_steps=1,receipt_sha256=P.sha(rp)))
    (run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in ledger))
    (run/'H1_EXECUTION_EVENTS.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in events))
    for arm,folder in F.prediction_paths(run).items():
        blob=binary(folder/'PREDICTIONS.jsonl.gz',b'OPAQUE_CONSTRUCTED_NOT_POSES')
        put(folder/'PREDICTION_SEAL.json',dict(complete=True,rows=238,before_reference_decode=True,current_real_reference_reads=0,
            ordered_ids=['id'+str(i)for i in range(238)],checkpoints={arm+'_s'+str(seed):dict(checkpoint_sha256=checkpoint_fits[(arm,seed)]['checkpoint_sha256'])for seed in F.SEEDS},
            tiny_head_forward_calls=714,new_backbone_forwards=0,predictions_sha256=P.sha(blob)))
    oldart={};oldgallery=[];visualgalleries={}
    def gallery(folder):
        content=dict(partition='TEST',raw_RGB_copied=False,reference_pose_or_GT_published=False,
            observations=[dict(q_xy=[10.,20.],reliability_weight=.5,fit_reliability_weight=.75,
                conditional_covariance_px2=[[1.,0.],[0.,2.]],source='CONSTRUCTED',dimension=1)])
        paths=[put(folder/'case_0000.json',content)]
        paths.extend(binary(folder/name,b'\xff\xd8\xffCONSTRUCTED')for name in ('case_0000.jpg','case_0000_families.jpg','case_0000_context.jpg'))
        page=folder/'index.html';page.write_text('<a href="case_0000.json">metadata</a><img src="case_0000.jpg">');paths.append(page)
        mp=put(folder/'GALLERY_MANIFEST.json',dict(complete=True,CAL_RGB_opened=0,raw_RGB_copied=False,public_reference_pose_or_GT=False,
            case_count=1,cases=[dict(id='id0',panel='case_0000.jpg',family_panel='case_0000_families.jpg',context_panel='case_0000_context.jpg',json='case_0000.json')],
            all_PRIMARY_transition_ids=['id0'],rights_scope_sha256=P.sha(rights),privacy_plan_sha256=P.sha(privacy),artifact_sha256={p.name:P.sha(p)for p in paths}))
        visualgalleries[str(mp.relative_to(run))]=P.sha(mp);return mp
    for name in P.GALLERIES:
        mp=gallery(run/name);oldgallery.append(dict(name=name,manifest_sha256=P.sha(mp)))
    derived=run/'DERIVED_COMPLETION_V1';derived_files=[]
    for name in F.NEW_GALLERIES:derived_files.append(gallery(derived/('GALLERY_'+name)))
    def scatter(folder):
        paths=[put(folder/'SCATTER_DATA.json',dict(constructed=True))]
        page=folder/'index.html';page.write_text('<a href="SCATTER_DATA.json">saved coordinates</a>');paths.append(page)
        paths.append(binary(folder/'REPRESENTATIVE_FULL.png',b'\x89PNG\r\n\x1a\nCONSTRUCTED'))
        svg=folder/'REPRESENTATIVE_FULL.svg';svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>');paths.append(svg)
        return put(folder/'FIGURE_MANIFEST.json',dict(complete=True,all_outliers_retained=True,PRIMARY_frames_per_dataset=192,artifacts={p.name:P.sha(p)for p in paths}))
    oldscatter=scatter(run/'S1_S3_S4_COMPARISON_FIGURES');oldart[str(oldscatter)]=P.sha(oldscatter)
    derived_files.append(scatter(derived/'FULL_SCATTER'))
    orient=run/'S1_ORIENTATION_SUPPLEMENT';jpg=binary(orient/'ORIENTATION_NATIVE_AXES.jpg',b'\xff\xd8\xffCONSTRUCTED')
    osp=put(orient/'DISPLAY_SEAL.json',dict(complete=True,raw_RGB_copied=False,GT_numeric_JSON_exported=False,artifact_sha256={jpg.name:P.sha(jpg)}));oldart[str(osp)]=P.sha(osp)
    oldvisual=put(run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json',dict(complete=True,galleries=oldgallery));oldart[str(oldvisual)]=P.sha(oldvisual)
    put(run/'DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json',dict(complete=True,artifact_seals=oldart))
    paired={}
    for pop,n in F.POPS.items():
        paired[pop]={}
        for i in range(30):
            ci=dict(cluster_count=6,resamples=10000,seed=20260917,CI95=[-1.,1.])
            record=dict(before='BEFORE',after='AFTER',denominator=n,success_difference_count=0,success_difference_percentage_points=0.,
                counts={k:n if k=='BOTH_FAILURE'else 0 for k in F.GROUPS},ids={k:['id'+str(i)for i in range(n)]if k=='BOTH_FAILURE'else[]for k in F.GROUPS},
                deltas={axis:dict(mean=0.,common_available_n=n,recording_cluster_CI=deepcopy(ci),named_session_cluster_sensitivity_CI=deepcopy(ci))for axis in ('success_rate','translation_cm','rotation_deg')},
                recording_clusters={str(i):dict(n=32)for i in range(6)},leave_one_cluster_out={str(i):dict(n=160)for i in range(6)})
            paired[pop]['contrast'+str(i)]=record
    def export(folder,seal,values,**metadata):
        artifacts={}
        for name,value in values.items():
            path=folder/name
            if name.endswith('.json'):put(path,value)
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            artifacts[name]=P.sha(path)
        return put(folder/seal,dict(complete=True,artifacts=artifacts,**metadata))
    derived_files.append(export(derived/'PUBLIC_METRICS40','EXPORT_SEAL.json',{'ALL_POPULATION_METRICS.csv':'population,mean\nPRIMARY192,0\n'},
        schema='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1',arms=['arm'+str(i)for i in range(40)],summary_rows=160,scalar_row_rows=9520,metrics_recomputed=False))
    metrics=derived/'PUBLIC_METRICS40/EXPORT_SEAL.json';ms=P.read(metrics)
    budget_schema=dict(schema='POSTHOC_SAVED_SCALAR_ERROR_BUDGET_V1',populations=F.POPS,no_outlier_exclusion=True,no_metric_recalculation=True,
        fixed_bands=dict(R=['<=5','(5,45]','>45']),units=dict(R=dict(errors_sums_and_mean_contributions='deg',squared_deviations_and_variance_contributions='deg^2')))
    budget=export(run/'ERROR_BUDGET40_V1','ERROR_BUDGET_SEAL.json',{
        'JOINT_CATEGORIES.csv':'population,arm,n\n'+''.join('PRIMARY192,arm0,0\n'for _ in range(800)),
        'ERROR_BANDS.csv':'population,arm,band_n\n'+''.join('PRIMARY192,arm0,0\n'for _ in range(1120)),
        'CONSERVATION_CHECKS.csv':'population,arm,mean_absolute_residual\n'+''.join('PRIMARY192,arm0,0\n'for _ in range(320)),
        'ERROR_BUDGET_SCHEMA.json':budget_schema,'INPUT_BINDINGS.json':dict(export_seal_sha256=P.sha(metrics),artifacts=ms['artifacts'])},
        schema='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1',arms=ms['arms'],populations=F.POPS,campaign_scope='ALL40_COMPLETED_ARMS',
        joint_rows=800,band_rows=1120,conservation_rows=320,source_sha256=P.sha(code/'error_budget.py'),scalar_exporter_source_sha256=P.sha(code/'export_results.py'),input_export_seal_sha256=P.sha(metrics),
        posthoc_descriptive=True,preregistered_test=False,selector_performance_claim=False,no_outlier_exclusion=True,raw_reference_reads=0,pose_or_RGB_reads=0,new_pose_metric_calls=0,new_solves=0,new_fits=0,independent_real_test_n=0)
    derived_files.append(export(derived/'PUBLIC_CONTRASTS30','PUBLIC_CONTRASTS_SEAL.json',{'ALL_POPULATION_PAIRED_CONTRASTS.json':paired,'CONTRAST_SCHEMA.json':dict(constructed=True)},public_poses_or_GT=False,new_metric_or_statistical_calculations=0))
    export(run/'D_PUBLIC_SCALARS_V3','EXPORT_SEAL.json',{'SCOPE_AND_LIMITATIONS.json':dict(constructed=True),'POPULATION_METRICS.csv':'population,mean\nPRIMARY192,0\n'},
        schema='PUBLIC_D_SAVED_SCALAR_EXPORT_SEAL_V1',raw_reference_decodes=0,mixed_pose_rows_decoded=0)
    postbanks={}
    for alias,folder in F.POSTBANK_PATHS.items():
        summary=put(run/folder/'SUMMARY.json',dict(constructed=True))
        packet=binary(run/folder/'BANK_COVERAGE_ROWS.jsonl.gz',b'OPAQUE_CONSTRUCTED_BANK_NEVER_DECODED')
        postbanks[alias]=put(run/folder/'ANALYSIS_SEAL.json',dict(complete=True,actual_new_pose_calls=0,artifacts={summary.name:P.sha(summary),packet.name:P.sha(packet)}))
        if alias in ('S2','F'):derived_files.append(postbanks[alias])
    seq=run/'FINAL_SEQUENCE_RECOVERY_V1';bindings={}
    for folder,source,stages,isderived in [(seq,'final_sequence.py',F.SEQUENCE_STAGES,False),(derived,'derived_completion.py',F.DERIVED_STAGES,True)]:
        plan=put(folder/'PLAN.json',dict(sources={str(code/source):P.sha(code/source)},fixed_inputs={},d1=str(d1),stage_order=list(stages),controller_sha256=P.sha(code/source)))
        go=put(run/('DERIVED_GO.json'if isderived else'SEQUENCE_GO.json'),dict(approved=True,plan_sha256=P.sha(plan),controller_sha256=P.sha(code/source)))
        journal=folder/'ACTUAL_COMMAND_LEDGER.jsonl';journal.write_text(json.dumps(dict(event='CONTROLLER_START',actual_argv=['CONSTRUCTED']))+'\n')
        checkpoints={}
        for index,name in enumerate(stages):
            cp=put(folder/'checkpoints'/(name+'.json'),dict(complete=True,identity=dict(input_bindings={'PLAN.json':P.sha(plan),'ROOT_GO':P.sha(go)}),completed_files={str(p):P.sha(p)for p in derived_files}if index==0 and isderived else{}))
            checkpoints[cp.name]=P.sha(cp)
        extra=dict(schema='FULL_CAMPAIGN_DERIVED_COMPLETION_SEAL_V1',upstream_sequence_seal_sha256=P.sha(seq/'FINAL_SEQUENCE_SEAL.json'),contrasts=30,scalar_arms=40,
            scalar_population_rows=160,scalar_frame_arm_rows=9520,gallery_groups=5,gallery_seed_count=3,scatter_datasets=69,raw_RGB_copied=False,public_poses_or_GT=False,all_outliers_and_failures_retained=True,public_contrasts_files={})if isderived else{}
        terminal=put(folder/('DERIVED_COMPLETION_SEAL.json'if isderived else'FINAL_SEQUENCE_SEAL.json'),dict(complete=True,plan_sha256=P.sha(plan),ROOT_GO_sha256=P.sha(go),source_sha256=P.sha(code/source),stage_order=list(stages),checkpoints=checkpoints,actual_command_ledger_sha256=P.sha(journal),**extra))
        key='derived_completion'if isderived else'final_sequence';bindings[key]=dict(path=str(terminal.relative_to(run)),sha256=P.sha(terminal));bindings[key+'_GO']=dict(path=str(go.relative_to(run)),sha256=P.sha(go))
    cost=put(run/'master/PUBLIC_COST_SUMMARY.json',dict(schema='PUBLIC_RECOVERY_AWARE_MASTER_COST_SUMMARY_V1',all_declared_H1_fits_complete=True,all_declared_H1_prediction_seals_complete=True,
        H1=dict(completed_new_fits=33,reservations=33,tiny_head_real_prediction_forwards=7854),complete_cost_measurement=False,unknown_cost_token='UNKNOWN_NOT_ZERO'))
    fifth=put(run/'ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json',dict(complete=True,CPU_seconds=1.,elapsed_wall_seconds=2.,Result='success',ExecMainStatus='0',NRestarts=0,
        source_sha256=P.sha(code/'cost_waiter.py'),launch_receipt_sha256='CONSTRUCTED',relation='Inclusive; do not add nested child timers'))
    visual=put(run/'ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW.json',dict(schema='ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW_V1',complete=True,approved_for_public_display=True,review_scope='LIMITED_SPOT_REVIEW',human_owner_certifications=0,raw_RGB_copied=False,gallery_seals=visualgalleries))
    policy=put(run/'ROOT_FINAL_PUBLIC_SIZE_POLICY.json',dict(schema='ROOT_REVIEWED_PUBLIC_SIZE_POLICY_V1',approved=True,author='ROOT',maximum_file_bytes=4*1024*1024,maximum_total_bytes=32*1024*1024,no_case_or_outlier_omission=True,reason='Constructed complete fixture'))
    report=run/'FINAL_REPORT_V1';report_artifacts={}
    for name,rows in [('FULL40_PRIMARY_METRICS.csv',40),('REPRESENTATIVE_PRIMARY_METRICS.csv',16),('PAIRED30_PRIMARY_COMPACT.csv',30),('FIXED_COHORTS.csv',165)]:
        path=report/name;path.parent.mkdir(exist_ok=True);path.write_text('arm,scalar\n'+''.join('CONSTRUCTED,0\n'for _ in range(rows)));report_artifacts[name]=P.sha(path)
    path=put(report/'FIXED_COHORTS.json',dict(constructed=True));report_artifacts[path.name]=P.sha(path)
    report_seal=put(report/'REPORT_SEAL.json',dict(schema='FINAL_KO_REPORT_SAVED_SCALAR_SEAL_V1',complete=True,scientific_and_derived_complete=True,report_ready=True,whole_campaign_complete=False,publication_pending=True,
        source_sha256=P.sha(code/'final_report.py'),metrics_export_seal_sha256=P.sha(metrics),paired_export_seal_sha256=P.sha(derived/'PUBLIC_CONTRASTS30/PUBLIC_CONTRASTS_SEAL.json'),error_budget_seal_sha256=P.sha(budget),cost_summary_sha256=P.sha(cost),
        postbank_seal_sha256={key:P.sha(path)for key,path in postbanks.items()},input_bindings={str(path):P.sha(path)for path in postbanks.values()},
        artifacts=report_artifacts,documents={name:P.sha(docs/name)for name in F.REPORT_DOCUMENTS},metadata_documents={name:P.sha(docs/name)for name in F.REPORT_METADATA_DOCUMENTS},
        raw_reference_reads=0,pose_payload_reads=0,model_tensor_reads=0,new_pose_metric_calls=0,new_solves=0,new_fits=0,new_forwards=0,new_renders=0))
    execution=[]
    for name in ('S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py','F_CORE_FOUR_CPU_CONTROLLER.py','ROOT_LAUNCH_RECOVERY_SERVICES.py','ROOT_CAPTURE_SERVICE_LIFECYCLE.py'):
        path=run/name;path.write_text('"""Constructed execution-source placeholder."""\n');execution.append(dict(path=name,sha256=P.sha(path)))
    ref=lambda p:dict(path=str(p.relative_to(run)),sha256=P.sha(p))
    spec=dict(schema=F.SPEC,code_filenames=[p.name for p in code.iterdir()],documents=[dict(name=p.name,sha256=P.sha(p))for p in docs.iterdir()],
        stage_statuses=stage_statuses,size_policy=ref(policy),cost_summary=ref(cost),derived_service_cost=ref(fifth),
        final_visual_review=ref(visual),execution_sources=execution,public_receipt_index=[dict(alias=role,role=role,scope='Constructed metadata only',**ref(cost))for role in ('provenance','cache','calls','argv','environment','model','calibration')],**bindings)
    spec['error_budget40']=ref(budget)
    spec['report_appendices']=ref(report_seal)
    return repo,run,put(root/'SPEC.json',spec),paired


def tests():
    checks=[]
    def reject(label,call):
        try:call()
        except (ValueError,PermissionError):checks.append(label);return
        raise AssertionError(label+' accepted')
    with TemporaryDirectory(prefix='final-public-constructed-')as tmp:
        root=Path(tmp);repo,run,spec,paired=fixture(root)
        manifest=root/'FINAL.json';result=F.build(repo,run,spec,manifest);assert result['PASS']and result['actual_new_H1_fits']==33
        value=P.read(manifest);assert len(P.read(next(e['source']for e in value['entries']if e['destination'].endswith('PUBLIC_PROVENANCE_INDEX.json')))['fit_receipts'])==33
        checks.append('complete33fit_11prediction_11gallery_120comparison_fixture')
        assert value['whole_campaign_complete']is False and value['scientific_and_derived_complete']and value['publication_ready']and value['publication_status']==P.PUBLICATION_STATUS
        checks.append('scientific_readiness_separate_from_pending_commit_push')
        assert len([e for e in value['entries']if '/tables/ERROR_BUDGET40_V1/'in e['destination']])==6;checks.append('all6_final40_error_budget_files')
        assert len([e for e in value['entries']if '/tables/final_report/'in e['destination']])==5;checks.append('exact5_source_and_closedscalar_bound_report_appendices')
        assert {e['destination'].split('/')[-1]for e in value['entries']if e['kind']=='documentation'}>=F.REPORT_METADATA_DOCUMENTS;checks.append('four_sealed_stage_method_documents_selected')
        saved=F.paired_csvs(paired);assert len(list(csv.DictReader(io.StringIO(saved['PAIRED_CONTRAST_SUMMARY.csv']))))==120
        assert len(list(csv.DictReader(io.StringIO(saved['ALL_PAIRED_CATEGORY_IDS.csv']))))==12900;checks.append('all12900_category_ID_rows_retained')
        sample=next(e for e in value['entries']if e['kind']=='gallery_derived_metadata');before=Path(sample['source']).read_bytes()
        P.guard_json(P.read(sample['source']));assert 'conditional_covariance_px2'in before.decode()and'reliability_weight'in before.decode();checks.append('measurement_q_covariance_scalar_reliability_preserved')
        for key in ('actual_pose','R_physical','reference_pose','model_state_dict','training_normalizer','tensor_values'):
            reject('private_'+key,lambda key=key:P.guard_json({key:dict(value=[1.,2.])}))
        reject('without_ROOT_release',lambda:P.materialize(manifest,root/'NO_GO'))
        reject('wrong_manifest_SHA',lambda:P.materialize(manifest,root/'BAD_GO',True,'WRONG'))
        copied=P.materialize(manifest,root/'constructed_copy',True,P.sha(manifest));assert copied['whole_campaign_complete']is False and copied['scientific_and_derived_complete']and copied['publication_ready']and copied['publication_status']==P.PUBLICATION_STATUS and not copied['commit_or_push_performed']
        size_audit=P.read(value['size_audit']['path']);assert copied['actual_staged_total_bytes']==size_audit['actual_staging_total_bytes']and copied['PUBLIC_CONTENTS_bytes']==size_audit['PUBLIC_CONTENTS_bytes']
        checks.append('PUBLIC_CONTENTS_actual_bytes_in_explicit_size_cap')
        assert (root/'constructed_copy'/sample['destination']).read_bytes()==before;checks.append('fresh_tiny_constructed_copy_exact_case_ancestry')
        reject('reuse_existing_output',lambda:P.materialize(manifest,root/'constructed_copy',True,P.sha(manifest)))
        bad=deepcopy(value);bad['entries'].pop();badpath=put(root/'TAMPERED.json',bad);reject('manifest_file_omission',lambda:F.verify(badpath))
        bad=deepcopy(value);bad['whole_campaign_complete']=True;reject('premature_campaign_complete_claim',lambda:F.verify(put(root/'PREMATURE.json',bad)))
        casepath=Path(sample['source']);casepath.write_bytes(before+b' ');reject('changed_parent_case_archive',lambda:F.verify(manifest));casepath.write_bytes(before)
        paired_bad=deepcopy(paired);paired_bad['PRIMARY192']['contrast0']['ids']['BOTH_FAILURE'].pop();reject('paired_ID_omission',lambda:F.paired(paired_bad))
        paired_bad=deepcopy(paired);paired_bad['CAL46']['contrast0']['deltas']['success_rate']['recording_cluster_CI']['resamples']=999;reject('bootstrap_draw_change',lambda:F.paired(paired_bad))
        altered=P.read(spec);altered['template_only']=True;incomplete=put(root/'INCOMPLETE.json',altered);reject('template_cannot_be_final',lambda:F.build(repo,run,incomplete,root/'INCOMPLETE_FINAL.json'))
        ledger=run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl';old=ledger.read_bytes();ledger.write_bytes(b'\n'.join(old.splitlines()[:-1])+b'\n');reject('32fits_not33',lambda:F.fit_accounts(run,F.Inputs()));ledger.write_bytes(old)
        execution=run/'ROOT_LAUNCH_RECOVERY_SERVICES.py';old=execution.read_bytes();execution.write_bytes(old+b'# changed\n');reject('execution_source_tamper',lambda:F.collect(repo,run,P.read(spec),F.Inputs()));execution.write_bytes(old)
        policy=run/'ROOT_FINAL_PUBLIC_SIZE_POLICY.json';low=P.read(policy);low['maximum_total_bytes']=100;lowpath=put(run/'LOW_SIZE_POLICY.json',low)
        low_spec=P.read(spec);low_spec['size_policy']=dict(path=lowpath.name,sha256=P.sha(lowpath));lp=put(root/'LOW_SPEC.json',low_spec)
        reject('size_limit_never_drops_cases',lambda:F.build(repo,run,lp,root/'LOW_FINAL.json'))
        assert P.read(root/'LOW_FINAL_SIZE_AUDIT.json')['no_files_truncated']and not(root/'LOW_FINAL.json').exists();checks.append('explicit_size_audit_preserved_before_refusal')
        reject('scalar_export_matrix',lambda:F.scalar_tree(dict(bank=[[1.,2.],[3.,4.]])))
        reject('nonfinite_scalar',lambda:F.scalar_tree(dict(value=float('nan'))))
        budget=run/'ERROR_BUDGET40_V1/ERROR_BUDGET_SEAL.json';original=budget.read_bytes();metric=run/'DERIVED_COMPLETION_V1/PUBLIC_METRICS40/EXPORT_SEAL.json'
        for label,key,wrong in [('initial25_not_final40','campaign_scope','INITIAL25_COMPLETED_ARMS_ONLY'),('wrong_budget_export_parent','input_export_seal_sha256','WRONG'),('missing_budget_rows','band_rows',1119),('budget_pose_reads','pose_or_RGB_reads',1),('budget_preregistered_claim','preregistered_test',True)]:
            changed=json.loads(original);changed[key]=wrong;budget.write_text(json.dumps(changed))
            reject(label,lambda:F.error_budget40(repo,run,dict(path=str(budget.relative_to(run)),sha256=P.sha(budget)),F.Inputs(),P.read(metric),P.sha(metric),lambda *args:None))
        budget.write_bytes(original)
        reject('numeric_rotation_array_not_unit_schema',lambda:P.guard_json(dict(R=[1.,0.,0.,0.])))
        report=run/'FINAL_REPORT_V1/REPORT_SEAL.json';original=report.read_bytes()
        for label,key,wrong in [('report_wrong_closed_export','metrics_export_seal_sha256','WRONG'),('report_model_tensor_read','model_tensor_reads',1),('report_premature_publication','whole_campaign_complete',True),('report_missing_postbank','postbank_seal_sha256',{}),('report_missing_metadata_document','metadata_documents',{})]:
            changed=json.loads(original);changed[key]=wrong;report.write_text(json.dumps(changed))
            changed_spec=P.read(spec);changed_spec['report_appendices']['sha256']=P.sha(report)
            reject(label,lambda:F.collect(repo,run,changed_spec,F.Inputs()))
        report.write_bytes(original)
        statuspath=repo/P.DOC_PREFIX/'STAGE_STATUS.csv';status_original=statuspath.read_bytes();statuspath.write_text(statuspath.read_text().replace('P0,COMPLETE_NEW','P0,COMPLETE'))
        changed=json.loads(original);changed['metadata_documents']['STAGE_STATUS.csv']=P.sha(statuspath);report.write_text(json.dumps(changed))
        changed_spec=P.read(spec);changed_spec['report_appendices']['sha256']=P.sha(report)
        for document in changed_spec['documents']:
            if document['name']=='STAGE_STATUS.csv':document['sha256']=P.sha(statuspath)
        reject('unregistered_stage_status_even_when_resealed',lambda:F.collect(repo,run,changed_spec,F.Inputs()))
        statuspath.write_bytes(status_original);report.write_bytes(original)
    return dict(PASS=True,constructed_tests=len(checks),checks=checks,constructed_only=True,actual_dataset_rows_read=0,
        actual_gallery_files_copied=0,actual_model_or_reference_decodes=0,scientific_imports_or_calls=0,
        tiny_fixture_copy_only=True,public_final_sha256=P.sha(F.__file__),public_package_sha256=P.sha(P.__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=tests();P.write_new(args.output,result);print(json.dumps(result))
