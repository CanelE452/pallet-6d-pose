"""Fake saved scalar fixtures only; no campaign data/model/GT reads."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import final_report as R


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);R.write_json(path,value);return R.sha(path)


def entry(path):return dict(path=str(path),sha256=R.sha(path))


def summary(pop,arm):
    n=R.POPS[pop];r={k:'0'for k in R.FIELDS};r.update(population=pop,arm=arm,denominator=str(n),
        successes=str(n//2),success_rate=str((n//2)/n),no_pose='0',fallback_known_n=str(n),fallback_false_n=str(n))
    for a,u in [('T','cm'),('R','deg')]:
        for k in R.STATS:
            r[a+'_'+k+(''if k in('n','missing')else'_'+u+('2'if k=='sample_variance'else''))]=str(n)if k=='n'else'0'if k=='missing'else'1.25'
    return r


def pair_data(table):
    result={}
    for pop,n in R.POPS.items():
        result[pop]={}
        for before,after,name in R.PAIR_DEFS:
            for s in R.SEEDS:
                a,b=R.prefix_arm(before,s),R.prefix_arm(after,s);good=n//2
                ci=dict(CI95=[0.,0.],cluster_count=6,seed=20260917,resamples=10000)
                result[pop][name+'_s'+str(s)]=dict(before=a,after=b,denominator=n,
                    success_difference_count=0,success_difference_percentage_points=0.,
                    counts=dict(BOTH_SUCCESS=good,RECOVERY=0,DAMAGE=0,BOTH_FAILURE=n-good),
                    ids=dict(BOTH_SUCCESS=[f'id{i}'for i in range(good)],RECOVERY=[],DAMAGE=[],BOTH_FAILURE=[f'id{i}'for i in range(good,n)]),
                    deltas={axis:dict(mean=0.,common_available_n=n,improved=0,worsened=0,
                        recording_cluster_CI=ci,named_session_cluster_sensitivity_CI=ci)for axis in('success_rate','translation_cm','rotation_deg')},
                    recording_clusters={f'cluster{i}':dict(success_difference_count=0)for i in range(6)},
                    leave_one_cluster_out={f'cluster{i}':dict(n=n-1)for i in range(6)})
    return result


class ReportTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def change(self,p,**v):
        d=json.loads(Path(p).read_text());d.update(v);Path(p).write_text(json.dumps(d));return d

    def metrics_fixture(self):
        root=self.r/'metrics';root.mkdir();rows=[summary(p,a)for p in R.POPS for a in R.arms40()]
        R.write_csv(root/'ALL_POPULATION_METRICS.csv',R.FIELDS,rows);(root/'ROW_METRICS.csv').write_text('identity,scalar\n')
        put(root/'METRICS_SCHEMA.json',dict(sample_variance_ddof=1,sample_std_ddof=1,populations=R.POPS,summary_columns=list(R.FIELDS),independent_real_test_n=0))
        put(root/'INPUT_BINDINGS.json',dict(sources=[],reference_sha256='reference_hash'))
        seal=dict(complete=True,schema='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1',arms=R.arms40(),populations=R.POPS,
            summary_rows=160,scalar_row_rows=9520,identities=238,all_outliers_and_failures_retained=True,
            metrics_recomputed=False,raw_references_decoded=0,solve_fit_forward_calls=0,
            artifacts={n:R.sha(root/n)for n in('ALL_POPULATION_METRICS.csv','ROW_METRICS.csv','METRICS_SCHEMA.json','INPUT_BINDINGS.json')})
        put(root/'EXPORT_SEAL.json',seal);return dict(path=str(root),sha256=R.sha(root/'EXPORT_SEAL.json'))

    def paired_fixture(self,table):
        root=self.r/'paired';root.mkdir();put(root/'ALL_POPULATION_PAIRED_CONTRASTS.json',pair_data(table));put(root/'CONTRAST_SCHEMA.json',{})
        put(root/'PUBLIC_CONTRASTS_SEAL.json',dict(complete=True,public_poses_or_GT=False,new_metric_or_statistical_calculations=0,
            artifacts={n:R.sha(root/n)for n in('ALL_POPULATION_PAIRED_CONTRASTS.json','CONTRAST_SCHEMA.json')}))
        return dict(path=str(root),sha256=R.sha(root/'PUBLIC_CONTRASTS_SEAL.json'))

    def cohort_fixture(self):
        fixed=dict(original31_selection_failure=[f'id{i}'for i in range(31)],original41_candidate_absent=[f'id{i}'for i in range(31,72)],
            PF_union38_candidate_absent=[f'id{i}'for i in range(31,69)],PF_new_good=[f'id{i}'for i in range(69,72)],
            PF_good_old_bank_good_H1_fail=[f'id{i}'for i in range(12)])
        fp=self.r/'fixed.json';put(fp,dict(sets=fixed));banks=[]
        groups={'S1':['S1_UNION'],'S2':['S2_C0','S2_CP','S2_CB'],'S3_S4':['S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE'],'F':['F_COMMON','F_BI']}
        for alias,arms in groups.items():
            root=self.r/alias;raw={}
            for arm in arms:
                raw[arm]=dict(fixed_cohorts={k:dict(n=len(ids),generated_good=len(ids),eligible_good=len(ids),
                    seeds={str(s):dict(selected_good=0,transitions={g:(ids if g=='BOTH_FAILURE'else[])for g in R.GROUPS})for s in R.SEEDS})for k,ids in fixed.items()})
            put(root/'SUMMARY.json',raw);put(root/'ANALYSIS_SEAL.json',dict(complete=True,fixed_sets_sha256=R.sha(fp),actual_new_pose_calls=0,
                artifacts={'SUMMARY.json':R.sha(root/'SUMMARY.json')}))
            banks.append(dict(name=alias,seal=str(root/'ANALYSIS_SEAL.json'),sha256=R.sha(root/'ANALYSIS_SEAL.json'),summary=str(root/'SUMMARY.json')))
        return dict(fixed_sets=entry(fp),banks=banks)

    def choice_fixture(self):
        names=['S1','S2_C0','S2_CP','S2_CB','S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE'];p=self.r/'choice.json'
        put(p,dict(schema='F_FROZEN_SYSTEM_CHOICE_V1',complete=True,before_F_optional_union_and_H1=True,primary_configuration_ranking=False,
            primary_development_fallback_bit_used=True,original120_direct_parity=False,independent_real_test_n=0,all_negative_fallback=False,
            selected_configuration='S1',ranked_configuration_names=names,
            own_validation={n:dict(validation_frames=235,median_joint_success=10,median_validation_loss=.5,production_wall_seconds=1.,
                by_seed={str(s):dict(joint_success=10,validation_loss=.5)for s in R.SEEDS})for n in names}))
        return entry(p)

    def budget_fixture(self,metrics_sha):
        root=self.r/'budget';root.mkdir()
        for name,n in [('JOINT_CATEGORIES.csv',800),('ERROR_BANDS.csv',1120),('CONSERVATION_CHECKS.csv',320)]:
            R.write_csv(root/name,('saved',),[dict(saved=i)for i in range(n)])
        put(root/'ERROR_BUDGET_SCHEMA.json',{})
        put(root/'INPUT_BINDINGS.json',dict(export_seal_sha256=metrics_sha))
        put(root/'ERROR_BUDGET_SEAL.json',dict(complete=True,schema='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1',campaign_scope='ALL40_COMPLETED_ARMS',
            arms=R.arms40(),populations=R.POPS,input_export_seal_sha256=metrics_sha,joint_rows=800,band_rows=1120,conservation_rows=320,
            posthoc_descriptive=True,preregistered_test=False,selector_performance_claim=False,no_outlier_exclusion=True,independent_real_test_n=0,
            raw_reference_reads=0,pose_or_RGB_reads=0,new_pose_metric_calls=0,new_solves=0,new_fits=0,
            artifacts={p.name:R.sha(p)for p in root.iterdir()if p.is_file()}))
        return dict(path=str(root),sha256=R.sha(root/'ERROR_BUDGET_SEAL.json'))

    def test_all40_all4_saved_metric_values_exact(self):
        spec=self.metrics_fixture();table,seal,bindings=R.metrics(R.Inputs(),spec)
        self.assertEqual(len(table),160);self.assertEqual(table[('PRIMARY192','N0')]['T_sample_variance_cm2'],'1.25')

    def test_partial25_export_rejected(self):
        spec=self.metrics_fixture();p=Path(spec['path'])/'EXPORT_SEAL.json';self.change(p,arms=R.arms40()[:25]);spec['sha256']=R.sha(p)
        with self.assertRaises(ValueError):R.metrics(R.Inputs(),spec)

    def test_saved_file_tamper_rejected(self):
        spec=self.metrics_fixture();(Path(spec['path'])/'ALL_POPULATION_METRICS.csv').write_text('tamper')
        with self.assertRaises(ValueError):R.metrics(R.Inputs(),spec)

    def test_all30_byfour_paired_saved_values(self):
        table,_,_=R.metrics(R.Inputs(),self.metrics_fixture());spec=self.paired_fixture(table)
        data,_=R.paired(R.Inputs(),spec,table);self.assertEqual(sum(map(len,data.values())),120)

    def test_paired_redefined_ID_rejected(self):
        table,_,_=R.metrics(R.Inputs(),self.metrics_fixture());spec=self.paired_fixture(table);root=Path(spec['path']);p=root/'ALL_POPULATION_PAIRED_CONTRASTS.json'
        d=json.loads(p.read_text());d['PRIMARY192']['final_matched_BI_effect_s20261018']['ids']['BOTH_FAILURE'][0]='id0';p.write_text(json.dumps(d))
        s=json.loads((root/'PUBLIC_CONTRASTS_SEAL.json').read_text());s['artifacts'][p.name]=R.sha(p);(root/'PUBLIC_CONTRASTS_SEAL.json').write_text(json.dumps(s));spec['sha256']=R.sha(root/'PUBLIC_CONTRASTS_SEAL.json')
        with self.assertRaises(ValueError):R.paired(R.Inputs(),spec,table)

    def test_fixed165_rows_and_all_IDs_preserved(self):
        spec=self.cohort_fixture();data,seals=R.fixed_cohorts(R.Inputs(),spec)
        self.assertEqual(set(seals),{'S1','S2','S3_S4','F'});self.assertEqual(len(data['arms']),11)
        self.assertEqual(len(data['sets']['original41_candidate_absent']),41)

    def test_missing_saved_cohort_is_NA_not_fake_zero(self):
        spec=self.cohort_fixture();b=spec['banks'][0];p=Path(b['summary']);d=json.loads(p.read_text());d['S1_UNION']['fixed_cohorts'].pop('PF_new_good');p.write_text(json.dumps(d))
        seal=Path(b['seal']);s=json.loads(seal.read_text());s['artifacts'][p.name]=R.sha(p);seal.write_text(json.dumps(s));b['sha256']=R.sha(seal)
        data,_=R.fixed_cohorts(R.Inputs(),spec)
        self.assertFalse(data['arms']['S1_UNION']['PF_new_good']['available'])
        self.assertNotIn('selected_good',data['arms']['S1_UNION']['PF_new_good'])

    def test_transition_ids_only_has_no_eager_default_error(self):
        spec=self.cohort_fixture();b=spec['banks'][0];p=Path(b['summary']);d=json.loads(p.read_text())
        for cohort in d['S1_UNION']['fixed_cohorts'].values():
            for seed in cohort['seeds'].values():seed['transition_ids']=seed.pop('transitions')
        p.write_text(json.dumps(d));seal=Path(b['seal']);s=json.loads(seal.read_text());s['artifacts'][p.name]=R.sha(p)
        seal.write_text(json.dumps(s));b['sha256']=R.sha(seal)
        result,_=R.fixed_cohorts(R.Inputs(),spec);self.assertTrue(result['arms']['S1_UNION']['PF_new_good']['available'])

    def test_frozen_choice_copied_not_PRIMARY_selected(self):
        spec=self.choice_fixture();d=R.frozen_choice(R.Inputs(),spec);self.assertEqual(d['selected_configuration'],'S1')
        p=Path(spec['path']);self.change(p,primary_configuration_ranking=True);spec['sha256']=R.sha(p)
        with self.assertRaises(ValueError):R.frozen_choice(R.Inputs(),spec)

    def test_bad_or_private_links_rejected(self):
        for target in('/private/image.jpg','../outside.md','https://random.example/x'):
            with self.assertRaises(ValueError):R.validate_links('[x]('+target+')',{'safe.md'})
        R.validate_links('[x](safe.md)',{'safe.md'})

    def test_allowed_seven_enum_machine_statuses(self):
        values={k:'COMPLETE_NEW'for k in('P0','S1','D_QCOV','S2','S3','S4','F','R')};values.update(P0='COMPLETE_EXACT_REUSE',D_QCOV='DIAGNOSTIC_ONLY')
        self.assertEqual(R.statuses(dict(stage_statuses=values)),values)
        values['R']='SUCCESS'
        with self.assertRaises(ValueError):R.statuses(dict(stage_statuses=values))

    def test_error_budget40_sixfile_gate_and_parent(self):
        spec=self.budget_fixture('closed40');s=R.budget(R.Inputs(),spec,'closed40');self.assertEqual(s['conservation_rows'],320)
        with self.assertRaises(ValueError):R.budget(R.Inputs(),spec,'other_export')

    def test_unfinished_cost_reservations_do_not_become33(self):
        p=self.r/'cost.json';put(p,dict(schema='PUBLIC_RECOVERY_AWARE_MASTER_COST_SUMMARY_V1',complete_cost_measurement=False,
            unknown_cost_token='UNKNOWN_NOT_ZERO',all_declared_H1_fits_complete=False))
        with self.assertRaises(ValueError):R.cost_summary(R.Inputs(),entry(p))

    def test_method_ledger_actual_callable_and_completion_binding(self):
        repo=self.r/'repo';source=repo/'module.py';source.parent.mkdir();source.write_text('def solve_e():\n    pass\n')
        draft=self.r/'draft.json';put(draft,[dict(id='METHOD-01',stage='BASELINE',source='module.py',source_sha256='old',callable='solve_e',line=0,
            evidence_sha256='history',actual_method='unchanged',interpretation_or_limit='exact baseline',execution_status='INCOMPLETE')])
        result=R.method_ledger(R.Inputs(),entry(draft),repo,dict(S1='closedS1'))
        self.assertEqual(result[0]['line'],1);self.assertEqual(result[0]['completion_binding_sha256'],'closedS1')
        self.assertEqual(result[0]['historical_source_sha256'],'old');self.assertEqual(result[0]['execution_status'],'COMPLETE_EXACT_REUSE')
        self.assertEqual(result[0]['status'],'COMPLETE_EXACT_REUSE')

    def test_no_unbound_ROOT_GO_output(self):
        p=self.r/'spec.json';g=self.r/'GO.json';put(p,dict(schema=R.SCHEMA));put(g,dict(approved=False,metadata_only=True))
        with self.assertRaises(ValueError):R.materialize(p,g,self.r/'out')
        self.assertFalse((self.r/'out').exists())

    def test_no_pose_or_GT_json_decode(self):
        p=self.r/'FIXED_REFERENCES.json';put(p,dict(secret='not decoded'))
        with self.assertRaises(ValueError):R.Inputs().read(p)

    def test_render_answers8_questions_and_no_winner_claim(self):
        table,_,_=R.metrics(R.Inputs(),self.metrics_fixture());paired=pair_data(table)
        cohorts,_=R.fixed_cohorts(R.Inputs(),self.cohort_fixture());choice=R.frozen_choice(R.Inputs(),self.choice_fixture())
        cost=dict(fullnormal_runtime_minimize_calls=dict(known_lower_bound=999,old_D4_prefix_included_not_added_again=112387,original_interrupted_arm_directories=8),
            D_diagnostic_runtime_minimize_calls=dict(old_Sigma_plus_Gamma=786,fixed_Gamma=64),
            H1=dict(completed_new_fits=33,reservations=33,tiny_head_real_prediction_forwards=7854,new_backbone_prediction_forwards=0),
            _derived_service=dict(CPU_seconds=2.5,elapsed_wall_seconds=3.),_master_child=dict(child_CPU_seconds=1.,wall_seconds=2.))
        links={k:'selected.json'for k in('methodology','structure','full40','metrics','budget','paired','cohorts','D','gallery','scatter','cost','cost_fifth','provenance','reproduce')}
        statuses={k:'COMPLETE_NEW'for k in('P0','S1','D_QCOV','S2','S3','S4','F','R')}
        text=R.render(table,paired,choice,cost,cohorts,[],statuses,links)
        for i in range(1,9):self.assertIn(f'## {i}.',text)
        self.assertIn('whole_campaign_complete=false',text);self.assertIn('UNKNOWN_NOT_ZERO',text)
        self.assertIn('N3/SubPix',text);self.assertIn('native S',text)
        self.assertIn('PRIMARY로 후보를 재순위화하지 않았다',text)
        self.assertNotIn(str(self.r),text)

    def test_materialize_fixture_exact5_appendices_no_wholegoal_claim(self):
        # One actual materialization of fake scalar metadata, with authority
        # gates mocked independently of their focused corruption tests.
        table,ms,mb=R.metrics(R.Inputs(),self.metrics_fixture());paired=pair_data(table)
        cohorts,bank_seals=R.fixed_cohorts(R.Inputs(),self.cohort_fixture());choice=R.frozen_choice(R.Inputs(),self.choice_fixture())
        repo=self.r/'repo';ns=repo/'scripts/research/pallet_bi_full_execution_v1';ns.mkdir(parents=True)
        for name in('cost_waiter.py','derived_completion.py','final_sequence.py'):(ns/name).write_text('fixture source')
        cost=dict(fullnormal_runtime_minimize_calls=dict(known_lower_bound=999,old_D4_prefix_included_not_added_again=112387,original_interrupted_arm_directories=8),
            D_diagnostic_runtime_minimize_calls=dict(old_Sigma_plus_Gamma=786,fixed_Gamma=64),
            H1=dict(completed_new_fits=33,reservations=33,tiny_head_real_prediction_forwards=7854,new_backbone_prediction_forwards=0),
            S1_candidate_cost_evaluations=dict(scientific_actual=1352,new_unique_PF=1348,old_cost_fallback_reconstruction=4,postseal_diagnostic=130))
        cp=self.r/'public_cost.json';put(cp,cost)
        fifth=self.r/'fifth.json';put(fifth,dict(complete=True,Result='success',ExecMainStatus='0',NRestarts=0,CPU_seconds=2.,elapsed_wall_seconds=3.))
        waiter=self.r/'waiter.json';put(waiter,dict(complete=True,campaign_complete=False,source_sha256=R.sha(ns/'cost_waiter.py'),
            output_bindings={str(cp.resolve()):R.sha(cp),str(fifth.resolve()):R.sha(fifth)},actual_master_child=dict(child_CPU_seconds=1.,wall_seconds=2.)))
        stage={k:'COMPLETE_NEW'for k in('P0','S1','D_QCOV','S2','S3','S4','F','R')};stage.update(P0='COMPLETE_EXACT_REUSE',D_QCOV='DIAGNOSTIC_ONLY')
        metricroot=self.r/'metric_scope';pairroot=self.r/'pair_scope'
        links={k:'selected.json'for k in('methodology','structure','full40','metrics','budget','paired','cohorts','D','gallery','scatter','cost','cost_fifth','provenance','reproduce')}
        spec=dict(schema=R.SCHEMA,repo=str(repo),source_bindings={},stage_statuses=stage,metrics=dict(path=str(metricroot),sha256='metricseal'),
            paired=dict(path=str(pairroot),sha256='pairedseal'),budget=dict(path='unused',sha256='budgetseal'),f_choice=dict(),cost=entry(cp),
            cost_waiter=entry(waiter),fifth_service=entry(fifth),cohorts={},derived=dict(sha256='derived'),sequence=dict(sha256='sequence'),
            D=dict(sha256='Dseal'),method_ledger_draft={},saved_diagnostic={},links=links,authorities=[],selected_relative_files=['selected.json',*R.DOCUMENTS])
        sp=self.r/'spec.json';put(sp,spec);go=self.r/'GO.json';put(go,dict(approved=True,metadata_only=True,source_sha256=R.sha(R.__file__),spec_sha256=R.sha(sp)))
        derived=dict(upstream_sequence_seal_sha256='sequence',scalar_arms=40,contrasts=30,
            _authenticated_checkpoint_artifact_hashes={str((metricroot/'EXPORT_SEAL.json').resolve()):'metricseal'},
            public_contrasts_files={str((pairroot/'PUBLIC_CONTRASTS_SEAL.json').resolve()):'pairedseal'})
        droot=self.r/'D_public';droot.mkdir();R.write_csv(droot/'POPULATION_METRICS.csv',('population','variant','cell','successes'),[])
        R.write_csv(droot/'CONTRAST_COUNTS.csv',('population','variant','contrast','family_switches','within_before_family_success_delta','family_readout_success_delta','joint_success_delta'),[])
        dseal=dict(schema='PUBLIC_D_SAVED_SCALAR_EXPORT_SEAL_V1',raw_reference_decodes=0,mixed_pose_rows_decoded=0)
        ledger=[dict(id='method',source='saved.py',callable='saved',line=1,execution_status='COMPLETE_NEW')]
        with patch.object(R,'metrics',return_value=(table,ms,mb)),patch.object(R,'authorities'),\
            patch.object(R,'paired',return_value=(paired,{})),patch.object(R,'budget',return_value={}),\
            patch.object(R,'frozen_choice',return_value=choice),patch.object(R,'cost_summary',return_value=cost),\
            patch.object(R,'fixed_cohorts',return_value=(cohorts,bank_seals)),patch.object(R,'terminal',side_effect=[derived,{}]),\
            patch.object(R,'saved_diagnostic',return_value=dict(PL=[],IMAGE_feature_associations={})),\
            patch.object(R,'bundle',return_value=(droot,dseal)),patch.object(R,'method_ledger',return_value=ledger):
            result=R.materialize(sp,go,self.r/'out')
        self.assertEqual(set(result['artifacts']),set(R.APPENDICES));self.assertEqual(result['primary_rows'],40)
        self.assertEqual(result['representative_rows'],16);self.assertEqual(result['paired_primary_rows'],30);self.assertEqual(result['fixed_cohort_rows'],165)
        self.assertFalse(result['whole_campaign_complete']);self.assertTrue(result['publication_pending'])
        self.assertEqual(result['new_solves'],0);self.assertEqual(result['new_fits'],0)
        output=self.r/'out';rows=R.csv_read(output/'FULL40_PRIMARY_METRICS.csv',R.FIELDS)
        self.assertEqual(rows[0]['T_sample_variance_cm2'],'1.25')
        self.assertIn('METHOD_DIFFERENCE_LEDGER.json',result['metadata_documents'])


if __name__=='__main__':unittest.main()
