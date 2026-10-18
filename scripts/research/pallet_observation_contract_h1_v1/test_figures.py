"""Constructed reporting integration; no real inputs, optimizers or model forwards."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.pallet_risk_information_v1.common import write, save_rows, sha
from . import figures as F
from . import registry as R


class FigureTests(unittest.TestCase):
    def test_postseal_guard_precedes_all_file_reads(self):
        with self.assertRaises(PermissionError):F.verified_postseal_inputs(Path('/absent'),Path('/absent'),False)

    def test_ecdf_keeps_zero_extreme_missing_and_full_denominator(self):
        records=[dict(id=str(i),metrics={'a':dict(translation_cm=v,rotation_deg=v)})for i,v in enumerate((0.,2000.,None))]
        values=F.ecdf_rows(records,['a']);self.assertEqual(len(values),6)
        axis=[r for r in values if r['axis']=='translation_cm']
        self.assertEqual([r['error']for r in axis],[0.,2000.,None]);self.assertEqual(axis[1]['cumulative_fraction_of_full_denominator'],2/3)
        self.assertIsNone(axis[2]['cumulative_fraction_of_full_denominator']);self.assertTrue(all(r['denominator']==3 for r in axis))

    def test_same_seed_comparator_and_strict_tradeoff_deltas(self):
        self.assertEqual(F.comparator('H1_NO_B_FEATURE_s20261020'),'D1_H1_MLP_s20261020')
        self.assertEqual(F.comparator('N0'),'D1_H1_MLP_s20261018')
        records=[dict(metrics={'b':dict(success=False,translation_cm=20.,rotation_deg=20.),'a':dict(success=False,translation_cm=10.,rotation_deg=21.)})]
        d=F.deltas(records,'b','a',True);self.assertEqual(d['translation_cm']['mean'],-10.);self.assertEqual(d['rotation_deg']['worsened_n'],1)

    def test_full_seven_plot_render_with_constructed_seals_and_real_csv_writers(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);evaluation=root/'evaluation';evaluation.mkdir();output=root/'figures';arms=list(R.arms(False))
            write(root/'PREDICTION_MANIFEST.json',dict(status='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING',real_reference_inputs=False,arms=arms))
            scored=[]
            for i in range(192):
                metrics={a:dict(success=(i+j)%4==0,translation_cm=float(i+j),rotation_deg=float((i+j)%180))for j,a in enumerate(arms)}
                scored.append(dict(id='TOY'+str(i),partition='TEST',metrics=metrics,action_sources={a:'N0'if j%2 else'OTHER'for j,a in enumerate(arms)},full_bank_has_success=i%5!=0))
            primary={a:dict(denominator=192,successes=48,T_gt50cm=100,T_gt100cm=50,R_gt45deg=120,no_pose=0)for a in arms}
            write(evaluation/'EVALUATION.json',dict(arms=arms,populations={'PRIMARY192':primary}))
            save_rows(evaluation/'SCORED_ROWS.jsonl.gz',scored)
            write(evaluation/'EVALUATION_SEAL.json',dict(status='COMPLETE',prediction_manifest_sha256=sha(root/'PREDICTION_MANIFEST.json'),
                artifacts={n:sha(evaluation/n)for n in('EVALUATION.json','SCORED_ROWS.jsonl.gz')}))
            original=[];new=[]
            for group,destination in(('D1_H1_MLP',original),('H1_NO_B_FEATURE',new)):
                for seed in R.SEEDS:
                    folder=root/'training'/(group+'_s'+str(seed));folder.mkdir(parents=True)
                    def metric(frames):return dict(loss=.5,frames=frames,success_5cm5deg=10,translation_cm={'mean':20.},rotation_deg={'mean':10.},frame_balanced_Brier=.2,tails={'T_gt50cm':5,'T_gt100cm':2,'R_gt45deg':3})
                    history=[dict(epoch=i,train=metric(768),validation=metric(235))for i in(1,2)]
                    write(folder/'HISTORY.json',history)
                    destination.append(dict(arm=group,seed=seed,best_epoch=1,checkpoint=str(folder/'unused.pt'),history_sha256=sha(folder/'HISTORY.json')))
            original_manifest=root/'ORIGINAL_MANIFEST.json';write(original_manifest,dict(runs=original));write(root/'NO_B_TRAINING_MANIFEST.json',dict(runs=new))
            def pm():return dict(denominator=100,successes=40,translation_cm={'mean':10.},rotation_deg={'mean':5.},T_gt50cm=2,T_gt100cm=1,R_gt45deg=1,changes=1)
            policy=dict(schema='FROZEN_H1_GUARDED_POLICY_CALIBRATION_V1',PRIMARY_or_CAL_reference_reads=0,seeds={})
            for seed in R.SEEDS:
                policy['seeds'][str(seed)]=dict(status='CONFIRMATION_FAILED_KEEP_H1',active_tau={'kind':'NEGATIVE_INFINITY'},calibrated_trial_tau={'kind':'FINITE','value':2.},development_baseline=pm(),
                    all_thresholds=[dict(tau=t,metrics=pm(),admissible=i>0)for i,t in enumerate(({'kind':'NEGATIVE_INFINITY'},{'kind':'FINITE','value':0.},{'kind':'FINITE','value':2.}))],confirmation=dict(baseline=pm(),selected=pm(),passed=False))
            policy_path=root/'POLICY.json';write(policy_path,policy)
            F.generate(root,evaluation,output,original_manifest,policy_path,postseal_GO=True)
            self.assertEqual(len(list(output.glob('*.png'))),7);self.assertEqual(len(list(output.glob('*.csv'))),7)
            manifest=json.loads((output/'FIGURES.json').read_text());self.assertFalse(manifest['actual_pixel_review_completed'])
            self.assertEqual(len(manifest['images']),7)
            for record in manifest['images']:self.assertEqual(record['sha256'],sha(output/record['file']))


if __name__=='__main__':unittest.main()
