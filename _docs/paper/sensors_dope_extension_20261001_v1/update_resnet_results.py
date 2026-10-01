"""CPU-only, receipt-bound ResNet18 manuscript insertion; no inference or evaluation.

The tabular formatter is shared with DOPE through an explicit naming/schema
adapter; numerical values and uncertainty statuses are copied without alteration.
No actual artifacts are read in --selfcheck or --pending.
"""
from pathlib import Path
import argparse
import copy
import csv
import json
import math
import update_dope_results as D

HERE, ROOT, TABLES = D.HERE, D.ROOT, D.TABLES
EXPERIMENT = ROOT/'_docs/experiments/pallet_resnet18_refiner_20261001_v1'
ARMS = ('RESNET18','D1','D2','D3','P1','P2','P3')
NATIVE_NAME = 'SimpleBaseline-derived ResNet18 full-image pallet9'


def status_file(complete=False, observed=''):
    if complete:
        third = ('The three declared fixed-estimator applications have completed baseline/P/D '
                 'development measurements. Their effects are reported separately, including '
                 'null or adverse outcomes; completion does not imply improvement for every '
                 'estimator or independent generalization.')
        abstract = observed
        conclusion = ('The ResNet18 application completes the declared third fixed-estimator '
                      'development experiment, with its own source-trained baseline and heads. '
                      'The measured outcomes do not establish arbitrary-backbone transfer, '
                      'independent TEST generalization or stable physical T/R improvement.')
        budget = ('The ResNet18 baseline completed the fixed 60 epochs on 55,980 source images: '
                  '209,940 updates and 3,358,800 exposures, retaining the final 12-image batch. '
                  'Adam used learning rate $10^{-3}$ and weight decay $10^{-4}$, with factor-ten '
                  'drops at the start of epochs 40 and 50. Only calibration loss was monitored; '
                  'epoch 60 was fixed, with seed 42 and source-only photometric augmentation. '
                  'The source-trained baseline was frozen before the six new P/D head fits.')
    else:
        third = D.THIRD
        abstract = 'ResNet18 results remain pending in this revision.'
        conclusion = ('The required ResNet18 application remains unresolved; neither a '
                      'completed interface nor results from another estimator establish its outcome.')
        budget = ('The planned ResNet18 baseline is 60 epochs of 55,980 source images: '
                  '209,940 updates and 3,358,800 exposures, retaining the final 12-image batch. '
                  'Adam starts at $10^{-3}$ with weight decay $10^{-4}$ and factor-ten drops '
                  'at the start of epochs 40 and 50. Only calibration loss is monitored and '
                  'final epoch 60 is fixed. Seed 42 and source-only photometric augmentation '
                  'are prescribed. These are preparation details; actual training and outcomes '
                  'have not been inserted.')
    main = observed if complete else 'ResNet18 baseline completion and before/after measurements remain pending.'
    return '% ResNet18 updater owns every third-estimator status macro.\n' + '\n'.join(
        rf'\newcommand{{\{key}}}{{{value}}}' for key,value in (
            ('ThirdEstimatorStatus',third),('ResnetAbstractStatus',abstract),
            ('ResnetMainStatus',main),('ResnetConclusionStatus',conclusion),
            ('ResnetBaselineStatus',budget)))+'\n'


def pending():
    output = {name.replace('dope_','resnet_'):value.replace('DOPE','ResNet18').replace('dope','resnet')
              for name,value in D.pending().items() if name != 'dope_status.tex'}
    output['resnet_status.tex'] = status_file()
    output['resnet_baseline.tex'] = ('ResNet18 baseline completion and its full 60-epoch '
        'training/calibration loss curve remain pending. No simulated curve is shown.\n')
    return output


def adapt_results(results,paired,training,selection,heldout):
    """Name-only adapter to the shared seven-arm formatter, never a metric fallback."""
    assert results['schema']=='resnet18_refiner_dev_results_v1'
    assert results['RESNET18_AP_claim'] is False and set(results['methods'])==set(ARMS)
    assert set(heldout['results'])==set(ARMS)
    r,p,t,s,h = copy.deepcopy((results,paired,training,selection,heldout))
    r['schema']='dope_refiner_dev_results_v1';r['DOPE_AP_claim']=r.pop('RESNET18_AP_claim')
    r['methods']['DOPE']=r['methods'].pop('RESNET18')
    h['results']['DOPE']=h['results'].pop('RESNET18')
    p['results']={k.replace('RESNET18','DOPE'):v for k,v in p['results'].items()}
    return r,p,t,s,h


def validate_baseline(baseline):
    assert baseline['complete'] is True and baseline['final_checkpoint_only'] is True
    assert (baseline['epochs'],baseline['updates'],baseline['source_exposures'])==(60,209940,3358800)
    assert (baseline['train_images'],baseline['calibration_images'],baseline['seed'],baseline['real_training'])==(55980,1004,42,0)
    assert [r['epoch'] for r in baseline['curves']]==list(range(1,61))
    for r in baseline['curves']:
        assert r['step']==3499*r['epoch'] and r['training_frames']==55980
        assert r['calibration']['frames']==1004
        expected_lr=.001*(.1 if r['epoch']>=40 else 1)*(.1 if r['epoch']>=50 else 1)
        assert math.isclose(r['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15)
        assert all(math.isfinite(v) and v>=0 for v in (r['training_loss'],r['calibration']['loss']))
    assert baseline['training_counts_scope']=='All60 epochs; exposures, not unique-image counts'
    counts=baseline['training_counts']
    assert counts['total_channels']==3358800*9
    assert 0<=counts['supervised_channels']<=counts['total_channels']
    assert 0<=counts['all_masked_frames']<=3358800


def render(results,paired,training,selection,heldout,baseline):
    validate_baseline(baseline)
    out=D.render(*adapt_results(results,paired,training,selection,heldout))
    out={name.replace('dope_','resnet_'):value.replace('DOPE','ResNet18').replace('dope','resnet')
         for name,value in out.items() if name!='dope_status.tex'}
    base=results['methods']['RESNET18'];p=results['seed_mean']['P']
    observed=(f"For frozen ResNet18 on reused DEV, the conditional keypoint median is "
        f"{D.number(base['median_px'])} pixels before refinement and {D.number(p['median_px'])} "
        'pixels for the mean of three P seed statistics; full-GT PCK10 is '
        f"{D.number(base['ALL_GT_PCK']['10'],scale=100)} and "
        f"{D.number(p['ALL_GT_PCK']['10'],scale=100)} percent, respectively. "
        'All missing and unfavorable outcomes remain in the declared tables.')
    out['resnet_status.tex']=status_file(True,observed)
    counts=baseline['training_counts']
    out['resnet_baseline.tex']=(
        f"The completed baseline contains {baseline['parameters']} learned parameters and "
        '209,940 updates across 60 epochs, with 3,358,800 source image exposures. '
        f"The recorded exposure-level mask has {counts['supervised_channels']} supervised "
        f"channels among {counts['total_channels']} slots and {counts['all_masked_frames']} "
        'all-masked frame exposures. These counts are not unique-image coverage. '
        'Each curve value is a full frame-weighted epoch mean; the 1,004-image calibration '
        'curve selected neither a checkpoint nor a stopping epoch.\n'
        r'\begin{figure}[t]\centering\includegraphics[width=.94\linewidth]{figures/resnet_baseline_curve.pdf}'+'\n'
        r'\caption{ResNet18 source baseline: all 60 epochs and fixed final checkpoint. '
        'Training and calibration use the declared fixed-nine-channel masked heatmap '
        r'half-MSE. This loss curve is not a pose-accuracy or convergence certificate.}'+'\n'
        r'\label{fig:resnetbaseline}\end{figure}'+'\n')
    return out


def render_runtime(runtime):
    assert runtime['schema']=='resnet18_refiner_runtime_v1'
    assert set(runtime['results'])=={'RESNET18','D1','P1'}
    adapted=copy.deepcopy(runtime);adapted['schema']='dope_refiner_runtime_v1'
    adapted['results']['DOPE']=adapted['results'].pop('RESNET18')
    for row in adapted['measurements']:
        assert row['arm'] in ('RESNET18','D1','P1')
        if row['arm']=='RESNET18':row['arm']='DOPE'
    return D.render_runtime(adapted).replace('DOPE','ResNet18').replace('dope','resnet')


def baseline_curve(baseline):
    """Plot only stored complete-epoch statistics; never run the baseline model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validate_baseline(baseline)
    rows=baseline['curves'];fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    ax.plot([r['epoch'] for r in rows],[r['training_loss'] for r in rows],label='TRAIN: all 55,980 frames/epoch')
    ax.plot([r['epoch'] for r in rows],[r['calibration']['loss'] for r in rows],label='Calibration: 1,004 frames')
    ax.set(xlabel='Completed epoch',ylabel='Masked heatmap half-MSE',title='Fixed 60-epoch source baseline; final checkpoint only')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    pdf=HERE/'figures/resnet_baseline_curve.pdf';fig.savefig(pdf);plt.close(fig)
    path=HERE/'generated_tables/RESNET_BASELINE_CURVE.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','step','lr','training_frames','training_loss','calibration_frames','calibration_loss','calibration_supervised_channels'])
        w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in ('epoch','step','lr','training_frames','training_loss')}|
            dict(calibration_frames=r['calibration']['frames'],calibration_loss=r['calibration']['loss'],calibration_supervised_channels=r['calibration']['supervised_channels']))
    return [D.bind(pdf),D.bind(path)]


def generate(folder,runtime_path=None):
    assets=[];evidence=[]
    if folder is None:
        output=pending();state='RESNET_RESULTS_PENDING';done=False
    else:
        folder=Path(folder).resolve();assert folder==EXPERIMENT.resolve()
        # Completion claims require the preceding DOPE stage to remain fully bound.
        dope=D.read(HERE/'DOPE_RESULTS_BINDINGS.json')
        assert dope['status']=='DOPE_RESULTS_AND_RUNTIME_INSERTED'
        for value in dope['evidence']+dope['generated_tables']:D.verify(value)
        evidence.append(D.bind(HERE/'DOPE_RESULTS_BINDINGS.json'))
        names=['DEV_RESULTS.json','DEV_PAIRED_RESULTS.json','TRAINING_COMPLETE.json','SELECTION.json','SYNTHETIC_HELDOUT.json','BASELINE_TRAINING_COMPLETE.json']
        values=[D.read(folder/n) for n in names];result,paired,training,selection,heldout,baseline=values
        for key in ('code','evaluation_lock','predictions','frame_results','full_precision_errors','paired_results'):D.verify(result[key])
        for b in result['source_bindings']:D.verify(b)
        assert result['paired_results']==D.bind(folder/'DEV_PAIRED_RESULTS.json')
        for left,right in ((result['baseline_training_complete'],D.bind(folder/'BASELINE_TRAINING_COMPLETE.json')),
                           (result['baseline'],baseline['final_checkpoint']),
                           (result['baseline_protocol'],baseline['protocol'])):
            # Production common.bound omits bytes; bytes are checked when present.
            assert D.verify(left).resolve()==D.verify(right).resolve()
            assert left['sha256']==right['sha256']
        for key in ('final_checkpoint','protocol','source'):D.verify(baseline[key])
        assert heldout['selection']['sha256']==D.bind(folder/'SELECTION.json')['sha256']
        D.verify(selection['protocol']);D.verify(selection['validation'])
        output=render(*values);evidence += [D.bind(folder/n) for n in names]
        assets=baseline_curve(baseline);done=True;state='RESNET_RESULTS_INSERTED_RUNTIME_PENDING'
        if runtime_path is not None:
            runtime_path=Path(runtime_path).resolve();assert runtime_path==folder/'RUNTIME.json'
            runtime=D.read(runtime_path);plan=D.read(D.verify(runtime['plan']))
            D.verify(runtime['code']);D.verify(runtime['raw_rows'])
            assert plan['predictions']['sha256']==result['predictions']['sha256']
            assert plan['protocol']['sha256']==selection['protocol']['sha256']
            assert plan['training']['sha256']==D.bind(folder/'TRAINING_COMPLETE.json')['sha256']
            for b in plan['codes']:D.verify(b)
            output['resnet_runtime.tex']=render_runtime(runtime)
            evidence += [D.bind(runtime_path),runtime['plan']]
            state='RESNET_RESULTS_AND_RUNTIME_INSERTED'
    for name,value in output.items():
        assert name.startswith('resnet_') and name.endswith('.tex')
        (TABLES/name).write_text(value)
    receipt=dict(schema='sensors_resnet_manuscript_insert_v1',status=state,draft=True,not_submitted=True,
        third_estimator_name=NATIVE_NAME,third_estimator_pending=not done,
        third_estimator_outcomes_inserted=done,three_fixed_estimator_measurements_complete=done,
        independent_TEST_available=False,stable_physical_TR_improvement_established=False,
        all_estimators_improve_inferred=False,new_training_or_inference=0,source_numbers_recomputed=False,
        runtime_results_inserted=runtime_path is not None,code=D.bind(__file__),formatter_code=D.bind(D.__file__),
        evidence=evidence,generated_tables=[D.bind(TABLES/n) for n in output],generated_assets=assets)
    (HERE/'RESNET_RESULTS_BINDINGS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=state,generated_tables=len(output),generated_assets=len(assets))))


def selfcheck():
    D.selfcheck()
    assert set(pending())=={'resnet_status.tex','resnet_results.tex','resnet_full.tex','resnet_paired.tex','resnet_source.tex','resnet_runtime.tex','resnet_baseline.tex'}
    assert all('pending' in value.lower() for value in pending().values())
    assert 'pending' not in status_file(True,'Measured adverse results remain reported.').lower()
    baseline=dict(complete=True,final_checkpoint_only=True,epochs=60,updates=209940,source_exposures=3358800,
        train_images=55980,calibration_images=1004,seed=42,real_training=0,parameters=123,
        training_counts_scope='All60 epochs; exposures, not unique-image counts',
        training_counts=dict(total_channels=3358800*9,supervised_channels=3358800*8,all_masked_frames=0),
        curves=[dict(epoch=e,step=3499*e,training_frames=55980,training_loss=1/e,
            lr=.001*(.1 if e>=40 else 1)*(.1 if e>=50 else 1),
            calibration=dict(frames=1004,loss=2/e,supervised_channels=8000)) for e in range(1,61)])
    validate_baseline(baseline)
    bad=copy.deepcopy(baseline);bad['updates']=210000
    try:validate_baseline(bad)
    except AssertionError:pass
    else:raise AssertionError('Incorrect update denominator accepted')
    # Full invented seven-arm renderer including adverse P median, missingness and unavailable CI.
    methods={a:dict(gt_denominator=2818,full_frame_denominator=319,median_px=3. if a=='RESNET18' else 4.,
        p90_px=8.,matched_frames=250,failed_or_excluded_frames=69,supervised_points=2200,missing_supervised_keypoints=200,
        ALL_GT_PCK={'10':.6},pose=dict(denominator=319,n=270,failure_count=49,coverage=270/319,translation_median_cm=4.,rotation_median_deg=2.)) for a in ARMS}
    r=dict(complete=True,schema='resnet18_refiner_dev_results_v1',role='REUSED_DEV',RESNET18_AP_claim=False,
        positive_frames=319,sessions=13,gt_denominator=2818,same_conditional_support=True,
        geometry_derived_reference_not_independent_physical_metrology=True,methods=methods,
        seed_mean={a:methods[a+'1'] for a in ('D','P')},predictions={'invented':True})
    ci=dict(status='COMPLETE',delta=1.,low=.5,high=2.)
    paired=dict(complete=True,strict_support_no_silent_intersection=True,predictions=r['predictions'],results={
        k:dict(conditional_keypoint_median={'session':ci},pose={'translation_error_cm':ci,'rotation_error_deg':dict(status='SUPPORT_MISMATCH_NO_PAIRED_CONDITIONAL_CI')})
        for k in ('P_minus_RESNET18','D_minus_RESNET18','P_minus_D')})
    training=dict(complete=True,fits=6,steps_per_fit=6000,total_head_updates=36000,seeds=[dict(seed=s,real_training=0,
        final_checkpoint_only=True,backbone_retrained=False,updates_per_arm=6000,exposures_per_arm=96000,
        usable_training_rows=12345,parameters={'P':222,'D':333}) for s in (1,2,3)])
    selection=dict(complete=True,real_selection=False,rules={a:dict(lam=1.,max_move_image_diagonal_fraction=None) for a in ('D','P')},temperatures={f'P{s}':1. for s in (1,2,3)})
    heldout=dict(complete=True,results={a:dict(frames=1985,gt9=100,observed9=80,missing9=20,median_px=4.,p90_px=9.,pck10_all_gt=.7) for a in ARMS})
    output=render(r,paired,training,selection,heldout,baseline)
    assert 'ResNet18+P mean & 4.000' in output['resnet_results.tex']
    assert 'ResNet18 & 3.000' in output['resnet_results.tex']
    assert 'unavailable' in output['resnet_paired.tex'] and '618' in output['resnet_full.tex']
    assert not any('DOPE' in v for v in output.values())
    runtime=dict(schema='resnet18_refiner_runtime_v1',complete=True,PASS=True,frames=26,sessions=13,repeats=5,batch=1,
        measured_calls=390,warmup_calls=60,all_cache_replays_PASS=True,no_fastest_selection=True,
        partial_or_failed_rows_discarded=False,historical_YOLO_latency_is_separate=True,point_atol_px=1e-10,
        measurements=[dict(arm=a) for a in ('RESNET18','D1','P1') for _ in range(130)],
        results={a:dict(failures_included=True,pose_status_counts={'OK':100,'NO_DETECTION':30},
            **{k:dict(n=130,median=v,p90=v+1) for k,v in [('keypoints_ms',4.),('pose_ms',2.),('full_ms',7.)]}) for a in ('RESNET18','D1','P1')})
    text=render_runtime(runtime);assert 'ResNet18+P1 & 4.000 & 2.000 & 7.000' in text
    print('PURE_RESNET_INSERT_SELFCHECK_PASS; invented data only, no model/data reads or plots')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pending',action='store_true');g.add_argument('--results-dir',type=Path);g.add_argument('--selfcheck',action='store_true')
    p.add_argument('--runtime',type=Path)
    a=p.parse_args()
    if a.runtime and not a.results_dir:p.error('--runtime requires --results-dir')
    selfcheck() if a.selfcheck else generate(a.results_dir,a.runtime)
