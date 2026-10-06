"""Saved-result and preservation verification without NN or pose forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import ROOT,DOC,OUTPUT,BANKS,COST,OLD_DOC,HARD_DOC,Data,read,write,sha,hash_value,code_bindings,verify_derived


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def preservation():
    initial=read(OUTPUT/'INITIAL_USER_STATE.json')
    changed=[]
    for entry in initial['entries']:
        p=ROOT/entry['path']
        if 'symlink' in entry:
            if not p.is_symlink() or str(p.readlink())!=entry['symlink']:changed.append(entry['path'])
        if 'bytes' in entry:
            if not p.is_file():changed.append(entry['path']);continue
            st=p.stat()
            if st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']:changed.append(entry['path'])
            if 'sha256' in entry and sha(p)!=entry['sha256']:changed.append(entry['path'])
    assert not changed, 'Existing user files changed: '+str(changed[:10])
    assert hashlib.sha256(git('diff','--binary')).hexdigest()==initial['tracked_diff_sha256'], 'User tracked diff changed'
    teacher=Path('/home/minjae/Documents/github/pallet-pose-teacher-20261006')
    teacher_diff=subprocess.check_output(['git','-C',str(teacher),'diff','--cached','--binary'])
    assert hashlib.sha256(teacher_diff).hexdigest()==initial['teacher_cached_diff_sha256']
    assert git('branch','--show-current').decode().strip()=='main'
    return dict(status='PASS',existing_user_entries_checked=len(initial['entries']),changed=0,
        user_tracked_diff_unchanged=True,teacher_staged_diff_unchanged=True,
        initial_head=initial['head'],branch='main',scope='existing working files and raw Git diffs, excluding newly owned namespace')


def frozen_state():
    states=read(OUTPUT/'FROZEN_INPUT_STATES.json')
    for entry in states['files']:
        st=Path(entry['path']).stat()
        assert st.st_size==entry['bytes'] and st.st_mtime_ns==entry['mtime_ns'], entry['path']
    return dict(status='PASS',files=len(states['files']),
        verification='current unchanged size/mtime; setup freshly hashed costs/banks/small inputs once and reused prior audited feature content hashes; this stat check is not a new full content audit')


def distribution(values):
    a=np.asarray(values,dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(mean=float(a.mean()),median=float(np.median(a)),P90=float(np.quantile(a,.9)),count=len(a))


def numeric_equal(actual,expected):
    assert np.isclose(actual,expected,rtol=1e-12,atol=1e-12), (actual,expected)


def independent_reaggregate(rows,summary):
    available=[r['pose'] for r in rows if r['pose']['available']]
    assert summary['pose']['total_frames']==len(rows)
    assert summary['pose']['available']==len(available)
    assert summary['pose']['failures']==len(rows)-len(available)
    checks=3
    for metric in ('translation_cm','rotation_deg','ADDsym_m'):
        actual=distribution([r[metric] for r in available])
        for key,value in actual.items():numeric_equal(value,summary['pose'][metric][key]);checks+=1
    corners=[r['corner'] for r in rows if r['corner']['evaluable']]
    obs=np.array([v for r in corners for v in r['observed_errors']],dtype=float)
    full=np.array([v for r in corners for v in r['errors']],dtype=float)
    cs=summary['corner']
    assert cs['corners']==len(full) and cs['observed_corners']==len(obs)
    assert cs['evaluable_frames']==len(corners) and cs['total_frames']==len(rows)
    for key,value in [('matched_pooled_corner8_median_px',np.median(obs)),
        ('matched_pooled_corner8_P90_px',np.quantile(obs,.9)),('gross20',(full>20).mean())]:
        numeric_equal(value,cs[key]);checks+=1
    for threshold in (5,10,20):numeric_equal((full<=threshold).mean(),cs['PCK'][str(threshold)]);checks+=1
    assert summary['NoOp']==sum(r['selected_index']==0 for r in rows)
    return checks+5


def verify():
    started=time.monotonic()
    setup=read(DOC/'LOSS_SETUP.json'); protocol=read(DOC/'PROTOCOL.json')
    assert setup['status']=='PASS' and protocol['code_bindings']==code_bindings()
    verify_derived(setup)
    assert setup['eligible_TRAIN']==55915 and setup['excluded_all_F_invalid']==0
    assert setup['numeric_tests']['status']=='PASS' and not setup['unsupported_nonzero_target_mass_rows']
    summary=read(DOC/'SUMMARY.json'); fits=read(DOC/'TRAIN_RECEIPTS.json')['methods']
    protected=preservation(); inputs=frozen_state()
    registry=read(OLD_DOC/'results/A_ID_MANIFEST.json')['IDs']
    train=set(registry['train']); synth=registry['synthetic_evaluation']; real=registry['real_evaluation']
    assert not train.intersection(synth) and not train.intersection(real)
    data=Data(ROOT)
    source_lookup={data.source['records'][data.indices[row]]['id']:row for row in data.eval_rows}
    source_banks=np.load(BANKS/'source_banks.npy',mmap_mode='r')
    real_bank=np.load(BANKS/'sampling_masks/REAL_DEV_generated_banks.npz')
    real_lookup={str(fid):i for i,fid in enumerate(real_bank['ids'])}
    assert len(real_lookup)==319 and (real_bank['counts']==201).all()
    train_cost=np.load(COST/'cost_ADDsym_m.npy',mmap_mode='r')
    train_available=np.load(COST/'F_available.npy',mmap_mode='r')
    native_checks=0; numeric_checks=0; final_F=0; batches=0; examples=0; probes=0; records=[]
    pnp={k:0 for k in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')}
    for method in ('SOFT6D','EXPECT6D'):
        fit=fits[method]
        assert fit['complete'] and fit['updates']==6000 and fit['exposures']==96000 and fit['excluded_exposures']==0
        ckpath=Path(fit['checkpoint_path'])
        assert sha(ckpath)==fit['checkpoint_sha256'] and ckpath.stat().st_size==fit['checkpoint_bytes']
        assert fit['code_bindings']==protocol['code_bindings'] and fit['new_final_F_calls']==0 and fit['new_backbone_forwards']==0
        assert fit['first_step']['GT_separated_from_forward'] and fit['first_step']['original_teacher_and_support_parity']
        probe=read(DOC/f'TRAIN_PROBE_{method}_seed1.json')
        assert probe['complete'] and len(probe['rows'])==256
        assert [r['id'] for r in probe['rows']]==setup['calibration']['ids'][:256]
        assert probe['execution']['new_F_calls']==0 and probe['execution']['new_refiner_examples']==256
        assert probe['execution']['new_refiner_batches']==16 and probe['GT_inference_access'] is False
        assert probe['checkpoint_sha256']==fit['checkpoint_sha256']
        assert probe['code_sha256']==sha(Path(__file__).with_name('evaluation.py'))
        assert [r['source_cache_row'] for r in probe['rows']]==setup['calibration']['rows'][:256]
        guard=read(OUTPUT/'probes'/f'{method}_seed1_ATTEMPT.json')
        assert guard['status']=='DONE' and guard['result_sha256']==sha(DOC/f'TRAIN_PROBE_{method}_seed1.json')
        expected_binding=hash_value(dict(method=method,checkpoint=fit['checkpoint_sha256'],
            protocol=sha(DOC/'PROTOCOL.json'),setup=sha(DOC/'LOSS_SETUP.json'),code=sha(Path(__file__).with_name('evaluation.py'))))
        assert probe['binding']==guard['binding']==expected_binding
        for row in probe['rows']:
            c=train_cost[row['source_cache_row']]; index=row['selected_index']; oracle_index=int(c.argmin())
            assert row['oracle_index']==oracle_index and row['oracle_exact']==(index==oracle_index)
            assert row['selected_F_available']==bool(train_available[row['source_cache_row'],index])
            numeric_equal(row['oracle_ADDsym_m'],c[oracle_index])
            if row['selected_F_available']:
                numeric_equal(row['selected_ADDsym_m'],c[index]);numeric_equal(row['oracle_gap_m'],c[index]-c[oracle_index])
            else:assert row['selected_ADDsym_m'] is None
            numeric_checks+=5
        probes+=len(probe['rows'])
        for split,ids in (('SYNTH_HELDOUT',synth),('REAL_DEV',real)):
            path=DOC/'results'/f'{split}_{method}_seed1.jsonl'
            rows=[json.loads(v) for v in path.read_text().splitlines() if v.strip()]
            receipt=read(DOC/'results'/f'{split}_{method}_seed1_EXECUTION.json')
            assert [r['id'] for r in rows]==ids and len(set(ids))==len(ids)
            assert receipt['rows_artifact']['sha256']==sha(path) and receipt['full_denominator']==len(rows)
            assert receipt['checkpoint_sha256']==fit['checkpoint_sha256']
            assert receipt['GT_inference_access'] is False and receipt['readout']=='unchanged J'
            assert receipt['code_sha256']==sha(Path(__file__).with_name('evaluation.py'))
            oracle_rows=[r for r in read(OLD_DOC/f'results/A_{split}_ORACLE.json')['rows'] if r['arm']=='GEO']
            oracle={r['id']:r for r in oracle_rows}
            raw={r['id']:r for r in read(OLD_DOC/f'results/A_{split}_BASELINES.json')['rows']['RAW']}
            for row in rows:
                fid=row['id']; index=row['selected_index']
                assert type(index) is int and 0<=index<row['action_count']
                assert row['action_count']==oracle[fid]['actions'] and row['session']==raw[fid]['session']
                assert row['F_attempt'] and row['F_complete']
                assert row['pose']['available']==row['F_available']
                if split=='SYNTH_HELDOUT':q=source_banks[source_lookup[fid],index]
                else:q=real_bank['points'][real_lookup[fid],index]
                recorded=np.array(row['native_points'],dtype=float)
                assert np.array_equal(q,recorded,equal_nan=True), 'Native bank candidate changed: '+fid
                if row['pose']['available']:
                    assert row['pose']['ADDsym_m']>=oracle[fid]['oracle_ADDsym_m']-1e-7
                native_checks+=1
            count={k:sum(r['PnP_counts'][k] for r in rows) for k in pnp}
            assert count==receipt['execution']['PnP_counts']
            for key in pnp:pnp[key]+=count[key]
            assert receipt['execution']['new_final_F_completed']==len(rows)
            assert receipt['execution']['new_final_F_attempts']==len(rows)
            assert receipt['execution']['new_refiner_examples']==len(rows)
            assert receipt['execution']['new_refiner_batches']==(125 if split=='SYNTH_HELDOUT' else 319)
            guard=read(OUTPUT/'evaluations'/f'{split}_{method}_seed1_ATTEMPT.json')
            assert guard['status']=='DONE' and guard['rows_sha256']==sha(path)
            assert guard['execution']==receipt['execution']
            final_F+=len(rows); batches+=receipt['execution']['new_refiner_batches'];examples+=receipt['execution']['new_refiner_examples']
            numeric_checks+=independent_reaggregate(rows,summary['splits'][split]['summaries'][method])
            baselines=read(OLD_DOC/f'results/A_{split}_BASELINES.json')['rows']
            controls={'N3':baselines['N3_seed1'],
                'SOFT2D_GEO':read(OLD_DOC/f'results/A_{split}_FIT_GEO_seed1.json')['rows']['GEO_J'],
                'HARD6D_GEO':read(HARD_DOC/f'results/{split}_POSE_TARGET_GEO_J_seed1.json')['rows']}
            for control,control_rows in controls.items():
                lookup={r['id']:r for r in control_rows}
                for metric in ('translation_cm','rotation_deg','ADDsym_m'):
                    delta=[r['pose'][metric]-lookup[r['id']]['pose'][metric] for r in rows
                           if r['pose']['available'] and lookup[r['id']]['pose']['available']]
                    paired=summary['splits'][split]['paired_comparisons'][method][control]['statistics'][metric]
                    for statistic in paired.values():
                        numeric_equal(float(np.mean(delta)),statistic['mean_paired_difference'])
                        numeric_equal(float(np.median(delta)),statistic['median_paired_difference'])
                        assert statistic['common_eligible_frames']==len(delta)
                        assert statistic['seed']==20260917 and statistic['resamples']==10000
                        numeric_checks+=5
            records.append(dict(method=method,split=split,rows=len(rows),sha256=sha(path),pose_available=sum(r['F_available'] for r in rows)))
    assert fits['SOFT6D']['initial_state_sha256']==fits['EXPECT6D']['initial_state_sha256']
    assert fits['SOFT6D']['order_sha256']==fits['EXPECT6D']['order_sha256']
    assert final_F==4608 and probes==512 and batches==888 and examples==4608
    assert sum(f['updates'] for f in fits.values())==12000
    owned=[p for folder in (DOC,Path(__file__).parent) for p in folder.rglob('*') if p.is_file()]
    assert not any(p.suffix.lower() in ('.tex','.bib','.pdf','.png','.pt','.npy','.npz') for p in owned)
    result=dict(schema='quick_pose_loss_verification_v1',status='PASS',seconds=time.monotonic()-started,
        scope='saved raw-row independent metric aggregation, exact native coordinate/index and identity checks, small artifact hashes and protected user state; no scientific forwards',
        preservation=protected,immutable_input_state=inputs,initialization_and_order_equal=True,
        TRAIN_heldout_identity_disjoint=True,TRAIN_rows=55915,eligible_exposures=192000,excluded_exposures=0,
        exact_native_candidates_checked=native_checks,raw_metric_aggregation_checks=numeric_checks,rows=records,
        execution=dict(formal_updates=12000,formal_exposures=192000,final_TRAIN_probe_examples=probes,
            final_evaluation_examples=examples,final_evaluation_refiner_batches=batches,final_F_calls=final_F,PnP_counts=pnp,
            final_TRAIN_probe_batches=32,evaluation_and_probe_batches=batches+32,evaluation_and_probe_examples=examples+probes,
            cost_regeneration=0,backbone_forwards=0,extra_seeds=0,paper_edits=0,paper_builds=0),
        verifier_additional_NN_forwards=0,verifier_additional_final_F=0,verifier_optimizer_updates=0,
        bootstrap_verification='paired actual-row point estimates/denominators and shared seed/resample metadata rechecked; interval algorithm independently fixture-tested; actual 10000-draw intervals were not regenerated by this verifier',
        toy_execution=dict(invocations=4,score_backward_calls=32,formal_model_forwards=0,final_F=0,optimizer_updates=0),
        formal_training_code_bindings=protocol['code_bindings'],final_code_bindings={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        interruption_policy='safe run CLI rejects any incomplete prior fit; no interrupted fit or hidden replay occurred; completed results alone are reusable',
        result_sha256=sha(DOC/'SUMMARY.json'))
    write(DOC/'VERIFICATION.json',result)
    print('QUICK_VERIFICATION',result['status'],native_checks,numeric_checks,round(result['seconds'],2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args();verify()
