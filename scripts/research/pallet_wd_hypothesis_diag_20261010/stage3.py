"""Fixed Metric3D accuracy gate, then SYNTH-only margin selection and S4.

Only the user-specified official depth model is inferred. Existing coordinates
and the sealed two 8-corner fits are reused; no new PnP or training is performed.
"""
import argparse
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import common as C
from . import depth as D
from . import statistics as S
from . import verdict as V
from . import stage1_resume as R


def _selections(population):
    seal=C.read(C.DOC/f'STAGE1_SELECTION_SEAL_{population}.json')
    path=C.DOC/seal['selection_path']
    assert C.sha(path)==seal['selection_sha256']
    return list(C.rows(path))


def _rgb_records(population,rows,final_only=False):
    by={r['id']:r for r in rows}
    if population=='REAL':
        audit=C.read(C.ROOT/'_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AUDIT.json')
        bindings={r['id']:r['image'] for r in audit['inputs']}
    else:
        manifest=C.read(C.SOURCE/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')
        bindings={r['id']:dict(path=r['image'],sha256=r['image_sha256'],pad=r['reflect_pad_px'])
                  for r in manifest['records'] if r['partition']=='heldout'}
    result=[]
    for fid,r in sorted(by.items()):
        if population=='REAL' and ((r['session'] in C.FINAL4)!=final_only):continue
        b=bindings[fid];path=Path(b['path'])
        if not path.is_absolute():path=C.SOURCE/path
        assert C.sha(path)==b['sha256']
        pad=int(b.get('pad',0));assert pad==0 or (population=='SYNTH' and pad==100)
        result.append(dict(population=population,id=fid,image_path=str(path),
            K=r['fixed_metadata']['K'],raw_hw=r['raw_hw'],crop_lrtb=[pad]*4))
    return result


def _truth_front(population,rows):
    ids=sorted({r['id'] for r in rows})
    if population=='REAL':
        g=C.read(C.SOURCE/C.GT)['frames'];out={}
        P=C.pose_api()
        for fid in ids:
            ref=g[fid.replace(':','__',1)];d=ref['physical_dimensions_m']
            x=P.cuboid(d['across'],d['height'],d['along'])
            z=(x@np.asarray(ref['R_gt_representative']).T+np.asarray(ref['t_gt']))[:4,2].mean()
            assert z>0;out[fid]=float(z)
        return out
    with np.load(C.SOURCE/C.GEOMETRY,allow_pickle=False) as a:
        g={k:a[k] for k in ('stems','Xcf','R','t')}
    index={str(s):i for i,s in enumerate(g['stems'])}
    out={}
    for fid in ids:
        i=index[fid];z=(g['Xcf'][i]@g['R'][i].T+g['t'][i])[:4,2].mean()
        assert z>0;out[fid]=float(z)
    return out


def _measure(population,rows,cache_dir,manifest):
    cached={r['id']:r for r in manifest['rows'] if r['population']==population}
    by=defaultdict(list)
    for r in rows:
        if r['id'] in cached:by[r['id']].append(r)
    measured=[]
    for fid,records in sorted(by.items()):
        b=cached[fid];path=cache_dir/b['cache'];assert D.sha256(path)==b['cache_sha256']
        with np.load(path,allow_pickle=False) as a:depth=a['depth_m']
        for r in records:
            estimate=D.front_depth_median(depth,r['qFinal'])
            measured.append(dict(population=population,id=fid,session=r['session'],
                method=r['method'],seed=r['seed'],**estimate))
    # All inference/depth measurements precede reference accuracy reads.
    path=C.DOC/f'DEPTH_MEASUREMENTS_{population}.jsonl.gz'
    C.write_rows(path,measured)
    C.write(C.DOC/f'DEPTH_MEASUREMENT_SEAL_{population}.json',dict(
        status='SEALED_BEFORE_REFERENCE_DEPTH_ACCURACY',rows=len(measured),
        measurement_path=path.name,measurement_sha256=C.sha(path),GT_input=False))
    return measured


def _accuracy(population,rows,measured):
    truth=_truth_front(population,rows);scored=[];groups=defaultdict(list)
    for r in measured:
        z=truth[r['id']]
        error=(r['median_m']-z)/z if r['available'] else None
        q=dict(r,reference_front_mean_z_m=z,relative_error=error,
               absolute_relative_error=None if error is None else abs(error))
        scored.append(q);groups[r['method']].append(q)
    result={}
    for method,records in sorted(groups.items()):
        perseed={}
        for seed in C.SEEDS:
            selected=[r for r in records if r['seed']==seed]
            valid=[r for r in selected if r['available']]
            abs_errors=[r['absolute_relative_error'] for r in valid]
            signed=[r['relative_error'] for r in valid]
            perseed[str(seed)]=dict(frames=len(selected),available=len(valid),
                absolute_relative_error=S.describe(abs_errors),signed_relative_error=S.describe(signed),
                signed_bias=None if not signed else float(np.mean(signed)),
                unavailable_ids=[r['id'] for r in selected if not r['available']])
        by=defaultdict(list)
        for r in records:by[r['id']].append(r)
        means=[];biases=[];missing=[]
        for fid,values in sorted(by.items()):
            assert len(values)==3 and {r['seed'] for r in values}==set(C.SEEDS)
            if not all(r['available'] for r in values):missing.append(fid);continue
            means.append(float(np.mean([r['absolute_relative_error'] for r in values])))
            biases.append(float(np.mean([r['relative_error'] for r in values])))
        result[method]=dict(frames=len(by),complete_three_seed_frames=len(means),
            seed_mean_absolute_relative_error=S.describe(means),
            signed_relative_error_seed_mean=S.describe(biases),
            signed_bias=None if not biases else float(np.mean(biases)),
            unavailable_primary_ids=missing,per_seed=perseed,
            aggregation='absolute errors first, then three-seed mean per frame, then frame median')
    C.write_rows(C.DOC/f'DEPTH_ACCURACY_ROWS_{population}.jsonl.gz',scored)
    C.write(C.DOC/f'DEPTH_ACCURACY_{population}.json',dict(population=population,
        methods=result,reference='front corners0..3 mean camera z; only for accuracy scoring',
        threshold_model_selection_final_test_frames=0))
    return result


def _choose(row,measurement,margin):
    selection=row['selection'];candidates=selection['candidates'];chosen=selection['hyp']
    gap=selection.get('formal_score_gap_px');trigger=gap is not None and gap<margin
    depths={}
    if trigger and measurement and measurement['available']:
        P=C.pose_api()
        for name,c in candidates.items():
            a=c['actual_pose']
            if not a.get('available'):continue
            cam=P.cuboid(*a['cf_extents'])@np.asarray(a['R_cf']).T+np.asarray(a['centroid'])
            depths[name]=float(cam[:4,2].mean())
        distances=[(abs(z-measurement['median_m']),i,name) for i,(name,z) in enumerate(depths.items())]
        if distances:chosen=min(distances)[2]
    actual=candidates[chosen]['actual_pose'] if chosen in candidates else selection['actual_pose']
    result=copy.deepcopy(row);result.pop('selection')
    result.update(rule='S4',hyp=chosen,actual_pose=actual,
        depth_tie=dict(m_px=margin,trigger=trigger,measured_depth_available=bool(measurement and measurement['available']),
                      measured_front_depth_m=None if measurement is None else measurement['median_m'],
                      candidate_front_mean_z_m=depths),
        fallback=bool(trigger and not(measurement and measurement['available'])),
        inference_reference_inputs=False,original_F_calls=0,new_PnP_calls=0)
    return result


def _seal_choose(population,rows,measured,margin,label):
    lookup={(r['id'],r['method'],r['seed']):r for r in measured}
    selected=[_choose(r,lookup.get((r['id'],r['method'],r['seed'])),margin) for r in rows]
    path=C.DOC/f'STAGE3_SELECTIONS_{label}_{population}.jsonl.gz'
    C.write_rows(path,selected)
    seal=dict(status='SEALED_BEFORE_REFERENCE_POSE_SCORING',rows=len(selected),
              selection_path=path.name,selection_sha256=C.sha(path),margin_px=margin,
              coordinates_changed=False,GT_for_selection=False,new_F_calls=0,new_PnP_calls=0)
    C.write(C.DOC/f'STAGE3_SELECTION_SEAL_{label}_{population}.json',seal)
    return selected


def _score(population,selected,label):
    truth,diagnostics=R.minimal_truth(population,with_margin=False);P=C.pose_api()
    scored=[]
    for r in selected:
        row=copy.deepcopy(r);row['pose']=P.metric((r['id'],r['actual_pose'],truth[r['id']]))
        row.update(elevation_deg=diagnostics[r['id']]['elevation_deg'])
        scored.append(row)
    C.write_rows(C.DOC/f'STAGE3_ROWS_{label}_{population}.jsonl.gz',scored)
    baseline=[]
    keys={(r['id'],r['method'],r['seed']) for r in selected}
    for r in C.rows(C.DOC/f'STAGE1_ROWS_{population}.jsonl.gz'):
        if (r['id'],r['method'],r['seed']) not in keys:continue
        b=copy.deepcopy(r);b.update(pose=r['pose']['S0'],hyp=r['hypS0']);baseline.append(b)
    return S.compare(baseline,scored,'REAL_DEV' if population=='REAL' else 'SYNTH_HELDOUT','S4')


def accuracy_gate(primary,forwards):
    assert primary['frames']==231
    median=primary['seed_mean_absolute_relative_error']['median']
    passed=primary['complete_three_seed_frames']==231 and median is not None and median<=V.DEPTH_ABS_REL_MEDIAN_MAX
    return dict(status='PASS' if passed else 'FAIL_STOP_S4',frames=231,primary_method=V.PRIMARY_METHOD,
        absolute_relative_error_median=median,threshold=.05,
        all_primary_seed_estimates_available=primary['complete_three_seed_frames']==231,
        missing_ids=primary['unavailable_primary_ids'],S4_executed=False,margin_tuning_executed=False,
        final_test4_in_accuracy_or_m_selection=False,
        aggregation='mean three seed absolute errors per ID, then median over231 IDs',
        model_forwards=forwards)


def _verify_sources():
    for b in C.read(C.DOC/'SOURCE_LOCK_STAGE3.json')['bindings']:
        assert C.sha(C.ROOT/b['path'])==b['sha256'], ('Stage3 source changed',b['path'])
    C.verify_core()


def run(private_dir):
    assert C.read(C.DOC/'STAGE2_REPORT_RECEIPT.json')['status']=='COMPLETE'
    assert not (C.DOC/'DEPTH_GATE.json').exists(), 'Preserve completed depth evidence'
    C.verify_core()
    paths=('stage3.py','depth.py','statistics.py','verdict.py','stage1_resume.py')
    C.write(C.DOC/'SOURCE_LOCK_STAGE3.json',dict(status='LOCKED_BEFORE_DEPTH_MODEL_CONSTRUCTION',
        bindings=[C.binding(Path(__file__).parent/n,C.ROOT) for n in paths],
        gate='N3_THEN_SUBPIX REAL231: all three absolute errors available per ID; mean per ID then median <=.05',
        missing_primary_depth='FAIL accuracy gate; do not silently reduce231',
        m_grid=list(V.DEPTH_MARGINS_PX),m_selection='SYNTH primary confusion seed mean minimum, smallest m exact tie',
        final_test_threshold_model_selection_frames=0))
    private=Path(private_dir).resolve();modelroot=private/'metric3d'
    rows={p:_selections(p) for p in ('REAL','SYNTH')}
    inputs=_rgb_records('SYNTH',rows['SYNTH'])+_rgb_records('REAL',rows['REAL'])
    assert len(inputs)==1985+231
    cache_dir=private/'depth_accuracy_cache'
    manifest=D.cache(inputs,modelroot,cache_dir)
    C.write(C.DOC/'DEPTH_INFERENCE_RECEIPT.json',dict(status=manifest['status'],
        frames=manifest['frames'],model_forwards=manifest['model_forwards'],checkpoint=manifest['checkpoint'],
        private_cache=True,GT_used_for_inference=False,training_updates=0,
        inference_lock_sha256=D.sha256(cache_dir/'DEPTH_INFERENCE_LOCK.json')))
    measured={p:_measure(p,rows[p],cache_dir,manifest) for p in ('SYNTH','REAL')}
    accuracy={p:_accuracy(p,rows[p],measured[p]) for p in ('SYNTH','REAL')}
    primary=accuracy['REAL'][V.PRIMARY_METHOD]
    gate=accuracy_gate(primary,manifest['model_forwards'])
    C.write(C.DOC/'DEPTH_GATE.json',gate)
    if gate['status']!='PASS':
        _verify_sources()
        C.write(C.DOC/'STAGE3_EXECUTION.json',dict(status='COMPLETE_ACCURACY_GATE_FAIL_STOP_S4',
            accuracy_gate=gate['status'],actual_depth_forwards=manifest['model_forwards'],
            actual_model_constructions=1,new_F_calls=0,new_PnP_calls=0,training_updates=0,
            S4_REAL_evaluations=0,margin_tuning_evaluations=0,final_test4_tuning_frames=0,
            source_hashes_unchanged=True))
        print('DEPTH_GATE FAIL_STOP_S4',gate['absolute_relative_error_median'],flush=True);return gate
    grid={};gridresults={}
    primaryrows=[r for r in rows['SYNTH'] if r['method']==V.PRIMARY_METHOD]
    for margin in V.DEPTH_MARGINS_PX:
        label=f'm{int(margin)}'
        selected=_seal_choose('SYNTH',primaryrows,measured['SYNTH'],margin,label)
        result=_score('SYNTH',selected,label);gridresults[margin]=result
        confusion=result['metrics'][V.PRIMARY_METHOD]['seed_mean']['S4']['rates']['confusion_rate']['rate']
        grid[str(int(margin))]=dict(confusion_rate=confusion,primary=result['primary'])
    margin=min(V.DEPTH_MARGINS_PX,key=lambda m:(grid[str(int(m))]['confusion_rate'],m))
    C.write(C.DOC/'S4_MARGIN_LOCK.json',dict(status='FROZEN_ON_SYNTH_BEFORE_ANY_REAL_S4_SELECTION',
        margin_px=margin,grid=grid,selection='minimum primary confusion seed mean; smaller margin exact tie',
        real_reference_for_margin_selection=False,final_test4_for_margin_selection=False))
    results={}
    selected=_seal_choose('SYNTH',rows['SYNTH'],measured['SYNTH'],margin,'S4')
    results['SYNTH']=_score('SYNTH',selected,'S4')
    remaining=_rgb_records('REAL',rows['REAL'],final_only=True);assert len(remaining)==88
    final_cache=private/'depth_final88_cache';final_manifest=D.cache(remaining,modelroot,final_cache)
    final_measured=[]
    # This supplementary measurement is sealed but never used for accuracy/tuning.
    lookup={r['id']:r for r in final_manifest['rows']}
    for fid,b in sorted(lookup.items()):
        assert D.sha256(final_cache/b['cache'])==b['cache_sha256']
        with np.load(final_cache/b['cache'],allow_pickle=False) as a:depth=a['depth_m']
        for r in rows['REAL']:
            if r['id']==fid:
                final_measured.append(dict(population='REAL',id=fid,session=r['session'],
                    method=r['method'],seed=r['seed'],**D.front_depth_median(depth,r['qFinal'])))
    C.write_rows(C.DOC/'DEPTH_MEASUREMENTS_FINAL88.jsonl.gz',final_measured)
    selected=_seal_choose('REAL',rows['REAL'],measured['REAL']+final_measured,margin,'S4')
    results['REAL']=_score('REAL',selected,'S4')
    for p,result in results.items():C.write(C.DOC/f'RESULTS_S4_{p}.json',result)
    verdict=V.evaluate(results['SYNTH']['primary'],results['REAL']['primary'],rule='S4')
    C.write(C.DOC/'VERDICT_S4.json',verdict)
    _verify_sources()
    C.write(C.DOC/'STAGE3_EXECUTION.json',dict(status='COMPLETE',accuracy_gate='PASS',
        actual_depth_forwards=manifest['model_forwards']+final_manifest['model_forwards'],
        actual_model_constructions=2,new_F_calls=0,new_PnP_calls=0,training_updates=0,
        margin_px=margin,S4_REAL_evaluations=1,final_test4_tuning_frames=0))
    return verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();run(args.private_dir)
