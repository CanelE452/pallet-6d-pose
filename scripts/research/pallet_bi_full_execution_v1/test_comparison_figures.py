"""Constructed saved-metric metadata only; no dataset/plot/reference calls."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from . import comparison_figures as C


def save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt')as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')


def tests():
    data={};arms=[name+'_s'+str(seed)for name in ('BASE','NEW')for seed in C.SEEDS]
    for i in range(238):
        metric=dict(translation_cm=.2,rotation_deg=.4,success=True,pose_available=True)
        identity=dict(id=f'CONSTRUCTED_{i:03d}',partition='TEST'if i<49 else'REMAINDER'if i<192 else'CAL',session='SESSION',cluster_id='REC_'+str(i%6))
        data[identity['id']]=dict(identity,metrics={arm:dict(metric)for arm in arms})
    for seed in C.SEEDS:
        data['CONSTRUCTED_005']['metrics']['BASE_s'+str(seed)]=dict(translation_cm=10000.,rotation_deg=179.,success=False,pose_available=True)
        data['CONSTRUCTED_006']['metrics']['NEW_s'+str(seed)]=dict(translation_cm=None,rotation_deg=None,success=False,pose_available=False)
    plan=dict(pairs=[dict(stage='CONSTRUCTED',name='BASE_NEW',before='BASE',after='NEW')],representative=dict(stage='CONSTRUCTED',pair='BASE_NEW',seed=C.SEEDS[0]))
    payload=C.build_payload(plan,data);assert len(payload['datasets'])==3
    dataset=payload['datasets'][0];assert len(dataset['points'])==192
    assert all(int(row['id'].rsplit('_',1)[-1])<192 for row in dataset['points'])
    stats=C.view_stats(dataset,20,20);assert stats['full_x_max']>10000 and stats['full_y_max']>179
    assert stats['sides']['before']['outside_zoom']==1 and stats['sides']['after']['unavailable']==1
    assert dataset['points'][5]['before']['translation_cm']==10000
    assert dataset['points'][6]['after']['translation_cm']is None
    for row in data.values():row['metrics']['FROZEN_PF']=dict(row['metrics'][arms[0]])
    static=deepcopy(plan);static['pairs'][0].update(before='FROZEN_PF',before_seeded=False)
    frozen=C.build_payload(static,data)['datasets']
    assert all(d['before_arm']=='FROZEN_PF'and not d['before_seeded']for d in frozen)
    assert all(d['static_estimator_note']for d in frozen)
    assert frozen[0]['points'][0]['before']==frozen[2]['points'][0]['before']
    assert 'FROZEN_PF_s'+str(C.SEEDS[0])not in data['CONSTRUCTED_000']['metrics']
    bad=deepcopy(plan);bad['pairs']*=2
    try:C.build_payload(bad,data)
    except ValueError:pass
    else:raise AssertionError('Duplicate pair output name accepted')
    contaminated=dict(translation_cm=1,rotation_deg=2,success=True,pose_available=True,R=[[1]],t=[2],reference=dict(R=[[1]],t=[2]))
    assert set(C.public_metric(contaminated))=={'translation_cm','rotation_deg','success','pose_available'}
    with TemporaryDirectory(prefix='comparison-figures-metadata-')as tmp:
        root=Path(tmp);ev=root/'evaluation';ind=root/'independent';pred=root/'predictions'
        save(pred/'PREDICTIONS.jsonl.gz',[dict(id=fid,constructed=True)for fid in data]);C.M.write(pred/'PREDICTION_SEAL.json',dict(complete=True))
        save(ev/'SCORED_ROWS.jsonl.gz',data.values());save(ind/'INDEPENDENT_ROWS.jsonl.gz',data.values())
        C.M.write(ev/'EVALUATION_SEAL.json',dict(complete=True,arms=arms,reference_sha256='CONSTRUCTED_NO_REFERENCE_FILE',
            prediction_bindings=[dict(path=str(pred/'PREDICTIONS.jsonl.gz'),sha256=C.M.sha(pred/'PREDICTIONS.jsonl.gz'),seal_sha256=C.M.sha(pred/'PREDICTION_SEAL.json'))],
            artifacts={'SCORED_ROWS.jsonl.gz':C.M.sha(ev/'SCORED_ROWS.jsonl.gz')}))
        C.M.write(ind/'INDEPENDENT_VERIFICATION.json',dict(PASS=True,canonical_evaluation_seal_sha256=C.M.sha(ev/'EVALUATION_SEAL.json')))
        for name in ('INDEPENDENT_SUMMARY.json','INDEPENDENT_PAIRED.json'):C.M.write(ind/name,dict(constructed=True))
        plan['sources']=[dict(evaluation_dir=str(ev),independent_dir=str(ind),independent_artifacts={name:C.M.sha(ind/name)for name in C.INDEPENDENT})]
        mp=root/'MANIFEST.json';C.M.write(mp,plan);_,merged,_=C.authenticate(mp);assert len(merged)==238
        # Authentic but disagreeing independent values must fail agreement.
        changed=deepcopy(list(data.values()));changed[0]['metrics'][arms[0]]['translation_cm']=42.;save(ind/'INDEPENDENT_ROWS.jsonl.gz',changed)
        plan2=deepcopy(plan);plan2['sources'][0]['independent_artifacts']['INDEPENDENT_ROWS.jsonl.gz']=C.M.sha(ind/'INDEPENDENT_ROWS.jsonl.gz')
        mp2=root/'CHANGED.json';C.M.write(mp2,plan2)
        try:C.authenticate(mp2)
        except ValueError:pass
        else:raise AssertionError('Canonical/independent disagreement accepted')
        # Parse the actual embedded JavaScript without opening a browser or
        # importing Matplotlib. Escape closes any hostile JSON script tag.
        embedded=json.dumps(payload).replace('<','\\u003c')
        page=C.HTML.replace('__DATA__',embedded);script=page.rsplit('<script>',1)[1].split('</script>',1)[0]
        js=root/'figure.js';js.write_text(script);node=shutil.which('node')
        if node:subprocess.run([node,'--check',str(js)],check=True,capture_output=True)
    return dict(PASS=True,constructed_metadata_only=True,actual_dataset_rows_read=0,images_opened=0,references_decoded=0,
        actual_metric_cost_solve_fit_forward_calls=0,actual_plots_rendered=0,PRIMARY192_all3seeds_retained=True,
        full_outlier_and_zoom_outside_count_preserved=True,no_pose_not_zero_imputed=True,CAL_not_exported=True,
        static_PF_arm_keeps_actual_name_and_one_frozen_estimator_disclosure=True,
        public_fields_exclude_R_t_reference=True,canonical_independent_disagreement_and_duplicate_pairs_rejected=True,
        actual_JavaScript_syntax_checked=bool(node),source_sha256=C.M.sha(C.__file__))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=tests();C.M.write(a.output,r);print(json.dumps(r,sort_keys=True))
