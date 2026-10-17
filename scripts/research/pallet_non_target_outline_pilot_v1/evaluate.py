"""Post-seal evaluator; references never enter supplier or selector processes."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

from scripts.research.pallet_statewise_correspondence_diagnosis_v1.analysis import metric


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path,'rt')if str(path).endswith('.gz')else Path(path).open())as f:
        return [json.loads(l)for l in f]


def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def csvwrite(path,values):
    with Path(path).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def run(prediction_folder,neutral_package,state_private,output,go):
    if not go:raise ValueError('Explicit postseal evaluation flag required')
    started,cpu=time.monotonic(),time.process_time()
    pred,package,state,out=map(Path,(prediction_folder,neutral_package,state_private,output))
    seal=read(pred/'PREDICTION_SEAL.json')
    for name,digest in seal['artifacts'].items():
        if sha(pred/name)!=digest:raise ValueError('Prediction seal changed')
    if sha(package)!=seal['neutral_package_sha256']:raise ValueError('Prediction input package changed')
    predictions=rows(pred/'PREDICTIONS.jsonl.gz')
    if len(predictions)!=18 or len({(r['id'],r['variant'])for r in predictions})!=18:
        raise ValueError('Six frames and three complete methods required')
    pkg=read(package);frames=rows(package.parent/pkg['frames']['path']);byid={f['id']:f for f in frames}
    # Reference read occurs strictly after all immutable prediction checks above.
    reference_path=state/'score_v2/FIXED_REFERENCES.json';refseal=read(state/'score_v2/SCORE_SEAL.json')
    if sha(reference_path)!=refseal['artifacts']['FIXED_REFERENCES.json']:raise ValueError('Original reference changed')
    refs=read(reference_path)['references']
    scored=[]
    for p in predictions:
        f=byid[p['id']];m=metric(p['pose'],refs[f['partition']][p['id']])
        scored.append(dict(id=p['id'],ordinal=f['ordinal'],partition=f['partition'],variant=p['variant'],
            candidate_id=p['candidate_id'],family=p['selected_family'],accepted=p['accepted'],fallback=p['fallback'],
            translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success_le5=m['success_le5'],
            pose_available=m['pose_available'],reference_rotation_available=m['reference_rotation_available']))
    lookup={(r['id'],r['variant']):r for r in scored};transitions=[]
    for f in frames:
        baseline=lookup[(f['id'],'BASELINE_E')]
        for variant in ('SUPPORT_AWARE_RERANK','MASS_MATCHED_CONTROL'):
            current=lookup[(f['id'],variant)]
            b,s=baseline['success_le5'],current['success_le5']
            transitions.append(dict(id=f['id'],ordinal=f['ordinal'],variant=variant,
                baseline_candidate_id=baseline['candidate_id'],new_candidate_id=current['candidate_id'],
                candidate_changed=baseline['candidate_id']!=current['candidate_id'],
                family_changed=baseline['family']!=current['family'],baseline_success=b,new_success=s,
                transition='RECOVERY'if s and not b else 'DAMAGE'if b and not s else 'PRESERVED_SUCCESS'if s else 'PERSISTENT_FAILURE',
                translation_delta_cm=current['translation_cm']-baseline['translation_cm'],
                rotation_delta_deg=current['rotation_deg']-baseline['rotation_deg']))
    summary={}
    for variant in seal['variants']:
        own=[r for r in scored if r['variant']==variant];tr=[r for r in transitions if r['variant']==variant]
        t=[r['translation_cm']for r in own if r['translation_cm']is not None]
        r=[r['rotation_deg']for r in own if r['rotation_deg']is not None]
        summary[variant]=dict(denominator=6,successes=sum(x['success_le5']is True for x in own),
            recoveries=sum(x['transition']=='RECOVERY'for x in tr),damages=sum(x['transition']=='DAMAGE'for x in tr),
            changed_candidates=sum(x['candidate_changed']for x in tr),changed_families=sum(x['family_changed']for x in tr),
            translation_mean_cm=statistics.mean(t),translation_sample_SD_cm=statistics.stdev(t),translation_max_cm=max(t),
            rotation_mean_deg=statistics.mean(r),rotation_sample_SD_deg=statistics.stdev(r),rotation_max_deg=max(r),
            translation_gt50cm=sum(x>50 for x in t),translation_gt100cm=sum(x>100 for x in t))
    base=summary['BASELINE_E']['successes']
    for v in ('SUPPORT_AWARE_RERANK','MASS_MATCHED_CONTROL'):
        if summary[v]['successes']-base!=summary[v]['recoveries']-summary[v]['damages']:
            raise ValueError('Paired success arithmetic failed')
    out.mkdir(parents=True,exist_ok=False)
    csvwrite(out/'METRICS.csv',scored);csvwrite(out/'TRANSITIONS.csv',transitions)
    write(out/'SUMMARY.json',dict(scope='ASSISTED6 prior-selected explanatory cases, not automatic system performance',
        AUTO238_executed=False,populations_combined=False,metrics=summary))
    write(out/'EVALUATION_RECEIPT.json',dict(prediction_seal_sha256=sha(pred/'PREDICTION_SEAL.json'),
        reference_sha256=sha(reference_path),reference_seal_sha256=sha(state/'score_v2/SCORE_SEAL.json'),
        predictions_sealed_before_reference_open=True,threshold='same pose T<=5cm AND physical/C2 R<=5deg',
        physical_reference_is_independent_truth=False,new_pose_fit=0,new_NN=0,
        artifacts={f.name:sha(f)for f in out.iterdir()},elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('prediction-folder','neutral-package','state-private','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--postseal-go',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.prediction_folder,a.neutral_package,a.state_private,a.output,a.postseal_go),indent=2))
