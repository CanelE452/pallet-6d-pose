"""Summarize existing independent source reports; never selects a model."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
reports={name:json.loads((HERE/(name+'_ALL4942_SOURCE.json')).read_text()) for name in ['SYNTH','MIXED']}
assert all(a['n']==4942 and a['sources']==dict(SYNTH=1742,YCB=3200) for a in reports.values())
for key in ['FROZEN_RAW_SAM_PLUS_ROUGH_BOX_FULL','ALWAYS_VISIBLE','ALWAYS_EXTERNAL_SUPPORT','ALWAYS_OUTSIDE','UNIFORM_JOINT']:
    assert reports['SYNTH']['results'][key]==reports['MIXED']['results'][key],key
assert reports['SYNTH']['strata_denominators']==reports['MIXED']['strata_denominators']
result=dict(schema='V64_FROZEN_SOURCE_CONTROL_COMPARISON_V1',
            scope='4942 identical public/synthetic source validation queries; no private access or model selection',
            predeclared_primary='MIXED_MULTITARGET_UNWEIGHTED',
            exact_radius2_order=['exact','radius2'],
            note='Joint confidence U requires confidence about the whole exact/near pair; pE marginal and all-GT recall retained separately.',
            source_counts=reports['MIXED']['sources'],
            fixed_baselines_identical_between_replays=True,
            macro_NLL={k:a['macro_source_joint_NLL']['FROZEN_HEAD'] for k,a in reports.items()},
            input_hashes={name:hashlib.sha256((HERE/(name+'_ALL4942_SOURCE.json')).read_bytes()).hexdigest() for name in reports},
            evidence_denominators=reports['MIXED']['strata_denominators'],
            mechanism_caution='MIXED passing the nondegenerate source diagnostic does not establish adequate external recall or private reliability.',
            reports={})
summary=[]
for tag,report in reports.items():
    result['reports'][tag]={}
    for model in ['FROZEN_HEAD','TRAIN_EMPIRICAL_JOINT_PRIOR','ALWAYS_VISIBLE','ALWAYS_EXTERNAL_SUPPORT','ALWAYS_OUTSIDE','FROZEN_RAW_SAM_PLUS_ROUGH_BOX_FULL']:
        result['reports'][tag][model]={}
        for domain in ['SYNTH','YCB']:
            d=report['results'][model][domain]
            result['reports'][tag][model][domain]={'joint_NLL':d['joint_NLL'],'scales':{}}
            for scale in ['exact','radius2']:
                z=d['scales'][scale]
                result['reports'][tag][model][domain]['scales'][scale]=z
                for decoder in ['logical_support_diagnostic','joint_MAP_diagnostic','confident_joint_MAP_U']:
                    m=z[decoder]
                    for c in m['per_class']:
                        summary.append(dict(train_candidate=tag,model=model,domain=domain,scale=scale,decoder=decoder,
                                            class_name=['VISIBLE','EXTERNAL_SUPPORT','OUTSIDE'][c['class_index']],
                                            n=m['n'],coverage=m['coverage'],joint_NLL=d['joint_NLL'],**c))
result['MIXED_YCB_external_evidence']={}
for model in ['FROZEN_HEAD','FROZEN_RAW_SAM_PLUS_ROUGH_BOX_FULL']:
    result['MIXED_YCB_external_evidence'][model]={}
    for label in ['YCB_exact_external_corroborated','YCB_exact_external_ambiguous','YCB_strict_external_corroborated_5x5',
                  'YCB_strict_external_ambiguous_5x5','YCB_radius2_EXTERNAL_SUPPORT_CORROBORATED_ANY','YCB_radius2_EXTERNAL_SUPPORT_AMBIGUOUS']:
        d=reports['MIXED']['evidence_results'][model][label]
        scale='radius2' if label.startswith('YCB_radius2_EXTERNAL') else'exact'
        result['MIXED_YCB_external_evidence'][model][label]=d['scales'][scale]
(HERE/'SOURCE_CONTROL_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
with(HERE/'SOURCE_CONTROL_COMPARISON.csv').open('w',newline='')as f:
    writer=csv.DictWriter(f,list(summary[0]));writer.writeheader();writer.writerows(summary)
print(json.dumps(dict(n=4942,macro_NLL=result['macro_NLL'],fixed_baselines_identical=True)))
