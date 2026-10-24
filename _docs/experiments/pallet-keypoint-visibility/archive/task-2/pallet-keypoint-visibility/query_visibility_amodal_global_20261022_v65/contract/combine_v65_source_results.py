"""Summarize the two locked source runs and confidence-policy cost, no selection."""
import json
from pathlib import Path
import numpy as np
import saved_evaluate_frozen_source as base
from source_support_contract import JOINT_STATES,confident_joint_map,joint_marginals

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
PAIRS=np.asarray(JOINT_STATES)


def confidence_cost(path):
    with np.load(path,allow_pickle=False)as z:
        p=base.softmax(z['joint_logits']);y=z['truth_joint'];source=z['source']
    decision=confident_joint_map(p);statep=joint_marginals(p);best=PAIRS[p.argmax(1)]
    result={}
    for domain in ['SYNTH','YCB']:
        result[domain]={}
        domain_mask=source==domain
        for scale,name in enumerate(['exact','radius2']):
            truthE=domain_mask&(PAIRS[y,scale]==1)
            mapE=domain_mask&(best[:,scale]==1)
            acceptedE=domain_mask&(decision['classes'][:,scale]==1)
            withheld=~decision['accepted']
            result[domain][name]=dict(GT_E=int(truthE.sum()),
                jointMAP_E_calls=int(mapE.sum()),jointMAP_E_correct=int((mapE&truthE).sum()),
                confidence_E_calls=int(acceptedE.sum()),confidence_E_correct=int((acceptedE&truthE).sum()),
                correct_E_calls_lost_to_U=int((mapE&truthE&withheld).sum()),
                incorrect_E_calls_removed_by_U=int((mapE&~truthE&withheld).sum()),
                GT_E_pair_U=int((truthE&withheld).sum()),
                GT_E_pE_ge_half_but_joint_pair_U=int((truthE&(statep[:,scale,1]>=.5)&withheld).sum()),
                explanation='U accepts only a whole exact/near joint state with probability>=.5; E marginal evidence can be divided between EV andEE. This is a fixed policy cost, not a fitted new decision.')
    return result


def main():
    release=HERE/'RELEASE_RULE_BEFORE_FIT.json'
    result=dict(schema='V65_TWO_LOCKED_FITS_SOURCE_RELEASE_SUMMARY_V1',
                scope='Source-only, same4942queries, no private/model inference or checkpoint selection.',
                predeclared_primary='MIXED_GLOBAL_UNWEIGHTED',
                release_rule_sha256=base.sha(release),comparisons={},confidence_policy_cost={},
                narrative='Global shape supervision improves the preregistered MIXED source engineering criteria, while SYNTH-only transfer remains HOLD. Lower jointNLL describes probability fit; it does not by itself establish external-occlusion discrimination. Confidence-U improves E precision by withholding both correct and incorrect E calls, so all-GT recall remains necessary.',
                source_semantics='E is full-minus-visible source support; semantic self and certified physical-human corner visibility remain unsupported.',
                no_private_data_or_new_models_GPU_training_PnP_calls=True)
    records=[]
    for tag in ['SYNTH','MIXED']:
        newpath=HERE/(tag+'_ALL4942_SOURCE.json')
        oldpath=V64/'contract'/(tag+'_ALL4942_SOURCE.json')
        new,old=json.loads(newpath.read_text()),json.loads(oldpath.read_text())
        assert new['n']==old['n']==4942 and new['sources']==old['sources']
        result['comparisons'][tag]=dict(
            v64_macro_source_joint_NLL=old['macro_source_joint_NLL']['FROZEN_HEAD'],
            v65_macro_source_joint_NLL=new['macro_source_joint_NLL']['FROZEN_HEAD'],
            release=new['comparative_release_vs_frozen_v64'],
            per_domain_NLL={s:dict(v64=old['results']['FROZEN_HEAD'][s]['joint_NLL'],v65=new['results']['FROZEN_HEAD'][s]['joint_NLL'])for s in ['SYNTH','YCB']},
            bindings={str(p):base.sha(p)for p in [newpath,oldpath]})
        for version,report in [('v64',old),('v65',new)]:
            for domain in ['SYNTH','YCB']:
                for scale in ['exact','radius2']:
                    stats=report['results']['FROZEN_HEAD'][domain]['scales'][scale]
                    for decoder in ['logical_support_diagnostic','joint_MAP_diagnostic','confident_joint_MAP_U']:
                        d=stats[decoder]
                        for cls in d['per_class']:
                            records.append(dict(version=version,training=tag,domain=domain,scale=scale,decoder=decoder,
                                                class_name=base.CLASS_NAMES[cls['class_index']],n=d['n'],coverage=d['coverage'],**cls))
        output=ROOT/'source_comparison'/(tag+'_GLOBAL_UNWEIGHTED.npz')
        result['confidence_policy_cost'][tag]=confidence_cost(output)
    mixed=json.loads((HERE/'MIXED_ALL4942_SOURCE.json').read_text())
    evidence=mixed['evidence_results']['FROZEN_HEAD']
    result['MIXED_YCB_exact_E_evidence']={}
    for label in ['YCB_exact_external_corroborated','YCB_exact_external_ambiguous',
                  'YCB_strict_external_corroborated_5x5','YCB_strict_external_ambiguous_5x5']:
        stats=evidence[label]['scales']['exact']
        result['MIXED_YCB_exact_E_evidence'][label]={key:stats[key]for key in ['logical_support_diagnostic','joint_MAP_diagnostic','confident_joint_MAP_U']}
    for tag in ['SYNTH','MIXED']:
        result['comparisons'][tag]['saved_GPU_CPU_numeric_parity']=json.loads((HERE/(tag+'_SAVED_REPLAY_PARITY.json')).read_text())
    (HERE/'SOURCE_RELEASE_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    base.write_csv(HERE/'V64_V65_SOURCE_CLASS_COMPARISON.csv',records)
    print(json.dumps(dict(macro_NLL={k:[v['v64_macro_source_joint_NLL'],v['v65_macro_source_joint_NLL']]for k,v in result['comparisons'].items()},
                          release={k:v['release']['RELEASE']for k,v in result['comparisons'].items()},
                          MIXED_YCB_exact_confidence_cost=result['confidence_policy_cost']['MIXED']['YCB']['exact'])))


if __name__=='__main__':main()
