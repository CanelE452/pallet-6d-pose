"""Public-only independent human-state strata and gate-order review."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values=sorted(values);n=len(values)
    if not n:return dict(n=0,mean=None,variance=None,std=None,median=None,P90=None,max=None)
    mean=math.fsum(values)/n
    variance=math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None
    def quantile(p):
        at=(n-1)*p;low=math.floor(at);high=math.ceil(at);fraction=at-low
        return values[low]+fraction*(values[high]-values[low])
    return dict(n=n,mean=mean,variance=variance,std=math.sqrt(variance) if variance is not None else None,
                median=quantile(.5),P90=quantile(.9),max=values[-1])


def run(docs):
    labels=read(docs/'VISIBILITY_LABELS.json')
    result=read(docs/'VISIBILITY_RESULTS.json')
    assert labels['used_in_inference'] is False and result['inference_used_human_states'] is False
    assert sha(docs/'VISIBILITY_LABELS.json')==result['labels_sha256']
    assert sha(docs/'PREDICTIONS.jsonl.gz')==result['predictions_sha256']
    states=result['states']
    by_corner={(r['id'],int(r['corner'])):r['category'] for r in labels['labels']}
    expected=dict(DIRECT_VISIBLE=1776,SELF_OCCLUDED=462,EXTERNAL_OCCLUDED=218,OUT_OF_FRAME=43)
    assert len(labels['labels'])==len(by_corner)==2499
    assert dict(Counter(by_corner.values()))==expected==result['human_reference_state_counts']
    assert len({fid for fid,k in by_corner})==319 and all(0<=k<8 for fid,k in by_corner)
    with gzip.open(docs/'PREDICTIONS.jsonl.gz','rt') as stream:
        raw=[json.loads(line) for line in stream]
    lookup={(r['seed'],r['method'],r['id']):r for r in raw}
    assert len(lookup)==len(raw)==5742
    count=0;maxstat=Counter();details=[]
    for seed,methods in result['by_seed'].items():
        assert len(methods)==6
        for method,reported in methods.items():
            rr=[r for r in raw if r['seed']==int(seed) and r['method']==method]
            assert len(rr)==319 and len({r['id'] for r in rr})==319
            grouped={s:[] for s in states};missing=0
            for r in rr:
                valid=r['corner']['canonical_valid']
                assert len(valid)==len(r['corner']['canonical_errors'])==len(r['canonical_observed'])==8
                for k,v in enumerate(valid):
                    key=(r['id'],k)
                    if not v:
                        assert key not in by_corner;missing+=1;continue
                    assert key in by_corner
                    grouped[by_corner[key]].append((r,k))
            assert missing==53==reported['missing_reference_corner_slots']
            assert reported['original_corner_slots']==2552 and reported['reference_corners']==2499
            assert sum(len(v) for v in grouped.values())==2499
            for state,pairs in grouped.items():
                row=reported['states'][state]
                full=[r['corner']['canonical_errors'][k] for r,k in pairs]
                observed=[r['corner']['canonical_errors'][k] for r,k in pairs if r['canonical_observed'][k]]
                assert all(v is not None and math.isfinite(v) for v in full)
                assert row['reference_corners']==len(full)
                assert row['observed_corners']==len(observed)
                assert row['frames']==len({r['id'] for r,k in pairs})
                computed=stats(observed)
                for name,value in computed.items():
                    old=row['observed_error_px'][name]
                    if value is None:assert old is None
                    elif name=='n':assert old==value
                    else:
                        delta=abs(value-old);maxstat[name]=max(maxstat[name],delta)
                        assert math.isclose(value,old,rel_tol=1e-12,abs_tol=1e-8),(seed,method,state,name,value,old)
                pck=sum(v<=10 for v in full)/len(full) if full else None
                assert pck==row['PCK10']
                assert sum(v>20 for v in full)==row['gross20_count']
                for baseline,damage in row['damage_vs'].items():
                    old=[lookup[(int(seed),baseline,r['id'])]['corner']['canonical_errors'][k] for r,k in pairs]
                    assert damage['good5_to_bad10']==sum(a<5 and b>10 for a,b in zip(old,full))
                    assert damage['bad20_to_good10']==sum(a>20 and b<=10 for a,b in zip(old,full))
                details.append(dict(seed=int(seed),method=method,state=state,reference_corners=len(full),observed_corners=len(observed),
                    frames=row['frames'],PCK10=pck,gross20_count=row['gross20_count']))
                count+=1
            assert sum(v['observed_corners'] for v in reported['states'].values())==2445
    code=Path(__file__).resolve().parent
    gate_files=['preflight.py','capture.py','coordinates.py','evaluate.py','visibility.py']
    config=read(docs/'CONFIG_LOCK.json');seal=read(docs/'COORDINATES_SEAL.json')
    assert seal['config_lock_sha256']==sha(docs/'CONFIG_LOCK.json')
    assert config['fusion_method_lock_sha256']==sha(docs/'FUSION_METHOD_LOCK.json')
    for binding in config['source_bindings']:
        assert sha(code.parents[2]/binding['path'])==binding['sha256']
    cap_ast=ast.parse((code/'capture.py').read_text())
    project=next(n for n in cap_ast.body if isinstance(n,ast.FunctionDef) and n.name=='baseline_points')
    fields={n.args[1].value for n in ast.walk(project) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='field'}
    assert fields=={'method','seed','id','qFinal'}
    evaluation=(code/'evaluate.py').read_text()
    assert evaluation.index("seal = C.read(C.DOC / 'COORDINATES_SEAL.json')")<evaluation.index('E, frames, targets, _, _ = load_real()')
    assert 'evaluation_references_loaded=False' in (code/'coordinates.py').read_text()
    assert seal['evaluation_references_loaded'] is False and seal['F_calls_before_seal']==0
    return dict(schema='independent_public_auxiliary_verification_v1',status='PASS',
        implementation='Python math.fsum, scalar sample variance and sorted linear quantiles; no visibility/aggregation/evaluator imports.',
        public_prediction_rows=len(raw),seed_method_groups=18,human_state_groups_verified=count,
        human_reference_state_counts=expected,missing_reference_slots_per_seed_method=53,
        maximum_distribution_absolute_difference=dict(maxstat),verified_strata=details,
        checks=dict(unique_human_state_identity_corner_labels=True,full2499_and_observed2445_denominators=True,
            all_seed_6method_PCK_gross_damage_correct=True,mean_variance_std_median_P90_max_correct=True,
            labels_used_only_post_evaluation=True,baseline_inference_projection_no_GT_pose_error_fields=True,
            fusion_method_and_source_hashes_sealed_before_coordinate_generation=True,reference_loading_after_coordinate_seal=True),
        gate_order_review=dict(baseline_projection_fields=sorted(fields),
            limitation='Source/control-flow and sealed-hash review supports the reproduction order. It does not claim filesystem access tracing beyond recorded execution.',
            verified_source_sha256={n:sha(code/n) for n in gate_files}),
        source_bindings=[dict(path=n,sha256=sha(docs/n)) for n in ['VISIBILITY_LABELS.json','VISIBILITY_RESULTS.json','PREDICTIONS.jsonl.gz','CONFIG_LOCK.json','COORDINATES_SEAL.json']],
        verifier=dict(path='scripts/research/pallet_feature_gradient_joint_20261010/verify_auxiliary.py',sha256=sha(Path(__file__))),
        execution=dict(model_calls=0,F_calls=0,optimizer_updates=0,private_input_reads=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--docs',type=Path,default=Path(__file__).resolve().parents[3]/'_docs/experiments/pallet_feature_gradient_joint_20261010')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=run(a.docs)
    output=a.output or a.docs/'PUBLIC_AUXILIARY_VERIFICATION.json'
    with output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','human_state_groups_verified','human_reference_state_counts','missing_reference_slots_per_seed_method','maximum_distribution_absolute_difference','checks']},ensure_ascii=False))


if __name__=='__main__':main()
