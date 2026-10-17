"""Fixed-head synthetic distributions, then a separate postprediction evaluation.

No fitting or model choice is performed. The distributions command has the
same reference firewall as the existing selector. Only evaluate opens targets,
after both the original pose predictions and complete distributions are sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import resource
import statistics
import time

from .common import read, rows, save_rows, sha, write

SEEDS=(20261018,20261019,20261020)
KINDS=('L_LINEAR','L_HEAD')
ARMS=tuple(k+'_s'+str(s) for k in KINDS for s in SEEDS)


def statistics_of(values):
    x=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    def percentile(p):
        z=(len(x)-1)*p;i=int(z)
        return x[i]+(z-i)*(x[min(i+1,len(x)-1)]-x[i])
    if not x:return dict(n=0,mean=None,sample_std=None,sample_variance=None,median=None,P90=None,P95=None,max=None)
    variance=statistics.variance(x) if len(x)>1 else None
    return dict(n=len(x),mean=statistics.fmean(x),sample_std=math.sqrt(variance) if variance is not None else None,
                sample_variance=variance,median=statistics.median(x),P90=percentile(.9),P95=percentile(.95),max=x[-1])


def _cost(start,cpu):
    return dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1)


def distributions(run_dir,output=None):
    """Repeat only the tiny frozen heads to archive log probabilities, GT-free."""
    from . import learned_head as L
    import torch
    L.cpu_config();start,cpu=time.monotonic(),time.process_time();run=Path(run_dir)
    feature_dir=run/'synthetic_features_v1';training=run/'head_training_v1';pred=run/'learned_synthetic_v1'
    features=feature_dir/'HEAD_FEATURES.jsonl.gz';featureseal=feature_dir/'FEATURE_SEAL.json'
    fdigest=L.bound(features,featureseal);manifest=read(training/'TRAINING_MANIFEST.json');ps=read(pred/'PREDICTION_SEAL.json')
    if (manifest.get('complete') is not True or manifest.get('total_training_runs')!=6
            or ps.get('before_reference_decode') is not True or ps.get('reference_reads')!=0
            or ps['features_sha256']!=fdigest or ps['training_manifest_sha256']!=sha(training/'TRAINING_MANIFEST.json')
            or sha(pred/'PREDICTIONS.jsonl.gz')!=ps['predictions_sha256']):
        raise ValueError('Complete matching frozen six-head pose predictions required first')
    out=Path(output or run/'head_synthetic_distributions_v1');out.mkdir(parents=True,exist_ok=False)
    models={};checkpoints={};result=[]
    with L.prediction_no_refs():
        original={r['id']:r for r in rows(pred/'PREDICTIONS.jsonl.gz')}
        for r in manifest['runs']:
            path=Path(r['checkpoint']);arm=r['kind']+'_s'+str(r['seed'])
            if sha(path)!=r['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
            ck=torch.load(path,map_location='cpu',weights_only=False);m=L.make_model(ck['kind'],ck['input_dim'])
            m.load_state_dict(ck['model_state_dict']);m.eval();m.requires_grad_(False)
            models[arm]=(m,ck['normalizer']);checkpoints[arm]=r['checkpoint_sha256']
        if set(models)!=set(ARMS):raise ValueError('Exactly the six predeclared heads required')
        for frame in rows(features):
            L.validate_frame(frame);values={}
            for arm,(model,normalizer) in models.items():
                x,indices=L.model_inputs(frame,normalizer)
                with torch.inference_mode():
                    logits=model(x).squeeze(-1);logp=logits.log_softmax(0);probs=logits.softmax(0);index=int(logits.argmax())
                if not torch.isfinite(logp).all():raise ValueError('Nonfinite frozen-head log probabilities')
                ids=[frame['candidates'][i]['candidate_id'] for i in indices];none=index==len(ids)
                chosen='N0' if none else ids[index];p=original[frame['id']]['predictions'][arm]
                if (p['selected_candidate_id']!=chosen or p['none_selected'] is not none
                        or abs(p['probability']-float(probs[index]))>1e-6):
                    raise ValueError('Distribution argmax differs from sealed pose prediction')
                values[arm]=dict(candidate_ids=ids,none_index=len(ids),log_probabilities=logp.tolist(),
                                 selected_index=index,none_selected=none)
            result.append(dict(id=frame['id'],split=frame['split'],predictions=values,reference_inputs=False))
        if len(result)!=1003 or set(original)!={r['id'] for r in result}:raise ValueError('Exact1003 expansion required')
        if [r['id'] for r in result]!=ps['ordered_ids']:raise ValueError('Original synthetic prediction order changed')
        save_rows(out/'DISTRIBUTIONS.jsonl.gz',result)
    receipt=dict(schema='FIXED_HEAD_SYNTHETIC_DISTRIBUTION_SEAL_V1',complete=True,rows=len(result),arms=list(ARMS),
                 distributions_sha256=sha(out/'DISTRIBUTIONS.jsonl.gz'),prediction_seal_sha256=sha(pred/'PREDICTION_SEAL.json'),
                 predictions_sha256=ps['predictions_sha256'],features_sha256=fdigest,feature_seal_sha256=sha(featureseal),
                 training_manifest_sha256=sha(training/'TRAINING_MANIFEST.json'),checkpoints=checkpoints,
                 ordered_ids=[r['id'] for r in result],before_reference_decode=True,reference_reads=0,
                 tiny_head_forward_calls=len(result)*len(models),new_backbone_SAM_or_GPU_forwards=0,new_training_updates=0,
                 additional_inference_scope='One extra frozen tiny-head pass solely to retain full log-softmax; no fitting or selection changes',
                 source_sha256=sha(__file__),cost=_cost(start,cpu))
    write(out/'DISTRIBUTION_SEAL.json',receipt);return receipt


def source_metric(pose,target):
    """Original numeric renderer matrices and per-asset Y-axis group, unchanged."""
    import numpy as np
    if not pose or not pose.get('available'):return dict(translation_cm=None,rotation_deg=None,success=False)
    R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    G=np.asarray(target['R_fixed'],float);g=np.asarray(target['t'],float);order=int(target['symmetry_order'])
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Invalid available synthetic prediction')
    if order not in (1,2,4):raise ValueError('Unknown original source symmetry')
    angles=[]
    for k in range(order):
        a=2*math.pi*k/order;c,s=math.cos(a),math.sin(a);Q=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        angles.append(math.degrees(math.acos(float(np.clip((np.trace((G@Q).T@R)-1)/2,-1,1)))))
    tc=float(np.linalg.norm(t-g)*100);rc=min(angles)
    return dict(translation_cm=tc,rotation_deg=rc,success=bool(tc<=5 and rc<=5))


def evaluate(run_dir,output=None,*,evaluation_GO=False):
    if evaluation_GO is not True:raise PermissionError('Explicit root postprediction evaluation GO required before IO')
    import numpy as np
    start,cpu=time.monotonic(),time.process_time();run=Path(run_dir);dist=run/'head_synthetic_distributions_v1'
    pred=run/'learned_synthetic_v1';feature_dir=run/'synthetic_features_v1';syn=run/'synthetic_v1';labels=run/'head_labels_v1'
    ds=read(dist/'DISTRIBUTION_SEAL.json');ps=read(pred/'PREDICTION_SEAL.json')
    if (ds.get('complete') is not True or ds.get('before_reference_decode') is not True or ds.get('reference_reads')!=0
            or sha(dist/'DISTRIBUTIONS.jsonl.gz')!=ds['distributions_sha256']
            or sha(pred/'PREDICTION_SEAL.json')!=ds['prediction_seal_sha256']
            or sha(pred/'PREDICTIONS.jsonl.gz')!=ds['predictions_sha256']):raise ValueError('Both prediction seals must pass before GT decode')
    fs=read(feature_dir/'FEATURE_SEAL.json');ls=read(labels/'LABEL_SEAL.json');provision=read(syn/'PROVISION_SEAL.json')
    feature_path=feature_dir/'HEAD_FEATURES.jsonl.gz';target_path=syn/'SYNTH_GT_LABELS.jsonl.gz';label_path=labels/'SYNTHETIC_HEAD_TARGETS.jsonl.gz'
    if (sha(feature_path)!=fs['head_features_sha256'] or sha(feature_path)!=ds['features_sha256']
            or sha(target_path)!=provision['artifacts'][target_path.name] or sha(target_path)!=ls['GT_labels_sha256']
            or sha(label_path)!=ls['labels_sha256'] or ls['features_sha256']!=ds['features_sha256']):raise ValueError('Original source/feature/label binding changed')
    # All target reads occur strictly below the two complete prediction seals.
    features={r['id']:r for r in rows(feature_path)};targets={r['id']:r for r in rows(target_path)}
    training_labels={r['id']:r for r in rows(label_path)};poses={r['id']:r for r in rows(pred/'PREDICTIONS.jsonl.gz')}
    values=list(rows(dist/'DISTRIBUTIONS.jsonl.gz'));lock=read(syn/'DATA_SPLIT_LOCK.json')
    expected=set(lock['expansion']['ids'])
    if any(set(x)!=expected for x in (features,targets,training_labels,poses)) or {r['id'] for r in values}!=expected or len(values)!=1003:
        raise ValueError('Exact prelocked1003 identity union required')
    scored=[]
    for r in values:
        fid=r['id'];frame=features[fid];target=targets[fid];label=training_labels[fid]
        if not(r['split']==frame['split']==target['split']==label['split']):raise ValueError('Asset split changed')
        cs=frame['candidates'];good=[i for i,c in enumerate(cs) if c['eligible'] and source_metric(c['actual_pose'],target)['success']]
        expected_target=np.zeros(len(cs)+1);expected_target[good if good else [-1]]=1/len(good) if good else 1.
        if good!=label['positive_candidate_indices'] or not np.array_equal(expected_target,np.asarray(label['target_distribution'])):
            raise ValueError('Independent original-symmetry candidate label disagreement')
        eligible=[c['candidate_id'] for c in cs if c['eligible']];goodids={cs[i]['candidate_id'] for i in good};arms={}
        for arm in ARMS:
            d=r['predictions'][arm];p=poses[fid]['predictions'][arm];logp=d['log_probabilities']
            if d['candidate_ids']!=eligible or d['none_index']!=len(eligible) or len(logp)!=len(eligible)+1:
                raise ValueError('Frozen softmax domain/eligibility changed')
            positive=[i for i,cid in enumerate(eligible) if cid in goodids] if goodids else [len(eligible)]
            ce=-statistics.fmean(logp[i] for i in positive);m=source_metric(p['actual_pose'],target)
            arms[arm]={**m,'none_selected':p['none_selected'],'cross_entropy':ce,
                       'selected_candidate_id':p['selected_candidate_id'],'fallback':p['fallback']}
        scored.append(dict(id=fid,split=r['split'],original_symmetry_order=target['symmetry_order'],
                           candidate_oracle_success=bool(good),eligible_good_count=len(good),eligible_count=len(eligible),
                           N0=source_metric(frame['fallback'],target),arms=arms))
    populations={}
    for split in ('train','validation'):
        subset=[r for r in scored if r['split']==split];n=len(subset);oracle=sum(r['candidate_oracle_success'] for r in subset)
        methods={}
        for arm in ARMS:
            methods[arm]=dict(denominator=n,successes=sum(r['arms'][arm]['success'] for r in subset),
                abstentions=sum(r['arms'][arm]['none_selected'] for r in subset),
                no_good_frames=n-oracle,oracle_good_frames=oracle,
                abstain_with_good=sum(r['candidate_oracle_success'] and r['arms'][arm]['none_selected'] for r in subset),
                abstain_with_no_good=sum(not r['candidate_oracle_success'] and r['arms'][arm]['none_selected'] for r in subset),
                candidate_selected_with_no_good=sum(not r['candidate_oracle_success'] and not r['arms'][arm]['none_selected'] for r in subset),
                success_after_abstain_N0=sum(r['arms'][arm]['none_selected'] and r['arms'][arm]['success'] for r in subset),
                translation_cm=statistics_of([r['arms'][arm]['translation_cm'] for r in subset]),
                rotation_deg=statistics_of([r['arms'][arm]['rotation_deg'] for r in subset]),
                CE=statistics_of([r['arms'][arm]['cross_entropy'] for r in subset]),
                good_CE=statistics_of([r['arms'][arm]['cross_entropy'] for r in subset if r['candidate_oracle_success']]),
                no_good_CE=statistics_of([r['arms'][arm]['cross_entropy'] for r in subset if not r['candidate_oracle_success']]))
        populations[split]=dict(denominator=n,original_symmetry_counts=dict(Counter(str(r['original_symmetry_order']) for r in subset)),
                                candidate_oracle_successes=oracle,no_good_frames=n-oracle,N0_successes=sum(r['N0']['success'] for r in subset),arms=methods)
    training=read(run/'head_training_v1/TRAINING_MANIFEST.json')
    loss_checks={}
    for r in training['runs']:
        arm=r['kind']+'_s'+str(r['seed']);actual=populations['validation']['arms'][arm]['CE']['mean']
        loss_checks[arm]=dict(training_recorded_best_validation_loss=r['best_validation_loss'],postprediction_validation_CE=actual,
                              absolute_difference=abs(actual-r['best_validation_loss']),
                              within_float32_frame_batch_roundoff_1e5=abs(actual-r['best_validation_loss'])<=1e-5)
    out=Path(output or run/'synthetic_head_evaluation_v1');out.mkdir(parents=True,exist_ok=False)
    save_rows(out/'SCORED_ROWS.jsonl.gz',scored)
    result=dict(schema='FIXED_HEAD_SYNTHETIC_POSTPREDICTION_EVALUATION_V1',complete=True,rows=len(scored),arms=list(ARMS),
                populations=populations,first_representative_seed=SEEDS[0],all_seeds_preserved=True,validation_loss_checks=loss_checks,
                source_symmetry='original per-asset C1/C2; no phase or matrix relabel',no_model_choice_or_fit=True,
                independent_test=False,validation_used_for_early_stopping=True,
                caveats=['Source asset holdout is new-head development, not independent backbone test',
                         'Train mixedC1/C2; validation C1 only; PRIMARY originalC2',
                         'Candidate oracle is a postprediction fixed-reference upper bound',
                         'Abstention falls back to original N0 and is not a safety guarantee'],
                input_bindings=dict(distribution_seal=sha(dist/'DISTRIBUTION_SEAL.json'),pose_prediction_seal=sha(pred/'PREDICTION_SEAL.json'),
                                    training_manifest=sha(run/'head_training_v1/TRAINING_MANIFEST.json'),GT=sha(target_path),labels=sha(label_path),
                                    label_seal=sha(labels/'LABEL_SEAL.json'),split_lock=sha(syn/'DATA_SPLIT_LOCK.json')),
                source_sha256=sha(__file__),cost=_cost(start,cpu))
    write(out/'SYNTHETIC_LEARNED_EVALUATION.json',result)
    lines=['# 고정 head의 합성 train/validation 결과','',
           '모든6개 고정 checkpoint를 그대로 사용했다. 추가 학습·seed 선택·GT 수정은 없으며 validation은 early stopping에 사용한 개발자료다.','']
    for split,pop in populations.items():
        lines += [f"## {split}: {pop['denominator']}장",'',
                  f"후보 oracle 공동성공 {pop['candidate_oracle_successes']}장, 좋은 eligible 후보 없음 {pop['no_good_frames']}장, 원 N0 성공 {pop['N0_successes']}장.",'',
                  '| arm | 공동성공 | 보류 | 전체CE | good CE | no-good CE |','|---|---:|---:|---:|---:|---:|']
        fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
        for arm in ARMS:
            a=pop['arms'][arm];lines.append(f"| {arm} | {a['successes']}/{pop['denominator']} | {a['abstentions']} | {fmt(a['CE']['mean'])} | {fmt(a['good_CE']['mean'])} | {fmt(a['no_good_CE']['mean'])} |")
        lines.append('')
    lines += ['대표 seed는20261018로 고정했으며 다른 seed도 모두 남긴다. 서로 다른 source 대칭·관측 지원 범위 때문에 이 수치만으로 실사 전이를 보장할 수 없다.']
    (out/'SYNTHETIC_LEARNED_EVALUATION_KO.md').write_text('\n'.join(lines)+'\n')
    write(out/'EVALUATION_SEAL.json',dict(schema='SYNTHETIC_HEAD_EVALUATION_SEAL_V1',complete=True,rows=len(scored),
          explicit_root_GO=True,input_bindings=result['input_bindings'],source_sha256=sha(__file__),
          result_sha256=sha(out/'SYNTHETIC_LEARNED_EVALUATION.json'),scored_rows_sha256=sha(out/'SCORED_ROWS.jsonl.gz'),
          markdown_sha256=sha(out/'SYNTHETIC_LEARNED_EVALUATION_KO.md'),
          artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},cost=result['cost']))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['distributions','evaluate']);p.add_argument('--run-dir',required=True)
    p.add_argument('--output');p.add_argument('--evaluation-GO',action='store_true');a=p.parse_args()
    r=distributions(a.run_dir,a.output) if a.stage=='distributions' else evaluate(a.run_dir,a.output,evaluation_GO=a.evaluation_GO)
    print(dict(complete=r['complete'],rows=r['rows']))


if __name__=='__main__':main()
