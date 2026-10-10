"""Public-only independent 222-candidate arithmetic and ablation audit.

Uses math.fsum/exp/log for posterior moments and NumPy for 2x2 eigensystems.
Does not import capture, fusion, evaluator, models, OpenCV or private inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def independent_moments(logits,network,gain,T):
    d=np.asarray(network,dtype=np.float64)/float(gain)
    probabilities=[];means=[];covariances=[];null=[];entropy=[];sums=[]
    for raw in logits:
        zz=[float(v)/float(T) for v in raw];maximum=max(zz)
        e=[math.exp(v-maximum) for v in zz];denominator=math.fsum(e)
        p=[v/denominator for v in e]
        m=[math.fsum(p[j]*float(d[j,k]) for j in range(222)) for k in range(2)]
        C=[[math.fsum(p[j]*(float(d[j,a])-m[a])*(float(d[j,b])-m[b]) for j in range(222))
            for b in range(2)] for a in range(2)]
        probabilities.append(p);means.append(m);covariances.append(C)
        null.append(p[-1]);sums.append(math.fsum(p))
        entropy.append(-math.fsum(v*math.log(v) for v in p if v>0))
    return dict(mean=np.asarray(means),C=np.asarray(covariances),null=np.asarray(null),
                entropy=np.asarray(entropy),sum=np.asarray(sums))


def run(docs):
    lock=read(docs/'INPUT_LOCK.json')
    parity=read(docs/'POSTERIOR_PARITY.json')
    seal=read(docs/'COORDINATES_SEAL.json')
    assert lock['status']==parity['status']==seal['status']=='PASS'
    assert sha(docs/'INPUT_LOCK.json')==parity['input_lock_sha256']==seal['input_lock_sha256']
    assert sha(docs/'FUSION_METHOD_LOCK.json')==parity['fusion_method_lock_sha256']==seal['fusion_method_lock_sha256']
    assert sha(docs/'POSTERIOR_CAPTURE.jsonl.gz')==parity['capture']['sha256']
    assert sha(docs/'NEW_COORDINATES_SEALED.jsonl.gz')==seal['coordinates_sha256']
    posterior=rows(docs/'POSTERIOR_CAPTURE.jsonl.gz')
    coordinates=rows(docs/'NEW_COORDINATES_SEALED.jsonl.gz')
    ids=lock['population']['frame_ids']
    assert len(posterior)==957 and len(coordinates)==1914
    original_buffer=np.asarray(parity['candidate_buffer'],dtype=np.float32)
    assert original_buffer.shape==(222,2) and np.array_equal(original_buffer[-1],np.zeros(2))
    assert np.all(np.linalg.norm(original_buffer[:-1],axis=-1)>0)
    # The original buffer is angle-major, 17 increasing radii per each of13 angles.
    radii=np.linalg.norm(original_buffer[:-1].astype(np.float64),axis=-1).reshape(13,17)
    assert np.all(np.diff(radii,axis=-1)>0)
    assert np.allclose(radii,np.arange(1,18)[None]*.08/17,rtol=0,atol=1e-8)
    coord={(r['seed'],r['method'],r['id']):r for r in coordinates}
    assert len(coord)==1914
    methods=['FG_JOINT_POSTERIOR','JOINT_FIXED_ISOTROPIC']
    maxima=Counter();counts=Counter();capture_order=[]
    def compare(name,a,b,tolerance=1e-8):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
        error=float(np.max(np.abs(a-b),initial=0))
        maxima[name]=max(maxima[name],error)
        assert error<=tolerance,(name,error,tolerance)
        counts[name]+=1
    for r in posterior:
        seed,fid=r['seed'],r['id'];capture_order.append((seed,fid))
        network=np.asarray(r['candidate_displacements_network'],dtype=np.float32)
        logits=np.asarray(r['logits'],dtype=np.float64)
        assert network.shape==(222,2) and logits.shape==(8,222)
        assert np.array_equal(network[-1],np.zeros(2))
        # FP32 multiplication is the frozen head's documented candidate arithmetic.
        expected_network=original_buffer*np.float32(r['network_box_diagonal'])
        compare('original_buffer_candidate_order_and_scaling',network,expected_network,tolerance=0)
        assert r['gain']>0 and np.isfinite(r['gain']) and r['temperature']>0
        compare('forward_native_qN_parity',r['forward_qN'],r['qN'],tolerance=1e-3)
        assert r['GT_inference_inputs'] is False
        independent=independent_moments(logits,network,r['gain'],r['temperature'])
        compare('capture_mean',independent['mean'],r['candidate_native_mean'])
        compare('capture_C',independent['C'],r['candidate_native_covariance'])
        compare('capture_null',independent['null'],r['posterior_null_probability'])
        compare('capture_entropy',independent['entropy'],r['posterior_entropy'])
        compare('capture_probability_sum',independent['sum'],r['posterior_probability_sum'],tolerance=1e-12)
        compare('all222_probability_sum_to_one',independent['sum'],np.ones(8),tolerance=1e-12)
        cap=.01*math.hypot(*r['raw_hw'])
        pair=[coord[(seed,m,fid)] for m in methods]
        for c in pair:
            assert c['qN']==r['qN'] and c['q0']==r['q0'] and c['prediction_support']==r['prediction_support']
            assert c['raw_hw']==r['raw_hw']
            record=c['correction']['diagnostics']['corner_records']
            assert len(record)==8 and [x['corner'] for x in record]==list(range(8))
            for k,v in enumerate(record):
                if 'C' not in v:
                    assert v['status'] in ('unsupported_prediction','nonfinite_initial','missing_initial_sentinel')
                    continue
                C=independent['C'][k]
                compare('record_C_from_all222',C,v['C'])
                compare('record_mean_from_all222',independent['mean'][k],v['candidate_mean_native_px'])
                compare('record_null',independent['null'][k],v['null_probability'])
                compare('record_entropy',independent['entropy'][k],v['posterior_entropy'])
                compare('record_probability_sum',independent['sum'][k],v['probability_sum'],tolerance=1e-12)
                assert v['candidate_count']==222 and v['last_null_included'] is True
                C=(C+C.T)/2
                eig=np.linalg.eigvalsh(C)
                compare('record_C_spectrum',eig,v['C_eigenvalues_px2'])
                # Diagonalize C+I independently, then clip its eigenvalues.
                stabilized_values,directions=np.linalg.eigh(C+np.eye(2))
                limited=np.clip(stabilized_values,1.,cap*cap)
                Sigma=directions@np.diag(limited)@directions.T
                compare('record_posterior_Sigma',Sigma,v['Sigma_posterior'])
                compare('record_posterior_Sigma_spectrum',limited,v['Sigma_posterior_eigenvalues_px2'])
                expected_Sigma=Sigma if c['method']=='FG_JOINT_POSTERIOR' else np.eye(2)*16
                compare('actual_prior_only_ablation_Sigma',expected_Sigma,v['Sigma'])
                compare('actual_prior_spectrum',np.linalg.eigvalsh(expected_Sigma),v['Sigma_eigenvalues_px2'])
        a,b=[c['correction']['diagnostics']['corner_records'] for c in pair]
        for x,y in zip(a,b):
            for key in ('prediction_support','point_support','candidate_count','last_null_included','fixed_window_center','C','C_eigenvalues_px2',
                        'candidate_mean_native_px','null_probability','posterior_entropy','probability_sum','Sigma_posterior',
                        'Sigma_posterior_eigenvalues_px2','A','b','S','A_eigenvalues','posterior_mean_is_not_prior_center'):
                assert (key in x)==(key in y),(seed,fid,x['corner'],key)
                if key in x:
                    if isinstance(x[key],bool):assert x[key]==y[key]
                    else:compare('paired_arm_identical_'+key,x[key],y[key],tolerance=0)
            counts['same_window_image_equation_pair']+=1
    assert capture_order==[(s,i) for s in (1,2,3) for i in ids]
    for method in methods:
        assert {k for k in coord if k[1]==method}=={(s,method,i) for s in (1,2,3) for i in ids}
    return dict(schema='independent_public_posterior_numeric_verification_v1',status='PASS',
        implementation='Python math.fsum/exp/log computes all222 softmax and moments; independent NumPy C+I eigensystem and clipping. No fusion/capture/model/evaluator imports.',
        posterior_rows=957,new_coordinate_rows=1914,candidate_entries=957*8*222,corner_pairs=957*8,
        maximum_absolute_difference=dict(maxima),checks_count=dict(counts),
        tolerances=dict(moment_and_covariance_absolute=1e-8,probability_sum_absolute=1e-12,
                        forward_qN_absolute_px=1e-3,original_buffer_candidate_order_scaling_absolute=0,paired_ablation_A_b_S_absolute=0),
        checks=dict(all222_including_null=True,original_buffer_order_and_FP32_scaling=True,gain_applied_native_covariance=True,
                    all957_forward_qN_parity=True,all1914_corner_covariance_Sigma_verified=True,
                    same_mu_q0_support_window_image_A_b_S=True,isotropic_only_prior_Sigma_substitution=True),
        source_bindings=[dict(path=f,sha256=sha(docs/f)) for f in ('INPUT_LOCK.json','FUSION_METHOD_LOCK.json','POSTERIOR_PARITY.json',
                         'POSTERIOR_CAPTURE.jsonl.gz','COORDINATES_SEAL.json','NEW_COORDINATES_SEALED.jsonl.gz')],
        verifier=dict(path='scripts/research/pallet_feature_gradient_joint_20261010/verify_posterior.py',sha256=sha(Path(__file__))),
        execution=dict(model_calls=0,F_calls=0,optimizer_updates=0,raw_private_inputs_read=0,GT_or_DEV_metric_reads=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--docs',type=Path,default=Path(__file__).resolve().parents[3]/'_docs/experiments/pallet_feature_gradient_joint_20261010')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=run(a.docs)
    output=a.output or a.docs/'POSTERIOR_NUMERIC_VERIFICATION.json'
    with output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],posterior_rows=result['posterior_rows'],new_coordinate_rows=result['new_coordinate_rows'],
                         maximum_absolute_difference=result['maximum_absolute_difference']),ensure_ascii=False))


if __name__=='__main__':main()
