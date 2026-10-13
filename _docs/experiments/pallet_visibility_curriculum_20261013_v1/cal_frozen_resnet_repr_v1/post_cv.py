"""Post-feature-seal uncertainty and frozen all-CAL/session-CV predictions."""
import collections
from pathlib import Path

import numpy as np

from experiment import read, write, sha, HERE


def uncertainty():
    data=read(HERE/'OOF_PREDICTIONS.json'); result=read(HERE/'RESULTS.json')
    families=sorted({r['family'] for r in data})
    reference=[r for r in data if r['family']=='RGB26_BASELINE']
    keys=[(r['id'],r['corner']) for r in reference];y=np.asarray([r['label'] for r in reference],bool)
    sessions=sorted({r['session'] for r in reference});groups=np.asarray([sessions.index(r['session']) for r in reference])
    values={f:{(r['id'],r['corner']):r['OOF_probability'] for r in data if r['family']==f} for f in families}
    probabilities={f:np.asarray([values[f][k] for k in keys]) for f in families}
    rng=np.random.default_rng(20261013);draws=rng.integers(0,len(sessions),(10000,len(sessions)))
    multiplicity=np.stack([(draws==i).sum(1) for i in range(len(sessions))],axis=1)
    positive=np.asarray([sum(y&(groups==i)) for i in range(len(sessions))]);negative=np.asarray([sum(~y&(groups==i)) for i in range(len(sessions))])
    denominator=(multiplicity@positive)*(multiplicity@negative);valid=denominator>0
    aucs={};recalls={}
    for family,p in probabilities.items():
        pair=np.zeros((len(sessions),len(sessions)))
        for i in range(len(sessions)):
            pi=p[y&(groups==i)]
            for j in range(len(sessions)):
                pn=p[~y&(groups==j)]
                pair[i,j]=((pi[:,None]>pn[None,:])+.5*(pi[:,None]==pn[None,:])).sum()
        aucs[family]=np.einsum('bi,ij,bj->b',multiplicity,pair,multiplicity)[valid]/denominator[valid]
        threshold=result['summary'][family]['OOF']['threshold']
        tp=np.asarray([sum(y&(groups==i)&(p>=threshold)) for i in range(len(sessions))])
        tn=np.asarray([sum(~y&(groups==i)&(p<threshold)) for i in range(len(sessions))])
        recalls[family]=(multiplicity[valid]@tp/(multiplicity[valid]@positive),multiplicity[valid]@tn/(multiplicity[valid]@negative))
    comparisons={}
    for family in families:
        delta=aucs[family]-aucs['RGB26_BASELINE']
        direct=recalls[family][0]-recalls['RGB26_BASELINE'][0]
        external=recalls[family][1]-recalls['RGB26_BASELINE'][1]
        comparisons[family]=dict(AUC_mean_difference=result['summary'][family]['OOF']['AUC']-result['summary']['RGB26_BASELINE']['OOF']['AUC'],
            AUC_CI95=np.quantile(delta,[.025,.975]).tolist(),direct_recall_delta_CI95=np.quantile(direct,[.025,.975]).tolist(),
            external_recall_delta_CI95=np.quantile(external,[.025,.975]).tolist())
    curves={}
    from sklearn.metrics import confusion_matrix
    for family,p in probabilities.items():
        curves[family]=[]
        for t in read(HERE/'PROTOCOL.json')['threshold_grid']:
            pred=p>=t;tp=int((y&pred).sum());tn=int((~y&~pred).sum())
            outside=groups==sessions.index('eval_outside')
            curves[family].append(dict(threshold=t,direct_recall=tp/int(y.sum()),external_recall=tn/int((~y).sum()),
                outside_external_recall=int((outside&~y&~pred).sum())/int((outside&~y).sum())))
    write(HERE/'OOF_UNCERTAINTY.json',dict(sessions=sessions,bootstrap_repeats=10000,
        valid_two_class_bootstraps=int(valid.sum()),comparisons=comparisons,fixed_threshold_curves=curves,
        caveat='CAL C/threshold/model-family selection is reused; bootstrap does not remove selection optimism'))
    print('PAIRED_REPR_BOOTSTRAP',comparisons,flush=True)


def export_predictions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    p=read(HERE/'PROTOCOL.json');seal=read(HERE/'FEATURE_SEAL.json')
    assert seal['features_sha256']==sha(HERE/'FEATURES.npz')
    keys=read(HERE/'FEATURE_KEYS.json');array=np.load(HERE/'FEATURES.npz');rgb=array['RGB26']
    oracle=read(HERE.parent/'ORACLE_CAL_INPUTS.json')['frames'];assert all(f['partition']=='CAL' for f in oracle)
    oracle={f['id']:f for f in oracle}
    selected=[];labels=[]
    for i,key in enumerate(keys):
        state=oracle[key['id']]['scoring_metadata']['human_states_native'][key['corner']]
        if state in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
            selected.append(i);labels.append(state=='DIRECT_VISIBLE')
    selected=np.asarray(selected);y=np.asarray(labels,bool);groups=np.asarray([r['session'] for r in keys])
    oof=read(HERE/'OOF_PREDICTIONS.json');expected={(r['family'],r['id'],r['corner']):r['OOF_probability'] for r in oof}
    summaries=read(HERE/'RESULTS.json')['summary'];models=HERE/'CV_MODELS';models.mkdir(exist_ok=False)
    outputs=[];bindings=[];parity_maximum={}
    for family in ('RGB26_PLUS_F4_PCA32','RGB26_PLUS_F4F8_PCA32'):
        frozen=array['F4'] if family=='RGB26_PLUS_F4_PCA32' else np.column_stack([array['F4'],array['F8']])
        full=read(HERE/f'{family}_MODEL.json');rep=full['representation']
        standardized=(frozen-np.asarray(rep['backbone_mean']))/np.asarray(rep['backbone_scale'])
        reduced=(standardized-np.asarray(rep['PCA_mean']))@np.asarray(rep['PCA_components']).T
        x=np.column_stack([rgb,reduced]);z=(x-np.asarray(full['mean']))/np.asarray(full['scale'])
        final_p=1/(1+np.exp(-np.clip(z@np.asarray(full['coefficients'])+full['intercept'],-60,60)))
        cv_p=np.zeros(len(keys))
        for heldout in sorted(set(groups)):
            train=selected[groups[selected]!=heldout];test=np.flatnonzero(groups==heldout)
            train_y=np.asarray([oracle[keys[i]['id']]['scoring_metadata']['human_states_native'][keys[i]['corner']]=='DIRECT_VISIBLE' for i in train])
            bs=StandardScaler().fit(frozen[train]);pca=PCA(n_components=32,svd_solver='full',whiten=False).fit(bs.transform(frozen[train]))
            train_x=np.column_stack([rgb[train],pca.transform(bs.transform(frozen[train]))]);test_x=np.column_stack([rgb[test],pca.transform(bs.transform(frozen[test]))])
            scale=StandardScaler().fit(train_x)
            head=LogisticRegression(C=summaries[family]['C'],class_weight='balanced',max_iter=3000,random_state=20261013).fit(scale.transform(train_x),train_y)
            cv_p[test]=head.predict_proba(scale.transform(test_x))[:,1]
            model=dict(schema='CAL_FROZEN_REPR_LOGO_MODEL_V1',family=family,heldout_session=heldout,
                train_sessions=sorted(set(groups[train])),training_points=len(train),C=summaries[family]['C'],
                backbone_mean=bs.mean_.tolist(),backbone_scale=bs.scale_.tolist(),PCA_mean=pca.mean_.tolist(),PCA_components=pca.components_.tolist(),
                mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coefficients=head.coef_[0].tolist(),intercept=float(head.intercept_[0]))
            path=models/f'{family}_{heldout}.json';write(path,model);bindings.append(dict(path=str(path),sha256=sha(path)))
        differences=[]
        for i in selected:
            key=keys[i];reference=expected[family,key['id'],key['corner']]
            # Float32 PCA GEMM can differ slightly when the prediction batch
            # includes the unlabelled eligible points. Threshold decisions
            # must remain exact; probability tolerance is separately recorded.
            np.testing.assert_allclose(cv_p[i],reference,atol=1e-6,rtol=0)
            assert (cv_p[i]>=full['threshold']) == (reference>=full['threshold'])
            differences.append(abs(cv_p[i]-reference))
        parity_maximum[family]=max(differences)
        for key,full_probability,cv_probability in zip(keys,final_p,cv_p):
            outputs.append(dict(**key,family=family,threshold=full['threshold'],
                full_CAL_probability=float(full_probability),heldout_session_probability=float(cv_probability)))
        bindings.append(dict(path=str(HERE/f'{family}_MODEL.json'),sha256=sha(HERE/f'{family}_MODEL.json')))
    write(HERE/'POSE_HEAD_PROBABILITIES.json',outputs)
    write(HERE/'POSE_HEAD_SEAL.json',dict(complete=True,before_pose_inference=True,before_pose_scoring=True,
        probabilities_sha256=sha(HERE/'POSE_HEAD_PROBABILITIES.json'),model_bindings=bindings,
        labeled_OOF_probability_maximum_abs_difference=parity_maximum,
        labeled_OOF_probability_parity_atol=1e-6,labeled_OOF_threshold_decision_parity=True,
        all_485_eligible_points_scored=True,
        feature_seal_sha256=sha(HERE/'FEATURE_SEAL.json'),TEST_accessed=False,FINAL_accessed=False))
    print('POSE_HEADS_FROZEN',len(outputs),'records',flush=True)


if __name__=='__main__':
    uncertainty();export_predictions()
