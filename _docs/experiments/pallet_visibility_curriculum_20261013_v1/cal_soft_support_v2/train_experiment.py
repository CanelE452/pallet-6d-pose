"""Explicit CAL-only legacy/V2 feature comparison, with no TEST label access."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import calibration as A
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

OUT=Path(__file__).resolve().parent
INPUT=C.SCRATCH/'cal_segmentation_v2'
LABELS=C.SCRATCH/'CAL_SCORING_METADATA.json'


@contextmanager
def cal_only_access():
    import builtins
    import io
    originals=builtins.open,io.open,Path.open
    accesses=[]

    def wrap(function):
        def guarded(path,*args,**kwargs):
            name=str(path)
            if any(part in name for part in ('/test_v1/','FROZEN_TEST_PROTOCOL','TEST_INPUTS',
                                             'ORACLE_CAL_INPUTS','TARGETS.json','/annotations/')):
                raise AssertionError('Forbidden input for this calibration comparison: '+name)
            if 'SCORING_METADATA' in name and Path(path).resolve()!=LABELS.resolve():
                raise AssertionError('Only the isolated 76-frame CAL labels may be read: '+name)
            if 'SCORING_METADATA' in name:
                accesses.append(str(Path(path).resolve()))
            return function(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:
        yield accesses
    finally:
        builtins.open,io.open,Path.open=originals


def main():
    prereg=C.read(OUT/'PREREGISTER.json')
    assert prereg['before_training'] and prereg['classification_C_grid']==[.01,.1,1,10]
    paths=[INPUT/'PREDICTIONS.jsonl.gz',INPUT/'PROTOCOL.json',LABELS,
           C.ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1/calibration.py',
           C.ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1/test_calibration_features.py',
           OUT/'code_before_soft_support_v2/calibration.py',C.DOC/'models/visibility_logistic_v1.json']
    hashes={str(p):C.sha(p) for p in paths}
    with cal_only_access() as accesses:
        labels=C.read(LABELS)['frames']
        assert len(labels)==76 and all(r['partition']=='CAL' for r in labels)
        rows=[r for r in C.rows(INPUT/'PREDICTIONS.jsonl.gz') if r['method']=='RGB_CURRICULUM']
        assert len(rows)==76 and {r['id'] for r in rows}=={r['id'] for r in labels}
        for version in ('legacy_v1','soft_support_v2'):
            A.train(INPUT,OUT/(version+'.json'),'RGB_CURRICULUM',version)
        old=C.read(OUT/'legacy_v1.json');new=C.read(OUT/'soft_support_v2.json')
        old_fit=C.read(OUT/'legacy_v1.training.json');new_fit=C.read(OUT/'soft_support_v2.training.json')
        frozen=C.read(C.DOC/'models/visibility_logistic_v1.json')
        summaries={}
        for version,model,fit in [('legacy_v1',old,old_fit),('soft_support_v2',new,new_fit)]:
            best=next(r for r in fit['candidates'] if r['C']==model['C'])
            summaries[version]=dict(points=model['training_points'],visible=model['visible_points'],
                external=model['external_points'],C=model['C'],threshold=model['visible_threshold'],
                pooled_LOGO_AUC=model['crossfit_AUC'],balanced_accuracy=model['crossfit_balanced_accuracy'],
                per_session_AUC=best['folds'],coefficients_soft_columns=model['coefficients'][4:6],
                feature_names_soft_columns=model['feature_scalars'][4:6],
                model_sha256=C.sha(OUT/(version+'.json')),
                candidate_AUCs=[{k:r[k] for k in ('C','AUC')} for r in fit['candidates']])
        Xlegacy=np.concatenate([A.features(r['appearance']) for r in rows])
        Xnew=np.concatenate([A.features(r['appearance'],'soft_support_v2') for r in rows])
        max_delta=0.
        for row in rows:
            raw=deepcopy(row['appearance'])
            result=A.predict(raw,frozen)
            explicit=A.predict(raw,dict(frozen,feature_version='legacy_v1'))
            assert result==explicit
            assert raw==row['appearance']
        old_vs_frozen={key:float(np.max(np.abs(np.asarray(old[key])-frozen[key])))
                       for key in ('mean','scale','coefficients','intercept')}
    assert hashes=={p:C.sha(p) for p in hashes}
    report=dict(schema='CAL_SOFT_SUPPORT_V2_COMPARE',CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        input_and_code_sha256=hashes,training_label_accesses=sorted(set(accesses)),
        legacy_default_features_and_old_model_predict_preserved=True,
        frozen_default_predict_vs_explicit_legacy_max_difference=0.,
        new_legacy_fit_vs_existing_frozen_model_differences=old_vs_frozen,
        all_RAW76_legacy_soft_feature_columns_zero=bool((Xlegacy[:,4:6]==0).all()),
        all_RAW76_V2_soft_feature_ranges=[Xnew[:,4:6].min(0).tolist(),Xnew[:,4:6].max(0).tolist()],
        nonzero_soft_feature_rows=int(np.any(Xnew[:,4:6]!=0,axis=1).sum()),
        summary=summaries,delta_pooled_LOGO_AUC=new['crossfit_AUC']-old['crossfit_AUC'],
        delta_balanced_accuracy=new['crossfit_balanced_accuracy']-old['crossfit_balanced_accuracy'],
        limits=['CAL-pooled OOF C/cutoff selection is exploratory and not nested validation',
                'This visibility target does not prove correct KP coordinates or ownership of same-colour pixels',
                'V2 raw RGB head must consume the raw RGB descriptor, not switched CNN binary feature meanings'],
        production_models_replaced=False,pose_stage_parameters_changed=False)
    C.write(OUT/'VERIFICATION.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
