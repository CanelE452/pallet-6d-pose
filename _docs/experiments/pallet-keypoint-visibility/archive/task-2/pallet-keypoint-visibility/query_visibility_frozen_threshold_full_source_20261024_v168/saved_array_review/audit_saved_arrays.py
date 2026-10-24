"""V168 saved-array verification only; no producer import, labels, features or models."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
COMPLETE=V/'retrospective_full_source_v1/COMPLETE.json'
INPUTS=V/'readout_preparation/FIXED_OUTPUT_HEAD_SCALAR_AND_LABEL_BINDINGS.json'
EXPECTED_COMPLETE='a2d73ed139e8f7ad89dd11b56816fe5553dcdf740831a9208b90f98df9e88ac0'
FAMILIES={'OLD6144':6144,'NEW20480':20480,'PALLOX512':512,'MR1840':1840,'PART1144':1144}
ADDED={'retrospective_primary_visibility_prediction','retrospective_secondary_entity_gated_visibility','frozen_PUBLIC_CAL_E_logit_threshold_float64'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(x in str(p).lower() for x in ('current223','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'],str(p)
    return p
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def run():
    start=time.monotonic();assert sha(COMPLETE)==EXPECTED_COMPLETE
    complete=read(COMPLETE);assert complete['PASS'] and complete['completed']
    entry=read(checked(complete['entry']));assert entry['approved_actual_saved_full_source_readout'] and entry['source_only']
    assert complete['inputs']==binding(INPUTS)
    inputs=read(INPUTS);bindings=[binding(COMPLETE),binding(INPUTS),complete['entry'],complete['code'],complete['protocol']]
    for b in bindings:checked(b)
    receipts=[];all_fields=0;row_count=0;panel_keys=None
    for model,result in zip(inputs['models'],complete['results']):
        assert model['arm']==result['arm'];threshold=np.float64(model['threshold_float64'])
        report=read(checked(result['report']));oldreport=read(checked(model['source_results']))
        bindings.extend([result['report'],model['source_results'],result['panels']])
        assert report['threshold_float64']==float(threshold)
        pairs=[];u_original=0;u_part=0
        for family,n in FAMILIES.items():
            ob=model['outputs'][family];nb=result['outputs'][family]
            old=load(checked(ob));new=load(checked(nb));bindings.extend([ob,nb])
            assert 'features' not in old and set(new)==set(old)|ADDED
            assert all(exact(a,new[k]) for k,a in old.items()),(model['arm'],family,'changed original field')
            score=old['raw_three_logits'][:,0];available=old['query_available']
            assert score.dtype==np.float32 and len(score)==n and available.dtype==np.bool_
            assert old['raw_three_logits'].shape==(n,3)
            t=new['frozen_PUBLIC_CAL_E_logit_threshold_float64'];assert t.dtype==np.float64 and t.shape==() and t==threshold
            pred=np.full(n,-1,np.int8);pred[available]=(score[available].astype(np.float64)>=threshold).astype(np.int8)
            assert exact(pred,new['retrospective_primary_visibility_prediction'])
            secondary=np.full(n,-1,np.int8);keep=available&(old['raw_three_logits'][:,1]>=0);secondary[keep]=pred[keep]
            assert exact(secondary,new['retrospective_secondary_entity_gated_visibility'])
            assert np.array_equal(pred==-1,~available)
            fixed0=np.full(n,-1,np.int8);fixed0[available]=(score[available]>=0).astype(np.int8)
            assert exact(fixed0,old['primary_visibility_prediction'])
            u=int((~available).sum())
            if family=='PART1144':
                u_part=u
                for k in ('native_visibility_target','point_quality_target','owner_target'):assert np.all(new[k]==-1)
            else:u_original+=u
            assert new['native_slot'].shape==(n,) and np.all((new['native_slot']>=0)&(new['native_slot']<8))
            all_fields+=len(old);row_count+=n
            pairs.append(dict(family=family,N=n,original_fields_byte_identical=len(old),inputU=u,direct_float64_prediction_exact=True))
        assert (u_original,u_part)==(535,178)
        gates=report['native']['gates'];assert len(gates)==35
        for name,g in gates.items():
            assert g['PASS']==bool(g['value'] is not None and g['value']>=g['limit']),name
            assert g['limit']==oldreport['native']['gates'][name]['limit'] and g['N']==oldreport['native']['gates'][name]['N']
        passed=sum(g['PASS'] for g in gates.values())
        assert passed==report['native']['native35_passed']==result['native35_passed']==18
        assert not report['native']['native35_all_pass'] and not report['existing_source_gates_all_pass']
        assert report['synthetic_point_quality']==oldreport['synthetic_point_quality']
        assert report['selected_model'] is None and not report['adopted'] and not report['fully_solved']
        tables=read(checked(result['panels']))['panels'];keys={(r['family'],r['original_or_reporting_split'],r['group']) for r in tables}
        assert len(keys)==len(tables)==249
        if panel_keys is not None:assert keys==panel_keys
        panel_keys=keys
        for family in FAMILIES:
            folds={r['original_or_reporting_split'] for r in tables if r['family']==family}
            expected={'OLD6144':{'ALL','TRAIN','VALIDATION'},'NEW20480':{'ALL','TRAIN','VALIDATION'},'PALLOX512':{'ALL','TRAIN_LAB','VAL_HALL'},'MR1840':{'ALL','SOURCE_DEVELOPMENT','PUBLIC_CAL_APPARENT'},'PART1144':{'ALL','SOURCE_TRAIN_WEAK_RESPONSE'}}[family]
            assert folds==expected
            for fold in folds:
                assert (family,fold,'ALL') in keys
                assert all((family,fold,f'native::{k}') in keys for k in range(8))
        for family,rkey in [('OLD6144','old_native'),('NEW20480','new_native')]:
            for fold in ('ALL','TRAIN','VALIDATION'):
                for group in oldreport['native'][rkey]:
                    prefix=fold+'::'
                    if group.startswith(prefix) and group[len(prefix):].split('::')[0] in ('material','aspect','foreign_family','geometry_family'):
                        assert (family,fold,group[len(prefix):]) in keys
        # Gate summary checked directly from saved counts, without redesigning the report helper.
        for rkey,gate in [('old_native','NATIVE'),('new_native','NEW')]:
            cm=report['native'][rkey]['VALIDATION']['CM_D_E_rows_D_E_U_columns']
            for row,label in [(0,'D'),(1,'E')]:
                gkey=gate+'::VALIDATION::'+label
                if gkey in gates:assert gates[gkey]['value']==cm[row][row]/sum(cm[row])
        receipts.append(dict(arm=model['arm'],threshold_float64=float(threshold),pairs=pairs,native35_passed=passed,panel_count=len(tables),all8_native_panels_preserved=True,old_new_factor_panels_complete=True,source_qualification=False))
    original_status=checked(inputs['PART_statuses']);saved_status=checked(complete['all198_statuses']);bindings.extend([inputs['PART_statuses'],complete['all198_statuses']])
    statuses=[json.loads(s) for s in saved_status.read_text().splitlines() if s]
    original=[json.loads(s) for s in original_status.read_text().splitlines() if s]
    assert statuses==original and len(statuses)==198
    missing=[s for s in statuses if not s['constructed_payload']];assert len(missing)==55
    assert all(s['feature_index'] is None and s['feature_indices'] is None and s['payload'] is None for s in missing)
    assert not complete['threshold_head_or_model_selection'] and not complete['independent_qualification_claimed']
    for b in bindings:checked(b)
    receipt=dict(completed=True,PASS=True,review_kind='root/author jointly reviewed producer plus saved-array verification; not a fresh independent performance study',code=binding(__file__),bindings=bindings,models=receipts,actual_retrospective_NPZ_reloaded=10,original_V166_NPZ_reloaded=10,total_paired_rows=row_count,original_fields_byte_compared=all_fields,original_U_each=535,PART_U_each=178,status_rows=198,missing55_null_preserved=True,original0_full_report_exactfirst_and_V167_MR_CAL_exact='producer receipt verified; not independently reimplemented in this bounded saved-array audit',no_new_threshold_search_or_model_forward_or_fit=True,RGB_features_weights_private_current_V112_access=0,seconds=time.monotonic()-start)
    with (H/'REVIEW.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'PASS':True,'review':binding(H/'REVIEW.json'),'models':[{k:r[k] for k in ('arm','native35_passed','panel_count')} for r in receipts]}),flush=True)
if __name__=='__main__':run()
