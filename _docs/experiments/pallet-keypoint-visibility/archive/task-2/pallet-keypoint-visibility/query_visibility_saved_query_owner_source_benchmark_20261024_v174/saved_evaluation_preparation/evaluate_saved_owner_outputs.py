"""V174 saved source outputs only. --prepare uses invented arrays and JSON metadata.

Real saved score/soft-truth bodies require the separately frozen root --entry.
No project producer is imported; no model, feature, pixel or training operation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
SCOPE = V / 'ROOT_SCOPE_BEFORE_SAVED_SOURCE_OWNER_SCORES_AND_NEW_CENSUS_RESULTS.json'
SCOPE_SHA = '093f02342fac858932eca0bea8ad14c5f6fed90f362a843e066aee3bd85c2f97'
READY = V / 'INPUT_PATH_READINESS.json'
CENSUS = P / 'query_visibility_full_source_query_causal_census_20261024_v173/literal_raster_census_v1/COMPLETE.json'
CENSUS_SHA = '025ea05d354404046bcaf92cef0af29653c77defd7e0a4c9faa7d9115fd29a96'
PROTOCOL = H / 'PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json'
CHECKS = H / 'GENERATED_INVENTED_CHECKS.json'
OUT = V / 'saved_owner_source_benchmark_v1'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
OWNER6 = ['INCIDENT_TARGET_DECLARED_CF', 'OTHER_TARGET', 'FOREIGN', 'BG', 'AMBIGUOUS', 'INVALID_SUPPORT']
OWNER3 = ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED']
META = ['original_global_index', 'original_occurrence_id', 'scene_index', 'base_index',
        'native_slot', 'query_xy', 'L', 'split', 'variant', 'material', 'aspect',
        'geometry_family', 'foreign_family', 'signed_quarter', 'original_semantic_state',
        'query_xy_missing', 'L_missing', 'selected_index_missing', 'query_available',
        'factual_query_available', 'feature_constructed', 'crop_constructed',
        'context_available', 'Q_inside', 'RGB_central4_available',
        'native_continuous_valid', 'native_nearest_valid']
MODELS = ['C45', 'C46', 'v61']


def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def check(binding):
    q = Path(binding['path'])
    require(q.is_relative_to(P), 'Bound source outside project')
    require(not any(x in str(q).lower() for x in ['_v112', '_v169', 'current223']), 'Forbidden binding')
    require(sha(q) == binding['sha256'], 'Binding SHA mismatch: ' + str(q))
    return q


def owner_predictions(logits, available):
    require(logits.shape == (len(available), 3), 'Owner shape')
    pred = np.full(len(available), -1, np.int8)
    finite = np.isfinite(logits).all(1)
    usable = available & finite
    pred[usable] = np.argmax(logits[usable], axis=1).astype(np.int8)
    return pred, finite


def presence_predictions(raw, available):
    require(raw.shape == available.shape, 'v61 shape')
    pred = np.full(len(raw), -1, np.int8)
    finite = np.isfinite(raw)
    usable = available & finite
    # Original v61 visible threshold0: D/visible -> TARGET0; E -> NON_TARGET1.
    pred[usable] = (raw[usable] < 0).astype(np.int8)
    return pred, finite


def truth_mass(owner6, missing):
    require(owner6.shape == (len(missing), 6), 'Exact point owner6 shape')
    require(np.isnan(owner6[missing]).all(), 'Missing q must retain allNaN source truth')
    w = owner6[~missing]
    require(np.isfinite(w).all() and (w >= 0).all(), 'Nonmissing weights')
    require(np.allclose(w.sum(1), 1, rtol=0, atol=1e-12), 'Source owner unit mass')
    known = np.zeros((len(missing), 3), np.float64)
    known[~missing] = np.column_stack((w[:, 0] + w[:, 1], w[:, 2], w[:, 3]))
    ambiguity = np.zeros(len(missing), np.float64)
    invalid = ambiguity.copy()
    ambiguity[~missing] = w[:, 4]
    invalid[~missing] = w[:, 5]
    # Missing q gets inventory-only unknown1; never a fabricated raster class.
    return known, ambiguity, invalid, missing.astype(np.float64)


def metrics(known, ambiguity, invalid, missing_mass, prediction, mask, classes):
    k = known[mask]
    z = prediction[mask]
    require(((z >= -1) & (z < classes)).all(), 'Prediction category')
    cm = np.zeros((classes, classes + 1), np.float64)
    for j in range(classes + 1):
        cm[:, j] = k[z == (j if j < classes else -1)].sum(0)
    denom = cm.sum(1)
    pp = cm[:, :classes].sum(0)
    recall = [float(cm[i, i] / denom[i]) if denom[i] else None for i in range(classes)]
    precision = [float(cm[i, i] / pp[i]) if pp[i] else None for i in range(classes)]
    unknown = ambiguity + invalid + missing_mass
    u_by_pred = [float(unknown[mask][z == (j if j < classes else -1)].sum()) for j in range(classes + 1)]
    require(abs(float(cm.sum() + unknown[mask].sum()) - int(mask.sum())) < 1e-8, 'Full inventory mass conservation')
    return dict(rows=int(mask.sum()), known_mass=float(cm.sum()),
                expected_soft_CM=cm.tolist(), truth_known_mass=denom.tolist(),
                expected_precision_on_known_truth_mass=precision,
                expected_recall_including_predictionU=recall,
                prediction_counts=[int((z == (j if j < classes else -1)).sum()) for j in range(classes + 1)],
                predictionU_rows=int((z < 0).sum()),
                ambiguous_mass=float(ambiguity[mask].sum()), invalid_mass=float(invalid[mask].sum()),
                missing_q_inventory_unknown_rows=int(missing_mass[mask].sum()),
                unknown_mass_by_prediction=u_by_pred)


def same_rows(actual, expected):
    require(actual.shape == expected.shape, 'Row metadata shape')
    if np.issubdtype(expected.dtype, np.number):
        eq = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    else:
        eq = actual.astype(str) == expected.astype(str)
    return eq.all(1) if eq.ndim == 2 else eq


def invented():
    tests = []
    def passed(name, ok):
        require(ok, name); tests.append(dict(name=name, PASS=True))
    logits = np.array([[2, 2, 1], [0, 1, 1], [1, 1, 1], [np.nan]*3], np.float32)
    z, finite = owner_predictions(logits, np.array([1, 1, 0, 0], bool))
    passed('Owner first-index tie T/F; inputU before argmax/NaN', z.tolist() == [0, 1, -1, -1])
    v, _ = presence_predictions(np.array([0., -.1, 1., np.nan]), np.array([1, 1, 0, 0], bool))
    passed('v61 threshold0 direction and equality, originalU first', v.tolist() == [0, 1, -1, -1])
    w = np.array([[.2,.3,.1,.25,.1,.05], [0,0,.5,.5,0,0], [np.nan]*6], np.float64)
    k,a,i,m = truth_mass(w, np.array([0,0,1], bool))
    passed('Incident plus OTHER target merge without cutoff', np.allclose(k[0], [.5,.1,.25], atol=1e-15))
    passed('Ambiguity, invalid and missing inventory unknown separate', a[0] == .1 and i[0] == .05 and m.tolist() == [0,0,1])
    r = metrics(k,a,i,m,np.array([0,-1,2]),np.ones(3,bool),3)
    passed('Expected confusion fractions and U truth-denominator mass', np.allclose(r['expected_soft_CM'], [[.5,0,0,0],[.1,0,0,.5],[.25,0,0,.5]]))
    passed('Precision/recall use known mass, U included in recall', r['expected_recall_including_predictionU'][0] == 1 and abs(r['expected_precision_on_known_truth_mass'][0] - .5/.85) < 1e-15)
    passed('Unknown mass not dropped', abs(sum(r['unknown_mass_by_prediction']) - 1.15) < 1e-15)
    kb = np.column_stack((k[:,0], k[:,1:].sum(1)))
    rb = metrics(kb,a,i,m,np.array([0,-1,1]),np.ones(3,bool),2)
    passed('Presence collapse keeps FOREIGN and BG separate from T only', np.allclose(rb['expected_soft_CM'], [[.5,0,0],[.35,0,1]]))
    q = np.array([[1,2],[np.nan,np.nan],[3,4]], np.float64)
    passed('Exact q equality includes null q, rejects coordinate change', same_rows(q,q.copy()).all() and not same_rows(q,q+1)[0])
    passed('ID/native/split mismatch rejected by row mask', same_rows(np.array(['a','x']),np.array(['a','b'])).tolist() == [True,False])
    empty=metrics(k,a,i,m,np.array([0,-1,2]),np.zeros(3,bool),3)
    passed('Empty strata retained with null P/R', empty['rows']==0 and empty['expected_precision_on_known_truth_mass']==[None]*3)
    passed('Full inventory equals known plus unknown mass', abs(r['known_mass'] + sum(r['unknown_mass_by_prediction']) - 3) < 1e-15)
    passed('Models remain separate; paircohort is conjunction only', (np.array([1,1,0],bool)&np.array([1,0,1],bool)).tolist()==[True,False,False])
    try: truth_mass(np.array([[.9,0,0,0,0,0.]]), np.array([False]))
    except AssertionError: rejected=True
    else: rejected=False
    passed('Non-unit source mass fails closed; never renormalized', rejected)
    return tests


def prepare():
    require(sha(SCOPE) == SCOPE_SHA and sha(CENSUS) == CENSUS_SHA, 'Frozen metadata')
    ready = read(READY); complete = read(CENSUS)
    require(ready['PASS'] and complete['completed'] and complete['PASS'], 'Metadata completion')
    require(complete['owner_class_order'] == OWNER6, 'Truth class order')
    tests = invented()
    write(CHECKS, dict(completed=True,PASS=True,code=bind(__file__),checks=tests,
                      actual_score_truth_feature_weight_or_model_bodies_read=0))
    packets = {name:ready['owner_models'][name]['saved_packet'] for name in MODELS[:2]}
    packets['v61'] = ready['v61']['saved_packet']
    protocol = dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),
                    readiness=bind(READY),checks=bind(CHECKS),census_complete=bind(CENSUS),
                    sidecars=complete['sidecars'],saved_packets=packets,
                    expected_owner6=OWNER6,expected_owner3=OWNER3,
                    truth_key='scene_owner_point',truth_only_exact_original_q=True,
                    owner3_rule='query_available first, then original argmax first-index ties T/F/BG',
                    v61_rule='original visible logit>=0 T / <0 nonT / originalinputU; no F-versus-BG claim',
                    owner_metrics='Expected float64 3x4 known T/F/BG soft mass confusion with predictionU column, P/R, unknown allocation; no GT cutoff.',
                    binary_metrics='All3 separate 2x3 T/nonT/U mass confusions; owner F+BG collapsed; no new threshold.',
                    missing_q='Keep source allNaN; output inventory missing-unknown1 only, not measured raster truth.',
                    alignment='Every saved row compared by original ID/global/scene/base/native/q/L/no_detection; availability kept permodel. Mismatch remains U in all6144 inventory and makes overall PASS false, no remap/retry.',
                    cohorts='ALL6144, originalTRAIN4096 apparent, primaryVALIDATION2048; each original native/material/aspect/variant/semantic stratum. Full inventory and separately all3 matched-available cohort.',
                    source_causal7_not_used_as_owner_truth=True,
                    original_source_qualification_not_overridden=True,
                    limitation='Point ownership is not crop-part presence, physical native identity/correctness or native external/self visibility. No labels changed or training/adoption/model/threshold choice.',
                    runtime=dict(executable=PYTHON,numpy='1.26.4',torch_not_imported=True),
                    output=str(OUT),selected_model=None,fully_solved=False)
    write(PROTOCOL,protocol)
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(
        approved_saved_source_owner_benchmark=False,source_only=True,code=bind(__file__),
        protocol=bind(PROTOCOL),census_complete=bind(CENSUS),no_fit_forward_encoder_or_model_selection=True))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),
        protocol=bind(PROTOCOL),checks=bind(CHECKS),readiness=bind(READY),prepared_only=True,
        actual_score_or_truth_bodies_read=False,actual_evaluation_performed=False))


def run(entry_path):
    # Root entry and all frozen JSON guards precede any NPZ hash/header/body.
    e=read(entry_path);require(e.get('approved_saved_source_owner_benchmark') is True and e.get('source_only') is True, 'No root saved-only entry')
    require(e.get('no_fit_forward_encoder_or_model_selection') is True, 'Wrong task scope')
    require(e['code'] == bind(__file__) and e['protocol'] == bind(PROTOCOL), 'Entry/code/protocol')
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['checks']==bind(CHECKS), 'Frozen preparation')
    require(e['census_complete']==p['census_complete'], 'Entry census binding')
    check(p['scope']);check(p['readiness']);check(p['census_complete']);check(p['checks'])
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and np.__version__=='1.26.4', 'Pinned runtime')
    require(not OUT.exists(), 'Preserve previous attempt; no automatic retry')
    OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL)))
    try:
        pieces=[]
        for split,n in [('TRAIN',4096),('VALIDATION',2048)]:
            with np.load(check(p['sidecars'][split]),allow_pickle=False) as z:
                require(z['owner_class_order'].astype(str).tolist()==OWNER6, 'Sidecar order')
                require(not bool(z['training_admission']) and not bool(z['initializer_physical_native_identity_certified']), 'No admission/physical gold')
                d={k:z[k].copy() for k in META};d['owner6']=z[p['truth_key']].copy()
                require(len(d['original_global_index'])==n and (d['split'].astype(str)==split).all(), 'Sidecar split count')
                pieces.append(d)
        d={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
        order=np.argsort(d['original_global_index']);d={k:x[order] for k,x in d.items()}
        require(np.array_equal(d['original_global_index'],np.arange(6144)), 'All original6144 IDs once')
        known,a,iv,missing=truth_mass(d['owner6'],d['query_xy_missing'])
        model_data={};alignment=[]
        for name in MODELS:
            with np.load(check(p['saved_packets'][name]),allow_pickle=False) as z:
                ok=np.ones(6144,bool);reasons=[[] for _ in range(6144)]
                for src,dst in [('global_index','original_global_index'),('occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L')]:
                    fieldok=same_rows(z[src],d[dst]);ok &= fieldok
                    for j in np.flatnonzero(~fieldok): reasons[j].append(src)
                nodet=d['selected_index_missing'];fieldok=same_rows(z['no_detection'],nodet);ok &= fieldok
                for j in np.flatnonzero(~fieldok):reasons[j].append('no_detection')
                av=z['query_available'].astype(bool).copy()
                require(av.shape==(6144,), 'Full saved input availability')
                if name!='v61':
                    require(z['owner_class_order'].astype(str).tolist()==OWNER3, 'Saved owner class order')
                    raw=z['raw_owner_three_logits'].copy();pred,finite=owner_predictions(raw,av&ok)
                else:
                    raw=z['raw_visible_logit'].copy();pred,finite=presence_predictions(raw,av&ok)
                    require(np.array_equal(pred[ok],z['prediction_D0_E1_Uminus1'][ok]), 'Original v61 threshold/U parity')
                nonfinite=ok&av&~finite
                model_data[name]=dict(pred=pred,matched=ok,available=av,finite=finite,raw=raw,reasons=reasons)
                alignment.append(dict(model=name,matched_rows=int(ok.sum()),mismatch_rows=int((~ok).sum()),
                    original_inputU=int((~av).sum()),available_nonfinite=int(nonfinite.sum()),
                    saved_query_availability_equals_original_census=bool(np.array_equal(av,d['query_available']))))
        paired=np.logical_and.reduce([x['matched']&x['available']&x['finite'] for x in model_data.values()])
        cohorts={'ALL':np.ones(6144,bool),'TRAIN':d['split'].astype(str)=='TRAIN','VALIDATION':d['split'].astype(str)=='VALIDATION'}
        groups={}
        for cohort,mask in cohorts.items():
            groups[cohort]=mask
            for factor in ['native_slot','material','aspect','variant','original_semantic_state']:
                for val in np.unique(d[factor]):groups[cohort+'::'+factor+'::'+str(val)]=mask&(d[factor]==val)
        reports={}
        binary=np.column_stack((known[:,0],known[:,1:].sum(1)))
        for cname,gate in [('FULL_INVENTORY',np.ones(6144,bool)),('ALL3_MATCHED_AVAILABLE',paired)]:
            reports[cname]={}
            for group,mask in groups.items():
                mm=mask&gate;record={}
                for name,x in model_data.items():
                    pred=x['pred'];bp=pred.copy() if name=='v61' else np.where(pred<0,-1,(pred!=0).astype(np.int8))
                    metric=dict(binary_target_presence=metrics(binary,a,iv,missing,bp,mm,2))
                    if name!='v61':metric['owner3']=metrics(known,a,iv,missing,pred,mm,3)
                    record[name]=metric
                reports[cname][group]=record
        fields=['original_global_index','original_occurrence_id','scene_index','base_index','native_slot','split','material','aspect','variant','original_semantic_state','query_available','query_xy_missing','L_missing','w_TARGET','w_FOREIGN','w_BG','ambiguous_mass','invalid_mass','missing_q_inventory_unknown']
        fields += [name+'_'+x for name in MODELS for x in ['prediction','original_available','matched','mismatch_fields']]
        ledger=OUT/'ALL6144_SAVED_OWNER_AND_EXACT_POINT_MASS.csv'
        with ledger.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(6144):
                row={k:d[k][i].item() for k in fields[:13]}
                row.update(w_TARGET=known[i,0],w_FOREIGN=known[i,1],w_BG=known[i,2],ambiguous_mass=a[i],invalid_mass=iv[i],missing_q_inventory_unknown=missing[i])
                for name,x in model_data.items():
                    row.update({name+'_prediction':int(x['pred'][i]),name+'_original_available':bool(x['available'][i]),name+'_matched':bool(x['matched'][i]),name+'_mismatch_fields':'|'.join(x['reasons'][i])})
                w.writerow(row)
        report=OUT/'EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json';write(report,dict(completed=True,primary='FULL_INVENTORY/VALIDATION',TRAIN_apparent_only=True,truth='exact original q point owner mass only',owner3_class_order=['T','F','BG','predictionU'],binary_class_order=['T','nonT','predictionU'],reports=reports))
        passed=all(x['mismatch_rows']==0 and x['available_nonfinite']==0 and x['saved_query_availability_equals_original_census'] for x in alignment)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=passed,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),
              rows=6144,TRAIN=4096,VALIDATION=2048,all_rows_retained=True,paired_matched_available_rows=int(paired.sum()),
              alignment=alignment,ledger=bind(ledger),reports=bind(report),census_complete=p['census_complete'],saved_packets=p['saved_packets'],
              no_source_truth_threshold_or_model_choice=True,new_fit_forward_encoder_PnP_qL_label_or_training_admission=0,
              original_native_source_qualification_unchanged=True,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False))
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path)
    args=parser.parse_args()
    prepare() if args.prepare else run(args.entry)
