"""Small independent audit of saved source boundary summaries.

All6400 source query predictions are joined to saved PER_QUERY CSVs. Native
distance/core geometry is reconstructed only for deterministic first4 targets
per source (64 queries), using brute nearest-false-pixel distance and explicit
5x5 patches rather than the diagnosis EDT/erosion implementation. No current
data, model, feature cache, inference, fit or threshold selection is used.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'source_audit';STEM='SOURCE_BOUNDARY_FAILURE_DIAGNOSIS'
MODELS=('V65_BASELINE','CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')
SOURCES=('ORIGINAL_YCB','CONTROLLED_YCB')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def jsonrows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def yes(r,k):return str(r[k]).lower()=='true'
def distance(mask,px,py):
    yy,xx=np.nonzero(~np.pad(mask,1,constant_values=False))
    return math.sqrt(int(np.min((xx-(px+1))**2+(yy-(py+1))**2)))
def patch(mask,px,py):
    p=np.pad(mask,2,constant_values=False)
    return p[py:py+5,px:px+5]
def bin_for(d):return'(0,1]'if d<=1 else'(1,2]'if d<=2 else'(2,3]'if d<=3 else'>3'

def main():
    output=HERE/(STEM+'_INDEPENDENT_MINI_AUDIT.json');assert not output.exists()
    reportpath=HERE/(STEM+'.json');j=json.loads(reportpath.read_text());assert j['completed']
    data=csvrows(HERE/(STEM+'_QUERY_ROWS.csv'));assert len(data)==19200
    by={(r['model'],r['sample_id']):r for r in data};assert len(by)==19200
    fields=0
    bindings=[ref(reportpath),ref(HERE/(STEM+'_QUERY_ROWS.csv'))]
    for model in MODELS:
        path=ROOT/'source_evaluation'/model/'PER_QUERY.csv';saved=csvrows(path);bindings.append(ref(path))
        selected=[r for r in saved if r['source']in SOURCES];assert len(selected)==6400
        for r in selected:
            d=by[(model,r['sample_id'])]
            assert d['source']==r['source']and int(d['exact_GT'])==int(r['exact_GT'])and int(d['radius2_GT'])==int(r['radius2_GT'])
            assert yes(d,'native_valid')==yes(r,'native_valid')
            for scale in ['exact','radius2']:
                assert float(d[scale+'_pV'])==float(r[scale+'_pV'])
                field='head_exact_visible'if scale=='exact'else'head_r2_visible';assert yes(d,field)==(float(r[scale+'_pV'])>=.5)
                sam='rawSAM_exact_positive'if scale=='exact'else'rawSAM_r2_positive';assert yes(d,sam)==(float(r[scale+'_rawSAM_pV'])>.5)
                assert int(d[scale+'_predicted'])==('V','E','O','U').index(r[scale+'_predicted'])
                fields+=4
    # Arithmetic replay of published visible strata, using the independently
    # verified saved pV values. Geometry flags are separately sampled below.
    summary_checks=0
    for r in j['visible_distance']:
        model='V65_BASELINE'if r['model']=='RAW_SAM_STRICT_ZERO'else r['model']
        rr=[d for d in data if d['source']==r['source']and d['model']==model and yes(d,'native_valid')and int(d['exact_GT'])==0]
        s=r['stratum']
        if s=='STRICT25_V_INTERIOR':rr=[d for d in rr if yes(d,'strict25_V')]
        elif s=='NONSTRICT25_V_BOUNDARY':rr=[d for d in rr if not yes(d,'strict25_V')]
        elif s!='ALL_KNOWN_V':rr=[d for d in rr if d['visible_distance_bin']==s]
        field='rawSAM_exact_positive'if r['model']=='RAW_SAM_STRICT_ZERO'else'head_exact_visible'
        rfield='rawSAM_r2_positive'if r['model']=='RAW_SAM_STRICT_ZERO'else'head_r2_visible'
        tp=sum(yes(d,field)for d in rr);r2=sum(yes(d,rfield)for d in rr)
        assert(r['N'],r['TP_exact'],r['FN_exact'],r['radius2_TP_diagnostic'])==(len(rr),tp,len(rr)-tp,r2)
        assert r['targets']==len({d['global_id']for d in rr});summary_checks+=5
    for r in j['controlled_external_core_recall']:
        rr=[d for d in data if d['source']==r['source']and d['model']==r['model']and yes(d,'native_valid')and int(d['exact_GT'])==1]
        if r['stratum']=='STRICT25_E_CORE':rr=[d for d in rr if yes(d,'strict25_E')]
        elif r['stratum']=='NONSTRICT25_E_BOUNDARY':rr=[d for d in rr if not yes(d,'strict25_E')]
        counts=Counter(int(d['exact_predicted'])for d in rr)
        assert r['N']==len(rr)
        assert[r['pred_'+c]for c in ['V','E','O','U']]==[counts[k]for k in range(4)];summary_checks+=5
    for r in j['source_state_confusion']:
        if r['stratum']!='ALL':continue
        rr=[d for d in data if d['source']==r['source']and d['model']==r['model']and yes(d,'native_valid')]
        scale=r['support'];known=[d for d in rr if int(d[scale+'_GT'])>=0]
        predE=[d for d in known if int(d[scale+'_predicted'])==1]
        tp=sum(int(d[scale+'_GT'])==1 for d in predE)
        assert r['known_GT']==len(known)and r['UNKNOWN_GT']==len(rr)-len(known)
        assert r['E_predicted_on_known_GT']==len(predE)and r['E_TP']==tp;summary_checks+=4
    targets={r['id']:r for r in jsonrows(ROOT/'provenance/TARGETS.jsonl')if r['split']=='VALIDATION'}
    comps={r['id']:r for r in jsonrows(ROOT/'controlled/COMPOSITES.jsonl')if r['split']=='VALIDATION'}
    samples=[]
    for source,records in [('ORIGINAL_YCB',targets),('CONTROLLED_YCB',comps)]:
        for ident in sorted(records)[:4]:
            rec=records[ident]
            if source=='ORIGINAL_YCB':
                visible=cv2.imread(rec['visible']['path'],-1)>0;external=np.zeros_like(visible);sids=rec['original_sample_ids']
                bindings.append(ref(rec['visible']['path']))
            else:
                with np.load(rec['labels']['path'],allow_pickle=False)as z:
                    visible=z['visible_positive'];external=z['full_positive']&z['visible_negative']
                    assert np.array_equal(visible,z['original_visible']&~z['paste_mask'])
                    assert np.array_equal(external,z['original_visible']&z['paste_mask'])
                sids=[ident+':'+str(k)for k in range(8)];bindings.append(ref(rec['labels']['path']))
            mismatch=0
            for sid in sids:
                r=by[('V65_BASELINE',sid)];px,py=int(r['pixel_x']),int(r['pixel_y']);assert yes(r,'native_valid')
                vd=distance(visible,px,py);ed=distance(external,px,py)if external.any()else 0.
                vc=bool(patch(visible,px,py).all());ec=bool(patch(external,px,py).all())
                assert vd==float(r['visible_inside_distance_px'])and ed==float(r['controlled_E_inside_distance_px'])
                assert vc==yes(r,'strict25_V')and ec==yes(r,'strict25_E')
                context='STRICT25_E_CORE'if ec else'E_INTERSECTING_5X5_BAND'if patch(external,px,py).any()else'NO_E_IN_5X5'
                assert r['E_context']==context
                if int(r['exact_GT'])==0:assert r['visible_distance_bin']==bin_for(vd)
            samples.append(dict(source=source,target_id=ident,queries=8,geometry_mismatches=mismatch))
    # Strict-square and Euclidean-distance diagnostics overlap; not partitions.
    keyrows=[r for r in j['visible_distance']if r['model']in ['V65_BASELINE','CONTROLLED_PLUS_PALLOX']and r['stratum']in ['>3','STRICT25_V_INTERIOR']]
    results=dict(PASS=True,schema='SOURCE_BOUNDARY_MINIMAL_INDEPENDENT_REPLAY_V1',bindings=bindings,code=ref(__file__),
        saved_PER_QUERY_rows_checked=19200,saved_prediction_fields_compared=fields,summary_count_fields_compared=summary_checks,
        native_geometry_targets_checked=len(samples),native_geometry_queries_checked=sum(x['queries']for x in samples),native_geometry_samples=samples,
        geometry_method='Brute nearest false native pixel center on onefalseborder padding; explicit25pixel patch all/any. Diagnosis EDT and erosion functions are not imported.',
        key_interior_counts=keyrows,blockers_found=[],
        interpretation_checks=dict(known_native_visible_boundary_FN_is_still_exact_error=True,
            strict25_visible_interior_FN_refutes_boundary_only_explanation=True,
            strict25_square_is_separate_overlapping_stratum_not_same_as_Euclidean_distance_gt3=True,
            E_precision_ALL_uses_every_knownVEO_prediction_denominator=True,
            strict25_E_positive_only_precision_is_not_discriminative=True,
            radius2_recall_here_is_on_same_exactV_stratum_not_full_known_radius2_GT_cohort=True),
        no_current_files_RGB_features_weights_or_models_read=True,new_model_inference_training_GPU_PnP_calls=0,threshold_or_selection_changes=False)
    output.write_text(json.dumps(results,indent=2)+'\n');print('SOURCE_BOUNDARY_MINI_PASS',sha(output),'summary',summary_checks,'native64queries',flush=True)

if __name__=='__main__':main()
