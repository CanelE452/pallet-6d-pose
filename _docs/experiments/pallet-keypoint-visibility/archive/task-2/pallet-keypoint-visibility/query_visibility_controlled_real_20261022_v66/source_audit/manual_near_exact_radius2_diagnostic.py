"""Saved V65/V66 support scores at identical manual-reference-near D/E IDs.

Reads only existing semantic/provenance and sealed score files. No evaluator
import, model, feature, RGB, pose solve, threshold fit or model choice.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1];PALLET=ROOT.parent
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';GROUPS=('WOOD_PRIMARY105','SQUARE118')
SIDECAR=Path('/home/minjae/Documents/github/pallet-pose/_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def key(r):return(r['id'],int(r['native_corner']))
def yes(r,k):return r.get(k,'').lower()=='true'
def ratio(a,b):return a/b if b else None
def decision(r,support):
    if not yes(r,'valid'):return'UNRESOLVED'
    field='p_visible_exact'if support=='exact'else'p_visible_radius2'
    if not r.get(field):return'UNRESOLVED'
    p=float(r[field]);assert math.isfinite(p)and 0<=p<=1
    return'VISIBLE'if p>=.5 else'ABSENT_NONVISIBLE'

def main():
    out=ROOT/'source_audit/MANUAL_NEAR_EXACT_RADIUS2_DIAGNOSTIC.json';assert not out.exists()
    manifest_path=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
    nearD_path=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
    original_audit=ROOT/'source_audit/CURRENT_SAVED_METRICS_INDEPENDENT_AUDIT.json'
    assert sha(original_audit)=='1dedddd25ed08cdb9ea85061c580438dc5b36f10bab59b46f68429ede199c563'
    models={'V65_MIXED_GLOBAL':PALLET/'query_visibility_amodal_global_20261022_v65/private_evaluation/MIXED_GLOBAL_UNWEIGHTED/REAL_QUERY_SCORES.csv',
            'V66_CONTROLLED_PLUS_PALLOX':ROOT/'private_evaluation/CONTROLLED_PLUS_PALLOX/REAL_QUERY_SCORES.csv'}
    tables={};bindings=[ref(manifest_path),ref(nearD_path),ref(SIDECAR),ref(original_audit),ref(__file__)]
    for model,path in models.items():
        sealpath=path.with_name('INFERENCE_SEAL.json');seal=json.loads(sealpath.read_text());assert seal['completed']is True
        b=next(b for b in seal['outputs']if Path(b['path']).resolve()==path.resolve());assert sha(path)==b['sha256']
        rr=read(path);tables[model]={key(r):r for r in rr};assert len(tables[model])==1784
        bindings.extend([ref(path),ref(sealpath)])
    assert set(tables['V65_MIXED_GLOBAL'])==set(tables['V66_CONTROLLED_PLUS_PALLOX'])
    manifest=read(manifest_path);mm={key(r):r for r in manifest};assert len(mm)==1784
    for k in mm:
        a,b=[tables[m][k]for m in models]
        assert a['image_sha256']==b['image_sha256']==mm[k]['image_sha256']
        assert a['valid']==b['valid']
        for c in ['query_x','query_y']:assert float(a[c])==float(b[c])==float(mm[k][c])
    side=json.loads(SIDECAR.read_text());lookup={(r['population'],r['frame_id'],int(r['corner_id'])):r for r in side['rows']}
    selected={D:[],E:[]};eligibility=[];candidate_provenance=[]
    for state in [D,E]:
        for group in GROUPS:
            allrows=[r for r in manifest if r['human_state']==state and r['group']==group]
            le3=[r for r in allrows if r['initial_reference_error_px']and float(r['initial_reference_error_px'])<=3]
            eligible=[r for r in le3 if yes(r,'paired_same_query_RGB_D_vs_E')and yes(r,'legacy_direct_score_comparable_location')]
            sourcecounts=Counter();nmanual=0
            for r in eligible:
                population,frame=('DEV319',r['id'])if group=='WOOD_PRIMARY105'else('GREEN0918',r['id'].split(':')[1])
                h=lookup[(population,frame,int(r['reference_corner_id']))];assert h['category']==state and h['image_sha256']==r['image_sha256']
                xy=h['reference_xy'];assert abs(float(xy[0])-float(r['human_reference_x']))<1e-12 and abs(float(xy[1])-float(r['human_reference_y']))<1e-12
                provenance=h.get('label_evidence',{}).get('annotation_coordinate_source','UNAVAILABLE');sourcecounts[provenance]+=1
                if provenance!='manual_click':continue
                nmanual+=1;selected[state].append(r)
                candidate_provenance.append(dict(id=r['id'],native_corner=int(r['native_corner']),group=group,human_state=state,
                    initial_reference_error_px=float(r['initial_reference_error_px']),annotation_coordinate_source=provenance,
                    reference_source=h['reference_source'],reference_coordinate_origin=h['label_evidence'].get('reference_coordinate_origin'),
                    model_prediction_exposure=h['label_evidence'].get('model_prediction_exposure'),reference_overlay_exposure=h['label_evidence'].get('reference_overlay_exposure')))
            eligibility.append(dict(human_state=state,group=group,all_semantic_rows=len(allrows),error_le3_before_existing_cohort_flags=len(le3),
                pairedDE_and_same_legacy_RGB_location_le3=len(eligible),coordinate_provenance_counts=dict(sourcecounts),manual_click_N=nmanual))
    saved_near=read(nearD_path);oldD={key(r)for r in saved_near if r['annotation_coordinate_source']=='manual_click'}
    assert {key(r)for r in selected[D]}==oldD
    assert Counter(r['group']for r in selected[D])==Counter(WOOD_PRIMARY105=153,SQUARE118=223)
    result=[];confusion=[]
    for model,table in tables.items():
        for group in GROUPS:
            for support in ['exact','radius2']:
                dc=Counter(decision(table[key(r)],support)for r in selected[D]if r['group']==group)
                ec=Counter(decision(table[key(r)],support)for r in selected[E]if r['group']==group)
                for state,c in [(D,dc),(E,ec)]:
                    n=sum(c.values());result.append(dict(model=model,group=group,support=support,human_state=state,N=n,
                        predicted_visible=c['VISIBLE'],predicted_absent_nonvisible=c['ABSENT_NONVISIBLE'],unresolved=c['UNRESOLVED'],
                        semantic_D_visible_recall=ratio(c['VISIBLE'],n)if state==D else None,
                        semantic_E_false_visible_rate=ratio(c['VISIBLE'],n)if state==E else None,
                        semantic_E_absent_recall=ratio(c['ABSENT_NONVISIBLE'],n)if state==E else None))
                confusion.append(dict(model=model,group=group,support=support,D_support=sum(dc.values()),E_support=sum(ec.values()),
                    TP=dc['VISIBLE'],FN=dc['ABSENT_NONVISIBLE'],FP=ec['VISIBLE'],TN=ec['ABSENT_NONVISIBLE'],D_unresolved=dc['UNRESOLVED'],E_unresolved=ec['UNRESOLVED']))
    published_path=ROOT/'contract/PRIVATE_PRIMARY_RESULTS.json';published=json.loads(published_path.read_text());bindings.append(ref(published_path))
    checked=0
    for row in result:
        if row['support']!='exact'or row['human_state']!=D:continue
        old=next(r for r in published['manual_near_D_results']if r['model']==row['model']and r['group']==row['group'])
        for a,b in [('N','N'),('predicted_visible','pred_visible'),('predicted_absent_nonvisible','pred_absent_nonvisible'),('unresolved','pred_other_unresolved')]:assert row[a]==old[b];checked+=1
    for row in result:
        if row['human_state']!=D or row['model']!='V66_CONTROLLED_PLUS_PALLOX'or row['support']!='radius2':continue
        expect=(151,153)if row['group']=='WOOD_PRIMARY105'else(218,223);assert(row['predicted_visible'],row['N'])==expect
    assert sha(original_audit)=='1dedddd25ed08cdb9ea85061c580438dc5b36f10bab59b46f68429ede199c563'
    report=dict(PASS=True,schema='SAVED_MANUAL_NEAR_D_E_EXACT_RADIUS2_DIAGNOSTIC_V1',bindings=bindings,
        model_comparison='Already frozen v65 and source-selected v66 saved outputs at identical IDs/query coordinates/valid flags; no new forward or selection.',
        threshold=.5,cohort='Existing paired_same_query_RGB_D_vs_E and legacy_direct_score_comparable_location; human initial/reference error<=3 and explicit annotation_coordinate_source=manual_click.',
        manual_D_ID_set_identical_to_existing153_223=True,manual_D_published_exact_count_fields_verified=checked,
        E_manual_flag_availability='Explicit manual_click provenance exists in the original all-state human sidecar. No manual flag is inferred from label/source naming.',
        eligibility=eligibility,metrics=result,binary_confusion_on_this_manual_near_D_E_subset=confusion,selected_ID_provenance=candidate_provenance,
        interpretation='R2 is learned any-visible native5x5 support. Increased semanticD recall and semanticE false-visible rate quantify target-definition tradeoffs. They are not exact-pixel segmentation accuracy: real instance occupancy masks and independent subpixel corner truth are unavailable.',
        hidden_reference_caveat='Manual-click hidden E coordinates remain semantic reference annotations; the flag does not certify physical pixel ownership or create an independent6D/subpixel reference.',
        original_current_audit_unchanged=True,source_winner_weights_thresholds_or_normalization_changed=False,
        saved_current_GT_scores_read=True,RGB_feature_checkpoint_reads=0,new_model_inference=0,new_training=0,GPU_used=False,PnP_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print('MANUAL_NEAR_DIAGNOSTIC_PASS',sha(out),flush=True)
    print('eligibility',json.dumps(eligibility),flush=True)
    print('metrics',json.dumps(result),flush=True)

if __name__=='__main__':main()
