"""Independent saved-output arithmetic only; no semantic evaluator import.

The parent authorized CURRENT saved GT/scores after selected inference/eval
seals completed. No RGB, feature cache, checkpoint, model or fitting is opened.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]
PALLET=ROOT.parent
GROUPS=('WOOD_PRIMARY105','SQUARE118')
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
PAIRS=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))
JOINT=('VV','EV','EE','OV','OE','OO')
CLASS=('V','E','O','U')
SAVED=('VISIBLE_TARGET','EXTERNAL_SUPPORT_PROXY','OUTSIDE_FULL_SUPPORT','UNKNOWN')
MODEL='V66_CONTROLLED_PLUS_PALLOX'

def read(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def yes(r,key):return r.get(key,'').lower()=='true'
def key(r):return(r['id'],int(r['native_corner']))
def ratio(a,b):return a/b if b else None
def binary(score,scale):
    if not yes(score,'valid'):return'OTHER_UNRESOLVED'
    field='p_visible_exact'if scale=='exact'else'p_visible_radius2'
    if not score.get(field):return'OTHER_UNRESOLVED'
    p=float(score[field]);assert math.isfinite(p)and 0<=p<=1
    return'VISIBLE'if p>=.5 else'ABSENT_NONVISIBLE'

def decoded(score):
    if not yes(score,'valid'):return dict(joint=-1,name='U',classes=('U','U'))
    p=[float(score['joint_p'+str(j)])for j in range(6)]
    assert all(math.isfinite(x)and 0<=x<=1 for x in p)and abs(sum(p)-1)<1e-12
    j=max(range(6),key=lambda k:p[k]);accepted=p[j]>=.5
    states=PAIRS[j]if accepted else(3,3)
    out=dict(joint=j if accepted else-1,name=JOINT[j]if accepted else'U',classes=tuple(CLASS[x]for x in states))
    exact=[p[0],p[1]+p[2],p[3]+p[4]+p[5]];near=[p[0]+p[1]+p[3],p[2]+p[4],p[5]]
    for k,name in enumerate(['exact','radius2']):
        for cls,field in enumerate(['p_visible_','p_external_support_','p_outside_full_']):
            assert abs(float(score[field+name])-[exact,near][k][cls])<1e-12
        assert score['confident_'+name]==SAVED[states[k]]
    assert abs(float(score['p_visible'])-exact[0])<1e-12
    assert yes(score,'confidence_U')==not_accepted(accepted)
    return out

def not_accepted(x):return not x

def metrics(rr,scores,scale,negative_states):
    assert all(r['human_state']in(D,)+negative_states for r in rr)
    pos=[r for r in rr if r['human_state']==D];neg=[r for r in rr if r['human_state']in negative_states]
    pc=Counter(binary(scores[key(r)],scale)for r in pos);nc=Counter(binary(scores[key(r)],scale)for r in neg)
    tp,fn,fp,tn=pc['VISIBLE'],pc['ABSENT_NONVISIBLE'],nc['VISIBLE'],nc['ABSENT_NONVISIBLE']
    pu,nu=pc['OTHER_UNRESOLVED'],nc['OTHER_UNRESOLVED'];N=len(rr);decided=tp+fn+fp+tn
    return dict(N=N,frames=len({r['id']for r in rr}),visible_support=len(pos),occluded_support=len(neg),
        TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=0,U_occluded=0,out_of_task_S_visible=0,out_of_task_S_occluded=0,
        other_unresolved_visible=pu,other_unresolved_occluded=nu,
        visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,len(pos)),
        occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,len(neg)),
        abstention_rate=0.0 if N else None,out_of_task_S_rate=0.0 if N else None,
        unresolved_rate=ratio(pu+nu,N),decision_coverage=ratio(decided,N),correct_fraction_all_GT=ratio(tp+tn,N),selective_accuracy=ratio(tp+tn,decided))

def main():
    output=ROOT/'source_audit/CURRENT_SAVED_METRICS_INDEPENDENT_AUDIT.json';assert not output.exists()
    manifest_path=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
    scores_path=ROOT/'private_evaluation/CONTROLLED_PLUS_PALLOX/REAL_QUERY_SCORES.csv'
    near_path=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
    published_path=ROOT/'contract/PRIVATE_PRIMARY_RESULTS.json';published_csv=ROOT/'contract/PRIVATE_PRIMARY_RESULTS.csv'
    seal_path=scores_path.with_name('INFERENCE_SEAL.json');seal=json.loads(seal_path.read_text())
    assert seal['completed']is True and seal['candidate']=='CONTROLLED_PLUS_PALLOX'
    registered={str(Path(b['path']).resolve()):b for b in seal['outputs']}
    assert sha(scores_path)==registered[str(scores_path.resolve())]['sha256']
    selection_path=ROOT/'SOURCE_SELECTION_SEAL.json'
    assert sha(selection_path)==seal['source_selection_seal']['sha256']
    manifest=read(manifest_path);rawscores=read(scores_path);near=read(near_path)
    published=json.loads(published_path.read_text());csvpublished=read(published_csv)
    assert published['selected_primary']==MODEL and published['source_selection_seal_sha256']==sha(selection_path)
    assert len(manifest)==len(rawscores)==1784
    mm={key(r):r for r in manifest};scores={key(r):r for r in rawscores}
    assert len(mm)==len(scores)==1784 and set(mm)==set(scores)
    decoded_scores={}
    for k,row in scores.items():
        base=mm[k];assert row['image_sha256']==base['image_sha256']
        assert float(row['query_x'])==float(base['query_x'])and float(row['query_y'])==float(base['query_y'])
        decoded_scores[k]=decoded(row)
    pub={(r['cohort'],r['group'],r['task'],r['model']):r for r in published['metrics']}
    pubcsv={(r['cohort'],r['group'],r['task'],r['model']):r for r in csvpublished}
    comparisons=[];discrepancies=[];nfields=0
    for cohort in ['paired_same_query_RGB','same_slot_pipeline']:
        for group in GROUPS:
            for task,negative in [('D_vs_E',(E,)),('D_vs_ES',(E,S))]:
                rr=[r for r in manifest if r['group']==group and yes(r,cohort+'_'+task)]
                for scale,name in [('exact',MODEL),('radius2',MODEL+'_R2_SUPPORT_DIAGNOSTIC')]:
                    m=metrics(rr,scores,scale,negative);pk=(cohort,group,task,name)
                    assert pk in pub and pk in pubcsv
                    for field,value in m.items():
                        for label,publishedrow in [('JSON',pub[pk]),('CSV',pubcsv[pk])]:
                            actual=publishedrow[field];nfields+=1
                            if value is None:ok=actual is None or actual==''
                            else:ok=actual not in(None,'')and math.isclose(float(actual),float(value),rel_tol=0,abs_tol=1e-12)
                            if not ok:discrepancies.append(dict(cohort=cohort,group=group,task=task,scale=scale,container=label,field=field,expected=value,published=actual))
                    comparisons.append(dict(cohort=cohort,group=group,task=task,support=scale,**m))
    # Retain all old eligibility flags. No invalid/mapping repair or resampling.
    pairde=[r for r in manifest if yes(r,'paired_same_query_RGB_D_vs_E')]
    assert Counter(r['group']for r in pairde)==Counter(WOOD_PRIMARY105=645,SQUARE118=592)
    confidence=[];jointcounts=[]
    for group in GROUPS:
        for state in [D,E]:
            rr=[r for r in pairde if r['group']==group and r['human_state']==state]
            c=Counter(decoded_scores[key(r)]['name']for r in rr)
            jc=dict(group=group,human_state=state,N=len(rr),**{j:c[j]for j in JOINT},U=c['U']);jointcounts.append(jc)
            expected=next(r for r in published['source_support_six_joint_counts_paired_D_E']if r['group']==group and r['human_state']==state)
            if expected!=jc:discrepancies.append(dict(kind='six_joint',group=group,human_state=state,expected=jc,published=expected))
            for k,name in enumerate(['exact','radius2']):
                cc=Counter(decoded_scores[key(r)]['classes'][k]for r in rr)
                confidence.append(dict(group=group,human_state=state,support=name,N=len(rr),**{c:cc[c]for c in CLASS}))
    manual=[r for r in near if r['annotation_coordinate_source']=='manual_click']
    assert len(near)==461 and len(manual)==376 and all(r['human_state']==D and float(r['initial_reference_error_px'])<=3 for r in near)
    assert Counter(r['group']for r in manual)==Counter(WOOD_PRIMARY105=153,SQUARE118=223)
    manual_result=[]
    for group in GROUPS:
        rr=[r for r in manual if r['group']==group]
        for scale in ['exact','radius2']:
            c=Counter(binary(scores[key(r)],scale)for r in rr)
            got=dict(group=group,support=scale,N=len(rr),visible=c['VISIBLE'],absent_nonvisible=c['ABSENT_NONVISIBLE'],unresolved=c['OTHER_UNRESOLVED'],semantic_D_recall=ratio(c['VISIBLE'],len(rr)))
            manual_result.append(got)
            if scale=='exact':
                old=next(r for r in published['manual_near_D_results']if r['group']==group and r['model']==MODEL)
                fields={'N':'N','visible':'pred_visible','absent_nonvisible':'pred_absent_nonvisible','unresolved':'pred_other_unresolved','semantic_D_recall':'semantic_D_recall'}
                for a,b in fields.items():
                    nfields+=1
                    if not math.isclose(float(got[a]),float(old[b]),rel_tol=0,abs_tol=1e-12):discrepancies.append(dict(kind='manual',group=group,field=a,expected=got[a],published=old[b]))
    direct=[r for r in pairde if r['human_state']==D]
    absent=[r for r in direct if binary(scores[key(r)],'exact')=='ABSENT_NONVISIBLE']
    decomposition=Counter(decoded_scores[key(r)]['name']for r in absent)
    assert len(direct)==1147 and len(absent)==802 and sum(decomposition.values())==802
    assert decomposition==Counter(OV=551,EE=4,OO=115,U=132)
    external=[r for r in pairde if r['human_state']==E]
    external_conf=Counter(decoded_scores[key(r)]['classes'][0]for r in external)
    external_binary=Counter(binary(scores[key(r)],'exact')for r in external)
    assert len(external)==90 and external_conf['E']==13 and external_binary['ABSENT_NONVISIBLE']==84
    # The published semantic/source-support count table includes D/E/S and both
    # old cohorts. Reconstruct it without calling a semantic evaluator.
    source_dist=[]
    for old in published['source_support_output_counts']:
        rr=[r for r in manifest if r['group']==old['group']and r['human_state']==old['human_state']and yes(r,old['cohort']+'_D_vs_ES')]
        k=0 if old['scale']=='exact'else 1;c=Counter(decoded_scores[key(r)]['classes'][k]for r in rr)
        got=dict(cohort=old['cohort'],group=old['group'],human_state=old['human_state'],scale=old['scale'],N=len(rr),
            VISIBLE_SUPPORT_PROXY=c['V'],EXTERNAL_SUPPORT_PROXY=c['E'],OUTSIDE_FULL_SUPPORT_PROXY=c['O'],U=c['U'])
        if got!=old:discrepancies.append(dict(kind='human_vs_source_support',expected=got,published=old))
        source_dist.append(got)
    assert not discrepancies,discrepancies[:3]
    report=dict(PASS=True,schema='INDEPENDENT_CURRENT_SAVED_METRICS_DIRECT_CSV_AUDIT_V1',
        scope='Parent-authorized saved score/semantic GT audit after selected frozen inference/evaluation. No model/feature/semantic-evaluator import or new threshold/model choice.',
        bindings=[ref(p)for p in [Path(__file__),manifest_path,scores_path,near_path,published_path,published_csv,seal_path,selection_path]],
        model=MODEL,fixed_threshold=.5,published_numeric_fields_compared=nfields,published_mismatches=discrepancies,
        full_slot_accounting=dict(rows=1784,valid_scores=sum(yes(r,'valid')for r in rawscores),invalid_scores=sum(not yes(r,'valid')for r in rawscores),
            mapping_status=dict(Counter(r['mapping_status']for r in manifest)),human_state=dict(Counter(r['human_state']for r in manifest)),
            flag_counts={f:sum(yes(r,f)for r in manifest)for f in ['paired_same_query_RGB_D_vs_E','paired_same_query_RGB_D_vs_ES','same_slot_pipeline_D_vs_E','same_slot_pipeline_D_vs_ES']},
            mapping_nonverified_rows=[dict(status=s,N=n,score_valid=sum(yes(scores[key(r)],'valid')for r in manifest if r['mapping_status']==s))for s,n in Counter(r['mapping_status']for r in manifest).items()if s!='VERIFIED_FROZEN_BASELINE_MAPPING']),
        binary_metrics=comparisons,paired_D_E_confidence_V_E_O_U=confidence,paired_D_E_six_joint=jointcounts,
        manual_reference_near_D_le3=manual_result,
        D_false_absence=dict(total_D=1147,binary_exact_absent=802,confident_six_joint_decomposition=dict(decomposition),
            interpretation='OV is model source-support output exact outside / radius2 visible. It is not verified coordinate-error truth: current exact-pixel target-ownership masks are unavailable; human corner/reference semantics cannot identify the exact query pixel class.'),
        semantic_E_vs_binary_absence=dict(N=90,binary_exact_absence=84,binary_absence_rate=84/90,
            confident_source_E_agreement=13,confident_source_E_agreement_rate=13/90,
            confident_exact_V_E_O_U={c:external_conf[c]for c in CLASS},
            warning='84/90 absence is not84/90 external-occlusion recognition. Source E agreement13/90 is diagnostic agreement with human semantic E, not independent exact-pixel mechanism GT.'),
        source_support_distribution_reconstructed=source_dist,
        frozen_winner_or_weights_modified=False,source_fitting_or_normalization_or_threshold_changes=False,
        actual_saved_GT_scores_read=True,RGB_images_opened=0,feature_or_checkpoint_files_opened=0,
        semantic_evaluator_imported=False,new_model_inference=0,new_training=0,GPU_used=False,PnP_calls=0)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('CURRENT_SAVED_AUDIT_PASS',sha(output),'fields',nfields,'Dabsence',len(absent),dict(decomposition),'E_conf',external_conf['E'],'Eabsence',external_binary['ABSENT_NONVISIBLE'],flush=True)
    print('accounting',json.dumps(report['full_slot_accounting']),flush=True)
    print('manual',json.dumps(manual_result),flush=True)

if __name__=='__main__':main()
