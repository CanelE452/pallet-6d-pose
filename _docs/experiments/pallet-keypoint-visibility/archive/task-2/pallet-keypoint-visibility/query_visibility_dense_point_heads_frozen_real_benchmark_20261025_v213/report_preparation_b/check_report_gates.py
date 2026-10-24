"""Production report helper fixtures; authored metadata only, no images/results."""
from pathlib import Path
import hashlib,json,os,stat,sys
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;P=H.parent.parent
def rb(p):
    assert P in p.parents and all('v112'not in s.casefold()for s in p.parts);c=Path('/')
    for s in p.parts[1:]:
        c=c/s;assert not stat.S_ISLNK(os.lstat(c).st_mode)
    return p.read_bytes()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def main():
    path=H/'build_fixed12_and_korean_report.py';m=type(sys)('authored_report');m.__file__=str(path);exec(compile(rb(path),str(path),'exec'),m.__dict__)
    passed=[]
    def test(n,ok=True):assert ok,n;passed.append(n)
    def rejects(n,fn):
        try:fn()
        except (AssertionError,KeyError):test(n)
        else:raise AssertionError('Expected rejection '+n)
    b=dict(path='AUTHORED_ONLY',sha256='a'*64)
    test('Exact_review_complete_alias',m.complete_alias(dict(complete=b),b))
    test('Exact_review_evaluation_complete_alias',m.complete_alias(dict(evaluation_complete=b,complete=b),b))
    rejects('Conflicting_review_alias_rejected',lambda:m.complete_alias(dict(complete=b,evaluation_complete=dict(path='DRIFT',sha256='b'*64)),b))
    arms={'C62':['C62_DENSE_POINT_RAW_Q8',False,0.],'C63':['C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',True,.25]}
    p=dict(evaluation_code=b,evaluation_protocol=b,source_fit_complete_path='AUTHORED_FIT',new_arms=arms)
    e=dict(prediction_seal='P',evaluation_complete='E',independent_numeric_review='Q')
    pred=dict(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,source_fit_complete=dict(path='AUTHORED_FIT'),original_rows2144=2144,model_domain_rows4288=4288,arms={a:dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=weak,weak_loss_weight=w)for a,weak,w in arms.values()})
    ev=dict(prediction_seal='P',code=b,protocol=b,models=m.MODELS,old_twelve_primary_metrics_exact=True)
    qa=dict(complete='E');store={'P':pred,'E':ev,'Q':qa};old=m.completed;m.completed=lambda k:store[k]
    try:
        test('Actual_report_gate_both_seals_old12_and_new_roles',m.evaluation_gate(e,p)==(pred,ev,qa))
        pred['arms'][arms['C63'][0]]['weak_loss_weight']=0.
        rejects('Actual_gate_wrong_weak_bridge_weight_rejected',lambda:m.evaluation_gate(e,p))
        pred['arms'][arms['C63'][0]]['weak_loss_weight']=.25
        ev['old_twelve_primary_metrics_exact']=False
        rejects('Actual_gate_old12_metric_drift_rejected',lambda:m.evaluation_gate(e,p));ev['old_twelve_primary_metrics_exact']=True
        pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']=False
        rejects('Actual_gate_new_predictions_not_closed_rejected',lambda:m.evaluation_gate(e,p))
    finally:m.completed=old
    rows=[]
    for i in range(461):
        r=dict(human_state='DIRECT_VISIBLE',initial_reference_error_px='2.5')
        for model in m.MODELS:r[model+'_D_E_U']='DEU'[i%3]
        rows.append(r)
    r=dict(human_state='DIRECT_VISIBLE',initial_reference_error_px='')
    for model in m.MODELS:r[model+'_D_E_U']='U'
    rows.append(r)
    keys={'native8':'primary_native8_D_E_with_U','type_U_inventory':'all1784_model_inventory','manual_near_D':'manual_near_D','coordinate_error_bins':'coordinate_error_bins','paired_recovery_damage':'paired_recovery_and_damage'}
    current=dict(tables={k:[dict(authored=k)]for k in keys.values()});s=m.supplementary_tables(current,rows)
    test('Reference461_and_missing_are_separate_all14',len(s['reference_association'])==28 and all(r['N']==461 for r in s['reference_association'][:14])and all(r['N']==1 and r['U']==1 for r in s['reference_association'][14:]))
    test('All_reference_D_E_U_denominators_conserved',all(r['D']+r['E']+r['U']==r['N']for r in s['reference_association']))
    test('Native_type_manual_bins_paired_tables_preserved',all(s[k]is current['tables'][v]for k,v in keys.items()))
    test('Fixed12_unchanged_and_new14_models',m.FIXED_IDS==[0,1,2,3,4,5,7,8,256,300,840,996]and len(m.MODELS)==14 and m.MODELS[-2:]==['C62','C63'])
    out=dict(completed=True,PASS=True,invented_only=True,code=bind(path),checks=passed,count=len(passed),actual_image_payload_or_score_reads=0,model_forward_or_fit=0)
    target=H/'NEW_REPORT_GATE_CHECKS.json'
    with target.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,count=len(passed),receipt=bind(target))))
if __name__=='__main__':main()
